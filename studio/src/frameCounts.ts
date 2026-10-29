import type { Binding } from './model.ts';
import {
  bindingIndex,
  currentValue,
  graphNodes,
  type GraphNode,
  type ObjectInfo,
} from './workflowGraph.ts';
import { setsFrames } from './workflowTraits.ts';

/** Inputs on several nodes that hold one clip's frame count, by node class and input name. */
interface DurationRule {
  family: string;
  members: Array<{ type: string; input: string; role: string }>;
  /** The clip's frame rate: fixed by the nodes, or read from these inputs. */
  fps: number | Array<{ type: string; input: string }>;
  /** The frame counts the nodes render exactly: base plus a whole number of steps. */
  grid: { base: number; step: number };
  relation: string;
}

const RULES: DurationRule[] = [
  {
    family: 'MiniMax H3',
    members: [
      { type: 'MiniMaxH3ImageToVideo', input: 'length', role: 'conditioning node' },
      { type: 'MiniMaxH3ReferenceToVideo', input: 'length', role: 'reference conditioning node' },
      { type: 'EmptyMiniMaxH3LatentAV', input: 'length', role: 'latent node' },
    ],
    // ComfyUI's MiniMax H3 nodes fix the clip at 24 fps and round a count up to 5 + 17k frames.
    fps: 24,
    grid: { base: 5, step: 17 },
    relation:
      'The conditioning node and the latent node describe the same clip, so they need the same frame count.',
  },
  {
    family: 'LTX',
    members: [
      { type: 'EmptyLTXVLatentVideo', input: 'length', role: 'video latent' },
      { type: 'LTXVImgToVideo', input: 'length', role: 'image to video node' },
      { type: 'LTXVEmptyLatentAudio', input: 'frames_number', role: 'audio latent' },
    ],
    fps: [
      { type: 'LTXVConditioning', input: 'frame_rate' },
      { type: 'LTXVEmptyLatentAudio', input: 'frame_rate' },
    ],
    // The LTX video latent keeps 1 + 8k frames and drops the rest.
    grid: { base: 1, step: 8 },
    relation:
      'The video latent and the audio latent describe the same clip, so they need the same frame count.',
  },
];

export interface DurationMember {
  binding: Binding;
  node: GraphNode;
  role: string;
}
/** One clip's frame-count inputs in a workflow, which a Duration control sets together. */
export interface DurationGroup {
  family: string;
  /** The inputs Duration writes. */
  members: DurationMember[];
  /** Members a wire feeds, which no draft value can set. */
  wired: DurationMember[];
  /** A fixed frame rate, or null when `rates` hold it. */
  fps: number | null;
  rates: DurationMember[];
  grid: { base: number; step: number };
  min: number;
  max: number;
  relation: string;
}

const bound = (binding: Binding, key: 'min' | 'max') =>
  binding.schema?.[key] ?? binding[key] ?? binding.validation?.[key];

/** The clip frame-count groups a workflow holds, one per family whose nodes it carries. */
export function durationGroups(
  workflow: Record<string, any> | null | undefined,
  bindings: Binding[],
): DurationGroup[] {
  const nodes = graphNodes(workflow);
  const bindingsOf = bindingIndex(bindings);
  const groups: DurationGroup[] = [];
  for (const rule of RULES) {
    const found = (entries: Array<{ type: string; input: string; role?: string }>) =>
      nodes.flatMap((node) =>
        entries
          .filter((entry) => entry.type === node.type)
          .flatMap((entry) => {
            const binding = bindingsOf(node).get(entry.input);
            return binding ? [{ binding, node, role: entry.role ?? 'frame rate' }] : [];
          }),
      );
    const members = found(rule.members);
    if (!members.length) continue;
    const editable = members.filter((member) => member.binding.resolved !== false);
    const limits = editable.map((member) => member.binding);
    groups.push({
      family: rule.family,
      members: editable,
      wired: members.filter((member) => member.binding.resolved === false),
      fps: typeof rule.fps === 'number' ? rule.fps : null,
      rates: typeof rule.fps === 'number' ? [] : found(rule.fps),
      grid: rule.grid,
      min: Math.max(rule.grid.base, ...limits.map((b) => Number(bound(b, 'min') ?? -Infinity))),
      max: Math.min(...limits.map((b) => Number(bound(b, 'max') ?? Infinity))),
      relation: rule.relation,
    });
  }
  return groups;
}

/** Whether one Duration control can set every member: none is wired, and one at least is editable. */
export function synchronized(group: DurationGroup): boolean {
  return group.members.length > 0 && group.wired.length === 0;
}

/** Each member's frame count in the draft, or null where it holds no number. */
export function clipFrames(group: DurationGroup, values: Record<string, unknown>) {
  return group.members.map((member) => {
    const value = currentValue(member.binding, values);
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  });
}

/** The clip's frame rate, or the reason it is unknown. */
export function clipRate(
  group: DurationGroup,
  values: Record<string, unknown>,
): { fps: number | null; reason?: string } {
  if (group.fps !== null) return { fps: group.fps };
  const editable = group.rates.filter((rate) => rate.binding.resolved !== false);
  const rates = [...new Set(editable.map((rate) => currentValue(rate.binding, values)))];
  if (!editable.length) return { fps: null, reason: 'No frame rate input sets this clip.' };
  if (rates.length > 1)
    return { fps: null, reason: `The frame rate inputs differ: ${rates.join(' and ')} fps.` };
  const fps = rates[0];
  return typeof fps === 'number' && Number.isFinite(fps) && fps > 0
    ? { fps }
    : { fps: null, reason: 'The frame rate input holds no positive number.' };
}

/** The nearest frame count the nodes render exactly, within every member's limits. */
export function snapFrames(group: DurationGroup, frames: number): number {
  const { base, step } = group.grid;
  const lowest = base + Math.max(0, Math.ceil((group.min - base) / step)) * step;
  const highest = base + Math.floor((group.max - base) / step) * step;
  const nearest = base + Math.round((frames - base) / step) * step;
  return Math.min(highest, Math.max(lowest, nearest));
}

/** The draft values with every member set to one frame count. */
export function setDuration(
  group: DurationGroup,
  frames: number,
  values: Record<string, unknown>,
): Record<string, unknown> {
  return {
    ...values,
    ...Object.fromEntries(group.members.map((member) => [member.binding.id, frames])),
  };
}

/** Seconds to two decimals, as the Duration control shows them. */
export const clipSeconds = (frames: number, fps: number) => Math.round((frames / fps) * 100) / 100;

/** Display notes for an input: the node it belongs to, its unit and how it relates to others. */
export interface InputMeta {
  context?: string;
  unit?: string;
  note?: string;
}

/**
 * Notes for frame-count and frame-rate inputs, keyed by binding id. A clip frame count that a
 * known family holds on several nodes names its node and role, and says how the counts relate;
 * any other frame count, such as a Wan latent's length, gets its unit, and its node when another
 * frame count shares its name.
 */
export function inputMetadata(
  workflow: Record<string, any> | null | undefined,
  bindings: Binding[],
  info?: ObjectInfo,
): Record<string, InputMeta> {
  const meta: Record<string, InputMeta> = {};
  const bindingsOf = bindingIndex(bindings);
  const counts = graphNodes(workflow).flatMap((node) => {
    const own = bindingsOf(node);
    const length = own.get('length');
    return length && setsFrames(node, info, own) ? [{ node, binding: length }] : [];
  });
  for (const { node, binding } of counts)
    meta[binding.id] = {
      unit: 'frames',
      ...(counts.length > 1 ? { context: `node ${node.id}` } : {}),
    };
  for (const group of durationGroups(workflow, bindings)) {
    const all = [...group.members, ...group.wired];
    const wired = group.wired.map((member) => `the ${member.role}`).join(' and ');
    const note =
      all.length < 2
        ? undefined
        : wired
          ? `${group.relation} The count of ${wired} comes from a wire, so keep this one equal to it.`
          : `${group.relation} Duration sets ${all.length === 2 ? 'both' : 'all of them'}.`;
    for (const member of all)
      meta[member.binding.id] = {
        context: `${group.family} ${member.role}, node ${member.node.id}`,
        unit: 'frames',
        ...(note ? { note } : {}),
      };
    for (const rate of group.rates)
      meta[rate.binding.id] = {
        context: `${group.family} ${rate.node.type}, node ${rate.node.id}`,
        unit: 'fps',
        ...(group.rates.length > 1
          ? { note: 'The clip frame rate; keep these inputs equal.' }
          : {}),
      };
  }
  return meta;
}
