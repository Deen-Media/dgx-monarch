import type { Binding } from './model.ts';
import {
  bindingIndex,
  currentValue,
  nodeName,
  workflowGraph,
  type GraphNode,
  type ObjectInfo,
  type WorkflowGraph,
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
  /** For a wired input, the value binding of the primitive node that feeds it, when one does. */
  feed?: Binding;
}
/** One clip's frame-count inputs in a workflow, which a Duration control sets together. */
export interface DurationGroup {
  /** Unique in the workflow, for element ids. */
  key: string;
  family: string;
  /** The clip's number when the workflow holds more than one clip of this family. */
  clip?: number;
  /** The inputs Duration writes. */
  members: DurationMember[];
  /** Members a wire feeds, which no draft value can set. */
  wired: DurationMember[];
  /** A fixed frame rate, or null when `rates` hold it. */
  fps: number | null;
  rates: DurationMember[];
  /** The sampler nodes the clip's inputs feed; empty when Studio cannot tell which. */
  samplers: string[];
  grid: { base: number; step: number };
  min: number;
  max: number;
  relation: string;
}

const bound = (binding: Binding, key: 'min' | 'max') =>
  binding.schema?.[key] ?? binding[key] ?? binding.validation?.[key];

/**
 * A node that samples a clip: it takes a model or a guider, stock or DGX Monarch's own socket
 * type, and returns a latent.
 */
const samples = (node: GraphNode) =>
  node.outputTypes.includes('LATENT') &&
  [...node.inputTypes.values()].some((type) => /^(DGXM_)?(MODEL|GUIDER)$/.test(type));

/** The samplers a node's outputs reach, following the wires up to the first sampler on each path. */
function samplersOf(start: GraphNode, graph: WorkflowGraph): Set<string> {
  const found = new Set<string>();
  const seen = new Set([start.key]);
  const queue = [start];
  while (queue.length)
    for (const reader of graph.readers.get(queue.shift()!.key) ?? []) {
      if (seen.has(reader.key)) continue;
      seen.add(reader.key);
      if (samples(reader)) found.add(reader.key);
      else queue.push(reader);
    }
  return found;
}

/**
 * The clip frame-count groups a workflow holds. Inputs of one family join a group only when they
 * feed the same sampler, so two clips get two groups. An input whose sampler Studio cannot find
 * stays alone, with no sampler, and no Duration control sets it. Each instance of a repeated
 * subgraph keeps its own nodes, so each gets its own groups.
 */
export function durationGroups(
  workflow: Record<string, any> | null | undefined,
  bindings: Binding[],
  info?: ObjectInfo,
): DurationGroup[] {
  const graph = workflowGraph(workflow, info);
  const bindingsOf = bindingIndex(bindings);
  const groups: DurationGroup[] = [];
  for (const rule of RULES) {
    const found = (entries: Array<{ type: string; input: string; role?: string }>) =>
      graph.nodes.flatMap((node) =>
        entries
          .filter((entry) => entry.type === node.type)
          .flatMap((entry): DurationMember[] => {
            const binding = bindingsOf(node).get(entry.input);
            if (!binding) return [];
            const source = node.sources.get(entry.input);
            const primitive = source?.primitive ? graph.byKey.get(source.key) : undefined;
            const feed = primitive && bindingsOf(primitive).get('value');
            return [{ binding, node, role: entry.role ?? 'frame rate', ...(feed ? { feed } : {}) }];
          }),
      );
    const rates = typeof rule.fps === 'number' ? [] : found(rule.fps);
    const entries = [
      ...found(rule.members).map((entry) => ({ entry, rate: false })),
      ...rates.map((entry) => ({ entry, rate: true })),
    ].map((item) => ({ ...item, samplers: samplersOf(item.entry.node, graph) }));
    // Entries that feed a common sampler belong to one clip.
    const clips: Array<typeof entries> = [];
    for (const item of entries) {
      const joined = clips.filter((clip) =>
        clip.some((other) => [...item.samplers].some((key) => other.samplers.has(key))),
      );
      clips.splice(0, clips.length, ...clips.filter((clip) => !joined.includes(clip)), [
        ...joined.flat(),
        item,
      ]);
    }
    const found_ = clips.filter((clip) => clip.some((item) => !item.rate));
    found_.forEach((clip, index) => {
      const members = clip.filter((item) => !item.rate).map((item) => item.entry);
      const editable = members.filter((member) => member.binding.resolved !== false);
      const limits = editable.map((member) => member.binding);
      groups.push({
        key: `${rule.family.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${index + 1}`,
        family: rule.family,
        ...(found_.length > 1 ? { clip: index + 1 } : {}),
        members: editable,
        wired: members.filter((member) => member.binding.resolved === false),
        fps: typeof rule.fps === 'number' ? rule.fps : null,
        rates: clip.filter((item) => item.rate).map((item) => item.entry),
        samplers: [...new Set(clip.flatMap((item) => [...item.samplers]))],
        grid: rule.grid,
        min: Math.max(rule.grid.base, ...limits.map((b) => Number(bound(b, 'min') ?? -Infinity))),
        max: Math.min(...limits.map((b) => Number(bound(b, 'max') ?? Infinity))),
        relation: rule.relation,
      });
    });
  }
  return groups;
}

/**
 * Whether one Duration control can set the clip: it knows the clip's sampler, no member is wired,
 * and one member at least is editable.
 */
export function synchronized(group: DurationGroup): boolean {
  return group.members.length > 0 && group.wired.length === 0 && group.samplers.length > 0;
}

/** The family, with the clip's number when the workflow holds more than one, as labels show it. */
export const clipName = (group: DurationGroup) =>
  `${group.family}${group.clip ? ` clip ${group.clip}` : ''}`;

/** Each member's frame count in the draft, or null where it holds no number. */
export function clipFrames(group: DurationGroup, values: Record<string, unknown>) {
  return group.members.map((member) => {
    const value = currentValue(member.binding, values);
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  });
}

/**
 * The clip's frame rate, or the reason it is unknown. Every rate input must hold the same rate. A
 * wired rate counts only when a primitive node feeds it; any other wire leaves the rate unknown.
 */
export function clipRate(
  group: DurationGroup,
  values: Record<string, unknown>,
): { fps: number | null; reason?: string } {
  if (group.fps !== null) return { fps: group.fps };
  if (!group.rates.length) return { fps: null, reason: 'No frame rate input sets this clip.' };
  const rates: unknown[] = [];
  for (const rate of group.rates) {
    const binding = rate.binding.resolved === false ? rate.feed : rate.binding;
    if (!binding)
      return {
        fps: null,
        reason: `A wire sets frame_rate on node ${nodeName(rate.node)}, so the clip's frame rate is unknown.`,
      };
    rates.push(currentValue(binding, values));
  }
  const unique = [...new Set(rates)];
  if (unique.length > 1)
    return { fps: null, reason: `The frame rate inputs differ: ${unique.join(' and ')} fps.` };
  const fps = unique[0];
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
  const counts = workflowGraph(workflow, info).nodes.flatMap((node) => {
    const own = bindingsOf(node);
    const length = own.get('length');
    return length && setsFrames(node, info, own) ? [{ node, binding: length }] : [];
  });
  for (const { node, binding } of counts)
    meta[binding.id] = {
      unit: 'frames',
      ...(counts.length > 1 ? { context: `node ${nodeName(node)}` } : {}),
    };
  for (const group of durationGroups(workflow, bindings, info)) {
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
        context: `${clipName(group)} ${member.role}, node ${nodeName(member.node)}`,
        unit: 'frames',
        ...(note ? { note } : {}),
      };
    for (const rate of group.rates)
      meta[rate.binding.id] = {
        context: `${clipName(group)} ${rate.node.type}, node ${nodeName(rate.node)}`,
        unit: 'fps',
        ...(group.rates.length > 1
          ? { note: 'The clip frame rate; keep these inputs equal.' }
          : {}),
      };
  }
  return meta;
}
