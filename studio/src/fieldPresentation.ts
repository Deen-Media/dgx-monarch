import { bindingType, fieldLabel } from './model.ts';
import type { Binding, Draft } from './model.ts';
import { bindingIndex, workflowGraph } from './workflowGraph.ts';
import type { GraphNode, ObjectInfo } from './workflowGraph.ts';

/** A text input's part in a job: the prompt, or the negative prompt. */
export type PromptRole = 'prompt' | 'negative';

/** How Create's Inputs and All inputs tabs name one field. */
export interface FieldPresentation {
  /** The name the field shows. */
  label: string;
  /** The node and the input that hold the field, such as `CLIPTextEncode · text · node 5`. */
  source: string;
  role?: PromptRole;
}

type Json = Record<string, any>;
type Readers = Map<string, Array<{ node: GraphNode; input: string; slot: number }>>;
const ROLE_NAMES: Record<PromptRole, string> = { prompt: 'Prompt', negative: 'Negative prompt' };
const SOCKET_ROLES: Record<string, PromptRole> = { positive: 'prompt', negative: 'negative' };
/** Input names that say the role, for a node whose text reaches both sampler inputs, or neither. */
const NAMED_ROLES: Record<string, PromptRole> = {
  prompt: 'prompt',
  positive: 'prompt',
  positive_prompt: 'prompt',
  negative: 'negative',
  negative_prompt: 'negative',
};
// Text reaches a sampler through a text encoder and a few conditioning nodes, such as Flux
// guidance or a zeroed copy; a longer path is not followed.
const MAX_HOPS = 8;
const FOLLOWED = new Set(['CONDITIONING', 'STRING']);
const isObject = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

/** A text input that Create edits in a text box. */
export const isText = (b: Binding) => b.resolved !== false && bindingType(b) === 'STRING';

/** The label the workflow's layout gives a field, or '' when the field keeps its input name. */
export function layoutLabel(b: Binding): string {
  const label = b.display?.label ?? b.label;
  return typeof label === 'string' && label && label !== b.input_name ? label : '';
}

/**
 * Who reads each node's outputs, on which input and from which output slot, following text and
 * conditioning only.
 */
function inputReaders(nodes: GraphNode[]): Readers {
  const readers: Readers = new Map();
  for (const node of nodes)
    for (const [input, source] of node.sources) {
      const type = node.inputTypes.get(input);
      if (!source || (type && !FOLLOWED.has(type) && !SOCKET_ROLES[input])) continue;
      if (!readers.has(source.key)) readers.set(source.key, []);
      readers.get(source.key)!.push({ node, input, slot: source.slot });
    }
  return readers;
}

/**
 * The roles a node's text reaches. Each output follows its own branches, within a few nodes, to
 * the nearest sampler inputs named `positive` or `negative`, so one output that reaches a sampler
 * first does not cut short another, as when a node encodes both texts and only its negative output
 * passes through a zeroed copy. Within one output the nearest input decides: a prompt whose zeroed
 * copy feeds the negative input reaches the positive input first, so it stays the prompt.
 */
function reachedRoles(readers: Readers, start: GraphNode): Set<PromptRole> {
  const roles = new Set<PromptRole>();
  const first = readers.get(start.key) ?? [];
  for (const slot of new Set(first.map((reader) => reader.slot))) {
    let frontier = first.filter((reader) => reader.slot === slot);
    const seen = new Set([start.key]);
    for (let hop = 0; hop < MAX_HOPS && frontier.length; hop++) {
      const found = new Set<PromptRole>();
      const next: typeof frontier = [];
      for (const { node, input } of frontier) {
        if (SOCKET_ROLES[input]) found.add(SOCKET_ROLES[input]);
        else if (!seen.has(node.key)) {
          seen.add(node.key);
          next.push(...(readers.get(node.key) ?? []));
        }
      }
      if (found.size) {
        for (const role of found) roles.add(role);
        break;
      }
      frontier = next;
    }
  }
  return roles;
}

/**
 * Each text input's role in the graph. A node whose text reaches one sampler input gives every
 * text input it holds that role. When it reaches both, as a node that encodes a prompt and a
 * negative prompt together does, or neither, an input named after a role takes that role.
 */
export function graphRoles(
  workflow: Json | null | undefined,
  bindings: Binding[],
  info?: ObjectInfo,
): Map<string, PromptRole> {
  const roles = new Map<string, PromptRole>();
  if (!isObject(workflow)) return roles;
  const graph = workflowGraph(workflow, info);
  const readers = inputReaders(graph.nodes);
  const own = bindingIndex(bindings.filter(isText));
  for (const node of graph.nodes) {
    const texts = [...own(node).values()];
    if (!texts.length) continue;
    const reached = reachedRoles(readers, node);
    for (const binding of texts) {
      const role = reached.size === 1 ? [...reached][0] : NAMED_ROLES[binding.input_name ?? ''];
      if (role) roles.set(binding.id, role);
    }
  }
  return roles;
}

/**
 * The prompt and negative prompt inputs: the ones chosen under Ordered prompt styles, and for a
 * role not chosen there, the ones the graph shows. A role the graph gives several inputs stays
 * with each of them.
 */
export function promptRoles(
  draft: Pick<Draft, 'workflow' | 'style_bindings'>,
  bindings: Binding[],
  info?: ObjectInfo,
): Map<string, PromptRole> {
  const texts = bindings.filter(isText);
  const chosen = new Map<string, PromptRole>();
  const named = (id: string | undefined) =>
    id ? texts.find((b) => b.id === id || b.aliases?.includes(id)) : undefined;
  const prompt = named(draft.style_bindings?.prompt);
  const negative = named(draft.style_bindings?.negative_prompt);
  if (prompt) chosen.set(prompt.id, 'prompt');
  if (negative && !chosen.has(negative.id)) chosen.set(negative.id, 'negative');
  const taken = new Set(chosen.values());
  for (const [id, role] of graphRoles(draft.workflow, texts, info))
    if (!taken.has(role) && !chosen.has(id)) chosen.set(id, role);
  return chosen;
}

/** The workflow's nodes by graph id and node id, as bindings name them. */
function rawNodes(workflow: Json | null | undefined): Map<string, Json> {
  const nodes = new Map<string, Json>();
  if (!isObject(workflow)) return nodes;
  if (!Array.isArray(workflow.nodes)) {
    for (const [id, node] of Object.entries(workflow))
      if (isObject(node) && typeof node.class_type === 'string') nodes.set(`root\u0000${id}`, node);
    return nodes;
  }
  const visit = (graph: Json, fallback: string, depth: number) => {
    const gid = graph.id === undefined ? fallback : String(graph.id);
    if (depth > 32) return;
    for (const node of Array.isArray(graph.nodes) ? graph.nodes.filter(isObject) : []) {
      if (!nodes.has(`${gid}\u0000${node.id}`)) nodes.set(`${gid}\u0000${node.id}`, node);
      if (isObject(node.subgraph)) visit(node.subgraph, `${gid}/node/${node.id}`, depth + 1);
    }
    const definitions = isObject(graph.definitions) ? graph.definitions.subgraphs : undefined;
    (Array.isArray(definitions) ? definitions : []).forEach((item: unknown, i: number) => {
      if (isObject(item)) visit(item, `${gid}/definition/${i}`, depth + 1);
    });
  };
  visit(workflow, 'root', 0);
  return nodes;
}

/** A node's name for people: its title, its class's display name, or its class. */
function nodeTitle(node: Json | undefined, info?: ObjectInfo): string {
  const title = node?.title ?? node?._meta?.title;
  if (typeof title === 'string' && title.trim()) return title.trim();
  const type = node?.type ?? node?.class_type;
  const shown = (info?.[type] as { display_name?: unknown } | undefined)?.display_name;
  return typeof shown === 'string' && shown ? shown : typeof type === 'string' ? type : '';
}

/** The first suffix that tells every field of a group apart, or the last one offered. */
function distinct(group: Binding[], suffixes: Array<(b: Binding) => string>) {
  const pick = suffixes.find((suffix) => new Set(group.map(suffix)).size === group.length);
  return pick ?? suffixes[suffixes.length - 1];
}

/**
 * Each field's name and source. A label the workflow's layout sets wins. A text input with a known
 * role reads Prompt or Negative prompt; when several share a role, each adds its input name, its
 * node's title or its node id. A multi-line text input without a role, and a text input whose name
 * another text input repeats, read their node's title or class and their input name, with the node
 * id when that still repeats. Every other field keeps its input name.
 */
export function presentFields(
  draft: Pick<Draft, 'workflow' | 'style_bindings'>,
  bindings: Binding[],
  info?: ObjectInfo,
): Map<string, FieldPresentation> {
  const roles = promptRoles(draft, bindings, info);
  const nodes = rawNodes(draft.workflow);
  const title = (b: Binding) =>
    nodeTitle(
      nodes.get(`${b.graph_id ?? 'root'}\u0000${b.node_id}`) ?? nodes.get(`root\u0000${b.node_id}`),
      info,
    );
  const input = (b: Binding) => String(b.input_name ?? b.label ?? b.id);
  const node = (b: Binding) => `node ${[...(b.instance_path ?? []), b.node_id].join('/')}`;
  const where = (b: Binding) => [title(b), input(b)].filter(Boolean).join(' · ');
  const plain = bindings.filter((b) => isText(b) && !layoutLabel(b) && !roles.has(b.id));
  const named = plain.filter(
    (b) => b.schema?.multiline || plain.some((o) => o !== b && input(o) === input(b)),
  );
  const labels = new Map<string, string>();
  for (const role of ['prompt', 'negative'] as const) {
    const group = bindings.filter((b) => roles.get(b.id) === role && !layoutLabel(b));
    const suffix = distinct(group, [input, title, node]);
    for (const b of group)
      labels.set(b.id, group.length > 1 ? `${ROLE_NAMES[role]} · ${suffix(b)}` : ROLE_NAMES[role]);
  }
  for (const b of named) {
    const repeated = named.some((o) => o !== b && where(o) === where(b));
    labels.set(b.id, repeated && b.node_id ? `${where(b)} · ${node(b)}` : where(b));
  }
  const result = new Map<string, FieldPresentation>();
  for (const b of bindings) {
    const role = roles.get(b.id);
    const label = layoutLabel(b) || labels.get(b.id) || String(fieldLabel(b));
    result.set(b.id, {
      label,
      source: b.node_id ? `${where(b)} · ${node(b)}` : where(b),
      ...(role ? { role } : {}),
    });
  }
  return result;
}
