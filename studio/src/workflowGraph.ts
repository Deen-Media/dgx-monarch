import type { Binding } from './model.ts';

/** The driver's node schemas, as the catalog carries them: inputs, outputs and output nodes. */
export interface NodeSchema {
  input?: { required?: Record<string, unknown>; optional?: Record<string, unknown> };
  output?: unknown[];
  output_node?: boolean;
}
export type ObjectInfo = Record<string, NodeSchema>;

/** Where a wired input reads from: a running node's output slot, or a primitive's value. */
export interface Source {
  key: string;
  slot: number;
  /** The wire comes from a PrimitiveNode, which holds a value and does not run. */
  primitive?: boolean;
}

/** One node that runs, or a primitive that holds a value, in one instance of its graph. */
export interface GraphNode {
  /** The root graph's id, or the id of the subgraph definition that holds the node. */
  graphId: string;
  /** The host subgraph instance node ids from the root graph down; empty at the root. */
  path: string[];
  id: string;
  /** Unique across instances: the path and the node id. */
  key: string;
  /** The node class. */
  type: string;
  /** Each wired input's source; null when the wire reaches nothing that runs. */
  sources: Map<string, Source | null>;
  /** Output slots that a running node reads. */
  used: Set<number>;
  /** Socket types by input name, and output types by slot. */
  inputTypes: Map<string, string>;
  outputTypes: string[];
}

/** The nodes that run, as the companion compiles the graph, and who reads whom. */
export interface WorkflowGraph {
  nodes: GraphNode[];
  byKey: Map<string, GraphNode>;
  /** The nodes that read each node's outputs, by the source node's key. */
  readers: Map<string, GraphNode[]>;
}

type Json = Record<string, any>;
type Row = [string, number, string, number];
const isObject = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const keyOf = (path: string[], id: string) => JSON.stringify([...path, id]);
const PASSIVE = new Set(['Reroute', 'Note', 'MarkdownNote']);
const inactive = (node: Json) => node.mode === 2 || node.mode === 4;

/** ComfyUI's API prompt: every entry but `extra` is a node with a class_type, as the companion reads it. */
function apiNodes(workflow: Json): [string, Json][] | null {
  const entries = Object.entries(workflow).filter(
    ([key, node]) => !(key === 'extra' && isObject(node) && !('class_type' in node)),
  );
  return entries.length &&
    entries.every(([, node]) => isObject(node) && typeof node.class_type === 'string')
    ? entries
    : null;
}

/** Every input the class declares, required and optional, with its raw schema. */
export function declaredInputs(
  info: ObjectInfo | undefined,
  type: string,
): Record<string, unknown> {
  const schema = info?.[type]?.input;
  return { ...(schema?.required ?? {}), ...(schema?.optional ?? {}) };
}

function schemaTypes(info: ObjectInfo | undefined, type: string) {
  const inputs = Object.entries(declaredInputs(info, type)).flatMap(([name, spec]) =>
    Array.isArray(spec) && typeof spec[0] === 'string' ? [[name, spec[0]] as [string, string]] : [],
  );
  const outputs = info?.[type]?.output ?? [];
  return { inputTypes: new Map(inputs), outputTypes: outputs.map(String) };
}

function link(row: unknown): [string, Row] | null {
  const values = Array.isArray(row)
    ? row.slice(0, 5)
    : isObject(row)
      ? ['id', 'origin_id', 'origin_slot', 'target_id', 'target_slot'].map((key) => row[key])
      : [];
  const [id, source, slot, target, targetSlot] = values;
  return id == null || source == null || target == null
    ? null
    : [String(id), [String(source), Number(slot), String(target), Number(targetSlot)]];
}

interface Context {
  gid: string;
  graph: Json;
  path: string[];
  nodes: Map<string, Json>;
  links: Map<string, Row>;
  children: Map<string, Context>;
  parent?: Context;
  host?: Json;
}

/**
 * The workflow's running nodes. A UI workflow expands only the subgraphs that an active instance
 * reaches from the root graph, as the companion's compiler does: muted and bypassed nodes neither
 * run nor open their subgraphs, and a definition no instance uses never runs. Each wired input is
 * followed through reroutes, bypassed nodes and subgraph boundaries to the node it reads.
 */
export function workflowGraph(workflow: Json | null | undefined, info?: ObjectInfo): WorkflowGraph {
  const nodes: GraphNode[] = [];
  const byKey = new Map<string, GraphNode>();
  const readers = new Map<string, GraphNode[]>();
  const add = (node: GraphNode) => {
    nodes.push(node);
    byKey.set(node.key, node);
  };
  const finish = () => {
    for (const node of nodes)
      for (const source of node.sources.values()) {
        const from = source && !source.primitive ? byKey.get(source.key) : undefined;
        if (!source || !from) continue;
        from.used.add(source.slot);
        if (!readers.has(from.key)) readers.set(from.key, []);
        readers.get(from.key)!.push(node);
      }
    return { nodes, byKey, readers };
  };
  if (!isObject(workflow)) return finish();
  const api = apiNodes(workflow);
  if (api) {
    for (const [id, raw] of api)
      add({
        graphId: 'root',
        path: [],
        id,
        key: keyOf([], id),
        type: raw.class_type,
        sources: new Map(),
        used: new Set(),
        ...schemaTypes(info, raw.class_type),
      });
    for (const [id, raw] of api)
      for (const [name, value] of Object.entries(isObject(raw.inputs) ? raw.inputs : {}))
        if (Array.isArray(value)) {
          const from = keyOf([], String(value[0]));
          byKey
            .get(keyOf([], id))!
            .sources.set(
              name,
              byKey.has(from) && Number.isInteger(value[1]) ? { key: from, slot: value[1] } : null,
            );
        }
    return finish();
  }
  // Index the graph definitions as the companion does: the root, its definitions and any graph
  // embedded in a node, each by its id or by the place that holds it.
  const graphs = new Map<string, Json>();
  const index = (graph: Json, fallback: string, depth: number) => {
    const gid = graph.id === undefined ? fallback : String(graph.id);
    if (depth > 32 || graphs.has(gid)) return;
    graphs.set(gid, graph);
    const definitions = isObject(graph.definitions) ? graph.definitions.subgraphs : undefined;
    (Array.isArray(definitions) ? definitions : []).forEach((item: unknown, i: number) => {
      if (isObject(item)) index(item, `${gid}/definition/${i}`, depth + 1);
    });
    for (const node of Array.isArray(graph.nodes) ? graph.nodes : [])
      if (isObject(node) && isObject(node.subgraph))
        index(node.subgraph, `${gid}/node/${node.id}`, depth + 1);
  };
  index(workflow, 'root', 0);
  const contexts: Context[] = [];
  const expand = (
    gid: string,
    path: string[],
    ancestry: Set<string>,
    parent?: Context,
    host?: Json,
  ) => {
    const graph = graphs.get(gid)!;
    const context: Context = {
      gid,
      graph,
      path,
      nodes: new Map(),
      links: new Map(),
      children: new Map(),
      parent,
      host,
    };
    for (const node of Array.isArray(graph.nodes) ? graph.nodes : [])
      if (isObject(node) && typeof node.type === 'string') context.nodes.set(String(node.id), node);
    for (const row of Array.isArray(graph.links) ? graph.links : []) {
      const parsed = link(row);
      if (parsed) context.links.set(...parsed);
    }
    contexts.push(context);
    for (const [nid, node] of context.nodes) {
      if (inactive(node)) continue;
      const child = isObject(node.subgraph)
        ? node.subgraph.id === undefined
          ? `${gid}/node/${nid}`
          : String(node.subgraph.id)
        : node.type;
      if (graphs.has(child) && !ancestry.has(child) && child !== gid && path.length < 32)
        context.children.set(
          nid,
          expand(child, [...path, nid], new Set([...ancestry, gid]), context, node),
        );
    }
    return context;
  };
  expand(graphs.keys().next().value!, [], new Set());
  const linked = (context: Context, id: unknown, seen: Set<string>): Source | null => {
    const row = context.links.get(String(id));
    return row ? source(context, row[0], row[1], seen) : null;
  };
  // Where an output slot's value comes from, as the compiler's own source lookup decides.
  const source = (
    context: Context,
    nid: string,
    slot: number,
    seen: Set<string>,
  ): Source | null => {
    const tag = keyOf(context.path, `${nid}#${slot}`);
    if (seen.has(tag) || seen.size > 256) return null;
    seen.add(tag);
    if (context.parent && nid === String(context.graph.inputNode?.id ?? '-10')) {
      const face = Array.isArray(context.graph.inputs) ? context.graph.inputs[slot] : undefined;
      const socket = (context.host?.inputs ?? []).find((item: Json) => item?.name === face?.name);
      return socket?.link != null ? linked(context.parent, socket.link, seen) : null;
    }
    const node = context.nodes.get(nid);
    if (!node || node.mode === 2) return null;
    if (node.mode === 4) {
      const type = node.outputs?.[slot]?.type;
      const through = (node.inputs ?? []).filter(
        (item: Json) => item?.type === type && item.link != null,
      );
      return through.length === 1 ? linked(context, through[0].link, seen) : null;
    }
    const child = context.children.get(nid);
    if (child) {
      const exit = String(child.graph.outputNode?.id ?? '-20');
      const rows = [...child.links.values()].filter(
        ([, , target, at]) => target === exit && at === slot,
      );
      return rows.length === 1 ? source(child, rows[0][0], rows[0][1], seen) : null;
    }
    if (node.type === 'Reroute')
      return slot === 0 && node.inputs?.[0]?.link != null
        ? linked(context, node.inputs[0].link, seen)
        : null;
    return {
      key: keyOf(context.path, nid),
      slot,
      ...(node.type === 'PrimitiveNode' ? { primitive: true } : {}),
    };
  };
  for (const context of contexts)
    for (const [nid, node] of context.nodes) {
      if (inactive(node) || context.children.has(nid) || PASSIVE.has(node.type)) continue;
      const sockets: Json[] = (Array.isArray(node.inputs) ? node.inputs : []).filter(isObject);
      const fromSchema = schemaTypes(info, node.type);
      add({
        graphId: context.gid,
        path: context.path,
        id: nid,
        key: keyOf(context.path, nid),
        type: node.type,
        sources: new Map(
          sockets
            .filter((socket) => socket.link != null)
            .map((socket) => [String(socket.name), linked(context, socket.link, new Set())]),
        ),
        used: new Set(),
        inputTypes: sockets.some((socket) => socket.type)
          ? new Map(sockets.map((socket) => [String(socket.name), String(socket.type)]))
          : fromSchema.inputTypes,
        outputTypes:
          Array.isArray(node.outputs) && node.outputs.length
            ? node.outputs.map((socket: Json) => String(socket?.type))
            : fromSchema.outputTypes,
      });
    }
  return finish();
}

/** The running nodes of a workflow; see workflowGraph. */
export function graphNodes(workflow: Json | null | undefined, info?: ObjectInfo): GraphNode[] {
  return workflowGraph(workflow, info).nodes;
}

/**
 * A lookup of each node's bindings by input name. A binding names its node by graph id, subgraph
 * instance path and node id, so each instance of a repeated subgraph keeps its own bindings.
 */
export function bindingIndex(bindings: Binding[]): (node: GraphNode) => Map<string, Binding> {
  const index = new Map<string, Map<string, Binding>>();
  for (const binding of bindings) {
    if (!binding.node_id || !binding.input_name) continue;
    const key = `${binding.graph_id ?? ''}\u0000${keyOf(binding.instance_path ?? [], binding.node_id)}`;
    const own = index.get(key) ?? new Map<string, Binding>();
    if (!own.has(binding.input_name)) own.set(binding.input_name, binding);
    index.set(key, own);
  }
  const none = new Map<string, Binding>();
  return (node) =>
    index.get(`${node.graphId}\u0000${node.key}`) ?? index.get(`\u0000${node.key}`) ?? none;
}

/** The value a binding holds in the draft, or its graph value when the draft has none. */
export function currentValue(binding: Binding | undefined, values: Record<string, unknown>) {
  return binding ? (values[binding.id] ?? binding.value) : undefined;
}

/** A node's name for people: its id, after the subgraph instances that hold it. */
export function nodeName(node: GraphNode): string {
  return [...node.path, node.id].join('/');
}
