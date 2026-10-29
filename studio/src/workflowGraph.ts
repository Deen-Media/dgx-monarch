import type { Binding } from './model.ts';

/** The driver's node schemas, as the catalog carries them: inputs, outputs and output nodes. */
export interface NodeSchema {
  input?: { required?: Record<string, unknown>; optional?: Record<string, unknown> };
  output?: unknown[];
  output_node?: boolean;
}
export type ObjectInfo = Record<string, NodeSchema>;

/** One node that runs, read from a UI workflow or an API prompt. */
export interface GraphNode {
  /** The root graph's id, or a subgraph definition's, as binding identities use it. */
  graphId: string;
  id: string;
  /** The node class, or a subgraph definition id for a subgraph instance. */
  type: string;
  /** Input names that a wire feeds. */
  wired: Set<string>;
  /** Output slots that a wire leaves from. */
  used: Set<number>;
}

type Json = Record<string, any>;
const isObject = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

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

/**
 * Every node of the workflow that runs. A muted or bypassed node (mode 2 or 4) is left out, and
 * subgraph definitions are read as the companion indexes them.
 */
export function graphNodes(workflow: Json | null | undefined): GraphNode[] {
  if (!isObject(workflow)) return [];
  const api = apiNodes(workflow);
  if (api) {
    const used = new Map<string, Set<number>>();
    for (const [, node] of api)
      for (const value of Object.values(isObject(node.inputs) ? node.inputs : {}))
        if (Array.isArray(value) && value.length === 2 && Number.isInteger(value[1])) {
          const source = String(value[0]);
          used.set(source, (used.get(source) ?? new Set()).add(value[1]));
        }
    return api.map(([id, node]) => ({
      graphId: 'root',
      id,
      type: node.class_type,
      wired: new Set(
        Object.entries(isObject(node.inputs) ? node.inputs : {})
          .filter(([, value]) => Array.isArray(value))
          .map(([name]) => name),
      ),
      used: used.get(id) ?? new Set(),
    }));
  }
  const nodes: GraphNode[] = [];
  const visit = (graph: Json, fallback: string, depth: number) => {
    if (depth > 32) return;
    const graphId = graph.id === undefined ? fallback : String(graph.id);
    for (const node of Array.isArray(graph.nodes) ? graph.nodes : []) {
      if (!isObject(node) || typeof node.type !== 'string') continue;
      if (node.mode !== 2 && node.mode !== 4)
        nodes.push({
          graphId,
          id: String(node.id),
          type: node.type,
          wired: new Set(
            (Array.isArray(node.inputs) ? node.inputs : [])
              .filter((socket: unknown) => isObject(socket) && socket.link != null)
              .map((socket: Json) => String(socket.name)),
          ),
          used: new Set(
            (Array.isArray(node.outputs) ? node.outputs : []).flatMap((socket: unknown, slot) =>
              isObject(socket) && Array.isArray(socket.links) && socket.links.length ? [slot] : [],
            ),
          ),
        });
      if (isObject(node.subgraph)) visit(node.subgraph, `${graphId}/node/${node.id}`, depth + 1);
    }
    const definitions = isObject(graph.definitions) ? graph.definitions.subgraphs : undefined;
    (Array.isArray(definitions) ? definitions : []).forEach((subgraph: unknown, index: number) => {
      if (isObject(subgraph)) visit(subgraph, `${graphId}/definition/${index}`, depth + 1);
    });
  };
  visit(workflow, 'root', 0);
  return nodes;
}

/**
 * A lookup of each node's bindings by input name, built once for all bindings. A binding names its
 * node by graph and node identity; a subgraph used twice gives its first instance.
 */
export function bindingIndex(bindings: Binding[]): (node: GraphNode) => Map<string, Binding> {
  const index = new Map<string, Map<string, Binding>>();
  for (const binding of bindings) {
    if (!binding.node_id || !binding.input_name) continue;
    const key = `${binding.graph_id ?? ''}\u0000${binding.node_id}`;
    const own = index.get(key) ?? new Map<string, Binding>();
    if (!own.has(binding.input_name)) own.set(binding.input_name, binding);
    index.set(key, own);
  }
  const none = new Map<string, Binding>();
  return (node) =>
    index.get(`${node.graphId}\u0000${node.id}`) ?? index.get(`\u0000${node.id}`) ?? none;
}

/** Every input the class declares, required and optional, with its raw schema. */
export function declaredInputs(
  info: ObjectInfo | undefined,
  type: string,
): Record<string, unknown> {
  const schema = info?.[type]?.input;
  return { ...(schema?.required ?? {}), ...(schema?.optional ?? {}) };
}

/** The value a binding holds in the draft, or its graph value when the draft has none. */
export function currentValue(binding: Binding | undefined, values: Record<string, unknown>) {
  return binding ? (values[binding.id] ?? binding.value) : undefined;
}
