// Test support: the bundled workflow catalog, the recorded driver schema the Python suite
// inspects it against, and bindings shaped as the companion's inspection lists them.
import { readFile } from 'node:fs/promises';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
/** @type {Array<{ name: string; title: string; workflow: Record<string, any> }>} */
export const bundled = (await read('../../src/dgx_monarch/studio/data/workflows.json')).workflows;
/** @type {Record<string, any>} */
export const recorded = await read('../../tests/fixtures/studio/object_info_bundled.json');

/** A bundled workflow's graph by its file stem without the dgx-monarch- prefix. */
export function graph(stem) {
  const row = bundled.find((entry) => entry.name === `dgx-monarch-${stem}`);
  if (!row) throw new Error(`no bundled workflow dgx-monarch-${stem}`);
  return row.workflow;
}

const SCALAR = new Set(['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO']);

/**
 * The bindings of these node classes: each widget in the order the recorded schema declares it,
 * with that schema, and unresolved where a wire feeds it. Only for classes that draw no seed
 * control widget, so their widget values line up with their declared widgets.
 */
export function widgetBindings(workflow, types) {
  const graphId = String(workflow.id);
  return workflow.nodes
    .filter((node) => types.includes(node.type))
    .flatMap((node) => {
      const schema = recorded[node.type];
      const spec = (name) => schema.input.required?.[name] ?? schema.input.optional?.[name];
      const names = [...schema.input_order.required, ...(schema.input_order.optional ?? [])].filter(
        (name) =>
          Array.isArray(spec(name)[0]) || SCALAR.has(spec(name)[0]) || spec(name)[1]?.widgetType,
      );
      if (names.length !== node.widgets_values.length)
        throw new Error(`${node.type} has ${node.widgets_values.length} widget values`);
      return names.map((name, index) => {
        const [kind, options = {}] = spec(name);
        const wired = (node.inputs ?? []).some((socket) => socket.name === name && socket.link);
        return {
          id: `${graphId}:${node.id}:${name}`,
          graph_id: graphId,
          node_id: String(node.id),
          input_name: name,
          label: name,
          value: node.widgets_values[index],
          resolved: !wired,
          schema: Array.isArray(kind)
            ? { type: 'COMBO', choices: kind, ...options }
            : { ...options, type: options.widgetType ?? kind },
        };
      });
    });
}

/** The format choices of the save nodes, as the inspection lists them. */
export function formatBindings(workflow) {
  const graphId = String(workflow.id);
  return workflow.nodes.flatMap((node) => {
    const values = node.widgets_values ?? [];
    const binding = (name, value) => ({
      id: `${graphId}:${node.id}:${name}`,
      graph_id: graphId,
      node_id: String(node.id),
      input_name: name,
      value,
    });
    if (node.type === 'SaveVideo')
      return [binding('format', values[1]), binding('format.codec', values[2])];
    if (node.type === 'SaveAudioAdvanced') return [binding('format', values[1])];
    return [];
  });
}
