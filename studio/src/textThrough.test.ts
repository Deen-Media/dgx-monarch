import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { graphRoles } from './fieldPresentation.ts';
import { carries } from './textThrough.ts';
import type { Binding } from './model.ts';

// tests/test_studio_template_layouts.py runs the same file through the layout check's rule.
const file = new URL('../../tests/fixtures/studio/text_through_cases.json', import.meta.url);
const { cases, readings } = JSON.parse(readFileSync(file, 'utf8'));
/** The lead's ruled readings, in the file's order: each has one graph. */
const RULED = [
  'Soft Switch',
  'Pack Join Text',
  'wildcard_text',
  'positive',
  'row 5',
  'P1',
  'P1b',
  'P2',
  'P2b',
  'P3',
  'P3b',
  'P4',
  'P5r',
  'P6',
  'P7',
  'P8',
  'P10',
  'P11',
  'P12',
  'P13',
  'encoder system_prompt',
];

type Input = 'multiline' | 'one-line' | [string, number, string];
type Spec = { input: { required: Record<string, unknown[]> }; output: string[] };

/** Each typed input's role, or null, in one reading's graph, as an API prompt Studio inspects. */
function roles(nodes: Record<string, [string, string[], Record<string, Input>]>) {
  const workflow: Record<string, object> = {};
  const info: Record<string, Spec> = {};
  const bindings: Binding[] = [];
  for (const [id, [type, outputs, inputs]] of Object.entries(nodes)) {
    const spec = (info[type] ??= { input: { required: {} }, output: outputs });
    const values: Record<string, unknown> = {};
    for (const [name, input] of Object.entries(inputs)) {
      if (Array.isArray(input)) {
        values[name] = [input[0], input[1]];
        spec.input.required[name] = [input[2]];
        continue;
      }
      const schema = { type: 'STRING', multiline: input === 'multiline' };
      values[name] = 'text';
      spec.input.required[name] = ['STRING', { multiline: schema.multiline }];
      bindings.push({ id: `${id}.${name}`, node_id: id, input_name: name, resolved: true, schema });
    }
    workflow[id] = { class_type: type, inputs: values };
  }
  const found = graphRoles(workflow, bindings, info);
  return Object.fromEntries(bindings.map((b) => [b.id, found.get(b.id) ?? null]));
}

test('the text-through cases hold for Studio, as they do for the layout check', () => {
  assert.equal(cases.length, 69);
  for (const row of cases) {
    const typed = row.typed ? { multiline: row.typed === 'multiline' } : undefined;
    const node = { type: row.class, outputTypes: row.outputs };
    assert.equal(carries(node, row.input, typed), row.carries, JSON.stringify(row));
  }
});

test('every ruled reading holds in its graph: Prompt, Negative prompt, or no role', () => {
  assert.deepEqual(
    readings.map((row: { rule: string }) => row.rule),
    RULED,
  );
  for (const { rule, nodes, roles: expected } of readings)
    assert.deepEqual(roles(nodes), expected, rule);
});

test('a class named after an object property is a class with no row', () => {
  const node = { type: 'constructor', outputTypes: ['STRING'] };
  assert.equal(carries(node, 'text'), true);
  assert.equal(carries(node, 'text', { multiline: false }), false);
});
