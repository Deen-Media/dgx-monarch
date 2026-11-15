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
  'X2',
  'S3',
];

type Input = 'multiline' | 'one-line' | [string, number, string];
type Node = [string, string[], Record<string, Input>];
type Spec = { input: { required: Record<string, unknown[]> }; output: string[] };

/** Each typed input's role, or null, in one reading's graph, as an API prompt Studio inspects. */
function roles(nodes: Record<string, Node>) {
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
  assert.equal(cases.length, 71);
  for (const row of cases) {
    // `unflagged` is a typed box whose schema sets no multiline flag at all.
    const typed =
      row.typed === 'unflagged'
        ? {}
        : row.typed
          ? { multiline: row.typed === 'multiline' }
          : undefined;
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

test('an input named for the negative prompt keeps that name, whichever input its text reaches', () => {
  const tail = (source: [string, number, string]): Record<string, Node> => ({
    20: ['CLIPTextEncode', ['CONDITIONING'], { text: source }],
    21: ['CLIPTextEncode', ['CONDITIONING'], { text: 'multiline' }],
    22: [
      'KSampler',
      ['LATENT'],
      { positive: ['20', 0, 'CONDITIONING'], negative: ['21', 0, 'CONDITIONING'] },
    ],
  });
  const both = { prompt: 'multiline', negative_prompt: 'multiline' } as const;
  // C1 and C2: a cloud node's prompt output, encoded locally, reaches only the positive input.
  assert.deepEqual(
    roles({ 1: ['BriaExpandImage', ['IMAGE', 'STRING'], both], ...tail(['1', 1, 'STRING']) }),
    { '1.prompt': 'prompt', '1.negative_prompt': 'negative', '21.text': 'negative' },
  );
  const edit = { ...both, structured_prompt: 'multiline' } as const;
  assert.deepEqual(
    roles({ 1: ['BriaImageEditNode', ['IMAGE', 'STRING'], edit], ...tail(['1', 1, 'STRING']) }),
    {
      '1.prompt': 'prompt',
      '1.negative_prompt': 'negative',
      '1.structured_prompt': 'prompt',
      '21.text': 'negative',
    },
  );
  // Q1: Qwen Image 2.1's encoder with its positive output alone, zeroed for the negative input.
  const qwen: Record<string, Node> = {
    1: ['TextEncodeQwenImage21', ['CONDITIONING', 'CONDITIONING', 'LATENT'], both],
    2: ['ConditioningZeroOut', ['CONDITIONING'], { conditioning: ['1', 0, 'CONDITIONING'] }],
    22: [
      'KSampler',
      ['LATENT'],
      { positive: ['1', 0, 'CONDITIONING'], negative: ['2', 0, 'CONDITIONING'] },
    ],
  };
  assert.deepEqual(roles(qwen), { '1.prompt': 'prompt', '1.negative_prompt': 'negative' });
  // C3: a cloud node alone, whose text reaches no sampler, names both boxes by their inputs.
  const kling: Record<string, Node> = {
    1: ['KlingTextToVideoNode', ['VIDEO', 'STRING', 'STRING'], both],
    2: ['SaveVideo', [], { video: ['1', 0, 'VIDEO'] }],
  };
  assert.deepEqual(roles(kling), { '1.prompt': 'prompt', '1.negative_prompt': 'negative' });
  // One way only: an encoder's `prompt` input wired to the negative side is the negative.
  const twoEdits: Record<string, Node> = {
    1: ['TextEncodeQwenImageEdit', ['CONDITIONING'], { prompt: 'multiline' }],
    2: ['TextEncodeQwenImageEdit', ['CONDITIONING'], { prompt: 'multiline' }],
    22: [
      'KSampler',
      ['LATENT'],
      { positive: ['1', 0, 'CONDITIONING'], negative: ['2', 0, 'CONDITIONING'] },
    ],
  };
  assert.deepEqual(roles(twoEdits), { '1.prompt': 'prompt', '2.prompt': 'negative' });
});

test('a class named after an object property is a class with no row', () => {
  const node = { type: 'constructor', outputTypes: ['STRING'] };
  assert.equal(carries(node, 'text'), true);
  assert.equal(carries(node, 'text', { multiline: false }), false);
});
