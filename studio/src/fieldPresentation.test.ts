import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bundled, graph, recorded, widgetBindings } from './workflowFixtures.js';
import { carries, graphRoles, presentFields, promptRoles } from './fieldPresentation.ts';
import { fieldIssue, issueText } from './fieldIssues.ts';
import type { Binding } from './model.ts';

const KREA = 'krea-2-text-to-image';
const labels = (names: Map<string, { label: string }>) =>
  Object.fromEntries([...names].map(([id, entry]) => [id, entry.label]));
/** The Krea 2 template's two text inputs, both named `text`, as the tester saw them. */
function krea() {
  const workflow = graph('krea2-t2i');
  return { workflow, bindings: widgetBindings(workflow, ['CLIPTextEncode']) as Binding[] };
}

/**
 * Every text input of a packaged template, shaped as the inspection lists it, prominent when its
 * App Mode layout lists it: enough to name the text fields without the companion.
 */
function textBindings(workflow: Record<string, any>): Binding[] {
  const layout = (workflow.extra?.linearData?.inputs ?? []).map(
    ([node, input]: [unknown, string]) => `${node}:${input}`,
  );
  return workflow.nodes.flatMap((node: Record<string, any>) => {
    const schema = recorded[node.type]?.input;
    if (!schema) return [];
    const specs = { ...(schema.required ?? {}), ...(schema.optional ?? {}) };
    return Object.entries(specs)
      .filter(([, spec]) => (spec as unknown[])[0] === 'STRING')
      .map(([name, spec]) => ({
        id: `${workflow.id}:${node.id}:${name}`,
        graph_id: String(workflow.id),
        node_id: String(node.id),
        input_name: name,
        label: name,
        prominent: layout.includes(`${node.id}:${name}`),
        resolved: !(node.inputs ?? []).some((s: any) => s.name === name && s.link != null),
        schema: { ...((spec as unknown[])[1] as object), type: 'STRING' },
      }));
  });
}

test('the Krea 2 boxes read Prompt and Negative prompt, from the sampler inputs they feed', () => {
  const { workflow, bindings } = krea();
  const names = presentFields({ workflow }, bindings, recorded);
  assert.deepEqual(labels(names), {
    [`${KREA}:5:text`]: 'Prompt',
    [`${KREA}:6:text`]: 'Negative prompt',
  });
  assert.equal(names.get(`${KREA}:5:text`)?.role, 'prompt');
  assert.equal(names.get(`${KREA}:6:text`)?.role, 'negative');
  assert.equal(names.get(`${KREA}:5:text`)?.source, 'CLIPTextEncode · text · node 5');
});

test('without a known role, each text box names its node: title or class, input, then node id', () => {
  const { workflow, bindings } = krea();
  // The same graph with the encoders wired to nothing that samples: no role is known.
  const unwired = { ...workflow, links: workflow.links.filter((row: number[]) => row[3] !== 8) };
  const names = presentFields({ workflow: unwired }, bindings, recorded);
  assert.deepEqual(labels(names), {
    [`${KREA}:5:text`]: 'CLIPTextEncode · text · node 5',
    [`${KREA}:6:text`]: 'CLIPTextEncode · text · node 6',
  });
  assert.equal(names.get(`${KREA}:5:text`)?.role, undefined);
  // A titled node uses its title, and a driver's display name stands in for a missing title.
  const titled = {
    ...unwired,
    nodes: unwired.nodes.map((node: Record<string, any>) =>
      node.id === 5 ? { ...node, title: 'Scene' } : node,
    ),
  };
  const info = {
    ...recorded,
    CLIPTextEncode: { ...recorded.CLIPTextEncode, display_name: 'CLIP Text Encode (Prompt)' },
  };
  assert.deepEqual(labels(presentFields({ workflow: titled }, bindings, info)), {
    [`${KREA}:5:text`]: 'Scene · text',
    [`${KREA}:6:text`]: 'CLIP Text Encode (Prompt) · text',
  });
});

test('the inputs chosen under Ordered prompt styles decide the roles they name', () => {
  const { workflow, bindings } = krea();
  // Choosing the second box as the prompt makes it Prompt; the first loses the graph's role.
  const swapped = presentFields(
    { workflow, style_bindings: { prompt: `${KREA}:6:text` } },
    bindings,
    recorded,
  );
  assert.deepEqual(labels(swapped), {
    [`${KREA}:5:text`]: 'CLIPTextEncode · text',
    [`${KREA}:6:text`]: 'Prompt',
  });
  // A chosen prompt alone leaves the graph's negative prompt in place.
  const roles = promptRoles(
    { workflow, style_bindings: { prompt: `${KREA}:5:text` } },
    bindings,
    recorded,
  );
  assert.deepEqual(
    [...roles],
    [
      [`${KREA}:5:text`, 'prompt'],
      [`${KREA}:6:text`, 'negative'],
    ],
  );
});

test('a label the layout sets wins, and other fields keep their input names', () => {
  const { workflow, bindings } = krea();
  const relabelled = bindings.map((b) =>
    b.node_id === '5' ? { ...b, display: { label: 'Scene description' } } : b,
  );
  const seed = {
    id: `${KREA}:8:seed`,
    graph_id: KREA,
    node_id: '8',
    input_name: 'seed',
    label: 'seed',
    type: 'INT',
  };
  const names = presentFields({ workflow }, [...relabelled, seed], recorded);
  assert.equal(names.get(`${KREA}:5:text`)?.label, 'Scene description');
  assert.equal(names.get(`${KREA}:5:text`)?.role, 'prompt');
  assert.equal(names.get(`${KREA}:6:text`)?.label, 'Negative prompt');
  assert.equal(names.get(seed.id)?.label, 'seed');
  assert.equal(names.get(seed.id)?.source, 'DGXMonarchKSampler · seed · node 8');
});

test('an API prompt, as Reuse settings loads one, names its roles and titles the same way', () => {
  const prompt = {
    '5': {
      class_type: 'CLIPTextEncode',
      inputs: { text: 'a fox', clip: ['4', 0] },
      _meta: { title: 'Scene' },
    },
    '6': { class_type: 'CLIPTextEncode', inputs: { text: '', clip: ['4', 0] } },
    '7': { class_type: 'CLIPTextEncode', inputs: { text: 'unused', clip: ['4', 0] } },
    '8': { class_type: 'KSampler', inputs: { positive: ['5', 0], negative: ['6', 0] } },
  };
  const text = (id: string) => ({
    id: `root:${id}:text`,
    graph_id: 'root',
    node_id: id,
    input_name: 'text',
    label: 'text',
    schema: { type: 'STRING', multiline: true },
  });
  const names = presentFields({ workflow: prompt }, ['5', '6', '7'].map(text), recorded);
  assert.deepEqual(labels(names), {
    'root:5:text': 'Prompt',
    'root:6:text': 'Negative prompt',
    'root:7:text': 'CLIPTextEncode · text',
  });
  assert.equal(names.get('root:5:text')?.source, 'Scene · text · node 5');
});

test('one node that encodes both texts names them by input; a shared role adds the input', () => {
  const both = graph('qwen-image21-t2i');
  const roles = graphRoles(both, textBindings(both), recorded);
  assert.equal(roles.get(`${both.id}:6:prompt`), 'prompt');
  assert.equal(roles.get(`${both.id}:6:negative_prompt`), 'negative');
  const kandinsky = graph('kandinsky5-image');
  const names = presentFields({ workflow: kandinsky }, textBindings(kandinsky), recorded);
  assert.equal(names.get(`${kandinsky.id}:6:clip_l`)?.label, 'Prompt · clip_l');
  assert.equal(names.get(`${kandinsky.id}:7:qwen25_7b`)?.label, 'Negative prompt · qwen25_7b');
});

test('each output finds its own role, so a zeroed negative branch keeps Negative prompt', () => {
  // Qwen Image 2.1's encoder with only its negative output passing through a zeroed copy: the
  // positive output reaches the sampler first, and the negative output still reaches its input.
  const base = graph('qwen-image21-t2i');
  const zeroOut = {
    id: 10,
    type: 'ConditioningZeroOut',
    inputs: [{ name: 'conditioning', type: 'CONDITIONING', link: 6 }],
    outputs: [{ name: 'CONDITIONING', type: 'CONDITIONING', links: [11] }],
  };
  const workflow = {
    ...base,
    nodes: [
      ...base.nodes.map((node: Record<string, any>) =>
        node.id === 7
          ? {
              ...node,
              inputs: node.inputs.map((socket: Record<string, any>) =>
                socket.name === 'negative' ? { ...socket, link: 11 } : socket,
              ),
            }
          : node,
      ),
      zeroOut,
    ],
    links: [
      ...base.links.map((row: unknown[]) =>
        row[0] === 6 ? [6, 6, 1, 10, 0, 'CONDITIONING'] : row,
      ),
      [11, 10, 0, 7, 2, 'CONDITIONING'],
    ],
  };
  const names = presentFields({ workflow }, textBindings(workflow), recorded);
  assert.equal(names.get(`${base.id}:6:prompt`)?.label, 'Prompt');
  assert.equal(names.get(`${base.id}:6:negative_prompt`)?.label, 'Negative prompt');
  assert.equal(names.get(`${base.id}:6:negative_prompt`)?.role, 'negative');
  // A prompt whose zeroed copy feeds the negative input, as in Flux 1, stays the prompt.
  const flux = graph('flux1-t2i');
  const fluxNames = presentFields({ workflow: flux }, textBindings(flux), recorded);
  assert.deepEqual(
    [...fluxNames.values()].filter((name) => name.role).map((name) => [name.label, name.role]),
    [['Prompt', 'prompt']],
  );
});

const node = (class_type: string, inputs: Record<string, unknown>) => ({ class_type, inputs });
const text = ['STRING', {}];
/** The roles of the typed text inputs listed, as `node:input` pairs, in a synthetic API graph. */
function roles(workflow: object, typed: Array<[number, string]>, info: Record<string, any>) {
  const bindings = typed.map(([id, name]) => ({
    id: `${id}:${name}`,
    node_id: String(id),
    input_name: name,
    resolved: true,
    schema: { type: 'STRING' },
  })) as Binding[];
  return Object.fromEntries(graphRoles(workflow, bindings, { ...recorded, ...info }));
}

test('text keeps the prompt role through a node unless its input shapes the text', () => {
  // A typed prompt through Replace Text and a node pack's styler (a class Studio has no row for:
  // its text and its style carry, its separator does not) to an encoder; and an instruction to a
  // text generator, whose own text reaches a second encoder while a typed prompt exists.
  const workflow = {
    1: node('PrimitiveStringMultiline', { value: 'a red fox' }),
    2: node('StringReplace', { string: ['1', 0], find: 'fox', replace: 'cat' }),
    3: node('PackStyle', { text: ['2', 0], style: 'oil painting', separator: ', ' }),
    4: node('CLIPTextEncode', { text: ['3', 0], clip: ['11', 0] }),
    5: node('DGXMonarchKSampler', { positive: ['4', 0], negative: ['6', 0] }),
    6: node('CLIPTextEncode', { text: 'blurry', clip: ['11', 0] }),
    7: node('PrimitiveStringMultiline', { value: 'Describe the picture in one line.' }),
    8: node('TextGenerateV2', { clip: ['11', 0], prompt: ['7', 0], system_prompt: 'You caption.' }),
    9: node('CLIPTextEncode', { text: ['8', 0], clip: ['11', 0] }),
    10: node('DGXMonarchKSampler', { positive: ['9', 0], negative: ['6', 0] }),
    11: node('CLIPLoader', {}),
  };
  const info = {
    PackStyle: { input: { required: { text, style: text, separator: text } }, output: ['STRING'] },
  };
  const typed: Array<[number, string]> = [
    [1, 'value'],
    [2, 'find'],
    [2, 'replace'],
    [3, 'style'],
    [3, 'separator'],
    [6, 'text'],
    [7, 'value'],
    [8, 'system_prompt'],
  ];
  assert.deepEqual(roles(workflow, typed, info), {
    '1:value': 'prompt',
    '3:style': 'prompt',
    '6:text': 'negative',
  });
});

test('a typed prompt is the Prompt through Soft Switch, node pack text nodes and a lone generator', () => {
  // The review's probes: each typed prompt reaches the sampler's positive input as shown.
  const S = ['STRING', { multiline: true }];
  const info = {
    'Text Concatenate': {
      input: { required: { delimiter: text, text_a: S, text_b: S } },
      output: ['STRING'],
    },
    ImpactWildcardProcessor: {
      input: { required: { wildcard_text: S, populated_text: S } },
      output: ['STRING'],
    },
    PackPositive: { input: { required: { positive: S } }, output: ['STRING'] },
    ComfySoftSwitchNode: {
      input: { required: { switch: ['BOOLEAN'] }, optional: { on_true: S, on_false: S } },
      output: ['STRING'],
    },
  };
  const sampled = (source: unknown) => ({
    20: node('CLIPTextEncode', { text: source, clip: ['30', 0] }),
    21: node('CLIPTextEncode', { text: 'blurry', clip: ['30', 0] }),
    22: node('DGXMonarchKSampler', { positive: ['20', 0], negative: ['21', 0] }),
    30: node('CLIPLoader', {}),
  });
  const typed = (pairs: Array<[number, string]>) =>
    [...pairs, [21, 'text']] as Array<[number, string]>;
  const graphs: Array<[object, Array<[number, string]>, Record<string, string>]> = [
    [
      {
        1: node('PrimitiveStringMultiline', { value: 'a red fox' }),
        2: node('PrimitiveStringMultiline', { value: 'a blue fox' }),
        3: node('ComfySoftSwitchNode', { switch: true, on_true: ['1', 0], on_false: ['2', 0] }),
        ...sampled(['3', 0]),
      },
      typed([
        [1, 'value'],
        [2, 'value'],
      ]),
      { '1:value': 'prompt', '2:value': 'prompt' },
    ],
    [
      {
        1: node('PrimitiveStringMultiline', { value: 'a red fox' }),
        2: node('Text Concatenate', { delimiter: ', ', text_a: ['1', 0], text_b: 'detailed' }),
        ...sampled(['2', 0]),
      },
      typed([
        [1, 'value'],
        [2, 'text_b'],
        [2, 'delimiter'],
      ]),
      { '1:value': 'prompt', '2:text_b': 'prompt' },
    ],
    [
      {
        1: node('ImpactWildcardProcessor', { wildcard_text: 'a __animal__', populated_text: '' }),
        ...sampled(['1', 0]),
      },
      typed([[1, 'wildcard_text']]),
      { '1:wildcard_text': 'prompt' },
    ],
    [
      { 1: node('PackPositive', { positive: 'a red fox' }), ...sampled(['1', 0]) },
      typed([[1, 'positive']]),
      { '1:positive': 'prompt' },
    ],
    [
      {
        1: node('PrimitiveStringMultiline', { value: 'a red fox' }),
        2: node('TextGenerateV2', {
          clip: ['30', 0],
          prompt: ['1', 0],
          system_prompt: 'Be brief.',
        }),
        ...sampled(['2', 0]),
      },
      typed([
        [1, 'value'],
        [2, 'system_prompt'],
      ]),
      { '1:value': 'prompt' },
    ],
  ];
  for (const [workflow, pairs, expected] of graphs)
    assert.deepEqual(roles(workflow, pairs, info), { ...expected, '21:text': 'negative' });
});

test('a Basic Guider takes one conditioning, so the text that reaches it is the prompt', () => {
  // p-h3-image: the request feeds the prompt upsampler and an If/Else switch, then the
  // conditioning node, then Basic Guider (DGX Monarch), whose one input is `conditioning`.
  const workflow = graph('p-h3-image');
  const bindings = textBindings(workflow);
  const names = presentFields({ workflow }, bindings, recorded);
  const request = bindings.find((b) => b.input_name === 'value')!;
  assert.deepEqual(
    [names.get(request.id)?.label, names.get(request.id)?.role],
    ['Prompt', 'prompt'],
  );
});

test('no packaged template shows two text boxes with one name, and each names its prompt', () => {
  for (const row of bundled) {
    const bindings = textBindings(row.workflow);
    const names = presentFields({ workflow: row.workflow }, bindings, recorded);
    for (const tab of ['Inputs', 'All inputs']) {
      const shown = bindings.filter(
        (b) => b.resolved !== false && (tab === 'All inputs' || b.prominent),
      );
      const texts = shown.map((b) => names.get(b.id)!.label);
      assert.equal(new Set(texts).size, texts.length, `${row.name} ${tab}: ${texts}`);
    }
    // The Fleet workflow takes a list of prompts in its sampler, not one prompt box.
    if (row.name === 'dgx-monarch-fleet') continue;
    const prompts = bindings.filter((b) => b.prominent && names.get(b.id)?.role === 'prompt');
    assert.ok(prompts.length, `${row.name} shows no Prompt on Inputs`);
  }
});

test('a field issue uses the name its field shows, and its node only when that name repeats', () => {
  const { workflow, bindings } = krea();
  const names = presentFields({ workflow }, bindings, recorded);
  const prompt = bindings.find((b) => b.node_id === '5')!;
  const issue = fieldIssue(prompt, 'A value is required', workflow, names);
  assert.equal(issueText(issue), 'Prompt: A value is required.');
  assert.equal(issue.node, undefined);
  // Two seeds, as a graph that samples in two stages has, keep their nodes apart.
  const seed = (node: string): Binding => ({
    id: `${KREA}:${node}:seed`,
    graph_id: KREA,
    node_id: node,
    input_name: 'seed',
    label: 'seed',
    type: 'INT',
  });
  const twins = [...bindings, seed('8'), seed('9')];
  const twinNames = presentFields({ workflow }, twins, recorded);
  assert.equal(
    issueText(fieldIssue(twins[2], 'Enter a finite number', workflow, twinNames)),
    'seed on node 8: Enter a finite number.',
  );
  // Without the names the issue keeps the input name and the node, as before.
  assert.equal(
    issueText(fieldIssue(prompt, 'A value is required', workflow)),
    'text on node 5: A value is required.',
  );
});

test('the text-through cases hold for Studio, as they do for the layout check', () => {
  // tests/test_studio_template_layouts.py runs the same file through its own walk.
  const file = new URL('../../tests/fixtures/studio/text_through_cases.json', import.meta.url);
  const { cases } = JSON.parse(readFileSync(file, 'utf8'));
  assert.ok(cases.length >= 20);
  for (const row of cases)
    assert.equal(
      carries({ type: row.class, outputTypes: row.outputs }, row.input),
      row.carries,
      JSON.stringify(row),
    );
});
