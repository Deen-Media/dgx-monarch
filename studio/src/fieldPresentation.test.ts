import test from 'node:test';
import assert from 'node:assert/strict';
import { bundled, graph, recorded, widgetBindings } from './workflowFixtures.js';
import { graphRoles, presentFields, promptRoles } from './fieldPresentation.ts';
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
