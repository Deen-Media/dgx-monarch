import test from 'node:test';
import assert from 'node:assert/strict';
import type { StudioClient } from './api.ts';
import { readFileSync } from 'node:fs';
import {
  bindingValues,
  freshGraphNotice,
  inspectDraft,
  LORA_SLOT_CLASSES,
  rebuildBindings,
} from './workflowInspector.ts';
import { componentFunctions } from './renderHarness.js';
import { graph, recorded } from './workflowFixtures.js';
import { droppedInspection, InspectionGuard } from './inspectionGuard.ts';
import { emptyDraft } from './model.ts';
import { loadWorkflow } from './workflowChoice.ts';

test('inspection keeps graph-owned sockets out of draft values and preserves editable overrides', async () => {
  const workflow = { extra: { customMetadata: { retained: true } }, nodes: [] };
  const client = {
    request: async () => ({
      workflow,
      errors: ['original seed invalid'],
      bindings: [
        { id: 'root:1:seed', resolved: true, value: -1, schema: { type: 'INT', default: 7 } },
        { id: 'root:2:strength', resolved: true, value: 0, schema: { type: 'FLOAT', default: 1 } },
        { id: 'root:3:text', resolved: false, value: ['2', 0], schema: { type: 'STRING' } },
      ],
    }),
  } as unknown as StudioClient;
  const result = await inspectDraft(client, workflow, 'Fixture', {
    'root:1:seed': 42,
    'deleted:id': 5,
  });
  assert.deepEqual(result.patch.values, { 'root:1:seed': 42, 'root:2:strength': 0 });
  assert.deepEqual(result.patch.workflow, workflow);
  assert.deepEqual(result.errors, ['original seed invalid']);
  assert.equal(result.controlTab, 'advanced');
});

test('instance bindings retain only their explicit unique legacy alias', async () => {
  const client = {
    request: async () => ({
      bindings: [
        {
          id: '@instance/root/host:1:seed',
          aliases: ['definition:1:seed'],
          value: 1,
          schema: { type: 'INT' },
        },
        { id: '@instance/root/other:1:seed', value: 2, schema: { type: 'INT' } },
      ],
      connections: [
        { node_id: 'mapped', input_name: 'model', type: 'DGXM_MODEL', source: ['loader', 0] },
      ],
    }),
  } as unknown as StudioClient;
  const result = await inspectDraft(client, {}, 'Shared graph', { 'definition:1:seed': 99 });
  assert.deepEqual(result.patch.values, {
    '@instance/root/host:1:seed': 99,
    '@instance/root/other:1:seed': 2,
  });
  assert.equal(result.connections[0].type, 'DGXM_MODEL');
});

test('inspection sends the draft values so edited dynamic choices list their child inputs', async () => {
  const sent: unknown[] = [];
  const workflow = { nodes: [] };
  const client = {
    request: async (_path: string, body: unknown) => {
      sent.push(body);
      return { bindings: [] };
    },
  } as unknown as StudioClient;
  await inspectDraft(client, workflow, 'Video', { 'main:2:format.codec': 'h264' });
  await inspectDraft(client, workflow, 'Video');
  assert.deepEqual(sent, [{ workflow, values: { 'main:2:format.codec': 'h264' } }, { workflow }]);
});

test('rebuilt bindings keep edits, give revealed inputs their value and drop vanished ones', () => {
  const values = bindingValues(
    [
      { id: 'main:2:format.codec', resolved: true, value: 'auto', schema: { type: 'COMBO' } },
      {
        id: 'main:2:format.codec.encoding',
        resolved: true,
        value: 'auto',
        schema: { type: 'COMBO' },
      },
      { id: 'main:2:crf', resolved: true, schema: { type: 'FLOAT', default: 18 } },
      { id: 'main:3:video', resolved: false, value: ['1', 0] },
    ],
    { 'main:2:format.codec': 'h264', 'main:2:old.child': 'kept before', 'main:2:crf': 30 },
  );
  assert.deepEqual(values, {
    'main:2:format.codec': 'h264',
    'main:2:format.codec.encoding': 'auto',
    'main:2:crf': 30,
  });
});

// Run the shipped Generation field handler: only a dynamic choice changes which inputs exist.
const fieldChangeCode = await componentFunctions('./views/Generation.svelte', ['fieldChange']);
const fieldHandler = new Function(
  'draft',
  'changeDraft',
  'rebind',
  `'use strict'; ${fieldChangeCode}; return fieldChange;`,
);

test('a dynamic choice rebuilds the bindings; any other field only edits the draft', () => {
  const calls: unknown[] = [];
  const fieldChange = fieldHandler(
    { values: { a: 1 } },
    (change: unknown) => calls.push(['change', change]),
    (values: unknown) => calls.push(['rebind', values]),
  );
  fieldChange({ id: 'main:2:format', schema: { type: 'COMBO', dynamic: true } }, 'mp4');
  fieldChange({ id: 'main:5:sampler', schema: { type: 'COMBO' } }, 'euler');
  fieldChange({ id: 'main:3:steps', schema: { type: 'INT' } }, 20);
  assert.deepEqual(calls, [
    ['rebind', { a: 1, 'main:2:format': 'mp4' }],
    ['change', { values: { a: 1, 'main:5:sampler': 'euler' } }],
    ['change', { values: { a: 1, 'main:3:steps': 20 } }],
  ]);
});

// Run the shipped App rebind with the shipped rebuildBindings: a later ordinary edit survives.
const rebindCode = await componentFunctions('./App.svelte', ['rebind']);
const rebindHarness = new Function(
  'rebuildBindings',
  'InspectionGuard',
  'connection',
  `'use strict';
  const guard = new InspectionGuard();
  let draft = { workflow: { nodes: [] }, workflowName: 'Video', values: {} };
  let bindings = [], connections = [], bindingErrors = [], errors = [];
  const changeDraft = (change) => { guard.edit(change); draft = { ...draft, ...change }; };
  const withBusy = async (action) => { try { await action(); } catch (cause) { errors.push(String(cause)); } };
  ${rebindCode}
  return { rebind, changeDraft, view: () => ({ draft, bindings, errors }) };`,
);

function inspection(answered: Promise<void>, sent: unknown[]) {
  return {
    request: async (_path: string, body: { values?: unknown }) => {
      sent.push(body.values);
      await answered;
      return {
        bindings: [
          { id: 'main:2:format.codec', resolved: true, value: 'auto' },
          { id: 'main:2:format.codec.encoding', resolved: true, value: 'auto' },
          { id: 'main:3:steps', resolved: true, value: 10 },
        ],
      };
    },
  };
}

test('App rebuilds the bindings for an edited dynamic choice and keeps later edits', async () => {
  const { InspectionGuard } = await import('./inspectionGuard.ts');
  let release!: () => void;
  const sent: unknown[] = [];
  const connection = { client: inspection(new Promise<void>((done) => (release = done)), sent) };
  const app = rebindHarness(rebuildBindings, InspectionGuard, connection);
  const running = app.rebind({ 'main:2:format.codec': 'h264', 'main:3:steps': 10 });
  // An ordinary field edit lands while the inspection runs.
  app.changeDraft({ values: { ...app.view().draft.values, 'main:3:steps': 25 } });
  release();
  await running;
  assert.deepEqual(sent, [{ 'main:2:format.codec': 'h264', 'main:3:steps': 10 }]);
  assert.deepEqual(
    app.view().bindings.map((binding: { id: string }) => binding.id),
    ['main:2:format.codec', 'main:2:format.codec.encoding', 'main:3:steps'],
  );
  assert.deepEqual(app.view().draft.values, {
    'main:2:format.codec': 'h264',
    'main:2:format.codec.encoding': 'auto',
    'main:3:steps': 25,
  });
  assert.equal(app.view().draft.workflowName, 'Video');
  assert.deepEqual(app.view().errors, []);
});

test('a rebuild yields to a newer inspection or another workflow and refuses a new connection', async () => {
  const { InspectionGuard } = await import('./inspectionGuard.ts');
  const values = { 'main:2:format.codec': 'h264' };
  const draft = { workflow: { nodes: [] }, workflowName: 'Video', values } as never;
  const connection = { client: inspection(Promise.resolve(), []) } as never;
  const newer = new InspectionGuard();
  const superseded = rebuildBindings(connection, newer, () => draft, values);
  newer.begin({});
  assert.equal(await superseded, null);
  let current = draft as { workflow: unknown };
  const replaced = rebuildBindings(
    connection,
    new InspectionGuard(),
    () => current as never,
    values,
  );
  current = { ...current, workflow: { nodes: [] } };
  assert.equal(await replaced, null);
  const moved = { client: inspection(Promise.resolve(), []) };
  const reconnected = rebuildBindings(moved as never, new InspectionGuard(), () => draft, values);
  moved.client = inspection(Promise.resolve(), []);
  await assert.rejects(reconnected, /companion connection changed/);
  assert.ok(await rebuildBindings(connection, new InspectionGuard(), () => draft, values));
});

// Run the shipped App inspect and changeDraft, with the shipped loadWorkflow the picker calls.
const pickCode = await componentFunctions('./App.svelte', ['changeDraft', 'inspect', 'setNotice']);
const pickHarness = new Function(
  'inspectDraft',
  'InspectionGuard',
  'droppedInspection',
  'loadWorkflow',
  'freshGraphNotice',
  'connection',
  'initial',
  `'use strict';
  const guard = new InspectionGuard();
  let draft = initial, undoDraft = null, workflowId = '', dirty = false, notice = '', error = '';
  let bindings = [], connections = [], bindingErrors = [], rawWorkflow = '', controlTab = '';
  const errors = [], submission = { preview: null };
  const $state = { snapshot: (value) => value };
  const fail = (cause) => errors.push(String(cause));
  const withBusy = async (action) => { try { await action(); } catch (cause) { fail(cause); } };
  ${pickCode}
  return { inspect, pick: (entry) => loadWorkflow(entry, inspect), view: () => ({ draft, errors, notice }) };`,
);
const BOTH_DROPPED =
  'This workflow starts from its own values. Choose the prompt input for styles and the LoRA insertion boundary again.';
const STACK_TAKES_OVER =
  'This workflow has LoRAs of its own. At Generate, the LoRA stack replaces them by position, and those it does not reach run at strength 0. Remove the stack on the LoRAs tab to keep them.';

test('a workflow picked by hand names none of the old graph; the same graph inspected again keeps its controls', async () => {
  const controls = {
    style_bindings: { prompt: 'old-graph:6:text', negative_prompt: 'old-graph:7:text' },
    lora_insertion: {
      loader: 'LoraLoader',
      model_targets: [{ node_id: '9', input_name: 'model' }],
      clip_targets: [{ node_id: '6', input_name: 'clip' }],
    },
  };
  const user = {
    styles: [{ id: 'style-ink', name: 'Ink', prompt: 'ink wash, {prompt}', enabled: true }],
    lora_stack: [{ id: 'detail', name: 'detail', enabled: true, model_strength: 0.8 }],
    project_id: 'project',
    seed_policy: 'fixed',
  };
  const old = { id: 'old-graph', nodes: [] };
  const connection = {
    client: {
      request: async (_path: string, body: { workflow: { id: string } }) => ({
        workflow: body.workflow,
        bindings: [
          { id: `${body.workflow.id}:6:text`, resolved: true, value: 'a fox' },
          { id: `${body.workflow.id}:7:text`, resolved: true, value: '' },
        ],
      }),
    },
    catalog: {},
  };
  const initial = { ...emptyDraft(), workflow: old, workflowName: 'Old', ...controls, ...user };
  const app = pickHarness(
    inspectDraft,
    InspectionGuard,
    droppedInspection,
    loadWorkflow,
    freshGraphNotice,
    connection,
    initial,
  );
  // A connect or a draft import inspects the current graph again with its values.
  assert.equal(await app.inspect(old, 'Old', true), '');
  let { draft } = app.view();
  assert.deepEqual(
    { style_bindings: draft.style_bindings, lora_insertion: draft.lora_insertion },
    controls,
  );
  assert.equal(app.view().notice, '', 'nothing is dropped, so nothing is said');
  const said = await app.pick({
    id: 'new',
    name: 'New',
    title: 'New',
    workflow: { id: 'new-graph', nodes: [] },
  });
  ({ draft } = app.view());
  assert.deepEqual(app.view().errors, []);
  // The notice line names what the new workflow dropped, and the picker hands the same text on.
  assert.equal(app.view().notice, BOTH_DROPPED);
  assert.equal(said, BOTH_DROPPED);
  assert.equal(draft.workflowName, 'New');
  assert.deepEqual(draft.values, { 'new-graph:6:text': 'a fox', 'new-graph:7:text': '' });
  assert.equal(draft.style_bindings, undefined);
  assert.equal(draft.lora_insertion, undefined);
  const { styles, lora_stack, project_id, seed_policy } = draft;
  assert.deepEqual({ styles, lora_stack, project_id, seed_policy }, user);
  assert.equal(JSON.stringify(draft).includes('old-graph'), false);
});

test('the load notice names only the control the draft held, and a stack over LoRA slots of its own', () => {
  const stack = [{ id: 'detail', name: 'detail', enabled: false, model_strength: 1 }];
  const insertion = { loader: 'LoraLoader', model_targets: [], clip_targets: [] };
  const slotless = graph('flux3-klein-base-edit');
  const slotted = graph('kromoss-t2i-turbo');
  const held = (change: object) => ({ ...emptyDraft(), ...change });
  assert.equal(freshGraphNotice(held({}), slotless, recorded), '');
  assert.equal(
    freshGraphNotice(held({ style_bindings: { prompt: 'g:6:text' } }), slotless, recorded),
    'This workflow starts from its own values. Choose the prompt input for styles again.',
  );
  // A choice cleared under Ordered prompt styles leaves an empty mapping, which holds nothing.
  assert.equal(freshGraphNotice(held({ style_bindings: {} }), slotless, recorded), '');
  assert.equal(
    freshGraphNotice(held({ lora_insertion: insertion }), slotless, recorded),
    'This workflow starts from its own values. Choose the LoRA insertion boundary again.',
  );
  assert.equal(
    freshGraphNotice(
      held({ style_bindings: { negative_prompt: 'g:7:text' }, lora_insertion: insertion }),
      slotless,
      recorded,
    ),
    BOTH_DROPPED,
  );
  // Any stack takes over the slots, even with every entry off; an empty one leaves them alone.
  assert.equal(freshGraphNotice(held({ lora_stack: stack }), slotless, recorded), '');
  assert.equal(freshGraphNotice(held({ lora_stack: stack }), slotted, recorded), STACK_TAKES_OVER);
  assert.equal(freshGraphNotice(held({}), slotted, recorded), '');
  // A workflow with LoRA loaders refuses or ignores a boundary, so the notice asks for none there.
  const cleared =
    'This workflow starts from its own values. This workflow has LoRAs of its own, so the LoRA insertion boundary is cleared.';
  assert.equal(freshGraphNotice(held({ lora_insertion: insertion }), slotted, recorded), cleared);
  assert.equal(
    freshGraphNotice(held({ lora_insertion: insertion, lora_stack: stack }), slotted, recorded),
    `${cleared} ${STACK_TAKES_OVER.replace('This workflow has LoRAs of its own. ', '')}`,
  );
  assert.equal(
    freshGraphNotice(
      held({ style_bindings: { prompt: 'g:6:text' }, lora_insertion: insertion }),
      slotted,
      recorded,
    ),
    'This workflow starts from its own values. Choose the prompt input for styles again. This workflow has LoRAs of its own, so the LoRA insertion boundary is cleared.',
  );
  // A bypassed loader is no slot: the companion compiles it out.
  const bypassed = structuredClone(slotted);
  for (const node of bypassed.nodes) if (LORA_SLOT_CLASSES.has(node.type)) node.mode = 4;
  assert.equal(freshGraphNotice(held({ lora_stack: stack }), bypassed, recorded), '');
});

test("Studio's LoRA slot classes are the companion's", () => {
  const source = readFileSync(
    new URL('../../src/dgx_monarch/studio/lora_graph.py', import.meta.url),
    'utf8',
  );
  const set = source.match(/^LORA_NODES = \{([^}]*)\}/m)![1];
  assert.deepEqual(
    [...set.matchAll(/"([^"]+)"/g)].map((match) => match[1]).sort(),
    [...LORA_SLOT_CLASSES].sort(),
  );
});

test('an edited workflow or a picked one says what it dropped and when the stack takes over its LoRAs', async () => {
  const connection = {
    client: {
      request: async (_path: string, body: { workflow: { id: string } }) => ({
        workflow: body.workflow,
        bindings: [{ id: `${body.workflow.id}:6:text`, resolved: true, value: 'a fox' }],
      }),
    },
    catalog: { object_info: recorded },
  };
  const stack = [{ id: 'detail', name: 'detail', enabled: true, model_strength: 0.8 }];
  const old = { id: 'old-graph', nodes: [] };
  const initial = {
    ...emptyDraft(),
    workflow: old,
    workflowName: 'Old',
    style_bindings: { prompt: 'old-graph:6:text' },
    lora_stack: stack,
  };
  const app = pickHarness(
    inspectDraft,
    InspectionGuard,
    droppedInspection,
    loadWorkflow,
    freshGraphNotice,
    connection,
    initial,
  );
  // Validate edited workflow with changed JSON inspects without the draft's values.
  const edited = { id: 'old-graph', nodes: [], extra: { note: 'edited' } };
  await app.inspect(edited, 'Old', false);
  assert.equal(
    app.view().notice,
    'This workflow starts from its own values. Choose the prompt input for styles again.',
  );
  assert.equal(app.view().draft.style_bindings, undefined);
  // A workflow with a LoRA loader of its own, loaded with the stack still in the draft.
  const kromoss = graph('kromoss-t2i-turbo');
  await app.pick({ id: 'k', name: 'Kromoss', title: 'Kromoss', workflow: kromoss });
  assert.equal(app.view().notice, STACK_TAKES_OVER);
  assert.deepEqual(app.view().draft.lora_stack, stack, 'the stack itself stays');
});
