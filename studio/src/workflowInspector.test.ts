import test from 'node:test';
import assert from 'node:assert/strict';
import type { StudioClient } from './api.ts';
import { bindingValues, inspectDraft, rebuildBindings } from './workflowInspector.ts';
import { componentFunctions } from './renderHarness.js';

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
