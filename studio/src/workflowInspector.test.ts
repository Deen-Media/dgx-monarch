import test from 'node:test';
import assert from 'node:assert/strict';
import type { StudioClient } from './api.ts';
import { inspectDraft } from './workflowInspector.ts';

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
