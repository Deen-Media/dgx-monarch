import test from 'node:test';
import assert from 'node:assert/strict';
import { proxy } from 'svelte/internal/client';
import { componentFunctions } from './renderHarness.js';
import type { StudioClient } from './api.ts';
import { InspectionGuard } from './inspectionGuard.ts';
import { issuesStale, refreshIssues } from './issueRefresh.ts';
import { rebuildBindings } from './workflowInspector.ts';

// A Studio tester's case: the template names a model file the driver lacks.
const ERROR = '3.unet_name: Value is not an available choice';

/** A companion that lists two fields and answers each inspection with the next errors, on release. */
function companion(...answers: string[][]) {
  const sent: Array<{ values: Record<string, unknown> }> = [];
  const gates: Array<() => void> = [];
  const client = {
    request: async (path: string, body: { values: Record<string, unknown> }) => {
      assert.equal(path, '/workflows/inspect');
      sent.push(body);
      await new Promise<void>((done) => gates.push(done));
      return {
        errors: answers.shift() ?? [],
        connections: [],
        bindings: [
          {
            id: 'main:2:format',
            resolved: true,
            value: 'mp4',
            schema: { type: 'COMBO', dynamic: true },
          },
          { id: 'main:3:unet_name', resolved: true, value: 'gone', schema: { type: 'COMBO' } },
        ],
      };
    },
  } as unknown as StudioClient;
  const release = async () => {
    while (!gates.length) await new Promise((done) => setImmediate(done));
    gates.shift()!();
    await new Promise((done) => setImmediate(done));
  };
  return { client, sent, release };
}

// Run the shipped App rebind with the shipped rebuildBindings, as workflowInspector.test.ts does.
const rebindCode = await componentFunctions('./App.svelte', ['rebind']);
const rebindHarness = new Function(
  'rebuildBindings',
  'InspectionGuard',
  'connection',
  'shown',
  `'use strict';
  const guard = new InspectionGuard();
  let draft = { workflow: { nodes: [] }, workflowName: 'Video', values: { 'main:2:format': 'mp4', 'main:3:unet_name': 'gone' } };
  let bindings = [], connections = [], bindingErrors = shown, errors = [];
  const changeDraft = (change) => { guard.edit(change); draft = { ...draft, ...change }; };
  const withBusy = async (action) => { try { await action(); } catch (cause) { errors.push(String(cause)); } };
  ${rebindCode}
  return { guard, rebind, changeDraft, view: () => ({ draft, bindingErrors, errors }) };`,
);

test('a rebuild that kept a field edited while it ran leaves its errors stale, so Create checks again', async () => {
  const { client, sent, release } = companion([ERROR], []);
  const shown = [ERROR];
  const app = rebindHarness(rebuildBindings, InspectionGuard, { client }, shown);
  assert.equal(issuesStale(shown, app.guard, client), false);
  // A dynamic choice changes, and while its inspection runs the tester repairs unet_name.
  const running = app.rebind({ ...app.view().draft.values, 'main:2:format': 'webm' });
  const repair = { ...app.view().draft.values, 'main:3:unet_name': 'kroma.safetensors' };
  app.changeDraft({ values: repair });
  await release();
  await running;
  // The rebuild keeps the repair but answers for the values sent before it.
  assert.equal(app.view().draft.values['main:3:unet_name'], 'kroma.safetensors');
  const rebuilt = app.view().bindingErrors;
  assert.deepEqual([...rebuilt], [ERROR]);
  assert.equal(sent[0].values['main:3:unet_name'], 'gone');
  // Its list carries the rebuild's ticket, so it is stale at once, also through Svelte's state proxy.
  assert.equal(issuesStale(rebuilt, app.guard, client), true);
  assert.equal(issuesStale(proxy(rebuilt), app.guard, client), true);
  // The check Create then runs sends the repair, and the error leaves.
  let applied: string[] = rebuilt;
  const refresh = {
    guard: app.guard,
    apply: (next: { errors: string[] }) => (applied = next.errors),
  };
  const checking = refreshIssues(refresh, { client: () => client, draft: () => app.view().draft });
  await release();
  assert.equal(await checking, true);
  assert.equal(sent[1].values['main:3:unet_name'], 'kroma.safetensors');
  assert.deepEqual(applied, []);
  assert.equal(issuesStale(proxy(applied), app.guard, client), false);
});

test('a rebuild with no edit while it ran shows a current list', async () => {
  const { client, release } = companion([ERROR]);
  const app = rebindHarness(rebuildBindings, InspectionGuard, { client }, [ERROR]);
  const running = app.rebind({ ...app.view().draft.values, 'main:2:format': 'webm' });
  await release();
  await running;
  const rebuilt = app.view().bindingErrors;
  assert.deepEqual([...rebuilt], [ERROR]);
  assert.equal(issuesStale(proxy(rebuilt), app.guard, client), false);
});
