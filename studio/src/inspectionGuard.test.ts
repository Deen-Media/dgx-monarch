import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DROPPED_REVIEW,
  InspectionGuard,
  droppedInspection,
  supersedesInspection,
} from './inspectionGuard.ts';

const client = { name: 'companion' };

test('edits outside the workflow and its values do not drop a template inspection', () => {
  const guard = new InspectionGuard();
  const ticket = guard.begin(client);
  for (const change of [
    { operation: 'video' },
    { seed_policy: 'random' },
    { project_id: 'project' },
    { lora_stack: [] },
    { styles: [] },
  ])
    guard.edit(change);
  assert.equal(guard.inspected(ticket, client), 'current');
});

test('a workflow or value edit during an inspection drops it with a notice', () => {
  for (const change of [{ workflow: { nodes: [] } }, { values: { seed: 3 } }]) {
    const guard = new InspectionGuard();
    const ticket = guard.begin(client);
    guard.edit(change);
    const state = guard.inspected(ticket, client);
    assert.equal(state, 'edited');
    assert.match(droppedInspection('Flux', state), /inspection of Flux was discarded/);
  }
  assert.equal(supersedesInspection({ workflow: undefined, operation: 'image' }), false);
});

test('a newer inspection replaces an older one without a notice', () => {
  const guard = new InspectionGuard();
  const older = guard.begin(client);
  const newer = guard.begin(client);
  assert.equal(guard.inspected(newer, client), 'current');
  // Applying the newer result edits the workflow and values.
  guard.edit({ workflow: {}, values: {} });
  assert.equal(guard.inspected(older, client), 'newer');
  assert.equal(droppedInspection('Older', 'newer'), '');
});

test('a reconnect drops an inspection from the old companion', () => {
  const guard = new InspectionGuard<object>();
  const ticket = guard.begin(client);
  const state = guard.inspected(ticket, { name: 'new companion' });
  assert.equal(state, 'reconnected');
  assert.match(droppedInspection('', state), /connection changed/);
});

test('any draft edit or a reconnect makes a preview stale, and starting one does not', () => {
  const guard = new InspectionGuard<object>();
  const inspection = guard.begin(client);
  const preview = guard.snapshot(client);
  assert.equal(guard.inspected(inspection, client), 'current');
  assert.equal(guard.previewed(preview, client), true);
  guard.edit({ seed_policy: 'random' });
  assert.equal(guard.previewed(preview, client), false);
  const next = guard.snapshot(client);
  assert.equal(guard.previewed(next, { name: 'new companion' }), false);
  assert.match(DROPPED_REVIEW, /Review the generation again/);
});
