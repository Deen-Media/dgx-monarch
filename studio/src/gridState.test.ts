import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as gridState from './gridState.ts';
import {
  defaultGridFields,
  gridIdentity,
  gridSessionText,
  restoredGridFields,
} from './gridState.ts';
import { GridReviewGuard } from './gridReview.ts';
import { componentFunctions } from './renderHarness.js';

test('a large grid retains a small durable identity for reload and reconciliation', () => {
  const value = {
    id: 'reviewed-grid',
    digest: 'digest',
    ready: true,
    total: 1024,
    job_count: 1024,
    expires_at: 1234,
    state: 'queued',
    cells: Array.from({ length: 1024 }, (_, index) => ({
      index,
      coordinates: { prompt: 'x'.repeat(10000) },
      style_records: [{ data: { prompt: 'x'.repeat(10000) } }],
    })),
    submission: { workflow: 'x'.repeat(100000) },
  };
  const saved = gridIdentity(value)!;
  assert.ok(JSON.stringify(saved).length < 300);
  assert.equal(saved.id, value.id);
  assert.equal(saved.digest, value.digest);
  assert.equal(saved.ready, true);
  assert.equal(saved.expires_at, value.expires_at);
  assert.equal(gridIdentity(null), null);
});

test('a saved session keeps the reviewed definition so a reload shows what Dispatch sends', () => {
  const definition = {
    ...defaultGridFields(),
    axisA: 'steps',
    valuesA: '[10,20]',
    seedPolicy: 'increment',
    baseSeed: 5,
    excluded: '2',
  };
  const preview = { id: 'review', digest: 'd', cells: [{ index: 0, coordinates: {} }] };
  const session = {
    preview,
    grid: null,
    requestKey: 'key',
    retryKey: '',
    retryIndices: [],
    dispatchPending: true,
    retryPending: false,
    definition,
  };
  const stored = JSON.parse(gridSessionText(session));
  assert.deepEqual(stored.preview.cells, []);
  assert.equal(stored.requestKey, 'key');
  assert.equal(stored.dispatchPending, true);
  assert.deepEqual(restoredGridFields(stored.definition), definition);
  assert.equal(JSON.parse(gridSessionText({ ...session, preview: null })).definition, null);
});

test('a malformed stored definition is ignored rather than shown', () => {
  const good = { ...defaultGridFields(), axisA: 'steps' };
  assert.deepEqual(restoredGridFields(good), good);
  for (const bad of [
    null,
    [],
    'text',
    { ...good, baseSeed: '1' },
    { ...good, axisA: 3 },
    { ...good, seedPolicy: 'random' },
    { ...good, valuesA: 'x'.repeat(70000) },
    Object.fromEntries(Object.entries(good).filter(([key]) => key !== 'excluded')),
  ])
    assert.equal(restoredGridFields(bad), null, JSON.stringify(bad)?.slice(0, 40));
  const extra = restoredGridFields({ ...good, injected: 'x' }) as unknown as Record<
    string,
    unknown
  >;
  assert.equal('injected' in extra, false);
});

// Run the shipped Grids prepare and remember handlers, so the stored session is what a reload reads.
const prepareCode = await componentFunctions('./views/Grids.svelte', [
  'prepare',
  'remember',
  'fail',
]);
const gridsHarness = new Function(
  'client',
  'gridState',
  'GridReviewGuard',
  'sessionStorage',
  `'use strict';
  let recoveryBlocked = false, dispatchPending = false, retryPending = false, loadGeneration = 0;
  let busy = false, error = '', confirmed = false, preview = null, grid = null, selected = [];
  let cellPage = 0, requestKey = '', retryKey = '', retryIndices = [], reviewedFields = null;
  let reviewedKey = null;
  let fields = { ...gridState.defaultGridFields(), axisA: 'steps', valuesA: '[10,20]' };
  const draft = { workflow: { id: 'w' } }, storage = 'dgxm.studio.grid.v1';
  const definition = gridState.gridDefinitionKey(draft, fields);
  const reviewGuard = new GridReviewGuard();
  const $state = { snapshot: (value) => structuredClone(value) };
  ${prepareCode}
  return { prepare, remember, edit: (key, value) => { fields[key] = value; }, error: () => error };`,
);

test('Grids stores the reviewed definition, not later edits, with the retained preview', async () => {
  const stored = new Map<string, string>();
  const storage = { setItem: (key: string, value: string) => stored.set(key, value) };
  const client = { request: async () => ({ id: 'review', digest: 'd', cells: [] }) };
  const view = gridsHarness(client, gridState, GridReviewGuard, storage);
  await view.prepare();
  assert.equal(view.error(), '');
  const session = JSON.parse(stored.get('dgxm.studio.grid.v1')!);
  assert.equal(session.preview.id, 'review');
  const reviewed = { ...defaultGridFields(), axisA: 'steps', valuesA: '[10,20]' };
  assert.deepEqual(restoredGridFields(session.definition), reviewed);
  view.edit('valuesA', '[30]');
  view.remember();
  const later = JSON.parse(stored.get('dgxm.studio.grid.v1')!);
  assert.deepEqual(later.definition, reviewed, 'a later edit never rewrites the reviewed fields');
});

test('Grids restores the reviewed fields before the review guard reads the definition', async () => {
  const view = await readFile(new URL('./views/Grids.svelte', import.meta.url), 'utf8');
  const restore = view.indexOf('gridState.restoredGridFields(');
  const guard = view.indexOf('reviewGuard.restore(current)');
  assert.ok(restore > 0 && guard > restore, 'fields are assigned before reviewGuard.restore');
  assert.match(view, /restoreSaved\(restored\.value\)/, 'onMount restores through restoreSaved');
});

test('a review identity names the draft and the fields, not their key order', () => {
  const fields = { ...defaultGridFields(), axisA: 'steps' };
  const draft = { workflow: { nodes: [{ id: 1 }] }, values: { a: 1, b: 2 } };
  const key = gridState.reviewIdentity(draft, fields);
  const reordered = { values: { b: 2, a: 1 }, workflow: { nodes: [{ id: 1 }] } };
  assert.equal(gridState.reviewIdentity(reordered, { ...fields }), key);
  assert.notEqual(gridState.reviewIdentity({ ...draft, values: { a: 1, b: 3 } }, fields), key);
  assert.notEqual(gridState.reviewIdentity(draft, { ...fields, valuesA: '[3]' }), key);
  assert.ok(key.length < 40, 'the stored identity is small');
  assert.equal(gridState.restoredReview({ reviewed: key, dispatchPending: false }, key), 'fresh');
  assert.equal(
    gridState.restoredReview({ reviewed: 'other', dispatchPending: false }, key),
    'stale',
  );
  assert.equal(gridState.restoredReview({ dispatchPending: false }, key), 'stale');
  assert.equal(
    gridState.restoredReview({ reviewed: 'other', dispatchPending: true }, key),
    'pending',
  );
});

// Run the shipped prepare and restore handlers against two drafts, as when the
// workspace draft changes while the Grids view is closed.
const restoreCode = await componentFunctions('./views/Grids.svelte', [
  'prepare',
  'remember',
  'persist',
  'fail',
  'restoreSaved',
]);
const restoreHarness = new Function(
  'client',
  'gridState',
  'GridReviewGuard',
  'sessionStorage',
  'draft',
  `'use strict';
  let recoveryBlocked = false, dispatchPending = false, retryPending = false, loadGeneration = 0;
  let busy = false, error = '', confirmed = false, preview = null, grid = null, selected = [];
  let cellPage = 0, requestKey = '', retryKey = '', retryIndices = [];
  let reviewedFields = null, reviewedKey = null;
  let fields = { ...gridState.defaultGridFields(), axisA: 'steps', valuesA: '[10,20]' };
  const storage = 'dgxm.studio.grid.v1';
  const definition = gridState.gridDefinitionKey(draft, fields);
  const reviewGuard = new GridReviewGuard();
  const $state = { snapshot: (value) => structuredClone(value) };
  ${restoreCode}
  return {
    prepare,
    restoreSaved,
    guard: reviewGuard,
    view: () => ({ preview, requestKey, error, fields, dispatchPending }),
    definition: () => gridState.gridDefinitionKey(draft, fields),
  };`,
);

test('a restored grid review is dropped when the draft changed, unless its dispatch is pending', async () => {
  const stored = new Map<string, string>();
  const storage = { setItem: (key: string, value: string) => stored.set(key, value) };
  const client = { request: async () => ({ id: 'review', digest: 'd', ready: true, cells: [] }) };
  const reviewedDraft = { workflow: { id: 'w' }, values: { prompt: 'a cat' } };
  const editedDraft = { workflow: { id: 'w' }, values: { prompt: 'a dog' } };
  const first = restoreHarness(client, gridState, GridReviewGuard, storage, reviewedDraft);
  await first.prepare();
  const saved = JSON.parse(stored.get('dgxm.studio.grid.v1')!);
  assert.equal(saved.reviewed, gridState.reviewIdentity(reviewedDraft, saved.definition));
  // Same draft: the review returns and a later edit still invalidates it.
  const same = restoreHarness(client, gridState, GridReviewGuard, storage, reviewedDraft);
  same.restoreSaved(saved);
  assert.equal(same.view().preview?.id, 'review');
  assert.equal(same.view().requestKey, saved.requestKey);
  assert.equal(same.view().error, '');
  assert.equal(same.guard.invalidate(same.definition(), false), false);
  // Changed draft: the old preview would run the earlier prompt, so it is dropped.
  const changed = restoreHarness(client, gridState, GridReviewGuard, storage, editedDraft);
  changed.restoreSaved(saved);
  assert.equal(changed.view().preview, null);
  assert.equal(changed.view().requestKey, '');
  assert.equal(changed.view().error, gridState.STALE_RESTORE);
  assert.equal(JSON.parse(stored.get('dgxm.studio.grid.v1')!).preview, null);
  // A session saved before reviews carried an identity is dropped the same way.
  const legacy = restoreHarness(client, gridState, GridReviewGuard, storage, reviewedDraft);
  legacy.restoreSaved({ ...saved, reviewed: undefined });
  assert.equal(legacy.view().preview, null);
  // An uncertain dispatch keeps its preview and request key for reconciliation.
  const pending = restoreHarness(client, gridState, GridReviewGuard, storage, editedDraft);
  pending.restoreSaved({ ...saved, dispatchPending: true });
  assert.equal(pending.view().preview?.id, 'review');
  assert.equal(pending.view().requestKey, saved.requestKey);
  assert.equal(pending.view().dispatchPending, true);
  assert.equal(pending.view().error, gridState.EARLIER_PENDING);
});
