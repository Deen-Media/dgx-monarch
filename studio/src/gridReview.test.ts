import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { GridReviewGuard, STALE_REVIEW, staleReview } from './gridReview.ts';
import { defaultGridFields, GRID_FIELD_KEYS, gridDefinitionKey } from './gridState.ts';

const draft = { workflow: { id: 'first' }, values: { seed: 1 } };
const fields = { ...defaultGridFields(), axisA: 'steps' };

test('changing any editable grid field invalidates its frozen review', () => {
  // The Grids view derives its definition with gridDefinitionKey and calls staleReview from its
  // edit effect, so this drives the shipped path for every field GridDefinition edits.
  const original = gridDefinitionKey(draft, fields);
  for (const key of GRID_FIELD_KEYS) {
    const guard = new GridReviewGuard();
    guard.accept(original, original);
    assert.equal(staleReview(guard, original, false, true), null);
    const changed = { ...fields, [key]: key === 'baseSeed' ? 7 : 'changed' };
    assert.equal(
      staleReview(guard, gridDefinitionKey(draft, changed), false, true),
      STALE_REVIEW,
      key,
    );
  }
  const guard = new GridReviewGuard();
  guard.accept(original, original);
  const edited = gridDefinitionKey({ ...draft, values: { seed: 2 } }, fields);
  assert.equal(staleReview(guard, edited, false, true), STALE_REVIEW, 'draft');
});

test('the Grids view binds every definition field that the key covers', async () => {
  const [view, editor] = await Promise.all(
    ['./views/Grids.svelte', './components/GridDefinition.svelte'].map((path) =>
      readFile(new URL(path, import.meta.url), 'utf8'),
    ),
  );
  assert.match(view, /\$derived\(gridState\.gridDefinitionKey\(draft, fields\)\)/);
  assert.match(view, /staleReview\(reviewGuard, definition, dispatchPending, !!preview\)/);
  for (const key of GRID_FIELD_KEYS) {
    assert.ok(view.includes(`bind:${key}={fields.${key}}`), key);
    assert.match(editor, new RegExp(`\\b${key} = \\$bindable\\(`), key);
  }
});

test('an edit during validation rejects the returned review before submission is available', async () => {
  const guard = new GridReviewGuard();
  let current = 'first-definition';
  const requested = current;
  let finish!: (value: { id: string }) => void;
  const response = new Promise<{ id: string }>((resolve) => {
    finish = resolve;
  });
  const validation = response.then((review) => {
    guard.accept(requested, current);
    return review;
  });
  current = 'changed-while-loading';
  finish({ id: 'old-definition-review' });
  await assert.rejects(validation, /changed during validation/);
  guard.accept(current, current);
  assert.equal(guard.invalidate(current, false), false);
});

test('edits preserve a pending identity until the same dispatch has reconciled', () => {
  const guard = new GridReviewGuard();
  guard.restore('restored-definition');
  assert.equal(staleReview(guard, 'edited-definition', true, true), null);
  assert.equal(staleReview(guard, 'edited-definition', false, true), STALE_REVIEW);
  const empty = new GridReviewGuard();
  empty.restore('restored-definition');
  assert.equal(staleReview(empty, 'edited-definition', false, false), null, 'no preview to drop');
});
