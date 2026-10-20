import test from 'node:test';
import assert from 'node:assert/strict';
import { GridReviewGuard } from './gridReview.ts';

test('changing any editable grid field invalidates its frozen review', () => {
  const definition = {
    draft: { workflow: { id: 'first' }, values: { seed: 1 } },
    axisA: 'steps',
    valuesA: '[1,2]',
    axisB: '',
    valuesB: '[1]',
    seedPolicy: 'fixed',
    seedBinding: '',
    baseSeed: 0,
    excluded: '',
  };
  for (const key of Object.keys(definition)) {
    const guard = new GridReviewGuard();
    const original = JSON.stringify(definition);
    guard.accept(original, original);
    assert.equal(guard.invalidate(original, false), false);
    assert.equal(
      guard.invalidate(JSON.stringify({ ...definition, [key]: 'changed' }), false),
      true,
      key,
    );
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
  const retained = { id: 'review-1', requestKey: 'original-key' };
  let preview: typeof retained | null = retained;
  if (guard.invalidate('edited-definition', true)) preview = null;
  assert.equal(preview, retained);
  if (guard.invalidate('edited-definition', false)) preview = null;
  assert.equal(preview, null);
});
