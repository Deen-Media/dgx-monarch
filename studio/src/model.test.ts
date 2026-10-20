import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bindingError,
  gridCombinations,
  consumeEvents,
  boundedLog,
  restoreDraft,
} from './model.ts';
test('typed limits reject unsafe values without clamping', () => {
  const b = { id: 'steps', type: 'INT', min: 1, max: 100 };
  assert.equal(bindingError(b, 10), null);
  assert.match(bindingError(b, 0)!, /Minimum/);
  assert.match(bindingError(b, 1.2)!, /whole/);
  assert.match(bindingError(b, Infinity)!, /finite/);
  assert.match(bindingError({ ...b, status: 'ambiguous' }, 20)!, /repair/);
});
test('grid preserves values and bounds Cartesian multiplication', () => {
  assert.deepEqual(
    gridCombinations([
      { id: 'seed', values: [1, 2] },
      { id: 'cfg', values: [3, 4] },
    ]),
    [
      { seed: 1, cfg: 3 },
      { seed: 1, cfg: 4 },
      { seed: 2, cfg: 3 },
      { seed: 2, cfg: 4 },
    ],
  );
  assert.throws(
    () =>
      gridCombinations(
        [
          { id: 'a', values: [1, 2] },
          { id: 'b', values: [1, 2] },
        ],
        3,
      ),
    /limit/,
  );
  assert.throws(
    () =>
      gridCombinations([
        { id: 'a', values: [1] },
        { id: 'a', values: [2] },
      ]),
    /different/,
  );
});
test('delayed and duplicated events cannot regress cursor', () => {
  const result = consumeEvents(4, [{ seq: 7 }, { seq: 2 }, { seq: 5 }, { seq: 7 }, { seq: 'bad' }]);
  assert.equal(result.cursor, 7);
  assert.deepEqual(result.events, [{ seq: 5 }, { seq: 7 }]);
});
test('console retention accounts for Unicode bytes and line count', () => {
  assert.deepEqual(boundedLog(['a', 'b', 'c'], 2), ['b', 'c']);
  assert.deepEqual(boundedLog(['old', '🦋'], 100, 4), ['🦋']);
});
test('restoration preserves unknown metadata without treating malformed stacks as lists', () => {
  const result = restoreDraft({
    values: { seed: 5 },
    extension: { kept: true },
    lora_stack: 'bad',
  });
  assert.equal(result.values.seed, 5);
  assert.deepEqual(result.lora_stack, []);
  assert.deepEqual((result as any).extension, { kept: true });
});
test('malformed nested draft controls are rejected before a render can replace the working draft', () => {
  for (const value of [
    { styles: [null] },
    { styles: ['saved-id'] },
    { lora_stack: [null] },
    { lora_stack: [7] },
    { parent_artifact_ids: [{}] },
    { lora_insertion: { loader: 'LoraLoader', model_targets: 'not-a-list' } },
    { lora_insertion: { loader: 'LoraLoader', clip_targets: [null] } },
    { lora_insertion: { loader: 'LoraLoader', model_targets: [{ node_id: '1' }] } },
    { style_bindings: [] },
    { style_bindings: { prompt: {} } },
    { workflow: [] },
    { workflowName: {} },
    { project_id: 3 },
  ])
    assert.throws(() => restoreDraft(value), /Invalid draft/);
});
test('optional legacy insertion targets default to empty without dropping extensions', () => {
  const value = restoreDraft({
    lora_insertion: {
      loader: 'LoraLoaderModelOnly',
      model_targets: [{ node_id: '1', input_name: 'model' }],
    },
    extension: { retained: true },
  });
  assert.deepEqual(value.lora_insertion?.clip_targets, []);
  assert.deepEqual((value as any).extension, { retained: true });
  assert.deepEqual(restoreDraft({ values: [] }).values, {});
  assert.equal(restoreDraft({ lora_insertion: null }).lora_insertion, undefined);
});
