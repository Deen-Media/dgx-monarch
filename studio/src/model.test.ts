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
