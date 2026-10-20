import test from 'node:test';
import assert from 'node:assert/strict';
import { frameTarget } from './mediaPlayback.ts';

test('frame stepping waits for valid duration and frame-rate metadata', () => {
  for (const duration of [NaN, Infinity, 0, -1])
    assert.equal(frameTarget(0, duration, 24, 1), null);
  for (const fps of [NaN, Infinity, 0, -24, undefined, '24'])
    assert.equal(frameTarget(0, 3, fps, 1), null);
  assert.equal(frameTarget(NaN, 3, 24, 1), null);
});

test('frame stepping uses the reported rate and clamps at both ends', () => {
  assert.equal(frameTarget(1, 3, 25, 1), 1.04);
  assert.equal(frameTarget(1, 3, 25, -1), 0.96);
  assert.equal(frameTarget(0, 3, 25, -1), 0);
  assert.equal(frameTarget(2.99, 3, 25, 1), 3);
});
