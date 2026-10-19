import test from 'node:test';
import assert from 'node:assert/strict';
import { encodeMaskAlpha } from './mask.ts';
test('mask upload selects painted pixels in Comfy inverse-alpha convention', () => {
  const editor = new Uint8ClampedArray([135, 249, 0, 255, 0, 0, 0, 0, 135, 249, 0, 128]);
  assert.deepEqual(
    [...encodeMaskAlpha(editor)],
    [255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 127],
  );
  assert.equal(editor[3], 255);
});
