import test from 'node:test';
import assert from 'node:assert/strict';
import { encodeMaskAlpha, maskGeometry, maskIsEmpty, maskPoint } from './mask.ts';
test('mask upload selects painted pixels in Comfy inverse-alpha convention', () => {
  const editor = new Uint8ClampedArray([135, 249, 0, 255, 0, 0, 0, 0, 135, 249, 0, 128]);
  assert.deepEqual(
    [...encodeMaskAlpha(editor)],
    [255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 127],
  );
  assert.equal(editor[3], 255);
});

for (const [width, height] of [
  [512, 512],
  [2048, 512],
  [512, 2048],
  [4096, 64],
  [64, 4096],
]) {
  test(`source ${width}x${height} fits centered stage without changing pixels or aspect`, () => {
    const geometry = maskGeometry(width, height);
    assert.equal(geometry.valid, true);
    assert.equal(geometry.width, width);
    assert.equal(geometry.height, height);
    for (const available of [180, 340, 1000]) {
      const stageWidth = Math.min(available, geometry.maxStageWidth);
      const stageHeight = (stageWidth * height) / width;
      assert.ok(stageHeight <= 460);
      assert.ok(stageWidth <= available);
      assert.ok(stageWidth <= width && stageHeight <= height);
      const rect = {
        left: (available - stageWidth) / 2,
        top: 50,
        width: stageWidth,
        height: stageHeight,
      };
      assert.deepEqual(
        maskPoint(rect.left + stageWidth / 2, 50 + stageHeight / 2, rect, width, height),
        { x: width / 2, y: height / 2 },
      );
      const quarter = maskPoint(
        rect.left + stageWidth / 4,
        50 + (stageHeight * 3) / 4,
        rect,
        width,
        height,
      )!;
      assert.ok(Math.abs(quarter.x - width / 4) < 1e-8);
      assert.ok(Math.abs(quarter.y - (height * 3) / 4) < 1e-8);
    }
  });
}

for (const invalid of [undefined, NaN, Infinity, 0, -4, 1.5, 4097, '512']) {
  test(`invalid mask dimension ${String(invalid)} has safe geometry and blocks upload`, () => {
    const geometry = maskGeometry(invalid, 512);
    assert.equal(geometry.valid, false);
    assert.ok(Number.isInteger(geometry.width) && geometry.width >= 1 && geometry.width <= 4096);
    assert.ok(Number.isFinite(geometry.maxStageWidth) && geometry.maxStageWidth > 0);
  });
}
test('captured pointer outside the image clamps to its edge and hidden stages cannot paint', () => {
  assert.deepEqual(
    maskPoint(-100, 1000, { left: 20, top: 20, width: 100, height: 200 }, 512, 1024),
    { x: 0, y: 1024 },
  );
  assert.equal(maskPoint(10, 10, { left: 0, top: 0, width: 0, height: 0 }, 512, 512), null);
});
test('a canvas with no painted pixel is an empty mask', () => {
  assert.equal(maskIsEmpty(new Uint8ClampedArray(16)), true);
  // Erased paint keeps its colour but has zero alpha.
  assert.equal(maskIsEmpty(new Uint8ClampedArray([135, 249, 0, 0, 0, 0, 0, 0])), true);
  assert.equal(maskIsEmpty(new Uint8ClampedArray([0, 0, 0, 0, 135, 249, 0, 1])), false);
  assert.throws(() => maskIsEmpty(new Uint8ClampedArray(3)), /RGBA/);
});
