/** Comfy LoadImage derives its mask as 1-alpha; green editor paint marks selected pixels. */
export function encodeMaskAlpha(pixels: Uint8ClampedArray): Uint8ClampedArray {
  if (pixels.length % 4) throw new Error('Mask pixels must be RGBA.');
  const output = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < pixels.length; i += 4) {
    output[i] = 255;
    output[i + 1] = 255;
    output[i + 2] = 255;
    output[i + 3] = 255 - pixels[i + 3];
  }
  return output;
}

/** One bounded source-pixel coordinate space drives the image, canvas and brush. */
export function maskGeometry(width: unknown, height: unknown, maxHeight = 460) {
  const validDimension = (value: unknown): value is number =>
    typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 4096;
  const valid = validDimension(width) && validDimension(height);
  const bounded = (value: unknown) =>
    typeof value === 'number' && Number.isFinite(value)
      ? Math.min(4096, Math.max(1, Math.floor(value)))
      : 1;
  const pixelWidth = bounded(width),
    pixelHeight = bounded(height);
  return {
    valid,
    width: pixelWidth,
    height: pixelHeight,
    maxStageWidth: Math.min(pixelWidth, (maxHeight * pixelWidth) / pixelHeight),
  };
}

/** Canvas bounds already exclude any centered margin around the image stage. */
export function maskPoint(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  width: number,
  height: number,
) {
  if (rect.width <= 0 || rect.height <= 0) return null;
  return {
    x: Math.max(0, Math.min(width, ((clientX - rect.left) * width) / rect.width)),
    y: Math.max(0, Math.min(height, ((clientY - rect.top) * height) / rect.height)),
  };
}
