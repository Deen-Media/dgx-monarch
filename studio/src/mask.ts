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
