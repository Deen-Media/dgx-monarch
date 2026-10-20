/** Frame stepping is available only after finite media metadata has arrived. */
export function frameTarget(
  current: number,
  duration: number,
  frameRate: unknown,
  direction: number,
): number | null {
  if (
    typeof frameRate !== 'number' ||
    !Number.isFinite(frameRate) ||
    frameRate <= 0 ||
    !Number.isFinite(duration) ||
    duration <= 0 ||
    !Number.isFinite(current)
  )
    return null;
  return Math.max(0, Math.min(duration, current + direction / frameRate));
}
