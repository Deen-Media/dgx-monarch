/** The keys that move along the Recent Outputs strip: Left to the previous output, Right to the next. */
const STEPS: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1 };

export type StepKey = Pick<
  KeyboardEvent,
  'key' | 'altKey' | 'ctrlKey' | 'metaKey' | 'shiftKey' | 'isComposing'
>;

/**
 * The index of the Recent Output an arrow key moves to from the tile at `index`, or null when the
 * key moves nothing: another key, a held modifier, text still being composed, or the end of the
 * strip. Only the strip's tiles listen, so arrows typed in a text field or a select never move it.
 */
export function recentStep(event: StepKey, index: number, count: number): number | null {
  const step = STEPS[event.key];
  if (!step || event.isComposing) return null;
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return null;
  const target = index + step;
  return target >= 0 && target < count ? target : null;
}
