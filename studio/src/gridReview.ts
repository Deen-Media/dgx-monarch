/** Keep a frozen grid review tied to the editable definition that produced it. */
export class GridReviewGuard {
  private reviewed: string | null = null;

  accept(requested: string, current: string) {
    if (requested !== current)
      throw new Error('Grid definition changed during validation. Validate every cell again.');
    this.reviewed = requested;
  }

  restore(current: string) {
    // A restored identity remains usable for reconciliation. Subsequent edits
    // must still invalidate an unaccepted review.
    this.reviewed = current;
  }

  invalidate(current: string, dispatchPending: boolean): boolean {
    if (dispatchPending || this.reviewed === null || this.reviewed === current) return false;
    this.reviewed = null;
    return true;
  }
}

export const STALE_REVIEW = 'Grid definition changed. Validate every cell again before submission.';
/**
 * The Grids edit effect: when an edit invalidates a frozen review that is on screen, return the
 * message to show; the view then drops the preview, its confirmation and its request key.
 */
export function staleReview(
  guard: GridReviewGuard,
  definition: string,
  dispatchPending: boolean,
  hasPreview: boolean,
): string | null {
  return guard.invalidate(definition, dispatchPending) && hasPreview ? STALE_REVIEW : null;
}
