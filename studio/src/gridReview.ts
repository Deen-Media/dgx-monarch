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
