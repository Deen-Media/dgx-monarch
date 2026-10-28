import type { Job } from './contracts';

/**
 * A request to open Queue on one job's row. The viewer status line's Queue link makes it, and
 * the next Queue view takes it as it mounts, once, so a later visit opens on no row. It lives
 * outside Queue, because Queue mounts again on every visit.
 */
export class QueueFocus {
  #id: string | null = null;

  /** Ask the next Queue view to open on this job's row. */
  request(id: string) {
    this.#id = id;
  }

  /** The requested job, once: null afterwards, and when nothing asked. */
  take(): string | null {
    const id = this.#id;
    this.#id = null;
    return id;
  }
}

/** One request per browser tab. */
export const queueFocus = new QueueFocus();

/** The element id of a job's Queue row, so Queue can scroll the requested row into view. */
export function queueRowId(id: string): string {
  return `queue-job-${id}`;
}

/**
 * Where Queue finds a job it was asked to open: among the active rows, which every history page
 * shows, or on a page of the loaded history. Null when neither holds it: Queue then reads the
 * job's summary by its identity and pins the row above the history pages.
 */
export function focusPlace(
  id: string,
  active: readonly Job[],
  terminal: readonly Job[],
  pageSize = 40,
): { job: Job; page: number | null } | null {
  const live = active.find((job) => job.id === id);
  if (live) return { job: live, page: null };
  const index = terminal.findIndex((job) => job.id === id);
  return index < 0 ? null : { job: terminal[index], page: Math.floor(index / pageSize) };
}
