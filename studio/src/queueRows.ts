import type { Job } from './contracts';

/** Job states the companion can still change; Queue keeps these rows above history. */
export const ACTIVE_JOB_STATES = [
  'submitting',
  'accepted',
  'queued',
  'running',
  'cancel_requested',
  'unknown',
];

export function isActive(job: Job): boolean {
  return ACTIVE_JOB_STATES.includes(job.state);
}

export interface SnapshotIds {
  active: string[];
  seen: string[];
}

/**
 * History loads only at mount and on filter changes, so a job that settles after that load
 * leaves both lists while the snapshot still holds it. Report the snapshot jobs that are settled
 * now and were active, or absent, in the previous snapshot, so Queue reloads history for them.
 * The first observation reports nothing because the mount load already covers it.
 */
export function settledJobs(previous: SnapshotIds | null, jobs: readonly Job[]) {
  const next = {
    active: jobs.filter(isActive).map((job) => job.id),
    seen: jobs.map((job) => job.id),
  };
  if (!previous) return { next, settled: [] as string[] };
  const active = new Set(previous.active),
    seen = new Set(previous.seen);
  const settled = jobs
    .filter((job) => !isActive(job) && (active.has(job.id) || !seen.has(job.id)))
    .map((job) => job.id);
  return { next, settled };
}

/** Keep the reader on the same history page after an automatic reload when it still exists. */
export function keptPage(page: number, rows: number, pageSize = 40): number {
  return Math.max(0, Math.min(page, Math.ceil(rows / pageSize) - 1));
}

/** Empty-state text that matches why the list is empty. */
export function emptyQueueText(query: string, status: string, loading: boolean) {
  if (loading)
    return { title: 'Loading job history…', detail: 'Active jobs appear as they arrive.' };
  if (query.trim() || status)
    return {
      title: 'No jobs match these filters.',
      detail: 'Clear the search or choose All history states to see every recorded job.',
    };
  return {
    title: 'No recorded jobs yet.',
    detail: 'Load a workflow and press Generate to make the first immutable job record.',
  };
}
