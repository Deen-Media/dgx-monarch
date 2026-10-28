import type { Job } from './contracts';
import type { Presentation } from './telemetryHistory';
import { firstSentence, jobFailure, type JobFailure } from './jobFailure.ts';

export { firstSentence };

/**
 * The browser twin of `format_duration_clock` in studio/presentation.py, the contract's
 * `duration_clock` rule: whole seconds rounded down, minutes uncapped (61:01), and unknown for a
 * missing, negative, non-finite or out-of-range value. A Python test keeps the two equal.
 */
export function formatDurationClock(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return 'unknown';
  if (value > Number.MAX_SAFE_INTEGER) return 'unknown';
  const whole = Math.floor(value);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

/** Seconds from `start` to `end`, or null when either is missing or the end comes first. */
export function duration(start: unknown, end: unknown): number | null {
  if (typeof start !== 'number' || typeof end !== 'number') return null;
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
  return end - start;
}

/** Seconds a snapshot clock may run on without a newer snapshot before it holds. */
export const CLOCK_LIMIT_S = 5;

/**
 * The companion's clock now: the newest snapshot's `server_time`, advanced by the browser's
 * monotonic timer since that snapshot arrived, at most `CLOCK_LIMIT_S` seconds. The browser's own
 * wall clock never enters, so a browser that runs fast or slow shows the same elapsed time.
 */
export function companionTime(
  anchor: { server: number; at: number } | null,
  monotonicMs: number,
): number | null {
  if (!anchor) return null;
  const since = Math.min(Math.max(0, (monotonicMs - anchor.at) / 1000), CLOCK_LIMIT_S);
  return anchor.server + since;
}

/**
 * Hold the clock through the small step back that request timing causes when a new snapshot
 * replaces the running estimate, so a time on screen never counts down. A larger step back is a
 * real change of the companion's clock and applies.
 */
export function steadyClock(previous: number | null, next: number | null): number | null {
  if (next === null || previous === null) return next;
  return next < previous && previous - next < 2 ? previous : next;
}

export type JobProgress = { value: number; max: number } | 'indeterminate' | null;
export interface JobLine {
  state: string;
  /** The state, with the elapsed or total time when both times are known. */
  headline: string;
  /** Steps from the driver's latest progress report, or the failure's one line. */
  detail: string;
  /** A failed job's error, read for display, or null. */
  failure: JobFailure | null;
  /** One node's reported steps, a bar with no known end, or no bar. */
  progress: JobProgress;
  /** The text of a link to Queue, or ''. */
  link: string;
  /** What a screen reader hears when the state changes; never the ticking time or steps. */
  announcement: string;
}

const LABELS: Record<string, string> = {
  submitting: 'Submitting',
  accepted: 'Submitting',
  queued: 'Queued',
  running: 'Running',
  cancel_requested: 'Cancel requested',
  succeeded: 'Finished',
  failed: 'Failed',
  cancelled: 'Cancelled',
  unknown: 'Outcome unknown',
};

/** The step count of one node's latest progress report; reports are never merged across nodes. */
function steps(job: Job): { value: number; max: number; text: string } | null {
  const report = job.progress;
  const max = report?.max,
    value = report?.value ?? 0;
  if (typeof max !== 'number' || !Number.isFinite(max) || max <= 0) return null;
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  const node = report?.node;
  const where = node === undefined || node === null || node === '' ? '' : ` on node ${node}`;
  return { value: Math.min(Math.max(value, 0), max), max, text: `step ${value} of ${max}${where}` };
}

/**
 * What the viewer's status line says about one job. `now` is the companion's clock, or null.
 * Each time uses one clock: a running job counts on the companion's clock from when the
 * companion first saw it run, since the driver may run on another host with its own clock, and
 * a job the driver reports ended shows the driver's own start to end time. No time appears when
 * a time is missing, and no line ever estimates the time left.
 */
export function jobLine(job: Job, now: number | null): JobLine {
  const state = job.state;
  const label = LABELS[state] ?? state;
  const end = typeof job.ended_at === 'number' ? job.ended_at : null;
  let headline = label,
    detail = '',
    failure: JobFailure | null = null,
    progress: JobProgress = null,
    link = '';
  if (state === 'running' || state === 'cancel_requested') {
    const reported = steps(job);
    if (reported) {
      detail = reported.text;
      progress = { value: reported.value, max: reported.max };
    } else if (state === 'running') progress = 'indeterminate';
    const elapsed =
      state !== 'running'
        ? null
        : end !== null
          ? duration(job.started_at, end)
          : duration(job.running_observed_at, now);
    if (elapsed !== null) headline = `${label} ${formatDurationClock(elapsed)}`;
  } else if (state === 'succeeded' || state === 'failed' || state === 'cancelled') {
    const total = duration(job.started_at, end);
    if (total !== null)
      headline = `${label} ${state === 'succeeded' ? 'in' : 'after'} ${formatDurationClock(total)}`;
  }
  if (state === 'failed') {
    failure = jobFailure(job.error);
    detail = failure?.summary ?? '';
    link = 'Open Queue';
  }
  if (state === 'unknown') link = 'Reconcile in Queue';
  const announcement = [label, state === 'failed' ? detail : '', state === 'unknown' ? link : '']
    .filter(Boolean)
    .join('. ');
  return { state, headline, detail, failure, progress, link, announcement };
}

/** The status token color the shared presentation contract gives a job state, if any. */
export function jobColor(
  presentation: Presentation | undefined,
  state: string,
): string | undefined {
  const token = presentation?.job_states?.[state];
  return token ? presentation?.status?.[token]?.hex : undefined;
}
