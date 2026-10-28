import type { StudioClient } from './api';
import type { Job } from './contracts';

/** List and snapshot rows are summaries; provenance actions require the full source record. */
export async function fullJob(client: StudioClient, id: string, candidate?: Job): Promise<Job> {
  if (
    candidate?.id === id &&
    !candidate.summary &&
    (candidate.submission?.workflow || candidate.submission?.prompt)
  )
    return candidate;
  const job = await client.request<Job>(`/jobs/${encodeURIComponent(id)}`);
  if (job.id !== id || job.summary)
    throw new Error('The companion did not return the requested full job provenance.');
  return job;
}

/**
 * One job's summary row, read by its identity. Its recorded seeds keep every digit, which a full
 * record's JSON numbers lose in the browser above 2**53 - 1, and it is a few hundred bytes.
 */
export async function jobSummary(client: StudioClient, id: string): Promise<Job> {
  const page = await client.request<{ items?: Job[] }>(
    `/jobs?job_id=${encodeURIComponent(id)}&limit=1`,
  );
  const job = page?.items?.[0];
  if (job?.id !== id) throw new Error('The companion returned no summary for this job.');
  return job;
}

/**
 * A retry keeps the original immutable request in `submission` and records the runtime it ran
 * against in this job-level field. Comparison reads the retry's own values from it.
 */
export const RETRY_RUNTIME_FIELD = 'retry_runtime';
const RETRY_RUNTIME_KEYS = ['versions', 'runtime_observation', 'schema_digest', 'schema_identity'];

/** The submission fields to compare, with a retry's own recorded runtime when present. */
export function comparableSubmission(job: Job): Record<string, unknown> {
  const own = job[RETRY_RUNTIME_FIELD];
  if (!own || typeof own !== 'object' || Array.isArray(own)) return job.submission;
  const recorded = own as Record<string, unknown>;
  const overrides = RETRY_RUNTIME_KEYS.filter((key) => recorded[key] !== undefined).map((key) => [
    key,
    recorded[key],
  ]);
  return { ...job.submission, ...Object.fromEntries(overrides) };
}
