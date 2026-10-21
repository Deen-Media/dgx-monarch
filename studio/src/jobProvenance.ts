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
