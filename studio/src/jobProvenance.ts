import type { StudioClient } from './api';
import type { Job } from './contracts';

/** List and snapshot rows are summaries; provenance actions require the full source record. */
export async function fullJob(client: StudioClient, id: string, candidate?: Job): Promise<Job> {
  if (candidate?.id === id && !candidate.summary && candidate.submission?.workflow)
    return candidate;
  const job = await client.request<Job>(`/jobs/${encodeURIComponent(id)}`);
  if (job.id !== id || job.summary)
    throw new Error('The companion did not return the requested full job provenance.');
  return job;
}
