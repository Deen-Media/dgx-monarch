import { requireRecordIdentity, type StudioClient } from './api.ts';
import type { Job } from './contracts';
import { isActive } from './queueRows.ts';
import { submissionRefused } from './submissionRefusal.ts';

/**
 * Active or unresolved work, as Queue counts it. A job whose outcome is unknown stays counted
 * until Queue reconciles it, so it does not vanish from Create.
 */
export function isActiveJob(job: Job): boolean {
  return isActive(job);
}

/** Work still in flight to or on the driver; activity animations follow only these jobs. */
export function isInFlightJob(job: Job): boolean {
  return ['queued', 'running', 'submitting', 'cancel_requested', 'accepted'].includes(job.state);
}

/** Ask before a job action; a kept identity means an earlier attempt may already have landed. */
export function jobActionQuestion(job: Job, action: 'cancel' | 'retry', kept: boolean): string {
  if (!kept)
    return action === 'cancel'
      ? `Request cancellation of ${job.id}? Cancellation is confirmed only by backend state.`
      : `Create a new linked retry of ${job.id} using its recorded configuration?`;
  return action === 'cancel'
    ? `An earlier cancellation request for ${job.id} ended without a recorded answer in this tab. Send it again with the same request identity? The companion records at most one cancellation for that identity, and only backend state confirms it.`
    : `An earlier retry request for ${job.id} ended without a recorded answer in this tab. Send it again with the same request identity? If that request created a retry, the companion returns it instead of making another.`;
}

/** Explain a result answered under an identity kept from an earlier attempt. */
export function recoveredJobAction(job: Job, action: 'cancel' | 'retry', record: { id: string }) {
  return action === 'cancel'
    ? `The companion answered the cancellation request kept from an earlier attempt for ${job.id}, so at most one cancellation is recorded. The job state shows whether it took effect.`
    : `The companion answered the retry request kept from an earlier attempt with job ${record.id}, so at most one retry of ${job.id} exists for it. It may be the retry the earlier attempt created.`;
}

/**
 * Keep the request identity if a transport failure leaves the result unknown.
 * A definite refusal means the companion accepted nothing under the key, so
 * the next click starts a fresh request instead of a recovery. Returns the
 * record the companion answered with, or nothing when the user declined.
 */
export async function requestJobAction(
  client: StudioClient,
  job: Job,
  action: 'cancel' | 'retry',
  notify: (message: string) => void = () => {},
): Promise<{ id: string } | undefined> {
  const storageKey = `dgxm.studio.${action}.${job.id}`;
  const kept = sessionStorage.getItem(storageKey);
  if (!window.confirm(jobActionQuestion(job, action, !!kept))) return;
  const key = kept || crypto.randomUUID();
  sessionStorage.setItem(storageKey, key);
  let answer: unknown;
  try {
    answer = await client.request<{ id: string }>(`/jobs/${encodeURIComponent(job.id)}/${action}`, {
      confirmed: true,
      idempotency_key: key,
    });
  } catch (error) {
    if (submissionRefused(error, action)) sessionStorage.removeItem(storageKey);
    throw error;
  }
  const record = requireRecordIdentity(answer as { id: string });
  sessionStorage.removeItem(storageKey);
  if (kept) notify(recoveredJobAction(job, action, record));
  return record;
}
