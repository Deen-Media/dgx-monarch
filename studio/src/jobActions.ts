import type { StudioClient } from './api';
import type { Job } from './contracts';

export function isActiveJob(job: Job): boolean {
  return ['queued', 'running', 'submitting', 'cancel_requested', 'accepted'].includes(job.state);
}

/** Keep the request identity if a transport failure leaves the result unknown. */
export async function requestJobAction(client: StudioClient, job: Job, action: 'cancel' | 'retry') {
  const description =
    action === 'cancel'
      ? `Request cancellation of ${job.id}? Cancellation is confirmed only by backend state.`
      : `Create a new linked retry of ${job.id} using its recorded configuration?`;
  if (!window.confirm(description)) return;
  const storageKey = `dgxm.studio.${action}.${job.id}`;
  const key = sessionStorage.getItem(storageKey) ?? crypto.randomUUID();
  sessionStorage.setItem(storageKey, key);
  await client.request(`/jobs/${encodeURIComponent(job.id)}/${action}`, {
    confirmed: true,
    idempotency_key: key,
  });
  sessionStorage.removeItem(storageKey);
}
