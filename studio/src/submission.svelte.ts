import { ApiError, requireRecordIdentity } from './api';
import { submissionRefused } from './submissionRefusal';
import type { Connection } from './connection.svelte';
import type { Draft } from './model';
import type { Job, JobPreview } from './contracts';

const PENDING_KEY = 'dgxm.studio.pending';

/** One reviewed submission identity survives timeouts, reloads, and reconciliation. */
export class Submission {
  preview = $state<JobPreview | null>(null);
  kind = $state('job');
  confirmed = $state(false);
  pending = $state<{ key: string; preview_id: string } | null>(null);
  confirmedMissing = $state(false);
  recoveryError = $state('');
  private key = '';
  private connection: Connection;
  constructor(connection: Connection) {
    this.connection = connection;
  }

  review(preview: JobPreview, kind = 'job') {
    if (this.recoveryError) throw new Error(this.recoveryError);
    if (this.pending) throw new Error('Reconcile the pending submission before reviewing another.');
    this.preview = preview;
    this.kind = kind;
    this.key = crypto.randomUUID();
    this.confirmed = false;
  }

  async prepare(draft: Draft) {
    if (this.recoveryError) throw new Error(this.recoveryError);
    if (this.pending) throw new Error('Reconcile the pending submission before reviewing another.');
    this.review(await this.connection.client.request<JobPreview>('/jobs/preview', draft));
  }

  private clearPending() {
    this.pending = null;
    this.confirmedMissing = false;
    sessionStorage.removeItem(PENDING_KEY);
  }

  async submit() {
    if (this.recoveryError) throw new Error(this.recoveryError);
    if (this.pending) throw new Error('Reconcile or retry the retained submission first.');
    if (!this.preview || !this.confirmed) throw new Error('Review and confirm a submission first.');
    this.pending = { key: this.key, preview_id: this.preview.id };
    try {
      sessionStorage.setItem(PENDING_KEY, JSON.stringify(this.pending));
    } catch (error) {
      this.pending = null;
      throw error;
    }
    try {
      const job = await this.connection.client.request<Job>('/jobs', {
        preview_id: this.preview.id,
        idempotency_key: this.key,
      });
      requireRecordIdentity(job);
      this.clearPending();
      await this.connection.refresh();
      return job;
    } catch (error) {
      if (submissionRefused(error, 'job')) {
        this.clearPending();
        this.preview = null;
        this.confirmed = false;
      } else if (this.pending) {
        this.connection.error =
          'Submission outcome unknown until reconciled. Do not create a new submission.';
      }
      await this.connection.refresh().catch(() => {});
      throw error;
    }
  }

  async reconcile() {
    if (!this.pending) return true;
    try {
      requireRecordIdentity(
        await this.connection.client.request(
          `/jobs/request/${encodeURIComponent(this.pending.key)}`,
        ),
      );
      this.clearPending();
      await this.connection.refresh();
      return true;
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        this.confirmedMissing = true;
        return false;
      }
      throw error;
    }
  }

  async retryRetained() {
    if (!this.pending || !this.confirmedMissing)
      throw new Error('Reconcile the request before retrying.');
    this.confirmedMissing = false;
    try {
      requireRecordIdentity(
        await this.connection.client.request('/jobs', {
          preview_id: this.pending.preview_id,
          idempotency_key: this.pending.key,
        }),
      );
    } catch (error) {
      if (submissionRefused(error, 'job')) {
        this.clearPending();
        this.preview = null;
        this.confirmed = false;
      }
      throw error;
    }
    this.clearPending();
    await this.connection.refresh();
  }

  observe(jobs: Job[]) {
    if (!this.pending || !jobs.some((job) => job.idempotency_key === this.pending?.key))
      return false;
    this.clearPending();
    return true;
  }

  clearUnreadableRequest() {
    try {
      sessionStorage.removeItem(PENDING_KEY);
      this.recoveryError = '';
    } catch {
      this.recoveryError =
        'The unreadable request could not be cleared from browser storage. New submissions remain blocked.';
    }
  }
}
