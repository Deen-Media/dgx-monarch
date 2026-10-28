import { ApiError, requireRecordIdentity } from './api';
import { submissionRefused } from './submissionRefusal';
import type { Connection } from './connection.svelte';
import type { Draft } from './model';
import type { Job, JobPreview } from './contracts';

const PENDING_KEY = 'dgxm.studio.pending';

/** One reviewed submission identity survives timeouts, reloads, and reconciliation. */
export class Submission {
  preview = $state<JobPreview | null>(null);
  /** The browser's monotonic time, in ms, when the review arrived; it anchors the expiry clock. */
  arrivedAt = 0;
  confirmed = $state(false);
  pending = $state<{ key: string; preview_id: string } | null>(null);
  confirmedMissing = $state(false);
  recoveryError = $state('');
  /** Why the pending submission is unresolved; the pending banner shows it. */
  outcome = $state('');
  /**
   * The request identity this page sent, by a submit or a retry of the retained request. An
   * identity restored after a reload is not it, so a job that had already ended then is old.
   */
  createdKey = '';
  private key = '';
  private connection: Connection;
  constructor(connection: Connection) {
    this.connection = connection;
  }

  review(preview: JobPreview) {
    if (this.recoveryError) throw new Error(this.recoveryError);
    if (this.pending) throw new Error('Reconcile the pending submission before reviewing another.');
    this.arrivedAt = performance.now();
    this.preview = preview;
    this.key = crypto.randomUUID();
    this.confirmed = false;
  }

  /**
   * Review a draft snapshot. The preview is dropped, and false returned, when
   * the companion connection or the draft changed while it was prepared.
   */
  async prepare(draft: Draft, current: () => boolean = () => true): Promise<boolean> {
    if (this.recoveryError) throw new Error(this.recoveryError);
    if (this.pending) throw new Error('Reconcile the pending submission before reviewing another.');
    const client = this.connection.client;
    const preview = await client.request<JobPreview>('/jobs/preview', draft);
    if (client !== this.connection.client || !current()) return false;
    this.review(preview);
    return true;
  }

  private clearPending() {
    this.pending = null;
    this.confirmedMissing = false;
    this.outcome = '';
    sessionStorage.removeItem(PENDING_KEY);
  }

  async submit() {
    if (this.recoveryError) throw new Error(this.recoveryError);
    if (this.pending) throw new Error('Reconcile or retry the retained submission first.');
    if (!this.preview || !this.confirmed) throw new Error('Review and confirm a submission first.');
    this.createdKey = this.key;
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
        this.outcome =
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
    this.createdKey = this.pending.key;
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
