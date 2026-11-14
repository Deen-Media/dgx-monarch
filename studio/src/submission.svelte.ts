import { ApiError, requireRecordIdentity } from './api';
import { draftRefusal, type DraftRefusal } from './draftRefusal';
import { reviewDecided } from './reviewSummary';
import { submissionRefused } from './submissionRefusal';
import type { Connection } from './connection.svelte';
import type { Binding, Draft } from './model';
import type { Job, JobPreview } from './contracts';

const PENDING_KEY = 'dgxm.studio.pending';
/** What an accepted submit says for a moment; Queue and the job's details keep its identifier. */
export const SUBMITTED = 'Job submitted. The viewer follows it, and Queue has its details.';
export const CONFIRMATION_MS = 5000;

/**
 * What a check of the draft leads to: start the job at once, open the review because it was
 * asked for or something needs a person, or nothing, as the draft or connection changed.
 */
export type Prepared = 'ready' | 'review' | 'dropped';

/** One reviewed submission identity survives timeouts, reloads, and reconciliation. */
export class Submission {
  preview = $state<JobPreview | null>(null);
  /** The companion's refusal of the draft at the check; the dialog shows it with its fix. */
  refusal = $state<DraftRefusal | null>(null);
  /** The bindings the review was made with; the summary reads labels and prompt inputs there. */
  reviewed = $state.raw<Binding[]>([]);
  /** The browser's monotonic time, in ms, when the review arrived; it anchors the expiry clock. */
  arrivedAt = 0;
  /** One box per backend warning, in order; the job starts only once each is ticked. */
  acknowledged = $state<boolean[]>([]);
  /** A brief line after an accepted submit; it clears itself after CONFIRMATION_MS. */
  confirmation = $state('');
  #shown = 0;
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
    this.refusal = null;
    this.key = crypto.randomUUID();
    this.acknowledged = (preview.warnings ?? []).map(() => false);
  }

  /** Tick or clear the box of one warning. */
  acknowledge(index: number, accepted: boolean) {
    this.acknowledged[index] = accepted;
  }

  /** Whether the review holds no error to fix and no warning left unticked. */
  get decided() {
    return !!this.preview && reviewDecided(this.preview, this.acknowledged);
  }

  /**
   * Check a draft snapshot with the companion, which makes the immutable record. It is 'ready'
   * when nothing needs a person and `asked` is false, so the caller submits at once; 'review'
   * when the caller asked for the dialog, a warning or error needs a person, or the companion
   * refused the draft; and 'dropped', with no review kept, when the companion connection or the
   * draft changed meanwhile. Any other failure throws for the banner.
   */
  async prepare(
    draft: Draft,
    current: () => boolean = () => true,
    asked = false,
    bindings: Binding[] = [],
  ): Promise<Prepared> {
    if (this.recoveryError) throw new Error(this.recoveryError);
    if (this.pending) throw new Error('Reconcile the pending submission before reviewing another.');
    const client = this.connection.client;
    this.clearConfirmation();
    let preview: JobPreview;
    try {
      preview = await client.request<JobPreview>('/jobs/preview', draft);
    } catch (error) {
      const refusal = draftRefusal(error);
      if (!refusal) throw error;
      if (client !== this.connection.client || !current()) return 'dropped';
      this.preview = null;
      this.refusal = refusal;
      return 'review';
    }
    if (client !== this.connection.client || !current()) return 'dropped';
    this.review(preview);
    this.reviewed = bindings;
    return asked || !this.decided ? 'review' : 'ready';
  }

  /** Show SUBMITTED, and clear it after CONFIRMATION_MS unless a later one replaced it. */
  private confirm() {
    const shown = ++this.#shown;
    this.confirmation = SUBMITTED;
    setTimeout(() => {
      if (shown === this.#shown) this.confirmation = '';
    }, CONFIRMATION_MS);
  }

  private clearConfirmation() {
    this.#shown++;
    this.confirmation = '';
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
    if (!this.preview) throw new Error('Check the draft before submitting it.');
    if (!this.decided) throw new Error('Tick each warning, or fix the draft, before generating.');
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
      this.confirm();
      await this.connection.refresh();
      return job;
    } catch (error) {
      if (submissionRefused(error, 'job')) {
        this.clearPending();
        this.preview = null;
        this.acknowledged = [];
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
        this.acknowledged = [];
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
