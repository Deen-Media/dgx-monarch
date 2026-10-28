import { untrack } from 'svelte';
import type { Connection } from './connection.svelte';
import type { Submission } from './submission.svelte';
import type { Artifact, Job } from './contracts';
import { liveArtifact } from './artifactDraft';
import { isInFlightJob } from './jobActions';
import {
  ENDED_STATES,
  NO_FOLLOW,
  adoptRequest,
  applyLookup,
  confirmOwn,
  followSnapshot,
  sameFollow,
  type Follow,
} from './jobFollow';
import { jobSummary } from './jobProvenance';
import { companionTime, jobColor, jobLine, steadyClock, type JobLine } from './jobStatus';
import { consentAction, type ConsentAction } from './jobFailure';

export interface FollowedJob extends JobLine {
  /** The job's identity; the status line's Queue link opens its row. */
  id: string;
  /** This tab submitted the job; otherwise it is the newest active job. */
  own: boolean;
  /** The job state's status token color from the shared presentation contract. */
  color?: string;
  /** The consent control when the failure asks for consent, or null. */
  consent?: ConsentAction | null;
}

/**
 * Owns what the output viewer shows: the chosen result, the comparison, and the job the viewer
 * follows. Setting `selected` or `comparison` is a user choice; the newest result of the
 * followed job never replaces a choice made after the follow began.
 */
export class JobTracker {
  follow = $state.raw(NO_FOLLOW);
  #selected = $state<Artifact | null>(null);
  #comparison = $state<Artifact | null>(null);
  #clock = $state<number | null>(null);
  #anchor: { server: number; at: number } | null = null;
  #request = '';
  /** The snapshot whose missing output the last artifacts request looked for. */
  #lookedFor: number | undefined;
  /** The artifacts request in flight for a succeeded job's output, if any. */
  lookup: Promise<void> | null = null;
  /** The followed job read by its identity after a truncated snapshot left it out. */
  #record = $state.raw<Job | null>(null);
  /** The snapshot the last identity read answered; one read per snapshot at most. */
  #readFor: number | undefined;
  /** The identity read in flight for the followed job, if any. */
  reading: Promise<void> | null = null;
  private connection: Connection;
  private submission: Submission;
  constructor(connection: Connection, submission: Submission) {
    this.connection = connection;
    this.submission = submission;
  }

  get selected() {
    return this.#selected;
  }
  set selected(artifact: Artifact | null) {
    this.#selected = artifact;
    this.follow = { ...this.follow, picked: true };
  }
  get comparison() {
    return this.#comparison;
  }
  set comparison(artifact: Artifact | null) {
    this.#comparison = artifact;
    this.follow = { ...this.follow, picked: true };
  }

  /** The snapshot's current copies of the chosen result and comparison. */
  shown = $derived.by(() => liveArtifact(this.connection.snapshot.artifacts, this.#selected));
  shownComparison = $derived.by(() =>
    liveArtifact(this.connection.snapshot.artifacts, this.#comparison),
  );
  /** The followed job as the snapshot lists it, or as read by its identity once the window drops it. */
  job = $derived.by(() => {
    const id = this.follow.id;
    if (!id) return undefined;
    const listed = this.connection.snapshot.jobs.find((job) => job.id === id);
    return listed ?? (this.#record?.id === id ? this.#record : undefined);
  });
  status = $derived.by((): FollowedJob | null => {
    const job = this.job;
    if (!job) return null;
    const { presentation, operations } = this.connection.capabilities;
    const line = jobLine(job, this.#clock);
    const consent = consentAction(line.failure, operations);
    const color = jobColor(presentation, job.state);
    return { ...line, id: job.id, own: this.follow.own, color, consent };
  });
  /** Whether any job is still in flight; activity animations follow this, not unresolved jobs. */
  working = $derived.by(() => this.connection.snapshot.jobs.some(isInFlightJob));
  private ticking = $derived(
    this.job?.state === 'running' &&
      typeof this.job.running_observed_at === 'number' &&
      typeof this.job.ended_at !== 'number',
  );

  /** Follow a job this tab just submitted; one it already follows keeps the user's choice. */
  own(id: string) {
    this.#request = '';
    this.follow = confirmOwn(this.follow, id);
  }

  /** Run in an effect: apply each snapshot to the followed job and the viewer. */
  observe() {
    const { jobs, artifacts, server_time: server, jobs_truncated } = this.connection.snapshot;
    const pending = this.submission.pending?.key;
    const follow = this.follow;
    untrack(() => {
      if (typeof server === 'number' && server !== this.#anchor?.server) {
        this.#anchor = { server, at: performance.now() };
        this.advance();
      }
      if (pending) this.#request = pending;
      const fresh = !!this.#request && this.#request === this.submission.createdKey;
      const adopted = adoptRequest(follow, this.#request, jobs, fresh);
      if (adopted) this.#request = '';
      const current = adopted ?? follow;
      const step = followSnapshot(
        current,
        this.known(jobs, current.id),
        artifacts,
        this.#comparison !== null,
      );
      this.apply(follow, step);
      this.readDropped(jobs, jobs_truncated === true, server);
      // One request per snapshot at most, so the bounded attempts span several polls.
      if (step.lookup && !this.lookup && server !== this.#lookedFor) {
        this.#lookedFor = server;
        this.lookup = this.lookUp(step.lookup).finally(() => (this.lookup = null));
      }
    });
  }

  /** The snapshot's jobs, with the followed job's record when the snapshot window left it out. */
  private known(jobs: readonly Job[], id: string | null): readonly Job[] {
    const record = this.#record;
    if (!id || record?.id !== id || jobs.some((job) => job.id === id)) return jobs;
    return [...jobs, record];
  }

  /**
   * Read the followed job by its identity when a truncated snapshot leaves it out: once per
   * snapshot while it may still change, and no more once the record shows it ended. A job the
   * snapshot carries is never read, and its old record is dropped.
   */
  private readDropped(jobs: readonly Job[], truncated: boolean, server: number | undefined) {
    const id = this.follow.id;
    if (!id) return;
    if (jobs.some((job) => job.id === id)) {
      if (this.#record) this.#record = null;
      return;
    }
    const record = this.#record?.id === id ? this.#record : null;
    if (!truncated || this.reading || server === this.#readFor) return;
    if (record && ENDED_STATES.includes(record.state)) return;
    this.#readFor = server;
    this.reading = this.readJob(id).finally(() => (this.reading = null));
  }

  private async readJob(id: string) {
    try {
      const job = await jobSummary(this.connection.client, id);
      if (this.follow.id === id) this.#record = job;
    } catch {
      // No record this time; the next snapshot asks again while the job may still change.
    }
  }

  /** Ask the artifacts route for a succeeded job's final output that the snapshot left out. */
  private async lookUp(jobId: string) {
    let items: Artifact[] | null = null;
    try {
      const page = await this.connection.client.request<{ items?: Artifact[] }>(
        `/artifacts?job_id=${encodeURIComponent(jobId)}&intermediate=false&sort=newest&limit=200`,
      );
      items = Array.isArray(page?.items) ? page.items : null;
    } catch {
      // A failed request is an answer without the output; the attempts stay bounded.
    }
    this.apply(this.follow, applyLookup(this.follow, jobId, items, this.#comparison !== null));
  }

  private apply(previous: Follow, step: { follow: Follow; show: Artifact | null }) {
    if (step.show) this.#selected = step.show;
    if (!sameFollow(step.follow, previous)) this.follow = step.follow;
  }

  /** Run in an effect: count the followed job's time once a second while it runs in view. */
  run(hidden: boolean) {
    if (hidden || !this.ticking) return;
    const timer = setInterval(() => this.advance(), 1000);
    return () => clearInterval(timer);
  }

  private advance() {
    this.#clock = steadyClock(this.#clock, companionTime(this.#anchor, performance.now()));
  }
}
