import type { Artifact, Job } from './contracts';
import { isActive } from './queueRows.ts';

/** The job the output viewer follows. */
export interface Follow {
  id: string | null;
  /** This tab submitted the job; otherwise it is the newest active job. */
  own: boolean;
  /** The job's end was handled, or it had already ended when the follow began. */
  settled: boolean;
  /** The user chose a result or a comparison since the follow began. */
  picked: boolean;
  /** Answers from the artifacts route that lacked the succeeded job's final output. */
  lookups: number;
}
export const NO_FOLLOW: Follow = {
  id: null,
  own: false,
  settled: false,
  picked: false,
  lookups: 0,
};
export const ENDED_STATES = ['succeeded', 'failed', 'cancelled'];
/** How many times the artifacts route is asked for an output the snapshot left out. */
export const OUTPUT_LOOKUPS = 3;

/**
 * Follow a job this tab submitted. A job that had already ended when the follow began, such as
 * one a reload finds by its request identity, never shows as a new result.
 */
export function ownFollow(id: string, ended = false): Follow {
  return { id, own: true, settled: ended, picked: false, lookups: 0 };
}

/**
 * Confirm that this tab submitted `id`. A job the viewer already follows keeps what happened
 * since its follow began, such as a result the user chose while the submit reply was on its way.
 */
export function confirmOwn(follow: Follow, id: string): Follow {
  return follow.id === id ? { ...follow, own: true } : ownFollow(id);
}

/** The newest job that is active or unresolved, by creation time. */
export function newestActive(jobs: readonly Job[]): Job | undefined {
  let newest: Job | undefined;
  for (const job of jobs) {
    if (!isActive(job)) continue;
    const created = job.created_at ?? 0,
      best = newest?.created_at ?? 0;
    if (!newest || created > best || (created === best && job.id > newest.id)) newest = job;
  }
  return newest;
}

/** A job's own output: not an upload or mask, which have no job, and not a live preview. */
export function isJobOutput(artifact: Artifact): boolean {
  return (
    !!artifact.job_id &&
    artifact.source !== 'live_preview' &&
    (artifact.kind === 'image' || artifact.kind === 'video')
  );
}

/**
 * The result to show for a finished job: its newest final image or video. Outputs that one
 * history entry lists share a creation time, so among equal times the first listed wins. The
 * snapshot and the artifacts route both list equal times most recently stored first, so on
 * either path the first listed is the last output the job stored.
 */
export function finalOutput(jobId: string, artifacts: readonly Artifact[]): Artifact | null {
  let found: Artifact | null = null;
  for (const artifact of artifacts) {
    if (artifact.job_id !== jobId || artifact.intermediate || !isJobOutput(artifact)) continue;
    if (!found || (artifact.created_at ?? 0) > (found.created_at ?? 0)) found = artifact;
  }
  return found;
}

/**
 * Own the job a pending request identity names, once the snapshot lists it. When this page sent
 * the request (`fresh`), its job is new even if it ended before the snapshot listed it, as when
 * the submit reply was lost. A reload restores the identity without `fresh`, so a job that had
 * already ended is followed without showing as a new result.
 */
export function adoptRequest(
  follow: Follow,
  key: string,
  jobs: readonly Job[],
  fresh = false,
): Follow | null {
  const job = key ? jobs.find((item) => item.idempotency_key === key) : undefined;
  if (!job) return null;
  if (job.id === follow.id) return follow.own ? follow : { ...follow, own: true };
  return ownFollow(job.id, !fresh && ENDED_STATES.includes(job.state));
}

/**
 * Apply one snapshot. When the followed job succeeds, show its final output, unless the user
 * chose a result or a comparison since the follow began or a comparison is open. The snapshot
 * holds at most 200 artifacts across all jobs, so when it lacks the output, `lookup` names the
 * job to ask the artifacts route about, until `OUTPUT_LOOKUPS` answers have come back without
 * it. With no job from this tab, follow the newest active job; when none is active, keep the
 * last one.
 */
export function followSnapshot(
  follow: Follow,
  jobs: readonly Job[],
  artifacts: readonly Artifact[],
  comparing: boolean,
): { follow: Follow; show: Artifact | null; lookup: string | null } {
  let next = follow,
    show: Artifact | null = null,
    lookup: string | null = null;
  const job = next.id ? jobs.find((item) => item.id === next.id) : undefined;
  if (job && !next.settled && ENDED_STATES.includes(job.state)) {
    if (job.state !== 'succeeded' || next.picked || comparing) next = { ...next, settled: true };
    else {
      show = finalOutput(job.id, artifacts);
      if (show || next.lookups >= OUTPUT_LOOKUPS) next = { ...next, settled: true };
      else lookup = job.id;
    }
  }
  // A job whose output is still being looked up keeps the follow, so its result is not lost.
  if (!next.own && !lookup) {
    const newest = newestActive(jobs);
    if (newest && newest.id !== next.id)
      next = { id: newest.id, own: false, settled: false, picked: false, lookups: 0 };
  }
  return { follow: next, show, lookup };
}

/**
 * Apply the artifacts route's answer about a succeeded job's outputs; `items` is null when the
 * request failed, which counts as an answer without the output. An answer for a job the viewer
 * no longer follows, or one that has settled, changes nothing.
 */
export function applyLookup(
  follow: Follow,
  jobId: string,
  items: readonly Artifact[] | null,
  comparing: boolean,
): { follow: Follow; show: Artifact | null } {
  if (follow.id !== jobId || follow.settled) return { follow, show: null };
  if (follow.picked || comparing) return { follow: { ...follow, settled: true }, show: null };
  const show = items ? finalOutput(jobId, items) : null;
  const lookups = follow.lookups + 1;
  return { follow: { ...follow, lookups, settled: !!show || lookups >= OUTPUT_LOOKUPS }, show };
}

export function sameFollow(left: Follow, right: Follow): boolean {
  return (
    left.id === right.id &&
    left.own === right.own &&
    left.settled === right.settled &&
    left.picked === right.picked &&
    left.lookups === right.lookups
  );
}
