import { emptyDraft } from './model';
import type { Artifact, Job } from './contracts';
import type { Draft } from './model';

/** Reuse retains the frozen requested configuration and explicit output lineage. */
export function artifactDraft(artifact: Artifact, job: Job, operation?: string): Draft {
  const requested: Partial<Draft> = job.submission.requested ?? {};
  return {
    ...emptyDraft(),
    ...requested,
    workflow: job.submission.workflow ?? null,
    workflowName: `From ${artifact.id}`,
    operation: operation ?? requested.operation ?? 'image',
    parent_artifact_ids: [artifact.id],
  };
}
