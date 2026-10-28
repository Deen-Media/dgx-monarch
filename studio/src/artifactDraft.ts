import { restoreDraft } from './model.ts';
import type { Artifact, Job, Workflow } from './contracts';
import type { Draft } from './model';
import { effectivePrompt } from './recordedSettings.ts';

/** Reuse the actual submitted graph; its seeds, styles and LoRAs are already resolved. */
export function artifactDraft(artifact: Artifact, job: Job, operation?: string): Draft {
  const requested = (job.submission.requested ?? {}) as Partial<Draft> & {
    source_workflow?: Workflow;
    source_workflowName?: string;
  };
  const prompt = effectivePrompt(job);
  const effective = prompt !== null;
  if (!effective && !job.submission.workflow)
    throw new Error('Source workflow and recorded effective inputs are unavailable for this job.');
  // restoreDraft fills the fields a record leaves out; a missing seed policy restores as the
  // companion's default, because that is the policy the recorded request ran with.
  const draft = {
    ...requested,
    ...(effective
      ? {
          source_workflow: requested.source_workflow ?? job.submission.workflow,
          source_workflowName: requested.source_workflowName ?? requested.workflowName,
          workflow: prompt as Workflow,
          workflowName: `Effective settings from ${artifact.id}`,
          values: {},
          styles: [],
          lora_stack: [],
          seed_policy: 'fixed',
        }
      : {
          workflow: job.submission.workflow ?? null,
          workflowName: `Requested settings from ${artifact.id}`,
        }),
    operation: operation ?? requested.operation ?? 'image',
    parent_artifact_ids: [artifact.id],
  };
  if (effective) {
    delete draft.style_bindings;
    delete draft.lora_insertion;
  }
  return structuredClone(restoreDraft(draft));
}

/** The snapshot's current copy of a chosen artifact, or the chosen copy when the snapshot omits it. */
export function liveArtifact(artifacts: Artifact[], item: Artifact | null): Artifact | null {
  return item ? (artifacts.find((artifact) => artifact.id === item.id) ?? item) : null;
}
