import { restoreDraft } from './model.ts';
import type { Artifact, Job, Workflow } from './contracts';
import type { Draft } from './model';
import { effectivePrompt } from './recordedSettings.ts';
import { roundedNumber } from './exactJson.ts';

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
  const large = largeNumbers(draft);
  // Name the prompt's inputs when they hold the number; the source workflow repeats them.
  const named = large.some((found) => found.input) ? large.filter((found) => found.input) : large;
  if (named.length)
    throw new Error(
      `Settings not loaded: ${named.map((found) => found.text).join(', ')}. A draft field holds whole numbers exactly only up to ${Number.MAX_SAFE_INTEGER}, so the draft is unchanged; Copy named inputs in the gallery keeps every digit.`,
    );
  return structuredClone(restoreDraft(draft));
}

/**
 * The whole numbers above 2**53 - 1 a draft would hold, each named by its prompt input
 * (`node id.input name`) or its path. A draft holds plain numbers, so it would carry such a
 * number rounded and send it changed: exactJob's BigInt gives the digits, and a number the
 * browser already rounded has none.
 */
function largeNumbers(value: unknown, path: string[] = []): { input: boolean; text: string }[] {
  if (typeof value === 'bigint' || roundedNumber(value)) {
    const input = path.length === 4 && path[0] === 'workflow' && path[2] === 'inputs';
    const name = input ? `${path[1]}.${path[3]}` : path.join('.');
    const text =
      typeof value === 'bigint'
        ? `${name} is ${value}`
        : `${name} is a whole number above ${Number.MAX_SAFE_INTEGER} that this browser holds rounded`;
    return [{ input, text }];
  }
  if (value === null || typeof value !== 'object') return [];
  return Object.entries(value).flatMap(([key, item]) => largeNumbers(item, [...path, key]));
}

/**
 * The notice after Reuse settings, Upscale or Load workflow loads a recorded draft. Send to edit
 * takes its own path and says what it bound.
 */
export function reuseNotice(operation?: string): string {
  return operation
    ? `Output lineage attached. Set the corresponding workflow asset input before generating ${operation}.`
    : 'Recorded settings loaded into a new editable draft.';
}

/** The snapshot's current copy of a chosen artifact, or the chosen copy when the snapshot omits it. */
export function liveArtifact(artifacts: Artifact[], item: Artifact | null): Artifact | null {
  return item ? (artifacts.find((artifact) => artifact.id === item.id) ?? item) : null;
}
