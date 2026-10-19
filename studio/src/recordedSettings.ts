import type { Job, Workflow } from './contracts';

/** Accept only a complete recorded API graph, never a requested-draft substitute. */
export function effectivePrompt(job: Job): Workflow | null {
  const prompt = job.submission.prompt;
  if (
    prompt === null ||
    typeof prompt !== 'object' ||
    Array.isArray(prompt) ||
    Object.keys(prompt).length === 0 ||
    !Object.values(prompt).every(
      (node) =>
        node !== null &&
        typeof node === 'object' &&
        typeof node.class_type === 'string' &&
        node.inputs !== null &&
        typeof node.inputs === 'object' &&
        !Array.isArray(node.inputs),
    )
  )
    return null;
  return prompt as Workflow;
}

export function recordedSettings(job: Job, kind: 'prompt' | 'seed'): string {
  const prompt = effectivePrompt(job);
  if (!prompt) throw new Error('Recorded effective inputs are unavailable for this job.');
  // Group by exact execution identity: flattened names could collide, and
  // multiple nodes may carry separate seeds, text or model inputs.
  const value =
    kind === 'prompt'
      ? prompt
      : Object.fromEntries(Object.entries(prompt).map(([id, node]) => [id, node.inputs]));
  return JSON.stringify(value, null, 2);
}
