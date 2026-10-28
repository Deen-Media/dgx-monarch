import type { Job, Seeds, Workflow } from './contracts';

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

/** The exact decimal digits the companion sends for a seed a browser number cannot hold. */
const DIGITS = /^-?\d+$/;

/**
 * The seeds a job ran with, keyed by execution identity, as a job summary carries them: a safe
 * integer as a number and a larger seed as its exact digits (contracts.ts, SeedValue). The
 * companion fills an imported job's seeds from its prompt. A number above
 * Number.MAX_SAFE_INTEGER comes from a full record and is already rounded, so it reads as unknown,
 * as does a missing list: null sends the reader to the job's summary.
 */
export function recordedSeeds(job: Job | null | undefined): Seeds | null {
  const recorded = job?.submission?.effective?.seeds;
  if (!recorded || typeof recorded !== 'object' || Array.isArray(recorded)) return null;
  const entries = Object.entries(recorded);
  if (entries.some(([, seed]) => typeof seed === 'number' && !Number.isSafeInteger(seed)))
    return null;
  return Object.fromEntries(
    entries.filter(([, seed]) => typeof seed === 'number' || DIGITS.test(String(seed))),
  );
}

/** One line for recorded seeds, every digit included, or '' when there is none to show. */
export function seedText(seeds: Seeds | null): string {
  const entries = Object.entries(seeds ?? {});
  if (entries.length === 1) return `Seed ${entries[0][1]}`;
  return entries.length
    ? `Seeds ${entries.map(([key, seed]) => `${key} = ${seed}`).join(', ')}`
    : '';
}

export function recordedSettings(job: Job, kind: 'prompt' | 'inputs' | 'seed'): string {
  if (kind === 'seed') {
    // A lone seed copies as the digits a seed field takes. Several keep their node identities as
    // JSON whose numbers carry every digit, which JSON.stringify cannot write above 2**53 - 1.
    const seeds = recordedSeeds(job);
    if (!seeds) throw new Error('Recorded seeds are unavailable for this job.');
    const entries = Object.entries(seeds);
    if (!entries.length) throw new Error('This job recorded no seed.');
    if (entries.length === 1) return String(entries[0][1]);
    return `{\n${entries.map(([key, seed]) => `  ${JSON.stringify(key)}: ${seed}`).join(',\n')}\n}`;
  }
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
