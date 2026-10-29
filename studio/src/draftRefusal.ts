import { ApiError } from './api.ts';

/** The companion's refusal of the draft's own content when Studio checked it for a job. */
export interface DraftRefusal {
  code: string;
  message: string;
}

/**
 * The refusal when the companion answered the check with HTTP 400 or 413 and a code: the draft
 * itself needs a change. Any other failure, such as a driver that is not running, a timeout or a
 * server fault, returns null and keeps its banner.
 */
export function draftRefusal(error: unknown): DraftRefusal | null {
  if (!(error instanceof ApiError) || !error.code) return null;
  if (error.status !== 400 && error.status !== 413) return null;
  return { code: error.code, message: error.message };
}

/** The Create tab that holds what a refusal names, and the words of the button that opens it. */
export interface AdjustTarget {
  tab: string;
  label: string;
}

const INPUTS = { tab: 'inputs', label: 'Open Inputs' };
const TARGETS: Record<string, AdjustTarget> = {
  lora_binding: { tab: 'loras', label: 'Open LoRAs' },
  lora_stack: { tab: 'loras', label: 'Open LoRAs' },
  styles: INPUTS,
  style_binding: INPUTS,
  seed_bounds: INPUTS,
  artifact_lineage: INPUTS,
};
// The companion names a refused value by its node and input, as in "5.steps: Value violates max=100".
const VALUE = /^[\w:-]+\.\w+: /;

/**
 * Where to fix a refusal, or null when no one tab holds the cause. A workflow refusal that names
 * a node's input is fixed among All inputs; any other names the graph, which Layout repairs.
 */
export function adjustTarget(refusal: DraftRefusal): AdjustTarget | null {
  if (refusal.code === 'workflow')
    return VALUE.test(refusal.message)
      ? { tab: 'advanced', label: 'Open All inputs' }
      : { tab: 'workflow', label: 'Open Layout' };
  return TARGETS[refusal.code] ?? null;
}
