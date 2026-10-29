import { DRIVER_NOT_RUNNING } from './firstRun.ts';

export interface Binding {
  id: string;
  aliases?: string[];
  label?: string;
  name?: string;
  input_name?: string;
  node_id?: string;
  /** The graph that holds the node: the root graph's id, or a subgraph definition's. */
  graph_id?: string;
  /** The subgraph instance node ids from the root graph down to the node; empty at the root. */
  instance_path?: string[];
  /** The compiled prompt node that receives this input; a subgraph instance gives it its own id. */
  execution_node_id?: string;
  type?: string;
  data_type?: string;
  value?: unknown;
  default?: unknown;
  options?: unknown[];
  enum?: unknown[];
  min?: number;
  max?: number;
  step?: number;
  description?: string;
  group?: string;
  status?: string;
  resolution_status?: string;
  resolved?: boolean;
  reason?: string | null;
  schema?: {
    type?: string;
    choices?: unknown[];
    min?: number;
    max?: number;
    step?: number;
    default?: unknown;
    /** A dynamic choice: changing it changes which child inputs the node declares. */
    dynamic?: boolean;
    /** Text the driver draws as a multi-line box, such as a prompt. */
    multiline?: boolean;
  };
  required?: boolean;
  validation?: {
    min?: number;
    max?: number;
    step?: number;
    options?: unknown[];
    /** A required value may not be empty text, as management operations require. */
    nonEmpty?: boolean;
  };
  display?: {
    label?: string;
    group?: string;
    order?: number;
    prominent?: boolean;
  };
  [key: string]: unknown;
}
export interface Lora {
  id: string;
  name: string;
  enabled: boolean;
  model_strength: number;
  clip_strength?: number;
  source_node?: string;
}
export interface Draft {
  workflow: Record<string, any> | null;
  workflowName: string;
  values: Record<string, unknown>;
  operation: string;
  project_id: string;
  styles: unknown[];
  lora_stack: Lora[];
  lora_insertion?: {
    loader: string;
    model_targets: Array<{ node_id: string; input_name: string }>;
    clip_targets: Array<{ node_id: string; input_name: string }>;
  };
  style_bindings?: { prompt?: string; negative_prompt?: string };
  seed_policy: string;
  parent_artifact_ids: string[];
}
export const emptyDraft = (): Draft => ({
  workflow: null,
  workflowName: '',
  values: {},
  operation: 'image',
  project_id: '',
  styles: [],
  lora_stack: [],
  // New drafts choose their seeds at review; Fixed sends the seed fields as they are.
  seed_policy: 'random',
  parent_artifact_ids: [],
});
/** The companion's policy for a request that names none, so a saved record keeps its meaning. */
const RECORDED_SEED_POLICY = 'fixed';
export function fieldLabel(b: Binding) {
  return b.display?.label ?? b.label ?? b.name ?? b.input_name ?? b.id;
}
export function bindingType(b: Binding) {
  return String(b.schema?.type ?? b.type ?? b.data_type ?? '').toUpperCase();
}
export function bindingOptions(b: Binding): unknown[] {
  return b.schema?.choices ?? b.options ?? b.enum ?? b.validation?.options ?? [];
}
export function bindingError(b: Binding, value: unknown): string | null {
  if (b.resolved === false)
    return b.reason ?? 'Unresolved input; repair workflow before submission.';
  const status = b.resolution_status ?? b.status;
  if (status && !['resolved', 'supported', 'ok'].includes(status))
    return `Binding ${status}; repair workflow before submission.`;
  // ComfyUI takes empty text for a required input, such as an empty negative prompt;
  // only the input's own rule refuses it.
  if (b.required && (value === undefined || (value === '' && b.validation?.nonEmpty)))
    return 'A value is required.';
  const type = bindingType(b);
  if (['INT', 'INTEGER', 'FLOAT', 'NUMBER'].includes(type)) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return 'Enter a finite number.';
    if (['INT', 'INTEGER'].includes(type) && !Number.isSafeInteger(value))
      return 'Enter a whole number within the browser safe integer range.';
    const min = b.schema?.min ?? b.min ?? b.validation?.min,
      max = b.schema?.max ?? b.max ?? b.validation?.max;
    if (min !== undefined && value < min) return `Minimum ${min}.`;
    if (max !== undefined && value > max) return `Maximum ${max}.`;
  }
  const options = bindingOptions(b);
  if (options.length && !options.includes(value)) return 'Choose an available value.';
  return null;
}
export function consumeEvents(current: number, events: any[]): { cursor: number; events: any[] } {
  const seen = new Set<number>();
  const accepted = events
    .filter((e) => {
      const seq = Number(e.sequence ?? e.seq ?? e.cursor);
      if (!Number.isSafeInteger(seq) || seq <= current || seen.has(seq)) return false;
      seen.add(seq);
      return true;
    })
    .sort(
      (a, b) => Number(a.sequence ?? a.seq ?? a.cursor) - Number(b.sequence ?? b.seq ?? b.cursor),
    );
  return { cursor: Math.max(current, ...seen), events: accepted };
}
export function boundedLog(lines: string[], lineLimit = 400, byteLimit = 128 * 1024) {
  const encoder = new TextEncoder();
  const result: string[] = [];
  let bytes = 0;
  for (const line of lines.slice(-lineLimit).reverse()) {
    const size = encoder.encode(line).length;
    if (bytes + size > byteLimit) break;
    bytes += size;
    result.push(line);
  }
  return result.reverse();
}
export function displayUnknown(value: unknown): string {
  return value === undefined || value === null
    ? 'Unknown'
    : typeof value === 'object'
      ? JSON.stringify(value)
      : String(value);
}
export function restoreDraft(value: unknown): Draft {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Invalid draft record.');
  const source = value as Partial<Draft>;
  const object = (entry: unknown): entry is Record<string, unknown> =>
    entry !== null && typeof entry === 'object' && !Array.isArray(entry);
  const invalid = (field: string): never => {
    throw new Error(`Invalid draft ${field}.`);
  };
  for (const field of ['workflowName', 'operation', 'project_id', 'seed_policy'] as const)
    if (source[field] !== undefined && typeof source[field] !== 'string') invalid(field);
  if (source.workflow !== undefined && source.workflow !== null && !object(source.workflow))
    invalid('workflow');
  for (const field of ['styles', 'lora_stack'] as const)
    if (Array.isArray(source[field]) && source[field]!.some((entry) => !object(entry)))
      invalid(field);
  if (
    Array.isArray(source.parent_artifact_ids) &&
    source.parent_artifact_ids.some((id) => typeof id !== 'string')
  )
    invalid('parent_artifact_ids');
  if (
    source.style_bindings !== undefined &&
    (!object(source.style_bindings) ||
      Object.values(source.style_bindings).some((entry) => typeof entry !== 'string'))
  )
    invalid('style_bindings');
  let insertion = source.lora_insertion ?? undefined;
  if (insertion !== undefined) {
    if (!object(insertion) || typeof insertion.loader !== 'string') invalid('lora_insertion');
    insertion = { ...insertion };
    for (const field of ['model_targets', 'clip_targets'] as const) {
      const targets = insertion[field] ?? [];
      if (
        !Array.isArray(targets) ||
        targets.some(
          (entry) =>
            !object(entry) ||
            typeof entry.node_id !== 'string' ||
            typeof entry.input_name !== 'string',
        )
      )
        invalid(`lora_insertion.${field}`);
      insertion[field] = targets;
    }
  }
  return {
    ...emptyDraft(),
    ...source,
    ...(source.lora_insertion !== undefined ? { lora_insertion: insertion } : {}),
    values: object(source.values) ? source.values : {},
    lora_stack: Array.isArray(source.lora_stack) ? source.lora_stack : [],
    styles: Array.isArray(source.styles) ? source.styles : [],
    parent_artifact_ids: Array.isArray(source.parent_artifact_ids)
      ? source.parent_artifact_ids
      : [],
    seed_policy: source.seed_policy ?? RECORDED_SEED_POLICY,
  };
}

/** An integer seed input, which the Random policy replaces at review as the companion does. */
export function isSeedBinding(b: Binding): boolean {
  return (
    b.resolved !== false &&
    bindingType(b) === 'INT' &&
    ['seed', 'noise_seed'].includes(b.input_name ?? '')
  );
}

/**
 * The draft change that puts recorded seeds into the draft's seed fields with the Fixed policy.
 * Seeds are keyed by execution identity, `node id.input name`. A single seed also fits a draft
 * with a single seed field whose node id differs. Returns the reason instead when none fits, or
 * when a seed is too large for a draft field to hold exactly.
 */
export function seedChange(
  draft: Draft,
  bindings: Binding[],
  seeds: Record<string, number | string>,
): { change: Partial<Draft>; notice: string } | { error: string } {
  const recorded = Object.entries(seeds);
  if (!draft.workflow) return { error: 'Load a workflow before using a recorded seed.' };
  if (!recorded.length) return { error: 'This output has no recorded seed.' };
  const fields = bindings.filter(isSeedBinding);
  const byKey = new Map(
    fields.map((b) => [`${b.execution_node_id ?? b.node_id}.${b.input_name}`, b]),
  );
  let matched = recorded.flatMap(([key, seed]) => {
    const field = byKey.get(key);
    return field ? [[field.id, seed] as const] : [];
  });
  if (!matched.length && recorded.length === 1 && fields.length === 1)
    matched = [[fields[0].id, recorded[0][1]]];
  if (!matched.length)
    return {
      error: `The draft has ${fields.length} seed field(s), and none matches the node of the ${recorded.length} recorded seed(s). Reuse settings loads the workflow that made this output.`,
    };
  // A draft value is a JavaScript number, exact only up to 2**53 - 1; a larger seed would round.
  const large = matched.find(([, seed]) => !Number.isSafeInteger(seed));
  if (large)
    return {
      error: `Seed ${large[1]} is too large for a draft field, which holds whole numbers exactly only up to ${Number.MAX_SAFE_INTEGER}. The draft is unchanged.`,
    };
  const left = recorded.length - matched.length;
  return {
    change: { values: { ...draft.values, ...Object.fromEntries(matched) }, seed_policy: 'fixed' },
    notice: `${matched.length === 1 ? `Seed ${matched[0][1]} is` : `${matched.length} seeds are`} now in the draft, and the seed policy is Fixed.${left ? ` ${left} recorded seed(s) matched no seed field.` : ''} Undo restores the previous draft.`,
  };
}

export function formatTime(value: unknown): string {
  if (value === undefined || value === null || value === '') return 'Unknown';
  const date = new Date(typeof value === 'number' ? value * 1000 : String(value));
  return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
}

/** The driver label when nothing accepts a connection at the driver origin. */
export const NOT_RUNNING = 'not running';

/** The driver state with its failure code, so an unavailable driver says why. */
export function driverStatus(driver: { state?: string; code?: string } | undefined): string {
  const state = driver?.state ?? 'unknown';
  if (state !== 'connected' && driver?.code === DRIVER_NOT_RUNNING) return NOT_RUNNING;
  return state !== 'connected' && driver?.code ? `${state} (${driver.code})` : state;
}
