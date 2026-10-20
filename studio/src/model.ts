export interface Binding {
  id: string;
  aliases?: string[];
  label?: string;
  name?: string;
  input_name?: string;
  node_id?: string;
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
  };
  required?: boolean;
  validation?: {
    min?: number;
    max?: number;
    step?: number;
    options?: unknown[];
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
  seed_policy: 'fixed',
  parent_artifact_ids: [],
});
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
  if (b.required && (value === undefined || value === '')) return 'A value is required.';
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
export function gridCombinations(
  axes: { id: string; values: unknown[] }[],
  max = 1024,
): Record<string, unknown>[] {
  if (!axes.length) return [];
  if (new Set(axes.map((a) => a.id)).size !== axes.length)
    throw new Error('Each axis must use a different field.');
  let result: Record<string, unknown>[] = [{}];
  for (const axis of axes) {
    if (!axis.id || !axis.values.length)
      throw new Error('Each axis needs a field and at least one value.');
    if (result.length * axis.values.length > max)
      throw new Error(`Grid exceeds the ${max}-cell limit.`);
    result = result.flatMap((row) => axis.values.map((value) => ({ ...row, [axis.id]: value })));
  }
  return result;
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
  };
}

export function formatTime(value: unknown): string {
  if (value === undefined || value === null || value === '') return 'Unknown';
  const date = new Date(typeof value === 'number' ? value * 1000 : String(value));
  return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
}
