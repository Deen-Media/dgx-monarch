export interface Cell {
  index: number;
  coordinates: Record<string, unknown>;
  excluded?: boolean;
  errors?: string[];
  preview_id?: string;
  job_id?: string;
  state?: string;
  attempts?: string[];
  artifact_ids?: string[];
}
export interface Grid {
  id: string;
  state?: string;
  digest?: string;
  cells: Cell[];
  ready?: boolean;
  total?: number;
  job_count?: number;
  expires_at?: number;
}

/** Persist enough to reconcile a request, never repeated workflow/cell payloads. */
export function gridIdentity(grid: Grid | null): Grid | null {
  if (!grid) return null;
  const { id, state, digest, ready, total, job_count, expires_at } = grid;
  return { id, state, digest, ready, total, job_count, expires_at, cells: [] };
}

/** The editable grid definition fields that GridDefinition binds. */
export interface GridFields {
  axisA: string;
  axisB: string;
  valuesA: string;
  valuesB: string;
  seedPolicy: string;
  seedBinding: string;
  baseSeed: number;
  excluded: string;
}
export function defaultGridFields(): GridFields {
  return {
    axisA: '',
    axisB: '',
    valuesA: '',
    valuesB: '',
    seedPolicy: 'fixed',
    seedBinding: '',
    baseSeed: 0,
    excluded: '',
  };
}
export const GRID_FIELD_KEYS = Object.keys(defaultGridFields()) as (keyof GridFields)[];
/** Stored definition text stays small; axis values are typed JSON lists, not payloads. */
const FIELD_TEXT_LIMIT = 64 * 1024;

/** The definition a frozen review belongs to. Any edited field changes it. */
export function gridDefinitionKey(draft: unknown, fields: GridFields): string {
  return JSON.stringify({
    draft,
    ...Object.fromEntries(GRID_FIELD_KEYS.map((key) => [key, fields[key]])),
  });
}

/** Read stored definition fields back, or null when any field is missing or malformed. */
export function restoredGridFields(value: unknown): GridFields | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const stored = value as Record<string, unknown>;
  for (const key of GRID_FIELD_KEYS) {
    const item = stored[key];
    if (key === 'baseSeed' ? !Number.isSafeInteger(item) : typeof item !== 'string') return null;
    if (typeof item === 'string' && item.length > FIELD_TEXT_LIMIT) return null;
  }
  if (!['fixed', 'increment'].includes(String(stored.seedPolicy))) return null;
  return Object.fromEntries(
    GRID_FIELD_KEYS.map((key) => [key, stored[key]]),
  ) as unknown as GridFields;
}

/** JSON with object keys sorted, so equal drafts give equal text whatever their key order. */
function sortedText(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) =>
    item && typeof item === 'object' && !Array.isArray(item)
      ? Object.fromEntries(
          Object.keys(item)
            .sort()
            .map((key) => [key, (item as Record<string, unknown>)[key]]),
        )
      : item,
  );
}

/**
 * A short identity of the draft and fields a review was made from. The session stores it instead
 * of the draft; it is change detection, not a security check (cyrb53, public domain, by bryc).
 */
export function reviewIdentity(draft: unknown, fields: GridFields): string {
  const text = sortedText({
    draft,
    fields: Object.fromEntries(GRID_FIELD_KEYS.map((key) => [key, fields[key]])),
  });
  let first = 0xdeadbeef,
    second = 0x41c6ce57;
  for (let index = 0; index < text.length; index++) {
    const code = text.charCodeAt(index);
    first = Math.imul(first ^ code, 2654435761);
    second = Math.imul(second ^ code, 1597334677);
  }
  first = Math.imul(first ^ (first >>> 16), 2246822507);
  first ^= Math.imul(second ^ (second >>> 13), 3266489909);
  second = Math.imul(second ^ (second >>> 16), 2246822507);
  second ^= Math.imul(first ^ (first >>> 13), 3266489909);
  return `${text.length}:${(4294967296 * (2097151 & second) + (first >>> 0)).toString(16)}`;
}

export const STALE_RESTORE =
  'This saved preview does not match the current draft and grid fields, or predates that check, so Studio dropped it. Validate every cell again before submission.';
export const EARLIER_PENDING =
  'This pending dispatch was validated against an earlier draft. Reconciling it sends that earlier review, not the current draft.';

/**
 * What a restored review may do. `fresh`: it matches the current draft and fields. `pending`: it
 * does not, but its dispatch outcome is unknown, so it stays for reconciliation. `stale`: drop it.
 */
export function restoredReview(
  saved: { reviewed?: unknown; dispatchPending: boolean },
  current: string,
): 'fresh' | 'pending' | 'stale' {
  if (typeof saved.reviewed === 'string' && saved.reviewed === current) return 'fresh';
  return saved.dispatchPending ? 'pending' : 'stale';
}

export interface GridSession {
  preview: Grid | null;
  grid: Grid | null;
  requestKey: string;
  retryKey: string;
  retryIndices: number[];
  dispatchPending: boolean;
  retryPending: boolean;
  /** The fields that produced the retained preview, so a reload shows what Dispatch sends. */
  definition: GridFields | null;
  /** The reviewIdentity of the draft and fields the retained preview was made from. */
  reviewed: string | null;
}
/** The stored grid session: request identities plus the reviewed definition, no cell payloads. */
export function gridSessionText(session: GridSession): string {
  return JSON.stringify({
    ...session,
    preview: gridIdentity(session.preview),
    grid: gridIdentity(session.grid),
    definition: session.preview ? session.definition : null,
    reviewed: session.preview ? session.reviewed : null,
  });
}

/** A sample value list for an axis. A style-stack value is a list of saved style IDs. */
export function axisExample(axis: string): string {
  return axis === 'styles' ? '[["ink"], ["ink", "night"]]' : '[20, 30, 40]';
}

/** Why Validate every cell cannot run yet, or '' when it can. */
export function validateBlocker(
  hasWorkflow: boolean,
  fields: Pick<GridFields, 'axisA' | 'axisB' | 'valuesA' | 'valuesB'>,
): string {
  if (!hasWorkflow) return 'Load a workflow in Create first. A grid varies the Create draft.';
  if (!fields.axisA) return 'Choose an input for Axis A.';
  const axes = [
    ['A', fields.axisA, fields.valuesA],
    ['B', fields.axisB, fields.valuesB],
  ];
  for (const [label, axis, text] of axes) {
    if (!axis) continue;
    const example = axisExample(axis);
    if (!text.trim()) return `List the values for Axis ${label}, for example ${example}.`;
    let values: unknown;
    try {
      values = JSON.parse(text);
    } catch {
      return `Axis ${label} values are not valid JSON. Write a list such as ${example}.`;
    }
    if (!Array.isArray(values))
      return `Axis ${label} values must be a list in square brackets, such as ${example}.`;
    if (!values.length) return `List at least one value for Axis ${label}.`;
  }
  return '';
}
