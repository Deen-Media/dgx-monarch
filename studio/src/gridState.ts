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
    valuesA: '[1,2]',
    valuesB: '[1]',
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
}
/** The stored grid session: request identities plus the reviewed definition, no cell payloads. */
export function gridSessionText(session: GridSession): string {
  return JSON.stringify({
    ...session,
    preview: gridIdentity(session.preview),
    grid: gridIdentity(session.grid),
    definition: session.preview ? session.definition : null,
  });
}
