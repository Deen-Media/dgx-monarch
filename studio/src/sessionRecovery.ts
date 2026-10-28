import type { Grid } from './gridState';
import type { OperationCapability, OperationRecord, OperationRequest } from './contracts';

const object = (value: unknown): value is Record<string, any> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const identity = (value: unknown): value is string => typeof value === 'string' && !!value.trim();
const review = (value: unknown) => object(value) && identity(value.id) && identity(value.digest);

export interface OperationRecovery {
  action: string;
  parameters: Record<string, unknown>;
  preview: OperationRecord | null;
  requestKey: string;
  pending: boolean;
  /** The identity of the Operations request this review answered, if one opened it. */
  request?: string;
}
export function validOperation(value: unknown): value is OperationRecovery {
  return (
    object(value) &&
    identity(value.action) &&
    object(value.parameters) &&
    typeof value.pending === 'boolean' &&
    identity(value.requestKey) &&
    review(value.preview) &&
    value.preview.action === value.action &&
    object(value.preview.parameters) &&
    typeof value.preview.available === 'boolean' &&
    typeof value.preview.requires_confirmation === 'boolean'
  );
}
/**
 * Whether a saved operation review comes back when Operations opens on `request`. A pending
 * outcome always does, and otherwise only a review of an operation the companion still lists:
 * when nothing new was asked, when `request` is the one that made the review, or when the palette
 * asks again for the reviewed operation. A new consent request brings its own fields instead.
 */
export function keepsReview(
  saved: OperationRecovery,
  request: OperationRequest | null,
  options: OperationCapability[],
): boolean {
  if (saved.pending) return true;
  if (!options.some((option) => option.id === saved.action)) return false;
  if (!request || (saved.request === request.id && saved.action === request.action)) return true;
  return request.action === saved.action && !request.consent;
}
export interface GridRecovery {
  preview: Grid | null;
  grid: Grid | null;
  requestKey: string;
  retryKey: string;
  retryIndices: number[];
  dispatchPending: boolean;
  retryPending: boolean;
}
export function validGrid(value: unknown): value is GridRecovery {
  if (
    !object(value) ||
    typeof value.requestKey !== 'string' ||
    typeof value.retryKey !== 'string' ||
    typeof value.dispatchPending !== 'boolean' ||
    typeof value.retryPending !== 'boolean' ||
    !Array.isArray(value.retryIndices) ||
    !value.retryIndices.every((n: unknown) => Number.isSafeInteger(n) && Number(n) >= 0)
  )
    return false;
  if (value.preview !== null && (!review(value.preview) || !Array.isArray(value.preview.cells)))
    return false;
  if (
    value.grid !== null &&
    (!object(value.grid) || !identity(value.grid.id) || !Array.isArray(value.grid.cells))
  )
    return false;
  return (
    (!value.dispatchPending || (value.preview !== null && identity(value.requestKey))) &&
    (!value.retryPending ||
      (value.grid !== null && identity(value.retryKey) && value.retryIndices.length > 0))
  );
}
export interface LibraryPlan {
  id: string;
  digest: string;
  ready: boolean;
  changes: Array<{ action: string }>;
  conflicts: Array<{
    key: string;
    kind: string;
    id: string;
    revision: number;
    current: unknown;
    incoming: unknown;
  }>;
  repair_list: unknown[];
}
export interface LibraryRecovery {
  plan: LibraryPlan;
  key: string;
  name: string;
}
export function validLibrary(value: unknown): value is LibraryRecovery {
  if (
    !object(value) ||
    !identity(value.key) ||
    typeof value.name !== 'string' ||
    !review(value.plan)
  )
    return false;
  const plan = value.plan;
  return (
    typeof plan.ready === 'boolean' &&
    Array.isArray(plan.changes) &&
    plan.changes.every((row: unknown) => object(row) && typeof row.action === 'string') &&
    Array.isArray(plan.repair_list) &&
    Array.isArray(plan.conflicts) &&
    plan.conflicts.every(
      (row: unknown) =>
        object(row) &&
        identity(row.key) &&
        identity(row.kind) &&
        identity(row.id) &&
        Number.isSafeInteger(row.revision),
    )
  );
}

/** Unreadable mutation receipts must survive until the user checks durable backend state. */
export function restoreSession<T>(key: string, validate: (value: unknown) => value is T) {
  try {
    const raw = sessionStorage.getItem(key);
    if (raw === null) return { value: null, blocked: false };
    const value: unknown = JSON.parse(raw);
    if (!validate(value)) throw new Error('Invalid stored request identity.');
    return { value, blocked: false };
  } catch {
    return { value: null, blocked: true };
  }
}

/** Authentication remains available when optional token persistence is disabled. */
export function savedToken(key: string): string {
  try {
    return sessionStorage.getItem(key) ?? '';
  } catch {
    return '';
  }
}
