import { requireRecordIdentity } from './api.ts';

/** A partial import response cannot establish completion of the retained request. */
export function requireLibraryReceipt(value: unknown): void {
  requireRecordIdentity(value);
  const result = value as { imported?: unknown; repair_list?: unknown };
  if (
    !Number.isSafeInteger(result.imported) ||
    Number(result.imported) < 0 ||
    !Array.isArray(result.repair_list)
  )
    throw new Error('Library import receipt is incomplete. Reconcile the retained request.');
}
