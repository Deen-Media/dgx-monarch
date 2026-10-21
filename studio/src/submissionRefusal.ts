import { ApiError } from './api.ts';

/** The definite refusals each request kind can receive; nothing was accepted under its key. */
const DEFINITE_REFUSALS = {
  library: [
    'library_confirmation',
    'library_preview',
    'revision_conflict',
    'record_limit',
    'library_limit',
  ],
  // A changed driver schema or a choice it no longer offers refuses a retry before it is reserved.
  'grid-retry': ['grid_retry', 'schema_changed', 'choice_unavailable', 'preview_expired'],
  grid: ['grid_preview', 'preview_expired', 'schema_changed', 'choice_unavailable'],
  job: ['preview_expired', 'schema_changed', 'choice_unavailable'],
  // Job actions: the companion releases or never records the key for these.
  cancel: ['confirmation', 'cancellation_scope'],
  retry: ['retry_scope', 'retry_provenance', 'schema_changed', 'choice_unavailable'],
} as const;

export type RefusalKind = keyof typeof DEFINITE_REFUSALS;

/** These backend refusals are emitted only after lookup of an accepted request. */
export function submissionRefused(error: unknown, kind: RefusalKind): boolean {
  if (!(error instanceof ApiError) || error.status < 400 || error.status >= 500) return false;
  return (DEFINITE_REFUSALS[kind] as readonly string[]).includes(error.code ?? '');
}
