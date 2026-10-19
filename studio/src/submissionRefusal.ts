import { ApiError } from './api.ts';

/** These backend refusals are emitted only after lookup of an accepted request. */
export function submissionRefused(
  error: unknown,
  kind: 'job' | 'grid' | 'grid-retry' | 'library',
): boolean {
  if (!(error instanceof ApiError) || error.status < 400 || error.status >= 500) return false;
  const codes =
    kind === 'library'
      ? [
          'library_confirmation',
          'library_preview',
          'revision_conflict',
          'record_limit',
          'library_limit',
        ]
      : kind === 'grid-retry'
        ? ['grid_retry']
        : kind === 'grid'
          ? ['grid_preview', 'preview_expired', 'schema_changed']
          : ['preview_expired', 'schema_changed'];
  return codes.includes(error.code ?? '');
}
