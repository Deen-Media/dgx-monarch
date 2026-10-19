import test from 'node:test';
import assert from 'node:assert/strict';
import { ApiError } from './api.ts';
import { requireLibraryReceipt } from './libraryImport.ts';
import { submissionRefused } from './submissionRefusal.ts';

test('a lost import acknowledgement followed by 401 never releases the retained copy identity', () => {
  const request = { idempotency_key: 'copy-request', preview_id: 'copy-plan' };
  let retained: typeof request | null = request;
  for (const failure of [new Error('response lost'), new ApiError(401, 'Unauthorized')]) {
    if (submissionRefused(failure, 'library')) retained = null;
    assert.deepEqual(retained, request);
  }
});
for (const failure of [
  new ApiError(409, 'Conflict', 'idempotency_conflict'),
  new ApiError(403, 'Forbidden'),
  new ApiError(409, 'Unknown conflict'),
  new ApiError(500, 'Failed'),
]) {
  test(`library ${failure.message} does not establish that import was unaccepted`, () => {
    assert.equal(submissionRefused(failure, 'library'), false);
  });
}
for (const code of [
  'library_confirmation',
  'library_preview',
  'revision_conflict',
  'record_limit',
  'library_limit',
]) {
  test(`${code} allows import re-review after definitive refusal`, () => {
    assert.equal(submissionRefused(new ApiError(409, 'Not imported', code), 'library'), true);
  });
}
for (const value of [
  {},
  null,
  { id: 'receipt' },
  { id: 'receipt', imported: 2 },
  { imported: 2, repair_list: [] },
  { id: 'receipt', imported: -1, repair_list: [] },
]) {
  test(`partial import acknowledgement cannot settle pending state: ${JSON.stringify(value)}`, () => {
    assert.throws(() => requireLibraryReceipt(value), /retained request/);
  });
}
test('a complete import receipt can settle the retained request', () => {
  requireLibraryReceipt({ id: 'receipt', imported: 2, repair_list: [] });
});
