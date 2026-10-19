import test from 'node:test';
import assert from 'node:assert/strict';
import { retainRecordPage, RECORD_BYTE_LIMIT, recordBytes } from './recordRetention.ts';
const record = (id: string, notes = '') => ({ id, revision: 1, data: { notes } });

test('small pages append while an oversized aggregate starts a complete new window', () => {
  assert.deepEqual(
    retainRecordPage([record('a')], [record('b')]).records.map((row) => row.id),
    ['a', 'b'],
  );
  const notes = 'a'.repeat(5 * 1024 * 1024);
  const first = [record('a', notes)],
    next = [record('b', notes)];
  const retained = retainRecordPage(first, next);
  assert.equal(retained.replaced, true);
  assert.equal(retained.records, next);
  assert.ok(
    retained.records.reduce((bytes, row) => bytes + recordBytes(row), 2) <= RECORD_BYTE_LIMIT,
  );
});

test('window limit counts UTF-8 bytes and rejects an oversized single page', () => {
  assert.throws(() => retainRecordPage([], [record('a', 'é'.repeat(5 * 1024 * 1024))]), /8 MiB/);
});

test('row limit starts a new window without dropping incoming records', () => {
  const first = Array.from({ length: 2000 }, (_, index) => record(String(index)));
  const next = [record('next')];
  assert.deepEqual(retainRecordPage(first, next), { records: next, replaced: true });
});
