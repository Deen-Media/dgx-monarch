import test from 'node:test';
import assert from 'node:assert/strict';
import { retainRecordPage, RECORD_BYTE_LIMIT, recordBytes } from './recordRetention.ts';
import { componentFunctions } from './renderHarness.js';
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

test('the project library keeps its rows while the same listing reloads', async () => {
  const loadCode = await componentFunctions('./views/WorkspaceRecords.svelte', ['load']);
  const harness = new Function(
    'client',
    'retainRecordPage',
    `'use strict';
    let sequence = 0, loadedFor = '', loading = false, loadError = '', nextOffset = null;
    let windowNotice = '', records = [], recordKind = 'presets', projectFilter = '';
    const failures = [];
    const fail = (cause) => failures.push(cause);
    ${loadCode}
    return {
      load,
      kind: (value) => { recordKind = value; },
      state: () => ({ ids: records.map((row) => row.id), loading, loadError, failures: failures.length }),
    };`,
  );
  const replies: Array<(value: unknown) => void> = [];
  const failures: Array<(cause: Error) => void> = [];
  const view = harness(
    {
      request: () =>
        new Promise((resolve, reject) => {
          replies.push(resolve);
          failures.push(reject);
        }),
    },
    retainRecordPage,
  );
  let pending = view.load();
  replies.shift()!({ records: [record('a')], next_offset: null });
  failures.shift();
  await pending;
  pending = view.load();
  assert.deepEqual(view.state().ids, ['a'], 'a revision reload keeps the current rows');
  failures.shift()!(new Error('Companion unavailable'));
  replies.shift();
  await pending;
  assert.deepEqual(view.state(), {
    ids: ['a'],
    loading: false,
    loadError: 'Companion unavailable',
    failures: 1,
  });
  view.kind('styles');
  pending = view.load();
  assert.deepEqual(view.state().ids, [], 'another kind never shows the previous kind');
  replies.shift()!({ records: [record('s')], next_offset: null });
  await pending;
  assert.deepEqual(view.state().ids, ['s']);
});
