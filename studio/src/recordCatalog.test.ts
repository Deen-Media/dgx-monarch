import test from 'node:test';
import assert from 'node:assert/strict';
import type { StudioClient } from './api.ts';
import { findRecord, loadRecordCatalog } from './recordCatalog.ts';
import { recordProjects, persistRecord } from './recordPersistence.ts';

test('catalog loads beyond the snapshot page and deduplicates moving records', async () => {
  const paths: string[] = [];
  const client = {
    request: async (path: string) => {
      paths.push(path);
      return paths.length === 1
        ? { records: [{ id: 'a', revision: 1, data: {} }], next_offset: 200 }
        : {
            records: [
              { id: 'a', revision: 2, data: {} },
              { id: 'b', revision: 1, data: {} },
            ],
            next_offset: null,
          };
    },
  } as unknown as StudioClient;
  assert.deepEqual(
    (await loadRecordCatalog(client, 'projects')).map((row) => [row.id, row.revision]),
    [
      ['a', 2],
      ['b', 1],
    ],
  );
  assert.deepEqual(paths, [
    '/records/projects?limit=200&offset=0',
    '/records/projects?limit=200&offset=200',
  ]);
});

test('explicit shared membership overrides legacy embedded project identity', () => {
  assert.deepEqual(recordProjects({ project_ids: [], project_id: 'old' }), []);
  assert.deepEqual(recordProjects({ draft: { project_id: 'legacy' } } as never), ['legacy']);
});

test('saving preserves reviewed identity and revision in a single PUT', async () => {
  const calls: unknown[][] = [];
  const result = { id: 'stable', revision: 3, data: { name: 'kept', project_ids: ['p'] } };
  const client = {
    request: async (...args: unknown[]) => {
      calls.push(args);
      return result;
    },
  } as unknown as StudioClient;
  let refreshed = false;
  assert.equal(
    await persistRecord(
      client,
      async () => {
        refreshed = true;
      },
      () => {},
      'presets',
      result.data,
      'stable',
      2,
    ),
    result,
  );
  assert.deepEqual(calls, [['/records/presets/stable', { revision: 2, data: result.data }, 'PUT']]);
  assert.equal(refreshed, true);
});

test('byte-bounded catalog pages can require more than ten requests', async () => {
  let calls = 0;
  const client = {
    request: async () => {
      const id = String(calls++);
      return { records: [{ id, revision: 1, data: {} }], next_offset: calls < 12 ? calls : null };
    },
  } as unknown as StudioClient;
  const records = await loadRecordCatalog(client, 'styles');
  assert.equal(calls, 12);
  assert.equal(records.length, 12);
  assert.equal(records.at(-1)?.id, '11');
});

test('catalog refuses a stalled cursor instead of polling forever', async () => {
  const client = {
    request: async () => ({ records: [{ id: 'a', revision: 1, data: {} }], next_offset: 0 }),
  } as unknown as StudioClient;
  await assert.rejects(loadRecordCatalog(client, 'styles'), /non-advancing library page/);
});

test('catalog refuses an oversized aggregate rather than return a partial selector', async () => {
  let calls = 0;
  const text = 'a'.repeat(5 * 1024 * 1024);
  const client = {
    request: async () => ({
      records: [{ id: String(++calls), revision: 1, data: { notes: text } }],
      next_offset: calls === 1 ? 1 : null,
    }),
  } as unknown as StudioClient;
  await assert.rejects(loadRecordCatalog(client, 'styles'), /8 MiB.*No partial catalog/);
  assert.equal(calls, 2);
});

test('read-back pages by identity, stops at the match and refuses a stalled cursor', async () => {
  const paths: string[] = [];
  const client = {
    request: async (path: string) => {
      paths.push(path);
      return paths.length === 1
        ? { records: [{ id: 'other', revision: 1, data: {} }], next_offset: 1 }
        : { records: [{ id: 'kept', revision: 1, data: { name: 'x' } }], next_offset: 2 };
    },
  } as unknown as StudioClient;
  assert.equal((await findRecord(client, 'styles', 'kept'))?.data.name, 'x');
  assert.equal(paths.length, 2);
  const empty = {
    request: async () => ({ records: [], next_offset: null }),
  } as unknown as StudioClient;
  assert.equal(await findRecord(empty, 'styles', 'kept'), null);
  const stalled = {
    request: async () => ({ records: [{ id: 'a', revision: 1, data: {} }], next_offset: 0 }),
  } as unknown as StudioClient;
  await assert.rejects(findRecord(stalled, 'styles', 'kept'), /non-advancing library page/);
});
