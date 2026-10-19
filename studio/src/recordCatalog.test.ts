import test from 'node:test';
import assert from 'node:assert/strict';
import type { StudioClient } from './api.ts';
import { loadRecordCatalog } from './recordCatalog.ts';
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
