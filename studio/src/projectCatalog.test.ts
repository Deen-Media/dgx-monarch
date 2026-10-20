import test from 'node:test';
import assert from 'node:assert/strict';
import { projectCatalog } from './projectCatalog.ts';
import type { StudioClient } from './api';

test('project selectors share complete paged results across repeated snapshot reads', async () => {
  const paths: string[] = [];
  const client = {
    request: async (path: string) => {
      paths.push(path);
      return path.includes('offset=100')
        ? {
            records: [{ id: 'older-project', revision: 1, data: { name: 'Older project' } }],
            next_offset: null,
          }
        : {
            records: [{ id: 'recent-project', revision: 2, data: { name: 'Recent project' } }],
            next_offset: 100,
          };
    },
  } as unknown as StudioClient;
  const header = projectCatalog(client, 'recent-project:2');
  const gallery = projectCatalog(client, 'recent-project:2');
  assert.equal(header, gallery);
  assert.deepEqual(
    (await header).map((row) => row.id),
    ['recent-project', 'older-project'],
  );
  await projectCatalog(client, 'recent-project:2');
  assert.equal(paths.length, 2);
  await projectCatalog(client, 'recent-project:3');
  assert.equal(paths.length, 4);
});
test('project catalog failure may retry and a different client never inherits old project labels', async () => {
  let count = 0;
  const client = {
    request: async () => {
      if (++count === 1) throw new Error('offline');
      return { records: [{ id: 'ours', revision: 1, data: { name: 'Ours' } }], next_offset: null };
    },
  } as unknown as StudioClient;
  await assert.rejects(projectCatalog(client, ''), /offline/);
  assert.equal((await projectCatalog(client, ''))[0].id, 'ours');
  const another = {
    request: async () => ({ records: [], next_offset: null }),
  } as unknown as StudioClient;
  assert.deepEqual(await projectCatalog(another, ''), []);
});
