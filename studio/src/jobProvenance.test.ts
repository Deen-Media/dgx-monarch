import test from 'node:test';
import assert from 'node:assert/strict';
import type { StudioClient } from './api.ts';
import type { Job } from './contracts.ts';
import { fullJob } from './jobProvenance.ts';

const provenance = {
  id: 'job/a',
  state: 'succeeded',
  created_at: 1,
  submission: { workflow: { node: { class_type: 'SaveImage' } }, prompt: { actual: true } },
} as Job;

test('provenance actions lazily fetch full data for compact snapshot/list rows', async () => {
  const paths: string[] = [];
  const client = {
    request: async (path: string) => {
      paths.push(path);
      return provenance;
    },
  } as unknown as StudioClient;
  const summary = {
    ...provenance,
    summary: true,
    submission: { requested: { operation: 'image' } },
  } as Job;
  assert.equal(await fullJob(client, summary.id, summary), provenance);
  assert.deepEqual(paths, ['/jobs/job%2Fa']);
  assert.equal(summary.submission.workflow, undefined);
});

test('known full immutable provenance needs no additional request', async () => {
  const client = {
    request: () => {
      throw new Error('Unexpected request');
    },
  } as unknown as StudioClient;
  assert.equal(await fullJob(client, provenance.id, provenance), provenance);
});

test('full external API prompt provenance is usable without a UI workflow', async () => {
  const external = {
    ...provenance,
    submission: {
      prompt: {
        '1': { class_type: 'Render', inputs: { seed: 42 } },
      },
    },
  };
  const client = {
    request: () => {
      throw new Error('Unexpected request');
    },
  } as unknown as StudioClient;
  assert.equal(await fullJob(client, external.id, external), external);
});

test('legacy list rows without workflows also request full provenance', async () => {
  let requested = false;
  const client = {
    request: async () => {
      requested = true;
      return provenance;
    },
  } as unknown as StudioClient;
  const legacy = { ...provenance, submission: {} };
  await fullJob(client, legacy.id, legacy);
  assert.equal(requested, true);
});

test('provenance actions refuse mismatched IDs or another summary from detail endpoint', async () => {
  for (const result of [
    { ...provenance, id: 'different' },
    { ...provenance, summary: true },
  ]) {
    const client = { request: async () => result } as unknown as StudioClient;
    await assert.rejects(fullJob(client, provenance.id), /requested full job provenance/);
  }
});
