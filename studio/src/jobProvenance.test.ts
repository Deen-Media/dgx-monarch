import test from 'node:test';
import assert from 'node:assert/strict';
import type { StudioClient } from './api.ts';
import type { Job } from './contracts.ts';
import { comparableSubmission, exactJob, fullJob, RETRY_RUNTIME_FIELD } from './jobProvenance.ts';
import { readExactJSON } from './exactJson.ts';
import { parameterDifferences } from './gallery.ts';

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

test('the exact read parses the full record with every digit and refuses a summary', async () => {
  const asked: unknown[] = [];
  const text = (id: string, extra = '') =>
    `{"id":"${id}"${extra},"submission":{"prompt":{"8":{"inputs":{"seed":18446744073709551615}}}}}`;
  const client = (body: string) =>
    ({
      request: async (path: string, payload: unknown, method: string, parse: typeof JSON.parse) => {
        asked.push([path, payload, method, parse === readExactJSON]);
        return parse(body);
      },
    }) as unknown as StudioClient;
  const job = await exactJob(client(text('job/a')), 'job/a');
  assert.equal(job.submission.prompt['8'].inputs.seed, 18446744073709551615n);
  assert.deepEqual(asked, [['/jobs/job%2Fa', undefined, 'GET', true]]);
  for (const body of [text('other'), text('job/a', ',"summary":true')])
    await assert.rejects(exactJob(client(body), 'job/a'), /requested full job provenance/);
});

test('comparison reads the runtime and versions a retry recorded for itself', () => {
  const recorded = {
    prompt: { '1': { inputs: { seed: 1 } } },
    versions: { dgx_monarch: '1.2.0', driver: '0.3.60' },
    runtime_observation: { reported: { comfyui_version: '0.3.60' } },
    schema_digest: 'old-schema',
    schema_identity: 'old-identity',
  };
  const original = { id: 'original', state: 'failed', created_at: 1, submission: recorded } as Job;
  const retry = {
    id: 'retry',
    state: 'succeeded',
    created_at: 2,
    retry_of: 'original',
    submission: structuredClone(recorded),
    [RETRY_RUNTIME_FIELD]: {
      observed_at: 99,
      versions: { dgx_monarch: '1.2.0', driver: '0.3.61' },
      runtime_observation: { reported: { comfyui_version: '0.3.61' } },
      schema_digest: 'new-schema',
      schema_identity: 'new-identity',
    },
  } as Job;
  const fresh = {
    ...retry,
    id: 'fresh',
    retry_of: undefined,
    submission: structuredClone(recorded),
  };
  fresh.submission.versions = { dgx_monarch: '1.2.0', driver: '0.3.61' };
  fresh.submission.runtime_observation = { reported: { comfyui_version: '0.3.61' } };
  fresh.submission.schema_digest = 'new-schema';
  fresh.submission.schema_identity = 'new-identity';
  delete fresh[RETRY_RUNTIME_FIELD];
  assert.deepEqual(
    parameterDifferences(comparableSubmission(original), comparableSubmission(retry)).map(
      (row) => row.key,
    ),
    [
      'runtime_observation.reported.comfyui_version',
      'schema_digest',
      'schema_identity',
      'versions.driver',
    ],
  );
  assert.deepEqual(
    parameterDifferences(comparableSubmission(retry), comparableSubmission(fresh)),
    [],
  );
  assert.equal(comparableSubmission(original), original.submission);
  assert.equal(retry.submission.schema_digest, 'old-schema', 'the immutable request is untouched');
});
