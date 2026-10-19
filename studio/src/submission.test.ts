import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { compileModule } from 'svelte/compiler';
import { ApiError } from './api.ts';
import { submissionRefused } from './submissionRefusal.ts';

async function submissionClass() {
  const source = await readFile(new URL('./submission.svelte.ts', import.meta.url), 'utf8');
  const plain = ts.transpile(source, {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
  });
  const compiled = compileModule(plain, { filename: 'submission.svelte.js', generate: 'client' }).js
    .code;
  const resolved = compiled.replace(/from ['"]([^'"]+)['"]/g, (_match, specifier) => {
    const url = specifier.startsWith('.')
      ? new URL(`${specifier}.ts`, import.meta.url).href
      : import.meta.resolve(specifier);
    return `from ${JSON.stringify(url)}`;
  });
  return (await import(`data:text/javascript;base64,${Buffer.from(resolved).toString('base64')}`))
    .Submission;
}
const Submission = await submissionClass();
function setup() {
  const store = new Map<string, string>();
  globalThis.sessionStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  } as Storage;
  const connection = {
    error: '',
    refresh: async () => {},
    client: {
      request: async (_path: string, _body?: unknown): Promise<unknown> => ({ id: 'job' }),
    },
  };
  const submission = new Submission(connection);
  submission.review({ id: 'preview' });
  submission.confirmed = true;
  return { submission, connection, store };
}
for (const code of ['preview_expired', 'schema_changed']) {
  test(`a definitive ${code} refusal allows a fresh reviewed submission`, async () => {
    const { submission, connection, store } = setup();
    connection.client.request = async () => {
      throw new ApiError(409, code, code);
    };
    await assert.rejects(submission.submit(), { code });
    assert.equal(submission.pending, null);
    assert.equal(submission.preview, null);
    assert.equal(store.has('dgxm.studio.pending'), false);
    submission.review({ id: 'fresh' });
    assert.equal(submission.preview.id, 'fresh');
  });
}
test('transport uncertainty retains the exact identity until an expired retry is definitively refused', async () => {
  const { submission, connection, store } = setup();
  let sent: unknown;
  connection.client.request = async (_path, body) => {
    sent = body;
    throw new Error('timeout');
  };
  await assert.rejects(submission.submit(), /timeout/);
  const retained = JSON.parse(store.get('dgxm.studio.pending')!);
  assert.deepEqual({ preview_id: retained.preview_id, idempotency_key: retained.key }, sent);
  assert.throws(() => submission.review({ id: 'fresh' }), /Reconcile/);
  connection.client.request = async () => {
    throw new ApiError(404, 'No accepted request', 'not_found');
  };
  assert.equal(await submission.reconcile(), false);
  assert.deepEqual(submission.pending, retained);
  connection.client.request = async (_path, body) => {
    assert.deepEqual(body, sent);
    throw new ApiError(409, 'Expired', 'preview_expired');
  };
  await assert.rejects(submission.retryRetained(), /Expired/);
  assert.equal(submission.pending, null);
  assert.equal(store.has('dgxm.studio.pending'), false);
});
for (const error of [
  new ApiError(409, 'Conflict', 'idempotency_conflict'),
  new ApiError(403, 'Forbidden'),
  new Error('timeout'),
]) {
  test(`${error.message} preserves a pending retry identity`, async () => {
    const { submission, connection } = setup();
    connection.client.request = async () => {
      throw error;
    };
    await assert.rejects(submission.submit());
    const retained = { ...submission.pending };
    submission.confirmedMissing = true;
    await assert.rejects(submission.retryRetained());
    assert.deepEqual(submission.pending, retained);
  });
}
test('storage failure does not send or leave an unsent pending identity', async () => {
  const { submission, connection } = setup();
  let sent = false;
  connection.client.request = async () => {
    sent = true;
  };
  sessionStorage.setItem = () => {
    throw new Error('full');
  };
  await assert.rejects(submission.submit(), /full/);
  assert.equal(sent, false);
  assert.equal(submission.pending, null);
});
test('successful acceptance clears the persisted identity', async () => {
  const { submission, store } = setup();
  assert.deepEqual(await submission.submit(), { id: 'job' });
  assert.equal(submission.pending, null);
  assert.equal(store.has('dgxm.studio.pending'), false);
});
test('grid refusal classification preserves uncertainty and separates submit from retry', () => {
  assert.equal(submissionRefused(new ApiError(409, 'expired', 'grid_preview'), 'grid'), true);
  assert.equal(
    submissionRefused(new ApiError(409, 'child expired', 'preview_expired'), 'grid'),
    true,
  );
  assert.equal(
    submissionRefused(new ApiError(409, 'changed state', 'grid_retry'), 'grid-retry'),
    true,
  );
  assert.equal(
    submissionRefused(new ApiError(409, 'conflict', 'idempotency_conflict'), 'grid'),
    false,
  );
  assert.equal(submissionRefused(new ApiError(500, 'failed', 'grid_preview'), 'grid'), false);
  assert.equal(submissionRefused(new Error('timeout'), 'grid'), false);
  assert.equal(submissionRefused(new ApiError(409, 'retry', 'grid_retry'), 'grid'), false);
});
test('unreadable pending identity blocks fresh work until explicitly cleared', async () => {
  const { submission, connection, store } = setup();
  store.set('dgxm.studio.pending', '{');
  submission.recoveryError = 'Check Queue';
  let sent = false;
  connection.client.request = async () => {
    sent = true;
  };
  assert.throws(() => submission.review({ id: 'fresh' }), /Check Queue/);
  await assert.rejects(submission.prepare({}), /Check Queue/);
  await assert.rejects(submission.submit(), /Check Queue/);
  assert.equal(sent, false);
  assert.equal(store.get('dgxm.studio.pending'), '{');
  submission.clearUnreadableRequest();
  assert.equal(store.has('dgxm.studio.pending'), false);
  submission.review({ id: 'fresh' });
});
for (const response of [{}, null, { id: '' }]) {
  test(`identity-free successful response cannot settle a request: ${JSON.stringify(response)}`, async () => {
    const { submission, connection, store } = setup();
    connection.client.request = async () => response;
    await assert.rejects(submission.submit(), /record identity/);
    const retained = { ...submission.pending };
    assert.ok(store.has('dgxm.studio.pending'));
    await assert.rejects(submission.reconcile(), /record identity/);
    submission.confirmedMissing = true;
    await assert.rejects(submission.retryRetained(), /record identity/);
    assert.deepEqual(submission.pending, retained);
  });
}
