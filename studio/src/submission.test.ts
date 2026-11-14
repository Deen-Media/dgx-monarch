import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { compileModule } from 'svelte/compiler';
import { ApiError } from './api.ts';
import { submissionRefused } from './submissionRefusal.ts';

async function submissionModule() {
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
  return await import(`data:text/javascript;base64,${Buffer.from(resolved).toString('base64')}`);
}
const { Submission, SUBMITTED, CONFIRMATION_MS } = await submissionModule();
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
  return { submission, connection, store };
}
test('clearing unreadable request remains visibly blocked when storage removal fails', () => {
  const { submission, store } = setup();
  store.set('dgxm.studio.pending', '{');
  submission.recoveryError = 'Unreadable request';
  sessionStorage.removeItem = () => {
    throw new Error('Storage denied');
  };
  assert.doesNotThrow(() => submission.clearUnreadableRequest());
  assert.match(submission.recoveryError, /remain blocked/);
  assert.equal(store.get('dgxm.studio.pending'), '{');
  assert.throws(() => submission.review({ id: 'next' }), /remain blocked/);
  sessionStorage.removeItem = (key) => {
    store.delete(key);
  };
  submission.clearUnreadableRequest();
  assert.equal(submission.recoveryError, '');
  assert.equal(store.has('dgxm.studio.pending'), false);
});
for (const code of ['preview_expired', 'schema_changed', 'choice_unavailable']) {
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
test('the identity this page sends is marked as created here, and a restored one is not', async () => {
  const { submission, connection, store } = setup();
  connection.client.request = async () => {
    throw new Error('timeout');
  };
  await assert.rejects(submission.submit(), /timeout/);
  const key = submission.pending!.key;
  assert.equal(submission.createdKey, key, 'a lost reply keeps the identity marked');
  // The marker is never stored, so a reload restores the identity without it.
  const stored = JSON.parse(store.get('dgxm.studio.pending')!);
  assert.deepEqual(Object.keys(stored).sort(), ['key', 'preview_id']);
  const restored = new Submission(connection);
  restored.pending = stored;
  assert.equal(restored.createdKey, '');
  restored.confirmedMissing = true;
  await assert.rejects(restored.retryRetained(), /timeout/);
  assert.equal(restored.createdKey, key, 'a retry of the retained request is sent from this page');
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
// Each code the companion raises after its key lookup, so nothing was accepted under the key.
const definite: Record<string, string[]> = {
  job: ['preview_expired', 'schema_changed', 'choice_unavailable'],
  grid: ['grid_preview', 'preview_expired', 'schema_changed', 'choice_unavailable'],
  'grid-retry': ['grid_retry', 'schema_changed', 'choice_unavailable', 'preview_expired'],
  cancel: ['confirmation', 'cancellation_scope'],
  retry: ['retry_scope', 'retry_provenance', 'schema_changed', 'choice_unavailable'],
};
for (const [kind, codes] of Object.entries(definite)) {
  for (const code of codes) {
    test(`a ${kind} request refused with ${code} is definite, but not as a server fault`, () => {
      const refusal = kind as Parameters<typeof submissionRefused>[1];
      assert.equal(submissionRefused(new ApiError(409, code, code), refusal), true);
      assert.equal(submissionRefused(new ApiError(500, code, code), refusal), false);
    });
  }
  test(`a ${kind} key conflict, transport failure or uncoded refusal stays unknown`, () => {
    const refusal = kind as Parameters<typeof submissionRefused>[1];
    assert.equal(
      submissionRefused(new ApiError(409, 'conflict', 'idempotency_conflict'), refusal),
      false,
    );
    assert.equal(submissionRefused(new ApiError(409, 'refused'), refusal), false);
    assert.equal(submissionRefused(new Error('timeout'), refusal), false);
  });
}
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
function held<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
test('a draft edit during preview preparation drops the older review', async () => {
  const { submission, connection } = setup();
  submission.preview = null;
  const response = held<unknown>();
  connection.client.request = () => response.promise;
  let edited = false;
  const preparing = submission.prepare({ values: { seed: 1 } }, () => !edited);
  edited = true;
  response.resolve({ id: 'older-preview' });
  assert.equal(await preparing, 'dropped');
  assert.equal(submission.preview, null);
  connection.client.request = async () => ({ id: 'current-preview' });
  assert.equal(await submission.prepare({ values: { seed: 2 } }, () => true), 'ready');
  assert.equal(submission.preview.id, 'current-preview');
});
test('a reconnect during preview preparation drops the old companion review', async () => {
  const { submission, connection } = setup();
  submission.preview = null;
  const response = held<unknown>();
  connection.client.request = () => response.promise;
  const preparing = submission.prepare({ values: {} });
  connection.client = { request: async () => ({ id: 'new-companion' }) };
  response.resolve({ id: 'old-companion-preview' });
  assert.equal(await preparing, 'dropped');
  assert.equal(submission.preview, null);
});
test('the unknown-outcome message stays in submission state after a refresh', async () => {
  const { submission, connection } = setup();
  let refreshed = 0;
  connection.refresh = async () => {
    refreshed++;
    connection.error = '';
  };
  connection.client.request = async () => {
    throw new Error('timeout');
  };
  await assert.rejects(submission.submit(), /timeout/);
  assert.ok(refreshed > 0);
  assert.match(submission.outcome, /outcome unknown until reconciled/);
  assert.equal(connection.error, '');
  connection.client.request = async () => ({ id: 'job' });
  assert.equal(await submission.reconcile(), true);
  assert.equal(submission.outcome, '');
  assert.equal(submission.pending, null);
});
test('a definitive refusal leaves no unknown-outcome message', async () => {
  const { submission, connection } = setup();
  connection.client.request = async () => {
    throw new ApiError(409, 'Expired', 'preview_expired');
  };
  await assert.rejects(submission.submit());
  assert.equal(submission.outcome, '');
});

test('a check with nothing to decide is ready, and a request or a warning opens the review', async () => {
  const { submission, connection } = setup();
  submission.preview = null;
  let warnings: unknown[] = [];
  connection.client.request = async () => ({ id: 'checked', errors: [], warnings });
  const bindings = [{ id: 'root:6:text' }];
  assert.equal(await submission.prepare({}, () => true, false, bindings), 'ready');
  assert.deepEqual(submission.reviewed, bindings);
  assert.equal(await submission.prepare({}, () => true, true), 'review');
  // A warning needs its own box ticked before the job can start, however it was asked for.
  warnings = ['Needs 96 GiB; 80 GiB is free.', { code: 'memory', message: 'Low memory.' }];
  assert.equal(await submission.prepare({}), 'review');
  assert.deepEqual(submission.acknowledged, [false, false]);
  assert.equal(submission.decided, false);
  await assert.rejects(submission.submit(), /Tick each warning/);
  assert.equal(submission.pending, null, 'nothing is sent before each warning is ticked');
  submission.acknowledge(0, true);
  assert.equal(submission.decided, false);
  submission.acknowledge(1, true);
  assert.equal(submission.decided, true);
  connection.client.request = async () => ({ id: 'job' });
  assert.deepEqual(await submission.submit(), { id: 'job' });
  // A new review starts with every box clear.
  connection.client.request = async () => ({ id: 'again', errors: [], warnings });
  assert.equal(await submission.prepare({}), 'review');
  assert.deepEqual(submission.acknowledged, [false, false]);
  // An error in the review blocks it whatever the boxes say.
  connection.client.request = async () => ({ id: 'broken', errors: ['Fix node 2.'], warnings: [] });
  assert.equal(await submission.prepare({}), 'review');
  assert.equal(submission.decided, false);
});

test('a refused draft opens the review with the refusal, and other failures keep the banner', async () => {
  const { submission, connection } = setup();
  for (const [status, code] of [
    [400, 'lora_binding'],
    [400, 'workflow'],
    [413, 'preview_limit'],
  ] as const) {
    connection.client.request = async () => {
      throw new ApiError(status, `refused ${code}`, code);
    };
    assert.equal(await submission.prepare({}), 'review');
    assert.deepEqual(submission.refusal, { code, message: `refused ${code}` });
    assert.equal(submission.preview, null);
  }
  // The next check that the companion accepts clears the refusal.
  connection.client.request = async () => ({ id: 'fixed', errors: [], warnings: [] });
  assert.equal(await submission.prepare({}), 'ready');
  assert.equal(submission.refusal, null);
  for (const error of [
    new ApiError(503, 'Driver not running', 'driver_not_running'),
    new ApiError(500, 'Server fault', 'internal'),
    new ApiError(400, 'No code'),
    new Error('timeout'),
  ]) {
    connection.client.request = async () => {
      throw error;
    };
    await assert.rejects(submission.prepare({}), (thrown) => thrown === error);
    assert.equal(submission.refusal, null);
  }
  // A refusal for a draft that changed meanwhile is dropped like a stale review.
  connection.client.request = async () => {
    throw new ApiError(400, 'stale', 'styles');
  };
  assert.equal(await submission.prepare({}, () => false), 'dropped');
  assert.equal(submission.refusal, null);
});

test('an accepted submit says so briefly, and a later line is never cleared early', async (t) => {
  mock.timers.enable({ apis: ['setTimeout'] });
  t.after(() => mock.timers.reset());
  const { submission, connection } = setup();
  await submission.submit();
  assert.equal(submission.confirmation, SUBMITTED);
  assert.doesNotMatch(SUBMITTED, /job-|[0-9a-f]{8}-/, 'the line names no identifier');
  mock.timers.tick(CONFIRMATION_MS - 1);
  assert.equal(submission.confirmation, SUBMITTED);
  // A second job just before the first line ends keeps its own full time.
  connection.client.request = async () => ({ id: 'second', errors: [], warnings: [] });
  assert.equal(await submission.prepare({}), 'ready');
  assert.equal(submission.confirmation, '', 'a new check clears the earlier line');
  connection.client.request = async () => ({ id: 'job-2' });
  await submission.submit();
  mock.timers.tick(1);
  assert.equal(submission.confirmation, SUBMITTED, "the first line's timer leaves the second");
  mock.timers.tick(CONFIRMATION_MS - 1);
  assert.equal(submission.confirmation, '');
  // A refused submit shows no confirmation.
  connection.client.request = async () => ({ id: 'third', errors: [], warnings: [] });
  await submission.prepare({});
  connection.client.request = async () => {
    throw new ApiError(409, 'Expired', 'preview_expired');
  };
  await assert.rejects(submission.submit());
  assert.equal(submission.confirmation, '');
});
