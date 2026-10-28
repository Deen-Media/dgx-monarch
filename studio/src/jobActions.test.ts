import test from 'node:test';
import assert from 'node:assert/strict';
import { ApiError } from './api.ts';
import { isActiveJob, isInFlightJob, requestJobAction } from './jobActions.ts';

const job = { id: 'job-1', state: 'failed', created_at: 1, submission: {} };

function setup(answer = true) {
  const store = new Map<string, string>();
  const questions: string[] = [];
  globalThis.sessionStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  } as Storage;
  globalThis.window = {
    confirm: (question: string) => {
      questions.push(question);
      return answer;
    },
  } as unknown as Window & typeof globalThis;
  const sent: { path: string; body: any }[] = [];
  let respond: (path: string, body: any) => Promise<unknown> = async () => ({ id: 'retry-1' });
  const client = {
    request: async (path: string, body: any) => {
      sent.push({ path, body });
      return respond(path, body);
    },
  };
  return {
    store,
    questions,
    sent,
    client: client as never,
    respond: (handler: typeof respond) => {
      respond = handler;
    },
  };
}

test('in-flight and unresolved states count as active work, as Queue counts them', () => {
  for (const state of ['queued', 'running', 'submitting', 'cancel_requested', 'accepted'])
    assert.equal(isActiveJob({ ...job, state }), true, state);
  // A job that ends unknown stays counted until Queue reconciles it, so it does not vanish.
  assert.equal(isActiveJob({ ...job, state: 'unknown' }), true);
  for (const state of ['failed', 'cancelled', 'succeeded'])
    assert.equal(isActiveJob({ ...job, state }), false, state);
});

test('only in-flight states drive the activity animations', () => {
  for (const state of ['queued', 'running', 'submitting', 'cancel_requested', 'accepted'])
    assert.equal(isInFlightJob({ ...job, state }), true, state);
  for (const state of ['failed', 'cancelled', 'succeeded', 'unknown'])
    assert.equal(isInFlightJob({ ...job, state }), false, state);
});

test('a declined confirmation sends nothing and keeps no identity', async () => {
  const context = setup(false);
  assert.equal(await requestJobAction(context.client, job, 'retry'), undefined);
  assert.deepEqual(context.sent, []);
  assert.equal(context.store.size, 0);
});

test('a confirmed action sends a fresh identity and clears it once the record arrives', async () => {
  const context = setup();
  const notices: string[] = [];
  const record = await requestJobAction(context.client, job, 'retry', (message) =>
    notices.push(message),
  );
  // App follows the retry this tab created in the output viewer.
  assert.deepEqual(record, { id: 'retry-1' });
  assert.equal(context.sent.length, 1);
  assert.equal(context.sent[0].path, '/jobs/job-1/retry');
  assert.equal(context.sent[0].body.confirmed, true);
  assert.match(context.sent[0].body.idempotency_key, /^[0-9a-f-]{36}$/);
  assert.equal(context.store.size, 0);
  assert.deepEqual(notices, []);
  assert.match(context.questions[0], /Create a new linked retry of job-1/);
});

for (const action of ['cancel', 'retry'] as const) {
  test(`a kept ${action} identity is reused and its answer comes with a recovery notice`, async () => {
    const context = setup();
    context.respond(async () => {
      throw new Error('timeout');
    });
    await assert.rejects(requestJobAction(context.client, job, action), /timeout/);
    const kept = context.store.get(`dgxm.studio.${action}.job-1`);
    assert.ok(kept);
    // The first attempt reached the companion; the same identity now returns its record.
    context.respond(async () => ({
      id: action === 'retry' ? 'retry-from-first-attempt' : 'job-1',
    }));
    const notices: string[] = [];
    await requestJobAction(context.client, job, action, (message) => notices.push(message));
    assert.equal(context.sent[1].body.idempotency_key, kept);
    assert.match(context.questions[1], /ended without a recorded answer/);
    assert.equal(notices.length, 1);
    assert.match(notices[0], /kept from an earlier attempt/);
    if (action === 'retry') assert.match(notices[0], /retry-from-first-attempt/);
    assert.equal(context.store.size, 0);
  });
}

test('an uncoded refusal or an identity-free answer keeps the identity for the next attempt', async () => {
  for (const answer of [
    async () => {
      throw new ApiError(409, 'Only a confirmed failed or cancelled graph task may be retried.');
    },
    async () => ({}),
  ]) {
    const context = setup();
    context.respond(answer);
    await assert.rejects(requestJobAction(context.client, job, 'retry'));
    assert.equal(context.store.size, 1);
  }
});

for (const [action, codes] of [
  ['cancel', ['confirmation', 'cancellation_scope']],
  ['retry', ['retry_scope', 'retry_provenance', 'schema_changed', 'choice_unavailable']],
] as const) {
  for (const code of codes) {
    test(`a definite ${code} refusal clears the ${action} identity, so the next click starts fresh`, async () => {
      const context = setup();
      context.respond(async () => {
        throw new ApiError(409, 'Refused before anything was reserved.', code);
      });
      await assert.rejects(requestJobAction(context.client, job, action), { code });
      assert.equal(context.store.size, 0);
      context.respond(async () => ({ id: action === 'retry' ? 'retry-2' : 'job-1' }));
      const notices: string[] = [];
      await requestJobAction(context.client, job, action, (message) => notices.push(message));
      assert.doesNotMatch(context.questions[1], /ended without a recorded answer/);
      assert.notEqual(context.sent[1].body.idempotency_key, context.sent[0].body.idempotency_key);
      assert.deepEqual(notices, []);
    });
  }
}

test('a key conflict, a server fault or a transport failure keeps the identity', async () => {
  for (const cause of [
    new ApiError(409, 'This key belongs to a different request.', 'idempotency_conflict'),
    new ApiError(502, 'Driver cancellation acknowledgement is malformed.', 'cancellation_receipt'),
    new ApiError(503, 'Driver unavailable.', 'driver_unavailable'),
    new Error('timeout'),
  ]) {
    const context = setup();
    context.respond(async () => {
      throw cause;
    });
    await assert.rejects(requestJobAction(context.client, job, 'cancel'));
    assert.equal(context.store.size, 1, cause.message);
  }
});

test('storage denial refuses before asking or sending', async () => {
  const context = setup();
  sessionStorage.getItem = () => {
    throw new Error('SecurityError');
  };
  await assert.rejects(requestJobAction(context.client, job, 'cancel'), /SecurityError/);
  assert.deepEqual(context.sent, []);
  assert.deepEqual(context.questions, []);
});
