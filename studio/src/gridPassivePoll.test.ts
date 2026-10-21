import test from 'node:test';
import assert from 'node:assert/strict';
import { ApiError } from './api.ts';
import { dispatchAbsent, GridPassivePoll, type PassivePollOptions } from './gridPassivePoll.ts';
import type { Grid } from './gridState.ts';
import { submissionRefused } from './submissionRefusal.ts';
import { componentFunctions } from './renderHarness.js';

function harness(read: (id: string) => Promise<Grid>, changes: Partial<PassivePollOptions> = {}) {
  const view = { target: 'grid-a' as string | undefined, ticket: 0, paused: false };
  const log = { applied: [] as string[], missing: [] as string[], reports: [] as string[] };
  const delays: number[] = [];
  let next: (() => void) | null = null;
  const poll = new GridPassivePoll({
    read,
    target: () => view.target,
    ticket: () => view.ticket,
    paused: () => view.paused,
    apply: (id) => log.applied.push(id),
    missing: (id) => log.missing.push(id),
    report: (text) => log.reports.push(text),
    schedule: (run, delay) => {
      delays.push(delay);
      next = run;
      return run;
    },
    cancel: () => {
      next = null;
    },
    ...changes,
  });
  return { poll, view, log, delays, next: () => next };
}
const grid = (id: string) => ({ id, cells: [] }) as Grid;
const flush = () => new Promise((resolve) => setImmediate(resolve));

test('failures back off up to the ceiling and a success resets the interval', async () => {
  let failing = true;
  const { poll, log, delays, next } = harness(async (id) => {
    if (failing) throw new Error('Companion unavailable');
    return grid(id);
  });
  poll.start();
  await flush();
  for (let attempt = 0; attempt < 6; attempt++) {
    next()!();
    await flush();
  }
  assert.deepEqual(delays, [6000, 12000, 24000, 48000, 60000, 60000, 60000]);
  assert.match(log.reports.at(-1)!, /tries again in 60 s: Companion unavailable/);
  failing = false;
  next()!();
  await flush();
  assert.equal(delays.at(-1), 3000);
  assert.equal(log.reports.at(-1), '', 'a success clears the passive status');
  assert.deepEqual(log.applied, ['grid-a']);
  poll.stop();
});

test('a slow reply for another grid or an earlier action never lands', async () => {
  const pending: Array<{ resolve: (value: Grid) => void; reject: (cause: Error) => void }> = [];
  const { poll, view, log, next } = harness(
    () =>
      new Promise((resolve, reject) => {
        pending.push({ resolve, reject });
      }),
  );
  poll.start();
  view.target = 'grid-b';
  pending.shift()!.reject(new Error('timeout for grid A'));
  await flush();
  next()!();
  view.ticket++;
  pending.shift()!.resolve(grid('grid-b'));
  await flush();
  assert.deepEqual(log, { applied: [], missing: [], reports: [] });
  next()!();
  pending.shift()!.resolve(grid('grid-b'));
  await flush();
  assert.deepEqual(log.applied, ['grid-b'], 'a current reply still lands');
  poll.stop();
});

test('a passive 404 settles the grid as absent instead of polling forever', async () => {
  const { poll, log, delays } = harness(async () => {
    throw new ApiError(404, 'Grid not found.', 'not_found');
  });
  poll.start();
  await flush();
  assert.deepEqual(log.missing, ['grid-a']);
  assert.deepEqual(log.reports, []);
  assert.deepEqual(delays, [3000], 'a settled absence does not back off');
  poll.stop();
  const refused = harness(async () => {
    throw new ApiError(409, 'Busy.', 'busy');
  });
  refused.poll.start();
  await flush();
  assert.deepEqual(refused.log.missing, [], 'only a 404 is definitive');
  assert.equal(refused.log.reports.length, 1);
  refused.poll.stop();
});

test('a paused or empty view keeps polling without a request and stop ends the loop', async () => {
  let reads = 0;
  const { poll, view, next } = harness(async (id) => {
    reads++;
    return grid(id);
  });
  view.paused = true;
  poll.start();
  await flush();
  view.paused = false;
  view.target = undefined;
  next()!();
  await flush();
  assert.equal(reads, 0);
  assert.notEqual(next(), null);
  poll.stop();
  assert.equal(next(), null);
  await poll.tick();
  assert.equal(next(), null, 'a stopped poll schedules nothing');
});

// Run the shipped Grids dispatch handler against a companion that no longer holds the preview.
const dispatchCode = await componentFunctions('./views/Grids.svelte', ['dispatch', 'fail']);
const dispatchHarness = new Function(
  'client',
  'submissionRefused',
  'dispatchAbsent',
  `'use strict';
  let recoveryBlocked = false, preview = { id: 'review' }, confirmed = true, loadGeneration = 0;
  let busy = false, error = '', dispatchPending = true, grid = null, requestKey = 'kept-key', saved = 0;
  const remember = () => { saved++; };
  const requireRecordIdentity = (value) => value;
  ${dispatchCode}
  return { dispatch, state: () => ({ dispatchPending, preview, confirmed, error, saved }) };`,
);

test('a restored dispatch receipt settles when the companion accepted nothing under its key', async () => {
  const refusing = (cause: Error) => ({
    request: async () => {
      throw cause;
    },
  });
  const absent = dispatchHarness(
    refusing(new ApiError(404, 'Record not found.', 'not_found')),
    submissionRefused,
    dispatchAbsent,
  );
  await absent.dispatch();
  const settled = absent.state();
  assert.equal(settled.dispatchPending, false);
  assert.equal(settled.preview, null);
  assert.equal(settled.confirmed, false);
  assert.match(settled.error, /submitted nothing/);
  for (const cause of [
    new Error('The operation was aborted.'),
    new ApiError(404, 'Not Found'),
    new ApiError(502, 'Driver unavailable.', 'driver_unavailable'),
  ]) {
    const pending = dispatchHarness(refusing(cause), submissionRefused, dispatchAbsent);
    await pending.dispatch();
    assert.equal(pending.state().dispatchPending, true, cause.message);
    assert.deepEqual(pending.state().preview, { id: 'review' }, cause.message);
  }
});

// Run the shipped Grids retry handler: a definite refusal ends the pending retry.
const retryCode = await componentFunctions('./views/Grids.svelte', ['retry', 'fail']);
const retryHarness = new Function(
  'client',
  'submissionRefused',
  `'use strict';
  let recoveryBlocked = false, grid = { id: 'grid-1' }, retryPending = true, selected = [];
  let loadGeneration = 0, busy = false, error = '', retryKey = 'kept-key', retryIndices = [2], saved = 0;
  const remember = () => { saved++; };
  const requireRecordIdentity = (value) => value;
  ${retryCode}
  return { retry, state: () => ({ retryPending, retryKey, retryIndices, error, saved }) };`,
);

test('a restored grid retry settles on a definite refusal and stays pending otherwise', async () => {
  const refusing = (cause: Error) => ({
    request: async () => {
      throw cause;
    },
  });
  for (const code of ['grid_retry', 'schema_changed', 'choice_unavailable']) {
    const refused = retryHarness(
      refusing(new ApiError(409, `refused: ${code}`, code)),
      submissionRefused,
    );
    await refused.retry();
    assert.deepEqual(refused.state().retryPending, false, code);
    assert.equal(refused.state().retryKey, '', code);
    assert.match(refused.state().error, new RegExp(code));
  }
  for (const cause of [
    new Error('The operation was aborted.'),
    new ApiError(409, 'This key belongs to a different request.', 'idempotency_conflict'),
    new ApiError(502, 'Driver unavailable.', 'driver_unavailable'),
  ]) {
    const pending = retryHarness(refusing(cause), submissionRefused);
    await pending.retry();
    assert.equal(pending.state().retryPending, true, cause.message);
    assert.equal(pending.state().retryKey, 'kept-key', cause.message);
  }
});
