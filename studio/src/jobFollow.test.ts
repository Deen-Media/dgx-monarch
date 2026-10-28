import test from 'node:test';
import assert from 'node:assert/strict';
import type { Artifact, Job } from './contracts.ts';
import {
  NO_FOLLOW,
  OUTPUT_LOOKUPS,
  adoptRequest,
  applyLookup,
  confirmOwn,
  finalOutput,
  followSnapshot,
  isJobOutput,
  newestActive,
  ownFollow,
  sameFollow,
  type Follow,
} from './jobFollow.ts';

const job = (id: string, state: string, created_at = 1, changes: Partial<Job> = {}) =>
  ({ id, state, created_at, submission: {}, ...changes }) as Job;
const output = (id: string, job_id: string | null, changes: Partial<Artifact> = {}) =>
  ({ id, job_id, kind: 'image', filename: `${id}.png`, created_at: 5, ...changes }) as Artifact;

/** Apply snapshots in order, as the viewer does on each poll. */
function replay(start: Follow, snapshots: [Job[], Artifact[]][], comparing = false) {
  let follow = start;
  const shown: (string | null)[] = [];
  for (const [jobs, artifacts] of snapshots) {
    const step = followSnapshot(follow, jobs, artifacts, comparing);
    follow = step.follow;
    shown.push(step.show?.id ?? null);
  }
  return { follow, shown };
}

test('the strip and the viewer take job outputs only', () => {
  assert.equal(isJobOutput(output('out', 'j')), true);
  assert.equal(isJobOutput(output('video', 'j', { kind: 'video' })), true);
  assert.equal(isJobOutput(output('temp', 'j', { intermediate: true })), true);
  assert.equal(isJobOutput(output('upload', null, { kind: 'input' as never })), false);
  assert.equal(isJobOutput(output('mask', null, { kind: 'mask' as never })), false);
  assert.equal(
    isJobOutput(output('live', 'j', { source: 'live_preview', intermediate: true })),
    false,
  );
});

test('the result of a finished job is its last stored final image or video', () => {
  const artifacts = [
    output('live', 'j', { source: 'live_preview', intermediate: true, created_at: 9 }),
    output('temp', 'j', { intermediate: true, created_at: 9 }),
    output('other-job', 'k', { created_at: 9 }),
    output('upscaled', 'j', { created_at: 7 }),
    output('base', 'j', { created_at: 7 }),
    output('older', 'j', { created_at: 6 }),
  ];
  // The snapshot lists the most recently stored artifact first among equal creation times.
  assert.equal(finalOutput('j', artifacts)?.id, 'upscaled');
  assert.equal(finalOutput('j', [output('temp', 'j', { intermediate: true })]), null);
  assert.equal(finalOutput('missing', artifacts), null);
});

test('a looked-up result is the one the snapshot would show', () => {
  // The snapshot and the artifacts route both list a job's outputs most recently stored first.
  const stored = [
    output('second', 'j', { created_at: 7 }),
    output('first', 'j', { created_at: 7 }),
  ];
  const done = [job('j', 'succeeded')];
  assert.equal(followSnapshot(ownFollow('j'), done, stored, false).show?.id, 'second');
  assert.equal(applyLookup(ownFollow('j'), 'j', stored, false).show?.id, 'second');
  // Equal times never reorder here: whichever order the companion sends decides.
  assert.equal(finalOutput('j', [...stored].reverse())?.id, 'first');
});

test('the tab follows the job it submitted and shows its result once it succeeds', () => {
  const start = ownFollow('mine');
  const { follow, shown } = replay(start, [
    [[job('mine', 'queued')], []],
    [[job('mine', 'running'), job('later', 'running', 9)], []],
    [[job('mine', 'succeeded'), job('later', 'running', 9)], [output('result', 'mine')]],
    [[job('mine', 'succeeded'), job('later', 'running', 9)], [output('result', 'mine')]],
  ]);
  assert.deepEqual(shown, [null, null, 'result', null], 'the result shows once');
  assert.equal(follow.id, 'mine', 'a newer job elsewhere does not take over');
});

test('a choice the user made after submitting, or an open comparison, keeps the viewer', () => {
  const done: [Job[], Artifact[]] = [[job('mine', 'succeeded')], [output('result', 'mine')]];
  const picked = replay({ ...ownFollow('mine'), picked: true }, [done]);
  assert.deepEqual(picked.shown, [null]);
  assert.equal(picked.follow.settled, true, 'the result never shows later either');
  const comparing = replay(ownFollow('mine'), [done, done], true);
  assert.deepEqual(comparing.shown, [null, null]);
  // Closing the comparison afterwards does not bring a stale result in.
  assert.deepEqual(replay(comparing.follow, [done]).shown, [null]);
});

test('a failed or cancelled job shows no result', () => {
  for (const state of ['failed', 'cancelled']) {
    const { follow, shown } = replay(ownFollow('mine'), [
      [[job('mine', state)], [output('partial', 'mine')]],
    ]);
    assert.deepEqual(shown, [null], state);
    assert.equal(follow.settled, true, state);
  }
});

test('a reload never shows the result of a job that had already ended as new', () => {
  const settled = [job('old', 'succeeded', 5), job('older', 'failed', 4)];
  const artifacts = [output('old-result', 'old')];
  const fresh = replay(NO_FOLLOW, [
    [settled, artifacts],
    [settled, artifacts],
  ]);
  assert.deepEqual(fresh.shown, [null, null]);
  assert.equal(fresh.follow.id, null);
  // A restored request identity names a job that had already ended: follow it, show nothing.
  const adopted = adoptRequest(NO_FOLLOW, 'key', [
    job('old', 'succeeded', 5, { idempotency_key: 'key' }),
  ]);
  assert.deepEqual(adopted, { id: 'old', own: true, settled: true, picked: false, lookups: 0 });
  assert.deepEqual(replay(adopted!, [[settled, artifacts]]).shown, [null]);
  assert.equal(adoptRequest(NO_FOLLOW, 'key', settled), null);
  assert.equal(adoptRequest(NO_FOLLOW, '', settled), null);
});

test('a job this page sent shows its result even when it ended before the follow began', () => {
  // The submit reply was lost and the job finished before the snapshot listed it.
  const listed = [job('fast', 'succeeded', 5, { idempotency_key: 'key' })];
  const adopted = adoptRequest(NO_FOLLOW, 'key', listed, true);
  assert.deepEqual(adopted, { id: 'fast', own: true, settled: false, picked: false, lookups: 0 });
  assert.deepEqual(replay(adopted!, [[listed, [output('fast-result', 'fast')]]]).shown, [
    'fast-result',
  ]);
  // The same identity restored after a reload names an old job.
  assert.equal(adoptRequest(NO_FOLLOW, 'key', listed, false)?.settled, true);
  // A cached prompt can succeed before the submit call returns; the tab still submitted it.
  assert.deepEqual(
    replay(ownFollow('fast'), [[[job('fast', 'succeeded')], [output('fast-result', 'fast')]]])
      .shown,
    ['fast-result'],
  );
});

test('confirming the job a follow already has keeps what happened since it began', () => {
  // Polling adopted the job while submit waited for its refresh, and the user chose a result.
  const during = { ...ownFollow('mine'), picked: true };
  assert.deepEqual(confirmOwn(during, 'mine'), during);
  const settled = { ...ownFollow('mine'), settled: true, lookups: 2 };
  assert.deepEqual(confirmOwn(settled, 'mine'), settled);
  const newest = { ...NO_FOLLOW, id: 'mine', picked: true };
  assert.deepEqual(confirmOwn(newest, 'mine'), { ...newest, own: true });
  assert.deepEqual(confirmOwn(during, 'next'), ownFollow('next'), 'another job starts fresh');
  assert.deepEqual(
    adoptRequest(newest, 'key', [job('mine', 'running', 1, { idempotency_key: 'key' })]),
    {
      ...newest,
      own: true,
    },
  );
});

test('a result the snapshot left out is asked of the artifacts route a bounded number of times', () => {
  // The snapshot's 200 artifacts hold other outputs, not this job's final image.
  const crowded = Array.from({ length: 200 }, (_, index) => output(`temp-${index}`, 'other'));
  const done: Job[] = [job('mine', 'succeeded')];
  let step = followSnapshot(ownFollow('mine'), done, crowded, false);
  assert.equal(step.show, null);
  assert.equal(step.lookup, 'mine');
  assert.equal(step.follow.settled, false, 'the result is not handled yet');
  const found = applyLookup(step.follow, 'mine', [output('result', 'mine')], false);
  assert.equal(found.show?.id, 'result');
  assert.equal(found.follow.settled, true);
  // Answers without the output, or failed requests, count until the bound settles the follow.
  let follow = step.follow;
  for (let answer = 1; answer <= OUTPUT_LOOKUPS; answer++) {
    assert.equal(followSnapshot(follow, done, crowded, false).lookup, 'mine', String(answer));
    const empty = applyLookup(
      follow,
      'mine',
      answer === 2 ? null : [output('temp', 'mine', { intermediate: true })],
      false,
    );
    assert.equal(empty.show, null);
    follow = empty.follow;
  }
  assert.deepEqual(follow, { ...ownFollow('mine'), settled: true, lookups: OUTPUT_LOOKUPS });
  step = followSnapshot(follow, done, crowded, false);
  assert.equal(step.lookup, null, 'no request after the bound');
  // A later snapshot that carries the output still shows it while lookups remain.
  assert.equal(
    followSnapshot({ ...ownFollow('mine'), lookups: 1 }, done, [output('result', 'mine')], false)
      .show?.id,
    'result',
  );
});

test('an answer for another job, a settled follow or a new choice shows nothing', () => {
  const waiting = ownFollow('mine');
  const items = [output('result', 'mine')];
  assert.deepEqual(applyLookup(waiting, 'old', items, false), { follow: waiting, show: null });
  const settled = { ...waiting, settled: true };
  assert.deepEqual(applyLookup(settled, 'mine', items, false), { follow: settled, show: null });
  assert.deepEqual(applyLookup({ ...waiting, picked: true }, 'mine', items, false).show, null);
  const comparing = applyLookup(waiting, 'mine', items, true);
  assert.equal(comparing.show, null);
  assert.equal(comparing.follow.settled, true);
});

test('the newest active job waits while the followed output is looked up', () => {
  const jobs = [job('done', 'succeeded', 2), job('next', 'running', 3)];
  const waiting = { ...NO_FOLLOW, id: 'done' };
  const step = followSnapshot(waiting, jobs, [], false);
  assert.equal(step.lookup, 'done');
  assert.equal(step.follow.id, 'done');
  const answered = applyLookup(step.follow, 'done', [output('done-result', 'done')], false);
  assert.equal(answered.show?.id, 'done-result');
  assert.equal(followSnapshot(answered.follow, jobs, [], false).follow.id, 'next');
});

test('with no job from this tab the viewer follows the newest active job', () => {
  const { follow, shown } = replay(NO_FOLLOW, [
    [[job('a', 'running', 2), job('b', 'queued', 3), job('z', 'succeeded', 9)], []],
    [[job('a', 'succeeded', 2), job('b', 'running', 3)], [output('a-result', 'a')]],
    [[job('a', 'succeeded', 2), job('b', 'succeeded', 3)], [output('b-result', 'b')]],
    [[job('a', 'succeeded', 2), job('b', 'succeeded', 3)], [output('b-result', 'b')]],
  ]);
  assert.deepEqual(shown, [null, null, 'b-result', null]);
  assert.deepEqual(follow, { id: 'b', own: false, settled: true, picked: false, lookups: 0 });
  // An unresolved job needs attention, so it counts as active and is followed.
  assert.equal(newestActive([job('lost', 'unknown', 4), job('done', 'succeeded', 8)])?.id, 'lost');
  assert.equal(newestActive([job('x', 'running', 4), job('y', 'running', 4)])?.id, 'y');
  assert.equal(newestActive([job('done', 'failed')]), undefined);
});

test('a new submission starts a fresh follow', () => {
  const picked = { ...ownFollow('first'), picked: true, settled: true };
  assert.equal(sameFollow(picked, ownFollow('first')), false);
  assert.deepEqual(ownFollow('second'), {
    id: 'second',
    own: true,
    settled: false,
    picked: false,
    lookups: 0,
  });
  assert.equal(sameFollow(ownFollow('a'), { ...ownFollow('a'), lookups: 1 }), false);
  assert.equal(sameFollow(ownFollow('a'), ownFollow('a')), true);
});
