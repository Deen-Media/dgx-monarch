import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { Job } from './contracts.ts';
import {
  CLOCK_LIMIT_S,
  companionTime,
  firstSentence,
  formatDurationClock,
  jobColor,
  jobLine,
  steadyClock,
} from './jobStatus.ts';

const job = (state: string, changes: Partial<Job> = {}) =>
  ({ id: 'j', state, created_at: 1, submission: {}, ...changes }) as Job;
const contract = JSON.parse(
  readFileSync(
    new URL('../../src/dgx_monarch/studio/data/presentation.json', import.meta.url),
    'utf8',
  ),
);

test('durations read as m:ss with whole seconds and uncapped minutes', () => {
  const cases: [unknown, string][] = [
    [0, '0:00'],
    [42.9, '0:42'],
    [60, '1:00'],
    [73.4, '1:13'],
    [3661, '61:01'],
    [Number.MAX_SAFE_INTEGER, '150119987579016:31'],
  ];
  for (const [value, text] of cases) assert.equal(formatDurationClock(value), text, String(value));
  for (const value of [-1, NaN, Infinity, null, undefined, '12', true, Number.MAX_SAFE_INTEGER + 2])
    assert.equal(formatDurationClock(value), 'unknown', String(value));
});

test('each job state has its own status line wording', () => {
  const lines = Object.fromEntries(
    ['submitting', 'accepted', 'queued', 'cancelled', 'succeeded', 'unknown'].map((state) => [
      state,
      jobLine(job(state), 100),
    ]),
  );
  assert.equal(lines.submitting.headline, 'Submitting');
  assert.equal(lines.accepted.headline, 'Submitting');
  assert.equal(lines.queued.headline, 'Queued');
  assert.equal(lines.queued.progress, null, 'a queued job draws no bar');
  assert.equal(lines.cancelled.headline, 'Cancelled');
  assert.equal(lines.succeeded.headline, 'Finished');
  assert.equal(lines.unknown.headline, 'Outcome unknown');
  assert.equal(lines.unknown.link, 'Reconcile in Queue');
  assert.equal(lines.unknown.announcement, 'Outcome unknown. Reconcile in Queue');
});

test('a running job counts on the companion clock from when the companion saw it run', () => {
  // The driver runs on another host whose clock is far behind; its start time never enters.
  const running = job('running', {
    started_at: 5,
    running_observed_at: 1000,
    progress: { value: 5, max: 20, node: '3' },
  });
  const line = jobLine(running, 1042.7);
  assert.equal(line.headline, 'Running 0:42');
  assert.equal(line.detail, 'step 5 of 20 on node 3');
  assert.deepEqual(line.progress, { value: 5, max: 20 });
  assert.equal(line.announcement, 'Running', 'neither the time nor the steps are announced');
  // The driver reported its end before history settlement stored the outputs: the time stops at
  // the driver's own start to end time.
  const ended = jobLine({ ...running, progress: null, ended_at: 78.2 }, 2000);
  assert.equal(ended.headline, 'Running 1:13');
  assert.equal(ended.progress, 'indeterminate');
  // Only the driver's start is known: no live time rather than one across two clocks.
  assert.equal(jobLine(job('running', { started_at: 1000 }), 1042.7).headline, 'Running');
});

test('a running job without a step count draws a bar with no known end', () => {
  const line = jobLine(job('running', { started_at: 10, progress: null }), 70);
  assert.equal(line.progress, 'indeterminate');
  assert.equal(line.detail, '');
  // A report without a denominator is not a share of the work either.
  assert.equal(jobLine(job('running', { progress: { value: 3 } }), 70).progress, 'indeterminate');
  const cancelling = jobLine(job('cancel_requested', { progress: { value: 2, max: 8 } }), 70);
  assert.equal(cancelling.headline, 'Cancel requested');
  assert.deepEqual(cancelling.progress, { value: 2, max: 8 });
  assert.equal(jobLine(job('cancel_requested'), 70).progress, null);
});

test('the bar shows one node report as given, never a share merged across nodes', () => {
  const line = jobLine(job('running', { progress: { value: 7, max: 7, node: 'vae' } }), null);
  assert.deepEqual(line.progress, { value: 7, max: 7 });
  assert.equal(line.detail, 'step 7 of 7 on node vae');
  assert.equal(line.headline, 'Running', 'no companion clock, no time');
});

test('no time appears when a timestamp is missing or the clocks disagree', () => {
  assert.equal(jobLine(job('running', { running_observed_at: null }), 50).headline, 'Running');
  assert.equal(jobLine(job('running', { running_observed_at: 60 }), 50).headline, 'Running');
  assert.equal(jobLine(job('running', { running_observed_at: 10 }), null).headline, 'Running');
  assert.equal(jobLine(job('succeeded', { started_at: 10 }), 50).headline, 'Finished');
  assert.equal(jobLine(job('succeeded', { ended_at: 10 }), 50).headline, 'Finished');
  assert.equal(
    jobLine(job('succeeded', { started_at: 10, ended_at: 83.9 }), 5).headline,
    'Finished in 1:13',
    'a finished job uses the driver times only',
  );
  assert.equal(
    jobLine(job('cancelled', { started_at: 10, ended_at: 52 }), 0).headline,
    'Cancelled after 0:42',
  );
  for (const state of ['queued', 'running', 'succeeded', 'failed'])
    assert.doesNotMatch(
      jobLine(job(state, { started_at: 1, ended_at: 5 }), 9).headline,
      /left|remaining|eta/i,
    );
});

test('a failed job shows its code, the first sentence of its message and a Queue link', () => {
  const failed = jobLine(
    job('failed', {
      started_at: 1,
      ended_at: 43,
      error: {
        code: 'OutOfMemoryError',
        message: 'CUDA out of memory. Tried to allocate 2.00 GiB on device 0.',
      },
    }),
    null,
  );
  assert.equal(failed.headline, 'Failed after 0:42');
  assert.equal(failed.detail, 'OutOfMemoryError: CUDA out of memory.');
  assert.equal(failed.link, 'Open Queue');
  assert.equal(failed.announcement, 'Failed. OutOfMemoryError: CUDA out of memory.');
  assert.equal(
    jobLine(job('failed'), null).detail,
    '',
    'a failure without an error says nothing more',
  );
  assert.equal(
    jobLine(job('failed', { error: { code: 'prompt_rejected' } }), null).detail,
    'prompt_rejected',
  );
});

test('the first sentence keeps file names and decimals whole', () => {
  assert.equal(
    firstSentence('Missing model.safetensors in loras. Check the path.'),
    'Missing model.safetensors in loras.',
  );
  assert.equal(firstSentence('Loaded 2.5 GiB! Then failed.'), 'Loaded 2.5 GiB!');
  assert.equal(
    firstSentence('Error(s) in loading state_dict:\n\tsize mismatch'),
    'Error(s) in loading state_dict:',
  );
  assert.equal(firstSentence('no terminator'), 'no terminator');
  assert.equal(firstSentence(undefined), '');
  const long = firstSentence('x'.repeat(500));
  assert.equal(long.length, 200);
  assert.ok(long.endsWith('…'));
});

test('the clock runs from the snapshot server time, not the browser wall clock', () => {
  const anchor = { server: 1000, at: 50_000 };
  assert.equal(companionTime(null, 50_000), null);
  assert.equal(companionTime(anchor, 50_000), 1000);
  assert.equal(companionTime(anchor, 51_500), 1001.5);
  // Without a newer snapshot the clock holds, so a lost connection cannot count on.
  assert.equal(companionTime(anchor, 90_000), 1000 + CLOCK_LIMIT_S);
  assert.equal(
    companionTime(anchor, 40_000),
    1000,
    'a timer reading before the anchor adds nothing',
  );
});

test('a new snapshot never makes the shown time count down', () => {
  assert.equal(steadyClock(null, 10), 10);
  assert.equal(steadyClock(10.9, 10.85), 10.9, 'request timing jitter holds the clock');
  assert.equal(steadyClock(10, 11), 11);
  assert.equal(steadyClock(100, 50), 50, 'a real clock change applies');
  assert.equal(steadyClock(10, null), null);
});

test('job states take their colors from the shared presentation contract', () => {
  const colors = Object.fromEntries(
    ['queued', 'running', 'succeeded', 'failed', 'unknown', 'cancelled', 'cancel_requested'].map(
      (state) => [state, jobColor(contract, state)],
    ),
  );
  assert.deepEqual(colors, {
    queued: '#4ade80',
    running: '#4ade80',
    succeeded: '#4ade80',
    failed: '#f87171',
    unknown: '#f87171',
    cancelled: '#facc15',
    cancel_requested: '#facc15',
  });
  assert.equal(jobColor(contract, 'invented'), undefined);
  assert.equal(jobColor(undefined, 'running'), undefined);
});
