import test from 'node:test';
import assert from 'node:assert/strict';
import type { Artifact, Job, WorkflowEntry } from './contracts.ts';
import { jobLine } from './jobStatus.ts';
import { beginsRun, jobWorkflow, NO_FOLLOW, ownFollow, pendingView } from './jobFollow.ts';
// The harness compiles the rune module; outside a render its deriveds recompute on each read.
import { renderComponent, visibleText } from './renderHarness.js';

const { JobTracker } = await import('./jobTracker.svelte.ts');
const noop = () => {};
const H3: WorkflowEntry = {
  id: 'bundled/h3',
  name: 'minimax_h3_mp4_sound',
  title: 'MiniMax H3 MP4 with sound',
  workflow: {},
};
const job = (id: string, state: string, changes: Partial<Job> = {}) =>
  ({
    id,
    state,
    created_at: 1,
    submission: { requested: { workflowName: H3.name } },
    ...changes,
  }) as Job;
const output = (id: string, job_id: string, kind = 'image') =>
  ({ id, job_id, kind, filename: `${id}.png`, created_at: 5 }) as Artifact;

function setup() {
  const connection = {
    snapshot: {
      jobs: [] as Job[],
      artifacts: [] as Artifact[],
      server_time: 1000,
      jobs_truncated: false,
    },
    capabilities: { operations: [] },
    workflows: [H3],
    client: { request: async () => ({ items: [] }) },
  };
  const submission = { pending: null as { key: string } | null, createdKey: '' };
  const tracker = new JobTracker(connection as never, submission as never);
  let polls = 0;
  const poll = (jobs: Job[], artifacts: Artifact[] = []) => {
    connection.snapshot = { jobs, artifacts, server_time: 1000 + ++polls, jobs_truncated: false };
    tracker.observe();
  };
  return { tracker, submission, poll, connection };
}

test('a new generation clears the earlier output until its own lands', () => {
  const { tracker, poll, connection } = setup();
  tracker.own('first');
  const earlier = [output('first-result', 'first')];
  poll([job('first', 'succeeded')], earlier);
  assert.equal(tracker.shown?.id, 'first-result');
  // Generate again: the viewer drops the earlier output at once, before any snapshot lists the job.
  tracker.own('second');
  assert.equal(tracker.shown, null);
  poll(
    [job('second', 'running', { running_observed_at: 990 }), job('first', 'succeeded')],
    earlier,
  );
  assert.equal(tracker.shown, null);
  assert.equal(tracker.status?.workflow, 'MiniMax H3 MP4 with sound');
  assert.equal(tracker.status?.headline, 'Running 0:12');
  // Recent Outputs reads the snapshot, which still lists the earlier output.
  assert.deepEqual(connection.snapshot.artifacts, earlier);
  poll(
    [job('second', 'succeeded'), job('first', 'succeeded')],
    [output('next', 'second'), ...earlier],
  );
  assert.equal(tracker.shown?.id, 'next');
});

test('a result chosen while the job runs stays, and so does an open comparison', () => {
  const picked = setup();
  picked.tracker.own('first');
  picked.poll([job('first', 'succeeded')], [output('first-result', 'first')]);
  picked.tracker.own('second');
  picked.tracker.selected = output('first-result', 'first');
  picked.poll(
    [job('second', 'succeeded')],
    [output('next', 'second'), output('first-result', 'first')],
  );
  assert.equal(picked.tracker.shown?.id, 'first-result');
  const comparing = setup();
  comparing.tracker.selected = output('left', 'old');
  comparing.tracker.comparison = output('right', 'old');
  comparing.tracker.own('mine');
  comparing.poll([job('mine', 'running')]);
  assert.equal(comparing.tracker.selected?.id, 'left');
  assert.equal(comparing.tracker.comparison?.id, 'right');
});

test('a run that fails or is cancelled leaves no earlier output on screen', () => {
  for (const state of ['failed', 'cancelled']) {
    const { tracker, poll } = setup();
    tracker.selected = output('old-result', 'old');
    tracker.own('mine');
    poll([job('mine', state)]);
    assert.equal(tracker.shown, null, state);
    assert.equal(pendingView(tracker.status!).tag, 'No output');
  }
});

test('a tab that submitted nothing clears for a job in flight, not for an unknown outcome', () => {
  const { tracker, poll } = setup();
  tracker.selected = output('chosen', 'old');
  poll([job('lost', 'unknown')]);
  assert.equal(tracker.follow.id, 'lost');
  assert.equal(tracker.shown?.id, 'chosen', 'an unresolved job is no new generation');
  poll([job('other-tab', 'queued', { created_at: 2 }), job('lost', 'unknown')]);
  assert.equal(tracker.follow.id, 'other-tab');
  assert.equal(tracker.shown, null);
  assert.equal(tracker.status?.own, false);
});

test('when one snapshot ends the followed job and starts a newer one, the newer one shows', () => {
  const { tracker, poll } = setup();
  poll([job('a', 'running')]);
  const finished = [output('a-result', 'a')];
  poll([job('b', 'queued', { created_at: 2 }), job('a', 'succeeded')], finished);
  assert.equal(tracker.follow.id, 'b');
  assert.equal(tracker.shown, null, 'the output of a stays in Recent Outputs');
});

test('a request this page sent clears the viewer even when its job ended before a poll', () => {
  const { tracker, submission, poll } = setup();
  tracker.selected = output('old-result', 'old');
  // submit() set the identity, then its reply was lost, so own() never ran.
  submission.createdKey = 'key';
  submission.pending = { key: 'key' };
  poll([job('mine', 'failed', { idempotency_key: 'key' })]);
  assert.equal(tracker.follow.id, 'mine');
  assert.equal(tracker.shown, null);
  const done = setup();
  done.tracker.selected = output('old-result', 'old');
  done.submission.createdKey = 'key';
  done.submission.pending = { key: 'key' };
  done.poll([job('mine', 'succeeded', { idempotency_key: 'key' })], [output('result', 'mine')]);
  assert.equal(done.tracker.shown?.id, 'result');
});

test('the follow helpers name the workflow and spot a new job in flight', () => {
  assert.equal(jobWorkflow(job('j', 'running'), [H3]), 'MiniMax H3 MP4 with sound');
  const imported = job('j', 'running', {
    submission: { requested: { workflowName: 'my_graph.json' } },
  });
  assert.equal(jobWorkflow(imported, [H3]), 'my_graph.json');
  assert.equal(jobWorkflow(job('j', 'running', { submission: {} }), [H3]), '');
  const jobs = [job('run', 'running'), job('done', 'succeeded'), job('lost', 'unknown')];
  assert.equal(beginsRun(NO_FOLLOW, ownFollow('run'), jobs), true);
  assert.equal(beginsRun(ownFollow('run'), { ...ownFollow('run'), own: false }, jobs), false);
  assert.equal(beginsRun(NO_FOLLOW, ownFollow('done'), jobs), false);
  assert.equal(beginsRun(NO_FOLLOW, ownFollow('lost'), jobs), false);
  assert.equal(beginsRun(NO_FOLLOW, ownFollow('missing'), jobs), false);
});

const viewer = (job: unknown, artifact: Artifact | null = null) =>
  renderComponent('./components/Viewer.svelte', {
    job,
    artifact,
    working: true,
    client: { request: async () => ({}) },
    onreuse: async () => {},
    onselect: noop,
    onqueue: noop,
  });
const followed = (state: string, changes: Partial<Job> = {}) => ({
  ...jobLine({ ...job('j', state), ...changes } as Job, 1042),
  id: 'j',
  own: true,
  workflow: 'MiniMax H3 MP4 with sound',
});

test('while the job runs the viewer names its workflow and progress, never an old file', async () => {
  const running = followed('running', {
    running_observed_at: 1000,
    progress: { value: 5, max: 20, node: '3' },
  });
  const text = visibleText(await viewer(running));
  assert.match(
    text,
    /^OUTPUT VIEWER MiniMax H3 MP4 with sound No output yet Running 0:42 · step 5 of 20 on node 3 /,
  );
  assert.match(text, /Waiting for the new output It appears here when the job finishes\./);
  assert.doesNotMatch(text, /\.png|\.mp4|No output selected|Make something remarkable/);
  assert.doesNotMatch(text, /Reuse settings|Send to edit/, 'no output footer');
  const failed = visibleText(await viewer(followed('failed', { error: { message: 'Stopped.' } })));
  assert.match(failed, /^OUTPUT VIEWER MiniMax H3 MP4 with sound No output Failed/);
  assert.match(failed, /No new output This job has no output to show\./);
  // The new output, once it lands, takes the viewer back to the file and its kind.
  const landed = visibleText(await viewer(followed('succeeded'), output('next', 'j', 'video')));
  assert.match(landed, /^OUTPUT VIEWER next\.png video Finished/);
  assert.doesNotMatch(landed, /Waiting for the new output/);
  const idle = visibleText(await viewer(null));
  assert.match(idle, /^OUTPUT VIEWER A space for your next idea No output selected/);
});
