import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { Artifact, Job } from './contracts.ts';
// The harness compiles the rune module; outside a render its deriveds recompute on each read.
import './renderHarness.js';

const { JobTracker } = await import('./jobTracker.svelte.ts');
const presentation = JSON.parse(
  readFileSync(
    new URL('../../src/dgx_monarch/studio/data/presentation.json', import.meta.url),
    'utf8',
  ),
);
const job = (id: string, state: string, changes: Partial<Job> = {}) =>
  ({ id, state, created_at: 1, submission: {}, ...changes }) as Job;
const output = (id: string, job_id: string) =>
  ({ id, job_id, kind: 'image', filename: `${id}.png`, created_at: 5 }) as Artifact;

function setup(
  route: (path: string) => Promise<unknown> = async () => ({ items: [] }),
  operations: unknown[] = [],
) {
  const requests: string[] = [];
  const connection = {
    snapshot: { jobs: [] as Job[], artifacts: [] as Artifact[], server_time: 1000 as number },
    capabilities: { presentation, operations },
    client: {
      request: (path: string) => {
        requests.push(path);
        return route(path);
      },
    },
  };
  const submission = {
    pending: null as { key: string; preview_id: string } | null,
    createdKey: '',
  };
  const tracker = new JobTracker(connection as never, submission as never);
  let polls = 0;
  /** Each poll is a new snapshot unless it names the server time it repeats. */
  const poll = (jobs: Job[], artifacts: Artifact[] = [], server_time = 1000 + ++polls) => {
    connection.snapshot = { jobs, artifacts, server_time };
    tracker.observe();
  };
  return { tracker, submission, poll, requests };
}

test('the viewer shows the result of the job this tab submitted, without touching a choice', () => {
  const { tracker, poll } = setup();
  poll([job('mine', 'submitting')]);
  tracker.own('mine');
  poll([job('mine', 'running', { started_at: 3, running_observed_at: 990 })], [], 1032.5);
  assert.equal(tracker.status?.headline, 'Running 0:42');
  assert.equal(tracker.status?.own, true);
  assert.equal(tracker.status?.color, '#4ade80');
  assert.equal(tracker.working, true);
  poll([job('mine', 'succeeded', { started_at: 990, ended_at: 1063 })], [output('result', 'mine')]);
  assert.equal(tracker.selected?.id, 'result');
  assert.equal(tracker.shown?.id, 'result');
  assert.equal(tracker.status?.headline, 'Finished in 1:13');
  assert.equal(tracker.working, false);
  // The user picks something else; a later poll never takes the viewer back.
  tracker.selected = output('chosen', 'old');
  poll([job('mine', 'succeeded')], [output('result', 'mine')]);
  assert.equal(tracker.selected?.id, 'chosen');
});

test('a result the user replaced or a comparison opened after submitting stays', () => {
  const picked = setup();
  picked.tracker.own('mine');
  picked.tracker.selected = output('chosen', 'old');
  picked.poll([job('mine', 'succeeded')], [output('result', 'mine')]);
  assert.equal(picked.tracker.selected?.id, 'chosen');
  const comparing = setup();
  comparing.tracker.own('mine');
  comparing.tracker.comparison = output('reference', 'old');
  comparing.poll([job('mine', 'succeeded')], [output('result', 'mine')]);
  assert.equal(comparing.tracker.selected, null);
  // The next submission starts over.
  comparing.tracker.comparison = null;
  comparing.tracker.own('next');
  comparing.poll([job('next', 'succeeded')], [output('next-result', 'next')]);
  assert.equal(comparing.tracker.selected?.id, 'next-result');
});

test('a pending request identity is followed once its job appears', () => {
  const { tracker, submission, poll } = setup();
  submission.pending = { key: 'key', preview_id: 'p' };
  poll([]);
  submission.pending = null;
  poll([job('mine', 'queued', { idempotency_key: 'key' })]);
  assert.deepEqual(tracker.follow, {
    id: 'mine',
    own: true,
    settled: false,
    picked: false,
    lookups: 0,
  });
  assert.equal(tracker.status?.headline, 'Queued');
  assert.equal(tracker.status?.id, 'mine', "the status line's Queue link opens this row");
});

test('a reload finds an ended job by its restored identity and shows no old result', () => {
  const { tracker, submission, poll } = setup();
  submission.pending = { key: 'key', preview_id: 'p' };
  poll([job('old', 'succeeded', { idempotency_key: 'key' })], [output('old-result', 'old')]);
  assert.equal(tracker.follow.id, 'old');
  assert.equal(tracker.selected, null);
  assert.equal(tracker.status?.headline, 'Finished');
});

test('a tab that submitted nothing follows the newest active job, unknown outcomes included', () => {
  const { tracker, poll } = setup();
  poll([job('done', 'succeeded')], [output('done-result', 'done')]);
  assert.equal(tracker.status, null, 'settled jobs at connect are not followed');
  poll([job('lost', 'unknown'), job('done', 'succeeded')]);
  assert.equal(tracker.status?.headline, 'Outcome unknown');
  assert.equal(tracker.status?.own, false);
  assert.equal(tracker.status?.color, '#f87171');
  assert.equal(tracker.working, false, 'an unresolved job does not animate the activity border');
});

test('a lost submit reply still shows the result of the job this page sent', () => {
  const { tracker, submission, poll } = setup();
  // submit() set the identity, then its reply was lost, so own() never ran.
  submission.createdKey = 'key';
  submission.pending = { key: 'key', preview_id: 'p' };
  poll([]);
  // The job finished before the tracker first saw it.
  poll([job('mine', 'succeeded', { idempotency_key: 'key' })], [output('result', 'mine')]);
  assert.equal(tracker.follow.id, 'mine');
  assert.equal(tracker.selected?.id, 'result');
});

test('own() keeps a choice made while the submit reply was on its way', () => {
  const { tracker, submission, poll } = setup();
  submission.createdKey = 'key';
  submission.pending = { key: 'key', preview_id: 'p' };
  // A poll adopts the running job before submit() returns.
  poll([job('mine', 'running', { idempotency_key: 'key' })]);
  assert.equal(tracker.follow.id, 'mine');
  tracker.selected = output('chosen', 'old');
  submission.pending = null;
  tracker.own('mine');
  assert.equal(tracker.follow.picked, true);
  poll([job('mine', 'succeeded')], [output('result', 'mine')]);
  assert.equal(tracker.selected?.id, 'chosen');
});

test('the artifacts route supplies a result the snapshot left out', async () => {
  const { tracker, poll, requests } = setup(async () => ({ items: [output('result', 'mine')] }));
  tracker.own('mine');
  const crowded = Array.from({ length: 200 }, (_, index) => output(`temp-${index}`, 'other'));
  poll([job('mine', 'succeeded')], crowded);
  assert.equal(tracker.status?.headline, 'Finished');
  assert.equal(tracker.follow.settled, false);
  await tracker.lookup;
  assert.deepEqual(requests, ['/artifacts?job_id=mine&intermediate=false&sort=newest&limit=200']);
  assert.equal(tracker.selected?.id, 'result');
  assert.equal(tracker.follow.settled, true);
  poll([job('mine', 'succeeded')], crowded);
  assert.equal(requests.length, 1, 'a settled follow asks nothing more');
});

test('the lookup stops after a bounded number of answers without the output', async () => {
  let fail = false;
  const { tracker, poll, requests } = setup(async () => {
    fail = !fail;
    if (fail) throw new Error('Network down');
    return { items: [] };
  });
  tracker.own('mine');
  const done = [job('mine', 'succeeded')];
  for (let round = 0; round < 6; round++) {
    poll(done, [], 2000 + round);
    // A repeated snapshot asks nothing new.
    poll(done, [], 2000 + round);
    await tracker.lookup;
  }
  assert.equal(requests.length, 3);
  assert.equal(tracker.follow.settled, true);
  assert.equal(tracker.follow.lookups, 3);
  assert.equal(tracker.selected, null);
});

test("the followed job's consent control follows the consent operation's availability", () => {
  const descriptor = `<<<DGXM-CONSENT-V1 ${JSON.stringify({
    consent_id: '0123456789abcdef0123456789abcdef',
    context_fingerprint: 'c0ffee00c0ffee00',
    env_fallback: 'DGXM_ALLOW_SLAB_RESCUE',
    evidence: 'weights will be byte-verified against the checkpoint during load',
    file_identity: 'size:1:mtime:2',
    human_reason: 'The stock load does not fit; slab residency does.',
    kind: 'rescue-slab',
    measured: {},
    panel_action: 'Load with slab residency',
    unet_name: 'flux/dev.safetensors',
    version: 1,
  })} DGXM-CONSENT-END>>>`;
  const refused = job('mine', 'failed', {
    error: { code: 'SlabResidencyRescueOffer', message: `[dgxm:C] no fit.\n${descriptor}` },
  });
  const row = (available: boolean) => ({
    id: 'consent.apply',
    label: 'consent · apply',
    available,
    reason: available ? undefined : 'Restart the companion with --allow-management.',
    parameters: [],
  });
  const control = (operations: unknown[], failed: Job = refused) => {
    const { tracker, poll } = setup(undefined, operations);
    tracker.own('mine');
    poll([failed]);
    return tracker.status?.consent;
  };
  assert.deepEqual(control([row(true)]), { label: 'Grant and retry', available: true, reason: '' });
  assert.equal(control([row(false)])?.label, 'Review consent');
  assert.equal(control([]), null, 'no consent operation, no control');
  assert.equal(control([row(true)], job('mine', 'failed', { error: { message: 'plain.' } })), null);
  const { tracker, poll } = setup(undefined, [row(true)]);
  tracker.own('mine');
  poll([refused]);
  assert.equal(tracker.status?.detail, 'Refused, class C (capacity-protective): no fit.');
  assert.equal(tracker.status?.failure?.consent?.panelAction, 'Load with slab residency');
});
