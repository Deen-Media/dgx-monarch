import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import type { Job } from './contracts.ts';
import { emptyQueueText, keptPage, settledJobs } from './queueRows.ts';
import { renderComponent, visibleText } from './renderHarness.js';

const job = (id: string, state: string) => ({ id, state, created_at: 1, submission: {} }) as Job;
const noop = async () => {};

test('a job that settles after the history load triggers a history reload', () => {
  let observed = settledJobs(null, [job('old', 'succeeded')]);
  assert.deepEqual(observed.settled, [], 'the mount history load covers the first snapshot');
  observed = settledJobs(observed.next, [job('retry', 'running'), job('old', 'succeeded')]);
  assert.deepEqual(observed.settled, []);
  observed = settledJobs(observed.next, [job('retry', 'running'), job('old', 'succeeded')]);
  assert.deepEqual(observed.settled, [], 'identical polls do not reload');
  observed = settledJobs(observed.next, [job('retry', 'succeeded'), job('old', 'succeeded')]);
  assert.deepEqual(observed.settled, ['retry']);
  observed = settledJobs(observed.next, [
    job('fast', 'failed'),
    job('retry', 'succeeded'),
    job('old', 'succeeded'),
  ]);
  assert.deepEqual(observed.settled, ['fast'], 'a job created and settled between polls counts');
});

test('an automatic reload keeps the history page when it still exists', () => {
  assert.equal(keptPage(1, 100), 1);
  assert.equal(keptPage(4, 100), 2);
  assert.equal(keptPage(3, 0), 0);
});

test('the empty state says when filters, not an empty history, hide every job', () => {
  assert.equal(emptyQueueText('', '', false).title, 'No recorded jobs yet.');
  assert.equal(emptyQueueText('alpha', '', false).title, 'No jobs match these filters.');
  assert.equal(emptyQueueText('', 'failed', false).title, 'No jobs match these filters.');
  assert.equal(emptyQueueText('', '', true).title, 'Loading job history…');
});

test('Queue renders snapshot jobs and an empty state that does not claim no jobs exist', async () => {
  const props = (jobs: Job[]) => ({
    connection: { client: { request: noop }, snapshot: { jobs }, capabilities: {} },
    busy: false,
    refresh: noop,
    fail: () => {},
    jobAction: noop,
  });
  const running = visibleText(
    await renderComponent('./views/Queue.svelte', props([job('j-run', 'running')])),
  );
  assert.match(running, /running workflow j-run/);
  assert.match(running, /1 active or unresolved jobs/);
  assert.doesNotMatch(running, /No recorded jobs yet/);
  const empty = visibleText(await renderComponent('./views/Queue.svelte', props([])));
  assert.match(empty, /Loading job history… Active jobs appear as they arrive\./);
  assert.doesNotMatch(empty, /No recorded jobs yet/, 'opening Queue never claims an empty history');
});

test('Queue reloads history in place when the snapshot reports a settled job', async () => {
  const source = await readFile(new URL('./views/Queue.svelte', import.meta.url), 'utf8');
  assert.match(source, /settledJobs\(observed, jobs\)/);
  assert.match(source, /if \(settled\.length\) untrack\(\(\) => void load\(true, true\)\)/);
  assert.match(source, /page = keepPosition \? keptPage\(page, history\.length\) : 0/);
});

test('JobProvenance mounts with the keyed record and no loaded provenance yet', async () => {
  const body = visibleText(
    await renderComponent('./components/JobProvenance.svelte', {
      client: { request: noop },
      summary: job('j-1', 'running'),
    }),
  );
  assert.equal(body, 'Refresh full provenance');
});
