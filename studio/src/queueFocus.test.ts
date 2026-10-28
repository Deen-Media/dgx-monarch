import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { Job } from './contracts.ts';
import { QueueFocus, focusPlace, queueFocus, queueRowId } from './queueFocus.ts';
import { componentFunctions, renderComponent } from './renderHarness.js';

const job = (id: string, state: string) => ({ id, state, created_at: 1, submission: {} }) as Job;
const noop = () => {};
const rows = (count: number, state = 'succeeded') =>
  Array.from({ length: count }, (_, index) => job(`old-${index}`, state));

test('a Queue request is taken once, so a later visit opens on no row', () => {
  const focus = new QueueFocus();
  assert.equal(focus.take(), null);
  focus.request('job-1');
  focus.request('job-2');
  assert.equal(focus.take(), 'job-2', 'the newest request wins');
  assert.equal(focus.take(), null);
  assert.equal(queueRowId('job-2'), 'queue-job-job-2');
});

test('Queue finds a requested row among the active rows or on its history page', () => {
  const running = job('run', 'running');
  const history = rows(90);
  assert.deepEqual(focusPlace('run', [running], history), { job: running, page: null });
  assert.deepEqual(focusPlace('old-0', [running], history), { job: history[0], page: 0 });
  assert.deepEqual(focusPlace('old-39', [], history), { job: history[39], page: 0 });
  assert.deepEqual(focusPlace('old-85', [], history), { job: history[85], page: 2 });
  assert.equal(focusPlace('gone', [running], history), null);
});

test('the status line link asks for its job, then opens Queue', async () => {
  const code = await componentFunctions('./components/JobStatusLine.svelte', ['openQueue']);
  const opened: string[] = [];
  const link = (followed: { id: string } | null) =>
    new Function('job', 'onqueue', 'queueFocus', `'use strict'; ${code} return openQueue;`)(
      followed,
      () => opened.push('queue'),
      queueFocus,
    );
  link({ id: 'job-9' })();
  assert.deepEqual(opened, ['queue']);
  assert.equal(queueFocus.take(), 'job-9');
  link(null)();
  assert.equal(queueFocus.take(), null, 'no job, no request');
});

// Run the shipped Queue handlers against plain state; a fake companion lists `history`.
const queueCode = await componentFunctions('./views/Queue.svelte', ['load', 'reveal']);
type QueueState = { page: number; selectedJob: Job | null; pinned: Job | null; focus: unknown };
function queue(options: {
  focus: string;
  active?: Job[];
  history?: Job[];
  snapshot?: Job[];
  summary?: Job | Error;
}) {
  return new Function(
    'options',
    'focusPlace',
    'queueRowId',
    `'use strict';
    const asked = [], failures = [], scrolled = [];
    let focus = options.focus, pinned = null, selectedJob = null, revision = 0, history = [],
      cursor = null, loading = true, query = '', status = '', page = 0, terminal = [];
    const active = options.active ?? [], jobs = [...active, ...(options.snapshot ?? [])];
    const keptPage = (value) => value;
    const fail = (cause) => failures.push(cause.message);
    const tick = async () => {};
    const document = { getElementById: (id) => ({ scrollIntoView: (how) => scrolled.push([id, how]) }) };
    const jobSummary = async (_client, id) => {
      asked.push('/jobs?job_id=' + id);
      if (options.summary instanceof Error) throw options.summary;
      return options.summary;
    };
    const client = { request: async () => {
      terminal = history = options.history ?? [];
      return { items: terminal, next_cursor: null };
    } };
    ${queueCode}
    const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
    return {
      load: async (reset) => (await load(reset), await settle()),
      state: () => ({ page, selectedJob, pinned, focus, asked, failures, scrolled }),
    };`,
  )(options, focusPlace, queueRowId) as {
    load: (reset: boolean) => Promise<void>;
    state: () => QueueState & { asked: string[]; failures: string[]; scrolled: unknown[] };
  };
}

test('Queue opens an active row once the first history load settles, and only once', async () => {
  const unknown = job('lost', 'unknown');
  const view = queue({ focus: 'lost', active: [unknown], history: rows(3) });
  await view.load(true);
  const state = view.state();
  assert.equal(state.selectedJob, unknown);
  assert.equal(state.pinned, null);
  assert.equal(state.focus, null, 'the request is used up');
  assert.deepEqual(state.asked, []);
  assert.deepEqual(state.scrolled, [['queue-job-lost', { block: 'nearest' }]]);
  await view.load(true);
  assert.equal(view.state().scrolled.length, 1, 'a later load opens nothing');
});

test('Queue turns to the history page that holds the requested row', async () => {
  const history = rows(90, 'failed');
  const view = queue({ focus: 'old-50', history });
  await view.load(true);
  const { page, selectedJob, pinned, asked } = view.state();
  assert.equal(page, 1);
  assert.equal(selectedJob, history[50]);
  assert.equal(pinned, null);
  assert.deepEqual(asked, []);
});

test('Queue pins a requested row its history lacks, read by its identity when needed', async () => {
  // The snapshot still lists the job, so no request is needed.
  const failed = job('failed-1', 'failed');
  const listed = queue({ focus: 'failed-1', history: rows(100), snapshot: [failed] });
  await listed.load(true);
  assert.equal(listed.state().pinned, failed);
  assert.equal(listed.state().selectedJob, failed);
  assert.deepEqual(listed.state().asked, []);
  // Neither the snapshot nor the loaded history holds it: read its summary.
  const summary = job('older', 'failed');
  const read = queue({ focus: 'older', history: rows(100), summary });
  await read.load(true);
  assert.deepEqual(read.state().asked, ['/jobs?job_id=older']);
  assert.equal(read.state().pinned, summary);
  assert.equal(read.state().selectedJob, summary);
  assert.deepEqual(read.state().scrolled, [['queue-job-older', { block: 'nearest' }]]);
  // A failed read reports the error and opens nothing.
  const lost = queue({ focus: 'older', summary: new Error('The companion returned no summary.') });
  await lost.load(true);
  assert.deepEqual(lost.state().failures, ['The companion returned no summary.']);
  assert.equal(lost.state().pinned, null);
  assert.equal(lost.state().selectedJob, null);
  assert.deepEqual(lost.state().scrolled, []);
});

test('Queue takes the request as it mounts and shows a pinned row above the history page', async () => {
  const source = readFileSync(new URL('./views/Queue.svelte', import.meta.url), 'utf8');
  assert.match(source, /let focus = queueFocus\.take\(\),/);
  assert.match(source, /\.filter\(\(job\) => job\.id !== pinned\?\.id && !active\.some/);
  assert.match(source, /\[\.\.\.active, \.\.\.top, \.\.\.terminal\.slice\(page \* 40/);
  const body = await renderComponent('./components/QueueRow.svelte', {
    client: { request: async () => ({}) },
    job: job('job-3', 'failed'),
    busy: false,
    ontoggle: noop,
    jobAction: async () => {},
  });
  assert.match(body, /<details class="record job-row" id="queue-job-job-3"/);
});
