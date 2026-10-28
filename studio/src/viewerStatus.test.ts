import test from 'node:test';
import assert from 'node:assert/strict';
import type { Artifact, Job } from './contracts.ts';
import { jobLine } from './jobStatus.ts';
import { renderComponent, visibleText } from './renderHarness.js';

const noop = () => {};
const client = { request: async () => ({}), thumbnail: async () => new Blob() };
const followed = (
  state: string,
  changes: Partial<Job> = {},
  now: number | null = 1042,
  own = true,
) => ({
  ...jobLine({ id: 'j', state, created_at: 1, submission: {}, ...changes } as Job, now),
  own,
  color: '#4ade80',
});
const artifact = (id: string, changes: Partial<Artifact> = {}) =>
  ({
    id,
    job_id: 'j',
    kind: 'image',
    filename: `${id}.png`,
    created_at: 1,
    ...changes,
  }) as Artifact;

function workspace(props: Record<string, unknown>) {
  return renderComponent('./views/WorkspaceOutput.svelte', {
    client,
    selected: null,
    comparison: null,
    artifacts: [],
    telemetry: {},
    history: [],
    driverState: 'connected',
    connectionState: 'connected',
    activeCount: 0,
    showTelemetry: true,
    onselect: noop,
    oncompare: noop,
    oncollapse: noop,
    navigate: noop,
    reuse: async () => {},
    ...props,
  });
}

test('the viewer status line replaces the active job count', async () => {
  const running = followed('running', {
    running_observed_at: 1000,
    progress: { value: 5, max: 20, node: '3' },
  });
  const body = await renderComponent('./components/Viewer.svelte', {
    job: running,
    working: true,
    client,
    onreuse: async () => {},
    onselect: noop,
    onqueue: noop,
  });
  const text = visibleText(body);
  assert.match(text, /Running 0:42 · step 5 of 20 on node 3/);
  assert.doesNotMatch(text, /active job/);
  assert.match(body, /role="progressbar"[^>]*aria-valuemax="20"[^>]*aria-valuenow="5"/);
  assert.match(body, /style="width: 25%;?"/);
  assert.match(body, /<span class="sr-only" role="status">Running<\/span>/);
  assert.match(body, /class="viewer-frame working"/);
});

test('a running job without a step count draws a bar with no value', async () => {
  const body = await renderComponent('./components/JobStatusLine.svelte', {
    job: followed('running', { running_observed_at: 1000 }),
  });
  assert.match(body, /class="job-progress indeterminate"/);
  assert.doesNotMatch(body, /aria-valuenow/);
});

test('a failed job names its code and first sentence and links to Queue', async () => {
  const failed = followed('failed', {
    started_at: 1,
    ended_at: 43,
    error: { code: 'RuntimeError', message: 'Sampler stopped. More detail follows.' },
  });
  const text = visibleText(
    await renderComponent('./components/JobStatusLine.svelte', { job: failed, onqueue: noop }),
  );
  assert.equal(
    text,
    'Failed after 0:42 · RuntimeError: Sampler stopped. Open Queue Failed. RuntimeError: Sampler stopped.',
  );
  const unknown = visibleText(
    await renderComponent('./components/JobStatusLine.svelte', {
      job: followed('unknown'),
      onqueue: noop,
    }),
  );
  assert.match(unknown, /^Outcome unknown Reconcile in Queue/);
});

test('with no followed job the viewer shows the selected kind and an empty status region', async () => {
  const body = await renderComponent('./components/Viewer.svelte', {
    artifact: artifact('a'),
    client,
    onreuse: async () => {},
    onselect: noop,
  });
  assert.match(visibleText(body), /^OUTPUT VIEWER a\.png image /);
  assert.match(body, /<span class="sr-only" role="status"><\/span>/);
  assert.doesNotMatch(body, /class="viewer-frame working"/);
});

test('the recent strip previews job outputs only', async () => {
  const body = await workspace({
    artifacts: [
      artifact('upload', { job_id: null as never, kind: 'input' as never }),
      artifact('mask', { job_id: null as never, kind: 'mask' as never }),
      artifact('live', { source: 'live_preview', intermediate: true }),
      artifact('final'),
      artifact('temp', { intermediate: true }),
    ],
  });
  const strip = visibleText(
    body.slice(body.indexOf('RECENT OUTPUTS'), body.indexOf('Spark telemetry')),
  );
  assert.equal(
    strip,
    'RECENT OUTPUTS View gallery → ▧ Loading preview final.png ▧ Loading preview Preview · temp.png',
  );
  assert.equal((body.match(/class="recent-thumb"/g) ?? []).length, 2);
  const empty = visibleText(
    await workspace({ artifacts: [artifact('upload', { job_id: null as never })] }),
  );
  assert.match(empty, /Completed outputs will appear here\./);
});

test('telemetry shows this tab job and keeps the raw sample closed', async () => {
  const body = await workspace({
    activeCount: 2,
    job: followed('succeeded', { started_at: 10, ended_at: 83 }),
  });
  const text = visibleText(body);
  assert.match(
    text,
    /Active or unresolved jobs 2 Driver connected This tab's job Finished in 1:13/,
  );
  assert.doesNotMatch(text, /render/i, 'the row never names the driver render block');
  assert.match(body, /<details><summary>Authoritative host sample<\/summary>/);
  const fallback = visibleText(await workspace({ job: followed('queued', {}, null, false) }));
  assert.match(fallback, /Newest active job Queued/);
  assert.match(visibleText(await workspace({})), /This tab's job None/);
});

test('strip tiles keep their thumbnails across snapshot polls', async () => {
  const { readFile } = await import('node:fs/promises');
  const strip = await readFile(new URL('./views/WorkspaceOutput.svelte', import.meta.url), 'utf8');
  assert.match(strip, /\{#each outputs as artifact \(artifact\.id\)\}/);
  // Each poll replaces every artifact object; the fetch keys on values, not on the object.
  const thumbnail = await readFile(
    new URL('./components/Thumbnail.svelte', import.meta.url),
    'utf8',
  );
  assert.match(thumbnail, /const id = \$derived\(artifact\.id\);/);
  assert.match(thumbnail, /\.thumbnail\(id, controller\.signal\)/);
});
