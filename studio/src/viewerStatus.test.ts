import test from 'node:test';
import assert from 'node:assert/strict';
import type { Artifact, Job } from './contracts.ts';
import { jobLine } from './jobStatus.ts';
import { CONSENT_HELP, TROUBLESHOOTING, type ConsentAction } from './jobFailure.ts';
import { renderComponent, visibleText } from './renderHarness.js';

const noop = () => {};
const client = { request: async () => ({}), thumbnail: async () => new Blob() };
const followed = (
  state: string,
  changes: Partial<Job> = {},
  now: number | null = 1042,
  own = true,
  consent: ConsentAction | null = null,
) => ({
  ...jobLine({ id: 'j', state, created_at: 1, submission: {}, ...changes } as Job, now),
  own,
  color: '#4ade80',
  consent,
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

const PANEL =
  ' Open the DGX Monarch panel and click "Load with slab residency" on the card for this model, then queue the render again. Headless: set DGXM_ALLOW_SLAB_RESCUE=1 on the driver and run again.';
const DESCRIPTOR = `<<<DGXM-CONSENT-V1 ${JSON.stringify({
  consent_id: '0123456789abcdef0123456789abcdef',
  context_fingerprint: 'c0ffee00c0ffee00',
  env_fallback: 'DGXM_ALLOW_SLAB_RESCUE',
  evidence: 'weights will be byte-verified against the checkpoint during load',
  family_hint: null,
  file_identity: 'size:1:mtime:2',
  human_reason:
    'flux/dev.safetensors needs 52.1 GiB under stock residency and only 40.2 GiB is available.',
  kind: 'rescue-slab',
  measured: { host: 'spark-a' },
  memo_context: {},
  panel_action: 'Load with slab residency',
  unet_name: 'flux/dev.safetensors',
  version: 1,
})} DGXM-CONSENT-END>>>`;
const offer = {
  started_at: 1,
  ended_at: 43,
  error: {
    code: 'SlabResidencyRescueOffer',
    message: `[dgxm:C guard=stock_load_preflight waivable=1] capacity rescue available for flux/dev.safetensors on spark-a.${PANEL} See docs/TROUBLESHOOTING.md #12.\n${DESCRIPTOR}`,
  },
};
const statusLine = async (job: unknown) =>
  renderComponent('./components/JobStatusLine.svelte', {
    job,
    onqueue: noop,
    onconsent: noop,
  });

test('a failed refusal shows its class, its first sentence and its troubleshooting entry', async () => {
  const body = await statusLine(
    followed('failed', {
      error: {
        code: 'ActorError',
        message:
          'A remote actor call has failed.\n [omitted] [dgxm:P] this topology cannot run on one host. Use single. See docs/TROUBLESHOOTING.md #16.',
      },
    }),
  );
  assert.equal(
    visibleText(body),
    'Failed · Refused, class P (physics): this topology cannot run on one host. Open Queue Troubleshooting #16 Failed. Refused, class P (physics): this topology cannot run on one host.',
  );
  assert.ok(body.includes(`href="${TROUBLESHOOTING}"`));
  assert.match(body, /target="_blank" rel="noopener noreferrer"/);
});

test('a failure that asks for consent offers Grant and retry beside its reason', async () => {
  const granted = await statusLine(
    followed('failed', offer, 1042, true, {
      label: 'Grant and retry',
      available: true,
      reason: '',
    }),
  );
  const text = visibleText(granted);
  assert.match(
    text,
    /^Failed after 0:42 · Refused, class C \(capacity-protective\): capacity rescue available for flux\/dev\.safetensors on spark-a\. Open Queue /,
  );
  assert.match(
    text,
    /Consent needed: flux\/dev\.safetensors needs 52\.1 GiB under stock residency and only 40\.2 GiB is available\. The ComfyUI panel's button for it is “Load with slab residency”\. Grant and retry Troubleshooting #12/,
  );
  assert.doesNotMatch(text, /Open the DGX Monarch panel/, 'the consent replaces the panel remedy');
  assert.doesNotMatch(text, /DGXM-CONSENT/);
  assert.equal((granted.match(/<button/g) ?? []).length, 2);
  assert.doesNotMatch(granted, /--allow-management/);
});

test('without management the consent control says why and links the Studio guide', async () => {
  const body = await statusLine(
    followed('failed', offer, 1042, true, {
      label: 'Review consent',
      available: false,
      reason:
        'This companion was started without --allow-management, so Studio cannot grant this consent.',
    }),
  );
  const text = visibleText(body);
  assert.match(
    text,
    /Review consent This companion was started without --allow-management, so Studio cannot grant this consent\. How to grant a consent from Studio/,
  );
  assert.ok(body.includes(`href="${CONSENT_HELP}"`));
  // With no consent operation at all there is no control, only the reason and the panel button.
  const none = visibleText(await statusLine(followed('failed', offer)));
  assert.match(none, /Consent needed: .* “Load with slab residency”\. Troubleshooting #12/);
  assert.doesNotMatch(none, /Grant and retry|Review consent/);
});

test('a refusal with a panel remedy and no consent block shows the remedy', async () => {
  const text = visibleText(
    await statusLine(
      followed('failed', {
        error: {
          code: 'StockLoadCapacityError',
          message: `[dgxm:C guard=stock_load_preflight waivable=1] the stock load does not fit.${PANEL} See docs/TROUBLESHOOTING.md #12.`,
        },
      }),
    ),
  );
  assert.match(
    text,
    new RegExp(
      `Open Queue ${PANEL.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} Troubleshooting #12`,
    ),
  );
  // A plain failure adds no second line.
  const plain = await statusLine(
    followed('failed', { error: { code: 'RuntimeError', message: 'Sampler stopped.' } }),
  );
  assert.doesNotMatch(plain, /job-failure/);
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
