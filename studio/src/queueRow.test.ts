import test from 'node:test';
import assert from 'node:assert/strict';
import type { Artifact, Job, OperationCapability } from './contracts.ts';
import { CONSENT_OPERATION } from './jobFailure.ts';
import { finalOutput } from './jobFollow.ts';
import { renderComponent, visibleText } from './renderHarness.js';

const noop = () => {};
const client = { request: async () => ({}), thumbnail: async () => new Blob() };
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
  measured: {},
  memo_context: {},
  panel_action: 'Load with slab residency',
  unet_name: 'flux/dev.safetensors',
  version: 1,
})} DGXM-CONSENT-END>>>`;
const job = (state: string, changes: Partial<Job> = {}) =>
  ({ id: `job-${state}`, state, created_at: 1760000000, submission: {}, ...changes }) as Job;
const refused = job('failed', {
  error: {
    code: 'SlabResidencyRescueOffer',
    message: `[dgxm:C guard=stock_load_preflight waivable=1] capacity rescue available for flux/dev.safetensors on spark-a. Open the DGX Monarch panel and click "Load with slab residency" on the card for this model, then queue the render again. Headless: set DGXM_ALLOW_SLAB_RESCUE=1 on the driver and run again. See docs/TROUBLESHOOTING.md #12.\n${DESCRIPTOR}`,
  },
});
const consentRow = (available: boolean): OperationCapability => ({
  id: CONSENT_OPERATION,
  label: 'consent · apply',
  available,
  reason: available
    ? undefined
    : 'Restart the companion with --allow-management to authorize management controls.',
  parameters: [],
});
const output = (id: string, changes: Partial<Artifact> = {}) =>
  ({
    id,
    job_id: 'job-failed',
    kind: 'image',
    filename: `${id}.png`,
    created_at: 2,
    ...changes,
  }) as Artifact;
const row = (props: Record<string, unknown>) =>
  renderComponent('./components/QueueRow.svelte', {
    client,
    busy: false,
    ontoggle: noop,
    jobAction: async () => {},
    onconsent: noop,
    ...props,
  });
// The summary's markup without Svelte's hydration comments.
const summaryOf = (body: string) =>
  body
    .slice(body.indexOf('<summary'), body.indexOf('</summary>') + '</summary>'.length)
    .replace(/<!--[\s\S]*?-->/g, '');

test('a failed row shows a short reason under its summary and the full failure inside', async () => {
  const body = await row({ job: refused, operations: [consentRow(true)], open: true });
  const summary = summaryOf(body);
  assert.match(
    summary,
    /<small class="job-reason">Refused, class C \(capacity-protective\): capacity rescue available for flux\/dev\.safetensors on spark-a\.<\/small><\/summary>/,
  );
  const details = visibleText(body.slice(body.indexOf('</summary>')));
  assert.match(
    details,
    /FAILURE Refusal class C, capacity-protective, guard stock_load_preflight , waivable Code SlabResidencyRescueOffer \[dgxm:C guard=stock_load_preflight waivable=1\] capacity rescue available/,
  );
  assert.match(details, /Troubleshooting #12/);
  assert.match(
    details,
    /Consent needed: flux\/dev\.safetensors needs 52\.1 GiB .* “Load with slab residency”\. Grant and retry Consent kind rescue-slab Model flux\/dev\.safetensors Consent id 0123456789abcdef0123456789abcdef/,
  );
  assert.doesNotMatch(
    details,
    /DGXM-CONSENT/,
    'the details show the fields, not the machine block',
  );
  assert.match(body, /Refresh full provenance/, 'the raw provenance stays in the open row');
});

test('without management the row offers Review consent and says why', async () => {
  const details = visibleText(await row({ job: refused, operations: [consentRow(false)] }));
  assert.match(
    details,
    /Review consent This companion was started without --allow-management, so Studio cannot grant this consent\./,
  );
  assert.doesNotMatch(details, /Grant and retry/);
});

test('only failed or unresolved rows explain their outcome', async () => {
  const unknown = job('unknown', {
    error: {
      code: 'driver_unavailable',
      message: 'Dispatch stopped before its outcome was recorded; reconcile before retrying.',
    },
  });
  assert.match(
    summaryOf(await row({ job: unknown })),
    /<small class="job-reason">driver_unavailable: Dispatch stopped before its outcome was recorded; reconcile before retrying\.<\/small>/,
  );
  for (const quiet of [
    job('succeeded'),
    job('running', { error: { code: 'late_ack', message: 'A late acknowledgement.' } }),
    job('failed'),
  ])
    assert.doesNotMatch(await row({ job: quiet }), /job-reason|failure-details/, quiet.state);
});

test('a row shows a 64 px thumbnail of its newest final output, and nothing without one', async () => {
  const withOutput = summaryOf(await row({ job: refused, output: output('final') }));
  assert.match(withOutput, /^<summary><span class="job-thumb">/);
  assert.match(visibleText(withOutput), /^▧ Loading preview failed/);
  const without = summaryOf(await row({ job: refused, output: null }));
  assert.doesNotMatch(without, /job-thumb|Loading preview/);
  const { readFile } = await import('node:fs/promises');
  const css = await readFile(new URL('./styles/pages.css', import.meta.url), 'utf8');
  assert.match(css, /\.job-thumb \{\n {2}width: 64px;\n {2}height: 64px;/);
});

test('Queue picks each row thumbnail by the newest final output rule', async () => {
  const active = job('unknown', { id: 'job-failed' });
  const artifacts = [
    output('older', { created_at: 1 }),
    output('newest', { created_at: 5 }),
    output('later-preview', { created_at: 9, intermediate: true }),
    output('other-job', { job_id: 'job-other', created_at: 10 }),
  ];
  // The viewer's rule: the job's newest final image or video, never a preview or another job's.
  assert.equal(finalOutput('job-failed', artifacts)?.id, 'newest');
  const queue = (snapshot: Record<string, unknown>) =>
    renderComponent('./views/Queue.svelte', {
      connection: { client, snapshot, capabilities: {} },
      busy: false,
      refresh: async () => {},
      fail: noop,
      jobAction: async () => {},
    });
  const body = await queue({ jobs: [active], artifacts });
  assert.match(body, /<img|Loading preview/);
  assert.equal((body.match(/class="job-thumb"/g) ?? []).length, 1);
  // The server render cannot fetch, so the placeholder names the kind; the source checks the id.
  const { readFile } = await import('node:fs/promises');
  const source = await readFile(new URL('./views/Queue.svelte', import.meta.url), 'utf8');
  assert.match(source, /output=\{finalOutput\(job\.id, artifacts\)\}/);
  assert.doesNotMatch(await queue({ jobs: [active] }), /job-thumb/);
  assert.match(
    await queue({ jobs: [active], artifacts: [], jobs_truncated: true }),
    /The live snapshot is limited/,
  );
});
