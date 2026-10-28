import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { Job, OperationCapability } from './contracts.ts';
import { CONSENT_OPERATION, consentRequest } from './jobFailure.ts';
import { componentFunctions } from './renderHarness.js';
import { keepsReview, type OperationRecovery } from './sessionRecovery.ts';

const DESCRIPTOR = `<<<DGXM-CONSENT-V1 ${JSON.stringify({
  consent_id: '0123456789abcdef0123456789abcdef',
  context_fingerprint: 'c0ffee00c0ffee00',
  env_fallback: 'DGXM_WAIVE_KNOWN_WRONG',
  evidence: 'every render under this waiver is stamped',
  family_hint: null,
  file_identity: 'size:1:mtime:2',
  human_reason: 'This combination produces measured wrong math, so the output will be stamped.',
  kind: 'waive-known-wrong:ring-pad',
  measured: { probe: 'ring_pad', world: 2 },
  memo_context: {},
  panel_action: 'Render under waiver (output stamped)',
  unet_name: 'chroma/base.safetensors',
  version: 1,
})} DGXM-CONSENT-END>>>`;
const asked = {
  id: 'job-9',
  state: 'failed',
  created_at: 1,
  submission: {},
  error: {
    code: 'UnsupportedModelError',
    message: `[dgxm:K guard=ring_pad waivable=1] pads.\n${DESCRIPTOR}`,
  },
} as Job;
const OPTIONS: OperationCapability[] = [
  {
    id: CONSENT_OPERATION,
    label: 'consent · apply',
    available: true,
    parameters: [
      { name: 'key', type: 'STRING', default: '' },
      { name: 'id', type: 'STRING', default: '' },
    ],
  },
  {
    id: 'reap.preview',
    label: 'reap · preview',
    available: true,
    parameters: [{ name: 'grace_s', type: 'INT', default: 30 }],
  },
];

// The harness compiles the rune module; App holds one opener and hands its methods out.
import './renderHarness.js';
const { OperationOpener } = await import('./operationOpener.svelte.ts');
const navigated: string[] = [];
const opener = new OperationOpener((view: string) => navigated.push(view));

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

test('Grant and retry opens Operations on consent.apply with the job context', () => {
  opener.consent(asked);
  assert.deepEqual(navigated, ['operations']);
  const first = opener.request!;
  assert.match(first.id, UUID, 'a random identity, so a reload never repeats one');
  assert.deepEqual(first, {
    id: first.id,
    action: 'consent.apply',
    parameters: { key: '', id: '' },
    consent: {
      jobId: 'job-9',
      kind: 'waive-known-wrong:ring-pad',
      consentId: '0123456789abcdef0123456789abcdef',
      unetName: 'chroma/base.safetensors',
      humanReason: 'This combination produces measured wrong math, so the output will be stamped.',
      panelAction: 'Render under waiver (output stamped)',
    },
  });
  // A job without a consent block, or none at all, opens nothing and keeps the last request.
  const { consent } = opener;
  consent({ ...asked, error: { message: 'plain.' } });
  consent(undefined);
  assert.deepEqual(navigated, ['operations']);
  assert.equal(opener.request, first);
  // The palette's operation rows still preselect an operation, each with a new identity.
  opener.operation('reap.preview');
  assert.deepEqual(opener.request, { id: opener.request?.id, action: 'reap.preview' });
  assert.match(opener.request?.id ?? '', UUID);
  const ids = new Set([first.id, opener.request?.id]);
  opener.consent(asked);
  ids.add(opener.request?.id);
  assert.equal(ids.size, 3, 'every request has its own identity');
  assert.deepEqual(navigated, ['operations', 'operations', 'operations']);
});

// Run the shipped Operations request handlers against plain state.
const operationsCode = await componentFunctions('./components/Operations.svelte', [
  'defaults',
  'open',
  'useCard',
]);
const operations = () =>
  new Function(
    'options',
    `'use strict';
    let action = '', parameters = {}, preview = { id: 'old-plan' }, confirmed = true,
      consent = null, appliedId = '';
    ${operationsCode}
    return { open, useCard, state: () => ({ action, parameters, preview, confirmed, consent, appliedId }) };`,
  )(OPTIONS);

test('Operations preselects the consent operation and fills its fields from the request', () => {
  const view = operations();
  const request = consentRequest(asked, 'request-7')!;
  view.open(request);
  assert.deepEqual(view.state(), {
    action: 'consent.apply',
    parameters: { key: '', id: '' },
    preview: null,
    confirmed: false,
    consent: request.consent,
    appliedId: 'request-7',
  });
  // Choosing the open operation again from the palette keeps its fields and review.
  view.useCard({ key: 'memo-key', id: 'p-0a1b2c3d4e5f' });
  assert.deepEqual(view.state().parameters, { key: 'memo-key', id: 'p-0a1b2c3d4e5f' });
  view.open({ id: 'request-8', action: 'consent.apply' });
  assert.deepEqual(view.state().parameters, { key: 'memo-key', id: 'p-0a1b2c3d4e5f' });
  assert.equal(view.state().appliedId, 'request-8');
  assert.equal(view.state().consent, request.consent);
  // Another consent request starts again from empty fields with its own job.
  view.open({
    ...consentRequest(asked, 'request-9')!,
    consent: { ...request.consent!, jobId: 'job-10' },
  });
  assert.deepEqual(view.state().parameters, { key: '', id: '' });
  assert.equal(view.state().consent?.jobId, 'job-10');
  // A palette choice of another operation takes its defaults and drops the consent context.
  view.open({ id: 'request-10', action: 'reap.preview' });
  assert.deepEqual(view.state().parameters, { grace_s: 30 });
  assert.equal(view.state().consent, null);
});

test('the status line, the viewer and Queue all reach the same consent handler', () => {
  const app = readFileSync(new URL('./App.svelte', import.meta.url), 'utf8');
  assert.equal(app.match(/onconsent=\{\(\) => opener\.consent\(tracker\.job\)\}/g)?.length, 2);
  assert.match(app, /onconsent=\{opener\.consent\}/);
  assert.match(app, /request=\{opener\.request\}/);
  assert.match(app, /onoperation=\{opener\.operation\}/);
  const viewer = readFileSync(new URL('./components/Viewer.svelte', import.meta.url), 'utf8');
  assert.match(viewer, /<JobStatusLine \{job\} \{onqueue\} \{onconsent\} \/>/);
  const operations = readFileSync(
    new URL('./components/Operations.svelte', import.meta.url),
    'utf8',
  );
  assert.match(operations, /if \(saved && keepsReview\(saved, request, options\)\) \{/);
  assert.match(operations, /pending, request: appliedId \}/);
  assert.match(
    operations,
    /if \(request && request\.id !== appliedId && !pending\) open\(request\);/,
  );
});

test('a saved review comes back only for the request that made it', () => {
  const consentAsk = consentRequest(asked, 'request-a')!;
  const saved = {
    action: 'consent.apply',
    parameters: { key: '', id: '' },
    preview: { id: 'plan', digest: 'd', action: 'consent.apply' },
    requestKey: 'kept',
    pending: false,
    request: 'request-a',
  } as unknown as OperationRecovery;
  // Leaving Operations and coming back for the same request keeps its plan.
  assert.equal(keepsReview(saved, consentAsk, OPTIONS), true);
  // After a reload the opener issues a new identity, so another job's request gets fresh fields.
  assert.equal(keepsReview(saved, { ...consentAsk, id: 'request-b' }, OPTIONS), false);
  // The palette asking again for the reviewed operation, or no request at all, keeps it.
  assert.equal(keepsReview(saved, { id: 'request-c', action: 'consent.apply' }, OPTIONS), true);
  assert.equal(keepsReview(saved, null, OPTIONS), true);
  assert.equal(keepsReview(saved, { id: 'request-d', action: 'reap.preview' }, OPTIONS), false);
  // A pending outcome always comes back; an operation the companion no longer lists does not.
  assert.equal(
    keepsReview({ ...saved, pending: true }, { ...consentAsk, id: 'request-e' }, []),
    true,
  );
  assert.equal(keepsReview(saved, null, []), false);
});
