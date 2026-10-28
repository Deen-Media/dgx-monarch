import test from 'node:test';
import assert from 'node:assert/strict';
import type { OperationCapability, OperationRecord } from './contracts.ts';
import {
  CONSENT_OPERATION,
  DESCRIPTOR_BEGIN,
  DESCRIPTOR_END,
  TEXT_LIMIT,
  consentAction,
  consentMatch,
  consentRequest,
  jobFailure,
} from './jobFailure.ts';

// Shapes dgx_monarch.refusal.refusal() and consent_descriptor.encode() write; the Python suite
// (tests/test_studio_failure_parse.py) runs this module on strings the real builders make.
const PANEL =
  ' Open the DGX Monarch panel and click "Load with slab residency" on the card for this model, then queue the render again. Headless: set DGXM_ALLOW_SLAB_RESCUE=1 on the driver and run again.';
const RESCUE = {
  consent_id: '0123456789abcdef0123456789abcdef',
  context_fingerprint: 'c0ffee00c0ffee00',
  env_fallback: 'DGXM_ALLOW_SLAB_RESCUE',
  evidence: 'weights will be byte-verified against the checkpoint during load',
  family_hint: 'flux',
  file_identity: 'size:35000000000:mtime:1760000000',
  human_reason:
    'flux/dev.safetensors needs 52.1 GiB of unified memory under stock residency (a 33.4 GiB file plus the host copy it is placed from) and only 40.2 GiB is available. Zero-copy slab residency fits it.',
  kind: 'rescue-slab',
  measured: { checkpoint_bytes: 35000000000, host: 'spark-a', mem_available_bytes: 43000000000 },
  memo_context: { combo_key: 'flux/dev.safetensors' },
  panel_action: 'Load with slab residency',
  unet_name: 'flux/dev.safetensors',
  version: 1,
};
const block = (body: unknown) => `${DESCRIPTOR_BEGIN}${JSON.stringify(body)}${DESCRIPTOR_END}`;
const OFFER =
  'capacity rescue available for flux/dev.safetensors on spark-a: stock residency needs 52.1 GiB of unified memory for a 33.4 GiB file and only 40.2 GiB is available, so the stock load would be killed out of memory.';
const rescue = (descriptor: unknown = RESCUE) =>
  `[dgxm:C guard=stock_load_preflight waivable=1] ${OFFER} Zero-copy slab residency fits this checkpoint.` +
  `${PANEL} See docs/TROUBLESHOOTING.md #12.\n[dgxm:measured {"checkpoint_bytes":35000000000}]\n${block(descriptor)}`;

test('a tagged refusal yields its class, first sentence and troubleshooting entry', () => {
  const failure = jobFailure({
    code: 'dgx_monarch.mesh_safety.MeshAttachError',
    message:
      'A remote actor call has failed.\n Traceback of where the remote call failed (most recent call last):\n  File "worker.py", line 3, in load\nMeshAttachError: [dgxm:P] this topology cannot run on one host. Use single instead. See docs/TROUBLESHOOTING.md #16.',
  });
  assert.ok(failure);
  assert.equal(failure.refusalClass, 'P');
  assert.equal(failure.guard, null);
  assert.equal(failure.waivable, false);
  assert.equal(failure.headline, 'this topology cannot run on one host.');
  assert.equal(
    failure.summary,
    'Refused, class P (physics): this topology cannot run on one host.',
  );
  assert.equal(failure.remedy, '', 'a troubleshooting pointer alone is not a panel remedy');
  assert.equal(failure.troubleshooting, 16);
  assert.equal(failure.consent, null);
  assert.match(failure.text, /^A remote actor call has failed\.\n/);
});

test('a consent descriptor yields the reason, the panel button and the request identity', () => {
  const failure = jobFailure({ code: 'SlabResidencyRescueOffer', message: rescue() });
  assert.ok(failure);
  assert.equal(failure.refusalClass, 'C');
  assert.equal(failure.guard, 'stock_load_preflight');
  assert.equal(failure.waivable, true);
  // The first sentence runs past the bound, so it ends with an ellipsis at 200 characters.
  assert.equal(failure.headline, `${OFFER.slice(0, 199).trimEnd()}…`);
  assert.equal(failure.headline.length, 200);
  assert.equal(failure.remedy, PANEL.trim());
  assert.equal(failure.troubleshooting, 12);
  assert.deepEqual(failure.consent, {
    kind: 'rescue-slab',
    consentId: RESCUE.consent_id,
    unetName: 'flux/dev.safetensors',
    humanReason: RESCUE.human_reason,
    panelAction: 'Load with slab residency',
  });
  assert.doesNotMatch(failure.text, /DGXM-CONSENT/, 'the shown text leaves the machine block out');
  assert.match(failure.text, /\[dgxm:measured /);
  // A class K waiver names its guard in the kind and its env value is the guard's short name.
  const waiver = jobFailure({
    message:
      '[dgxm:K guard=ring_pad waivable=1] this sequence length needs divisibility padding, and the exact pad exclusion is ulysses-only. Use a uly* or single topology (docs/ADAPTERS.md). Open the DGX Monarch panel and click "Render under waiver (output stamped)" on the card for this model, then queue the render again. Headless: set DGXM_WAIVE_KNOWN_WRONG=ring-pad on the driver and run again. See docs/TROUBLESHOOTING.md #21.\n' +
      block({
        ...RESCUE,
        kind: 'waive-known-wrong:ring-pad',
        panel_action: 'Render under waiver (output stamped)',
        env_fallback: 'DGXM_WAIVE_KNOWN_WRONG',
        measured: { probe: 'ring_pad', measured: '', topology: 'ring2', world: 2 },
      }),
  });
  assert.equal(waiver?.refusalClass, 'K');
  assert.equal(waiver?.consent?.kind, 'waive-known-wrong:ring-pad');
  assert.equal(waiver?.consent?.panelAction, 'Render under waiver (output stamped)');
  assert.match(
    waiver?.remedy ?? '',
    /^Open the DGX Monarch panel .*ring-pad on the driver and run again\.$/,
  );
  assert.equal(waiver?.troubleshooting, 21);
});

test('a plain message is shown whole, with its code and first sentence', () => {
  const failure = jobFailure({
    code: 'OutOfMemoryError',
    message: 'CUDA out of memory. Tried to allocate 2.00 GiB on device 0.',
  });
  assert.deepEqual(failure, {
    code: 'OutOfMemoryError',
    refusalClass: null,
    guard: null,
    waivable: false,
    headline: 'CUDA out of memory.',
    summary: 'OutOfMemoryError: CUDA out of memory.',
    text: 'CUDA out of memory. Tried to allocate 2.00 GiB on device 0.',
    remedy: '',
    troubleshooting: null,
    consent: null,
  });
  assert.equal(jobFailure({ code: 'prompt_rejected' })?.summary, 'prompt_rejected');
  assert.equal(jobFailure(null), null);
  assert.equal(jobFailure({}), null);
  assert.equal(jobFailure(undefined), null);
});

test('an over-long message is bounded, never cut into a guessed cause', () => {
  const message = `Error(s) in loading state_dict:\n${'\tsize mismatch for a weight. '.repeat(400)}`;
  const failure = jobFailure({ code: 'RuntimeError', message });
  assert.ok(failure);
  assert.equal(failure.text.length, TEXT_LIMIT);
  assert.ok(failure.text.endsWith('…'));
  assert.equal(
    failure.text.slice(0, -1),
    message
      .trimEnd()
      .slice(0, TEXT_LIMIT - 1)
      .trimEnd(),
  );
  assert.equal(failure.headline, 'Error(s) in loading state_dict:');
  assert.equal(failure.refusalClass, null);
});

test('tag-shaped text that is not exactly one canonical tag gives no class', () => {
  const two = jobFailure({ message: '[dgxm:P] first. [dgxm:K guard=ring_pad waivable=1] second.' });
  assert.equal(two?.refusalClass, null);
  assert.equal(two?.headline, '[dgxm:P] first.');
  for (const message of [
    '[dgxm:C guard=stock_load_preflight] a guard without its waivable flag.',
    '[dgxm:C waivable=1] a waivable flag without its guard.',
    '[dgxm;P] an escaped body tag.',
    '[dgxm:measured {"checkpoint_bytes":1}] only measured numbers.',
    '[dgxm:X] an unknown class.',
  ])
    assert.equal(jobFailure({ message })?.refusalClass, null, message);
});

test('a malformed consent block yields no request and stays in the shown text', () => {
  const without = (name: string) =>
    Object.fromEntries(Object.entries(RESCUE).filter(([key]) => key !== name));
  const cases: [string, string][] = [
    ['two blocks', `${rescue()}\n${block(RESCUE)}`],
    ['no end marker', rescue().replace(DESCRIPTOR_END, '')],
    ['not JSON', `[dgxm:C] x. ${DESCRIPTOR_BEGIN}{"version":1,${DESCRIPTOR_END}`],
    ['a newer version', rescue({ ...RESCUE, version: 2 })],
    ['an empty reason', rescue({ ...RESCUE, human_reason: '' })],
    ['no panel action', rescue(without('panel_action'))],
    ['an unknown kind', rescue({ ...RESCUE, kind: 'waive-everything' })],
    ['nested measurements', rescue({ ...RESCUE, measured: { nested: { a: 1 } } })],
    ['a family hint that is not text', rescue({ ...RESCUE, family_hint: 3 })],
    ['over 4096 bytes', rescue({ ...RESCUE, human_reason: 'é'.repeat(2100) })],
  ];
  for (const [name, message] of cases) {
    const failure = jobFailure({ message });
    assert.equal(failure?.consent, null, name);
    assert.match(failure?.text ?? '', /DGXM-CONSENT-V1/, name);
  }
  const infinite = rescue().replace('"checkpoint_bytes":35000000000,', '"checkpoint_bytes":1e999,');
  assert.equal(jobFailure({ message: infinite })?.consent, null, 'a non-finite measurement');
});

const consentRow = (available: boolean, reason?: string): OperationCapability => ({
  id: CONSENT_OPERATION,
  label: 'consent · apply',
  available,
  reason,
  parameters: [
    { name: 'key', type: 'STRING', default: '' },
    { name: 'id', type: 'STRING', default: '' },
  ],
});

test('the consent control follows the consent operation row, and only that row', () => {
  const failure = jobFailure({ message: rescue() });
  assert.deepEqual(consentAction(failure, [consentRow(true)]), {
    label: 'Grant and retry',
    available: true,
    reason: '',
  });
  const without = consentAction(failure, [
    consentRow(
      false,
      'Restart the companion with --allow-management to authorize management controls.',
    ),
  ]);
  assert.equal(without?.label, 'Review consent');
  assert.equal(without?.available, false);
  assert.match(without?.reason ?? '', /^This companion was started without --allow-management/);
  assert.equal(
    consentAction(failure, [consentRow(false, 'Another reason.')])?.reason,
    'Another reason.',
    'any other reason shows as the companion gave it',
  );
  assert.equal(consentAction(failure, []), null, 'no consent operation, no control');
  assert.equal(consentAction(failure, undefined), null);
  assert.equal(consentAction(jobFailure({ message: 'plain.' }), [consentRow(true)]), null);
  assert.equal(consentAction(null, [consentRow(true)]), null);
});

test('a consent request opens consent.apply with empty identity fields and the job context', () => {
  const job = { id: 'job-7', state: 'failed', created_at: 1, submission: {} };
  assert.deepEqual(consentRequest({ ...job, error: { message: rescue() } }, 'request-4'), {
    id: 'request-4',
    action: 'consent.apply',
    parameters: { key: '', id: '' },
    consent: {
      jobId: 'job-7',
      kind: 'rescue-slab',
      consentId: RESCUE.consent_id,
      unetName: 'flux/dev.safetensors',
      humanReason: RESCUE.human_reason,
      panelAction: 'Load with slab residency',
    },
  });
  assert.equal(consentRequest({ ...job, error: { message: 'plain.' } }, 'request-5'), null);
  assert.equal(consentRequest(job, 'request-6'), null);
});

test('a consent review says whether its plan holds the request the job asked for', () => {
  const id = RESCUE.consent_id;
  const plan = (adapter_plan: unknown) =>
    ({ id: 'p', action: CONSENT_OPERATION, adapter_plan }) as unknown as OperationRecord;
  const card = (key: string, consent_id: unknown) => ({
    key,
    id: `p-${key}`,
    consent_id,
    title: key,
  });
  assert.deepEqual(consentMatch(plan({ ready: true, card: card('a', id) }), id), {
    chosen: card('a', id),
    same: true,
  });
  assert.equal(
    (consentMatch(plan({ ready: true, card: card('b', 'other') }), id) as { same: boolean }).same,
    false,
  );
  assert.deepEqual(
    consentMatch(plan({ ready: false, pending: [card('a', 'other'), card('b', id)] }), id),
    { listed: [card('b', id)] },
  );
  assert.deepEqual(consentMatch(plan({ ready: false, pending: [] }), id), { listed: [] });
  assert.equal(consentMatch(plan(null), id), null, 'a plan that read no list says nothing');
  assert.equal(consentMatch(null, id), null);
  assert.deepEqual(
    consentMatch(plan({ ready: false, pending: [{ key: 3 }, 'x', card('c', id)] }), id),
    { listed: [card('c', id)] },
    'cards without an exact key and id are not offered',
  );
});
