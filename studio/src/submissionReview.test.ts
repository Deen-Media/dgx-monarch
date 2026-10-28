import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { emptyDraft } from './model.ts';
import { reviewSummary, startReviewClock, submitBlock } from './reviewSummary.ts';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';

const WARNING =
  'Studio checked these values against the inputs the driver reports. The backend checks memory, topology, identity and consent when the job runs and can still refuse it then.';
const bindings = [
  {
    id: 'root:6:text',
    node_id: '6',
    execution_node_id: '6',
    input_name: 'text',
    display: { label: 'Prompt' },
    schema: { type: 'STRING', multiline: true },
    value: 'cat',
  },
  {
    id: 'root:7:text',
    node_id: '7',
    execution_node_id: '7',
    input_name: 'text',
    schema: { type: 'STRING', multiline: true },
    value: 'blur',
  },
  {
    id: 'root:8:seed',
    node_id: '8',
    execution_node_id: '8',
    input_name: 'seed',
    schema: { type: 'INT' },
    value: 0,
  },
  {
    id: 'root:9:filename_prefix',
    node_id: '9',
    execution_node_id: '9',
    input_name: 'filename_prefix',
    schema: { type: 'STRING' },
    value: 'kroma',
  },
];
/** A random-seed review whose style rewrote the prompt, as the companion returns it. */
function preview(changes: Record<string, unknown> = {}) {
  return {
    id: 'review-1',
    digest: 'digest',
    errors: [],
    warnings: [WARNING],
    requested: {
      ...emptyDraft(),
      workflowName: 'Kroma INT8',
      operation: 'image',
      seed_policy: 'random',
      values: { 'root:6:text': 'cat', 'root:7:text': 'blur', 'root:8:seed': 0 },
      styles: [{ id: 'paint', name: 'Painting' }],
    },
    effective: {
      seeds: { '8.seed': 4242 },
      styles: { style_order: ['paint'], prompt: 'framed painting of cat' },
      style_bindings: { prompt: 'root:6:text' },
    },
    prompt: {
      '6': { class_type: 'CLIPTextEncode', inputs: { text: 'framed painting of cat' } },
      '7': { class_type: 'CLIPTextEncode', inputs: { text: 'blur' } },
      '8': { class_type: 'DGXMonarchKSampler', inputs: { seed: 4242, steps: 20 } },
      '9': { class_type: 'SaveImage', inputs: { filename_prefix: 'kroma' } },
    },
    workflow: {},
    model_identities: [{ node_id: '2', input: 'unet_name', name: 'kroma-int8.safetensors' }],
    requested_topology: [
      { node_id: '1', class_type: 'DGXMonarchInit', parameters: { world_size: 2 } },
      { node_id: '3', class_type: 'DGXMonarchLoraLoader', parameters: {} },
    ],
    created_at: 1000,
    expires_at: 1600,
    ...changes,
  };
}
async function review(
  changes: Record<string, unknown> = {},
  state: { pending?: unknown; confirmed?: boolean; busy?: boolean } = {},
) {
  const body = await renderComponent('./components/SubmissionReview.svelte', {
    submission: {
      preview: preview(changes),
      pending: state.pending ?? null,
      confirmed: state.confirmed ?? false,
    },
    bindings,
    busy: state.busy ?? false,
    submitPreview: async () => {},
  });
  return { body, text: visibleText(body) };
}
/** The opening tag of the Submit button. */
const submitTag = (body: string) => {
  const end = body.indexOf('Submit reviewed job') + 1 || body.indexOf('Submitting…') + 1;
  assert.ok(end > 0, 'missing Submit');
  const start = body.lastIndexOf('<button', end);
  return body.slice(start, body.indexOf('>', start) + 1);
};

test('the review opens with a plain summary and keeps the frozen record closed', async () => {
  const { body, text } = await review();
  for (const line of [
    'IMMUTABLE SUBMISSION',
    'Review generation',
    'Ready to submit. Confirm below.',
    'Workflow Kroma INT8 · image',
    'kroma-int8.safetensors (unet_name, node 2)',
    'Seed 4242',
    'Chosen at review by the Random policy. Your seed fields keep their values.',
    'Prompt · node 6 framed painting of cat Your draft text before review: cat',
    'text · node 7 blur',
    'Styles applied in this order: Painting.',
    'DGXMonarchInit (node 1): world_size 2',
    'The backend sets the topology that runs.',
    `Checks still to run ${WARNING}`,
    '10 minutes after it was made. After that, review again.',
    'Immutable submission record (JSON)',
  ])
    assert.ok(text.includes(line), line);
  assert.doesNotMatch(text, /Review job|DGXMonarchLoraLoader|filename_prefix/);
  // The unchanged negative text shows once, without a draft copy.
  assert.equal(text.split('blur').length, 2);
  // The record stays closed, and its JSON is not even rendered until it opens.
  assert.match(body, /<details class="review-record">/);
  assert.doesNotMatch(body, /<pre>|"digest"/);
});

test('Submit says why it is off, and turns on only for a confirmed, unblocked review', async () => {
  const unchecked = await review();
  assert.match(submitTag(unchecked.body), /disabled/);
  assert.match(submitTag(unchecked.body), /aria-describedby="submit-block"/);
  assert.ok(unchecked.text.includes('Submit is off until you check the box above.'));
  const ready = await review({}, { confirmed: true });
  assert.doesNotMatch(submitTag(ready.body), /disabled/);
  assert.doesNotMatch(ready.text, /Submit is off/);
  const busy = await review({}, { confirmed: true, busy: true });
  assert.match(submitTag(busy.body), /disabled/);
  assert.ok(busy.text.includes('Submitting…'));
  assert.ok(busy.text.includes('Submit is off while Studio finishes the current request.'));
});

test('a pending submission or a companion error blocks the review and says so', async () => {
  const pending = await review({}, { confirmed: true, pending: { key: 'k', preview_id: 'p' } });
  assert.ok(pending.text.includes('Blocked: an earlier submission is unresolved.'));
  assert.ok(
    pending.text.includes(
      'Submit is off until the earlier submission is reconciled. Use Reconcile request in the banner.',
    ),
  );
  assert.match(submitTag(pending.body), /disabled/);
  assert.doesNotMatch(pending.body, /review-status ready/);
  // While its own request is in flight, the request identity is already pending.
  const sending = await review(
    {},
    { confirmed: true, busy: true, pending: { key: 'k', preview_id: 'p' } },
  );
  assert.ok(sending.text.includes('Submitting. Wait for the outcome.'));
  assert.ok(sending.text.includes('Submit is off while this submission is sent.'));
  const refused = await review(
    { errors: ['Choose a model file for node 2.'] },
    { confirmed: true },
  );
  assert.ok(refused.text.includes('Blocked: Choose a model file for node 2.'));
  assert.ok(
    refused.text.includes(
      'Submit is off until the problems above are fixed. Return to the draft, correct them and review again.',
    ),
  );
  assert.match(submitTag(refused.body), /disabled/);
  assert.match((await review()).body, /class="review-status ready"/);
});

test('a fixed review names its seed source, and text without styles shows once', () => {
  const summary = reviewSummary(
    preview({
      requested: { ...preview().requested, seed_policy: 'fixed', styles: [] },
      effective: { seeds: { '8.seed': 0, '13.noise_seed': 5 } },
      prompt: { ...preview().prompt, '6': { class_type: 'Text', inputs: { text: 'cat' } } },
      warnings: [{ code: 'custom' }],
      expires_at: undefined,
    }) as never,
    bindings,
    false,
    false,
  );
  assert.equal(summary.seeds, 'Seeds 8.seed = 0, 13.noise_seed = 5');
  assert.equal(summary.seedSource, 'Taken from your seed fields by the Fixed policy.');
  assert.deepEqual(summary.texts, [
    { label: 'Prompt · node 6', text: 'cat' },
    { label: 'text · node 7', text: 'blur' },
  ]);
  assert.equal(summary.styles, '');
  assert.deepEqual(summary.warnings, ['{"code":"custom"}']);
  assert.equal(summary.expiry, '');
  // A workflow without multi-line text, and no style, shows every text input instead.
  const single = bindings.map((binding) => ({ ...binding, schema: { type: binding.schema.type } }));
  const unstyled = preview({ effective: { seeds: { '8.seed': 4242 } } });
  const texts = reviewSummary(unstyled as never, single, false, false).texts;
  assert.deepEqual(
    texts.map((entry) => entry.label),
    ['Prompt · node 6', 'text · node 7', 'filename_prefix · node 9'],
  );
  const none = reviewSummary(preview({ effective: { seeds: {} } }) as never, [], false, false);
  assert.equal(none.seeds, 'This workflow has no seed input.');
  assert.equal(none.seedSource, '');
});

test('submitBlock puts the most basic block first', () => {
  const ok = preview() as never;
  const blocked = preview({ errors: ['x'] }) as never;
  assert.equal(submitBlock(ok, { confirmed: true, busy: false, pending: false }), '');
  assert.match(
    submitBlock(blocked, { confirmed: false, busy: true, pending: true }),
    /problems above/,
  );
  assert.match(submitBlock(ok, { confirmed: false, busy: false, pending: true }), /reconciled/);
  assert.match(
    submitBlock(ok, { confirmed: false, busy: true, pending: false }),
    /current request/,
  );
  assert.match(submitBlock(ok, { confirmed: false, busy: false, pending: false }), /check the box/);
});

// Run the shipped dialog submit handler: every outcome closes the dialog.
const submitCode = await componentFunctions('./components/SubmissionReview.svelte', ['submit']);
const dialogSubmit = new Function(
  'submitPreview',
  `'use strict';
  const closed = [];
  const previewDialog = { close: () => closed.push('closed') };
  ${submitCode}
  return submit().then(
    () => closed,
    (cause) => ({ cause, closed }),
  );`,
);

test('the review dialog closes after a submit, whatever its outcome', async () => {
  assert.deepEqual(await dialogSubmit(async () => {}), ['closed']);
  const failed = await dialogSubmit(async () => {
    throw new Error('lost response');
  });
  assert.equal(failed.cause.message, 'lost response');
  assert.deepEqual(failed.closed, ['closed']);
  const source = readFileSync(
    new URL('./components/SubmissionReview.svelte', import.meta.url),
    'utf8',
  );
  assert.match(source, /onclick=\{submit\}/);
});

// Run the shipped App submit with its error handling: a failure reaches the banner.
const appCode = await componentFunctions('./App.svelte', [
  'submitPreview',
  'withBusy',
  'fail',
  'setNotice',
]);
const appSubmit = new Function(
  'submission',
  `'use strict';
  let pendingActions = 0, busy = false, error = '', notice = '', dirty = true;
  const owned = [];
  const tracker = { own: (id) => owned.push(id) };
  const connection = { fromLink: false };
  const errorText = (cause) => cause.message;
  ${appCode}
  return submitPreview().then(() => ({ busy, error, notice, dirty, owned }));`,
);

test('a refused or unknown submission leaves its error for the banner and the draft dirty', async () => {
  const unknown = await appSubmit({
    submit: async () => {
      throw new Error('Submission outcome unknown until reconciled.');
    },
  });
  assert.deepEqual(unknown, {
    busy: false,
    error: 'Submission outcome unknown until reconciled.',
    notice: '',
    dirty: true,
    owned: [],
  });
  // Only an accepted job becomes the one the viewer follows.
  const accepted = await appSubmit({ submit: async () => ({ id: 'job-1' }) });
  assert.deepEqual(accepted, {
    busy: false,
    error: '',
    notice: 'Submitted job-1.',
    dirty: false,
    owned: ['job-1'],
  });
});

const expiredAt = new Date(1600 * 1000).toLocaleTimeString();

test('a review says when it has expired and turns Submit off with the reason', () => {
  const ready = { confirmed: true, busy: false, pending: false };
  // One second before expires_at the review is still ready to submit.
  const before = reviewSummary(preview() as never, bindings, false, false, 1599);
  assert.equal(before.status, 'Ready to submit. Confirm below.');
  assert.match(before.expiry, /^This review expires at .*After that, review again\.$/);
  assert.equal(submitBlock(preview() as never, { ...ready, now: 1599 }), '');
  // At expires_at the companion refuses it, so the dialog says so before a submit tries.
  const after = reviewSummary(preview() as never, bindings, false, false, 1600);
  assert.equal(after.ready, false);
  assert.equal(after.status, `Blocked: this review expired at ${expiredAt}.`);
  assert.equal(after.expiry, `This review expired at ${expiredAt}, 10 minutes after it was made.`);
  assert.equal(
    submitBlock(preview() as never, { ...ready, now: 1600 }),
    `This review expired at ${expiredAt}. Return to the draft and review again.`,
  );
  // A request already on its way says so instead, and a clock not yet read expires nothing.
  assert.equal(
    submitBlock(preview() as never, { ...ready, busy: true, pending: true, now: 1700 }),
    'Submit is off while this submission is sent.',
  );
  assert.equal(submitBlock(preview() as never, { ...ready, now: null }), '');
  assert.equal(reviewSummary(preview() as never, bindings, false, false).ready, true);
});

test('the review clock reads at once, every minute and exactly at the expiry moment', () => {
  let monotonic = 5000;
  const readings: number[] = [];
  const scheduled: Record<string, { run: () => void; ms: number }> = {};
  const cancelled: unknown[] = [];
  const stop = startReviewClock(
    { created_at: 1000, expires_at: 1600 },
    5000,
    (now) => readings.push(now),
    {
      now: () => monotonic,
      every: (run, ms) => ((scheduled.every = { run, ms }), 'minute'),
      after: (run, ms) => ((scheduled.after = { run, ms }), 'expiry'),
      cancel: (every, after) => cancelled.push(every, after),
    },
  );
  // The clock starts at the review's created_at, the companion's clock when it made the review.
  assert.deepEqual(readings, [1000]);
  assert.equal(scheduled.every.ms, 60_000);
  assert.equal(scheduled.after.ms, 600_000);
  monotonic += 60_000;
  scheduled.every.run();
  assert.deepEqual(readings, [1000, 1060]);
  // The expiry timer fires a hair early on a coarse clock and still reads expired.
  monotonic = 5000 + 599_999.5;
  scheduled.after.run();
  assert.equal(readings.at(-1), 1600);
  assert.equal(
    reviewSummary(preview() as never, bindings, false, false, readings.at(-1)).ready,
    false,
  );
  stop();
  assert.deepEqual(cancelled, ['minute', 'expiry']);
  // A review that arrives late in its life schedules only what is left, and never a negative delay.
  startReviewClock({ created_at: 1000, expires_at: 1600 }, 0, () => {}, {
    now: () => 700_000,
    every: () => 'minute',
    after: (_run, ms) => ((scheduled.after = { run: _run, ms }), 'expiry'),
    cancel: () => {},
  });
  assert.equal(scheduled.after.ms, 0);
});

test('the dialog runs its clock only while open, from the time the review arrived', async () => {
  const { Submission } = await import('./submission.svelte.ts');
  const submission = new Submission({ client: {} } as never);
  const before = performance.now();
  submission.review({ id: 'review-2' } as never);
  assert.ok(submission.arrivedAt >= before && submission.arrivedAt <= performance.now());
  const source = readFileSync(
    new URL('./components/SubmissionReview.svelte', import.meta.url),
    'utf8',
  );
  assert.match(source, /if \(!isOpen \|\| !preview\) return;/);
  assert.match(source, /startReviewClock\(preview, submission\.arrivedAt,/);
  assert.match(source, /onclose=\{\(\) => \(isOpen = false\)\}/);
});

test('the review shows a seed beyond the browser range with every digit', () => {
  const large = preview({ effective: { seeds: { '8.seed': '18446744073709551615' } } });
  assert.equal(
    reviewSummary(large as never, bindings, false, false).seeds,
    'Seed 18446744073709551615',
  );
});
