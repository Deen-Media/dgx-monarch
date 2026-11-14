import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ApiError } from './api.ts';
import { adjustTarget, draftRefusal } from './draftRefusal.ts';
import { DROPPED_REVIEW } from './inspectionGuard.ts';
import { emptyDraft } from './model.ts';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';

const source = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8');

// Run the shipped App handlers against fakes: Generate submits at once only when nothing needs a
// person, and every other outcome opens the review or reports why nothing started.
const appCode = await componentFunctions('./App.svelte', [
  'prepareJob',
  'submitPreview',
  'withBusy',
  'fail',
]);
const appGenerate = new Function(
  'prepared',
  'submit',
  'review',
  `'use strict';
  let pendingActions = 0, busy = false, error = '', dirty = true;
  const calls = [], owned = [];
  const $state = { snapshot: (value) => value };
  const draft = { values: { seed: 1 } }, bindings = [{ id: 'root:6:text' }];
  const connection = { client: 'companion', fromLink: false };
  const guard = { snapshot: () => 'ticket', previewed: () => true };
  const submission = {
    prepare: async (...args) => (calls.push(['prepare', ...args.filter((a) => typeof a !== 'function')]), prepared),
    submit: async () => (calls.push(['submit']), submit()),
  };
  const tracker = { own: (id) => owned.push(id) };
  const previewPanel = { open: () => calls.push(['open']) };
  const errorText = (cause) => (cause instanceof Error ? cause.message : String(cause));
  const DROPPED_REVIEW = ${JSON.stringify(DROPPED_REVIEW)};
  ${appCode}
  return prepareJob(review).then(() => ({ busy, error, dirty, owned, calls }));`,
);

test('Generate starts a job that needs no decision at once, with no dialog', async () => {
  const draft = { values: { seed: 1 } };
  const bindings = [{ id: 'root:6:text' }];
  const done = await appGenerate('ready', () => ({ id: 'job-1' }), undefined);
  assert.deepEqual(done, {
    busy: false,
    error: '',
    dirty: false,
    owned: ['job-1'],
    calls: [['prepare', draft, false, bindings], ['submit']],
  });
});

test('Review settings, a warning or a refusal open the review, and nothing is submitted', async () => {
  for (const review of [true, false]) {
    const done = await appGenerate('review', () => assert.fail('submitted'), review);
    assert.deepEqual(
      done.calls.map(([name]: [string]) => name),
      ['prepare', 'open'],
    );
    assert.equal(done.calls[0][2], review);
    assert.deepEqual([done.dirty, done.owned, done.error], [true, [], '']);
  }
});

test('a check dropped by an edit, or a refused submit, says why and leaves the draft dirty', async () => {
  const dropped = await appGenerate('dropped', () => assert.fail('submitted'), undefined);
  assert.equal(dropped.error, DROPPED_REVIEW);
  assert.deepEqual(
    dropped.calls.map(([name]: [string]) => name),
    ['prepare'],
  );
  const refused = await appGenerate(
    'ready',
    () => {
      throw new Error('Driver capabilities changed; review a fresh preview.');
    },
    undefined,
  );
  assert.deepEqual([refused.busy, refused.dirty, refused.owned], [false, true, []]);
  assert.equal(refused.error, 'Driver capabilities changed; review a fresh preview.');
});

test('only a coded 400 or 413 is a refusal of the draft; other failures keep the banner', () => {
  assert.deepEqual(draftRefusal(new ApiError(400, 'Bad slot.', 'lora_binding')), {
    code: 'lora_binding',
    message: 'Bad slot.',
  });
  assert.deepEqual(
    draftRefusal(new ApiError(413, 'Too big.', 'preview_limit'))?.code,
    'preview_limit',
  );
  for (const error of [
    new ApiError(400, 'No code'),
    new ApiError(401, 'Token', 'authentication'),
    new ApiError(409, 'Schema', 'schema_version'),
    new ApiError(503, 'Driver', 'driver_not_running'),
    new Error('timeout'),
    'text',
  ])
    assert.equal(draftRefusal(error), null);
});

test('a refusal names the Create tab that holds its cause', () => {
  const target = (code: string, message = 'x') => adjustTarget({ code, message });
  assert.deepEqual(target('lora_binding'), { tab: 'loras', label: 'Open LoRAs' });
  assert.deepEqual(target('lora_stack'), { tab: 'loras', label: 'Open LoRAs' });
  for (const code of ['styles', 'style_binding', 'seed_bounds', 'artifact_lineage'])
    assert.deepEqual(target(code), { tab: 'inputs', label: 'Open Inputs' }, code);
  assert.deepEqual(target('workflow', '5.steps: Value violates max=100'), {
    tab: 'advanced',
    label: 'Open All inputs',
  });
  assert.deepEqual(target('workflow', '12:3.seed: Expected an integer'), {
    tab: 'advanced',
    label: 'Open All inputs',
  });
  assert.deepEqual(target('workflow', 'Node type unavailable: Foo'), {
    tab: 'workflow',
    label: 'Open Layout',
  });
  assert.equal(target('preview_limit'), null);
});

const refusalDialog = (code: string, message: string) =>
  renderComponent('./components/SubmissionReview.svelte', {
    submission: {
      preview: null,
      refusal: { code, message },
      reviewed: [],
      pending: null,
      acknowledged: [],
      acknowledge() {},
    },
    busy: false,
    submitPreview: async () => {},
  });

test('a refused draft opens the dialog on what needs attention and the way to fix it', async () => {
  const body = await refusalDialog('lora_binding', 'LoRA slot 1 (12.strength_model): too strong');
  const text = visibleText(body);
  for (const line of [
    'NEEDS ATTENTION',
    'Fix the draft',
    'The companion refused this draft, so no job started.',
    'LoRA slot 1 (12.strength_model): too strong',
    'Error code lora_binding',
    'Return to draft',
    'Open LoRAs',
  ])
    assert.ok(text.includes(line), line);
  // No record exists to show, and nothing can start from here.
  assert.doesNotMatch(text, /Immutable submission record|>Generate<|Generate anyway/);
  assert.doesNotMatch(body, /type="checkbox"/);
  const plain = visibleText(await refusalDialog('preview_limit', 'Too big.'));
  assert.ok(plain.includes('Return to draft'));
  assert.doesNotMatch(plain, /Open (LoRAs|Inputs|All inputs|Layout)/);
  // The button opens its tab through the bound controlTab and closes the dialog.
  assert.match(
    source('./components/SubmissionReview.svelte'),
    /function adjust\(tab: string\) \{\s*controlTab = tab;\s*previewDialog\.close\(\);/,
  );
  assert.match(source('./App.svelte'), /<SubmissionReview[^>]*bind:controlTab \/>/);
});

const generationProps = (overrides: Record<string, unknown> = {}) => ({
  client: {},
  draft: { ...emptyDraft(), workflow: { nodes: [] }, workflowName: 'Portrait', values: {} },
  bindings: [],
  rawWorkflow: '',
  controlTab: 'inputs',
  bindingErrors: [],
  catalog: {},
  connections: [],
  workflows: [],
  workflowId: '',
  requestLimit: 4 * 1024 * 1024,
  dirty: false,
  busy: false,
  pending: false,
  driverState: 'connected',
  changeDraft() {},
  inspect: async () => {},
  fail() {},
  undo() {},
  prepareJob: async () => {},
  ...overrides,
});

test('Generate starts the job, and Review settings asks for the review', () => {
  const generation = source('./views/Generation.svelte');
  // Each button calls prepareJob itself, so a click event never arrives as the review flag.
  assert.match(generation, /onclick=\{\(\) => prepareJob\(true\)\}[^>]*>Review settings</);
  assert.match(generation, /onclick=\{\(\) => prepareJob\(\)\}[^>]*>Generate <span>/);
});

test('an accepted submit shows its brief line in place of the hint, and says it once', async () => {
  const render = (props: Record<string, unknown>) =>
    renderComponent('./views/Generation.svelte', generationProps(props));
  const body = await render({ confirmation: 'Job submitted.' });
  assert.match(
    body,
    /<small class="submit-confirmation" aria-hidden="true">Job submitted\.<\/small>/,
  );
  assert.match(body, /<span class="sr-only" role="status">Job submitted\.<\/span>/);
  assert.doesNotMatch(body, /Generate checks the draft/);
  // The live region stays in place while empty, so a screen reader hears each new line.
  assert.match(await render({}), /<span class="sr-only" role="status"><\/span>/);
  assert.match(source('./App.svelte'), /confirmation=\{submission\.confirmation\}/);
});

test('every frozen record stays closed, renders only when opened and scrolls in a short box', () => {
  const dialog = source('./components/SubmissionReview.svelte');
  assert.match(
    dialog,
    /\{#if recordFor === preview\.id\}<pre>\{JSON\.stringify\(preview, null, 2\)\}<\/pre>\{\/if\}/,
  );
  // Queue's job details reach the same record after submission, closed until opened.
  const queue = source('./components/JobProvenance.svelte');
  assert.match(queue, /<summary>Immutable submission record \(JSON\)<\/summary>/);
  assert.match(queue, /\{#if submissionShown\}<pre>\{writeExactJSON\(job\.submission\)\}<\/pre>/);
  assert.match(queue, /\{#if fullShown\}<pre>\{text\}<\/pre>\{\/if\}/);
  assert.doesNotMatch(queue, /<details[^>]*\bopen\b/);
  assert.match(
    source('./styles/dialogs.css'),
    /\.review-record pre \{\s*max-height: min\(16rem, 35vh\);/,
  );
});
