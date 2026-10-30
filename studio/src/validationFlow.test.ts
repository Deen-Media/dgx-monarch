import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';
import type { StudioClient } from './api.ts';
import { InspectionGuard } from './inspectionGuard.ts';
import { followIssues, issuesStale, refreshIssues } from './issueRefresh.ts';
import {
  fieldIssues,
  fieldTab,
  inspectionIssues,
  issueText,
  reveal,
  summaryIssues,
} from './fieldIssues.ts';
import { emptyDraft } from './model.ts';
import type { Binding, Draft } from './model.ts';

// A Studio tester's case: the Kromoss template names a model file the driver lacks.
const ERROR = '3.unet_name: Value is not an available choice';
const workflow = {
  '3': { class_type: 'DGXMonarchUNETLoader', inputs: { unet_name: 'kromoss_bf16.safetensors' } },
  '10': { class_type: 'DGXMonarchKSampler', inputs: { seed: 42, steps: 50 }, _meta: {} },
};
const unet: Binding = {
  id: 'root:3:unet_name',
  node_id: '3',
  execution_node_id: '3',
  input_name: 'unet_name',
  schema: { type: 'COMBO', choices: ['kroma.safetensors', 'kromoss_bf16.safetensors'] },
  prominent: true,
  order: 9,
};
const seed: Binding = {
  id: 'root:10:seed',
  node_id: '10',
  execution_node_id: '10',
  input_name: 'seed',
  schema: { type: 'INT', min: 0, max: 2 ** 53 - 1 },
  prominent: true,
  order: 2,
};
const steps: Binding = {
  id: 'root:10:steps',
  node_id: '10',
  execution_node_id: '10',
  input_name: 'steps',
  schema: { type: 'INT', min: 1, max: 10000 },
  prominent: false,
  order: 12,
};
const draftOf = (values: Record<string, unknown>, extra: Partial<Draft> = {}): Draft => ({
  ...emptyDraft(),
  workflow,
  workflowName: 'Kromoss text to image',
  values: { 'root:3:unet_name': 'kromoss_bf16.safetensors', 'root:10:seed': 42, ...values },
  ...extra,
});

/** A companion whose inspections answer with the queued error lists, each when released. */
function companion(...answers: string[][]) {
  const sent: unknown[] = [];
  const gates: Array<() => void> = [];
  const client = {
    request: async (path: string, body: unknown) => {
      assert.equal(path, '/workflows/inspect');
      sent.push(body);
      await new Promise<void>((done) => gates.push(done));
      return {
        errors: answers.shift() ?? [],
        connections: [{ node_id: '10', input_name: 'model' }],
      };
    },
  } as unknown as StudioClient;
  const release = async () => {
    while (!gates.length) await new Promise((done) => setImmediate(done));
    gates.shift()!();
    await new Promise((done) => setImmediate(done));
  };
  return { client, sent, release };
}

/** App's state as the one-line wiring keeps it: the shown issues, set by the refresh. */
function app(errors: string[]) {
  const guard = new InspectionGuard();
  const state = { errors, connections: [] as unknown[], applied: 0 };
  const refresh = {
    guard,
    apply: (next: { errors: string[]; connections: unknown[] }) => {
      state.errors = next.errors;
      state.connections = next.connections;
      state.applied++;
    },
  };
  return { guard, state, refresh };
}

test('after a value edit Create inspects the draft again and drops the error it resolved', async () => {
  const { client, sent, release } = companion([]);
  const { guard, state, refresh } = app([ERROR]);
  let draft = draftOf({});
  // The list the template's inspection showed is current until an edit.
  assert.equal(issuesStale(state.errors, guard, client), false);
  guard.edit({ operation: 'workflow' });
  guard.edit({ seed_policy: 'fixed' });
  assert.equal(issuesStale(state.errors, guard, client), false);
  // The tester picks an available file.
  const change = { values: { ...draft.values, 'root:3:unet_name': 'kroma.safetensors' } };
  guard.edit(change);
  draft = { ...draft, ...change };
  assert.equal(issuesStale(state.errors, guard, client), true);
  const running = refreshIssues(refresh, { client: () => client, draft: () => draft });
  await release();
  assert.equal(await running, true);
  assert.deepEqual(sent, [{ workflow, values: change.values }]);
  assert.deepEqual(state.errors, []);
  assert.equal(state.connections.length, 1);
  assert.equal(issuesStale(state.errors, guard, client), false);
});

test('a refresh keeps a genuine error, and the kept list is not checked again', async () => {
  const { client, release } = companion([ERROR]);
  const { guard, state, refresh } = app([ERROR]);
  const draft = draftOf({ 'root:10:seed': 7 });
  guard.edit({ values: draft.values });
  const running = refreshIssues(refresh, { client: () => client, draft: () => draft });
  await release();
  assert.equal(await running, true);
  assert.deepEqual(state.errors, [ERROR]);
  assert.equal(issuesStale(state.errors, guard, client), false);
});

test('an answer that an edit, an inspection or a reconnect overtook is dropped without a notice', async () => {
  for (const overtake of ['edit', 'inspection', 'reconnect'] as const) {
    const { client, release } = companion([]);
    const { guard, state, refresh } = app([ERROR]);
    let draft = draftOf({ 'root:3:unet_name': 'kroma.safetensors' });
    let connected = client;
    guard.edit({ values: draft.values });
    const running = refreshIssues(refresh, { client: () => connected, draft: () => draft });
    if (overtake === 'edit') {
      draft = draftOf({ 'root:3:unet_name': 'kromoss_bf16.safetensors' });
      guard.edit({ values: draft.values });
    } else if (overtake === 'inspection') guard.begin(client);
    else connected = companion().client;
    await release();
    assert.equal(await running, false, overtake);
    assert.deepEqual(state, { errors: [ERROR], connections: [], applied: 0 }, overtake);
  }
});

test('Create waits for edits to settle, then sends one inspection while the issues are stale', async () => {
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    const { client, sent, release } = companion([]);
    const { guard, state, refresh } = app([ERROR]);
    let draft = draftOf({});
    const checking: boolean[] = [];
    const current = { client: () => client, draft: () => draft };
    const follow = () => followIssues(refresh, state.errors, current, (on) => checking.push(on));
    // Nothing to check before an edit, and no request.
    assert.equal(follow(), undefined);
    assert.deepEqual(checking, [false]);
    // Each edit restarts the wait, so typing sends one request.
    for (const name of ['kroma', 'kroma.safetensors']) {
      const change = { values: { ...draft.values, 'root:3:unet_name': name } };
      guard.edit(change);
      draft = { ...draft, ...change };
      const cancel = follow();
      mock.timers.tick(399);
      assert.equal(sent.length, 0);
      if (name === 'kroma') cancel!();
    }
    mock.timers.tick(1);
    assert.equal(sent.length, 1);
    await release();
    assert.deepEqual(state.errors, []);
    assert.deepEqual(checking, [false, true, true, false]);
    // With the list cleared, later edits ask nothing.
    guard.edit({ values: { ...draft.values, 'root:10:seed': 3 } });
    assert.equal(follow(), undefined);
    mock.timers.tick(1000);
    assert.equal(sent.length, 1);
  } finally {
    mock.timers.reset();
  }
});

test('App hands Create its guard and the setter for the issues and connections, on one line', () => {
  const source = readFileSync(new URL('./App.svelte', import.meta.url), 'utf8');
  assert.match(
    source,
    /\n {10}issueRefresh=\{\{ guard, apply: \(next\) => \(\{ errors: bindingErrors, connections \} = next\) \}\}\n/,
  );
  const create = readFileSync(new URL('./views/Generation.svelte', import.meta.url), 'utf8');
  assert.match(create, /followIssues\(\s*issueRefresh,\s*bindingErrors,/);
});

test('field issues name the field, its node and the problem, Inputs fields first', () => {
  const titled = {
    nodes: [
      { id: 10, type: 'KSampler', title: 'Sampler' },
      { id: 3, type: 'UNETLoader' },
    ],
    definitions: {
      subgraphs: [{ id: 'inner', nodes: [{ id: 5, type: 'CLIPTextEncode', title: 'Prompt' }] }],
    },
  };
  const inner: Binding = {
    id: '@instance/root/12:5:text',
    node_id: '5',
    graph_id: 'inner',
    instance_path: ['12'],
    input_name: 'text',
    schema: { type: 'STRING' },
    required: true,
    order: 1,
  };
  const issues = fieldIssues(
    [steps, inner, { ...unet, graph_id: 'root' }, { ...seed, graph_id: 'root' }],
    { 'root:3:unet_name': 'kromoss_bf16.safetensors', 'root:10:steps': 0 },
    titled,
  );
  assert.deepEqual(issues.map(issueText), [
    'seed on node 10 (Sampler): Enter a finite number.',
    'text on node 12/5 (Prompt): A value is required.',
    'steps on node 10 (Sampler): Minimum 1.',
  ]);
  // A node without a title goes by its id alone, as do the nodes of an API prompt.
  const plain = fieldIssues([unet], { 'root:3:unet_name': 'gone' }, titled);
  assert.equal(issueText(plain[0]), 'unet_name on node 3: Choose an available value.');
  const api = { ...workflow, '10': { ...workflow['10'], _meta: { title: 'KSampler' } } };
  assert.equal(
    issueText(fieldIssues([steps], { 'root:10:steps': 0 }, api)[0]),
    'steps on node 10 (KSampler): Minimum 1.',
  );
  assert.deepEqual(fieldIssues([unet], { 'root:3:unet_name': 'kroma.safetensors' }, workflow), []);
});

test('a workflow issue about a value names its field, and one field is listed once', () => {
  const codec: Binding = { ...unet, id: 'root:3:format.codec', input_name: 'format.codec' };
  const format: Binding = { ...unet, id: 'root:3:format', input_name: 'format' };
  const issues = inspectionIssues(
    [ERROR, '3.format.codec: Value is not an available choice', 'App Mode input 4: Widget missing'],
    [format, codec, unet, steps],
    workflow,
  );
  assert.deepEqual(
    issues.map((issue) => [issue.text, issue.field?.id]),
    [
      ['unet_name on node 3: Value is not an available choice.', unet.id],
      ['format.codec on node 3: Value is not an available choice.', codec.id],
      ['App Mode input 4: Widget missing', undefined],
    ],
  );
  // The browser already flags unet_name, so the companion's line about it is not repeated.
  const fields = fieldIssues([unet], { 'root:3:unet_name': 'gone' }, workflow);
  assert.deepEqual(
    summaryIssues(fields, issues).map((issue) => issue.text),
    [
      'unet_name on node 3: Choose an available value.',
      'format.codec on node 3: Value is not an available choice.',
      'App Mode input 4: Widget missing',
    ],
  );
});

test('Show field opens Inputs or All inputs, whichever lists the field', () => {
  assert.equal(fieldTab({ prominent: true }, 'assets'), 'inputs');
  assert.equal(fieldTab({ prominent: false }, 'inputs'), 'advanced');
  assert.equal(fieldTab({ prominent: true }, 'inputs'), 'inputs');
  assert.equal(fieldTab({ prominent: true }, 'advanced'), 'advanced');
  assert.equal(fieldTab({ prominent: false }, 'workflow'), 'advanced');
});

/** A page with one element per id that records what reveal did to it. */
function page(ids: Record<string, { disabled?: boolean; inClosedDetails?: boolean }>) {
  const log: string[] = [];
  return {
    log,
    getElementById(id: string) {
      const spec = ids[id];
      if (!spec) return null;
      const details = { open: !spec.inClosedDetails };
      return {
        disabled: spec.disabled,
        closest: () => (spec.inClosedDetails === undefined ? null : details),
        scrollIntoView: (options: ScrollIntoViewOptions) =>
          log.push(`scroll ${id} ${options.block}`),
        focus: () => log.push(`focus ${id}${details.open ? '' : ' closed'}`),
      };
    },
  };
}

test('reveal scrolls a field into view, opens a closed disclosure and focuses an enabled control', () => {
  const view = page({
    'field-a': { inClosedDetails: true },
    'field-b': { disabled: true },
  });
  assert.equal(reveal(view, 'field-a'), true);
  assert.equal(reveal(view, 'field-b'), true);
  assert.equal(reveal(view, 'field-c'), false);
  assert.deepEqual(view.log, ['scroll field-a center', 'focus field-a', 'scroll field-b center']);
});

// Run the shipped Show field handler against a fake page and Svelte's tick.
const showCode = await componentFunctions('./components/IssueSummary.svelte', ['show']);
const summaryHarness = new Function(
  'fieldTab',
  'reveal',
  'document',
  'start',
  `'use strict';
  let controlTab = start, listOpen = true;
  const tabs = [];
  const tick = async () => tabs.push(controlTab);
  ${showCode}
  return { show, tabs, tab: () => controlTab, listOpen: () => listOpen };`,
);

test('Show field switches to the tab that lists the field, then scrolls to it and focuses it', async () => {
  const view = page({ 'field-root:10:steps': {}, 'inspection-issues': {} });
  const summary = summaryHarness(fieldTab, reveal, view, 'inputs');
  await summary.show({ text: 'steps', field: { id: 'root:10:steps', prominent: false } });
  assert.equal(summary.tab(), 'advanced');
  assert.deepEqual(view.log, ['scroll field-root:10:steps center', 'focus field-root:10:steps']);
  // The list of issues closes, so the fields get their room back.
  assert.equal(summary.listOpen(), false);
  // A field Inputs should list but does not show falls back to All inputs.
  const hidden = page({ 'field-root:3:unet_name': {} });
  const other = summaryHarness(fieldTab, reveal, hidden, 'loras');
  await other.show({ text: 'unet_name', field: { id: 'root:3:unet_name', prominent: true } });
  assert.deepEqual(other.tabs, ['inputs']);
  assert.ok(hidden.log.includes('focus field-root:3:unet_name'));
  // An issue that names no field shows the workflow's list of issues.
  const list = page({ 'inspection-issues': {} });
  const plain = summaryHarness(fieldTab, reveal, list, 'workflow');
  await plain.show({ text: 'App Mode input 4: Widget missing' });
  assert.equal(plain.tab(), 'inputs');
  assert.deepEqual(list.log, ['scroll inspection-issues center', 'focus inspection-issues']);
});

function createProps(overrides: Record<string, unknown> = {}) {
  return {
    client: {},
    draft: draftOf({ 'root:10:steps': 50 }),
    bindings: [unet, seed, steps],
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
    rebind: async () => {},
    refreshCatalog: async () => {},
    ...overrides,
  };
}
const render = (overrides: Record<string, unknown> = {}) =>
  renderComponent('./views/Generation.svelte', createProps(overrides));
/** The footer's issue summary, as a reader sees it. */
const footerText = (body: string) =>
  visibleText(body.slice(body.lastIndexOf('<div', body.indexOf('id="generate-issues"'))));
/** The opening tag of the first button whose text starts with `label`. */
const buttonTag = (body: string, label: string) => {
  const end = body.indexOf(`>${label}`);
  assert.ok(end >= 0, `missing ${label}`);
  return body.slice(body.lastIndexOf('<button', end), end + 1);
};

test('the footer names the invalid field and keeps its details open to use while Generate is off', async () => {
  const body = await render({
    draft: draftOf({ 'root:3:unet_name': 'gone.safetensors', 'root:10:steps': 0 }),
    bindingErrors: [ERROR, 'App Mode input 4: Widget missing'],
  });
  for (const label of ['Review settings', 'Generate <span>'])
    assert.match(buttonTag(body, label), /disabled="" aria-describedby="generate-issues"/);
  const footer = footerText(body);
  assert.match(
    footer,
    /^Fix unet_name on node 3: Choose an available value\. Show field All 3 issues/,
  );
  assert.match(footer, /steps on node 10: Minimum 1\. Show field/);
  assert.match(footer, /App Mode input 4: Widget missing Show issues/);
  // Neither Show field nor the list of issues is ever disabled.
  const summary = body.slice(body.indexOf('id="generate-issues"'), body.indexOf('</details>'));
  assert.doesNotMatch(summary, /disabled/);
  assert.match(summary, /<details>/);
});

test('with only workflow issues Generate stays on and the footer names the first one', async () => {
  const body = await render({ bindingErrors: [ERROR] });
  assert.doesNotMatch(buttonTag(body, 'Generate <span>'), /disabled/);
  const footer = footerText(body);
  assert.match(
    footer,
    /^Check unet_name on node 3: Value is not an available choice\. Show field$/,
  );
  assert.match(body, /id="inspection-issues" tabindex="-1"/);
  const clean = await render();
  assert.doesNotMatch(clean, /generate-issues/);
  assert.match(visibleText(clean), /Generate checks the draft and starts the job\./);
});

test('under Random a valid seed field reads Random each run and cannot be edited', async () => {
  const ready = { 'root:10:steps': 50 };
  const random = await render({ draft: draftOf(ready, { seed_policy: 'random' }) });
  assert.match(random, /<input id="field-root:10:seed" value="Random each run" disabled=""\/>/);
  assert.match(visibleText(random), /Choose the Fixed seed policy to set this seed\./);
  const fixed = await render({ draft: draftOf(ready, { seed_policy: 'fixed' }) });
  assert.match(fixed, /<input id="field-root:10:seed" type="number"[^>]*>/);
  assert.doesNotMatch(buttonTag(fixed, 'Generate <span>'), /disabled/);
  assert.doesNotMatch(fixed, /Random each run|disabled="" aria-invalid/);
  // A seed the draft cannot submit stays open to repair, even under Random.
  const invalid = await render({
    draft: draftOf({ ...ready, 'root:10:seed': null }, { seed_policy: 'random' }),
  });
  assert.match(invalid, /<input id="field-root:10:seed" type="number"[^>]*>/);
  assert.match(
    visibleText(invalid),
    /Chosen for each job: the Random policy replaces this value\./,
  );
  assert.match(visibleText(invalid), /seed on node 10: Enter a finite number\./);
});

test('Operation reads as the job category and says the workflow decides what runs', async () => {
  const body = await render({ draft: draftOf({}, { operation: 'workflow' }) });
  assert.match(body, /<label for="generation-operation">Job category<\/label>/);
  assert.match(body, /<select id="generation-operation" aria-describedby="operation-note">/);
  assert.match(
    visibleText(body),
    /Labels the job in Queue, the gallery and its record, while the workflow decides what runs\./,
  );
});
