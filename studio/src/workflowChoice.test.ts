import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import type { WorkflowEntry } from './contracts.ts';
import { emptyDraft } from './model.ts';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';
import {
  byTitle,
  filterKeyAction,
  loadWorkflow,
  matchesWorkflow,
  paletteWorkflows,
  workflowChoices,
  workflowTitle,
} from './workflowChoice.ts';

const workflow = { nodes: [] };
const entry = (name: string, title: string) =>
  ({ id: `workflow.${name}`, name, title, workflow: { name } }) as WorkflowEntry;
const int8 = entry('dgx-monarch-kroma-t2i-int8', 'Kroma local INT8 text to image');
const bf16 = entry('dgx-monarch-kroma-t2i', 'Kroma v0.3.1 BF16 text to image');
const flux1 = entry('dgx-monarch-flux1-t2i', 'Flux 1 text to image');
const quick = entry('dgx-monarch-quickstart', 'quickstart');
const wan = entry('dgx-monarch-wan-t2v', 'Wan text to video');
// Catalog order is file stem order, as workflows.json lists them.
const catalog = [flux1, bf16, int8, quick, wan];
const titles = (entries: WorkflowEntry[]) => entries.map(workflowTitle);

test('a workflow is listed by its title, or by its file stem when the catalog gives none', () => {
  assert.equal(workflowTitle(int8), 'Kroma local INT8 text to image');
  assert.equal(workflowTitle({ ...int8, title: '' }), 'dgx-monarch-kroma-t2i-int8');
  assert.equal(workflowTitle({ id: 'x', name: 'stem-only', workflow } as never), 'stem-only');
});

test('the list is in title order, ignoring case, and the catalog keeps its own order', () => {
  const two = entry('dgx-monarch-flux2-t2i', 'Flux 2 text to image');
  const ten = entry('dgx-monarch-flux10-t2i', 'Flux 10 text to image');
  const input = [wan, ten, quick, int8, two, bf16, flux1];
  assert.deepEqual(titles(byTitle(input)), [
    'Flux 1 text to image',
    'Flux 2 text to image',
    'Flux 10 text to image',
    'Kroma local INT8 text to image',
    'Kroma v0.3.1 BF16 text to image',
    'quickstart',
    'Wan text to video',
  ]);
  assert.deepEqual(input[0], wan, 'sorting copies the catalog');
  // Equal titles fall back to the file stem, so the order never depends on the input.
  const twin = entry('dgx-monarch-a-twin', 'Wan text to video');
  assert.deepEqual(
    byTitle([wan, twin]).map((w) => w.name),
    ['dgx-monarch-a-twin', 'dgx-monarch-wan-t2v'],
  );
});

test('the filter matches every typed word anywhere in the title or the file stem, in any case', () => {
  for (const query of ['kroma', 'KROMA', 'int8', 'local int8', 'int8 kroma', 't2i-int8', 'Local'])
    assert.ok(matchesWorkflow(int8, query), query);
  assert.ok(matchesWorkflow(quick, 'quick'));
  assert.ok(matchesWorkflow(flux1, 'flux1'), 'the stem finds names the title spells another way');
  assert.ok(!matchesWorkflow(int8, 'kroma video'), 'every word must match');
  assert.ok(!matchesWorkflow(int8, 'bf16'));
  for (const blank of ['', '   ']) assert.ok(matchesWorkflow(int8, blank), 'no words match all');
});

test('the select keeps the current choice visible and selected while the filter hides it', () => {
  const kroma = workflowChoices(catalog, 'kroma', '');
  assert.deepEqual(titles(kroma.matches), [
    'Kroma local INT8 text to image',
    'Kroma v0.3.1 BF16 text to image',
  ]);
  assert.deepEqual(kroma.options, kroma.matches);
  const kept = workflowChoices(catalog, 'kroma', wan.id);
  assert.deepEqual(titles(kept.matches), titles(kroma.matches));
  assert.deepEqual(titles(kept.options), [...titles(kroma.matches), 'Wan text to video']);
  const matched = workflowChoices(catalog, 'int8', int8.id);
  assert.deepEqual(matched.options, [int8], 'a matching choice is listed once');
  assert.deepEqual(workflowChoices(catalog, '', '').options, byTitle(catalog));
});

test('a filter with no matches lists only the current choice', () => {
  const none = workflowChoices(catalog, 'no such workflow', bf16.id);
  assert.deepEqual(none.matches, []);
  assert.deepEqual(none.options, [bf16]);
  assert.deepEqual(workflowChoices(catalog, 'no such workflow', '').options, []);
});

test('Enter loads the first match only while no workflow is chosen; Escape clears', () => {
  const matches = workflowChoices(catalog, 'kroma', '').matches;
  assert.equal(filterKeyAction('Enter', '', matches), int8);
  assert.equal(filterKeyAction('Enter', wan.id, matches), undefined);
  assert.equal(filterKeyAction('Enter', '', []), undefined);
  assert.equal(filterKeyAction('Escape', '', matches), 'clear');
  assert.equal(filterKeyAction('Escape', wan.id, []), 'clear');
  for (const key of ['k', 'ArrowDown', 'Tab', ' '])
    assert.equal(filterKeyAction(key, '', matches), undefined);
});

test('the palette lists matching workflows by title once the query has a word', () => {
  assert.deepEqual(paletteWorkflows(catalog, ''), []);
  assert.deepEqual(paletteWorkflows(catalog, '  '), []);
  assert.deepEqual(titles(paletteWorkflows(catalog, 'kroma')), [
    'Kroma local INT8 text to image',
    'Kroma v0.3.1 BF16 text to image',
  ]);
  assert.deepEqual(paletteWorkflows(catalog, 'text video'), [wan]);
  assert.deepEqual(paletteWorkflows(catalog, 'quickstart'), [quick]);
});

test('typing kroma in the palette finds the bundled Kroma templates by title', async () => {
  const data = new URL('../../src/dgx_monarch/studio/data/workflows.json', import.meta.url);
  const bundled: WorkflowEntry[] = JSON.parse(await readFile(data, 'utf8')).workflows;
  assert.ok(bundled.every((w) => typeof w.title === 'string' && w.title && w.name));
  const found = paletteWorkflows(bundled, 'kroma');
  assert.ok(found.some((w) => w.title === 'Kroma local INT8 text to image'));
  assert.ok(found.every((w) => `${w.title} ${w.name}`.toLowerCase().includes('kroma')));
  assert.deepEqual(found, byTitle(found));
});

test('loading a workflow inspects it under its file stem with the template id', async () => {
  const calls: unknown[][] = [];
  await loadWorkflow(int8, async (...args) => void calls.push(args));
  assert.deepEqual(calls, [[{ name: int8.name }, int8.name, false, int8.id]]);
});

// Run the shipped Generation handlers against fakes.
const generationCode = await componentFunctions('./views/Generation.svelte', [
  'chooseWorkflow',
  'filterKeydown',
  'clearFilter',
]);
const generationHarness = new Function(
  'filterKeyAction',
  'loadWorkflow',
  'current',
  'matches',
  `'use strict';
  const calls = [];
  let choosing = '', workflowQuery = 'kroma';
  const choices = { matches };
  const filterInput = { focus: () => calls.push(['focus']) };
  const inspect = async (...args) => void calls.push(['inspect', ...args]);
  ${generationCode}
  return { filterKeydown, clearFilter, chooseWorkflow, view: () => ({ calls, choosing, workflowQuery }) };`,
);
function generation(current: string, matches: WorkflowEntry[]) {
  return generationHarness(filterKeyAction, loadWorkflow, current, matches);
}
function key(name: string, isComposing = false) {
  const event = {
    key: name,
    isComposing,
    prevented: false,
    preventDefault: () => (event.prevented = true),
  };
  return event;
}

test('Generation: Enter in the filter loads the first match the way the select does', () => {
  const view = generation('', [int8, bf16]);
  const enter = key('Enter');
  view.filterKeydown(enter);
  assert.ok(enter.prevented);
  assert.deepEqual(view.view().calls, [['inspect', int8.workflow, int8.name, false, int8.id]]);
  assert.equal(view.view().choosing, int8.id);
  const select = generation('', []);
  select.chooseWorkflow(int8);
  assert.deepEqual(select.view().calls, view.view().calls);
});

test('Generation: Enter keeps a chosen workflow, and Escape or Clear empties the filter', () => {
  const view = generation(wan.id, [int8]);
  const enter = key('Enter');
  view.filterKeydown(enter);
  assert.ok(!enter.prevented);
  assert.deepEqual(view.view(), { calls: [], choosing: '', workflowQuery: 'kroma' });
  const escape = key('Escape');
  view.filterKeydown(escape);
  assert.ok(escape.prevented);
  assert.equal(view.view().workflowQuery, '');
  const cleared = generation('', [int8]);
  cleared.clearFilter();
  assert.deepEqual(cleared.view(), { calls: [['focus']], choosing: '', workflowQuery: '' });
  const typing = key('k');
  cleared.filterKeydown(typing);
  assert.ok(!typing.prevented);
  // Enter that confirms an input-method candidate loads nothing.
  const composing = generation('', [int8]);
  const confirm = key('Enter', true);
  composing.filterKeydown(confirm);
  assert.ok(!confirm.prevented);
  assert.deepEqual(composing.view(), { calls: [], choosing: '', workflowQuery: 'kroma' });
});

// Run the shipped palette handler: it opens Create and loads through the same path.
const paletteCode = await componentFunctions('./components/CommandPalette.svelte', [
  'loadWorkflowCommand',
]);
const paletteHarness = new Function(
  'loadWorkflow',
  `'use strict';
  const calls = [];
  const navigate = (view) => calls.push(['navigate', view]);
  const inspect = async (...args) => void calls.push(['inspect', ...args]);
  ${paletteCode}
  return { loadWorkflowCommand, calls };`,
);

test('the palette opens Create and loads the workflow as the select does', () => {
  const palette = paletteHarness(loadWorkflow);
  palette.loadWorkflowCommand(int8);
  assert.deepEqual(palette.calls, [
    ['navigate', 'workspace'],
    ['inspect', int8.workflow, int8.name, false, int8.id],
  ]);
});

const generationProps = (overrides: Record<string, unknown> = {}) => ({
  client: {},
  draft: { ...emptyDraft(), workflow, workflowName: int8.name, values: {} },
  bindings: [],
  rawWorkflow: '',
  controlTab: 'inputs',
  bindingErrors: [],
  catalog: {},
  connections: [],
  workflows: catalog,
  workflowId: int8.id,
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

test('the workflow select shows titles in title order with each file stem beside them', async () => {
  const body = await renderComponent('./views/Generation.svelte', generationProps());
  const select = body.slice(body.indexOf('<select id="workflow"'), body.indexOf('</select>'));
  const options = [...select.matchAll(/<option value="([^"]*)"(?: title="([^"]*)")?/g)];
  assert.deepEqual(
    options.map(([, id]) => id),
    ['', ...byTitle(catalog).map((w) => w.id)],
  );
  assert.deepEqual(
    options.slice(1).map(([, , stem]) => stem),
    byTitle(catalog).map((w) => w.name),
  );
  assert.match(
    select,
    /title="dgx-monarch-kroma-t2i-int8" selected="">Kroma local INT8 text to image</,
  );
  assert.match(body, /<small class="workflow-file mono">dgx-monarch-kroma-t2i-int8<\/small>/);
  assert.match(body, /type="search" placeholder="Filter by title or file name"/);
  assert.match(body, /<button disabled="" aria-label="Clear workflow filter">Clear<\/button>/);
  assert.doesNotMatch(body, /No packaged workflow matches/);
  const imported = await renderComponent(
    './views/Generation.svelte',
    generationProps({ workflowId: '' }),
  );
  assert.doesNotMatch(imported, /class="workflow-file/);
});

async function selectOptions(props: Record<string, unknown>) {
  const body = await renderComponent('./views/Generation.svelte', generationProps(props));
  const select = body.slice(body.indexOf('<select id="workflow"'), body.indexOf('</select>'));
  const ids = [...select.matchAll(/<option value="([^"]*)"/g)].map(([, id]) => id);
  return { body, select, ids };
}

test('a filter lists its matches in title order beside the loaded workflow', async () => {
  const { body, select, ids } = await selectOptions({ workflowQuery: 'video' });
  assert.deepEqual(ids, ['', int8.id, wan.id]);
  assert.match(select, /selected="">Kroma local INT8 text to image</);
  assert.doesNotMatch(tagOf(body, 'Clear workflow filter'), /disabled/);
  assert.doesNotMatch(body, /No packaged workflow matches/);
});

test('a filter with no matches says so and keeps the loaded workflow selected', async () => {
  const { body, select, ids } = await selectOptions({ workflowQuery: 'no such workflow' });
  assert.match(visibleText(body), /No packaged workflow matches this filter\./);
  assert.deepEqual(ids, ['', int8.id]);
  assert.match(select, /selected="">Kroma local INT8 text to image</);
  const nothing = await selectOptions({ workflowQuery: 'no such workflow', workflowId: '' });
  assert.deepEqual(nothing.ids, ['']);
  assert.match(nothing.select, /<option value="" selected="">dgx-monarch-kroma-t2i-int8</);
});

function tagOf(body: string, label: string) {
  const at = body.indexOf(`aria-label="${label}"`);
  assert.ok(at >= 0, `missing ${label}`);
  return body.slice(body.lastIndexOf('<', at), body.indexOf('>', at) + 1);
}

test('the palette offers the workflow group only when the catalog has workflows', async () => {
  const props = { operations: [], navigate() {}, inspect: async () => {}, onoperation() {} };
  const connected = visibleText(
    await renderComponent('./components/CommandPalette.svelte', { ...props, workflows: catalog }),
  );
  assert.match(
    connected,
    /LOAD A WORKFLOW Type a title or file name to find a packaged workflow\./,
  );
  assert.doesNotMatch(connected, /Load workflow:/, 'an empty query lists no workflow rows');
  const offline = visibleText(
    await renderComponent('./components/CommandPalette.svelte', { ...props, workflows: [] }),
  );
  assert.doesNotMatch(offline, /LOAD A WORKFLOW/);
});
