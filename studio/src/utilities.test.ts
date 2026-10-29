import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { WorkflowEntry } from './contracts.ts';
import { emptyDraft, type Draft } from './model.ts';
import {
  UTILITIES,
  destinations,
  isUtility,
  navigation,
  railView,
  resolveView,
  utilities,
} from './navigation.ts';
import {
  FLEET_SAMPLER,
  FLEET_WORKFLOW,
  fleetNextStep,
  fleetPlan,
  fleetPlanText,
  hasFleetSampler,
} from './fleetSetup.ts';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';

const read = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8');
const catalog: { workflows: WorkflowEntry[] } = JSON.parse(
  read('../../src/dgx_monarch/studio/data/workflows.json'),
);
const fleetEntry = catalog.workflows.find((entry) => entry.id === FLEET_WORKFLOW)!;
const presentation = JSON.parse(read('../../src/dgx_monarch/studio/data/presentation.json'));
const other = { nodes: [{ id: 3, type: 'KSampler' }] };

test('the rail groups Grids, Fleet and Operations under Utilities; the palette lists each', () => {
  assert.deepEqual(
    navigation.map((item) => item.label),
    ['Create', 'Gallery', 'Queue', 'Utilities', 'Library', 'Settings'],
  );
  assert.deepEqual(
    destinations.map((item) => item.id),
    ['workspace', 'gallery', 'queue', 'grid', 'fleet', 'operations', 'projects', 'settings'],
  );
  assert.ok(utilities.every((item) => item.group === 'Utilities' && item.summary));
  assert.deepEqual(['grid', 'fleet', 'operations', 'queue', UTILITIES].map(railView), [
    UTILITIES,
    UTILITIES,
    UTILITIES,
    'queue',
    UTILITIES,
  ]);
  assert.equal(isUtility(UTILITIES), false, 'the alias is not a view of its own');
  // Route ids stay stable; the alias opens the utility tab used last, Grids at first.
  assert.equal(resolveView(UTILITIES), 'grid');
  assert.equal(resolveView('operations'), 'operations');
  assert.equal(resolveView('queue'), 'queue');
  assert.equal(resolveView(UTILITIES), 'operations');
  assert.equal(resolveView('fleet'), 'fleet');
  assert.equal(resolveView(UTILITIES), 'fleet');
  const app = read('./App.svelte');
  assert.match(app, /view = resolveView\(id\);/);
  assert.match(app, /view=\{railView\(view\)\}/);
  assert.match(app, /\{:else if isUtility\(view\)\}<Utilities/);
  const palette = read('./components/CommandPalette.svelte');
  assert.match(palette, /destinations\.filter/);
});

const connection = (workflows: WorkflowEntry[] = [fleetEntry]) => ({
  client: { request: async () => ({ items: [], next_cursor: null, records: [] }) },
  snapshot: { jobs: [], operations: [], telemetry: {}, artifacts: [], records: {}, driver: {} },
  metricHistory: [],
  capabilities: { presentation, operations: [] },
  workflows,
});
const utility = (view: string, changes: Record<string, unknown> = {}) =>
  renderComponent('./views/Utilities.svelte', {
    view,
    connection: connection(),
    draft: emptyDraft(),
    bindings: [],
    reuse: async () => {},
    restoreDraft: async () => {},
    changeDraft: () => {},
    inspect: async () => {},
    navigate: () => {},
    setNotice: () => {},
    request: null,
    onrefresh: () => {},
    ...changes,
  });

test('Utilities opens each tab with its one-line purpose and keeps the page explanations', async () => {
  for (const [view, heading] of [
    ['grid', 'What a grid does'],
    ['fleet', 'What Fleet does'],
    ['operations', 'Operations & diagnostics'],
  ]) {
    const body = await utility(view);
    const tabs = body.slice(body.indexOf('<nav'), body.indexOf('</nav>'));
    const current = [...tabs.matchAll(/<button[^>]*aria-current="page"[^>]*>([^<]*)</g)];
    const tab = utilities.find((item) => item.id === view)!;
    assert.deepEqual(
      current.map((match) => match[1]),
      [tab.label],
    );
    assert.deepEqual(
      [...tabs.matchAll(/<button[^>]*>([^<]*)</g)].map((match) => match[1]),
      ['Grids', 'Fleet', 'Operations'],
    );
    const text = visibleText(body);
    assert.ok(text.includes(tab.summary!), view);
    assert.ok(text.includes(heading.replace('&', '&amp;')) || text.includes(heading), heading);
  }
});

test('the Fleet page says what Fleet does, when to use it and what Set up Fleet will do', async () => {
  const text = visibleText(await utility('fleet'));
  for (const part of [
    'Each line of the prompts input becomes its own job, and the jobs go to the GPUs in turn, one job per GPU at a time.',
    'Line 1 uses seed noise_seed, line 2 noise_seed + 1 and so on, and the outputs come back in prompt order.',
    'Use Fleet when you want more pictures per hour',
    'on one GPU Fleet gains nothing.',
    'Set up Fleet loads the packaged workflow "Fleet: many prompts, one per GPU" into Create, sets Operation to Fleet and opens Create.',
    'Operation Fleet only labels the job for Gallery and this page; the Fleet sampler node in the workflow is what spreads the prompts over the GPUs.',
    'Set up Fleet in Create',
    'A run takes up to 1,024 jobs.',
  ])
    assert.ok(text.includes(part), part);
  const kept = visibleText(
    await utility('fleet', { draft: { ...emptyDraft(), workflow: fleetEntry.workflow } }),
  );
  assert.ok(
    kept.includes('Your Create draft already holds a Fleet sampler. Set up Fleet keeps it'),
  );
});

test('a Fleet sampler counts only where it runs, as the Operation select reads the graph', () => {
  assert.equal(hasFleetSampler(fleetEntry.workflow), true, 'the packaged Fleet workflow');
  assert.equal(hasFleetSampler(other), false);
  const sub = 'a1b2c3d4-0000-4000-8000-000000000001';
  const nested = (mode: number, instance = true) => ({
    nodes: [{ id: 3, type: 'KSampler' }, ...(instance ? [{ id: 7, type: sub, mode }] : [])],
    links: [],
    definitions: { subgraphs: [{ id: sub, nodes: [{ id: 1, type: FLEET_SAMPLER }], links: [] }] },
  });
  assert.equal(hasFleetSampler(nested(0)), true, 'inside a running subgraph instance');
  assert.equal(hasFleetSampler(nested(2)), false, 'a muted instance runs nothing');
  assert.equal(hasFleetSampler(nested(0, false)), false, 'a definition no instance uses');
  for (const mode of [2, 4])
    assert.equal(hasFleetSampler({ nodes: [{ id: 9, type: FLEET_SAMPLER, mode }] }), false);
  assert.equal(hasFleetSampler({ 9: { class_type: FLEET_SAMPLER, inputs: {} } }), true);
  assert.equal(hasFleetSampler({ 3: { class_type: 'KSampler', inputs: {} } }), false);
  for (const value of [null, undefined, 'text', [], {}])
    assert.equal(hasFleetSampler(value), false);
  assert.equal(fleetEntry.title, 'Fleet: many prompts, one per GPU');
});

test('the Fleet plan keeps a Fleet draft, loads the packaged workflow, or says it is missing', () => {
  const draft = (workflow: Draft['workflow']) => ({ ...emptyDraft(), workflow });
  assert.deepEqual(fleetPlan(draft(fleetEntry.workflow), [fleetEntry]), { kind: 'keep' });
  assert.deepEqual(fleetPlan(draft(other), [fleetEntry]), {
    kind: 'load',
    entry: fleetEntry,
    replaces: true,
  });
  assert.deepEqual(fleetPlan(draft(null), [fleetEntry]), {
    kind: 'load',
    entry: fleetEntry,
    replaces: false,
  });
  assert.deepEqual(fleetPlan(draft(other), []), { kind: 'missing' });
  const replacing = fleetPlanText(fleetPlan(draft(other), [fleetEntry]));
  assert.match(replacing, /into Create in place of your current workflow, sets Operation to Fleet/);
  assert.match(fleetPlanText({ kind: 'missing' }), new RegExp(`a ${FLEET_SAMPLER} node`));
  const next =
    'Fleet is ready in Create. Next, put your prompts in the prompts input, one per line: each line becomes its own job, line 1 uses noise_seed, line 2 noise_seed + 1 and so on. Then generate as usual.';
  assert.equal(fleetNextStep({ kind: 'keep' }), next);
  assert.equal(
    fleetNextStep({ kind: 'load', entry: fleetEntry, replaces: true }),
    `${next} Undo brings back the workflow you had before.`,
  );
  assert.equal(fleetNextStep({ kind: 'load', entry: fleetEntry, replaces: false }), next);
  assert.match(fleetNextStep({ kind: 'missing' }), /would render as one ordinary job/);
});

// Run the shipped Set up Fleet handler against fakes of App's callbacks.
const setUpCode = await componentFunctions('./views/Fleet.svelte', ['setUp']);
function fleetPage(start: Draft, loads: boolean) {
  return new Function(
    'start',
    'loads',
    'helpers',
    `'use strict';
    const { fleetPlan, fleetNextStep, hasFleetSampler, entry } = helpers;
    const calls = [];
    let draft = start, loading = false;
    const plan = fleetPlan(start, [entry]);
    const changeDraft = (change) => { calls.push(['changeDraft', change]); draft = { ...draft, ...change }; };
    // App's inspect reports a refusal itself and leaves the draft as it was.
    const inspect = async (workflow, name, preserve, id) => {
      calls.push(['inspect', name, preserve, id]);
      if (loads) draft = { ...draft, workflow, workflowName: name };
    };
    const loadWorkflow = (item, run) => run(item.workflow, item.name, false, item.id);
    const navigate = (view) => calls.push(['navigate', view]);
    const setNotice = (text) => calls.push(['notice', text]);
    ${setUpCode}
    return { setUp, calls, state: () => ({ draft, loading }) };`,
  )(start, loads, { fleetPlan, fleetNextStep, hasFleetSampler, entry: fleetEntry });
}

test('Set up Fleet loads the Fleet workflow, labels the draft, then opens Create with the next step', async () => {
  const page = fleetPage({ ...emptyDraft(), workflow: other, workflowName: 'mine' }, true);
  await page.setUp();
  assert.deepEqual(page.calls, [
    ['changeDraft', { operation: 'fleet' }],
    ['inspect', 'dgx-monarch-fleet', false, FLEET_WORKFLOW],
    ['navigate', 'workspace'],
    ['notice', fleetNextStep({ kind: 'load', entry: fleetEntry, replaces: true })],
  ]);
  assert.equal(page.state().draft.operation, 'fleet');
  assert.equal(page.state().loading, false);
  // A refused load stays on Fleet, where App shows its error, puts the label back and posts no
  // notice.
  const refused = fleetPage({ ...emptyDraft(), workflow: other }, false);
  await refused.setUp();
  assert.deepEqual(
    refused.calls.map((call) => call.slice(0, 2)),
    [
      ['changeDraft', { operation: 'fleet' }],
      ['inspect', 'dgx-monarch-fleet'],
      ['changeDraft', { operation: 'image' }],
    ],
  );
  assert.equal(refused.state().draft.operation, 'image');
  assert.equal(refused.state().loading, false);
  // A draft that already holds a Fleet sampler and the Fleet label is kept as it is.
  const kept = fleetPage(
    { ...emptyDraft(), workflow: fleetEntry.workflow, operation: 'fleet' },
    true,
  );
  await kept.setUp();
  assert.deepEqual(kept.calls, [
    ['navigate', 'workspace'],
    ['notice', fleetNextStep({ kind: 'keep' })],
  ]);
});
