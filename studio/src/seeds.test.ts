import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { emptyDraft, isSeedBinding, restoreDraft, seedChange } from './model.ts';
import { artifactDraft } from './artifactDraft.ts';
import { experimentDefinition } from './experimentRecords.ts';
import { recordedSeeds, recordedSettings, seedText } from './recordedSettings.ts';
import type { Artifact, Job } from './contracts.ts';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';

const workflow = { nodes: [], extra: {} };
const seedField = (node: string, name = 'seed', execution = node) => ({
  id: `main:${node}:${name}`,
  node_id: node,
  execution_node_id: execution,
  input_name: name,
  schema: { type: 'INT', min: 0, max: 2 ** 53 - 1 },
  value: 0,
});
const steps = { id: 'main:8:steps', node_id: '8', input_name: 'steps', schema: { type: 'INT' } };
const draft = (values: Record<string, unknown> = {}) => ({
  ...emptyDraft(),
  workflow,
  workflowName: 'Kroma INT8',
  values: { 'main:8:seed': 0, 'main:8:steps': 20, ...values },
});
const summary = (seeds: unknown, state = 'succeeded') =>
  ({
    summary: true,
    id: 'job-1',
    state,
    created_at: 1,
    submission: { requested: { operation: 'image' }, effective: { seeds } },
  }) as Job;

test('new drafts choose seeds at review; a record that names no policy keeps fixed', () => {
  assert.equal(emptyDraft().seed_policy, 'random');
  assert.equal(restoreDraft({}).seed_policy, 'fixed');
  assert.equal(restoreDraft({ seed_policy: 'random' }).seed_policy, 'random');
  assert.equal(restoreDraft({ seed_policy: 'fixed' }).seed_policy, 'fixed');
  // A saved grid definition and a reused request without a policy ran with the fixed default.
  const grid = experimentDefinition({
    grid_request: { draft: { workflow }, axes: [{ name: 'main:8:steps', values: [10, 20] }] },
  });
  assert.equal(grid.draft.seed_policy, 'fixed');
  const job = { id: 'j', state: 'failed', created_at: 1, submission: { workflow } } as Job;
  assert.equal(artifactDraft({ id: 'a', job_id: 'j' } as Artifact, job).seed_policy, 'fixed');
});

test('seed fields follow the companion rule: an integer seed or noise_seed input', () => {
  assert.equal(isSeedBinding(seedField('8')), true);
  assert.equal(isSeedBinding(seedField('9', 'noise_seed')), true);
  assert.equal(isSeedBinding({ ...seedField('8'), resolved: false }), false);
  assert.equal(isSeedBinding({ ...seedField('8'), schema: { type: 'COMBO' } }), false);
  assert.equal(isSeedBinding(steps), false);
});

test('recorded seeds keep every digit, and a rounded number reads as unknown', () => {
  assert.deepEqual(recordedSeeds(summary({ '8.seed': 2 ** 53 - 1 })), { '8.seed': 2 ** 53 - 1 });
  assert.deepEqual(recordedSeeds(summary({})), {});
  // The companion sends a seed above 2**53 - 1 as its digits; text that is not a whole number drops.
  const large = { '8.seed': '18446744073709551615', '9.seed': '9007199254740992', '1.seed': 'x' };
  assert.deepEqual(recordedSeeds(summary(large)), {
    '8.seed': '18446744073709551615',
    '9.seed': '9007199254740992',
  });
  // A number past the safe range was rounded by JSON parsing, so the reader asks for the summary.
  assert.equal(recordedSeeds(summary({ '8.seed': 2 ** 53 })), null);
  assert.equal(recordedSeeds(summary(null)), null);
  assert.equal(recordedSeeds(undefined), null);
  // A full record of an imported job has no seed list; its summary carries the prompt's seeds.
  const imported = {
    id: 'external',
    state: 'succeeded',
    created_at: 1,
    submission: { prompt: { '3': { class_type: 'KSampler', inputs: { seed: 11, steps: 20 } } } },
  } as Job;
  assert.equal(recordedSeeds(imported), null);
  assert.equal(seedText({ '3.seed': 11 }), 'Seed 11');
  assert.equal(seedText({ '8.seed': '18446744073709551615' }), 'Seed 18446744073709551615');
  assert.equal(
    seedText({ '3.seed': 11, '4.noise_seed': 0 }),
    'Seeds 3.seed = 11, 4.noise_seed = 0',
  );
  assert.equal(seedText({}), '');
  assert.equal(seedText(null), '');
});

test('Use this seed fills the matching seed field and sets the Fixed policy', () => {
  const fields = [seedField('8'), steps];
  const used = seedChange(draft(), fields, { '8.seed': 4242 });
  assert.deepEqual(used, {
    change: { values: { 'main:8:seed': 4242, 'main:8:steps': 20 }, seed_policy: 'fixed' },
    notice:
      'Seed 4242 is now in the draft, and the seed policy is Fixed. Undo restores the previous draft.',
  });
  // A subgraph instance matches by the execution identity the record uses.
  const nested = seedField('3', 'seed', '__studio_subgraph__2/3');
  const inside = seedChange(draft(), [nested, seedField('8')], {
    '__studio_subgraph__2/3.seed': 7,
  });
  assert.deepEqual('change' in inside && inside.change.values, {
    'main:8:seed': 0,
    'main:8:steps': 20,
    'main:3:seed': 7,
  });
  // One seed fits a draft with one seed field even when the node ids differ.
  const moved = seedChange(draft(), fields, { '12.seed': 9 });
  assert.equal('change' in moved && moved.change.values!['main:8:seed'], 9);
  // Two seeds, one matched: the notice says what was left.
  const partial = seedChange(draft(), fields, { '8.seed': 1, '13.noise_seed': 2 });
  assert.match('notice' in partial ? partial.notice : '', /1 recorded seed\(s\) matched no seed/);
});

test('Use this seed refuses when no seed field fits, naming both counts', () => {
  const fields = [seedField('8'), seedField('9', 'noise_seed')];
  assert.deepEqual(seedChange(draft(), fields, { '12.seed': 9 }), {
    error:
      'The draft has 2 seed field(s), and none matches the node of the 1 recorded seed(s). Reuse settings loads the workflow that made this output.',
  });
  assert.deepEqual(seedChange({ ...draft(), workflow: null }, fields, { '8.seed': 1 }), {
    error: 'Load a workflow before using a recorded seed.',
  });
  assert.deepEqual(seedChange(draft(), fields, {}), { error: 'This output has no recorded seed.' });
});

test('Use this seed refuses a seed a draft field cannot hold exactly, and never rounds it', () => {
  const fields = [seedField('8'), steps];
  const limit =
    'is too large for a draft field, which holds whole numbers exactly only up to 9007199254740991. The draft is unchanged.';
  for (const seed of ['18446744073709551615', '9007199254740992'])
    assert.deepEqual(seedChange(draft(), fields, { '8.seed': seed }), {
      error: `Seed ${seed} ${limit}`,
    });
  // The largest safe seed still applies as a number.
  const safe = seedChange(draft(), fields, { '8.seed': 9007199254740991 });
  assert.equal('change' in safe && safe.change.values!['main:8:seed'], 9007199254740991);
  // A large seed that matches no field stays out of the draft; the matching one applies.
  const mixed = seedChange(draft(), fields, {
    '8.seed': 7,
    '13.noise_seed': '18446744073709551615',
  });
  assert.equal('change' in mixed && mixed.change.values!['main:8:seed'], 7);
  assert.match('notice' in mixed ? mixed.notice : '', /1 recorded seed\(s\) matched no seed field/);
});

// Run the shipped App draft handlers: one seed change, one Undo.
const appCode = await componentFunctions('./App.svelte', [
  'changeDraft',
  'undo',
  'importDraft',
  'useSeed',
  'setNotice',
]);
const appHarness = new Function(
  'seedChange',
  'start',
  'fields',
  `'use strict';
  const $state = { snapshot: (value) => value };
  const guard = { edit() {} };
  const submission = { preview: { id: 'stale review' } };
  let draft = start, undoDraft = null, dirty = false, notice = '', error = '', workflowId = 'kroma';
  let bindings = fields, connections = [], bindingErrors = [], rawWorkflow = '';
  // Undo inspects the restored workflow again, which lists the same bindings.
  const inspect = async () => { bindings = fields; };
  ${appCode}
  return { useSeed, undo, view: () => ({ draft, dirty, notice, error, preview: submission.preview }) };`,
);

test('Use this seed is one draft change that clears the review, and Undo restores the draft', async () => {
  const app = appHarness(seedChange, draft(), [seedField('8'), steps]);
  app.useSeed({ '8.seed': 4242 });
  let view = app.view();
  assert.equal(view.draft.values['main:8:seed'], 4242);
  assert.equal(view.draft.seed_policy, 'fixed');
  assert.equal(view.dirty, true);
  assert.equal(view.preview, null);
  assert.match(view.notice, /^Seed 4242 is now in the draft/);
  await app.undo();
  view = app.view();
  assert.equal(view.draft.values['main:8:seed'], 0);
  assert.equal(view.draft.seed_policy, 'random');
  assert.equal(view.notice, 'Draft change undone.');
  // A refusal changes nothing and says why.
  const refused = appHarness(seedChange, draft(), [seedField('8'), seedField('9', 'noise_seed')]);
  refused.useSeed({ '12.seed': 9 });
  assert.match(refused.view().error, /none matches the node/);
  assert.equal(refused.view().draft.values['main:8:seed'], 0);
  assert.deepEqual(refused.view().preview, { id: 'stale review' });
});

test('the viewer footer shows the recorded seed and offers Use this seed', async () => {
  const used: unknown[] = [];
  const body = async (props: Record<string, unknown>) =>
    visibleText(
      await renderComponent('./components/OutputSeeds.svelte', {
        client: {},
        jobId: 'job-1',
        onuseseed: (seeds: unknown) => used.push(seeds),
        ...props,
      }),
    );
  assert.equal(await body({ job: summary({ '8.seed': 4242 }) }), 'Seed 4242 Use this seed');
  assert.equal(
    await body({ job: summary({ '8.seed': 1, '13.noise_seed': 2 }) }),
    'Seeds 8.seed = 1, 13.noise_seed = 2 Use these seeds',
  );
  assert.equal(await body({ job: summary({ '8.seed': 1 }), onuseseed: undefined }), 'Seed 1');
  assert.equal(await body({ job: summary({}) }), 'No seed input');
  assert.equal(
    await body({ job: summary({ '8.seed': '18446744073709551615' }) }),
    'Seed 18446744073709551615 Use this seed',
  );
  // Outside the snapshot the job's summary loads first.
  assert.equal(await body({ job: undefined }), 'Reading the recorded seed…');
  const viewer = (jobs?: Job[]) =>
    renderComponent('./components/Viewer.svelte', {
      artifact: { id: 'out', job_id: 'job-1', kind: 'image', filename: 'out.png', created_at: 1 },
      client: {},
      jobs,
      onuseseed: () => {},
      onreuse: async () => {},
      onselect: () => {},
    });
  assert.match(visibleText(await viewer([summary({ '8.seed': 4242 })])), /Seed 4242 Use this seed/);
  // Presentation mode passes no jobs, and its hidden footer reads nothing.
  assert.doesNotMatch(await viewer(), /Use this seed|Reading the recorded seed/);
});

test('Queue rows show the seeds of their job', async () => {
  const body = visibleText(
    await renderComponent('./views/Queue.svelte', {
      connection: {
        client: {},
        snapshot: {
          jobs: [
            summary({ '8.seed': 4242 }, 'queued'),
            { ...summary(null, 'running'), id: 'job-2' },
            { ...summary({ '3.seed': '18446744073709551615' }, 'queued'), id: 'job-3' },
          ],
        },
        capabilities: {},
      },
      busy: false,
      refresh: async () => {},
      fail: () => {},
      jobAction: async () => {},
    }),
  );
  assert.match(body, /queued image job-1 Seed 4242/);
  assert.match(body, /running image job-2 \S+/);
  assert.doesNotMatch(body, /job-2 Seed/);
  assert.match(body, /queued image job-3 Seed 18446744073709551615/);
});

test('under Random the seed field reads Random each run and says how to set it', async () => {
  const props = (seed_policy: string) => ({
    client: {},
    draft: { ...draft(), seed_policy },
    bindings: [
      { ...seedField('8'), prominent: true },
      { ...steps, prominent: true },
    ],
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
  });
  const random = visibleText(await renderComponent('./views/Generation.svelte', props('random')));
  assert.ok(
    random.includes(
      'Random: each job gets new seeds. Review settings shows them before it starts.',
    ),
  );
  assert.equal(random.split('Choose the Fixed seed policy to set this seed.').length, 2);
  const fixed = visibleText(await renderComponent('./views/Generation.svelte', props('fixed')));
  assert.ok(fixed.includes('Fixed: jobs use the seed fields as they are.'));
  assert.doesNotMatch(fixed, /Chosen for each job|Choose the Fixed seed policy/);
});

// Run the shipped Gallery copy handler against a fake clipboard.
const copyCode = await componentFunctions('./views/Gallery.svelte', ['copy']);
const galleryCopy = new Function(
  'recordedSeeds',
  'recordedSettings',
  'fullRecord',
  'summaryRecord',
  `'use strict';
  const copied = [], notices = [], fetched = [];
  const client = {};
  const navigator = { clipboard: { writeText: async (text) => copied.push(text) } };
  const setNotice = (text) => notices.push(text);
  const exactJob = async (_client, id) => (fetched.push('full ' + id), fullRecord);
  const jobSummary = async (_client, id) => (fetched.push('summary ' + id), summaryRecord);
  ${copyCode}
  return async (entry, kind) => (await copy(entry, kind), { copied, notices, fetched });`,
);

test('Gallery copies the seed alone, and named inputs through a separate action', async () => {
  const full = {
    id: 'job-1',
    state: 'succeeded',
    created_at: 1,
    submission: {
      prompt: { '8': { class_type: 'KSampler', inputs: { seed: 4242, steps: 20 } } },
      effective: { seeds: { '8.seed': 4242 } },
    },
  };
  const entry = { artifact: { id: 'out', job_id: 'job-1' }, job: summary({ '8.seed': 4242 }) };
  const copy = galleryCopy(recordedSeeds, recordedSettings, full);
  // The snapshot row carries the seed, so no full record loads for it.
  assert.deepEqual(await copy(entry, 'seed'), {
    copied: ['4242'],
    notices: ['Copied the recorded seed 4242.'],
    fetched: [],
  });
  const two = galleryCopy(recordedSeeds, recordedSettings, full);
  const both = await two({ ...entry, job: summary({ '8.seed': 1, '9.noise_seed': 2 }) }, 'seed');
  assert.deepEqual(JSON.parse(both.copied[0]), { '8.seed': 1, '9.noise_seed': 2 });
  assert.deepEqual(both.notices, ['Copied the recorded seeds.']);
  const inputs = await galleryCopy(recordedSeeds, recordedSettings, full)(entry, 'inputs');
  assert.deepEqual(JSON.parse(inputs.copied[0]), { '8': { seed: 4242, steps: 20 } });
  assert.deepEqual(inputs.fetched, ['full job-1']);
  // A seed above 2**53 - 1 copies as its exact digits.
  const large = summary({ '8.seed': '18446744073709551615' });
  assert.deepEqual(
    await galleryCopy(recordedSeeds, recordedSettings, full)({ ...entry, job: large }, 'seed'),
    {
      copied: ['18446744073709551615'],
      notices: ['Copied the recorded seed 18446744073709551615.'],
      fetched: [],
    },
  );
  // Without a snapshot row the copy reads the job's summary, never the rounded full record.
  const outside = await galleryCopy(
    recordedSeeds,
    recordedSettings,
    full,
    large,
  )({ ...entry, job: undefined }, 'seed');
  assert.deepEqual(outside.copied, ['18446744073709551615']);
  assert.deepEqual(outside.fetched, ['summary job-1']);
  // Each copy has its own action in the card menu.
  const source = readFileSync(new URL('./views/Gallery.svelte', import.meta.url), 'utf8');
  assert.match(source, /copy\(entry, 'seed'\)\}>Copy seed</);
  assert.match(source, /copy\(entry, 'inputs'\)\}>Copy named inputs</);
  assert.doesNotMatch(source, /Copy named inputs \/ seeds/);
});

test('one job summary reads by identity and refuses a different job', async () => {
  const { jobSummary } = await import('./jobProvenance.ts');
  const asked: string[] = [];
  const client = (items: unknown[]) => ({
    request: async (path: string) => (asked.push(path), { items }),
  });
  const row = summary({ '8.seed': '18446744073709551615' });
  assert.equal(await jobSummary(client([row]) as never, 'job-1'), row);
  assert.deepEqual(asked, ['/jobs?job_id=job-1&limit=1']);
  await assert.rejects(jobSummary(client([]) as never, 'job-1'), /no summary for this job/);
  await assert.rejects(
    jobSummary(client([{ ...row, id: 'other' }]) as never, 'job-1'),
    /no summary for this job/,
  );
});
