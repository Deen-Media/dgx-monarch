import test from 'node:test';
import assert from 'node:assert/strict';
import { renderComponent, visibleText } from './renderHarness.js';
import { formatBindings, graph, widgetBindings } from './workflowFixtures.js';
import { durationGroups } from './frameCounts.ts';
import { emptyDraft } from './model.ts';
import type { Binding } from './model.ts';

const H3 = ['MiniMaxH3ImageToVideo', 'EmptyMiniMaxH3LatentAV'];
/** The H3 MP4 graph with the conditioning and latent inputs, its save choices and draft values. */
function h3(stem = 'minimax-h3-mp4') {
  const workflow = graph(stem);
  const bindings: Binding[] = [...widgetBindings(workflow, H3), ...formatBindings(workflow)].map(
    (binding: Binding) => ({ ...binding, prominent: binding.input_name !== 'length' }),
  );
  const values = Object.fromEntries(bindings.map((binding) => [binding.id, binding.value]));
  return { workflow, bindings, values };
}
function generationProps(overrides: Record<string, unknown> = {}) {
  const { workflow, bindings, values } = h3();
  return {
    client: {},
    draft: { ...emptyDraft(), workflow, workflowName: 'dgx-monarch-minimax-h3-mp4', values },
    bindings,
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
const render = (props: Record<string, unknown>) =>
  renderComponent('./views/Generation.svelte', props);
const options = (body: string) =>
  [...body.matchAll(/<option value="([^"]+)"[^>]*>([^<]+)<\/option>/g)]
    .filter(([, value]) =>
      ['image', 'video', 'edit', 'upscale', 'workflow', 'fleet'].includes(value),
    )
    .map(([, value, label]) => `${value}:${label}`);

test('Create names what the H3 MP4 workflow saves and offers only the operations it supports', async () => {
  const body = await render(
    generationProps({ draft: { ...generationProps().draft, operation: 'video' } }),
  );
  assert.match(visibleText(body), /Output: one MP4 video with sound/);
  assert.deepEqual(options(body), ['video:Video', 'workflow:Workflow']);
  assert.doesNotMatch(body, /operation-need/);
  const t2va = h3('minimax-h3-t2va');
  const frames = await render(
    generationProps({
      draft: { ...emptyDraft(), workflow: t2va.workflow, values: t2va.values, operation: 'video' },
      bindings: t2va.bindings,
    }),
  );
  assert.match(visibleText(frames), /Output: PNG frames plus separate FLAC audio/);
});

test('an operation the graph does not support stays selected and names the input it needs', async () => {
  const body = await render(
    generationProps({ draft: { ...generationProps().draft, operation: 'edit' } }),
  );
  assert.deepEqual(options(body), [
    'video:Video',
    'edit:Edit (not supported)',
    'workflow:Workflow',
  ]);
  assert.match(body, /<option value="edit" selected="">Edit \(not supported\)<\/option>/);
  assert.match(body, /aria-describedby="operation-note operation-need"/);
  assert.match(
    visibleText(body),
    /Edit needs an image, video or mask input node, such as Load Image\. This workflow has none\./,
  );
  // Before an inspection lists the inputs nothing is known, so every operation stays listed.
  const unread = await render(generationProps({ bindings: [] }));
  assert.equal(options(unread).length, 6);
  assert.doesNotMatch(visibleText(unread), /Output:/);
});

test('the Inputs tab shows one Duration control after the size fields, writing both lengths', async () => {
  const body = await render(generationProps());
  const text = visibleText(body);
  assert.match(text, /Duration MiniMax H3 clip Frames Seconds/);
  assert.match(text, /124 frames is 5\.17 s at 24 fps\./);
  assert.match(
    text,
    /Duration writes one count to the conditioning node \(node 7\) and the latent node \(node 9\), rounded to the nearest count MiniMax H3 renders exactly: 5 plus a multiple of 17\./,
  );
  // The frame counts stay under All inputs, each named by its node, with its unit and relation.
  const all = visibleText(await render(generationProps({ controlTab: 'advanced' })));
  assert.match(all, /length MiniMax H3 conditioning node, node 7 int · frames/);
  assert.match(all, /length MiniMax H3 latent node, node 9 int · frames/);
  assert.match(
    all,
    /The conditioning node and the latent node describe the same clip, so they need the same frame count\. Duration sets both\./,
  );
  assert.equal(all.match(/Duration MiniMax H3 clip/g)?.length, 1);
});

test('the Duration control shows each count when they differ, and nothing to sync when one is wired', async () => {
  const { workflow, bindings, values } = h3();
  const [group] = durationGroups(workflow, bindings);
  const differ = visibleText(
    await renderComponent('./components/DurationControl.svelte', {
      group,
      values: { ...values, [`${workflow.id}:9:length`]: 141 },
      onchange() {},
    }),
  );
  assert.match(differ, /The frame counts differ: 124 and 141\. Enter a duration to set them all\./);
  const wired = bindings.map((binding) =>
    binding.id === `${workflow.id}:9:length` ? { ...binding, resolved: false } : binding,
  );
  const body = visibleText(
    await render(generationProps({ bindings: wired, controlTab: 'advanced' })),
  );
  assert.doesNotMatch(body, /Duration MiniMax H3 clip/);
  assert.match(
    body,
    /The count of the latent node comes from a wire, so keep this one equal to it\./,
  );
});
