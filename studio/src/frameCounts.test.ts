import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bundled,
  graph,
  recorded as recordedSchema,
  widgetBindings as fixtureBindings,
} from './workflowFixtures.js';
import type { Binding } from './model.ts';
import type { ObjectInfo } from './workflowGraph.ts';
import {
  clipFrames,
  clipRate,
  clipSeconds,
  durationGroups,
  inputMetadata,
  setDuration,
  snapFrames,
  synchronized,
} from './frameCounts.ts';

const recorded = recordedSchema as ObjectInfo;
const widgetBindings = (workflow: Record<string, any>, types: string[]): Binding[] =>
  fixtureBindings(workflow, types);
const H3 = ['MiniMaxH3ImageToVideo', 'EmptyMiniMaxH3LatentAV'];
const values = (bindings: Binding[]) =>
  Object.fromEntries(bindings.map((binding) => [binding.id, binding.value]));

test('each MiniMax H3 workflow pairs its two length inputs in one 24 fps Duration control', () => {
  for (const stem of ['minimax-h3-mp4', 'minimax-h3-t2va', 'minimax-h3-guide']) {
    const workflow = graph(stem);
    const bindings = widgetBindings(workflow, H3);
    const groups = durationGroups(workflow, bindings);
    assert.equal(groups.length, 1, stem);
    const [group] = groups;
    assert.equal(group.family, 'MiniMax H3');
    assert.deepEqual(
      group.members.map((member) => [member.role, member.binding.input_name]),
      [
        ['conditioning node', 'length'],
        ['latent node', 'length'],
      ],
      stem,
    );
    assert.ok(synchronized(group), stem);
    assert.deepEqual(clipFrames(group, values(bindings)), [124, 124], stem);
    assert.deepEqual(clipRate(group, {}), { fps: 24 });
    // 124 frames at 24 fps, the issue's "about 5.2 s".
    assert.equal(clipSeconds(124, 24), 5.17);
    assert.deepEqual([group.min, group.max], [5, 3600]);
  }
});

test('Duration writes one count, on the frame grid H3 renders, to both length inputs', () => {
  const workflow = graph('minimax-h3-mp4');
  const bindings = widgetBindings(workflow, H3);
  const [group] = durationGroups(workflow, bindings);
  // H3 renders 5 + 17k frames, so Duration takes the nearest such count within the limits.
  assert.equal(snapFrames(group, 124), 124);
  assert.equal(snapFrames(group, 130), 124);
  assert.equal(snapFrames(group, 133), 141);
  assert.equal(snapFrames(group, 8 * 24), 192);
  assert.equal(snapFrames(group, 1), 5);
  assert.equal(snapFrames(group, 99999), 3592);
  const draft = { ...values(bindings), other: 'kept' };
  const next = setDuration(group, snapFrames(group, 141), draft);
  const lengths = group.members.map((member) => member.binding.id);
  assert.deepEqual(lengths, [`${workflow.id}:7:length`, `${workflow.id}:9:length`]);
  assert.deepEqual(
    lengths.map((id) => next[id]),
    [141, 141],
  );
  assert.equal(next.other, 'kept');
  assert.equal(next[`${workflow.id}:7:width`], 1344, 'the size is left alone');
  // Counts that differ are each shown, so the control never hides either one.
  assert.deepEqual(clipFrames(group, { ...draft, [lengths[1]]: 141 }), [124, 141]);
});

test('the two H3 length fields carry their node, unit and relation, wired or not', () => {
  const workflow = graph('minimax-h3-mp4');
  const bindings = widgetBindings(workflow, H3);
  const meta = inputMetadata(workflow, bindings, recorded);
  const relation =
    'The conditioning node and the latent node describe the same clip, so they need the same frame count.';
  assert.deepEqual(meta[`${workflow.id}:7:length`], {
    context: 'MiniMax H3 conditioning node, node 7',
    unit: 'frames',
    note: `${relation} Duration sets both.`,
  });
  assert.deepEqual(meta[`${workflow.id}:9:length`], {
    context: 'MiniMax H3 latent node, node 9',
    unit: 'frames',
    note: `${relation} Duration sets both.`,
  });
  assert.equal(meta[`${workflow.id}:7:width`], undefined, 'only frame counts get notes');
  // When a wire feeds one count no draft value can set it, so there is no Duration control and
  // each field says how it relates to the other.
  const wired = bindings.map((binding) =>
    binding.id === `${workflow.id}:9:length` ? { ...binding, resolved: false } : binding,
  );
  const [group] = durationGroups(workflow, wired);
  assert.ok(!synchronized(group));
  assert.equal(
    inputMetadata(workflow, wired, recorded)[`${workflow.id}:7:length`].note,
    `${relation} The count of the latent node comes from a wire, so keep this one equal to it.`,
  );
});

test('LTX pairs the video and audio frame counts and reads the clip rate from its frame rate inputs', () => {
  const workflow = graph('ltx25-t2v');
  const bindings = widgetBindings(workflow, [
    'EmptyLTXVLatentVideo',
    'LTXVEmptyLatentAudio',
    'LTXVConditioning',
  ]);
  const [group] = durationGroups(workflow, bindings);
  assert.equal(group.family, 'LTX');
  assert.deepEqual(
    group.members.map((member) => [member.role, member.binding.input_name]),
    [
      ['video latent', 'length'],
      ['audio latent', 'frames_number'],
    ],
  );
  assert.deepEqual(clipFrames(group, values(bindings)), [121, 121]);
  assert.deepEqual(clipRate(group, values(bindings)), { fps: 24 });
  assert.deepEqual([group.min, group.max], [1, 1000]);
  assert.equal(snapFrames(group, 120), 121, 'LTX keeps 1 + 8k frames');
  const rate = `${workflow.id}:7:frame_rate`;
  assert.deepEqual(clipRate(group, { ...values(bindings), [rate]: 25 }), {
    fps: null,
    reason: 'The frame rate inputs differ: 25 and 24 fps.',
  });
  const meta = inputMetadata(workflow, bindings, recorded);
  assert.equal(meta[rate].unit, 'fps');
  assert.equal(meta[`${workflow.id}:11:frames_number`].context, 'LTX audio latent, node 11');
});

test('a frame count no family pairs gets its unit, and its node when another shares its name', () => {
  const dancer = graph('wandancer');
  const bindings = widgetBindings(dancer, ['WanDancerVideo']);
  assert.deepEqual(durationGroups(dancer, bindings), []);
  const meta = inputMetadata(dancer, bindings, recorded);
  assert.deepEqual(meta[`${dancer.id}:16:length`], { unit: 'frames', context: 'node 16' });
  assert.deepEqual(meta[`${dancer.id}:24:length`], { unit: 'frames', context: 'node 24' });
  const wan = graph('wan-t2v');
  const single = widgetBindings(wan, ['EmptyHunyuanLatentVideo']);
  assert.deepEqual(inputMetadata(wan, single, recorded)[`${wan.id}:8:length`], { unit: 'frames' });
  assert.deepEqual(durationGroups(graph('kroma-t2i'), []), []);
});

const socket = (name: string, type: string, link: number | null = null) => ({ name, type, link });
const out = (type: string, links: number[]) => ({ type, links });
/** One H3 clip: a conditioning node and a latent node that both feed one sampler. */
function h3Clip(base: number, link: number) {
  const [cond, latent, sampler] = [base, base + 1, base + 2];
  return {
    nodes: [
      { id: cond, type: 'MiniMaxH3ImageToVideo', outputs: [out('CONDITIONING', [link])] },
      { id: latent, type: 'EmptyMiniMaxH3LatentAV', outputs: [out('LATENT', [link + 1])] },
      {
        id: sampler,
        type: 'DGXMonarchKSampler',
        inputs: [
          socket('model', 'DGXM_MODEL'),
          socket('positive', 'CONDITIONING', link),
          socket('latent_image', 'LATENT', link + 1),
        ],
        outputs: [out('LATENT', [])],
      },
    ],
    links: [
      [link, cond, 0, sampler, 1, 'CONDITIONING'],
      [link + 1, latent, 0, sampler, 2, 'LATENT'],
    ],
  };
}
/** A length binding as the inspection lists it, for a node of a graph or a subgraph instance. */
function length(graphId: string, node: number, value: number, path: string[] = []): Binding {
  const scope = path.length ? `@instance/root/${path.join('/')}` : graphId;
  return {
    id: `${scope}:${node}:length`,
    graph_id: graphId,
    instance_path: path,
    node_id: String(node),
    input_name: 'length',
    value,
    resolved: true,
    schema: { type: 'INT', min: 5, max: 3600, step: 17 },
  };
}

test('two H3 clips in one graph get one Duration group each, and each writes only its own counts', () => {
  const [first, second] = [h3Clip(1, 1), h3Clip(4, 3)];
  const workflow = {
    id: 'two',
    nodes: [...first.nodes, ...second.nodes],
    links: [...first.links, ...second.links],
  };
  const bindings = [
    length('two', 1, 124),
    length('two', 2, 124),
    length('two', 4, 192),
    length('two', 5, 192),
  ];
  const groups = durationGroups(workflow, bindings);
  assert.deepEqual(
    groups.map((group) => [group.clip, group.members.map((m) => m.binding.id), group.samplers]),
    [
      [1, ['two:1:length', 'two:2:length'], [JSON.stringify(['3'])]],
      [2, ['two:4:length', 'two:5:length'], [JSON.stringify(['6'])]],
    ],
  );
  assert.ok(groups.every(synchronized));
  const draft = Object.fromEntries(bindings.map((b) => [b.id, b.value]));
  const next = setDuration(groups[0], 141, draft);
  assert.deepEqual(
    bindings.map((b) => next[b.id]),
    [141, 141, 192, 192],
  );
  assert.deepEqual(clipFrames(groups[1], draft), [192, 192]);
  const meta = inputMetadata(workflow, bindings);
  assert.equal(meta['two:4:length'].context, 'MiniMax H3 clip 2 conditioning node, node 4');
  // A count whose sampler cannot be found keeps its own group, which no control sets.
  const stray = {
    ...workflow,
    nodes: [...workflow.nodes, { id: 9, type: 'EmptyMiniMaxH3LatentAV' }],
  };
  const lone = durationGroups(stray, [...bindings, length('two', 9, 300)]).at(-1)!;
  assert.deepEqual([lone.members.map((m) => m.binding.id), lone.samplers], [['two:9:length'], []]);
  assert.ok(!synchronized(lone));
  assert.equal(
    inputMetadata(stray, [...bindings, length('two', 9, 300)])['two:9:length'].note,
    undefined,
  );
});

test('each instance of a repeated H3 subgraph gets its own Duration group and keeps its counts', () => {
  const clip = h3Clip(1, 1);
  const definition = {
    id: 'clip-def',
    nodes: clip.nodes,
    links: clip.links.map(([id, origin_id, origin_slot, target_id, target_slot, type]) => ({
      id,
      origin_id,
      origin_slot,
      target_id,
      target_slot,
      type,
    })),
    inputNode: { id: -10 },
    outputNode: { id: -20 },
  };
  const workflow = {
    id: 'root',
    nodes: [
      { id: 10, type: 'clip-def', mode: 0 },
      { id: 11, type: 'clip-def', mode: 0 },
    ],
    links: [],
    definitions: { subgraphs: [definition] },
  };
  const bindings = [
    length('clip-def', 1, 124, ['10']),
    length('clip-def', 2, 124, ['10']),
    length('clip-def', 1, 192, ['11']),
    length('clip-def', 2, 192, ['11']),
  ];
  const groups = durationGroups(workflow, bindings);
  assert.deepEqual(
    groups.map((group) => group.members.map((m) => m.binding.id)),
    [
      ['@instance/root/10:1:length', '@instance/root/10:2:length'],
      ['@instance/root/11:1:length', '@instance/root/11:2:length'],
    ],
  );
  const draft = Object.fromEntries(bindings.map((b) => [b.id, b.value]));
  assert.deepEqual(
    groups.map((group) => clipFrames(group, draft)),
    [
      [124, 124],
      [192, 192],
    ],
  );
  const next = setDuration(groups[1], 141, draft);
  assert.deepEqual(
    bindings.map((b) => next[b.id]),
    [124, 124, 141, 141],
  );
  assert.equal(
    inputMetadata(workflow, bindings)['@instance/root/11:1:length'].context,
    'MiniMax H3 clip 2 conditioning node, node 11/1',
  );
});

test('a wired frame rate leaves seconds unknown unless a primitive node holds it', () => {
  const LTX = ['EmptyLTXVLatentVideo', 'LTXVEmptyLatentAudio', 'LTXVConditioning'];
  const wire = (source: Record<string, unknown>) => {
    const workflow = structuredClone(graph('ltx25-t2v'));
    workflow.nodes
      .find((node: any) => node.id === 7)
      .inputs.push({ name: 'frame_rate', type: 'FLOAT', widget: { name: 'frame_rate' }, link: 99 });
    workflow.nodes.push({ id: 30, ...source });
    workflow.links.push([99, 30, 0, 7, 2, 'FLOAT']);
    return workflow;
  };
  const video = wire({ type: 'GetVideoComponents', outputs: [out('FLOAT', [99])] });
  const fed = widgetBindings(video, LTX);
  assert.equal(fed.find((b) => b.id === `${video.id}:7:frame_rate`)!.resolved, false);
  const [group] = durationGroups(video, fed);
  assert.deepEqual(clipRate(group, values(fed)), {
    fps: null,
    reason: "A wire sets frame_rate on node 7, so the clip's frame rate is unknown.",
  });
  const held = wire({
    type: 'PrimitiveNode',
    outputs: [out('FLOAT', [99])],
    widgets_values: [24, 'fixed'],
  });
  const primitive = (rate: number): Binding => ({
    id: `${held.id}:30:value`,
    graph_id: String(held.id),
    node_id: '30',
    input_name: 'value',
    value: rate,
  });
  const bindings = [...widgetBindings(held, LTX), primitive(24)];
  const [known] = durationGroups(held, bindings);
  assert.deepEqual(clipRate(known, values(bindings)), { fps: 24 });
  assert.deepEqual(clipRate(known, { ...values(bindings), [`${held.id}:30:value`]: 30 }), {
    fps: null,
    reason: 'The frame rate inputs differ: 30 and 24 fps.',
  });
});

test('every bundled H3 and LTX workflow finds its sampler and gets one synchronized Duration control', () => {
  const RULED = [
    'MiniMaxH3ImageToVideo',
    'MiniMaxH3ReferenceToVideo',
    'EmptyMiniMaxH3LatentAV',
    'EmptyLTXVLatentVideo',
    'LTXVImgToVideo',
    'LTXVEmptyLatentAudio',
    'LTXVConditioning',
  ];
  const seen: string[] = [];
  for (const row of bundled) {
    const groups = durationGroups(row.workflow, widgetBindings(row.workflow, RULED));
    if (!groups.length) continue;
    seen.push(row.name.replace(/^dgx-monarch-/, ''));
    assert.equal(groups.length, 1, row.name);
    assert.ok(synchronized(groups[0]), row.name);
    assert.equal(groups[0].samplers.length, 1, row.name);
  }
  assert.deepEqual(seen, [
    'ltx-i2v-av',
    'ltx-t2v',
    'ltx25-flf2v',
    'ltx25-i2v-guide',
    'ltx25-i2v',
    'ltx25-t2v',
    'minimax-h3-guide',
    'minimax-h3-mp4',
    'minimax-h3-t2va',
  ]);
});
