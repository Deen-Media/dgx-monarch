import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bundled,
  formatBindings as fixtureFormats,
  graph,
  recorded as recordedSchema,
} from './workflowFixtures.js';
import type { StudioClient } from './api.ts';
import type { Binding } from './model.ts';
import { inspectDraft } from './workflowInspector.ts';
import type { ObjectInfo } from './workflowGraph.ts';
import {
  fittedOperation,
  OPERATIONS,
  operationChoices,
  operationNeed,
  outputSummary,
  workflowTraits,
} from './workflowTraits.ts';

const recorded = recordedSchema as ObjectInfo;
const formatBindings = (workflow: Record<string, any>): Binding[] => fixtureFormats(workflow);
const traitsOf = (workflow: Record<string, any>, values: Record<string, unknown> = {}) =>
  workflowTraits(workflow, formatBindings(workflow), values, recorded);
const supported = (workflow: Record<string, any>) =>
  operationChoices(traitsOf(workflow), 'workflow')
    .filter((choice) => !choice.need)
    .map((choice) => choice.id)
    .join(',');

const STILL = 'image,workflow';
const STILL_EDIT = 'image,edit,workflow';
const CLIP = 'video,workflow';
const CLIP_EDIT = 'video,edit,workflow';
const PNG = 'PNG image';
const FRAMES = 'PNG frames';
const FRAMES_AUDIO = 'PNG frames plus separate FLAC audio';
const MP4 = 'one MP4 video with sound';
// The reviewed operations and output of each bundled workflow, by file stem without its prefix.
// A graph that loads an image, a video or a mask offers Edit; Upscale also needs an upscale
// node; a graph with a clip frame count, a video input or a video output makes video.
const EXPECTED: Record<string, [string, string]> = {
  'anima-t2i': [STILL, PNG],
  'boogu-t2i': [STILL, PNG],
  'chroma-t2i': [STILL, PNG],
  'cogvideox-i2v': [CLIP_EDIT, FRAMES],
  'cogvideox-t2v': [CLIP, FRAMES],
  'dual-spark-split': [STILL, PNG],
  'ernie-t2i': [STILL, PNG],
  fleet: ['image,workflow,fleet', PNG],
  'flux1-t2i': [STILL, PNG],
  'flux2-t2i': [STILL, PNG],
  'flux3-klein-base-edit': [STILL_EDIT, PNG],
  'flux3-klein-base-t2i': [STILL, PNG],
  'flux3-klein-edit': [STILL_EDIT, PNG],
  'flux3-klein-t2i': [STILL, PNG],
  'hunyuan-image': [STILL, PNG],
  'hunyuan-video': [CLIP, FRAMES],
  'identity-gate': [STILL, PNG],
  'ideogram4-t2i': [STILL, PNG],
  'kandinsky5-image': [STILL, PNG],
  'kandinsky5-video-lite': [CLIP, FRAMES],
  'kandinsky5-video-pro': [CLIP, FRAMES],
  'krea2-t2i': [STILL, PNG],
  'kroma-t2i-int8': [STILL, PNG],
  'kroma-t2i-upscale': [STILL, PNG],
  'kroma-t2i': [STILL, PNG],
  'kromoss-t2i-turbo': [STILL, PNG],
  'kromoss-t2i': [STILL, PNG],
  'lens-t2i': [STILL, PNG],
  'longcat-t2i': [STILL, PNG],
  'lora-low-rss': [STILL, PNG],
  'ltx-i2v-av': [CLIP_EDIT, FRAMES_AUDIO],
  'ltx-t2v': [CLIP, FRAMES],
  'ltx25-flf2v': [CLIP_EDIT, FRAMES_AUDIO],
  'ltx25-i2v-guide': [CLIP_EDIT, FRAMES_AUDIO],
  'ltx25-i2v': [CLIP_EDIT, FRAMES_AUDIO],
  'ltx25-t2v': [CLIP, FRAMES_AUDIO],
  'mage-flow-edit-turbo': [STILL_EDIT, PNG],
  'mage-flow-edit': [STILL_EDIT, PNG],
  'mage-flow-t2i-turbo': [STILL, PNG],
  'mage-flow-t2i': [STILL, PNG],
  'minimax-h3-guide': [CLIP_EDIT, FRAMES_AUDIO],
  'minimax-h3-mp4': [CLIP, MP4],
  'minimax-h3-t2va': [CLIP, FRAMES_AUDIO],
  'moss-image-base-t2i': [STILL, PNG],
  'moss-image-layout-t2i': [STILL, PNG],
  'moss-image-t2i': [STILL, PNG],
  'omnigen2-t2i': [STILL, PNG],
  'pid-4k': ['image,edit,upscale,workflow', PNG],
  'pixeldit-t2i': [STILL, PNG],
  quickstart: [STILL, PNG],
  'qwen-image-t2i': [STILL, PNG],
  'qwen-image21-edit-mask-rgba': [STILL_EDIT, PNG],
  'qwen-image21-edit-rgba': [STILL_EDIT, PNG],
  'qwen-image21-t2i': [STILL, PNG],
  'radiance-t2i': [STILL, PNG],
  'wan-animate2-base-quality': [CLIP_EDIT, MP4],
  'wan-animate2-bfloat16-warm': [CLIP_EDIT, MP4],
  'wan-animate2-distilled': [CLIP_EDIT, MP4],
  'wan-animate2-memory-saver': [CLIP_EDIT, MP4],
  'wan-animate2': [CLIP_EDIT, MP4],
  'wan-bernini': [CLIP, FRAMES],
  'wan-flowrvs': [CLIP_EDIT, FRAMES],
  'wan-scail': [CLIP_EDIT, FRAMES],
  'wan-scail2': [CLIP_EDIT, FRAMES],
  'wan-t2v': [CLIP, FRAMES],
  'wan21-i2v': [CLIP_EDIT, FRAMES],
  'wan22-i2v': [CLIP_EDIT, FRAMES],
  'wan22-t2v': [CLIP, FRAMES],
  wandancer: [CLIP_EDIT, FRAMES],
  'zimage-dct-t2i': [STILL, PNG],
  'zimage-t2i': [STILL, PNG],
};

test('each bundled workflow offers the operations its graph supports and names its output', () => {
  const rows = bundled.filter((row) => row.name.replace(/^dgx-monarch-/, '') in EXPECTED);
  assert.ok(rows.length >= 71, 'the table covers the bundled set it was reviewed on');
  for (const row of rows) {
    const [operations, output] = EXPECTED[row.name.replace(/^dgx-monarch-/, '')];
    assert.equal(supported(row.workflow), operations, row.name);
    assert.equal(outputSummary(traitsOf(row.workflow)), output, row.name);
  }
  // A template added later keeps the rules that need no review.
  for (const row of bundled) {
    const traits = traitsOf(row.workflow);
    assert.equal(operationNeed('workflow', traits), null, row.name);
    const loads = row.workflow.nodes.some((node: any) =>
      ['LoadImage', 'LoadImageMask', 'LoadVideo'].includes(node.type),
    );
    assert.equal(operationNeed('edit', traits) === null, loads, row.name);
    assert.notEqual(operationNeed('image', traits) === null, traits.motion, row.name);
  }
});

test('the Kroma text to image workflow offers no Edit, and a draft that holds Edit says why', () => {
  const traits = traitsOf(graph('kroma-t2i'));
  assert.deepEqual(
    operationChoices(traits, 'image').map((choice) => choice.id),
    ['image', 'workflow'],
  );
  const held = operationChoices(traits, 'edit');
  assert.deepEqual(
    held.map((choice) => choice.id),
    ['image', 'edit', 'workflow'],
  );
  assert.match(
    held.find((choice) => choice.id === 'edit')!.need!,
    /^Edit needs an image, video or mask input node, such as Load Image\. This workflow has none\.$/,
  );
  // Without a workflow there is nothing to read, so every operation stays listed.
  assert.deepEqual(
    operationChoices(null, 'image').map((choice) => choice.id),
    OPERATIONS.map((operation) => operation.id),
  );
});

test('a loaded workflow keeps a supported operation and otherwise takes its first supported one', () => {
  const kroma = traitsOf(graph('kroma-t2i'));
  assert.equal(fittedOperation('image', kroma), 'image');
  assert.equal(fittedOperation('edit', kroma), 'image');
  assert.equal(fittedOperation('workflow', kroma), 'workflow');
  assert.equal(fittedOperation('no such operation', kroma), 'image');
  assert.equal(fittedOperation('image', traitsOf(graph('minimax-h3-mp4'))), 'video');
  assert.equal(fittedOperation('upscale', traitsOf(graph('minimax-h3-guide'))), 'video');
  assert.equal(fittedOperation('edit', traitsOf(graph('minimax-h3-guide'))), 'edit');
  assert.equal(fittedOperation('fleet', traitsOf(graph('fleet'))), 'fleet');
  assert.equal(fittedOperation('fleet', traitsOf(graph('qwen-image21-edit-rgba'))), 'edit');
  assert.equal(fittedOperation('video', traitsOf(graph('pid-4k'))), 'upscale');
});

test('the output names the save nodes and follows their format choices in the draft', () => {
  const mp4 = graph('minimax-h3-mp4');
  const format = `${mp4.id}:14:format`;
  assert.equal(outputSummary(traitsOf(mp4, { [format]: 'webm' })), 'one WebM video with sound');
  assert.equal(outputSummary(traitsOf(mp4, { [format]: 'mkv' })), 'one MKV video with sound');
  // Save Video's auto container is MP4, or WebM for AV1.
  const codec = `${mp4.id}:14:format.codec`;
  assert.equal(
    outputSummary(traitsOf(mp4, { [format]: 'auto', [codec]: 'av1' })),
    'one WebM video with sound',
  );
  const t2va = graph('minimax-h3-t2va');
  assert.equal(
    outputSummary(traitsOf(t2va, { [`${t2va.id}:14:format`]: 'mp3' })),
    'PNG frames plus separate MP3 audio',
  );
  // Without bindings the formats that a choice sets are unknown, and the kinds remain.
  assert.equal(outputSummary(workflowTraits(mp4, [], {}, recorded)), 'one video with sound');
  assert.equal(
    outputSummary(workflowTraits(t2va, [], {}, recorded)),
    'PNG frames plus separate audio',
  );
  const silent = structuredClone(mp4);
  silent.nodes
    .find((node: any) => node.type === 'CreateVideo')
    .inputs.find((socket: any) => socket.name === 'audio').link = null;
  assert.equal(outputSummary(traitsOf(silent)), 'one MP4 video without sound');
  // Video Combine names its container in its format and takes its own soundtrack.
  const combine = (format: string, audio: boolean, save = true) => {
    const node = {
      id: 2,
      type: 'VHS_VideoCombine',
      inputs: [{ name: 'audio', type: 'AUDIO', link: audio ? 4 : null }],
    };
    const track = { id: 1, type: 'LoadAudio', outputs: [{ type: 'AUDIO', links: [4] }] };
    const links = audio ? [[4, 1, 0, 2, 0, 'AUDIO']] : [];
    const bindings = [
      { id: 'g:2:format', graph_id: 'g', node_id: '2', input_name: 'format', value: format },
      {
        id: 'g:2:save_output',
        graph_id: 'g',
        node_id: '2',
        input_name: 'save_output',
        value: save,
      },
    ];
    return outputSummary(workflowTraits({ id: 'g', nodes: [track, node], links }, bindings, {}));
  };
  assert.equal(combine('video/h264-mp4', true), 'one MP4 video with sound');
  assert.equal(combine('image/gif', false), 'one GIF video without sound');
  assert.equal(
    combine('video/h264-mp4', true, false),
    'previews only; this workflow saves no file',
  );
});

test('an API prompt, a muted node, an upload widget and a used mask output are read from the graph', () => {
  const api = {
    '1': { class_type: 'LoadImage', inputs: { image: 'a.png' } },
    '2': { class_type: 'SaveImage', inputs: { images: ['1', 0], filename_prefix: 'x' } },
    '3': { class_type: 'JoinImageWithAlpha', inputs: { image: ['1', 0], alpha: ['1', 1] } },
  };
  const traits = workflowTraits(api, [], {});
  assert.deepEqual([...traits.inputs], ['image', 'mask']);
  assert.equal(outputSummary(traits), 'PNG image');
  const ui = (type: string, mode = 0) => ({
    id: 'g',
    nodes: [
      { id: 1, type, mode, outputs: [{ name: 'IMAGE', links: [1] }] },
      { id: 2, type: 'PreviewImage', mode: 0 },
    ],
  });
  assert.equal(operationNeed('edit', workflowTraits(ui('LoadImage'), [], {})), null);
  assert.notEqual(operationNeed('edit', workflowTraits(ui('LoadImage', 2), [], {})), null);
  assert.notEqual(operationNeed('edit', workflowTraits(ui('LoadImage', 4), [], {})), null);
  // A loader the class list does not know counts through the upload widget its schema declares.
  const custom: ObjectInfo = {
    PackVideoLoader: { input: { required: { file: ['COMBO', { video_upload: true }] } } },
    PackMaskLoader: {
      input: { required: { image: [[], { image_upload: true }] } },
      output: ['MASK'],
    },
  };
  const video = workflowTraits(ui('PackVideoLoader'), [], {}, custom);
  assert.deepEqual([...video.inputs], ['video']);
  assert.ok(video.motion);
  assert.deepEqual([...workflowTraits(ui('PackMaskLoader'), [], {}, custom).inputs], ['mask']);
  // A preview shows a result and saves no file.
  assert.equal(
    outputSummary(workflowTraits(ui('LoadImage'), [], {})),
    'previews only; this workflow saves no file',
  );
  assert.equal(
    outputSummary(workflowTraits({ nodes: [] }, [], {})),
    'no save node; this workflow saves no file',
  );
});

test('a newly loaded workflow takes a supported operation, and a restored draft keeps its own', async () => {
  const workflow = graph('kroma-t2i');
  const graphId = String(workflow.id);
  const width = { id: `${graphId}:7:width`, graph_id: graphId, node_id: '7', input_name: 'width' };
  const client = (bindings: unknown[]) =>
    ({ request: async () => ({ workflow, bindings }) }) as unknown as StudioClient;
  const read = client([{ ...width, value: 1024, schema: { type: 'INT' } }]);
  const fit = { operation: () => 'edit', info: recorded };
  const operation = async (inspected: Promise<{ patch: Record<string, unknown> }>) =>
    (await inspected).patch.operation;
  assert.equal(await operation(inspectDraft(read, workflow, 'Kroma', undefined, fit)), 'image');
  // Prior values mean a draft that comes back, such as a preset or a reconnect: no fit.
  assert.equal(await operation(inspectDraft(read, workflow, 'Kroma', {}, fit)), undefined);
  assert.equal(await operation(inspectDraft(read, workflow, 'Kroma')), undefined);
  // An inspection that lists no inputs could not read the graph, so the operation stays.
  assert.equal(
    await operation(inspectDraft(client([]), workflow, 'Kroma', undefined, fit)),
    undefined,
  );
});

const socket = (name: string, type: string, link: number | null = null) => ({ name, type, link });
const out = (type: string, links: number[]) => ({ type, links });

test('only subgraphs that a running instance reaches count, so a bypassed video branch changes nothing', () => {
  const still = (mode: number) => ({
    id: 'still',
    nodes: [
      { id: 1, type: 'EmptyLatentImage', outputs: [out('LATENT', [1])] },
      {
        id: 2,
        type: 'KSampler',
        inputs: [socket('model', 'MODEL'), socket('latent_image', 'LATENT', 1)],
        outputs: [out('LATENT', [2])],
      },
      {
        id: 3,
        type: 'VAEDecode',
        inputs: [socket('samples', 'LATENT', 2)],
        outputs: [out('IMAGE', [3])],
      },
      { id: 4, type: 'SaveImage', inputs: [socket('images', 'IMAGE', 3)] },
      { id: 5, type: 'video-def', mode },
    ],
    links: [
      [1, 1, 0, 2, 1, 'LATENT'],
      [2, 2, 0, 3, 0, 'LATENT'],
      [3, 3, 0, 4, 0, 'IMAGE'],
    ],
    definitions: {
      subgraphs: [
        {
          id: 'video-def',
          nodes: [
            { id: 1, type: 'LoadVideo', outputs: [out('VIDEO', [1])] },
            { id: 2, type: 'SaveVideo', inputs: [socket('video', 'VIDEO', 1)] },
          ],
          links: [
            { id: 1, origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0, type: 'VIDEO' },
          ],
        },
        // A definition that no instance uses never runs.
        { id: 'unused-def', nodes: [{ id: 1, type: 'SaveVideo' }], links: [] },
      ],
    },
  });
  for (const mode of [2, 4]) {
    const traits = workflowTraits(still(mode), [], {});
    assert.equal(traits.motion, false, `mode ${mode}`);
    assert.deepEqual([...traits.inputs], []);
    assert.equal(operationNeed('image', traits), null);
    assert.equal(outputSummary(traits), 'PNG image');
  }
  const running = workflowTraits(still(0), [], {});
  assert.ok(running.motion);
  assert.deepEqual([...running.inputs], ['video']);
  assert.notEqual(operationNeed('image', running), null);
  assert.equal(outputSummary(running), 'one video plus PNG frames');
});

test("each saved video's sound follows its own Create Video node", () => {
  // A second branch joins the decoded frames without audio and saves them as well.
  const silentBranch = (workflow: Record<string, any>) => {
    workflow.nodes.push(
      {
        id: 30,
        type: 'CreateVideo',
        inputs: [socket('images', 'IMAGE', 90), socket('audio', 'AUDIO')],
        outputs: [out('VIDEO', [91])],
      },
      {
        id: 31,
        type: 'SaveVideo',
        inputs: [socket('video', 'VIDEO', 91)],
        widgets_values: ['video/silent', 'mp4', 'h264'],
      },
    );
    workflow.links.push([90, 11, 0, 30, 0, 'IMAGE'], [91, 30, 0, 31, 0, 'VIDEO']);
    return workflow;
  };
  const mixed = silentBranch(structuredClone(graph('minimax-h3-mp4')));
  assert.equal(
    outputSummary(traitsOf(mixed)),
    'one MP4 video with sound plus one MP4 video without sound',
  );
  // The branch with audio saves nothing once its Save Video is muted, so it promises no sound.
  const unsaved = silentBranch(structuredClone(graph('minimax-h3-mp4')));
  unsaved.nodes.find((node: any) => node.id === 14).mode = 2;
  assert.equal(outputSummary(traitsOf(unsaved)), 'one MP4 video without sound');
  // A video that no Create Video node makes leaves its sound unknown.
  const passed = {
    id: 'pass',
    nodes: [
      { id: 1, type: 'LoadVideo', outputs: [out('VIDEO', [1])] },
      { id: 2, type: 'SaveVideo', inputs: [socket('video', 'VIDEO', 1)] },
    ],
    links: [[1, 1, 0, 2, 0, 'VIDEO']],
  };
  assert.equal(outputSummary(workflowTraits(passed, [], {})), 'one video');
});

test('an operation chosen while a workflow loads stays, and an untouched one is fitted', async () => {
  const workflow = graph('kroma-t2i');
  const graphId = String(workflow.id);
  const bindings = [
    { id: `${graphId}:7:width`, graph_id: graphId, node_id: '7', input_name: 'width', value: 1024 },
  ];
  let answer: (value: unknown) => void = () => {};
  const client = {
    request: () => new Promise((resolve) => (answer = resolve)),
  } as unknown as StudioClient;
  let chosen = 'edit';
  const fit = { operation: () => chosen, info: recorded };
  const changed = inspectDraft(client, workflow, 'Kroma', undefined, fit);
  chosen = 'workflow';
  answer({ workflow, bindings });
  assert.equal((await changed).patch.operation, undefined, 'the newer choice stays');
  chosen = 'edit';
  const untouched = inspectDraft(client, workflow, 'Kroma', undefined, fit);
  answer({ workflow, bindings });
  assert.equal((await untouched).patch.operation, 'image', 'fitted when the choice did not change');
});
