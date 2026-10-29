import type { Binding } from './model.ts';
import {
  bindingIndex,
  currentValue,
  declaredInputs,
  workflowGraph,
  type GraphNode,
  type ObjectInfo,
  type WorkflowGraph,
} from './workflowGraph.ts';

export type OperationId = 'image' | 'video' | 'edit' | 'upscale' | 'workflow' | 'fleet';
/** The Operation select's choices, in the order it lists them. */
export const OPERATIONS: ReadonlyArray<{ id: OperationId; label: string }> = [
  { id: 'image', label: 'Image' },
  { id: 'video', label: 'Video' },
  { id: 'edit', label: 'Edit' },
  { id: 'upscale', label: 'Upscale' },
  { id: 'workflow', label: 'Workflow' },
  { id: 'fleet', label: 'Fleet' },
];
/** A loaded workflow keeps a supported operation, or takes the first of these that it supports. */
const PRIMARY: OperationId[] = ['fleet', 'upscale', 'video', 'edit', 'image', 'workflow'];

type Media = 'image' | 'mask' | 'video' | 'audio';
/** Input nodes by class, for a driver schema that does not list the class. */
const LOADERS: Record<string, Media> = {
  LoadImage: 'image',
  LoadImageOutput: 'image',
  LoadImageMask: 'mask',
  LoadVideo: 'video',
  LoadAudio: 'audio',
  VHS_LoadVideo: 'video',
  VHS_LoadVideoPath: 'video',
};
/**
 * Nodes that enlarge an image or a latent by a factor or a model. A resize to a fixed size, such
 * as Upscale Image or Scale To Total Pixels, also shrinks, so it does not count.
 */
const UPSCALERS = new Set([
  'ImageUpscaleWithModel',
  'UpscaleModelLoader',
  'ImageScaleBy',
  'LatentUpscaleBy',
  'PiDConditioning',
  'UltimateSDUpscale',
]);
const FLEET = 'DGXMonarchFleetKSampler';

interface Saver {
  kind: 'image' | 'video' | 'audio';
  format?: string;
  sound?: boolean;
  saved?: false;
}
/** Output nodes by class. A preview shows a result without saving a file. */
const SAVERS: Record<string, Saver> = {
  SaveImage: { kind: 'image', format: 'PNG' },
  PreviewImage: { kind: 'image', format: 'PNG', saved: false },
  SaveAnimatedWEBP: { kind: 'video', format: 'animated WebP', sound: false },
  SaveAnimatedPNG: { kind: 'video', format: 'animated PNG', sound: false },
  SaveWEBM: { kind: 'video', format: 'WebM', sound: false },
  SaveVideo: { kind: 'video' },
  SaveAudio: { kind: 'audio', format: 'FLAC' },
  SaveAudioMP3: { kind: 'audio', format: 'MP3' },
  SaveAudioOpus: { kind: 'audio', format: 'Opus' },
  SaveAudioAdvanced: { kind: 'audio' },
  PreviewAudio: { kind: 'audio', saved: false },
  VHS_VideoCombine: { kind: 'video' },
};
const CONTAINERS: Record<string, string> = {
  mp4: 'MP4',
  mkv: 'MKV',
  webm: 'WebM',
  mov: 'MOV',
  gif: 'GIF',
  webp: 'WebP',
};
const AUDIO_FORMATS: Record<string, string> = { flac: 'FLAC', mp3: 'MP3', opus: 'Opus' };

/** One node whose result the job keeps or shows. */
export interface WorkflowOutput {
  node: GraphNode;
  kind: 'image' | 'video' | 'audio';
  /** False for a preview, which writes no output file. */
  saved: boolean;
  format?: string;
  /** Whether a video carries a soundtrack; null when the graph does not say. */
  sound?: boolean | null;
}
/** What a workflow takes in and gives out, read from its graph, bindings and draft values. */
export interface WorkflowTraits {
  inputs: Set<Media>;
  upscale: boolean;
  fleet: boolean;
  /** The graph makes video: it holds a clip frame count, a video input or a video output. */
  motion: boolean;
  outputs: WorkflowOutput[];
}

const isObject = (value: unknown): value is Record<string, any> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

/** The media an input node loads: its class, or the upload widget its schema declares. */
function loads(node: GraphNode, info: ObjectInfo | undefined): Media | undefined {
  if (LOADERS[node.type]) return LOADERS[node.type];
  for (const spec of Object.values(declaredInputs(info, node.type))) {
    const options = Array.isArray(spec) && isObject(spec[1]) ? spec[1] : {};
    if (options.video_upload) return 'video';
    if (options.audio_upload) return 'audio';
    if (options.image_upload) {
      const outputs = info?.[node.type]?.output ?? [];
      return outputs.length && outputs.every((kind) => kind === 'MASK') ? 'mask' : 'image';
    }
  }
  return undefined;
}

/** Whether an image loader's mask output feeds the graph, as an alpha-masked edit reads it. */
function masks(node: GraphNode, info: ObjectInfo | undefined): boolean {
  const declared = info?.[node.type]?.output;
  const slot = Array.isArray(declared) ? declared.indexOf('MASK') : 1;
  return slot >= 0 && node.used.has(slot);
}

/** A node that sets a clip's frame count beside its width and height, such as a video latent. */
export function setsFrames(
  node: GraphNode,
  info: ObjectInfo | undefined,
  own: Map<string, Binding>,
) {
  const names = new Set([...Object.keys(declaredInputs(info, node.type)), ...own.keys()]);
  return names.has('length') && names.has('width') && names.has('height');
}

function saverOf(node: GraphNode, info: ObjectInfo | undefined): Saver | undefined {
  if (SAVERS[node.type]) return SAVERS[node.type];
  if (!/^(Save|Preview)/.test(node.type)) return undefined;
  const kinds = Object.values(declaredInputs(info, node.type)).map((spec) =>
    Array.isArray(spec) ? spec[0] : undefined,
  );
  const kind = kinds.includes('VIDEO')
    ? 'video'
    : kinds.includes('AUDIO')
      ? 'audio'
      : kinds.includes('IMAGE')
        ? 'image'
        : undefined;
  return kind && { kind, ...(node.type.startsWith('Preview') ? { saved: false as const } : {}) };
}

/** Whether a running node feeds this input. */
const fed = (node: GraphNode, input: string) => {
  const source = node.sources.get(input);
  return !!source && !source.primitive;
};

/**
 * Whether one saved video has sound: Video Combine takes its own soundtrack, and Save Video has
 * one when the Create Video node that feeds it takes audio. Any other source leaves it unknown.
 */
function soundOf(node: GraphNode, graph: WorkflowGraph): boolean | null {
  if (node.type === 'VHS_VideoCombine') return fed(node, 'audio');
  const source = node.sources.get('video');
  const maker = source && !source.primitive ? graph.byKey.get(source.key) : undefined;
  return maker?.type === 'CreateVideo' ? fed(maker, 'audio') : null;
}

/** The container or audio format an output node writes, from the draft's choice where it has one. */
function formatOf(
  node: GraphNode,
  saver: Saver,
  own: Map<string, Binding>,
  values: Record<string, unknown>,
) {
  if (saver.format) return saver.format;
  const format = String(currentValue(own.get('format'), values) ?? '');
  if (saver.kind === 'audio') return AUDIO_FORMATS[format];
  // Video Combine names a format such as video/h264-mp4 or image/gif.
  if (node.type === 'VHS_VideoCombine') return CONTAINERS[format.split(/[/-]/).pop() ?? ''];
  // Save Video's auto container is MP4, or WebM when the codec is AV1.
  if (format === 'auto')
    return currentValue(own.get('format.codec'), values) === 'av1' ? 'WebM' : 'MP4';
  return CONTAINERS[format];
}

export function workflowTraits(
  workflow: Record<string, any> | null | undefined,
  bindings: Binding[],
  values: Record<string, unknown>,
  info?: ObjectInfo,
): WorkflowTraits {
  const graph = workflowGraph(workflow, info);
  const { nodes } = graph;
  const inputs = new Set<Media>();
  const outputs: WorkflowOutput[] = [];
  let motion = false;
  const bindingsOf = bindingIndex(bindings);
  for (const node of nodes) {
    const own = bindingsOf(node);
    const media = loads(node, info);
    if (media) inputs.add(media);
    if (media === 'image' && masks(node, info)) inputs.add('mask');
    if (media === 'video' || node.type === 'CreateVideo' || setsFrames(node, info, own))
      motion = true;
    const saver = saverOf(node, info);
    if (!saver) continue;
    if (saver.kind === 'video') motion = true;
    // Video Combine's save_output off keeps its file temporary.
    const combine = node.type === 'VHS_VideoCombine';
    outputs.push({
      node,
      kind: saver.kind,
      saved:
        saver.saved !== false &&
        !(combine && currentValue(own.get('save_output'), values) === false),
      format: formatOf(node, saver, own, values),
      ...(saver.kind === 'video' ? { sound: saver.sound ?? soundOf(node, graph) } : {}),
    });
  }
  return {
    inputs,
    upscale: nodes.some((node) => UPSCALERS.has(node.type)),
    fleet: nodes.some((node) => node.type === FLEET),
    motion,
    outputs,
  };
}

/** Why the workflow cannot run an operation, or null when it can. */
export function operationNeed(id: string, traits: WorkflowTraits): string | null {
  const { inputs } = traits;
  const media = inputs.has('image') || inputs.has('video');
  const shows = (kind: string) => traits.outputs.some((output) => output.kind === kind);
  if (id === 'image')
    return shows('image') && !traits.motion
      ? null
      : 'Image needs a graph that saves a still image; this workflow makes video or saves no image.';
  if (id === 'video')
    return traits.motion && (shows('image') || shows('video'))
      ? null
      : 'Video needs a graph that makes video: a frame count, a video input or a video save node. This workflow has none.';
  if (id === 'edit')
    return media || inputs.has('mask')
      ? null
      : 'Edit needs an image, video or mask input node, such as Load Image. This workflow has none.';
  if (id === 'upscale')
    return media && traits.upscale
      ? null
      : 'Upscale needs an image or video input node and an upscale node, such as Upscale Image By. This workflow lacks one of them.';
  if (id === 'fleet')
    return traits.fleet
      ? null
      : 'Fleet needs the Fleet KSampler (DGX Monarch) node, which the Fleet workflow carries. This workflow has none.';
  return null;
}

/**
 * The Operation select's choices: those the workflow supports, and the draft's own choice even when
 * the workflow does not support it, so the select still shows it. Every choice without a workflow.
 */
export function operationChoices(traits: WorkflowTraits | null, current: string) {
  return OPERATIONS.map((operation) => ({
    ...operation,
    need: traits ? operationNeed(operation.id, traits) : null,
  })).filter((operation) => !operation.need || operation.id === current);
}

/** The draft's operation when the workflow supports it, otherwise the first one it supports. */
export function fittedOperation(current: string, traits: WorkflowTraits): OperationId {
  const known = OPERATIONS.find((operation) => operation.id === current);
  if (known && !operationNeed(known.id, traits)) return known.id;
  return PRIMARY.find((id) => !operationNeed(id, traits)) ?? 'workflow';
}

/** Words for some outputs: "one MP4 video", "2 WebM videos" or, with no count, "PNG frames". */
function phrase(outputs: WorkflowOutput[], noun: string, count: boolean) {
  const formats = [...new Set(outputs.map((output) => output.format).filter(Boolean))].join(
    ' and ',
  );
  const plural =
    noun === 'audio' ? noun : outputs.length > 1 || noun === 'frame' ? `${noun}s` : noun;
  const amount = !count ? '' : outputs.length === 1 ? 'one' : String(outputs.length);
  return [amount, formats, plural].filter(Boolean).join(' ');
}

/** What a run saves, in words: for example "one MP4 video with sound" or "PNG frames". */
export function outputSummary(traits: WorkflowTraits): string {
  const saved = traits.outputs.filter((output) => output.saved);
  if (!saved.length)
    return traits.outputs.length
      ? 'previews only; this workflow saves no file'
      : 'no save node; this workflow saves no file';
  const of = (kind: string) => saved.filter((output) => output.kind === kind);
  const parts: string[] = [];
  // Each saved video says whether it has sound, and says nothing where the graph does not tell.
  for (const [sound, suffix] of [
    [true, ' with sound'],
    [false, ' without sound'],
    [null, ''],
  ] as const) {
    const videos = of('video').filter((video) => (video.sound ?? null) === sound);
    if (videos.length) parts.push(`${phrase(videos, 'video', true)}${suffix}`);
  }
  const images = of('image');
  if (images.length) parts.push(phrase(images, traits.motion ? 'frame' : 'image', false));
  const audio = of('audio');
  if (audio.length)
    parts.push(`${parts.length ? 'separate ' : ''}${phrase(audio, 'audio', false)}`);
  return parts.join(' plus ');
}
