import type { GraphNode } from './workflowGraph.ts';

/**
 * The inputs whose text each stock ComfyUI class passes on, read from its source at 0.45.0. Every
 * stock class with a text input and a text or conditioning output has a row:
 * tests/fixtures/studio/stock_text_classes.json lists them, and a test fails when one lacks a row.
 * A row names the inputs that hold the content the node passes on or renders: a text node's text,
 * an encoder's prompts and the conditioning it adds to, a media node's prompt. It leaves out what
 * shapes or locates the text (a pattern, a find or replacement string, a separator, a template, a
 * key, a file name), ids, names, numbers and options, and everything a text generator is asked,
 * since a generator writes text of its own. An entry that ends in `.` names a group of inputs that
 * grows, such as Format Text's `values.a`. The two switches and Branch pass on their branches; the
 * Soft Switch is the one `nodes/graph_live.py` lists, which ComfyUI 0.45.0 does not register.
 */
export const TEXT_THROUGH: Record<string, readonly string[]> = {
  AddTextPrefix: ['texts', 'prefix'],
  AddTextSuffix: ['texts', 'suffix'],
  BranchNode: ['autogrow.'],
  BriaAddObject: [],
  BriaEraseByText: [],
  BriaExpandImage: ['prompt', 'negative_prompt'],
  BriaImageEditNode: ['prompt', 'negative_prompt', 'structured_prompt'],
  BriaImageGenerate: ['prompt', 'structured_prompt'],
  BriaReplaceImageBackground: ['background.prompt'],
  BriaReplaceObject: [],
  ByteDance2FirstLastFrameNode: ['model.prompt'],
  ByteDance2ReferenceNodeV2: ['model.prompt'],
  ByteDance2TextToVideoNode: ['model.prompt'],
  ByteDanceCreateImageAsset: [],
  ByteDanceCreateVideoAsset: [],
  ByteDanceSeedNode: [],
  CLIPTextEncode: ['text'],
  CLIPTextEncodeControlnet: ['conditioning', 'text'],
  CLIPTextEncodeFlux: ['clip_l', 't5xxl'],
  CLIPTextEncodeHiDream: ['clip_l', 'clip_g', 't5xxl', 'llama'],
  CLIPTextEncodeHunyuanDiT: ['bert', 'mt5xl'],
  CLIPTextEncodeKandinsky5: ['clip_l', 'qwen25_7b'],
  CLIPTextEncodeLumina2: ['user_prompt'],
  CLIPTextEncodePixArtAlpha: ['text'],
  CLIPTextEncodeSD3: ['clip_l', 'clip_g', 't5xxl'],
  CLIPTextEncodeSDXL: ['text_g', 'text_l'],
  CLIPTextEncodeSDXLRefiner: ['text'],
  CameraAngle: [],
  CaseConverter: ['string'],
  ClaudeNode: [],
  ComfySoftSwitchNode: ['on_true', 'on_false'],
  ComfySwitchNode: ['on_true', 'on_false'],
  ElevenLabsSpeechToText: [],
  FishAudioSpeechToText: [],
  Flux3LayoutPrompt: ['prompt'],
  GLIGENTextBoxApply: ['conditioning_to', 'text'],
  GeminiImage2Node: ['prompt'],
  GeminiImageNode: ['prompt'],
  GeminiNanoBanana2: ['prompt'],
  GeminiNanoBanana2V2: ['prompt'],
  GeminiNode: [],
  GeminiNodeV2: [],
  GeminiNodeV3: [],
  GeminiVideoOmni: ['model.prompt'],
  GeminiVideoOmniV2: ['model.prompt'],
  HeyGenCreateAvatarNode: ['source.prompt'],
  IdeogramPImage: ['prompt'],
  IdeogramTextToImageApi: ['model.prompt'],
  JsonExtractString: ['json_string'],
  KlingImage2VideoNode: ['prompt', 'negative_prompt'],
  KlingLipSyncTextToVideoNode: ['text'],
  KlingStartEndFrameNode: ['prompt', 'negative_prompt'],
  KlingTextToVideoNode: ['prompt', 'negative_prompt'],
  LTXVAddGeneratedKeyframes: [],
  LTXVGeneratedKeyframesToGuides: [],
  LumaRay32ExtendVideoNode: ['prompt'],
  LumaRay32ImageToVideoNode: ['prompt'],
  LumaRay32KeyframesToVideoNode: ['prompt'],
  LumaRay32TextToVideoNode: ['prompt'],
  LumaRay32VideoEditNode: ['prompt'],
  LumaRay32VideoReframeNode: ['prompt'],
  MOSSImageLayoutPrompt: ['prompt'],
  MakeTrainingDataset: ['texts'],
  MergeTextLists: ['texts'],
  MeshyImageToModelNode: ['should_texture.texture_prompt'],
  MeshyMultiImageToModelNode: ['should_texture.texture_prompt'],
  MeshyRefineNode: ['texture_prompt'],
  MeshyTextToModelNode: ['prompt'],
  MeshyTextureNode: ['text_style_prompt'],
  MiniMaxH3ImageConditioning: ['prompt'],
  MiniMaxH3ImageToVideo: ['prompt'],
  MiniMaxH3ReferenceToVideo: ['prompt'],
  MiniMaxMusic3TextEncode: ['caption', 'lyrics'],
  MinimaxH3ImageNode: ['prompt'],
  MinimaxHailuo03ContextIRNode: [],
  OpenAIChatNode: [],
  OpenRouterLLMNode: [],
  PH3ImageConditioning: ['prompt'],
  PatinaMaterialExtractNode: ['prompt'],
  PatinaMaterialNode: ['prompt'],
  PhotoMakerEncode: ['text'],
  PrimitiveString: ['value'],
  PrimitiveStringMultiline: ['value'],
  RecraftV4TextToImageNode: ['prompt', 'negative_prompt'],
  RecraftV4TextToVectorNode: ['prompt', 'negative_prompt'],
  RegexExtract: ['string'],
  RegexReplace: ['string'],
  ReplaceText: ['texts'],
  SaveConditioning: ['conditioning'],
  SaveText: ['text'],
  ShuffleImageTextDataset: ['texts'],
  ShuffleVideoTextDataset: ['texts'],
  StringConcatenate: ['string_a', 'string_b'],
  StringFormat: ['values.'],
  StringReplace: ['string'],
  StringSubstring: ['string'],
  StringTrim: ['string'],
  StripWhitespace: ['texts'],
  TencentHyNode: [],
  TencentTextToModelNode: ['prompt'],
  TextEncodeAceStepAudio: ['tags', 'lyrics'],
  'TextEncodeAceStepAudio1.5': ['tags', 'lyrics'],
  TextEncodeBooguEdit: ['prompt', 'negative_prompt'],
  TextEncodeFibo2: ['prompt', 'negative_prompt'],
  TextEncodeHunyuanVideo_ImageToVideo: ['prompt'],
  TextEncodeJoyImageEdit: ['prompt'],
  TextEncodeMOSSImageEdit: ['prompt'],
  TextEncodeMageFlowEdit: ['prompt', 'negative_prompt'],
  TextEncodeMingImageEdit: ['prompt'],
  TextEncodeQwenImage21: ['prompt', 'negative_prompt'],
  TextEncodeQwenImageEdit: ['prompt'],
  TextEncodeQwenImageEditPlus: ['prompt'],
  TextEncodeZImageOmni: ['prompt'],
  TextGenerate: [],
  TextGenerateLTX2Prompt: [],
  TextGenerateV2: [],
  TextToLowercase: ['texts'],
  TextToUppercase: ['texts'],
  TripoMeshCompleteNode: [],
  TripoP1TextToModelNode: ['prompt', 'negative_prompt'],
  TripoRetopologyNode: [],
  TripoSmartSegmentNode: [],
  TripoTextToModelNode: ['prompt', 'negative_prompt'],
  TripoTextureNode: ['texture_prompt'],
  TruncateText: ['texts'],
  WanTrackToVideo: [],
  WeVisOneDetect: [],
  YuE2GenerateABC: [],
  YuE2GenerateMusic: ['style', 'lyrics', 'abc'],
};

/**
 * Words in an input's name that say it shapes or locates the text instead of holding it. A name's
 * words split at `_`, dots, digits and capitals, and match whatever their case, so `System_Prompt`
 * and `Find` are denied while `profile` and `keywords` are not.
 */
const DENIED_WORDS = new Set([
  'system',
  'pattern',
  'regex',
  'find',
  'replace',
  'delimiter',
  'separator',
  'instruction',
  'instructions',
  'path',
  'file',
  'filename',
  'url',
  'folder',
  'directory',
  'dir',
  'key',
  'format',
  'template',
]);
const WORDS = /[A-Z]+(?![a-z])|[A-Z]?[a-z]+/g;
const denied = (input: string) =>
  (input.match(WORDS) ?? []).some((word) => DENIED_WORDS.has(word.toLowerCase()));

/**
 * Whether a node passes on the text of one input into its own output. `typed` is the widget's
 * box when the text is typed on this node, and absent when it arrives by link. A row decides for
 * its class. A class with no row, such as a node pack's, never passes on an input whose name a
 * denied word marks. Past that it passes on linked text, and a typed widget only when its box is
 * multi-line, since a one-line widget holds an option, a name or a key more often than a prompt.
 * A node that makes conditioning takes in every input the words leave, typed or linked, as an
 * encoder takes in all its text.
 */
export function carries(
  node: Pick<GraphNode, 'type' | 'outputTypes'>,
  input: string,
  typed?: { multiline?: boolean },
): boolean {
  if (Object.hasOwn(TEXT_THROUGH, node.type))
    return TEXT_THROUGH[node.type].some(
      (name) => name === input || (name.endsWith('.') && input.startsWith(name)),
    );
  if (denied(input)) return false;
  return !typed || typed.multiline === true || node.outputTypes.includes('CONDITIONING');
}
