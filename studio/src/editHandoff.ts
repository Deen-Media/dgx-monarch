import type { StudioClient } from './api';
import type { AssetReceipt } from './assetEditorState.svelte';
import type { Artifact, Inspection, Workflow, WorkflowEntry } from './contracts';
import type { Binding, Draft } from './model';
import { bindingError, bindingType, fieldLabel } from './model.ts';
import { byTitle, workflowTitle } from './workflowChoice.ts';
import {
  bindingIndex,
  declaredInputs,
  graphNodes,
  type GraphNode,
  type ObjectInfo,
} from './workflowGraph.ts';
import { operationNeed, workflowTraits } from './workflowTraits.ts';

const isObject = (value: unknown): value is Record<string, any> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

/**
 * The input of a node class that takes an image stored in the driver's input store: an image
 * upload widget that reads that store, as the companion's upload rule has it. Load Image counts
 * when the driver schema does not list the class.
 */
function uploadInput(type: string, info: ObjectInfo | undefined): string | null {
  const declared = Object.entries(declaredInputs(info, type));
  if (!declared.length) return type === 'LoadImage' ? 'image' : null;
  for (const [name, spec] of declared) {
    const options = Array.isArray(spec) && isObject(spec[1]) ? spec[1] : {};
    if (options.image_upload === true && (options.image_folder ?? 'input') === 'input') return name;
  }
  return null;
}

/** The running nodes that load an image from the input store, each with its upload input. */
export function imageLoaders(workflow: Workflow | null | undefined, info?: ObjectInfo) {
  return graphNodes(workflow, info).flatMap((node) => {
    const input = uploadInput(node.type, info);
    // A loader whose outputs are all masks, such as Load Image (as Mask), takes no image.
    if (!input || (node.outputTypes.length && !node.outputTypes.includes('IMAGE'))) return [];
    return [{ node, input }];
  });
}

/** The node classes a workflow runs that the driver does not list; the companion refuses them. */
export function missingClasses(workflow: Workflow | null | undefined, info?: ObjectInfo) {
  const types = graphNodes(workflow, info).map((node: GraphNode) => node.type);
  return [...new Set(types)].filter((type) => type !== 'PrimitiveNode' && !info?.[type]);
}

/**
 * A workflow whose purpose is editing an image, so Studio loads it for Edit: it runs on this
 * driver, loads an image from the input store, makes no video and has no upscale or Fleet node.
 */
export function editingWorkflow(workflow: Workflow, info: ObjectInfo): boolean {
  const traits = workflowTraits(workflow, [], {}, info);
  return (
    !traits.motion &&
    !traits.upscale &&
    !traits.fleet &&
    imageLoaders(workflow, info).length > 0 &&
    !missingClasses(workflow, info).length
  );
}

const byOrder = (a: Binding, b: Binding) => Number(a.order ?? 0) - Number(b.order ?? 0);

/**
 * The binding Send to edit fills: the image input that the workflow's App Mode layout lists
 * first, or the first image input in the graph when the layout lists none. A wired input, or one
 * Studio cannot set, does not count.
 */
export function editTarget(
  workflow: Workflow | null | undefined,
  bindings: Binding[],
  info?: ObjectInfo,
): Binding | null {
  const own = bindingIndex(bindings);
  const found = imageLoaders(workflow, info).flatMap(({ node, input }) => {
    const binding = own(node).get(input);
    if (!binding || binding.resolved === false) return [];
    return ['STRING', 'COMBO'].includes(bindingType(binding)) ? [binding] : [];
  });
  const shown = found.filter((binding) => binding.prominent || binding.display?.prominent);
  return [...shown].sort(byOrder)[0] ?? found[0] ?? null;
}

export const VIDEO_OUTPUT = 'This output is a video, and Send to edit takes an image.';
export const MASK_OUTPUT = 'This is an uploaded mask, and Send to edit takes an image.';
export const UNREAD_SCHEMA =
  "Studio has not read the driver's node list, so it cannot tell which workflows this driver runs. Refresh list on the LoRAs tab reads it again.";
export const NO_EDITOR =
  'The loaded workflow has no image input that Studio can set, and no workflow in the Workflow list edits an image on this driver.';
export const NEXT_STEP =
  'The draft is unchanged, and the output stays selected. Editing workflows come from the Workflow list and Import JSON on Create, and from workflows saved in Library. Load one that takes an image input, then press Send to edit again.';
export const SEVERAL =
  'Several workflows can edit this output. Choose one, and Studio loads it, puts the output in its image input and sets the Operation to Edit. Until you choose, the draft stays as it is.';

/** What Send to edit does with an output, decided before anything changes the draft. */
export type EditPlan =
  | { kind: 'keep'; target: Binding }
  | { kind: 'load'; choice: WorkflowEntry }
  | { kind: 'choose'; choices: WorkflowEntry[] }
  | { kind: 'none'; reason: string };

/**
 * Keep the loaded workflow when it supports Edit and has an image input to fill. Otherwise offer
 * the packaged editing workflows that run on this driver: load the one, let the user choose among
 * several, or say why there is none.
 */
export function editPlan(
  artifact: Pick<Artifact, 'kind'>,
  current: { workflow: Workflow | null; bindings: Binding[]; values: Draft['values'] },
  entries: readonly WorkflowEntry[],
  info?: ObjectInfo,
): EditPlan {
  // Uploads carry their own kinds, input and mask, beside the image and video of job outputs.
  const kind: string = artifact.kind;
  if (kind === 'video') return { kind: 'none', reason: VIDEO_OUTPUT };
  if (kind === 'mask') return { kind: 'none', reason: MASK_OUTPUT };
  const { workflow, bindings, values } = current;
  if (workflow && bindings.length) {
    const traits = workflowTraits(workflow, bindings, values, info);
    const target = editTarget(workflow, bindings, info);
    if (!operationNeed('edit', traits) && target) return { kind: 'keep', target };
  }
  if (!info || !Object.keys(info).length) return { kind: 'none', reason: UNREAD_SCHEMA };
  const choices = byTitle(entries).filter((entry) => editingWorkflow(entry.workflow, info));
  if (choices.length === 1) return { kind: 'load', choice: choices[0] };
  return choices.length ? { kind: 'choose', choices } : { kind: 'none', reason: NO_EDITOR };
}

/**
 * The draft with the output in the image input and the Operation set to Edit. A loaded workflow
 * starts from its own values and lineage; a kept one keeps its values and adds to its lineage.
 */
export function editDraft(
  draft: Draft,
  target: Binding,
  value: string,
  lineage: string[],
  load?: { workflow: Workflow; name: string } | null,
): Draft {
  const kept = load ? { values: {}, parent_artifact_ids: [] } : draft;
  return {
    ...draft,
    ...(load ? { workflow: load.workflow, workflowName: load.name } : {}),
    values: { ...kept.values, [target.id]: value },
    parent_artifact_ids: [...new Set([...kept.parent_artifact_ids, ...lineage])],
    operation: 'edit',
  };
}

/**
 * Why Generate waits for an edit's image, or null: under Edit, the image input Send to edit fills
 * must hold an image the driver offers.
 */
export function editImageNeed(draft: Draft, bindings: Binding[], info?: ObjectInfo) {
  if (draft.operation !== 'edit' || !draft.workflow || !bindings.length) return null;
  const target = editTarget(draft.workflow, bindings, info);
  if (!target) return null;
  const value = draft.values[target.id] ?? target.value;
  if (value !== undefined && value !== null && value !== '' && !bindingError(target, value))
    return null;
  return `Edit needs an image in the ${fieldLabel(target)} input on node ${target.node_id}. Use Send to edit on an output, or upload an image on the Assets tab.`;
}

/** What App lends Send to edit: the busy state, the draft import, navigation and the notice. */
export interface EditActions {
  withBusy: (action: () => Promise<void>) => Promise<void>;
  importDraft: (value: Draft, templateId?: string) => Promise<void>;
  navigate: (view: string) => void;
  setNotice: (message: string) => void;
}

/**
 * The state Send to edit reads, current at each read. `draft` returns the live draft, so a
 * workflow that changed while the output was copied is a different object.
 */
export interface EditContext {
  client: StudioClient;
  info: () => ObjectInfo | undefined;
  workflows: () => readonly WorkflowEntry[];
  workflowId: () => string;
  draft: () => Draft;
  bindings: () => Binding[];
}

/**
 * Bind an output as the image input of the kept workflow, or of `choice` once loaded. The
 * companion copies the output into the driver's input store first, so a refusal leaves the draft
 * unchanged; then one draft import makes the change, which one Undo reverts. The import inspects
 * the workflow with the bound value, so the image input lists the copy among its choices.
 */
export async function bindOutput(
  artifact: Artifact,
  choice: WorkflowEntry | null,
  context: EditContext,
  actions: EditActions,
) {
  const { client } = context;
  const start = context.draft().workflow;
  let workflow = start;
  let bindings = context.bindings();
  if (choice) {
    const inspected = await client.request<Inspection>('/workflows/inspect', {
      workflow: choice.workflow,
    });
    [workflow, bindings] = [inspected.workflow ?? choice.workflow, inspected.bindings ?? []];
  }
  const target = editTarget(workflow, bindings, context.info());
  const entry = choice ?? context.workflows().find((item) => item.id === context.workflowId());
  const label = entry
    ? workflowTitle(entry)
    : context.draft().workflowName || 'the loaded workflow';
  if (!target)
    throw new Error(`${label} has no image input that Studio can set, so the draft is unchanged.`);
  const receipt =
    (artifact.kind as string) === 'input' && typeof artifact.binding_value === 'string'
      ? (artifact as unknown as AssetReceipt)
      : await client.request<AssetReceipt>('/assets', { kind: 'input', artifact_id: artifact.id });
  if (context.draft().workflow !== start)
    throw new Error(
      `The output was copied as ${receipt.binding_value}, but the draft's workflow changed meanwhile, so it was not bound. Press Send to edit again.`,
    );
  const load = choice && workflow ? { workflow, name: choice.name } : null;
  const lineage = [artifact.id, receipt.id];
  const next = editDraft(context.draft(), target, receipt.binding_value, lineage, load);
  await actions.importDraft(next, choice ? choice.id : context.workflowId());
  // An inspection that failed leaves no bindings and shows its own error.
  if (!context.bindings().length) return;
  actions.navigate('workspace');
  actions.setNotice(
    `Send to edit put the output in the ${fieldLabel(target)} input on node ${target.node_id} of ${label} and set the Operation to Edit. Describe the change in the prompt, then Generate. Undo restores the previous draft.`,
  );
}

/** A plan that needs the user: the workflows to choose among, or why none can take the output. */
export type EditOffer = Extract<EditPlan, { kind: 'choose' | 'none' }>;

/**
 * Send to edit. Keeping the loaded workflow, or loading the one editing workflow, binds the
 * output at once. Several workflows, or none, go to `offer`, which shows the choice or the reason;
 * nothing changes the draft until the user chooses.
 */
export function sendToEdit(
  artifact: Artifact,
  context: EditContext,
  actions: EditActions,
  offer: (plan: EditOffer) => void,
): Promise<void> {
  const draft = context.draft();
  const current = { workflow: draft.workflow, bindings: context.bindings(), values: draft.values };
  const plan = editPlan(artifact, current, context.workflows(), context.info());
  if (plan.kind === 'choose' || plan.kind === 'none') return Promise.resolve(offer(plan));
  const choice = plan.kind === 'load' ? plan.choice : null;
  return actions.withBusy(() => bindOutput(artifact, choice, context, actions));
}
