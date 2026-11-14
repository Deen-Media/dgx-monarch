import test from 'node:test';
import assert from 'node:assert/strict';
import { bundled, recorded } from './workflowFixtures.js';
import {
  editDraft,
  editImageNeed,
  editingWorkflow,
  editPlan,
  editTarget,
  MASK_OUTPUT,
  NO_EDITOR,
  sendToEdit,
  UNREAD_SCHEMA,
  VIDEO_OUTPUT,
  type EditActions,
  type EditContext,
  type EditOffer,
} from './editHandoff.ts';
import { reuseNotice } from './artifactDraft.ts';
import { emptyDraft, type Binding, type Draft } from './model.ts';
import type { Artifact, WorkflowEntry } from './contracts.ts';

const entries = bundled as unknown as WorkflowEntry[];
const entry = (stem: string) => entries.find((row) => row.name === `dgx-monarch-${stem}`)!;
const IMAGE_CHOICES = recorded.LoadImage.input.required.image[0] as string[];
const output = { id: 'output', job_id: 'job', kind: 'image' } as Artifact;
const RECEIPT = { id: 'copy', binding_value: 'studio/copy.png', width: 128, height: 96 };

/**
 * Bindings as the companion's inspection lists a packaged workflow's App Mode inputs. As there, a
 * stored upload that the draft's values name joins the choices of an image upload input.
 */
const stored = (value: unknown) => typeof value === 'string' && value.startsWith('studio/');

function inspected(workflow: Record<string, any>, values: Record<string, unknown> = {}): Binding[] {
  const graphId = String(workflow.id);
  const nodes = new Map(workflow.nodes.map((node: any) => [String(node.id), node]));
  return workflow.extra.linearData.inputs.map(([node, name]: [number, string], order: number) => {
    const type = (nodes.get(String(node)) as any).type;
    const loader = name === 'image' && ['LoadImage', 'LoadImageMask'].includes(type);
    const choices = loader ? (recorded[type].input.required.image[0] as string[]) : undefined;
    const id = `${graphId}:${node}:${name}`;
    return {
      id,
      graph_id: graphId,
      node_id: String(node),
      input_name: name,
      label: name,
      order,
      prominent: true,
      resolved: true,
      value: loader ? (nodes.get(String(node)) as any).widgets_values[0] : undefined,
      schema: loader
        ? {
            type: 'COMBO',
            image_upload: true,
            choices: [...choices!, ...(stored(values[id]) ? [values[id]] : [])],
          }
        : { type: 'STRING' },
    };
  });
}

/** A draft with a packaged workflow loaded as Create loads it. */
function loaded(stem: string, change: Partial<Draft> = {}): Draft {
  const workflow = entry(stem).workflow;
  const values = Object.fromEntries(inspected(workflow).map((b) => [b.id, b.value ?? 'x']));
  return { ...emptyDraft(), workflow, workflowName: `dgx-monarch-${stem}`, values, ...change };
}

/** A driver schema without the classes some editing workflows need. */
function without(...classes: string[]) {
  return Object.fromEntries(Object.entries(recorded).filter(([name]) => !classes.includes(name)));
}

interface Setup {
  draft: Draft;
  workflowId?: string;
  info?: Record<string, any>;
  workflows?: WorkflowEntry[];
  assets?: (state: { draft: Draft }) => unknown;
  /** A binding the import lists under All inputs only, as App Mode may leave an image input. */
  hidden?: string;
}

/** Send to edit against a fake companion and App: it records each request, import and notice. */
function harness(setup: Setup) {
  const state = {
    draft: setup.draft,
    bindings: setup.draft.workflow ? inspected(setup.draft.workflow, setup.draft.values) : [],
    workflowId: setup.workflowId ?? '',
  };
  const calls: Array<{ path: string; body: any }> = [];
  const imports: Array<{ value: Draft; templateId?: string }> = [];
  const views: string[] = [];
  const notices: string[] = [];
  const errors: string[] = [];
  const offers: EditOffer[] = [];
  const tabs: string[] = [];
  const client = {
    async request(path: string, body: any) {
      calls.push({ path, body });
      if (path === '/workflows/inspect')
        return { workflow: body.workflow, bindings: inspected(body.workflow) };
      if (path === '/assets') return setup.assets ? setup.assets(state) : RECEIPT;
      throw new Error(`unexpected ${path}`);
    },
  };
  const context: EditContext = {
    client: client as never,
    info: () => setup.info,
    workflows: () => setup.workflows ?? entries,
    workflowId: () => state.workflowId,
    draft: () => state.draft,
    bindings: () => state.bindings,
  };
  const actions: EditActions = {
    async withBusy(action) {
      try {
        await action();
      } catch (error) {
        errors.push((error as Error).message);
      }
    },
    async importDraft(value, templateId) {
      imports.push({ value: structuredClone(value), templateId });
      // The import inspects the workflow with the draft's values, so a bound copy is a choice.
      state.draft = value;
      state.bindings = inspected(value.workflow!, value.values).map((binding) =>
        binding.id === setup.hidden ? { ...binding, prominent: false } : binding,
      );
      state.workflowId = templateId ?? '';
    },
    navigate: (view) => views.push(view),
    setNotice: (text) => notices.push(text),
    selectTab: (tab) => tabs.push(tab),
  };
  const send = (artifact: Artifact) =>
    sendToEdit(artifact, context, actions, (plan) => offers.push(plan));
  return { state, calls, imports, views, notices, errors, offers, tabs, send };
}

test('the editing workflows are the packaged ones that load an image, make no video and run on the driver', () => {
  const editing = (info: Record<string, any>) =>
    entries.filter((row) => editingWorkflow(row.workflow, info)).map((row) => row.name);
  // Of the 23 packaged graphs with an image input, image to video, motion transfer and the PiD
  // upscale stay out.
  assert.deepEqual(editing(recorded), [
    'dgx-monarch-flux3-klein-base-edit',
    'dgx-monarch-flux3-klein-edit',
    'dgx-monarch-mage-flow-edit-turbo',
    'dgx-monarch-mage-flow-edit',
    'dgx-monarch-qwen-image21-edit-mask-rgba',
    'dgx-monarch-qwen-image21-edit-rgba',
  ]);
  // A driver without a workflow's nodes cannot run it, so Studio does not offer it.
  assert.deepEqual(
    editing(without('ConditioningZeroOut', 'TextEncodeMageFlowEdit', 'TextEncodeQwenImage21')),
    ['dgx-monarch-flux3-klein-base-edit'],
  );
  assert.deepEqual(editing({ StudioCPUFixture: {} }), []);
});

test('Send to edit keeps a loaded workflow that supports Edit and binds the output to its image input', async () => {
  const draft = loaded('flux3-klein-base-edit', { parent_artifact_ids: ['mask-upload'] });
  const run = harness({ draft, workflowId: entry('flux3-klein-base-edit').id, info: recorded });
  await run.send(output);
  assert.deepEqual(run.offers, []);
  assert.deepEqual(run.errors, []);
  // The companion copies the output into the input store; no other workflow is inspected.
  assert.deepEqual(run.calls, [
    { path: '/assets', body: { kind: 'input', artifact_id: 'output' } },
  ]);
  const [{ value, templateId }] = run.imports;
  const image = 'flux.3-klein-base-image-editing:8:image';
  assert.equal(value.values[image], 'studio/copy.png');
  assert.deepEqual(value.workflow, draft.workflow);
  assert.deepEqual({ ...value.values, [image]: draft.values[image] }, draft.values);
  assert.equal(value.operation, 'edit');
  assert.deepEqual(value.parent_artifact_ids, ['mask-upload', 'output', 'copy']);
  assert.equal(templateId, entry('flux3-klein-base-edit').id);
  assert.deepEqual(run.views, ['workspace']);
  assert.match(run.notices[0], /the image input on node 8 of FLUX\.3 Klein Base image editing/);
  // The Inputs tab shows the prominent image input, so no other tab is chosen.
  assert.deepEqual(run.tabs, []);
  // With the copy bound, Generate no longer waits for an image.
  assert.equal(editImageNeed(run.state.draft, run.state.bindings, recorded), null);
});

test('an image input that App Mode leaves under All inputs opens that tab after the bind', async () => {
  const draft = loaded('flux3-klein-base-edit');
  const image = 'flux.3-klein-base-image-editing:8:image';
  const run = harness({
    draft,
    workflowId: entry('flux3-klein-base-edit').id,
    info: recorded,
    hidden: image,
  });
  await run.send(output);
  assert.deepEqual(run.errors, []);
  assert.equal(run.imports[0].value.values[image], 'studio/copy.png');
  assert.deepEqual(run.tabs, ['advanced']);
  assert.deepEqual(run.views, ['workspace']);
});

test('with one editing workflow on the driver, Send to edit loads it and binds the output', async () => {
  const draft = loaded('flux3-klein-base-t2i', { parent_artifact_ids: ['older'] });
  const info = without('ConditioningZeroOut', 'TextEncodeMageFlowEdit', 'TextEncodeQwenImage21');
  const run = harness({ draft, info });
  await run.send(output);
  const chosen = entry('flux3-klein-base-edit');
  assert.deepEqual(run.offers, []);
  assert.deepEqual(
    run.calls.map((call) => call.path),
    ['/workflows/inspect', '/assets'],
  );
  assert.equal(run.calls[0].body.workflow, chosen.workflow);
  const [{ value, templateId }] = run.imports;
  assert.deepEqual(value.workflow, chosen.workflow);
  assert.equal(value.workflowName, 'dgx-monarch-flux3-klein-base-edit');
  // The loaded workflow starts from its own values, with the output in its image input.
  assert.deepEqual(value.values, { 'flux.3-klein-base-image-editing:8:image': 'studio/copy.png' });
  assert.equal(value.operation, 'edit');
  assert.deepEqual(value.parent_artifact_ids, ['output', 'copy']);
  assert.equal(templateId, chosen.id);
  assert.deepEqual(run.views, ['workspace']);
  assert.match(
    run.notices[0],
    /node 8 of FLUX\.3 Klein Base image editing and set the Operation to Edit/,
  );
});

test('with several editing workflows, Send to edit offers the choice and changes nothing', async () => {
  const draft = loaded('flux3-klein-base-t2i');
  const before = structuredClone(draft);
  const run = harness({ draft, info: recorded });
  await run.send(output);
  assert.equal(run.offers.length, 1);
  const [offer] = run.offers;
  assert.equal(offer.kind, 'choose');
  assert.deepEqual(offer.kind === 'choose' && offer.choices.map((row) => row.title), [
    'FLUX.3 Klein Base image editing',
    'FLUX.3 Klein image editing',
    'Mage-Flow image editing quality',
    'Mage-Flow image editing Turbo',
    'Qwen Image 2.1 masked RGBA reference edit',
    'Qwen Image 2.1 ten-reference RGBA edit',
  ]);
  assert.deepEqual([run.calls, run.imports, run.views, run.notices], [[], [], [], []]);
  assert.equal(run.state.draft, draft);
  assert.deepEqual(draft, before);
});

test('with no editing workflow, Send to edit says why and changes nothing', async () => {
  const t2i = loaded('flux3-klein-base-t2i');
  const edit = loaded('flux3-klein-base-edit');
  const cases: Array<[Setup, Artifact, string]> = [
    // The tester's case on a driver that runs none of the packaged editing workflows.
    [{ draft: t2i, info: { StudioCPUFixture: {} } }, output, NO_EDITOR],
    [{ draft: t2i, info: recorded, workflows: [entry('flux3-klein-base-t2i')] }, output, NO_EDITOR],
    [{ draft: emptyDraft(), info: {} }, output, UNREAD_SCHEMA],
    [{ draft: edit, info: recorded }, { ...output, kind: 'video' }, VIDEO_OUTPUT],
    [{ draft: edit, info: recorded }, { ...output, kind: 'mask' } as never, MASK_OUTPUT],
  ];
  for (const [setup, artifact, reason] of cases) {
    const before = structuredClone(setup.draft);
    const run = harness(setup);
    await run.send(artifact);
    assert.deepEqual(run.offers, [{ kind: 'none', reason }]);
    assert.deepEqual([run.calls, run.imports, run.views, run.notices], [[], [], [], []]);
    assert.deepEqual(run.state.draft, before);
  }
  // A loaded editing workflow is kept even before the driver's node list is read.
  const plan = editPlan(
    output,
    { ...edit, bindings: inspected(edit.workflow!) },
    entries,
    undefined,
  );
  assert.equal(plan.kind, 'keep');
});

test('the output lands on the image input the App Mode layout lists first, never on a mask', () => {
  const target = (stem: string, reorder = (rows: Binding[]) => rows) => {
    const workflow = entry(stem).workflow;
    return editTarget(workflow, reorder(inspected(workflow)), recorded)?.node_id;
  };
  assert.equal(target('flux3-klein-base-edit'), '8');
  // Load Image (as Mask) on node 7 reads the input store too, but gives a mask.
  assert.equal(target('qwen-image21-edit-mask-rgba'), '6');
  // The canvas comes first in the layout of the ten-reference edit, whatever the listing order.
  assert.equal(
    target('qwen-image21-edit-rgba', (rows) => [...rows].reverse()),
    '6',
  );
  assert.equal(target('ltx25-flf2v'), '10');
  // An image input that a wire feeds cannot take a value.
  const wired = (rows: Binding[]) =>
    rows.map((row) =>
      row.node_id === '6' && row.input_name === 'image' ? { ...row, resolved: false } : row,
    );
  assert.equal(target('qwen-image21-edit-rgba', wired), '7');
  // An API prompt on a driver that has not listed Load Image yet still names its image input.
  const prompt = {
    '1': { class_type: 'LoadImage', inputs: { image: 'x.png' } },
    '2': { class_type: 'SaveImage', inputs: { images: ['1', 0] } },
  };
  const binding = {
    id: 'root:1:image',
    graph_id: 'root',
    node_id: '1',
    input_name: 'image',
    type: 'COMBO',
  };
  assert.equal(editTarget(prompt, [binding], {})?.id, 'root:1:image');
  // A loader that reads the output store cannot take a stored upload.
  const output = {
    LoadImage: {
      input: { required: { image: [['x.png'], { image_upload: true, image_folder: 'output' }] } },
    },
  };
  assert.equal(editTarget(prompt, [binding], output as never), null);
});

test('a refused copy, or a workflow changed while copying, leaves the draft unchanged', async () => {
  const draft = loaded('flux3-klein-base-edit');
  const refused = harness({
    draft,
    info: recorded,
    assets: () => {
      throw new Error('Only a saved image output of a job can become an input image.');
    },
  });
  await refused.send(output);
  assert.deepEqual(refused.errors, [
    'Only a saved image output of a job can become an input image.',
  ]);
  assert.deepEqual([refused.imports, refused.views, refused.notices], [[], [], []]);
  assert.equal(refused.state.draft, draft);

  const changed = harness({
    draft,
    info: recorded,
    assets: (state) => {
      state.draft = loaded('flux3-klein-edit');
      return RECEIPT;
    },
  });
  await changed.send(output);
  assert.match(changed.errors[0], /copied as studio\/copy\.png, but the draft's workflow changed/);
  assert.deepEqual(changed.imports, []);
});

test('an uploaded input binds as it is, with no copy', async () => {
  const run = harness({ draft: loaded('flux3-klein-base-edit'), info: recorded });
  await run.send({ id: 'upload', kind: 'input', binding_value: 'studio/upload.png' } as never);
  assert.deepEqual(run.calls, []);
  const [{ value }] = run.imports;
  assert.equal(value.values['flux.3-klein-base-image-editing:8:image'], 'studio/upload.png');
  assert.deepEqual(value.parent_artifact_ids, ['upload']);
});

test('under Edit, Generate waits until the image input holds an image the driver offers', () => {
  const draft = loaded('flux3-klein-base-edit', { operation: 'edit' });
  const image = 'flux.3-klein-base-image-editing:8:image';
  const need = (value: unknown, change: Partial<Draft> = {}) => {
    const next = { ...draft, values: { ...draft.values, [image]: value }, ...change };
    return editImageNeed(next, inspected(next.workflow!, next.values), recorded);
  };
  const reason =
    'Edit needs an image in the image input on node 8. Use Send to edit on an output, or upload an image on the Assets tab.';
  assert.equal(need(''), reason);
  // A file the driver does not offer is no image either.
  assert.equal(need('missing.png'), reason);
  // A driver with no input images offers none, whatever name the workflow still carries.
  const bare = inspected(draft.workflow!).map((b) =>
    b.id === image ? { ...b, schema: { ...b.schema, choices: [] } } : b,
  );
  const named = { ...draft, values: { ...draft.values, [image]: 'reference.png' } };
  assert.equal(editImageNeed(named, bare, recorded), reason);
  assert.equal(need(IMAGE_CHOICES[0]), null);
  // With no draft value the graph's own runs, and this driver offers reference.png.
  assert.ok(IMAGE_CHOICES.includes('reference.png'));
  assert.equal(need(undefined), null);
  assert.equal(need('studio/copy.png'), null);
  // Other operations, and graphs without an image input, are not this rule's to block.
  assert.equal(need('', { operation: 'image' }), null);
  const t2i = loaded('flux3-klein-base-t2i', { operation: 'edit' });
  assert.equal(editImageNeed(t2i, inspected(t2i.workflow!), recorded), null);
});

test('a kept workflow keeps its values and lineage; a loaded one starts from its own', () => {
  const draft = {
    ...emptyDraft(),
    workflow: {},
    values: { a: 1, b: 2 },
    parent_artifact_ids: ['x'],
  };
  const target = { id: 'b' } as Binding;
  assert.deepEqual(editDraft(draft, target, 'v', ['o', 'c']), {
    ...draft,
    values: { a: 1, b: 'v' },
    parent_artifact_ids: ['x', 'o', 'c'],
    operation: 'edit',
  });
  const loadedDraft = editDraft(draft, target, 'v', ['o', 'o'], {
    workflow: { id: 'w' },
    name: 'w',
  });
  assert.deepEqual(
    [
      loadedDraft.workflow,
      loadedDraft.workflowName,
      loadedDraft.values,
      loadedDraft.parent_artifact_ids,
    ],
    [{ id: 'w' }, 'w', { b: 'v' }, ['o']],
  );
});

test('Reuse settings, Upscale and Load workflow keep their notices', () => {
  assert.equal(reuseNotice(), 'Recorded settings loaded into a new editable draft.');
  assert.equal(
    reuseNotice('upscale'),
    'Output lineage attached. Set the corresponding workflow asset input before generating upscale.',
  );
});
