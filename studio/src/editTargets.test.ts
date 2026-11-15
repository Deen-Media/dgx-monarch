import test from 'node:test';
import assert from 'node:assert/strict';
import { bundled, recorded } from './workflowFixtures.js';
import {
  bindOutput,
  editImageNeed,
  editPlan,
  type EditActions,
  type EditContext,
} from './editHandoff.ts';
import { emptyDraft, type Binding, type Draft } from './model.ts';
import type { Artifact, WorkflowEntry } from './contracts.ts';

const entries = bundled as unknown as WorkflowEntry[];
const entry = (stem: string) => entries.find((row) => row.name === `dgx-monarch-${stem}`)!;
const output = { id: 'output', job_id: 'job', kind: 'image' } as Artifact;
const RECEIPT = { id: 'copy', binding_value: 'studio/copy.png', width: 128, height: 96 };

/**
 * A packaged workflow's App Mode inputs as the companion's inspection lists them; as there, a
 * stored upload that the values name joins the choices of an image upload input.
 */
function inspected(workflow: Record<string, any>, values: Record<string, unknown> = {}): Binding[] {
  const nodes = new Map(workflow.nodes.map((node: any) => [String(node.id), node]));
  return workflow.extra.linearData.inputs.map(([node, name]: [number, string], order: number) => {
    const type = (nodes.get(String(node)) as any).type;
    const id = `${workflow.id}:${node}:${name}`;
    const loader = name === 'image' && type === 'LoadImage';
    const choices = recorded.LoadImage.input.required.image[0] as string[];
    const stored = typeof values[id] === 'string' ? [values[id]] : [];
    return {
      id,
      graph_id: String(workflow.id),
      node_id: String(node),
      input_name: name,
      order,
      prominent: true,
      resolved: true,
      schema: loader
        ? { type: 'COMBO', image_upload: true, choices: [...choices, ...stored] }
        : { type: 'STRING' },
    };
  });
}

test('Send to edit binds an output to the MiniMax-H3-Image edit, and Generate stops waiting', async () => {
  // On a driver with the full recorded schema the MiniMax-H3-Image edit is one of the editing
  // workflows offered; choosing it binds the copy to node 11's image, the input its layout lists.
  const start = entry('minimax-h3-image-t2i').workflow;
  const t2iValues = Object.fromEntries(inspected(start).map((b) => [b.id, 'x']));
  let draft: Draft = { ...emptyDraft(), workflow: start, workflowName: 'm', values: t2iValues };
  let bindings = inspected(start);
  const plan = editPlan(
    output,
    { workflow: start, bindings, values: draft.values },
    entries,
    recorded,
  );
  assert.equal(plan.kind, 'choose');
  const choice =
    plan.kind === 'choose' ? plan.choices.find((row) => row.name.includes('minimax')) : null;
  assert.equal(choice?.name, 'dgx-monarch-minimax-h3-image-edit');
  const calls: string[] = [];
  const notices: string[] = [];
  const context: EditContext = {
    client: {
      async request(path: string, body: any) {
        calls.push(path);
        if (path === '/workflows/inspect')
          return { workflow: body.workflow, bindings: inspected(body.workflow) };
        if (path === '/assets') return RECEIPT;
        throw new Error(`unexpected ${path}`);
      },
    } as never,
    info: () => recorded,
    workflows: () => entries,
    workflowId: () => '',
    draft: () => draft,
    bindings: () => bindings,
  };
  const actions: EditActions = {
    withBusy: (action) => action(),
    async importDraft(value) {
      draft = value;
      bindings = inspected(value.workflow!, value.values);
    },
    navigate: () => {},
    setNotice: (text) => notices.push(text),
  };
  await bindOutput(output, choice!, context, actions);
  const edit = entry('minimax-h3-image-edit').workflow;
  assert.deepEqual(calls, ['/workflows/inspect', '/assets']);
  assert.equal(draft.workflowName, 'dgx-monarch-minimax-h3-image-edit');
  assert.deepEqual(draft.values, { [`${edit.id}:11:image`]: 'studio/copy.png' });
  assert.equal(draft.operation, 'edit');
  assert.deepEqual(draft.parent_artifact_ids, ['output', 'copy']);
  assert.match(notices[0], /the image input on node 11 of MiniMax-H3-Image image editing/);
  assert.equal(editImageNeed(draft, bindings, recorded), null);
});
