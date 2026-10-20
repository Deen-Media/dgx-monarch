import test from 'node:test';
import assert from 'node:assert/strict';
import { layoutWorkflow, unchangedWorkflow } from './workflowLayout.ts';
import { inspectDraft } from './workflowInspector.ts';

test('visual layout edits survive repair-editor validation and preserve typed values', async () => {
  const original = {
    '1': { class_type: 'Node', inputs: { seed: 5 } },
    extra: {
      retained: 'metadata',
      dgxm_studio: { bindings: { '1:seed': { target: '1:seed', extension: true } } },
    },
  };
  const bindings = [
    {
      id: '1:seed',
      label: 'Random seed',
      prominent: true,
      order: 2,
      group: 'Sampling',
      resolved: true,
      value: 5,
    },
  ];
  const { workflow, rawWorkflow } = layoutWorkflow(original, bindings);
  const edited = JSON.parse(rawWorkflow);
  assert.equal(edited.extra.dgxm_studio.bindings['1:seed'].label, 'Random seed');
  assert.equal(edited.extra.dgxm_studio.bindings['1:seed'].target, '1:seed');
  assert.equal(edited.extra.dgxm_studio.bindings['1:seed'].extension, true);
  assert.equal(edited.extra.retained, 'metadata');
  assert.equal(original.extra.dgxm_studio.bindings['1:seed'].extension, true);
  assert.equal('label' in original.extra.dgxm_studio.bindings['1:seed'], false);
  const preserve = unchangedWorkflow(workflow, edited);
  assert.equal(preserve, true);
  const inspected = await inspectDraft(
    { request: async () => ({ workflow: edited, bindings, connections: [], errors: [] }) } as any,
    edited,
    'workflow',
    preserve ? { '1:seed': 999 } : undefined,
  );
  assert.equal(inspected.patch.values['1:seed'], 999);
  assert.equal(inspected.patch.workflow.extra.dgxm_studio.bindings['1:seed'].prominent, true);
});
test('explicit workflow JSON changes retain their edited defaults when revalidated', async () => {
  const workflow = { '1': { class_type: 'Node', inputs: { seed: 5 } } };
  const edited = { '1': { class_type: 'Node', inputs: { seed: 23 } } };
  const preserve = unchangedWorkflow(workflow, edited);
  assert.equal(preserve, false);
  const inspected = await inspectDraft(
    {
      request: async () => ({
        workflow: edited,
        bindings: [{ id: '1:seed', resolved: true, value: 23 }],
      }),
    } as any,
    edited,
    'workflow',
    preserve ? { '1:seed': 999 } : undefined,
  );
  assert.equal(inspected.patch.values['1:seed'], 23);
});
