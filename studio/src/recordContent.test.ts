import test from 'node:test';
import assert from 'node:assert/strict';
import { recordContent } from './recordContent.ts';
import { emptyDraft } from './model.ts';
import type { RecordData, RecordKind } from './contracts.ts';

const originalDraft = {
  ...emptyDraft(),
  workflow: { '1': { class_type: 'Render', inputs: { seed: 7 } } },
  values: { 'root:1:seed': 7 },
};
const otherWorkspace = {
  draft: {
    ...emptyDraft(),
    workflow: { '2': { class_type: 'Other', inputs: { seed: 99 } } },
    values: { 'root:2:seed': 99 },
  },
  selectedArtifactId: 'new-output',
  activeJobIds: ['new-job'],
  layout: { width: 500, showTelemetry: false, showConsole: true },
};

for (const kind of ['presets', 'sessions', 'workflows'] as RecordKind[]) {
  test(`renaming a saved ${kind} record preserves its captured content across workspace changes`, () => {
    const saved: RecordData = {
      name: 'Original',
      draft: originalDraft,
      workflow: originalDraft.workflow,
      selected_artifact_id: 'old-output',
      active_job_ids: ['old-job'],
      layout: { width: 300, showTelemetry: true, showConsole: false },
      notes: 'Retained notes',
      extension: { future: true },
    };
    const before = structuredClone(saved);
    const updated = {
      ...recordContent(kind, saved, otherWorkspace),
      name: 'Renamed',
      project_ids: ['new-project'],
    };
    assert.deepEqual(updated, { ...before, name: 'Renamed', project_ids: ['new-project'] });
    assert.deepEqual(saved, before);
  });
}

test('new presets, sessions and workflows capture the current workspace', () => {
  assert.deepEqual(recordContent('presets', null, otherWorkspace), { draft: otherWorkspace.draft });
  assert.deepEqual(recordContent('sessions', null, otherWorkspace), {
    draft: otherWorkspace.draft,
    selected_artifact_id: 'new-output',
    active_job_ids: ['new-job'],
    layout: otherWorkspace.layout,
  });
  assert.deepEqual(recordContent('workflows', null, otherWorkspace), {
    workflow: otherWorkspace.draft.workflow,
  });
});
