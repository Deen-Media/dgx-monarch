import test from 'node:test';
import assert from 'node:assert/strict';
import { artifactDraft, liveArtifact } from './artifactDraft.ts';
import type { Artifact, Job } from './contracts.ts';
import { recordedSettings } from './recordedSettings.ts';

const artifact = { id: 'output', job_id: 'job' } as Artifact;
function job(requested: Record<string, unknown> = {}, prompt?: Record<string, unknown>): Job {
  return {
    id: 'job',
    state: 'succeeded',
    created_at: 1,
    submission: {
      workflow: { id: 'original-ui', nodes: [], extra: { extension: 'keep' } },
      requested: { workflowName: 'Original workflow', operation: 'image', ...requested },
      prompt,
    },
  };
}

test('reuse preserves edited effective inputs and original source metadata without aliasing', () => {
  const original = job(
    { values: { 'original-ui:1:text': 'edited' }, extension: { keep: true } },
    {
      '1': { class_type: 'Render', inputs: { text: 'edited', seed: 8 } },
    },
  );
  const before = structuredClone(original);
  const draft = artifactDraft(artifact, original);
  assert.deepEqual(draft.workflow, original.submission.prompt);
  assert.deepEqual(draft.values, {});
  assert.deepEqual((draft as any).source_workflow, original.submission.workflow);
  assert.equal((draft as any).source_workflowName, 'Original workflow');
  assert.deepEqual((draft as any).extension, { keep: true });
  assert.match(draft.workflowName, /^Effective settings/);
  assert.deepEqual(draft.parent_artifact_ids, ['output']);
  draft.workflow!['1'].inputs.text = 'new edit';
  (draft as any).source_workflow.extra.extension = 'changed';
  assert.deepEqual(original, before);
});

test('random seed reuse freezes every actual seed including expanded primitive fanout', () => {
  const original = job(
    { seed_policy: 'random', values: { 'original-ui:primitive:value': 5 } },
    {
      '1': { class_type: 'Render', inputs: { seed: 123 } },
      '2': { class_type: 'Render', inputs: { seed: 456 } },
    },
  );
  const draft = artifactDraft(artifact, original);
  assert.equal(draft.seed_policy, 'fixed');
  assert.equal(draft.workflow!['1'].inputs.seed, 123);
  assert.equal(draft.workflow!['2'].inputs.seed, 456);
  assert.deepEqual(draft.values, {});
});

test('expanded styles and inserted LoRAs are retained exactly once for reuse and edit handoff', () => {
  const original = job(
    {
      styles: [{ id: 'style', prompt: 'oil painting, {prompt}' }],
      style_bindings: { prompt: 'original-ui:1:text' },
      lora_stack: [{ id: 'paint.safetensors', enabled: true, model_strength: 0.5 }],
      lora_insertion: { loader: 'LoraLoaderModelOnly', model_targets: [], clip_targets: [] },
    },
    {
      '1': {
        class_type: 'Render',
        inputs: { text: 'oil painting, cat', model: ['__studio_lora__0', 0] },
      },
      __studio_lora__0: {
        class_type: 'LoraLoaderModelOnly',
        inputs: {
          lora_name: 'paint.safetensors',
          strength_model: 0.5,
          model: ['source', 0],
        },
      },
      source: { class_type: 'ModelLoader', inputs: {} },
    },
  );
  const draft = artifactDraft(artifact, original, 'edit');
  assert.deepEqual(draft.workflow, original.submission.prompt);
  assert.deepEqual(draft.styles, []);
  assert.deepEqual(draft.lora_stack, []);
  assert.equal(draft.style_bindings, undefined);
  assert.equal(draft.lora_insertion, undefined);
  assert.equal(draft.operation, 'edit');
});

test('legacy records without a usable effective prompt retain explicitly labelled requested settings', () => {
  for (const prompt of [undefined, {}, { broken: true }]) {
    const original = job({ seed_policy: 'random', values: { 'original-ui:1:seed': 5 } }, prompt);
    const draft = artifactDraft(artifact, original);
    assert.deepEqual(draft.workflow, original.submission.workflow);
    assert.deepEqual(draft.values, original.submission.requested!.values);
    assert.equal(draft.seed_policy, 'random');
    assert.match(draft.workflowName, /^Requested settings/);
  }
});

test('external prompt-only jobs can reconstruct recorded inputs without an original UI graph', () => {
  const original = job({}, { '1': { class_type: 'Render', inputs: { seed: 42 } } });
  delete original.submission.workflow;
  delete original.submission.requested;
  const draft = artifactDraft(artifact, original);
  assert.deepEqual(draft.workflow, original.submission.prompt);
  assert.equal(draft.seed_policy, 'fixed');
  assert.throws(() => artifactDraft(artifact, { ...original, submission: {} }), /unavailable/);
});

test('copied named inputs use actual random seeds and expanded text, grouped by execution identity', () => {
  const original = job(
    { seed_policy: 'random', values: { 'original-ui:1:seed': 5 } },
    {
      '1': { class_type: 'Render', inputs: { seed: 123, text: 'styled cat' } },
      '__studio_subgraph__2/3': { class_type: 'Render', inputs: { seed: 456 } },
    },
  );
  assert.deepEqual(JSON.parse(recordedSettings(original, 'inputs')), {
    '1': { seed: 123, text: 'styled cat' },
    '__studio_subgraph__2/3': { seed: 456 },
  });
  assert.deepEqual(JSON.parse(recordedSettings(original, 'prompt')), original.submission.prompt);
  assert.throws(() => recordedSettings(job({ values: { stale: 5 } }), 'inputs'), /unavailable/);
});

test('copy seed copies the seeds the job ran with, every digit included', () => {
  const two = job(
    { seed_policy: 'random', values: { 'original-ui:1:seed': 5 } },
    {
      '1': { class_type: 'Render', inputs: { seed: 123, text: 'styled cat' } },
      '__studio_subgraph__2/3': { class_type: 'Render', inputs: { seed: 456 } },
    },
  );
  // Seeds come from the summary list, which the companion fills for an imported job too.
  assert.throws(() => recordedSettings(two, 'seed'), /seeds are unavailable/);
  two.submission.effective = { seeds: { '1.seed': 123 } };
  assert.equal(recordedSettings(two, 'seed'), '123');
  // A seed above 2**53 - 1 arrives as its digits and copies as them.
  two.submission.effective = { seeds: { '1.seed': '18446744073709551615' } };
  assert.equal(recordedSettings(two, 'seed'), '18446744073709551615');
  two.submission.effective = {
    seeds: { '1.seed': 9007199254740991, '__studio_subgraph__2/3.seed': '9007199254740992' },
  };
  assert.equal(
    recordedSettings(two, 'seed'),
    '{\n  "1.seed": 9007199254740991,\n  "__studio_subgraph__2/3.seed": 9007199254740992\n}',
  );
  // A number the browser already rounded never copies as if it were exact.
  two.submission.effective = { seeds: { '1.seed': 2 ** 64 } };
  assert.throws(() => recordedSettings(two, 'seed'), /seeds are unavailable/);
  two.submission.effective = { seeds: {} };
  assert.throws(() => recordedSettings(two, 'seed'), /recorded no seed/);
});

test('the viewer shows the snapshot copy of a chosen artifact and keeps one the snapshot omits', () => {
  const chosen = { id: 'a', state: 'old' } as never;
  const fresh = { id: 'a', state: 'new' } as never;
  assert.equal(liveArtifact([fresh], chosen), fresh);
  assert.equal(liveArtifact([], chosen), chosen);
  assert.equal(liveArtifact([fresh], null), null);
});
