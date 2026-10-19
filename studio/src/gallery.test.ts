import test from 'node:test';
import assert from 'node:assert/strict';
import { filterGallery, galleryEntries, virtualRows, parameterDifferences } from './gallery.ts';
import type { Artifact, Job } from './contracts.ts';
test('gallery joins durable provenance for model, operation, status and project filters', () => {
  const artifacts = [
    { id: 'a', job_id: 'j', filename: 'a.png', created_at: 10, kind: 'image' },
  ] as Artifact[];
  const jobs = [
    {
      id: 'j',
      state: 'succeeded',
      submission: {
        project_id: 'p',
        requested: { operation: 'edit' },
        prompt: { '1': { inputs: { ckpt_name: 'model.safetensors' } } },
      },
    },
  ] as unknown as Job[];
  const entries = galleryEntries(artifacts, jobs);
  const filter = {
    query: 'model',
    kind: 'image',
    operation: 'edit',
    model: 'safetensors',
    status: 'succeeded',
    project: 'p',
    from: '',
    until: '',
    sort: 'newest',
  };
  assert.equal(filterGallery(entries, filter).length, 1);
  assert.equal(filterGallery(entries, { ...filter, status: 'failed' }).length, 0);
});
test('virtual gallery bounds mounted items independently of total library size', () => {
  const window = virtualRows(100000, 1000, 660, 33000);
  assert.equal(window.columns, 4);
  assert.ok(window.end - window.start <= 16);
  assert.equal(window.before, 32670);
  assert.equal(virtualRows(0, 1, 1, 0).end, 0);
});
test('comparison reports changed and missing values without guessing semantics', () => {
  assert.deepEqual(parameterDifferences({ seed: 1, lora: ['x'] }, { seed: 2 }), [
    { key: 'lora.0', left: 'x', right: undefined },
    { key: 'seed', left: 1, right: 2 },
  ]);
});
