import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GALLERY_KINDS,
  GALLERY_STATES,
  galleryEntries,
  galleryQuery,
  hasDimensions,
  virtualRows,
  parameterDifferences,
} from './gallery.ts';
import type { Artifact, Job } from './contracts.ts';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';

const filter = {
  query: '',
  kind: '',
  operation: '',
  model: '',
  status: '',
  project: '',
  from: '',
  until: '',
  sort: 'newest',
};

test('gallery joins durable provenance for model, operation, status and project', () => {
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
  const [entry] = galleryEntries(artifacts, jobs);
  assert.equal(entry.job, jobs[0]);
  assert.equal(entry.operation, 'edit');
  assert.equal(entry.status, 'succeeded');
  assert.equal(entry.project, 'p');
  assert.deepEqual(entry.models, ['model.safetensors']);
  assert.equal(galleryEntries(artifacts, [])[0].status, 'unknown');
});

test('every gallery filter option maps to a field the companion stores', () => {
  const query = (changes: Partial<typeof filter>, scope = {}) =>
    new URLSearchParams(galleryQuery({ ...filter, ...changes }, { projectId: '', ...scope }));
  for (const { value } of GALLERY_STATES) {
    const params = query({ status: value });
    assert.notEqual(params.get('status'), 'input');
    assert.notEqual(params.get('status'), 'intermediate');
  }
  assert.equal(query({ status: 'intermediate' }).get('intermediate'), 'true');
  assert.equal(query({ status: 'intermediate' }).has('status'), false);
  assert.equal(query({ status: 'failed' }).get('status'), 'failed');
  assert.deepEqual(
    GALLERY_KINDS.map((option) => option.value),
    ['', 'image', 'video', 'input', 'mask'],
  );
  assert.equal(query({ kind: 'input' }).get('kind'), 'input');
  const sized = query({}, { projectId: 'p', width: 512, height: null });
  assert.equal(sized.get('width'), '512');
  assert.equal(sized.has('height'), false);
  assert.equal(sized.get('project_id'), 'p');
  assert.equal(query({ sort: 'oldest' }).get('sort'), 'oldest');
});

test('size filters are offered only when loaded rows carry dimensions', () => {
  const output = { id: 'o', job_id: 'j', filename: 'o.png', created_at: 1, kind: 'image' };
  assert.equal(
    hasDimensions([{ ...output, metadata: { width: null } }] as unknown as Artifact[]),
    false,
  );
  assert.equal(hasDimensions([output] as Artifact[]), false);
  assert.equal(hasDimensions([{ ...output, metadata: { width: 512 } }] as Artifact[]), true);
});

test('Gallery opens without an empty-state flash, unreachable states or output size filters', async () => {
  const noop = async () => {};
  const body = await renderComponent('./views/Gallery.svelte', {
    projectSummaries: [],
    jobs: [],
    projectId: '',
    selected: null,
    client: { request: noop },
    navigate: () => {},
    onselect: () => {},
    oncompare: () => {},
    reuse: noop,
    setNotice: () => {},
  });
  const text = visibleText(body);
  assert.match(text, /Loading outputs…/);
  assert.doesNotMatch(text, /No matching outputs/);
  assert.match(text, /Width and height filters apply to uploaded inputs and masks/);
  assert.doesNotMatch(body, /type="number"/);
  assert.doesNotMatch(body, /<option value="input">input<\/option>/);
  assert.match(body, /<option value="intermediate">Intermediate previews<\/option>/);
  assert.match(body, /<option value="input">Uploaded inputs<\/option>/);
});

test('a settled filter change keeps the current page until the new page arrives', async () => {
  const loadCode = await componentFunctions('./views/Gallery.svelte', ['loadPage']);
  const harness = new Function(
    'client',
    `'use strict';
    let generation = 0, loading = true, loadError = '', nextCursor = null, scroller = undefined;
    let loaded = [{ id: 'kept' }];
    const query = 'limit=100';
    ${loadCode}
    return { loadPage, state: () => ({ loaded, loading, loadError }) };`,
  );
  let finish!: (page: unknown) => void;
  let refuse!: (cause: Error) => void;
  const replies = [
    new Promise((resolve) => {
      finish = resolve;
    }),
    new Promise((_resolve, reject) => {
      refuse = reject;
    }),
  ];
  const gallery = harness({ request: () => replies.shift() });
  const pending = gallery.loadPage('kind=video', true);
  assert.deepEqual(gallery.state().loaded, [{ id: 'kept' }], 'no blank grid during the request');
  assert.equal(gallery.state().loading, true);
  finish({ items: [{ id: 'video' }], next_cursor: null });
  await pending;
  assert.deepEqual(gallery.state().loaded, [{ id: 'video' }]);
  const failed = gallery.loadPage('kind=image', true);
  refuse(new Error('Companion unavailable'));
  await failed;
  assert.deepEqual(gallery.state(), {
    loaded: [{ id: 'video' }],
    loading: false,
    loadError: 'Companion unavailable',
  });
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
