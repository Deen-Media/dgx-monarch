import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';
import { parseExactJSON, readExactJSON, writeExactJSON } from './exactJson.ts';
import { COMPARE_DEPTH, differenceText, parameterDifferences } from './gallery.ts';
import * as gridState from './gridState.ts';
import { GridReviewGuard } from './gridReview.ts';

// Review F12: grid cell inspection, grid export and the comparison keep every digit of a whole
// number above 2**53 - 1, as the gallery and queue exports already do.
const MAX_SAFE = 9007199254740991n;
const UINT64_MAX = '18446744073709551615';

/** A companion that answers each path with recorded text, read by the parser the caller names. */
function companion(texts: Record<string, string>) {
  const asked: [string, boolean][] = [];
  const client = {
    request: async (path: string, _body?: unknown, _method?: string, parse = JSON.parse) => {
      asked.push([path, parse === readExactJSON]);
      if (!(path in texts)) throw new Error(`unexpected ${path}`);
      return parse(texts[path]);
    },
  };
  return { client, asked };
}

const cellCode = await componentFunctions('./components/GridCellDetails.svelte', [
  'inspect',
  'exportCell',
]);
/** The shipped Inspect full submission and Export cell handlers of one grid cell. */
function cell(client: unknown, ids: { jobId?: string; previewId?: string }) {
  return new Function(
    'client',
    'jobId',
    'previewId',
    'readExactJSON',
    'writeExactJSON',
    `'use strict';
    let detail = '', error = '', busy = false;
    const downloads = [];
    const exportText = (text, name) => downloads.push([name, text]);
    const loadOutputs = async () => {};
    ${cellCode}
    return async () => { await inspect(); exportCell(); return { detail, error, downloads }; };`,
  )(client, ids.jobId, ids.previewId, readExactJSON, writeExactJSON);
}

const gridCode = await componentFunctions('./views/Grids.svelte', ['exportGrid']);
/** The shipped Export complete preview and Export manifest and outcomes handler. */
function gridExport(client: unknown) {
  return new Function(
    'client',
    'readExactJSON',
    'writeExactJSON',
    `'use strict';
    const downloads = [];
    let error = '';
    const exportText = (text, name) => downloads.push([name, text]);
    const fail = (cause) => { error = cause instanceof Error ? cause.message : String(cause); };
    ${gridCode}
    return async (value, preview) => { await exportGrid(value, preview); return { downloads, error }; };`,
  )(client, readExactJSON, writeExactJSON);
}

/** A job or preview record as the companion writes it: Python's json keeps every digit. */
const record = (id: string, seeds: string[]) =>
  `{"id": "${id}", "submission": {` +
  `"prompt": {"8": {"class_type": "KSampler", "inputs": {"seed": ${seeds[0]}}}}, ` +
  `"effective": {"seeds": {"8.seed": ${seeds[0]}}}, ` +
  `"recorded": [${seeds.join(', ')}]}}`;

test('Inspect full submission shows and exports adjacent unsafe integers and the uint64 maximum', async () => {
  const seeds = ['9007199254740992', '9007199254740993', UINT64_MAX];
  const { client, asked } = companion({
    '/jobs/job-1': record('job-1', seeds),
    '/jobs/previews/preview-1': record('preview-1', seeds),
  });
  for (const ids of [{ jobId: 'job-1' }, { previewId: 'preview-1' }]) {
    const { detail, error, downloads } = await cell(client, ids)();
    assert.equal(error, '');
    const [[name, text]] = downloads;
    assert.equal(name, `grid-cell-${ids.jobId ?? ids.previewId}.json`);
    assert.equal(text, detail, 'the download is the text on screen');
    for (const seed of seeds) assert.ok(text.includes(seed), `${seed} in ${text}`);
    const exact = parseExactJSON(text) as { submission: { recorded: bigint[] } };
    assert.deepEqual(exact.submission.recorded, seeds.map(BigInt));
    assert.doesNotMatch(text, /18446744073709552000/);
  }
  assert.deepEqual(asked, [
    ['/jobs/job-1', true],
    ['/jobs/previews/preview-1', true],
  ]);
});

test('Export complete preview and Export manifest and outcomes keep every digit', async () => {
  const seeds = ['9007199254740992', '9007199254740993', UINT64_MAX];
  const text = `{"id": "grid-1", "cells": [], "submission": {"seeds": [${seeds.join(', ')}]}}`;
  const { client, asked } = companion({ '/grids/grid-1': text, '/grids/previews/grid-1': text });
  for (const preview of [false, true]) {
    const { downloads, error } = await gridExport(client)({ id: 'grid-1' }, preview);
    assert.equal(error, '');
    const [[name, written]] = downloads;
    assert.equal(name, preview ? 'grid-preview.json' : 'grid-results.json');
    assert.deepEqual(
      (parseExactJSON(written) as { submission: { seeds: bigint[] } }).submission.seeds,
      seeds.map(BigInt),
    );
  }
  assert.deepEqual(asked, [
    ['/grids/grid-1', true],
    ['/grids/previews/grid-1', true],
  ]);
});

const prepareCode = await componentFunctions('./views/Grids.svelte', [
  'prepare',
  'remember',
  'fail',
]);
/** The shipped Validate every cell handler with an incrementing seed from the base seed box. */
function prepareGrid(client: unknown, change: Record<string, unknown> = {}) {
  return new Function(
    'client',
    'gridState',
    'GridReviewGuard',
    'change',
    `'use strict';
    let recoveryBlocked = false, dispatchPending = false, retryPending = false, loadGeneration = 0;
    let busy = false, error = '', confirmed = false, preview = null, grid = null, selected = [];
    let cellPage = 0, requestKey = '', retryKey = '', retryIndices = [], reviewedFields = null;
    let reviewedKey = null;
    let fields = { ...gridState.defaultGridFields(), axisA: '8.steps', valuesA: '[20, 30, 40]',
      seedPolicy: 'increment', seedBinding: '8.seed', baseSeed: ${Number(MAX_SAFE)}, ...change };
    const draft = { workflow: { id: 'w' } }, storage = 'dgxm.studio.grid.v1';
    const sessionStorage = { setItem() {} };
    const definition = gridState.gridDefinitionKey(draft, fields);
    const reviewGuard = new GridReviewGuard();
    const $state = { snapshot: (value) => structuredClone(value) };
    ${prepareCode}
    return async () => { await prepare(); return { preview, error }; };`,
  )(client, gridState, GridReviewGuard, change);
}

test('an incrementing grid from base 2**53 - 1 records and exports each cell seed exactly', async () => {
  // Validate every cell sends the base seed as StudioClient writes a body.
  const sent: string[] = [];
  const review = {
    request: async (path: string, body: unknown) => {
      sent.push(`${path} ${JSON.stringify(body)}`);
      return { id: 'review-1', digest: 'd', ready: true, cells: [] };
    },
  };
  const { error } = await prepareGrid(review)();
  assert.equal(error, '');
  assert.match(sent[0], /^\/grids\/preview\?summary=true /);
  assert.match(
    sent[0],
    /"seed_policy":"increment","base_seed":9007199254740991,"seed_binding":"8\.seed"/,
  );
  // The companion adds the cell index to the base seed as a Python int: one cell past the base is
  // 2**53, the next 2**53 + 1, which a browser double cannot hold. Each record keeps its digits.
  const seeds = [0n, 1n, 2n].map((index) => String(MAX_SAFE + index));
  const texts: Record<string, string> = {};
  seeds.forEach((seed, index) => (texts[`/jobs/job-${index}`] = record(`job-${index}`, [seed])));
  texts['/grids/grid-1'] =
    `{"id": "grid-1", "cells": [${seeds.map((seed, index) => `{"index": ${index}, "job_id": "job-${index}", "seed": ${seed}}`).join(', ')}]}`;
  const { client } = companion(texts);
  const exported = [];
  for (const index of [0, 1, 2]) {
    const { downloads } = await cell(client, { jobId: `job-${index}` })();
    exported.push(downloads[0][1]);
  }
  exported.forEach((text, index) => {
    const exact = parseExactJSON(text) as {
      submission: { effective: { seeds: Record<string, unknown> } };
    };
    assert.equal(String(exact.submission.effective.seeds['8.seed']), seeds[index]);
  });
  assert.ok(exported[2].includes('9007199254740993') && !exported[2].includes('9007199254740992'));
  const { downloads } = await gridExport(client)({ id: 'grid-1' }, false);
  const cells = (parseExactJSON(downloads[0][1]) as { cells: { seed: unknown }[] }).cells;
  assert.deepEqual(
    cells.map((row) => String(row.seed)),
    seeds,
  );
});

test('the comparison keeps adjacent large seeds apart and names a deep field it compared whole', () => {
  const read = (seed: string) => readExactJSON(`{"seed": ${seed}, "steps": 20}`);
  assert.deepEqual(parameterDifferences(read('9007199254740992'), read('9007199254740993')), [
    { key: 'seed', left: 9007199254740992n, right: 9007199254740993n },
  ]);
  assert.deepEqual(
    parameterDifferences(read(UINT64_MAX), read('18446744073709551614')).map((row) => [
      differenceText(row.left),
      differenceText(row.right),
    ]),
    [[UINT64_MAX, '18446744073709551614']],
  );
  assert.deepEqual(parameterDifferences(read(UINT64_MAX), read(UINT64_MAX)), []);
  // A number the browser already rounded can never be said to match.
  const rounded = JSON.parse(`{"seed": ${UINT64_MAX}}`);
  const [row] = parameterDifferences(rounded, rounded);
  assert.equal(row.key, 'seed');
  assert.match(differenceText(row.left), /above 9007199254740991 that this browser holds rounded/);
  // A difference below the depth bound is one row for the whole nested value, not silence.
  let left: unknown = { v: 1 },
    right: unknown = { v: 2 };
  for (let depth = 0; depth < COMPARE_DEPTH + 1; depth++)
    [left, right] = [{ n: left }, { n: right }];
  const deep = parameterDifferences(left, right);
  assert.equal(deep.length, 1);
  assert.equal(deep[0].whole, true);
  assert.equal(deep[0].key.split('.').length, COMPARE_DEPTH + 1);
  assert.match(differenceText(deep[0].right), /"v": 2/);
  assert.deepEqual(parameterDifferences(left, structuredClone(left)), []);
});

test('the comparison reads exact records and says when it compared a field whole', () => {
  const source = readFileSync(
    new URL('./components/ComparisonDetails.svelte', import.meta.url),
    'utf8',
  );
  assert.match(source, /exactJob\(client, id\)/);
  assert.doesNotMatch(source, /fullJob|JSON\.stringify/);
  assert.match(source, /differenceText\(row\.left\)/);
  assert.match(
    source,
    /Fields nested deeper than \{COMPARE_DEPTH\} levels are compared as one value\./,
  );
  const cellSource = readFileSync(
    new URL('./components/GridCellDetails.svelte', import.meta.url),
    'utf8',
  );
  assert.doesNotMatch(cellSource, /JSON\.stringify|exportJSON/);
  const grids = readFileSync(new URL('./views/Grids.svelte', import.meta.url), 'utf8');
  assert.doesNotMatch(grids, /exportJSON/);
});

test('Validate every cell refuses, by its digits, a value this browser would send rounded', async () => {
  const ready = { axisA: '8.seed', axisB: '', valuesA: '[20, 30]', valuesB: '' };
  const blocker = (change: Record<string, unknown>) =>
    gridState.validateBlocker(true, { ...ready, ...change });
  assert.equal(blocker({ valuesA: '[9007199254740991, 1]' }), '');
  assert.equal(
    blocker({ valuesA: '[9007199254740993]' }),
    'Axis A holds 9007199254740993, a whole number above 9007199254740991 that this browser cannot send exactly. Use a smaller value.',
  );
  assert.equal(
    blocker({ axisB: '8.cfg', valuesB: `[9007199254740992, 9007199254740993, ${UINT64_MAX}]` }),
    `Axis B holds 9007199254740992, 9007199254740993 and ${UINT64_MAX}, whole numbers above 9007199254740991 that this browser cannot send exactly. Use smaller values.`,
  );
  // The Base seed box holds a plain number, so a typed seed above 2**53 - 1 is already rounded.
  const typed = Number('9007199254740993');
  assert.equal(blocker({ seedPolicy: 'increment', baseSeed: Number(MAX_SAFE) }), '');
  assert.equal(
    blocker({ seedPolicy: 'increment', baseSeed: typed }),
    'The base seed is above 9007199254740991, the largest whole number this browser sends exactly. Use a smaller base seed.',
  );
  assert.equal(blocker({ seedPolicy: 'fixed', baseSeed: typed }), '', 'Fixed sends no base seed');
  // The definition shows the line and keeps Validate off.
  const body = await renderComponent('./components/GridDefinition.svelte', {
    client: { request: async () => ({ records: [], next_offset: null }) },
    editable: [
      {
        id: '8.seed',
        type: 'INT',
        node_id: '8',
        label: 'Seed',
        input_name: 'seed',
        resolved: true,
      },
    ],
    ...gridState.defaultGridFields(),
    ...ready,
    valuesA: '[9007199254740993]',
    hasWorkflow: true,
    busy: false,
    blocked: false,
    onvalidate() {},
  });
  const button = body.slice(body.lastIndexOf('<button', body.indexOf('Validate every cell')));
  assert.match(button.slice(0, button.indexOf('>')), /disabled/);
  assert.match(visibleText(body), /Axis A holds 9007199254740993, a whole number above/);
  // Validate every cell itself checks the same, so nothing rounded is ever sent.
  for (const change of [{ axisA: '8.seed', valuesA: '[9007199254740993]' }, { baseSeed: typed }]) {
    const sent: string[] = [];
    const client = { request: async (path: string) => (sent.push(path), {}) };
    const { error, preview } = await prepareGrid(client, change)();
    assert.deepEqual(sent, []);
    assert.equal(preview, null);
    assert.match(error, /this browser (cannot send|sends) exactly/);
  }
});
