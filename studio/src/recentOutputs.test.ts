import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import type { Artifact } from './contracts.ts';
import { recentStep, type StepKey } from './recentOutputs.ts';
import { componentFunctions, renderComponent } from './renderHarness.js';

const key = (value: string, changes: Partial<StepKey> = {}): StepKey => ({
  key: value,
  altKey: false,
  ctrlKey: false,
  metaKey: false,
  shiftKey: false,
  isComposing: false,
  ...changes,
});
const output = (id: string) =>
  ({ id, job_id: 'j', kind: 'image', filename: `${id}.png`, created_at: 1 }) as Artifact;

test('Left and Right step along the strip and stop at its ends', () => {
  assert.equal(recentStep(key('ArrowRight'), 0, 3), 1);
  assert.equal(recentStep(key('ArrowRight'), 1, 3), 2);
  assert.equal(recentStep(key('ArrowLeft'), 2, 3), 1);
  assert.equal(recentStep(key('ArrowLeft'), 0, 3), null, 'the first output has no previous one');
  assert.equal(recentStep(key('ArrowRight'), 2, 3), null, 'the last output has no next one');
  for (const other of ['ArrowUp', 'ArrowDown', 'Enter', ' ', 'Home', 'a'])
    assert.equal(recentStep(key(other), 1, 3), null, other);
});

test('a held modifier or text being composed moves nothing', () => {
  for (const held of ['altKey', 'ctrlKey', 'metaKey', 'shiftKey', 'isComposing'] as const)
    assert.equal(recentStep(key('ArrowRight', { [held]: true }), 0, 3), null, held);
});

// WorkspaceOutput.svelte's handler, run as shipped against a strip of fake tiles.
const stepCode = await componentFunctions('./views/WorkspaceOutput.svelte', ['stepRecent']);
function strip(outputs: Artifact[]) {
  const selected: string[] = [],
    focused: number[] = [];
  const tiles = outputs.map((_, index) => ({ focus: () => focused.push(index) }));
  const step = new Function(
    'outputs',
    'onselect',
    'recentStep',
    `'use strict'; ${stepCode} return stepRecent;`,
  )(outputs, (artifact: Artifact) => selected.push(artifact.id), recentStep);
  const press = (value: string, index: number, changes: Partial<StepKey> = {}) => {
    let prevented = false;
    step(
      {
        ...key(value, changes),
        preventDefault: () => (prevented = true),
        currentTarget: { parentElement: { children: tiles } },
      },
      index,
    );
    return prevented;
  };
  return { press, selected, focused };
}

test('an arrow on a tile selects the next output, updates the viewer and moves focus', () => {
  const { press, selected, focused } = strip([
    output('newest'),
    output('middle'),
    output('oldest'),
  ]);
  assert.equal(press('ArrowRight', 0), true, 'the strip does not scroll as well');
  assert.equal(press('ArrowRight', 1), true);
  assert.equal(press('ArrowLeft', 2), true);
  assert.deepEqual(selected, ['middle', 'oldest', 'middle']);
  assert.deepEqual(focused, [1, 2, 1]);
  // At an end, with a modifier, or for another key, the browser keeps the key.
  assert.equal(press('ArrowLeft', 0), false);
  assert.equal(press('ArrowRight', 1, { altKey: true }), false);
  assert.equal(press('Enter', 1), false);
  assert.deepEqual(selected, ['middle', 'oldest', 'middle']);
});

test('only the strip tiles listen for the arrows and the selected tile says it is pressed', async () => {
  const source = await readFile(new URL('./views/WorkspaceOutput.svelte', import.meta.url), 'utf8');
  assert.equal((source.match(/onkeydown=/g) ?? []).length, 1);
  assert.match(
    source,
    /onclick=\{\(\) => onselect\(artifact\)\}\s+onkeydown=\{\(event\) => stepRecent\(event, index\)\}/,
  );
  const app = await readFile(new URL('./App.svelte', import.meta.url), 'utf8');
  assert.doesNotMatch(app, /Arrow(Left|Right)/, 'no page-wide arrow shortcut');
  const body = await renderComponent('./views/WorkspaceOutput.svelte', {
    client: { request: async () => ({}), thumbnail: async () => new Blob() },
    selected: output('middle'),
    comparison: null,
    artifacts: [output('newest'), output('middle'), output('oldest')],
    telemetry: {},
    history: [],
    connectionState: 'connected',
    activeCount: 0,
    showTelemetry: false,
    jobs: [],
    onuseseed: () => {},
    onselect: () => {},
    oncompare: () => {},
    navigate: () => {},
    reuse: async () => {},
  });
  const pressed = [...body.matchAll(/aria-pressed="(true|false)" title="([^"]+)"/g)].map(
    ([, state, title]) => `${title}:${state}`,
  );
  assert.deepEqual(pressed, ['newest.png:false', 'middle.png:true', 'oldest.png:false']);
});
