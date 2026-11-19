import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyDraft } from './model.ts';
import { defaultGridFields, validateBlocker } from './gridState.ts';
import { experimentDefinition } from './experimentRecords.ts';
import { renderComponent, visibleText } from './renderHarness.js';

const steps = { id: 'root:3:steps', type: 'INT', node_id: '3', label: 'Steps', resolved: true };
const ready = { axisA: steps.id, axisB: '', valuesA: '[20, 30, 40]', valuesB: '' };
const NO_WORKFLOW = 'Load a workflow in Create first. A grid varies the Create draft.';
const NO_AXIS = 'Choose an input for Axis A.';

test('the Validate reason names each missing piece of the definition', () => {
  assert.equal(validateBlocker(false, ready), NO_WORKFLOW);
  assert.equal(validateBlocker(true, { ...ready, axisA: '' }), NO_AXIS);
  const cases: [Partial<typeof ready>, string][] = [
    [{ valuesA: '' }, 'List the values for Axis A, for example [20, 30, 40].'],
    [{ valuesA: '  ' }, 'List the values for Axis A, for example [20, 30, 40].'],
    [
      { valuesA: '[20, 30,' },
      'Axis A values are not valid JSON. Write a list such as [20, 30, 40].',
    ],
    [{ valuesA: '20' }, 'Axis A values must be a list in square brackets, such as [20, 30, 40].'],
    [{ valuesA: '[]' }, 'List at least one value for Axis A.'],
    [{ axisB: 'root:3:cfg' }, 'List the values for Axis B, for example [20, 30, 40].'],
    [
      { axisB: 'root:3:cfg', valuesB: '{"cfg": 4}' },
      'Axis B values must be a list in square brackets, such as [20, 30, 40].',
    ],
    [
      { axisA: 'styles', valuesA: '[["ink"], ["ink" "night"]]' },
      'Axis A values are not valid JSON. Write a list such as [["ink"], ["ink", "night"]].',
    ],
  ];
  for (const [change, reason] of cases)
    assert.equal(validateBlocker(true, { ...ready, ...change }), reason, JSON.stringify(change));
  assert.equal(validateBlocker(true, ready), '');
  assert.equal(validateBlocker(true, { ...ready, valuesB: 'ignored without Axis B' }), '');
  assert.equal(validateBlocker(true, { ...ready, axisB: 'root:3:cfg', valuesB: '[4, 7]' }), '');
});

test('a new grid starts with empty value fields, also for a saved one-axis experiment', () => {
  const fields = defaultGridFields();
  assert.equal(fields.valuesA, '');
  assert.equal(fields.valuesB, '');
  assert.equal(
    validateBlocker(true, { ...fields, axisA: steps.id }),
    validateBlocker(true, { ...ready, valuesA: '' }),
  );
  const restored = experimentDefinition({
    grid_request: { draft: {}, axes: [{ name: steps.id, values: [20, 30] }] },
  });
  assert.equal(restored.valuesA, '[20,30]');
  assert.equal(restored.axisB, '');
  assert.equal(restored.valuesB, '');
});

/** The opening tag of the Validate button and the reason shown after it. */
async function validate(props: Record<string, unknown>) {
  const body = await renderComponent('./components/GridDefinition.svelte', {
    client: { request: async () => ({ records: [], next_offset: null }) },
    editable: [steps],
    ...defaultGridFields(),
    hasWorkflow: true,
    busy: false,
    blocked: false,
    onvalidate() {},
    ...props,
  });
  const end = body.indexOf('Validate every cell');
  assert.ok(end > 0, 'the Validate button renders');
  const start = body.lastIndexOf('<button', end);
  const reason = body.match(/<small id="grid-validate-reason"[^>]*>([^<]*)<\/small>/);
  return { tag: body.slice(start, body.indexOf('>', start) + 1), reason: reason?.[1] ?? '', body };
}

test('the disabled Validate button says why, for each cause', async () => {
  const causes: [Record<string, unknown>, string][] = [
    [{ hasWorkflow: false, ...ready }, NO_WORKFLOW],
    [{}, NO_AXIS],
    [{ axisA: steps.id }, 'List the values for Axis A, for example [20, 30, 40].'],
    [
      { ...ready, valuesA: '[20, 30' },
      'Axis A values are not valid JSON. Write a list such as [20, 30, 40].',
    ],
  ];
  for (const [props, reason] of causes) {
    const shown = await validate(props);
    assert.match(shown.tag, /disabled/, reason);
    assert.match(shown.tag, /aria-describedby="grid-validate-reason"/, reason);
    assert.equal(shown.reason, reason);
  }
  const enabled = await validate(ready);
  assert.doesNotMatch(enabled.tag, /disabled/);
  assert.doesNotMatch(enabled.tag, /aria-describedby/);
  assert.equal(enabled.reason, '');
  // A pending dispatch or recovery holds the button; its own notice explains that state.
  const held = await validate({ ...ready, blocked: true });
  assert.match(held.tag, /disabled/);
  assert.equal(held.reason, '');
});

test('the value fields start empty and show an example that fits the axis', async () => {
  const { body } = await validate({ axisA: steps.id, axisB: 'styles' });
  const areas = [...body.matchAll(/<textarea([^>]*)>([^<]*)<\/textarea>/g)];
  assert.equal(areas.length, 2);
  assert.match(areas[0][1], /placeholder="Example: \[20, 30, 40\]"/);
  assert.match(
    areas[1][1],
    /placeholder="Example: \[\[&quot;ink&quot;\], \[&quot;ink&quot;, &quot;night&quot;\]\]"/,
  );
  for (const area of areas) assert.equal(area[2], '');
  assert.doesNotMatch(body, /\[1,2\]|>\[1\]</);
});

test('the Grids screen opens with a how-to that uses the Create draft', async () => {
  const body = await renderComponent('./views/Grids.svelte', {
    client: { request: async () => ({ items: [], next_cursor: null }) },
    draft: { ...emptyDraft(), workflow: { nodes: [] }, workflowName: 'Portrait' },
    bindings: [steps],
    reuse: async () => {},
    restoreDraft: async () => {},
  });
  const text = visibleText(body);
  // The guide's own example names Axis A too, so place the guide against the form's heading.
  const heading = body.indexOf('<h3>Axis A</h3>');
  assert.ok(heading > 0, 'the form names Axis A in its heading');
  const form = visibleText(body.slice(0, heading)).length;
  const [what, how, first] = ['What a grid does', 'How to run a grid', 'Your first comparison'].map(
    (heading) => text.indexOf(heading),
  );
  assert.ok(
    0 < what && what < how && how < first && first < form,
    'the guide comes before the axes',
  );
  for (const step of [
    'Load a workflow in Create and set its inputs there. Each cell is that Create draft with the axis values filled in.',
    'Pick an input for Axis A and list its values as a JSON array, for example steps 20, 30, 40 as [20, 30, 40].',
    'Axis B is optional and multiplies the cells: three values on Axis A and two on Axis B make six cells.',
    'Press Validate every cell, tick the review box, then press Submit reviewed grid.',
    'When a cell finishes, press Inspect full submission to list its outputs, then View one or Compare it with the output in view.',
  ])
    assert.ok(text.includes(step), step);
  assert.match(body, /<ol>(?:<!--[^>]*-->|\s)*<li>/);
  assert.ok(text.includes(NO_AXIS), 'a loaded workflow with no axis asks for Axis A');
});

test('the Grids guide works one example through and walks a first comparison', async () => {
  const text = visibleText(await renderComponent('./components/GridGuide.svelte', {}));
  for (const line of [
    'For example, steps [20, 30, 40] on Axis A and cfg [3.5, 5] on Axis B make 3 × 2 = 6 cells with the same prompt and seed.',
    // Axis A is the outer loop of the backend's itertools.product, so Axis B changes first.
    'Cell 1 is 20 steps at CFG 3.5, cell 2 is 20 steps at CFG 5, cell 3 is 30 steps at CFG 3.5, and so on to cell 6, 40 steps at CFG 5.',
    'if 30 and 40 steps look alike, 30 steps is enough for this prompt.',
    'Your first comparison: 20 steps against 40',
    'Here, choose steps for Axis A and write [20, 40].',
    'Leave Axis B on No second axis and Seed policy on Fixed draft seeds, so both cells use the seed shown in Create.',
    'Press Validate every cell. The preview reads 2 cells · 2 jobs.',
    'press Inspect full submission on cell 1 and View its output. Then press Inspect full submission on cell 2 and Compare on its output.',
    'Only the steps changed, so any difference in the pictures comes from the steps.',
  ])
    assert.ok(text.includes(line), line);
  const ordered = ['What a grid does', 'How to run a grid', 'Your first comparison'];
  const places = ordered.map((heading) => text.indexOf(heading));
  assert.deepEqual(
    [...places].sort((a, b) => a - b),
    places,
    'what, how, then the first run',
  );
});
