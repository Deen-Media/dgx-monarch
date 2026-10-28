import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { emptyDraft } from './model.ts';
import { moveStyle, styleEndNote, styleName } from './promptStyles.ts';
import { recordContent } from './recordContent.ts';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';

const paint = { id: 'paint', name: 'Oil painting', prompt: 'oil painting', negative_prompt: '' };
const wrap = { id: 'wrap', name: 'Emphasis', prompt: '(({prompt}))', negative_prompt: '' };
const frame = { id: 'frame', name: 'Frame', prompt: 'framed {prompt}', negative_prompt: '' };
const prompt = { id: 'root:2:text', type: 'STRING', node_id: '2', label: 'Prompt' };

/** Each rendered button as its opening tag and visible label. */
function buttons(body: string) {
  return [...body.matchAll(/<button([^>]*)>((?:<!--[^>]*-->|[^<])*)<\/button>/g)].map(
    ([, tag, label]) => ({ tag, label: label.replace(/<!--[^>]*-->/g, '').trim() }),
  );
}
const stylesBody = (styles: unknown[], bindings: unknown[] = [prompt]) =>
  renderComponent('./components/PromptStyles.svelte', {
    draft: { ...emptyDraft(), workflow: { nodes: [] }, styles },
    bindings,
    changeDraft() {},
  });

test('a style moves one place and never past either end of the list', () => {
  const styles = [paint, wrap, frame];
  assert.deepEqual(moveStyle(styles, 0, 1), [wrap, paint, frame]);
  assert.deepEqual(moveStyle(styles, 1, 1), [paint, frame, wrap]);
  assert.deepEqual(moveStyle(styles, 2, -1), [paint, frame, wrap]);
  for (const [index, offset] of [
    [0, -1],
    [2, 1],
    [1, 0],
    [3, -1],
    [-1, 1],
    [0.5, 1],
  ])
    assert.equal(moveStyle(styles, index, offset), null, `${index} by ${offset}`);
  assert.deepEqual(styles, [paint, wrap, frame], 'the draft list is never changed in place');
  assert.equal(styleName(paint, 0), 'Oil painting');
  assert.equal(styleName({ id: 'unnamed' }, 1), 'Style 2');
});

test('Create explains the style order rule with the fox example and points to Library', async () => {
  const text = visibleText(await stylesBody([]));
  for (const sentence of [
    'Ordered prompt styles',
    'Styles apply from top to bottom, to the prompt and the negative prompt separately.',
    '{prompt} in a style stands for the text so far; a style without it adds its text at the end, after a comma.',
    'For example, "a red fox" with "oil painting" then "(({prompt}))" gives "((a red fox, oil painting))"; the other order gives "((a red fox)), oil painting".',
    'Make styles in Library under styles, then add one here with Append style.',
    'Submission review shows the effective text.',
  ])
    assert.ok(text.includes(sentence), sentence);
  // Without a text input there is nothing for a style to change.
  assert.equal(visibleText(await stylesBody([paint], [])), '');
});

test('each applied style has Move up, Move down and Remove, and the ends say why a move is off', async () => {
  const body = await stylesBody([paint, wrap, frame]);
  const text = visibleText(body);
  assert.ok(text.includes('1. Oil painting'), text);
  assert.ok(text.includes('3. Frame'), text);
  const rows = buttons(body).filter((button) => /^(Move up|Move down|×)$/.test(button.label));
  assert.deepEqual(
    rows.map(({ label }) => label),
    ['×', 'Move up', 'Move down', '×', 'Move up', 'Move down', '×', 'Move up', 'Move down'],
  );
  const [, firstUp, firstDown, , middleUp, middleDown, , lastUp, lastDown] = rows.map(
    ({ tag }) => tag,
  );
  assert.match(firstUp, /disabled/);
  assert.match(firstUp, /aria-describedby="style-note-0"/);
  assert.match(body, /<small id="style-note-0">Already first<\/small>/);
  for (const tag of [firstDown, middleUp, middleDown, lastUp]) {
    assert.doesNotMatch(tag, /disabled/);
    assert.doesNotMatch(tag, /aria-describedby/);
  }
  assert.match(lastDown, /disabled/);
  assert.match(lastDown, /aria-describedby="style-note-2"/);
  assert.match(body, /<small id="style-note-2">Already last<\/small>/);
  assert.doesNotMatch(body, /id="style-note-1"/, 'a middle style has no note');
  assert.match(rows[0].tag, /aria-label="Remove Oil painting"/);
  const only = await stylesBody([paint]);
  const [, up, down] = buttons(only).map(({ tag }) => tag);
  for (const tag of [up, down]) {
    assert.match(tag, /disabled/);
    assert.match(tag, /aria-describedby="style-note-0"/);
  }
  assert.match(only, /<small id="style-note-0">Only style<\/small>/);
  assert.equal(styleEndNote(1, 3), '');
});

// Run the shipped move handler with the shipped App changeDraft, undo and importDraft, as
// Generation passes App's draft and changeDraft to the styles panel.
const moveCode = await componentFunctions('./components/PromptStyles.svelte', ['move']);
const appCode = await componentFunctions('./App.svelte', [
  'changeDraft',
  'undo',
  'importDraft',
  'setNotice',
]);
const workspace = new Function(
  'initial',
  'moveStyle',
  `'use strict';
  let draft = initial, undoDraft = null, workflowId = '', dirty = false, error = '', notice = '';
  let bindings = [], connections = [], bindingErrors = [], rawWorkflow = '';
  const changes = [], guard = { edit: (change) => changes.push(change) };
  const submission = { preview: { id: 'reviewed' } };
  const $state = { snapshot: (value) => value };
  const inspect = async () => {};
  ${appCode}
  ${moveCode}
  return { move, undo, view: () => ({ draft, dirty, notice, changes, preview: submission.preview }) };`,
);

test('a move is one undoable draft change and a blocked move changes nothing', async () => {
  const app = workspace(
    { ...emptyDraft(), workflow: { nodes: [] }, styles: [paint, wrap] },
    moveStyle,
  );
  app.move(0, -1);
  app.move(1, 1);
  assert.deepEqual(app.view().changes, [], 'a move past an end is no draft change');
  assert.equal(app.view().dirty, false);
  app.move(0, 1);
  assert.deepEqual(app.view().changes, [{ styles: [wrap, paint] }]);
  assert.deepEqual(app.view().draft.styles, [wrap, paint]);
  assert.equal(app.view().dirty, true);
  assert.equal(app.view().preview, null, 'a reordered draft needs a new review');
  await app.undo();
  assert.deepEqual(app.view().draft.styles, [paint, wrap]);
  assert.equal(app.view().notice, 'Draft change undone.');
});

// Run the shipped Library save handler: a style record carries its text and no mode.
const createCode = await componentFunctions('./views/WorkspaceRecords.svelte', ['createRecord']);
const library = new Function(
  'recordContent',
  'editing',
  `'use strict';
  return (async () => {
    let busy = false, recordKind = 'styles', selectedRecord = editing, recordName = 'Ink';
    let recordText = 'ink wash, {prompt}', negativeText = 'blurry', gridId = '', membership = [];
    const draft = { styles: [] }, selected = null, activeJobs = [], layout = {};
    const $state = { snapshot: (value) => value };
    const saved = [], errors = [];
    const saveRecord = async (kind, data, id, revision) => saved.push({ kind, data, id, revision });
    const session = { id: () => 'style-new', settle() {}, failed() {} };
    const load = async () => {};
    const fail = (cause) => errors.push(String(cause));
    ${createCode}
    await createRecord();
    return { saved, errors };
  })();`,
);

test('Library saves a style without a mode, and an edit keeps the mode an older record carries', async () => {
  const created = await library(recordContent, null);
  assert.deepEqual(created.errors, []);
  assert.deepEqual(created.saved, [
    {
      kind: 'styles',
      id: 'style-new',
      revision: 0,
      data: {
        name: 'Ink',
        project_ids: [],
        prompt: 'ink wash, {prompt}',
        negative_prompt: 'blurry',
      },
    },
  ]);
  const older = { id: 'old', revision: 3, data: { name: 'Old', prompt: 'x', mode: 'template' } };
  const edited = await library(recordContent, older);
  assert.equal(edited.saved[0].data.mode, 'template', 'the record keeps its own content');
  assert.equal(edited.saved[0].data.prompt, 'ink wash, {prompt}');
  const source = await readFile(
    new URL('./views/WorkspaceRecords.svelte', import.meta.url),
    'utf8',
  );
  assert.doesNotMatch(source, /Style mode|styleMode|value="template"/);
  assert.match(
    visibleText(source.slice(source.indexOf("{#if recordKind === 'styles'}"))),
    /\{'\{prompt\}'\} stands for the text so far\. Studio adds text without \{'\{prompt\}'\} at the end, after a comma\./,
  );
});
