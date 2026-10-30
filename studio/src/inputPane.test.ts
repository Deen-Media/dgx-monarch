import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';
import { graph, recorded, widgetBindings } from './workflowFixtures.js';
import { emptyDraft } from './model.ts';
import type { Binding } from './model.ts';
import { fittedHeight } from './textGrowth.ts';
import { footerRoom } from './footerRoom.ts';

const KREA = 'krea-2-text-to-image';
const UUID = '3f2a9c1e-7b4d-4e8a-9c6f-2d1b0a9e8f7c';
const LONG_ID = `artifact${'0123456789abcdef'.repeat(9)}`;
/** Create with the Krea 2 template's prompt boxes, size fields and model choice. */
async function create(
  controlTab: string,
  change: { values?: object; parent_artifact_ids?: string[] } = {},
) {
  const workflow = graph('krea2-t2i');
  const bindings: Binding[] = widgetBindings(workflow, ['CLIPTextEncode', 'EmptyLatentImage']).map(
    (b: Binding) => ({ ...b, prominent: b.input_name !== 'batch_size' }),
  );
  bindings.push({
    id: `${KREA}:3:unet_name`,
    graph_id: KREA,
    node_id: '3',
    input_name: 'unet_name',
    label: 'unet_name',
    prominent: true,
    schema: { type: 'COMBO', choices: ['krea2_turbo_fp8_scaled.safetensors'] },
  });
  const values = {
    ...Object.fromEntries(
      bindings.map((b) => [b.id, b.value ?? 'krea2_turbo_fp8_scaled.safetensors']),
    ),
    ...change.values,
  };
  return renderComponent('./views/Generation.svelte', {
    client: {},
    draft: { ...emptyDraft(), workflow, workflowName: 'dgx-monarch-krea2-t2i', values, ...change },
    bindings,
    rawWorkflow: '',
    controlTab,
    bindingErrors: [],
    catalog: { object_info: recorded },
    connections: [],
    workflows: [],
    workflowId: '',
    requestLimit: 4 * 1024 * 1024,
    dirty: false,
    busy: false,
    pending: false,
    driverState: 'connected',
    changeDraft() {},
    inspect: async () => {},
    fail() {},
    undo() {},
    prepareJob: async () => {},
    rebind: async () => {},
    refreshCatalog: async () => {},
  });
}
/** The opening tag of the element whose id or for attribute names this field. */
const tag = (body: string, kind: string, attribute: string, id: string) =>
  body.match(new RegExp(`<${kind}[^>]*${attribute}="field-${id}"[^>]*>`))?.[0] ?? '';
const RAW_TYPES = /class="field-type">(string|int|float|combo|boolean)\b/g;

test('Inputs names the Krea 2 boxes Prompt and Negative prompt, with no raw type tags', async () => {
  const body = await create('inputs');
  const text = visibleText(body);
  assert.match(text, /^GENERATION Create .* Prompt .* Add negative prompt/);
  assert.doesNotMatch(text, /\btext\b string/);
  assert.deepEqual(body.match(RAW_TYPES), null);
  // The main prompt starts at seven rows and grows; the empty negative prompt waits folded.
  assert.match(
    tag(body, 'textarea', 'id', `${KREA}:5:text`),
    /class="prompt-text"[^>]*rows="7"|rows="7"[^>]*class="prompt-text"/,
  );
  const opener = body.match(/<button[^>]*class="field-opener"[^>]*>/)?.[0] ?? '';
  assert.match(opener, /aria-expanded="false"/);
  assert.match(opener, new RegExp(`aria-controls="field-${KREA}:6:text"`));
  assert.match(tag(body, 'textarea', 'id', `${KREA}:6:text`), /\shidden/);
  // A folded box shows no label: its button names it.
  assert.equal(tag(body, 'label', 'for', `${KREA}:6:text`), '');
  assert.doesNotMatch(visibleText(body), /Prompt .* Negative prompt Add negative prompt/);
  // A field's node shows on hover.
  assert.match(
    tag(body, 'label', 'for', `${KREA}:5:text`),
    /title="CLIPTextEncode · text · node 5"/,
  );
});

test('All inputs keeps the raw type tags and shows the negative prompt box open', async () => {
  const body = await create('advanced');
  const types = [...body.matchAll(RAW_TYPES)].map((match) => match[1]);
  assert.deepEqual([...new Set(types)].sort(), ['combo', 'int', 'string']);
  assert.match(visibleText(body), /Prompt string .* Negative prompt string/);
  assert.doesNotMatch(body, /field-opener/);
  assert.doesNotMatch(tag(body, 'textarea', 'id', `${KREA}:6:text`), /\shidden/);
});

test('a negative prompt that holds text stays open on Inputs', async () => {
  const body = await create('inputs', { values: { [`${KREA}:6:text`]: 'blurry' } });
  assert.doesNotMatch(body, /field-opener/);
  assert.match(visibleText(body), /Negative prompt blurry/);
  assert.doesNotMatch(tag(body, 'textarea', 'id', `${KREA}:6:text`), /\shidden/);
});

test('a long lineage id shows whole, wraps, and has a Copy button that holds it', async () => {
  const body = await create('inputs', { parent_artifact_ids: [UUID, LONG_ID] });
  const text = visibleText(body);
  assert.match(
    text,
    new RegExp(`Input lineage ${UUID} Copy ${LONG_ID} Copy Lineage records ancestry`),
  );
  assert.match(text, /Clear lineage/);
  for (const id of [UUID, LONG_ID]) {
    assert.match(body, new RegExp(`<code[^>]*>${id}</code>`));
    assert.match(
      body,
      new RegExp(`<button[^>]*title="${id}"[^>]*aria-label="Copy lineage id ${id}"`),
    );
  }
  const style = readFileSync(new URL('./components/InputLineage.svelte', import.meta.url), 'utf8');
  assert.match(style, /code \{[^}]*min-width: 0;[^}]*overflow-wrap: anywhere;/);
});

test('a draft without lineage shows no lineage block', async () => {
  assert.doesNotMatch(visibleText(await create('inputs')), /Input lineage|Clear lineage/);
});

test('Copy writes the whole id, and selects it for a manual copy when the clipboard refuses', async () => {
  const code = await componentFunctions('./components/InputLineage.svelte', ['copy']);
  const run = new Function(
    'clipboard',
    `'use strict';
    let status = '';
    const selected = [];
    const shown = ['first code', 'second code'];
    const navigator = { clipboard };
    const getSelection = () => ({ selectAllChildren: (node) => selected.push(node) });
    ${code}
    return async (id, index) => (await copy(id, index), { status, selected });`,
  );
  const written: string[] = [];
  const copied = await run({ writeText: async (text: string) => written.push(text) })(LONG_ID, 1);
  assert.deepEqual(written, [LONG_ID]);
  assert.deepEqual(copied, { status: 'Copied the whole id.', selected: [] });
  const refused = await run({ writeText: async () => Promise.reject(new Error('denied')) })(
    LONG_ID,
    1,
  );
  assert.deepEqual(refused, {
    status: 'Studio could not copy; the id is selected for a manual copy.',
    selected: ['second code'],
  });
});

test('one layout rule keeps the whole Create pane from scrolling sideways', () => {
  const css = readFileSync(new URL('./styles/generation.css', import.meta.url), 'utf8');
  const rule = css.slice(css.indexOf("/* Create's pane never scrolls sideways"));
  assert.match(rule, /\.controls-panel \{\s*overflow-wrap: anywhere;\s*\}/);
  assert.match(rule, /\.controls-panel,\s*\.controls-scroll \{\s*overflow-x: hidden;\s*\}/);
  assert.match(
    rule,
    /\.controls-panel \.input-row \{\s*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/,
  );
  assert.match(rule, /\.controls-panel fieldset \{\s*min-width: 0;/);
  // The hidden attribute wins over any field rule that sets display, such as `.field label`.
  assert.match(css, /\.field \[hidden\] \{\s*display: none;\s*\}/);
});

test('a short window keeps a field list floor and pins the footer, which fields stop above', () => {
  const css = readFileSync(new URL('./styles/generation.css', import.meta.url), 'utf8');
  const rule = css.slice(css.indexOf('/* On a short window the field list keeps'));
  assert.match(rule, /\.controls-scroll \{\s*min-height: min\(280px, 40vh\);/);
  assert.match(rule, /\.controls-panel \{\s*scroll-padding-bottom: var\(--footer-room, 0px\);/);
  assert.match(rule, /\.submission-footer \{\s*position: sticky;\s*bottom: 0;/);
  // The prompt's ceiling stays under the list's floor, so the whole box fits in the list.
  assert.match(rule, /textarea\.prompt-text \{\s*max-height: min\(30vh, 420px\);/);
  assert.match(
    rule,
    /@media \(max-width: 760px\) \{[^}]*html \{\s*scroll-padding-bottom: calc\(80px/,
  );
  const view = readFileSync(new URL('./views/Generation.svelte', import.meta.url), 'utf8');
  assert.match(view, /<div class="submission-footer" use:footerRoom>/);
});

test('the footer action tracks the footer height, brings focused controls clear of it, and cleans up', () => {
  const room: string[] = [];
  let watch = () => {};
  let watching = false;
  const listeners = new Map<string, (event: Event) => void>();
  const pane = {
    style: {
      setProperty: (name: string, value: string) => room.push(`${name}=${value}`),
      removeProperty: (name: string) => room.push(`${name} removed`),
    },
    addEventListener: (type: string, listener: (event: Event) => void) =>
      listeners.set(type, listener),
    removeEventListener: (type: string) => listeners.delete(type),
  };
  const box = (top: number, bottom: number) => () => ({ top, bottom });
  const inside = {
    getBoundingClientRect: box(730, 760),
    scrollIntoView: () => assert.fail('a footer control stays where it is'),
  };
  const footer = {
    parentElement: pane,
    offsetHeight: 151,
    contains: (n: unknown) => n === inside,
    getBoundingClientRect: box(717, 868),
  };
  const original = globalThis.ResizeObserver;
  const frame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = ((callback: () => void) => (callback(), 0)) as never;
  globalThis.ResizeObserver = class {
    constructor(callback: () => void) {
      watch = callback;
    }
    observe() {
      watching = true;
    }
    unobserve() {}
    disconnect() {
      watching = false;
    }
  } as unknown as typeof ResizeObserver;
  try {
    const action = footerRoom(footer as unknown as HTMLElement);
    footer.offsetHeight = 216;
    watch();
    // Once a focus settles, a field left under the footer scrolls clear of it; a field clear of
    // the footer and a control inside the footer stay put.
    const scrolled: string[] = [];
    const field = (name: string, top: number, bottom: number) => ({
      getBoundingClientRect: box(top, bottom),
      scrollIntoView: (options: { block: string }) => scrolled.push(`${name} ${options.block}`),
    });
    for (const target of [field('under', 660, 723), field('clear', 559, 596), inside])
      listeners.get('focusin')!({ target } as unknown as Event);
    assert.deepEqual(scrolled, ['under nearest']);
    action.destroy?.();
    assert.deepEqual(room, ['--footer-room=151px', '--footer-room=216px', '--footer-room removed']);
    assert.equal(watching, false);
    assert.equal(listeners.size, 0);
  } finally {
    globalThis.ResizeObserver = original;
    globalThis.requestAnimationFrame = frame;
  }
});

test('the main prompt fits its text between its empty height and its ceiling', () => {
  // A short prompt keeps the seven-row floor; a long one grows; a very long one stops and scrolls.
  assert.equal(fittedHeight(120, 2, 170, 420), 170);
  assert.equal(fittedHeight(300, 2, 170, 420), 302);
  assert.equal(fittedHeight(900, 2, 170, 420), 420);
});
