import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { compile, compileModule } from 'svelte/compiler';

// Server-render Create without a DOM, as componentRender.test.ts does: .svelte files and rune
// modules compile on import, and extensionless relative imports resolve to .ts.
registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith('.') && context.parentURL?.startsWith('file:'))
      for (const candidate of [specifier, `${specifier}.ts`]) {
        const url = new URL(candidate, context.parentURL);
        if (existsSync(fileURLToPath(url))) return { url: url.href, shortCircuit: true };
      }
    return next(specifier, context);
  },
  load(url, context, next) {
    const path = url.startsWith('file:') ? fileURLToPath(url) : '';
    const source = path && readFileSync(path, 'utf8');
    if (path.endsWith('.svelte'))
      return {
        format: 'module',
        source: compile(source, { filename: path, generate: 'server' }).js.code,
        shortCircuit: true,
      };
    if (path.endsWith('.svelte.ts')) {
      const plain = ts.transpile(source, {
        target: ts.ScriptTarget.ESNext,
        module: ts.ModuleKind.ESNext,
      });
      return {
        format: 'module',
        source: compileModule(plain, { filename: path, generate: 'server' }).js.code,
        shortCircuit: true,
      };
    }
    return next(url, context);
  },
});
globalThis.location = { host: '127.0.0.1:8765' } as Location;
const { render } = await import('svelte/server');
const Generation = (await import('./views/Generation.svelte')).default;
const { emptyDraft } = await import('./model.ts');

// An editing graph: Load Image feeds the edit node, whose image Save Image keeps.
const workflow = {
  '1': { class_type: 'LoadImage', inputs: { image: 'reference.png' } },
  '2': { class_type: 'StudioCPUFixtureEdit', inputs: { image: ['1', 0], prompt: 'brighter' } },
  '3': { class_type: 'SaveImage', inputs: { images: ['2', 0] } },
};
const info = {
  LoadImage: {
    input: { required: { image: [['example.png'], { image_upload: true }] } },
    output: ['IMAGE', 'MASK'],
  },
};
const image = {
  id: 'root:1:image',
  graph_id: 'root',
  node_id: '1',
  input_name: 'image',
  label: 'image',
  prominent: true,
  schema: { type: 'COMBO', choices: ['example.png'], image_upload: true },
};
const prompt = { id: 'root:2:prompt', node_id: '2', input_name: 'prompt', type: 'STRING' };

function body(value: unknown, change: Record<string, unknown> = {}, choices = ['example.png']) {
  const props = {
    client: {},
    draft: {
      ...emptyDraft(),
      workflow,
      workflowName: 'Edit',
      operation: 'edit',
      values: { 'root:1:image': value, 'root:2:prompt': 'brighter' },
      ...change,
    },
    bindings: [{ ...image, schema: { ...image.schema, choices } }, prompt],
    rawWorkflow: '',
    controlTab: 'inputs',
    bindingErrors: [],
    catalog: { object_info: info },
    connections: [],
    workflows: [],
    workflowId: '',
    requestLimit: 4 * 1024 * 1024,
    dirty: true,
    busy: false,
    pending: false,
    driverState: 'connected',
    changeDraft() {},
    inspect: async () => {},
    fail() {},
    undo() {},
    prepareJob: async () => {},
  };
  return render(Generation as never, { props }).body;
}

/** The opening tags of the footer's Review settings and Generate buttons. */
function buttons(html: string) {
  return ['Review settings</button>', 'Generate <span>'].map((label) => {
    const start = html.lastIndexOf('<button', html.indexOf(label));
    return html.slice(start, html.indexOf('>', start) + 1);
  });
}

const REASON =
  'Edit needs an image in the image input on node 1. Use Send to edit on an output, or upload an image on the Assets tab.';

test('under Edit, Generate and Review settings stay off with the reason while the image input is empty', () => {
  const empty = body('');
  for (const tag of buttons(empty)) assert.match(tag, /disabled/);
  assert.ok(empty.includes(`<small class="warning">${REASON}</small>`));
  // A driver whose input store is empty lists no choice, so no field check catches the gap.
  const none = body('', {}, []);
  for (const tag of buttons(none)) assert.match(tag, /disabled/);
  assert.ok(none.includes(REASON));
});

test('under Edit, an image the driver offers turns Generate on, and other operations keep their rules', () => {
  const ready = body('example.png');
  for (const tag of buttons(ready)) assert.doesNotMatch(tag, /disabled/);
  assert.ok(!ready.includes(REASON));
  // Under Image the empty input is a field issue as before, without the Edit reason.
  const image = body('', { operation: 'image' });
  for (const tag of buttons(image)) assert.match(tag, /disabled/);
  assert.ok(!image.includes(REASON));
  // The footer's issue summary (from the validation flow) names the field instead of counting it.
  assert.ok(image.includes('id="generate-issues"'));
  assert.match(image, /issue-line warning/);
});
