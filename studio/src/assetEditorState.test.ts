import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { compileModule } from 'svelte/compiler';

// Compile the actual rune module, as connection.test.ts does.
async function stateModule() {
  const source = await readFile(new URL('./assetEditorState.svelte.ts', import.meta.url), 'utf8');
  const plain = ts.transpile(source, {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
  });
  const compiled = compileModule(plain, {
    filename: 'assetEditorState.svelte.js',
    generate: 'client',
  }).js.code;
  const resolved = compiled.replace(/from ['"]([^'"]+)['"]/g, (_match, specifier) => {
    const url = specifier.startsWith('.')
      ? new URL(`${specifier}.ts`, import.meta.url).href
      : import.meta.resolve(specifier);
    return `from ${JSON.stringify(url)}`;
  });
  return import(`data:text/javascript;base64,${Buffer.from(resolved).toString('base64')}`);
}
const { AssetEditorState, maskCanvas, assetEditor } = await stateModule();

/** A stand-in canvas whose 2D context stores whole-canvas pixel arrays. */
function canvas(width: number, height: number) {
  let pixels = { width, height, data: new Uint8ClampedArray(width * height * 4) };
  return {
    width,
    height,
    getContext: () => ({
      getImageData: () => ({ ...pixels, data: new Uint8ClampedArray(pixels.data) }),
      putImageData: (next: typeof pixels) => {
        pixels = { ...next, data: new Uint8ClampedArray(next.data) };
      },
    }),
    paint(index: number) {
      pixels.data[index * 4 + 3] = 255;
    },
    painted: () => [...pixels.data].filter((_, i) => i % 4 === 3 && pixels.data[i] > 0).length,
  };
}

test('painted mask pixels survive the canvas leaving and coming back', () => {
  const state = new AssetEditorState(() => {});
  state.mode = 'mask';
  state.width = 4;
  state.height = 2;
  const first = canvas(4, 2);
  const action = maskCanvas(first, state);
  first.paint(1);
  first.paint(6);
  // Switching to Input image, another tab or another view removes the canvas.
  action.destroy();
  const second = canvas(4, 2);
  maskCanvas(second, state);
  assert.equal(second.painted(), 2);
});

test('saved paint is not drawn onto a canvas of another size, and clearing forgets it', () => {
  const state = new AssetEditorState(() => {});
  const first = canvas(4, 4);
  first.paint(0);
  state.capture(first);
  const resized = canvas(8, 8);
  state.restore(resized);
  assert.equal(resized.painted(), 0);
  state.capture(first);
  state.clearMask();
  const same = canvas(4, 4);
  state.restore(same);
  assert.equal(same.painted(), 0);
});

test('a new source replaces the old one, releases its URL and invalidates a receipt', () => {
  const revoked: string[] = [];
  const state = new AssetEditorState((url: string) => revoked.push(url));
  state.target = 'node:image';
  state.mode = 'input';
  const before = state.generation;
  state.useSource('blob:first', { name: 'first.png' }, 640, 480);
  state.receipt = { id: 'asset', width: 640, height: 480, binding_value: 'studio/first.png' };
  state.useSource('blob:second', { name: 'second.png' }, 800, 600);
  assert.deepEqual(revoked, ['blob:first']);
  assert.equal(state.source, 'blob:second');
  assert.equal(state.name, 'second.png');
  assert.equal(state.width, 800);
  assert.equal(state.receipt, null);
  assert.ok(state.generation > before);
  // The chosen target and mode are not part of the source.
  assert.equal(state.target, 'node:image');
  assert.equal(state.mode, 'input');
  state.reset();
  assert.deepEqual(revoked, ['blob:first', 'blob:second']);
  assert.equal(state.source, '');
  assert.equal(state.file, undefined);
});

test('the shared editor state is one object per tab', () => {
  assert.ok(assetEditor instanceof AssetEditorState);
  assert.equal(assetEditor.mode, 'input');
});
