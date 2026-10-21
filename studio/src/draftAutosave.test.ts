import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { compileModule } from 'svelte/compiler';
import { emptyDraft, restoreDraft } from './model.ts';
import { initialWorkspace } from './initialWorkspace.ts';

// Compile the actual rune module, as connection.test.ts does.
async function autosaveModule() {
  const source = await readFile(new URL('./draftAutosave.svelte.ts', import.meta.url), 'utf8');
  const plain = ts.transpile(source, {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
  });
  const compiled = compileModule(plain, {
    filename: 'draftAutosave.svelte.js',
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
const { DraftAutosave, DRAFT_KEY, STORAGE_UNAVAILABLE, draftPayload } = await autosaveModule();

function setup() {
  const store = new Map<string, string>();
  const writes: string[] = [];
  const downloads: { text: string; name: string }[] = [];
  const storage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      writes.push(value);
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  } as Storage;
  const autosave = new DraftAutosave(
    () => storage,
    (text: string, name: string) => downloads.push({ text, name }),
  );
  return { autosave, store, writes, downloads, storage };
}
const draft = (seed: number) => ({ ...emptyDraft(), workflowName: 'Flux', values: { seed } });
const record = (value: unknown, updated_at = '2026-10-20T00:00:00.000Z') =>
  JSON.stringify({ schema_version: '1.0', draft: value, updated_at });

test('a stored draft that fails restore is retained until the user acts', () => {
  const { autosave, store, writes, downloads } = setup();
  // A record an earlier build wrote, which this build's restoreDraft refuses.
  const unreadable = record({ ...draft(1), lora_insertion: { loader: 5 } });
  store.set(DRAFT_KEY, unreadable);
  globalThis.localStorage = { getItem: () => unreadable } as unknown as Storage;
  globalThis.sessionStorage = { getItem: () => null } as unknown as Storage;
  const restored = initialWorkspace();
  assert.equal(restored.draft, null);
  assert.equal(restored.saved, unreadable);
  autosave.restore(restored.saved, restored.draft);
  assert.equal(autosave.retained, true);
  // The first edit must not overwrite the retained record.
  autosave.save(draft(2), true);
  autosave.save(draft(3), true);
  assert.deepEqual(writes, []);
  assert.equal(store.get(DRAFT_KEY), unreadable);
  // Export downloads the raw stored text, not the empty in-memory draft.
  autosave.exportSaved();
  assert.deepEqual(downloads, [{ text: unreadable, name: 'monarch-saved-draft.json' }]);
  assert.equal(autosave.retained, true);
  autosave.resume();
  autosave.save(draft(4), true);
  assert.equal(writes.length, 1);
  assert.deepEqual(JSON.parse(store.get(DRAFT_KEY)!).draft.values, { seed: 4 });
});

test('a full or blocked store gets its own message and export action', () => {
  const { autosave, storage, downloads } = setup();
  autosave.restore('', null);
  assert.equal(autosave.retained, false);
  storage.setItem = () => {
    throw new DOMException('quota', 'QuotaExceededError');
  };
  autosave.save(draft(1), true);
  assert.equal(autosave.storageError, STORAGE_UNAVAILABLE);
  assert.equal(autosave.retained, false);
  autosave.exportDraft(draft(1));
  assert.equal(downloads[0].name, 'monarch-draft.json');
  assert.deepEqual(JSON.parse(downloads[0].text).values, { seed: 1 });
  storage.setItem = () => {};
  autosave.save(draft(2), true);
  assert.equal(autosave.storageError, '');
});

test('the same draft under a newer timestamp and another key order is not a conflict', () => {
  const { autosave } = setup();
  autosave.restore('', null);
  autosave.save(draft(1), true);
  const reordered = Object.fromEntries(Object.entries(draft(1)).reverse());
  autosave.observe(DRAFT_KEY, record(reordered, '2026-10-20T01:00:00.000Z'));
  assert.equal(autosave.conflict, false);
  autosave.observe('dgxm.studio.layout.v1', record(draft(9)));
  assert.equal(autosave.conflict, false);
  autosave.observe(DRAFT_KEY, record(draft(9)));
  assert.equal(autosave.conflict, true);
});

test('a removed draft record is not reported as a newer draft', () => {
  const { autosave, writes } = setup();
  autosave.restore('', null);
  autosave.save(draft(1), true);
  autosave.observe(DRAFT_KEY, null);
  assert.equal(autosave.conflict, false);
  // The next save writes the draft again, although its payload did not change.
  autosave.save(draft(1), true);
  assert.equal(writes.length, 2);
});

test('a save whose payload equals what this tab last saved writes nothing', () => {
  const { autosave, writes } = setup();
  const stored = record(draft(1));
  autosave.restore(stored, restoreDraft(JSON.parse(stored).draft));
  // Connecting re-inspects and rewrites an identical draft; that is not a save.
  autosave.save(draft(1), true);
  assert.deepEqual(writes, []);
  autosave.save(draft(2), true);
  autosave.save(draft(2), true);
  assert.equal(writes.length, 1);
  autosave.save(draft(3), false);
  assert.equal(writes.length, 1);
});

test('loading the newer draft does not echo a write back to the other tab', () => {
  const tabA = setup(),
    tabB = setup();
  tabA.autosave.restore('', null);
  tabB.autosave.restore('', null);
  tabA.autosave.save(draft(1), true);
  tabB.autosave.save(draft(2), true);
  // Both tabs share one browser store.
  tabB.storage.getItem = () => tabA.store.get(DRAFT_KEY) ?? null;
  tabB.autosave.observe(DRAFT_KEY, tabA.store.get(DRAFT_KEY)!);
  assert.equal(tabB.autosave.conflict, true);
  const loaded = tabB.autosave.loadNewer();
  assert.equal(tabB.autosave.conflict, false);
  assert.deepEqual(loaded.values, { seed: 1 });
  // Importing the loaded draft marks tab B dirty; its save is skipped.
  tabB.autosave.save(loaded, true);
  assert.equal(tabB.writes.length, 1);
  // Had tab B rewritten it under a new timestamp, tab A would see the same payload.
  tabA.autosave.observe(DRAFT_KEY, record(loaded, '2026-10-20T02:00:00.000Z'));
  assert.equal(tabA.autosave.conflict, false);
});

test('keeping this tab exports it and makes it the stored draft', () => {
  const { autosave, store, downloads } = setup();
  autosave.restore('', null);
  autosave.save(draft(1), true);
  autosave.observe(DRAFT_KEY, record(draft(5)));
  assert.equal(autosave.conflict, true);
  autosave.save(draft(1), true);
  assert.notEqual(JSON.parse(store.get(DRAFT_KEY)!).draft.values.seed, 5);
  store.set(DRAFT_KEY, record(draft(5)));
  autosave.keep(draft(1));
  assert.equal(downloads[0].name, 'monarch-this-tab.json');
  assert.equal(autosave.conflict, false);
  assert.deepEqual(JSON.parse(store.get(DRAFT_KEY)!).draft.values, { seed: 1 });
});

test('an unreadable newer draft reports why it cannot load', () => {
  const { autosave, store } = setup();
  autosave.restore('', null);
  assert.throws(() => autosave.loadNewer(), /No saved draft/);
  store.set(DRAFT_KEY, '{');
  assert.throws(() => autosave.loadNewer(), /newer draft could not be restored/);
});

test('payload comparison sorts nested keys and keeps arrays in order', () => {
  assert.equal(draftPayload({ b: 1, a: { d: 2, c: [3, 1] } }), '{"a":{"c":[3,1],"d":2},"b":1}');
  assert.notEqual(draftPayload({ a: [1, 2] }), draftPayload({ a: [2, 1] }));
});
