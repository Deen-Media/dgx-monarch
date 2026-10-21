import test from 'node:test';
import assert from 'node:assert/strict';
import { initialWorkspace } from './initialWorkspace.ts';
import { emptyDraft } from './model.ts';

function setup(draft: string | null, layout: string | null, pending: string | null) {
  globalThis.localStorage = {
    getItem: (key: string) => (key.includes('draft') ? draft : layout),
  } as Storage;
  globalThis.sessionStorage = { getItem: () => pending } as unknown as Storage;
}
const identity = { key: 'request-123', preview_id: 'preview-456' };
for (const damaged of ['draft', 'layout', 'both']) {
  test(`valid pending request survives malformed ${damaged} state`, () => {
    setup(
      damaged !== 'layout' ? '{' : JSON.stringify({ draft: emptyDraft() }),
      damaged !== 'draft' ? '{' : JSON.stringify({ width: 400 }),
      JSON.stringify(identity),
    );
    const restored = initialWorkspace();
    assert.deepEqual(restored.pending, identity);
    assert.equal(restored.pendingError, '');
    if (damaged !== 'layout') {
      // The unreadable draft text comes back for retention and export.
      assert.equal(restored.draft, null);
      assert.equal(restored.saved, '{');
    }
    if (damaged !== 'draft') assert.match(restored.error, /layout could not be restored/);
    else assert.equal(restored.error, '');
  });
}
test('a draft that fails restore returns its exact stored text', () => {
  const stored = JSON.stringify({ draft: { lora_insertion: { loader: 5 } } });
  setup(stored, null, null);
  const restored = initialWorkspace();
  assert.equal(restored.draft, null);
  assert.equal(restored.saved, stored);
});
test('optional storage denial does not lose the pending request', () => {
  setup(null, null, JSON.stringify(identity));
  localStorage.getItem = () => {
    throw new Error('denied');
  };
  const restored = initialWorkspace();
  assert.deepEqual(restored.pending, identity);
  assert.equal(restored.saved, '');
  assert.match(restored.error, /storage is unavailable/);
});
for (const pending of ['{', '{}', 'false', '{"key":"","preview_id":"preview"}']) {
  test(`unreadable pending state is quarantined: ${pending}`, () => {
    setup(null, null, pending);
    const restored = initialWorkspace();
    assert.equal(restored.pending, null);
    assert.match(restored.pendingError, /Check Queue/);
    assert.equal(sessionStorage.getItem('dgxm.studio.pending'), pending);
  });
}
test('absent pending state is ordinary first-run state', () => {
  setup(null, null, null);
  assert.equal(initialWorkspace().pendingError, '');
});
