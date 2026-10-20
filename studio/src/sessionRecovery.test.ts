import test from 'node:test';
import assert from 'node:assert/strict';
import { restoreSession, validOperation, validGrid, validLibrary } from './sessionRecovery.ts';

const operation = {
  action: 'restart',
  parameters: {},
  preview: {
    id: 'op-1',
    digest: 'digest',
    action: 'restart',
    parameters: {},
    available: true,
    requires_confirmation: true,
  },
  requestKey: 'request-1',
  pending: true,
};
const grid = {
  preview: { id: 'preview-1', digest: 'digest', cells: [] },
  grid: null,
  requestKey: 'request-1',
  retryKey: '',
  retryIndices: [],
  dispatchPending: true,
  retryPending: false,
};
const library = {
  plan: {
    id: 'import-1',
    digest: 'digest',
    ready: true,
    changes: [],
    conflicts: [],
    repair_list: [],
  },
  key: 'request-1',
  name: 'library.json',
};
function storage(raw: string | null) {
  const store = new Map<string, string>();
  if (raw !== null) store.set('pending', raw);
  globalThis.sessionStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  } as Storage;
}
for (const [name, valid, validate] of [
  ['operation', operation, validOperation],
  ['grid', grid, validGrid],
  ['library', library, validLibrary],
] as const) {
  test(`${name} request identity survives valid recovery and corrupt input is quarantined without mutation`, () => {
    storage(JSON.stringify(valid));
    assert.deepEqual(restoreSession('pending', validate as (v: unknown) => v is unknown), {
      value: valid,
      blocked: false,
    });
    for (const raw of ['{', '{}', 'null', 'false', '[]', '{"pending":true}']) {
      storage(raw);
      assert.deepEqual(restoreSession('pending', validate as (v: unknown) => v is unknown), {
        value: null,
        blocked: true,
      });
      assert.equal(sessionStorage.getItem('pending'), raw);
    }
    storage(null);
    assert.deepEqual(restoreSession('pending', validate as (v: unknown) => v is unknown), {
      value: null,
      blocked: false,
    });
  });
}
test('partial or inconsistent stored grid identities cannot silently unlock a duplicate dispatch', () => {
  for (const candidate of [
    { ...grid, requestKey: '' },
    { ...grid, preview: null },
    { ...grid, retryPending: true },
    { ...grid, retryIndices: [-1] },
    { ...grid, grid: { id: 'run-1' } },
    { ...grid, dispatchPending: 'true' },
  ])
    assert.equal(validGrid(candidate), false);
  assert.equal(
    validGrid({
      ...grid,
      grid: { id: 'run-1', cells: [] },
      retryPending: true,
      retryKey: 'retry-1',
      retryIndices: [0, 2],
    }),
    true,
  );
});
test('operation and library recovery require complete retry identities and renderable receipt shapes', () => {
  assert.equal(validOperation({ ...operation, preview: null }), false);
  assert.equal(validOperation({ ...operation, requestKey: '' }), false);
  for (const field of ['action', 'parameters', 'available', 'requires_confirmation']) {
    const preview: Record<string, unknown> = { ...operation.preview };
    delete preview[field];
    assert.equal(validOperation({ ...operation, preview }), false);
  }
  assert.equal(
    validOperation({ ...operation, preview: { ...operation.preview, action: 'another' } }),
    false,
  );
  assert.equal(validLibrary({ ...library, plan: { ...library.plan, conflicts: [{}] } }), false);
  assert.equal(validLibrary({ ...library, plan: { ...library.plan, repair_list: null } }), false);
});
test('storage access denial conservatively blocks mutation recovery', () => {
  storage(null);
  sessionStorage.getItem = () => {
    throw new Error('denied');
  };
  assert.deepEqual(restoreSession('pending', validOperation), { value: null, blocked: true });
});
