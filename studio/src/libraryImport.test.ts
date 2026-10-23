import test from 'node:test';
import assert from 'node:assert/strict';
import { ApiError } from './api.ts';
import { requireLibraryReceipt } from './libraryImport.ts';
import { submissionRefused } from './submissionRefusal.ts';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

// Execute the component's actual apply handler with storage denied, without a DOM.
const component = await readFile(
  new URL('./components/LibraryTransfer.svelte', import.meta.url),
  'utf8',
);
const script = component.slice(component.indexOf('>') + 1, component.indexOf('</script>'));
const ast = ts.createSourceFile('LibraryTransfer.ts', script, ts.ScriptTarget.Latest, true);
const applyNode = ast.statements.find(
  (node) => ts.isFunctionDeclaration(node) && node.name?.text === 'apply',
);
assert.ok(applyNode);
const applyCode = ts.transpile(applyNode.getText(ast), { target: ts.ScriptTarget.ESNext });
const runApply = new Function(
  'initialUncertain',
  'storageAPI',
  // Strict mode makes a handler that names a variable this harness does not declare fail loudly.
  `'use strict';
  return (async () => {
    let recoveryBlocked = false, plan = { id: 'plan', digest: 'digest', ready: true },
      confirmed = true, busy = false, reading = false, error = '', uncertain = initialUncertain,
      key = 'original-key', name = 'fixture', sent = 0;
    const storage = 'pending', sessionStorage = storageAPI,
      $state = { snapshot: value => value },
      client = { request: async () => { sent++; throw new Error('Unexpected send'); } };
    ${applyCode}
    await apply();
    return { uncertain, key, sent, busy, error };
  })();`,
);
const sourceHandlers = ['read', 'review']
  .map((name) => {
    const node = ast.statements.find(
      (node) => ts.isFunctionDeclaration(node) && node.name?.text === name,
    );
    assert.ok(node);
    return ts.transpile(node.getText(ast), { target: ts.ScriptTarget.ESNext });
  })
  .join('\n');
const sourceHarness = new Function(
  'client',
  `'use strict';
  let recoveryBlocked = false, uncertain = false, busy = false, reading = false,
    sourceGeneration = 0, document = { name: 'old source' }, name = 'old.json',
    plan = { id: 'old-plan' }, choices = {}, confirmed = false, error = '',
    key = 'old-key', mode = 'copy';
  const $state = { snapshot: value => value };
  ${sourceHandlers}
  return { read, review, state: () => ({ reading, plan, document, name, key, busy }) };`,
);

test('library reading blocks review of the previous source until the selected file is ready', async () => {
  let finish!: (value: string) => void;
  const contents = new Promise<string>((resolve) => {
    finish = resolve;
  });
  let sent = 0;
  const controller = sourceHarness({
    request: async () => {
      sent++;
      return {};
    },
  });
  const loading = controller.read({ name: 'new.json', size: 100, text: () => contents });
  await controller.review();
  assert.equal(sent, 0);
  assert.equal(controller.state().reading, true);
  finish('{"name":"new source"}');
  await loading;
  assert.equal(controller.state().reading, false);
  assert.equal(controller.state().plan, null);
  assert.equal(controller.state().name, 'new.json');
});

test('a delayed library review cannot install its plan after the source has changed', async () => {
  let finish!: (value: unknown) => void;
  const response = new Promise<unknown>((resolve) => {
    finish = resolve;
  });
  const controller = sourceHarness({ request: () => response });
  const reviewing = controller.review();
  await controller.read({ name: 'new.json', size: 100, text: async () => '{"name":"new source"}' });
  finish({ id: 'old-source-plan' });
  await reviewing;
  assert.equal(controller.state().plan, null);
  assert.equal(controller.state().key, 'old-key');
  assert.deepEqual(controller.state().document, { name: 'new source' });
  assert.equal(controller.state().busy, false);
});

for (const pending of [false, true]) {
  test(`import storage failure preserves prior uncertainty (${pending}) without sending`, async () => {
    const state = await runApply(pending, {
      setItem: () => {
        throw new Error('Storage denied');
      },
    });
    assert.equal(state.uncertain, pending);
    assert.equal(state.key, 'original-key');
    assert.equal(state.sent, 0);
    assert.equal(state.busy, false);
    assert.match(state.error, /not sent/);
  });
}

test('a lost import acknowledgement followed by 401 never releases the retained copy identity', () => {
  const request = { idempotency_key: 'copy-request', preview_id: 'copy-plan' };
  let retained: typeof request | null = request;
  for (const failure of [new Error('response lost'), new ApiError(401, 'Unauthorized')]) {
    if (submissionRefused(failure, 'library')) retained = null;
    assert.deepEqual(retained, request);
  }
});
for (const failure of [
  new ApiError(409, 'Conflict', 'idempotency_conflict'),
  new ApiError(403, 'Forbidden'),
  new ApiError(409, 'Unknown conflict'),
  new ApiError(500, 'Failed'),
]) {
  test(`library ${failure.message} does not establish that import was unaccepted`, () => {
    assert.equal(submissionRefused(failure, 'library'), false);
  });
}
for (const [code, status] of [
  ['library_confirmation', 409],
  ['library_preview', 409],
  ['revision_conflict', 409],
  ['record_limit', 409],
  ['library_limit', 409],
  ['record_window_limit', 413],
] as const) {
  test(`${code} allows import re-review after definitive refusal`, () => {
    assert.equal(submissionRefused(new ApiError(status, 'Not imported', code), 'library'), true);
  });
}
// Run the same handler against a companion that answers the import with `failure`.
const runRefused = new Function(
  'failure',
  'submissionRefused',
  `'use strict';
  return (async () => {
    let recoveryBlocked = false, plan = { id: 'plan', digest: 'digest', ready: true },
      confirmed = true, busy = false, reading = false, error = '', uncertain = false,
      key = 'original-key', name = 'fixture', notice = '', document = null, sent = 0;
    const kept = new Map(), storage = 'pending', $state = { snapshot: value => value },
      sessionStorage = { setItem: (k, v) => kept.set(k, v), removeItem: (k) => kept.delete(k) },
      requireLibraryReceipt = () => {}, onrefresh = async () => {},
      client = { request: async () => { sent++; throw failure; } };
    ${applyCode}
    await apply();
    return { uncertain, sent, busy, error, retained: kept.has(storage) };
  })();`,
);
test('a catalog that grew past the browser window after review clears the pending import', async () => {
  const refusal = new ApiError(413, 'No records were imported.', 'record_window_limit');
  const state = await runRefused(refusal, submissionRefused);
  assert.deepEqual(
    { uncertain: state.uncertain, sent: state.sent, busy: state.busy, retained: state.retained },
    { uncertain: false, sent: 1, busy: false, retained: false },
  );
  assert.match(state.error, /Review the import again before retrying\./);
  // A lost answer still keeps the retained identity for a retry.
  const lost = await runRefused(new Error('response lost'), submissionRefused);
  assert.equal(lost.uncertain, true);
  assert.equal(lost.retained, true);
  assert.match(lost.error, /Retry uses the same import identity\./);
});
for (const value of [
  {},
  null,
  { id: 'receipt' },
  { id: 'receipt', imported: 2 },
  { imported: 2, repair_list: [] },
  { id: 'receipt', imported: -1, repair_list: [] },
]) {
  test(`partial import acknowledgement cannot settle pending state: ${JSON.stringify(value)}`, () => {
    assert.throws(() => requireLibraryReceipt(value), /retained request/);
  });
}
test('a complete import receipt can settle the retained request', () => {
  requireLibraryReceipt({ id: 'receipt', imported: 2, repair_list: [] });
});
