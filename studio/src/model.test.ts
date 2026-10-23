import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  bindingError,
  consumeEvents,
  boundedLog,
  restoreDraft,
  driverStatus,
  NOT_RUNNING,
} from './model.ts';
test('typed limits reject unsafe values without clamping', () => {
  const b = { id: 'steps', type: 'INT', min: 1, max: 100 };
  assert.equal(bindingError(b, 10), null);
  assert.match(bindingError(b, 0)!, /Minimum/);
  assert.match(bindingError(b, 1.2)!, /whole/);
  assert.match(bindingError(b, Infinity)!, /finite/);
  assert.match(bindingError({ ...b, status: 'ambiguous' }, 20)!, /repair/);
});
test('a required text input may be empty unless its own rule forbids it', () => {
  const text = { id: 'negative', type: 'STRING', required: true };
  assert.equal(bindingError(text, ''), null);
  assert.equal(bindingError({ ...text, schema: { type: 'STRING', default: 'x' } }, ''), null);
  assert.equal(bindingError(text, undefined), 'A value is required.');
  assert.equal(
    bindingError({ ...text, validation: { nonEmpty: true } }, ''),
    'A value is required.',
  );
  assert.equal(bindingError({ ...text, validation: { nonEmpty: true } }, 'repo'), null);
  // A choice with no match and a number still refuse empty text by their own rules.
  const choice = { id: 'sampler', type: 'COMBO', required: true, options: ['euler', 'dpmpp_2m'] };
  assert.equal(bindingError(choice, ''), 'Choose an available value.');
  assert.equal(bindingError({ ...choice, options: ['', 'euler'] }, ''), null);
  assert.equal(
    bindingError({ id: 'steps', type: 'INT', required: true }, ''),
    'Enter a finite number.',
  );
});
test('the shipped MOSS-Image workflows leave double_layers empty and pass the field check', async () => {
  const read = async (path: string) =>
    JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
  const { workflows } = await read('../../src/dgx_monarch/studio/data/workflows.json');
  const contract = await read('../../tests/node_input_contract.json');
  const [, type, fallback] = contract.DGXMonarchSkipLayerGuidance.required.find(
    ([name]: [string]) => name === 'double_layers',
  );
  const shipped = workflows.flatMap((row: { id: string; workflow: { nodes: any[] } }) =>
    row.workflow.nodes
      .filter((node) => node.type === 'DGXMonarchSkipLayerGuidance')
      .map((node) => [row.id, node.dgxm_widgets.double_layers]),
  );
  assert.ok(shipped.some(([id]: [string]) => id === 'workflow.dgx-monarch-moss-image-t2i'));
  assert.ok(shipped.some(([, value]: [string, unknown]) => value === ''));
  for (const [id, value] of shipped) {
    const binding = { id, required: true, schema: { type, default: fallback } };
    assert.equal(bindingError(binding, value), null, id);
  }
});
test('delayed and duplicated events cannot regress cursor', () => {
  const result = consumeEvents(4, [{ seq: 7 }, { seq: 2 }, { seq: 5 }, { seq: 7 }, { seq: 'bad' }]);
  assert.equal(result.cursor, 7);
  assert.deepEqual(result.events, [{ seq: 5 }, { seq: 7 }]);
});
test('console retention accounts for Unicode bytes and line count', () => {
  assert.deepEqual(boundedLog(['a', 'b', 'c'], 2), ['b', 'c']);
  assert.deepEqual(boundedLog(['old', '🦋'], 100, 4), ['🦋']);
});
test('restoration preserves unknown metadata without treating malformed stacks as lists', () => {
  const result = restoreDraft({
    values: { seed: 5 },
    extension: { kept: true },
    lora_stack: 'bad',
  });
  assert.equal(result.values.seed, 5);
  assert.deepEqual(result.lora_stack, []);
  assert.deepEqual((result as any).extension, { kept: true });
});
test('malformed nested draft controls are rejected before a render can replace the working draft', () => {
  for (const value of [
    { styles: [null] },
    { styles: ['saved-id'] },
    { lora_stack: [null] },
    { lora_stack: [7] },
    { parent_artifact_ids: [{}] },
    { lora_insertion: { loader: 'LoraLoader', model_targets: 'not-a-list' } },
    { lora_insertion: { loader: 'LoraLoader', clip_targets: [null] } },
    { lora_insertion: { loader: 'LoraLoader', model_targets: [{ node_id: '1' }] } },
    { style_bindings: [] },
    { style_bindings: { prompt: {} } },
    { workflow: [] },
    { workflowName: {} },
    { project_id: 3 },
  ])
    assert.throws(() => restoreDraft(value), /Invalid draft/);
});
test('optional legacy insertion targets default to empty without dropping extensions', () => {
  const value = restoreDraft({
    lora_insertion: {
      loader: 'LoraLoaderModelOnly',
      model_targets: [{ node_id: '1', input_name: 'model' }],
    },
    extension: { retained: true },
  });
  assert.deepEqual(value.lora_insertion?.clip_targets, []);
  assert.deepEqual((value as any).extension, { retained: true });
  assert.deepEqual(restoreDraft({ values: [] }).values, {});
  assert.equal(restoreDraft({ lora_insertion: null }).lora_insertion, undefined);
});
test('an unavailable driver shows its failure code', () => {
  assert.equal(driverStatus({ state: 'connected', code: 'stale' }), 'connected');
  assert.equal(
    driverStatus({ state: 'unavailable', code: 'database_error' }),
    'unavailable (database_error)',
  );
  assert.equal(driverStatus({ state: 'unavailable' }), 'unavailable');
  assert.equal(driverStatus({}), 'unknown');
  assert.equal(driverStatus(undefined), 'unknown');
});
test('a driver origin that refuses connections reads as not running', () => {
  assert.equal(driverStatus({ state: 'unavailable', code: 'driver_not_running' }), 'not running');
  assert.equal(NOT_RUNNING, 'not running');
  assert.equal(driverStatus({ state: 'connected', code: 'driver_not_running' }), 'connected');
});
