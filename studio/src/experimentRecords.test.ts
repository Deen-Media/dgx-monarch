import test from 'node:test';
import assert from 'node:assert/strict';
import { experimentDefinition } from './experimentRecords.ts';

test('restoring an experiment uses its frozen definition and never its job outcomes as draft state', () => {
  const data = {
    grid_id: 'unavailable',
    grid_snapshot: { cells: [{ job_id: 'old-job', state: 'succeeded' }] },
    grid_request: {
      draft: {
        workflow: { '1': { class_type: 'Render', inputs: { seed: 7 } } },
        values: {},
        styles: [],
        lora_stack: [],
      },
      axes: [
        { name: 'styles', values: [[], ['paint']] },
        { name: 'root:1:seed', values: [7, 8] },
      ],
      seed_policy: 'fixed',
      excluded: [1, 3],
    },
  };
  const before = JSON.stringify(data);
  const restored = experimentDefinition(data);
  assert.equal(restored.axisA, 'styles');
  assert.equal(restored.valuesA, '[[],["paint"]]');
  assert.equal(restored.axisB, 'root:1:seed');
  assert.equal(restored.excluded, '2,4');
  assert.equal(restored.draft.seed_policy, 'fixed');
  assert.equal('job_id' in restored.draft, false);
  assert.equal(JSON.stringify(data), before);
});

test('an experiment definition that cannot fit the editor is refused without dropping axes', () => {
  assert.throws(
    () =>
      experimentDefinition({
        grid_request: {
          axes: [
            { name: 'a', values: [1] },
            { name: 'b', values: [2] },
            { name: 'c', values: [3] },
          ],
        },
      }),
    /one or two/,
  );
  assert.throws(() => experimentDefinition({ grid_id: 'missing' }), /no saved grid definition/);
});
