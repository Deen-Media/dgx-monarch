import test from 'node:test';
import assert from 'node:assert/strict';
import { gridIdentity } from './gridState.ts';

test('a large grid retains a small durable identity for reload and reconciliation', () => {
  const value = {
    id: 'reviewed-grid',
    digest: 'digest',
    ready: true,
    total: 1024,
    job_count: 1024,
    expires_at: 1234,
    state: 'queued',
    cells: Array.from({ length: 1024 }, (_, index) => ({
      index,
      coordinates: { prompt: 'x'.repeat(10000) },
      style_records: [{ data: { prompt: 'x'.repeat(10000) } }],
    })),
    submission: { workflow: 'x'.repeat(100000) },
  };
  const saved = gridIdentity(value)!;
  assert.ok(JSON.stringify(saved).length < 300);
  assert.equal(saved.id, value.id);
  assert.equal(saved.digest, value.digest);
  assert.equal(saved.ready, true);
  assert.equal(saved.expires_at, value.expires_at);
  assert.equal(gridIdentity(null), null);
});
