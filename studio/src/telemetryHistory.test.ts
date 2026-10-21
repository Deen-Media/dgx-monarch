import test from 'node:test';
import assert from 'node:assert/strict';
import {
  appendTelemetry,
  chartPath,
  slabReading,
  type MetricDefinition,
} from './telemetryHistory.ts';
const metric: MetricDefinition = {
  id: 'host.gpu.power',
  source: 'gpu.power_w',
  unit: 'W',
  name: 'Power',
  aggregation: 'none',
};
test('telemetry preserves nulls, rejects stale/out-of-order sampling and caps retention', () => {
  let history = appendTelemetry(
    [],
    {
      state: 'connected',
      sample: { t: 1, driver_host: { host: 'a', t: 1, gpu: { power_w: 10 } } },
    },
    [metric],
  );
  history = appendTelemetry(
    history,
    { state: 'connected', sample: { t: 2, driver_host: { host: 'a', t: 2 } } },
    [metric],
  );
  assert.deepEqual(history[0].points, [
    { time: 1, value: 10 },
    { time: 2, value: null },
  ]);
  assert.equal(appendTelemetry(history, { state: 'stale', sample: { t: 3 } }, [metric]), history);
  history = appendTelemetry(
    history,
    {
      state: 'connected',
      sample: { t: 3, driver_host: { host: 'a', t: 3, gpu: { power_w: 11 } } },
    },
    [metric],
    2,
  );
  assert.equal(history[0].points.length, 2);
  assert.match(chartPath(history[0].points), /^ M/);
});
test('rail rates use per-host counter delta and reset to unknown', () => {
  const rail = {
    ...metric,
    id: 'host.rail.rx',
    source: 'rails.*.rx_bytes',
    aggregation: 'per_host_per_rail_rate',
  };
  const sample = (t: number, value: number) => ({
    state: 'connected',
    sample: { t, driver_host: { host: 'a', t, rails_t: t, rails: { eth0: { rx_bytes: value } } } },
  });
  let history = appendTelemetry([], sample(1, 1e9), [rail]);
  history = appendTelemetry(history, sample(3, 5e9), [rail]);
  assert.equal(history[0].points.at(-1)?.value, 2);
  history = appendTelemetry(history, sample(4, 1), [rail]);
  assert.equal(history[0].points.at(-1)?.value, null);
});
test('sample times are bounded by the companion clock, not a browser clock behind the host', (t) => {
  const host = 1_792_522_000;
  t.mock.method(Date, 'now', () => (host - 30) * 1000);
  const sample = (time: number, observed?: number) => ({
    state: 'connected',
    ...(observed === undefined ? {} : { observed_at: observed }),
    sample: { t: time, driver_host: { host: 'a', t: time, gpu: { power_w: time - host } } },
  });
  let history = appendTelemetry([], sample(host, host + 0.2), [metric]);
  history = appendTelemetry(history, sample(host + 1.5, host + 1.7), [metric]);
  assert.deepEqual(
    history[0].points.map((point) => point.value),
    [0, 1.5],
    'a browser 30 s behind keeps every sample',
  );
  const future = appendTelemetry(history, sample(host + 3600, host + 3), [metric]);
  assert.equal(future, history, 'a sample far ahead of the companion clock is refused');
  history = appendTelemetry(history, sample(host + 3, host + 3.1), [metric]);
  assert.equal(history[0].points.at(-1)?.value, 3, 'later samples still arrive');
  assert.equal(
    appendTelemetry(history, sample(host + 4), [metric]),
    history,
    'without a companion time the browser clock still bounds samples',
  );
});
test('the slab chart records a known zero unless the Worker report could hide a slab', () => {
  const slab: MetricDefinition = {
    id: 'host.memory.slab',
    source: 'worker.memory.slab_gib',
    unit: 'GiB',
    name: 'Slab',
    aggregation: 'none',
    missing: 'slab_rule',
  };
  const resident = { patcher_class: 'ModelPatcher', model: 'sample-model' };
  const workers = [
    {
      host: { host: 'slab', t: 1 },
      memory: { ...resident, weight_residency: 'slab', slab_gib: 12 },
    },
    { host: { host: 'stock', t: 1 }, memory: { ...resident, weight_residency: 'cudaMalloc' } },
    { host: { host: 'idle', t: 1 }, memory: { safetensors_backend: 'pread' } },
    { host: { host: 'error', t: 1 }, memory_error: 'RuntimeError: memory detail failed' },
    { host: { host: 'silent', t: 1 } },
    { host: { host: 'hidden', t: 1 }, memory: { ...resident, weight_residency: 'slab' } },
    {
      host: { host: 'shared', t: 1 },
      memory: { ...resident, weight_residency: 'slab', slab_gib: 3 },
    },
  ];
  const history = appendTelemetry(
    [],
    {
      state: 'connected',
      sample: { t: 1, driver_host: { host: 'driver', t: 1 }, workers },
    },
    [slab],
  );
  const reading = (host: string) =>
    history.find((series) => series.host === host)?.points.at(-1)?.value;
  assert.equal(reading('slab'), 12);
  assert.equal(reading('stock'), 0);
  assert.equal(reading('idle'), 0);
  assert.equal(reading('driver'), 0);
  assert.equal(reading('error'), null);
  assert.equal(reading('silent'), null);
  assert.equal(reading('hidden'), null);
  // A driver box that also runs a Worker takes that Worker's reading, not a zero.
  const shared = appendTelemetry(
    [],
    {
      state: 'connected',
      sample: { t: 1, driver_host: { host: 'shared', t: 1 }, workers: [workers.at(-1)] },
    },
    [slab],
  );
  assert.deepEqual(shared[0].points, [{ time: 1, value: 3 }]);
  // Without the contract's rule a missing value stays unknown.
  const plain = appendTelemetry(
    [],
    { state: 'connected', sample: { t: 1, workers: [workers[1]] } },
    [{ ...slab, missing: 'unknown' }],
  );
  assert.equal(plain[0].points[0].value, null);
  assert.equal(slabReading({ memory: { slab_gib: -1 } }), 0);
  assert.equal(slabReading({ memory: { slab_gib: true } }), null);
  assert.equal(slabReading({ memory: { weight_residency: 3 } }), null);
});
