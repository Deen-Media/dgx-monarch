import test from 'node:test';
import assert from 'node:assert/strict';
import { appendTelemetry, chartPath, type MetricDefinition } from './telemetryHistory.ts';
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
