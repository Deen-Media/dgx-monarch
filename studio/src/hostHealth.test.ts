import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { hostSummary, staleAfter } from './hostHealth.ts';
import { appendTelemetry, type MetricSeries, type Presentation } from './telemetryHistory.ts';
import {
  RENDER_PROGRESS,
  chartChoice,
  sameMetricOn,
  seriesHost,
  seriesLabel,
} from './chartChoice.ts';
import { renderComponent, visibleText } from './renderHarness.js';

const presentation: Presentation = JSON.parse(
  readFileSync(
    new URL('../../src/dgx_monarch/studio/data/presentation.json', import.meta.url),
    'utf8',
  ),
);
/** One host's readings in the shape telemetry.host_stats() publishes. */
const host = (name: string, t: number, available: number, util: number) => ({
  host: name,
  t,
  mem_gib: { MemTotal: 121.6, MemAvailable: available, MemFree: 9.5, AnonPages: 12.25 },
  gpu: { util_pct: util, clock_mhz: 2400, power_w: 61.42, temp_c: 58 },
  gpu_proc_gib: 32.84,
  pool_gib: 0,
  rails_t: t,
  rails: { rail0: { rx_bytes: 4e9, tx_bytes: 3e9 }, rail1: { rx_bytes: 2e9, tx_bytes: 1e9 } },
});
const readiness = {
  schema_version: 1,
  overall: 'degraded',
  presentation: { schema_version: 1, health: 'ready', activity: 'rendering', maintenance: 'wait' },
  lifecycle: {
    worker_service: { state: 'ready', detail: '2 worker services ready.' },
    attached_mesh: { state: 'active', detail: 'The attached mesh has active work in flight.' },
    render_session: { state: 'active', detail: '1 render session is active.' },
  },
};
/** A pair mid-render: the driver and rank 0 share the first box, rank 1 runs on the second. */
const pair = (t = 1000, observed = t + 0.4) => ({
  state: 'connected',
  observed_at: observed,
  sample: {
    t,
    driver_host: host('spark-a', t, 62.2, 97),
    workers: [
      { rank: 0, world: 2, host: host('spark-a', t - 0.5, 61.9, 96) },
      { rank: 1, world: 2, host: host('spark-b', t - 0.4, 18.4, 95) },
    ],
    render: { active: true, active_renders: 1, step: 12, steps: 30, sec_per_step: 1.5 },
    readiness,
  },
});

test('a pair shows both hosts side by side with the driver and its Worker merged', () => {
  const summary = hostSummary(pair(), presentation);
  assert.deepEqual(
    summary.hosts.map((card) => [card.host, card.roles, card.age]),
    [
      ['spark-a', 'Driver · Worker rank 0 of 2', ''],
      ['spark-b', 'Worker rank 1 of 2', ''],
    ],
  );
  const [first, second] = summary.hosts;
  // The newer of two readings of one box wins: the driver's, here.
  assert.deepEqual(
    first.readings.map((reading) => [reading.name, reading.text]),
    [
      ['Available memory', '62.2 of 121.6 GiB'],
      ['GPU memory', '32.8 GiB'],
      ['GPU utilization', '97 %'],
      ['GPU power', '61.4 W'],
      ['GPU temperature', '58 °C'],
    ],
  );
  // Available memory carries the shared display threshold: above 20 GiB green, else yellow.
  assert.deepEqual(first.readings[0].status, {
    hex: '#4ade80',
    icon: '✓',
    title: 'More than 20 GiB available',
  });
  assert.equal(second.readings[0].text, '18.4 of 121.6 GiB');
  assert.deepEqual(second.readings[0].status, {
    hex: '#facc15',
    icon: '!',
    title: '20 GiB or less available',
  });
  assert.equal(summary.note, '');
  // The readiness rows read as the ComfyUI panel's card does, active in yellow.
  assert.deepEqual(
    summary.readiness.map((row) => [row.label, row.state, row.detail, row.mark.hex]),
    [
      ['Worker service', 'ready', '2 worker services ready.', '#4ade80'],
      ['Attached mesh', 'active', 'The attached mesh has active work in flight.', '#facc15'],
      ['Render session', 'active', '1 render session is active.', '#facc15'],
    ],
  );
});

test('one host says why it is alone, and Workers without readings are counted', () => {
  const { sample, ...rest } = pair();
  const alone = hostSummary({ ...rest, sample: { ...sample, workers: [] } }, presentation);
  assert.equal(alone.hosts.length, 1);
  assert.equal(alone.hosts[0].roles, 'Driver');
  assert.equal(
    alone.note,
    "Only the driver's host reports. Worker hosts join this list while a Monarch render's mesh is attached.",
  );
  const local = hostSummary(
    {
      ...rest,
      sample: { ...sample, workers: [{ rank: 0, world: 1, host: host('spark-a', 999, 60, 90) }] },
    },
    presentation,
  );
  assert.equal(local.hosts[0].roles, 'Driver · Worker rank 0 of 1');
  assert.equal(local.note, 'Every Worker that reports runs on this host.');
  const failed = hostSummary(
    { ...rest, sample: { ...sample, workers: [{ status_error: 'TimeoutError' }] } },
    presentation,
  );
  assert.equal(
    failed.note,
    "One Worker answered without host readings. Only the driver's host reports. Worker hosts join this list while a Monarch render's mesh is attached.",
  );
  // A Worker before setup names its host as text and has no readings yet.
  const early = hostSummary(
    { ...rest, sample: { ...sample, workers: [{ rank: 1, world: 2, host: 'spark-b' }] } },
    presentation,
  );
  assert.deepEqual(
    early.hosts.map((card) => [card.host, card.age]),
    [
      ['spark-a', ''],
      ['spark-b', 'No readings from this host yet.'],
    ],
  );
  assert.ok(early.hosts[1].readings.every((reading) => reading.text === 'unknown'));
  assert.deepEqual(early.hosts[1].readings[0].status, {
    hex: '#f87171',
    icon: '?',
    title: 'Available memory is unknown',
  });
  assert.equal(early.note, '');
});

test('readings older than the freshness rule say so; unknown values stay unknown', () => {
  assert.equal(staleAfter(presentation), 3);
  const late = pair(1000, 1020.4);
  assert.deepEqual(
    hostSummary(late, presentation).hosts.map((card) => card.age),
    ['Stale reading, 20 s old.', 'Stale reading, 21 s old.'],
  );
  // A host clock a little ahead of the companion is not stale; far ahead it cannot say.
  assert.equal(hostSummary(pair(1000, 998), presentation).hosts[0].age, '');
  assert.equal(hostSummary(pair(1000, 990), presentation).hosts[0].age, 'Reading time unknown.');
  const sparse = pair();
  sparse.sample.workers = [];
  sparse.sample.driver_host = { host: 'spark-a', t: 1000 } as never;
  const card = hostSummary(sparse, presentation).hosts[0];
  assert.deepEqual(
    card.readings.map((reading) => reading.text),
    ['unknown', 'unknown', 'unknown', 'unknown', 'unknown'],
  );
});

test('readiness and missing samples are read defensively', () => {
  const { sample, ...rest } = pair();
  const { readiness: _, ...older } = sample;
  assert.deepEqual(hostSummary({ ...rest, sample: older }, presentation).readiness, []);
  const malformed = hostSummary(
    { ...rest, sample: { ...sample, readiness: { schema_version: 2 } } },
    presentation,
  );
  assert.deepEqual(
    malformed.readiness.map((row) => [row.state, row.detail, row.mark.icon]),
    Array(3).fill(['unknown', 'Readiness is missing or malformed.', '?']),
  );
  const odd = hostSummary(
    {
      ...rest,
      sample: {
        ...sample,
        readiness: {
          ...readiness,
          lifecycle: {
            ...readiness.lifecycle,
            attached_mesh: { state: 'constructor', detail: 'x' },
          },
        },
      },
    },
    presentation,
  );
  assert.equal(odd.readiness[1].state, 'unknown');
  assert.equal(hostSummary({}, presentation).note, 'No telemetry sample yet.');
  assert.equal(
    hostSummary(
      { state: 'connected', sample: { fixture: true, render: { state: 'idle' } } },
      presentation,
    ).note,
    'The latest sample carries no host readings.',
  );
  const omitted = 'Driver telemetry exceeds the 2 MiB snapshot budget.';
  assert.equal(
    hostSummary({ state: 'connected', sample_omitted: omitted }, presentation).note,
    omitted,
  );
});

/** Two samples a second apart, so the rail rates have a value. */
function history(): MetricSeries[] {
  let series = appendTelemetry([], pair(1000), presentation.metrics);
  const later = pair(1001);
  for (const worker of later.sample.workers) worker.host.rails.rail0.rx_bytes += 2e9;
  later.sample.driver_host.rails.rail0.rx_bytes += 2e9;
  series = appendTelemetry(series, later, presentation.metrics);
  return series;
}

test('the chart offers hosts and metrics as two choices, rails named under their host', () => {
  const all = history();
  const choice = chartChoice(all, '', '');
  assert.deepEqual(choice.hosts, ['spark-a', 'spark-b', RENDER_PROGRESS]);
  assert.equal(choice.host, 'spark-a');
  // The chart opens on a metric that moves while a job renders.
  assert.equal(choice.series?.id, 'spark-a:host.gpu.utilization');
  assert.ok(choice.metrics.every((series) => seriesHost(series) === 'spark-a'));
  assert.deepEqual(
    choice.metrics.map(seriesLabel).filter((label) => label.startsWith('Rail')),
    ['Rail receive, rail0', 'Rail receive, rail1', 'Rail transmit, rail0', 'Rail transmit, rail1'],
  );
  const render = chartChoice(all, RENDER_PROGRESS, '');
  assert.deepEqual(render.metrics.map(seriesLabel), [
    'Step',
    'Total steps',
    'Step duration',
    'Last render duration',
  ]);
  // Another host keeps the metric, and the rail, in view.
  const rail = all.find((series) => series.id === 'spark-a/rail1:host.rail.rx')!;
  assert.equal(sameMetricOn(all, 'spark-b', rail)?.id, 'spark-b/rail1:host.rail.rx');
  assert.equal(sameMetricOn(all, RENDER_PROGRESS, rail)?.id, 'driver:render.step');
  // A choice the history no longer holds falls back to the first host.
  assert.equal(chartChoice(all, 'gone', 'gone:host.gpu.power').host, 'spark-a');
});

test('the chart shows a single host or metric as text, not a one-item select', async () => {
  const two = await renderComponent('./components/TelemetryCharts.svelte', {
    history: history(),
    presentation,
  });
  const selects = [...two.matchAll(/<select[\s\S]*?<\/select>/g)].map((match) => match[0]);
  assert.equal(selects.length, 2);
  assert.deepEqual(
    [...selects[0].matchAll(/<option[^>]*>([^<]*)<\/option>/g)].map((match) => match[1]),
    ['spark-a', 'spark-b', RENDER_PROGRESS],
  );
  assert.match(visibleText(two), /Host spark-a spark-b Render progress Metric Total memory/);
  const { sample, ...rest } = pair();
  const hostsOnly = presentation.metrics.filter((metric) => !metric.id.startsWith('render.'));
  const single = appendTelemetry([], { ...rest, sample: { ...sample, workers: [] } }, hostsOnly);
  const one = await renderComponent('./components/TelemetryCharts.svelte', {
    history: single,
    presentation,
  });
  assert.equal([...one.matchAll(/<select/g)].length, 1, 'only the Metric choice is a select');
  assert.match(visibleText(one), /Host spark-a The only source reporting\. Metric/);
  const lone = await renderComponent('./components/TelemetryCharts.svelte', {
    history: single.filter((series) => series.metric.id === 'host.gpu.power'),
    presentation,
  });
  assert.doesNotMatch(lone, /<select/);
  assert.match(
    visibleText(lone),
    /Host spark-a The only source reporting\. Metric GPU power The only metric this host reports\./,
  );
});

test('the overview renders every host, its readings and the readiness rows', async () => {
  const text = visibleText(
    await renderComponent('./components/HostHealth.svelte', {
      telemetry: pair(),
      presentation,
      heading: 'Hosts',
    }),
  );
  for (const part of [
    'Hosts spark-a Driver · Worker rank 0 of 2 Available memory ✓ 62.2 of 121.6 GiB GPU memory 32.8 GiB GPU utilization 97 %',
    'spark-b Worker rank 1 of 2 Available memory ! 18.4 of 121.6 GiB',
    'Worker service ✓ ready 2 worker services ready.',
    'Attached mesh ! active The attached mesh has active work in flight.',
    'Render session ! active 1 render session is active.',
  ])
    assert.ok(text.includes(part), part);
});
