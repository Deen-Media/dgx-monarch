import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { hostSummary } from './hostHealth.ts';
import { initialWorkspace, saveLayout, telemetryDetailsOpen } from './initialWorkspace.ts';
import { appendTelemetry, type Presentation } from './telemetryHistory.ts';
import { componentFunctions, renderComponent, visibleText } from './renderHarness.js';

const presentation: Presentation = JSON.parse(
  readFileSync(
    new URL('../../src/dgx_monarch/studio/data/presentation.json', import.meta.url),
    'utf8',
  ),
);
const LAYOUT_KEY = 'dgxm.studio.layout.v1';
/** One host's readings in the shape telemetry.host_stats() publishes. */
const host = (name: string, t: number, available: number) => ({
  host: name,
  t,
  mem_gib: { MemTotal: 121.6, MemAvailable: available },
  gpu: { util_pct: 40, power_w: 31.5, temp_c: 52 },
  gpu_proc_gib: 12.5,
});
const READY = {
  worker_service: { state: 'ready', detail: '2 worker services ready.' },
  attached_mesh: { state: 'ready', detail: 'The attached mesh is ready.' },
  render_session: { state: 'idle', detail: 'No render session is active.' },
};
/** A two-host sample whose readiness report holds `lifecycle`, with `report` fields over it. */
function telemetry(lifecycle: Record<string, unknown> = READY, report: object = {}, t = 1000) {
  return {
    state: 'connected',
    observed_at: t + 0.4,
    sample: {
      t,
      driver_host: host('spark-a', t, 62.2),
      workers: [
        { rank: 0, world: 2, host: host('spark-a', t - 0.5, 61.9) },
        { rank: 1, world: 2, host: host('spark-b', t - 0.4, 44.1) },
      ],
      readiness: { schema_version: 1, overall: 'ready', lifecycle, ...report },
    },
  };
}
const busy = {
  ...READY,
  attached_mesh: { state: 'active', detail: 'The attached mesh has active work in flight.' },
  render_session: { state: 'active', detail: '1 render session is active.' },
};
const line = (sample: object, clock = {}) => {
  const overall = hostSummary(sample, presentation, clock).overall;
  return overall && [overall.state, overall.layers, overall.mark.icon, overall.stale];
};

test('the overall readiness line names the worst layer state and the layers in it', () => {
  assert.deepEqual(line(telemetry()), ['ready', [], '✓', ''], 'an idle render session is ready');
  assert.deepEqual(line(telemetry(busy, { overall: 'degraded' })), [
    'active',
    ['Attached mesh', 'Render session'],
    '!',
    '',
  ]);
  const stopped = { ...busy, worker_service: { state: 'blocked', detail: 'No Worker answers.' } };
  assert.deepEqual(line(telemetry(stopped, { overall: 'blocked' })), [
    'blocked',
    ['Worker service'],
    '?',
    '',
  ]);
  assert.deepEqual(line(telemetry(READY, { schema_version: 2 })), [
    'unknown',
    ['Worker service', 'Attached mesh', 'Render session'],
    '?',
    '',
  ]);
  // A failed telemetry refresh makes the driver's verdict unknown over ready rows.
  assert.deepEqual(line(telemetry(READY, { overall: 'unknown' })), ['unknown', [], '?', '']);
  // The driver's verdict never makes the line better than its own rows.
  assert.equal(line(telemetry(stopped, { overall: 'ready' }))?.[0], 'blocked');
  const older = telemetry();
  delete (older.sample as Record<string, unknown>).readiness;
  assert.equal(line(older), null, 'an older driver sends no report, so there is no line');
});

test('a retained report keeps its last state on the line, marked stale with its age', () => {
  assert.deepEqual(line({ ...telemetry(busy), state: 'stale' }, { now: 1042.4, live: true }), [
    'active',
    ['Attached mesh', 'Render session'],
    '?',
    'stale, 42 s old',
  ]);
});

/** Browser storage for a test, or one that refuses every call. */
function storage(initial: Record<string, string> = {}, refuse = false) {
  const values = new Map(Object.entries(initial));
  const check = () => {
    if (refuse) throw new Error('storage is off');
  };
  globalThis.localStorage = {
    getItem: (key: string) => (check(), values.get(key) ?? null),
    setItem: (key: string, value: string) => (check(), void values.set(key, value)),
  } as Storage;
  globalThis.sessionStorage = { getItem: () => null } as unknown as Storage;
  return values;
}
/** The panel's own markup in a workspace with history, so every chart control renders. */
async function panel() {
  const definitions = presentation.metrics;
  let history = appendTelemetry([], telemetry(busy, {}, 1000), definitions);
  history = appendTelemetry(history, telemetry(busy, {}, 1001), definitions);
  return renderComponent('./components/TelemetryPanel.svelte', {
    telemetry: telemetry(busy, {}, 1001),
    history,
    presentation,
    driverState: 'connected',
    connectionState: 'connected',
    serverTime: 1001.5,
    activeCount: 1,
    job: null,
  });
}
const opening = (body: string) => body.match(/<details class="telemetry-details[^>]*>/)?.[0] ?? '';

test('the panel keeps both hosts and readiness in view and folds the detail away', async () => {
  storage();
  const body = await panel();
  const fold = body.indexOf('<details class="telemetry-details');
  assert.ok(fold > 0, 'the panel has an expander');
  assert.doesNotMatch(opening(body), /\bopen\b/, 'closed until opened');
  const top = visibleText(body.slice(0, fold)),
    folded = visibleText(body.slice(fold));
  assert.ok(top.startsWith('Spark telemetry'), top);
  // Readiness sits with the jobs and the driver, above the two host cards.
  for (const shown of [
    "Active or unresolved jobs 1 Driver connected Readiness ! active Attached mesh, Render session This tab's job None",
    'Hosts spark-a Driver · Worker rank 0 of 2 Available memory ✓ 62.2 of 121.6 GiB',
    'spark-b Worker rank 1 of 2 Available memory ✓ 44.1 of 121.6 GiB',
  ])
    assert.ok(top.includes(shown), shown);
  assert.ok(top.indexOf('Readiness') < top.indexOf('Hosts'));
  for (const hidden of ['Metric history', 'Inspect sample', 'Export history', 'in flight'])
    assert.ok(!top.includes(hidden), hidden);
  // Nothing is removed: the rows, the chart and its choices, inspection, export and raw sample.
  assert.ok(folded.startsWith('Detailed telemetry Worker service ✓ ready 2 worker services'));
  for (const kept of [
    'Attached mesh ! active The attached mesh has active work in flight.',
    'Metric history Host',
    'Inspect sample',
    'Sample table',
    'Export history',
    'Authoritative host sample',
    'Reclaimable cache',
    'Unknown counters stay unknown.',
  ])
    assert.ok(folded.includes(kept), kept);
  assert.match(body, /<details><summary>Authoritative host sample<\/summary>/);
});

test('the detailed telemetry stays open across a reload, kept with the layout', async () => {
  const values = storage({
    [LAYOUT_KEY]: JSON.stringify({ width: 400, showTelemetry: true, showConsole: false }),
  });
  assert.equal(telemetryDetailsOpen(), false);
  // The panel's own handler, as the details element calls it on a toggle.
  const toggle = new Function(
    'saveLayout',
    `'use strict'; let detailed = false;
     ${await componentFunctions('./components/TelemetryPanel.svelte', ['toggleDetails'])}
     return { toggleDetails, detailed: () => detailed };`,
  )(saveLayout);
  const element = { open: true };
  toggle.toggleDetails({ target: element, currentTarget: element });
  assert.equal(toggle.detailed(), true);
  assert.deepEqual(JSON.parse(values.get(LAYOUT_KEY)!), {
    width: 400,
    showTelemetry: true,
    showConsole: false,
    telemetryDetails: true,
  });
  assert.match(opening(await panel()), /\bopen\b/, 'a reload opens it again');
  // The workspace saves its own layout without dropping the panel's choice, and back.
  saveLayout({ width: 420, showTelemetry: true, showConsole: true });
  assert.equal(telemetryDetailsOpen(), true);
  assert.deepEqual(initialWorkspace().layout, {
    width: 420,
    showTelemetry: true,
    showConsole: true,
  });
  // A nested disclosure's toggle is not the panel's.
  toggle.toggleDetails({ target: { open: false }, currentTarget: element });
  assert.equal(telemetryDetailsOpen(), true);
  element.open = false;
  toggle.toggleDetails({ target: element, currentTarget: element });
  assert.equal(telemetryDetailsOpen(), false);
  assert.equal(JSON.parse(values.get(LAYOUT_KEY)!).width, 420);
  assert.doesNotMatch(opening(await panel()), /\bopen\b/);
});

test('without a stored choice, or without storage, the detail starts closed', () => {
  for (const stored of [null, '{', '[true]', '{"telemetryDetails":"yes"}']) {
    storage(stored === null ? {} : { [LAYOUT_KEY]: stored });
    assert.equal(telemetryDetailsOpen(), false, String(stored));
  }
  storage({}, true);
  assert.equal(telemetryDetailsOpen(), false);
  assert.doesNotThrow(() => saveLayout({ telemetryDetails: true }));
});

test('a retained report shows its stale label beside the readiness line', async () => {
  storage();
  const body = await renderComponent('./components/TelemetryPanel.svelte', {
    telemetry: { ...telemetry(busy), state: 'stale' },
    history: [],
    presentation,
    connectionState: 'connected',
    serverTime: 1042.4,
    activeCount: 0,
  });
  const top = visibleText(body.slice(0, body.indexOf('<details class="telemetry-details')));
  assert.ok(top.includes('Readiness ? active Attached mesh, Render session stale, 42 s old'), top);
  const older = telemetry();
  delete (older.sample as Record<string, unknown>).readiness;
  const none = await renderComponent('./components/TelemetryPanel.svelte', {
    telemetry: older,
    history: [],
    presentation,
    connectionState: 'connected',
    activeCount: 0,
  });
  assert.doesNotMatch(visibleText(none), /Readiness/, 'no report, no line');
});

test('Fleet keeps the hosts and the full readiness rows in one block, with no summary line', async () => {
  const fleet = visibleText(
    await renderComponent('./components/HostHealth.svelte', {
      telemetry: telemetry(busy),
      presentation,
    }),
  );
  assert.match(fleet, /spark-b Worker rank 1 of 2/);
  assert.match(fleet, /Render session ! active 1 render session is active\./);
  assert.doesNotMatch(fleet, /Readiness/);
  const rows = visibleText(
    await renderComponent('./components/HostHealth.svelte', {
      telemetry: telemetry(busy),
      presentation,
      part: 'readiness',
    }),
  );
  assert.doesNotMatch(rows, /spark-a|Available memory/, 'the rows alone');
  assert.match(rows, /^Worker service ✓ ready 2 worker services ready\./);
  const hosts = visibleText(
    await renderComponent('./components/HostHealth.svelte', {
      telemetry: telemetry(busy),
      presentation,
      part: 'hosts',
    }),
  );
  assert.match(hosts, /^spark-a Driver · Worker rank 0 of 2/);
  assert.doesNotMatch(hosts, /Worker service|Readiness/, 'the hosts alone');
});
