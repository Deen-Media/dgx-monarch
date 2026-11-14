import type { Telemetry } from './contracts';
import { readPath, SAMPLE_CLOCK_MARGIN_S, type Presentation } from './telemetryHistory.ts';

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
const finite = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/** The readings each host shows, by presentation contract metric id, top to bottom. */
const READINGS = [
  'host.memory.available',
  'host.memory.gpu',
  'host.gpu.utilization',
  'host.gpu.power',
  'host.gpu.temperature',
];
/** The lifecycle layers of the driver's readiness report, named as the ComfyUI panel names them. */
const LAYERS = [
  ['worker_service', 'Worker service'],
  ['attached_mesh', 'Attached mesh'],
  ['render_session', 'Render session'],
] as const;
/** The status token each layer state shows with; active shows yellow, as in the panel's card. */
const LAYER_TOKEN: Record<string, string> = {
  ready: 'ready',
  idle: 'idle',
  active: 'degraded',
  blocked: 'blocked',
  unknown: 'unknown',
};

export interface StatusMark {
  hex: string;
  icon: string;
}
export interface Reading {
  id: string;
  name: string;
  text: string;
  /** The display threshold's mark and what it means, for available memory only. */
  status?: StatusMark & { title: string };
}
export interface HostCard {
  host: string;
  /** What runs on the host: the driver, and each Worker by rank. */
  roles: string;
  readings: Reading[];
  /** Why the readings may not be current, or ''. */
  age: string;
}
export interface ReadinessRow {
  label: string;
  /** The state the report gave; while the readings are retained, the state when last read. */
  state: string;
  detail: string;
  mark: StatusMark;
  /** A stale label with the report's age while its readings are retained, or ''. */
  stale: string;
}
/** One line for the whole readiness report, for a panel too narrow for its rows. */
export interface ReadinessOverall {
  /**
   * ready, active, unknown or blocked: the worst layer's state, or the driver's own verdict when
   * that is worse, as when its telemetry refresh failed. An idle layer counts as ready.
   */
  state: string;
  /** The layers in that state, by name; none while every layer is ready or idle. */
  layers: string[];
  mark: StatusMark;
  /** The rows' stale label while the readings are retained, or ''. */
  stale: string;
}
export interface HealthSummary {
  hosts: HostCard[];
  /** Why the list has one host or none, or which Workers sent no readings; '' otherwise. */
  note: string;
  readiness: ReadinessRow[];
  /** The readiness rows in one line, or null when the driver sends no readiness report. */
  overall: ReadinessOverall | null;
  /** Why every reading shown is a retained one, with its age, or '' while they are current. */
  stale: string;
}
/**
 * What the overview needs to tell current readings from retained ones: the companion's time at
 * the newest snapshot, and whether Studio still reaches the companion.
 */
export interface Clock {
  now?: number | null;
  live?: boolean;
}

interface Source {
  host: string;
  roles: string[];
  block: Record<string, unknown>;
  time: number | null;
}

/** One entry per host name: the driver's own box, then each Worker's, merged when they share one. */
function sources(sample: Record<string, unknown>) {
  const hosts = new Map<string, Source>();
  const add = (host: string, role: string, block: Record<string, unknown>) => {
    const entry = hosts.get(host) ?? { host, roles: [], block: {}, time: null };
    entry.roles.push(role);
    const time = finite(block.t);
    // The driver and a Worker on one box read the same machine, so the newer reading wins.
    if (time !== null && (entry.time === null || time > entry.time))
      [entry.block, entry.time] = [block, time];
    hosts.set(host, entry);
  };
  const driver = record(sample.driver_host);
  if (typeof driver.host === 'string' && driver.host) add(driver.host, 'Driver', driver);
  let workers = 0,
    silent = 0;
  for (const item of Array.isArray(sample.workers) ? sample.workers : []) {
    workers++;
    const worker = record(item),
      stats = record(worker.host);
    // A Worker that has not run setup reports its host name as text, without readings.
    const name =
      typeof stats.host === 'string'
        ? stats.host
        : typeof worker.host === 'string'
          ? worker.host
          : '';
    if (!name) {
      silent++;
      continue;
    }
    const { rank, world } = worker;
    add(
      name,
      Number.isInteger(rank) && Number.isInteger(world)
        ? `Worker rank ${rank} of ${world}`
        : 'Worker',
      stats,
    );
  }
  return { hosts: [...hosts.values()], workers, silent };
}

function mark(presentation: Presentation | undefined, token: string): StatusMark {
  const status = presentation?.status?.[token];
  return { hex: status?.hex ?? '', icon: status?.icon ?? '' };
}

function unitText(value: number | null, unit: string, presentation?: Presentation): string {
  if (value === null) return 'unknown';
  return `${value.toFixed(presentation?.units?.[unit]?.precision ?? 2)} ${unit}`;
}

function readings(block: Record<string, unknown>, presentation?: Presentation): Reading[] {
  const metrics = presentation?.metrics ?? [];
  const metric = (id: string) => metrics.find((item) => item.id === id);
  const value = (id: string) => {
    const definition = metric(id);
    return definition ? readPath(block, definition.source) : null;
  };
  return READINGS.flatMap((id) => {
    const definition = metric(id);
    if (!definition) return [];
    const unit = definition.unit,
      reading = value(id);
    if (id !== 'host.memory.available')
      return [{ id, name: definition.name, text: unitText(reading, unit, presentation) }];
    const total = value('host.memory.total');
    const precision = presentation?.units?.[unit]?.precision ?? 2;
    const text =
      reading !== null && total !== null
        ? `${reading.toFixed(precision)} of ${unitText(total, unit, presentation)}`
        : unitText(reading, unit, presentation);
    const rule = presentation?.thresholds?.available_memory_gib;
    if (!rule || rule.operator !== '>') return [{ id, name: definition.name, text }];
    const token = reading === null ? 'unknown' : reading > rule.value ? rule.above : rule.otherwise;
    const title =
      reading === null
        ? 'Available memory is unknown'
        : reading > rule.value
          ? `More than ${rule.value} GiB available`
          : `${rule.value} GiB or less available`;
    return [{ id, name: definition.name, text, status: { ...mark(presentation, token), title } }];
  });
}

/** Readings older than this, when the companion fetched the sample, read as stale. */
export function staleAfter(presentation?: Presentation): number {
  const rule = presentation?.freshness;
  return Math.max(
    (rule?.stale_after_intervals ?? 3) * (rule?.default_interval_s ?? 1),
    rule?.minimum_stale_s ?? 3,
  );
}

/** An age in plain words: seconds up to two minutes, then whole minutes. */
const span = (seconds: number) =>
  seconds < 120 ? `${Math.round(seconds)} s` : `${Math.floor(seconds / 60)} min`;

/**
 * Readings the companion still shows after its telemetry stopped: after a failed poll it keeps
 * the last sample and its `observed_at`, so their difference stops growing. `now` is the
 * companion's time at the newest snapshot; without the companion there is no current time.
 */
interface Held {
  lost: boolean;
  now: number | null;
}
function held(telemetry: Telemetry, clock: Clock): Held | null {
  if (!telemetry?.sample) return null;
  const lost = clock.live === false;
  if (!lost && telemetry.state === 'connected') return null;
  return { lost, now: lost ? null : finite(clock.now) };
}

/** How old a reading taken at `time` is now, or null when the clocks cannot say. */
function since(time: number | null, now: number | null): number | null {
  if (time === null || now === null) return null;
  const seconds = now - time;
  // A host clock may run a little ahead of the companion's, as the charts allow.
  return seconds < -SAMPLE_CLOCK_MARGIN_S ? null : Math.max(0, seconds);
}

function age(
  source: Source,
  observed: number | null,
  kept: Held | null,
  presentation?: Presentation,
): string {
  if (!Object.keys(source.block).length) return 'No readings from this host yet.';
  if (kept) {
    const seconds = since(source.time, kept.now);
    return seconds === null ? 'Stale reading.' : `Stale reading, ${span(seconds)} old.`;
  }
  if (source.time === null || observed === null) return 'Reading time unknown.';
  const seconds = since(source.time, observed);
  if (seconds === null) return 'Reading time unknown.';
  return seconds > staleAfter(presentation) ? `Stale reading, ${span(seconds)} old.` : '';
}

/** Why the shown readings are retained, with the time since the last good one, or ''. */
function staleNote(kept: Held | null, observed: number | null): string {
  if (!kept) return '';
  if (kept.lost)
    return 'Studio cannot reach the companion, so these are the last readings it sent.';
  const seconds = since(observed, kept.now);
  return seconds === null
    ? "The companion's telemetry has stopped, so these are the last readings it received."
    : `The companion has had no new telemetry for ${span(seconds)}, so these are the last readings it received.`;
}

function readiness(
  sample: Record<string, unknown>,
  observed: number | null,
  kept: Held | null,
  presentation?: Presentation,
): ReadinessRow[] {
  // An older driver sends no readiness report; there is nothing to show then.
  if (!('readiness' in sample)) return [];
  const report = record(sample.readiness),
    valid = report.schema_version === 1,
    lifecycle = record(report.lifecycle);
  // A retained report says what held when it was read; what holds now is unknown.
  const seconds = kept ? since(observed, kept.now) : null;
  const stale = !kept ? '' : seconds === null ? 'stale' : `stale, ${span(seconds)} old`;
  return LAYERS.map(([id, label]) => {
    const layer = valid ? record(lifecycle[id]) : {};
    const state =
      typeof layer.state === 'string' && Object.hasOwn(LAYER_TOKEN, layer.state)
        ? layer.state
        : 'unknown';
    const detail =
      typeof layer.detail === 'string'
        ? layer.detail.replace(/\s+/g, ' ').trim().slice(0, 400)
        : '';
    return {
      label,
      state,
      detail: state === layer.state && detail ? detail : 'Readiness is missing or malformed.',
      mark: mark(presentation, kept ? 'unknown' : LAYER_TOKEN[state]),
      stale,
    };
  });
}

/** How bad a layer state is for the overall line; the worst one decides it. */
const RANK: Record<string, number> = { ready: 0, idle: 0, active: 1, unknown: 2, blocked: 3 };

function overall(
  rows: ReadinessRow[],
  verdict: unknown,
  kept: Held | null,
  presentation?: Presentation,
): ReadinessOverall | null {
  if (!rows.length) return null;
  let state = rows.reduce(
    (worst, row) => (RANK[row.state] > RANK[worst] ? row.state : worst),
    'ready',
  );
  // The driver's verdict can only make the line worse, never better than its own rows.
  if ((verdict === 'unknown' || verdict === 'blocked') && RANK[verdict] > RANK[state])
    state = verdict;
  return {
    state,
    layers:
      state === 'ready' ? [] : rows.filter((row) => row.state === state).map((row) => row.label),
    mark: mark(presentation, kept ? 'unknown' : LAYER_TOKEN[state]),
    stale: rows[0].stale,
  };
}

function note(hosts: number, workers: number, silent: number, telemetry: Telemetry): string {
  if (!hosts) {
    if (typeof telemetry.sample_omitted === 'string') return telemetry.sample_omitted;
    return telemetry.sample
      ? 'The latest sample carries no host readings.'
      : 'No telemetry sample yet.';
  }
  const parts: string[] = [];
  if (silent)
    parts.push(
      `${silent === 1 ? 'One Worker' : `${silent} Workers`} answered without host readings.`,
    );
  if (hosts === 1)
    parts.push(
      workers > silent
        ? 'Every Worker that reports runs on this host.'
        : "Only the driver's host reports. Worker hosts join this list while a Monarch render's mesh is attached.",
    );
  return parts.join(' ');
}

/**
 * Every host the latest sample reports, side by side, with the driver's readiness report as rows
 * and as one overall line. Values
 * come from the presentation contract's metric sources and units, so they read as the charts do;
 * unknown stays unknown, and memory categories are never added into one total. Readings the
 * companion retained after its telemetry stopped, or that Studio holds after losing the
 * companion, carry a stale label with their age on each host and readiness row.
 */
export function hostSummary(
  telemetry: Telemetry,
  presentation?: Presentation,
  clock: Clock = {},
): HealthSummary {
  const sample = record(telemetry?.sample),
    observed = finite(telemetry?.observed_at),
    kept = held(telemetry, clock);
  const { hosts, workers, silent } = sources(sample);
  const rows = readiness(sample, observed, kept, presentation),
    report = record(sample.readiness);
  return {
    hosts: hosts.map((source) => ({
      host: source.host,
      roles: source.roles.join(' · '),
      readings: readings(source.block, presentation),
      age: age(source, observed, kept, presentation),
    })),
    note: note(hosts.length, workers, silent, telemetry ?? {}),
    readiness: rows,
    overall: overall(rows, report.schema_version === 1 ? report.overall : null, kept, presentation),
    stale: staleNote(kept, observed),
  };
}
