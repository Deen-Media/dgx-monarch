import type { Telemetry } from './contracts';
export interface MetricDefinition {
  id: string;
  name: string;
  source: string;
  unit: string;
  aggregation: string;
  threshold?: string | null;
  /** `slab_rule` names the contract's rule for a missing value; anything else means unknown. */
  missing?: string;
}
export interface Presentation {
  metrics: MetricDefinition[];
  units: Record<string, { precision: number }>;
  memory: Record<string, { hex: string }>;
  freshness: { minimum_stale_s: number; stale_after_intervals: number; default_interval_s: number };
}
export interface MetricPoint {
  time: number;
  value: number | null;
}
export interface MetricSeries {
  id: string;
  host: string;
  metric: MetricDefinition;
  points: MetricPoint[];
}
const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
const number = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;
function read(value: unknown, path: string): number | null {
  let node = value;
  for (const key of path.split('.')) node = record(node)[key];
  return number(node);
}
/**
 * Slab GiB a host's chart records under the presentation contract's slab rule, the rule
 * `slab_reading` in studio/presentation.py applies for `dgxm top` and the sidebar. A Worker
 * reports `slab_gib` only while a slab holds its model, so a missing value is a known zero
 * unless the report could hide one. `undefined` means no Worker reports the host.
 */
export function slabReading(worker: unknown): number | null {
  if (worker === undefined) return 0;
  if (!worker || typeof worker !== 'object' || Array.isArray(worker)) return null;
  const report = worker as Record<string, unknown>;
  if (report.memory_error != null) return null;
  const memory = report.memory;
  if (!memory || typeof memory !== 'object' || Array.isArray(memory)) return null;
  const fields = memory as Record<string, unknown>;
  if ('slab_gib' in fields) {
    const value = number(fields.slab_gib);
    return value === null ? null : Math.max(0, value);
  }
  const residency = fields.weight_residency;
  // Every Worker that holds a model also reports its weight_residency.
  if (residency == null) return 'patcher_class' in fields || 'model' in fields ? null : 0;
  return typeof residency === 'string' && residency !== 'slab' ? 0 : null;
}
/** Host clocks may run slightly ahead of the companion that fetched their sample. */
export const SAMPLE_CLOCK_MARGIN_S = 5;
export function appendTelemetry(
  history: MetricSeries[],
  telemetry: Telemetry,
  definitions: MetricDefinition[],
  limit = 240,
): MetricSeries[] {
  if (telemetry.state !== 'connected') return history;
  const sample = record(telemetry.sample),
    sourceTime = number(sample.t);
  // Bound sample times by the companion's clock at fetch, not the browser's: a remote browser
  // behind the host would drop every sample. The bound keeps one far-future time from blocking
  // every later point at the order check below.
  const latest = (number(telemetry.observed_at) ?? Date.now() / 1000) + SAMPLE_CLOCK_MARGIN_S;
  if (sourceTime === null || sourceTime > latest) return history;
  const hosts: Record<string, unknown>[] = [
    record(sample.driver_host),
    ...(Array.isArray(sample.workers)
      ? sample.workers.map((worker) => ({ ...record(record(worker).host), worker }))
      : []),
  ];
  const updates: { host: string; metric: MetricDefinition; time: number; value: number | null }[] =
    [];
  const workerHosts = new Set(
    hosts.filter((scope) => 'worker' in scope).map((scope) => String(scope.host ?? 'unknown')),
  );
  for (const metric of definitions) {
    const scopes: Record<string, unknown>[] = metric.id.startsWith('render.')
      ? [{ host: 'driver', t: sourceTime, ...sample }]
      : hosts;
    for (const scope of scopes) {
      const host = String(scope.host ?? 'unknown'),
        time = number(scope.t) ?? sourceTime;
      if (metric.aggregation === 'per_host_per_rail_rate') {
        const rails = record(scope.rails),
          stamp = number(scope.rails_t);
        for (const [rail, counters] of Object.entries(rails))
          updates.push({
            host: `${host}/${rail}`,
            metric,
            time: stamp ?? time,
            value: read(counters, metric.source.split('.').at(-1)!),
          });
      } else {
        let value = read(scope, metric.source);
        if (metric.missing === 'slab_rule') {
          // The driver's own box takes its Worker's report when a Worker runs there.
          if (!('worker' in scope) && workerHosts.has(host)) continue;
          value = slabReading('worker' in scope ? scope.worker : undefined);
        }
        if (metric.id === 'host.memory.cache') {
          const available = read(scope, 'mem_gib.MemAvailable'),
            free = read(scope, 'mem_gib.MemFree');
          value =
            available === null || free === null
              ? null
              : Math.min(available, Math.max(0, available - free));
        }
        updates.push({ host, metric, time, value });
      }
    }
  }
  const output = new Map(history.map((series) => [series.id, series]));
  for (const item of updates) {
    const id = `${item.host}:${item.metric.id}`,
      prior = output.get(id),
      last = prior?.points.at(-1);
    if (last && item.time <= last.time) continue;
    // Rail counters retain raw value privately in series so reset and irregular intervals stay explicit.
    const priorCounter = (prior as MetricSeries & { counter?: number | null })?.counter;
    const rate = item.metric.aggregation === 'per_host_per_rail_rate';
    const value = rate
      ? last && priorCounter != null && item.value != null && item.value >= priorCounter
        ? (item.value - priorCounter) / (item.time - last.time) / 1e9
        : null
      : item.value;
    output.set(id, {
      id,
      host: item.host,
      metric: item.metric,
      points: [...(prior?.points ?? []), { time: item.time, value }].slice(-limit),
      ...(rate ? { counter: item.value } : {}),
    });
  }
  return [...output.values()].slice(-256);
}
export function chartPath(points: MetricPoint[], width = 600, height = 120): string {
  const known = points.filter((point) => point.value !== null);
  if (!known.length) return '';
  const min = Math.min(...known.map((point) => point.value!)),
    max = Math.max(...known.map((point) => point.value!));
  const start = points[0].time,
    span = Math.max(1, points.at(-1)!.time - start);
  let pen = false;
  return points
    .map((point) => {
      if (point.value === null) {
        pen = false;
        return '';
      }
      const text = `${pen ? 'L' : 'M'}${(((point.time - start) / span) * width).toFixed(1)},${(height - ((point.value - min) / (max - min || 1)) * height).toFixed(1)}`;
      pen = true;
      return text;
    })
    .join(' ');
}
