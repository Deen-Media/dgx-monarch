import type { MetricSeries } from './telemetryHistory';

/** The Host choice for the driver's render progress values, which belong to no single host. */
export const RENDER_PROGRESS = 'Render progress';
/** The metric a chart opens on: it moves while a job renders, unlike total memory. */
const OPENING_METRIC = 'host.gpu.utilization';

const railOf = (series: MetricSeries) =>
  series.metric.aggregation === 'per_host_per_rail_rate' ? series.host.indexOf('/') : -1;

/** The host a series belongs to: its host name, or Render progress for the render values. */
export function seriesHost(series: MetricSeries): string {
  if (series.metric.id.startsWith('render.')) return RENDER_PROGRESS;
  const slash = railOf(series);
  return slash > 0 ? series.host.slice(0, slash) : series.host;
}

/** A series' name in the Metric choice; a rail rate adds its rail. */
export function seriesLabel(series: MetricSeries): string {
  const slash = railOf(series);
  return slash > 0 ? `${series.metric.name}, ${series.host.slice(slash + 1)}` : series.metric.name;
}

/**
 * The chart's choices: hosts in the order they first reported with Render progress last, the
 * chosen host's metrics, and the series to draw. A choice no longer in the history falls back
 * to the first host and the opening metric.
 */
export function chartChoice(history: MetricSeries[], host: string, seriesId: string) {
  const hosts = [...new Set(history.map(seriesHost))].sort(
    (a, b) => Number(a === RENDER_PROGRESS) - Number(b === RENDER_PROGRESS),
  );
  const chosen = hosts.includes(host) ? host : (hosts[0] ?? '');
  const metrics = history.filter((series) => seriesHost(series) === chosen);
  const series =
    metrics.find((item) => item.id === seriesId) ??
    metrics.find((item) => item.metric.id === OPENING_METRIC) ??
    metrics[0];
  return { hosts, host: chosen, metrics, series };
}

/** The series that shows the same metric, and rail, on another host; or that host's first. */
export function sameMetricOn(
  history: MetricSeries[],
  host: string,
  current: MetricSeries | undefined,
): MetricSeries | undefined {
  const metrics = history.filter((series) => seriesHost(series) === host);
  return (
    metrics.find(
      (item) =>
        current &&
        item.metric.id === current.metric.id &&
        seriesLabel(item) === seriesLabel(current),
    ) ?? chartChoice(history, host, '').series
  );
}
