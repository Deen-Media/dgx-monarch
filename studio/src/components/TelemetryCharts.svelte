<script lang="ts">
  import type { MetricSeries, Presentation } from '../telemetryHistory';
  import { chartPath } from '../telemetryHistory';
  import { chartChoice, sameMetricOn, seriesLabel } from '../chartChoice';
  import { exportJSON } from '../files';
  let { history, presentation }: { history: MetricSeries[]; presentation?: Presentation } =
    $props();
  let host = $state(''),
    selected = $state(''),
    point = $state(-1);
  const choice = $derived(chartChoice(history, host, selected));
  const series = $derived(choice.series);
  /** A new host keeps the metric in view when that host reports it too. */
  function chooseHost(value: string) {
    selected = sameMetricOn(history, value, series)?.id ?? '';
    host = value;
    point = -1;
  }
  const current = $derived(
    series?.points[
      point < 0 ? series.points.length - 1 : Math.min(point, series.points.length - 1)
    ],
  );
  const known = $derived(
    series?.points.filter((point) => point.value !== null).map((point) => point.value!) ?? [],
  );
  function formatted(value: number | null | undefined) {
    return value == null
      ? 'Unknown'
      : `${value.toFixed(presentation?.units[series?.metric.unit ?? '']?.precision ?? 2)} ${series?.metric.unit ?? ''}`;
  }
</script>

<section class="metric-chart">
  <h3>Metric history</h3>
  {#if history.length}<div class="chart-choice">
      {#if choice.hosts.length > 1}<label
          >Host<select
            value={choice.host}
            onchange={(event) => chooseHost(event.currentTarget.value)}
            >{#each choice.hosts as item (item)}<option value={item}>{item}</option>{/each}</select
          ></label
        >{:else}<p>
          <span>Host</span><strong title={choice.host}>{choice.host}</strong><small
            >The only source reporting.</small
          >
        </p>{/if}
      {#if choice.metrics.length > 1}<label
          >Metric<select
            value={series?.id ?? ''}
            onchange={(event) => {
              selected = event.currentTarget.value;
              point = -1;
            }}
            >{#each choice.metrics as item (item.id)}<option value={item.id}
                >{seriesLabel(item)}</option
              >{/each}</select
          ></label
        >{:else if series}<p>
          <span>Metric</span><strong>{seriesLabel(series)}</strong><small
            >The only metric this host reports.</small
          >
        </p>{/if}
    </div>
    {#if series}<p>
        <strong>{formatted(current?.value)}</strong> · {current
          ? new Date(current.time * 1000).toLocaleTimeString()
          : 'No sample'}
      </p>
      <svg
        viewBox="-4 -4 608 128"
        role="img"
        aria-label={`${seriesLabel(series)} history; minimum ${formatted(known.length ? Math.min(...known) : null)}, maximum ${formatted(known.length ? Math.max(...known) : null)}`}
        ><path
          d={chartPath(series.points)}
          fill="none"
          stroke={presentation?.memory[series.metric.id.split('.').at(-1) ?? '']?.hex ?? '#22d3ee'}
          stroke-width="2"
        /></svg
      >
      <label
        >Inspect sample<input
          type="range"
          min="0"
          max={Math.max(0, series.points.length - 1)}
          value={point < 0 ? series.points.length - 1 : point}
          oninput={(event) => (point = Number(event.currentTarget.value))}
        /></label
      >
      <small
        >{series.points.length} samples (maximum 240), source: {series.metric.source}. Missing
        values break the line. Separate hosts and counters are never added together.</small
      >
      <details>
        <summary>Sample table</summary>
        <div class="sample-table">
          <table>
            <thead><tr><th>Time</th><th>{seriesLabel(series)}</th></tr></thead><tbody
              >{#each [...series.points].reverse() as sample}<tr
                  ><td>{new Date(sample.time * 1000).toLocaleTimeString()}</td><td
                    >{formatted(sample.value)}</td
                  ></tr
                >{/each}</tbody
            >
          </table>
        </div>
      </details>
      <button onclick={() => exportJSON(series, 'metric-history.json')}>Export history</button>
    {/if}
  {:else}<p class="muted">
      No timestamped metric samples reported. History begins when the driver reports a valid sample.
    </p>{/if}
</section>

<style>
  .metric-chart {
    display: grid;
    gap: 8px;
    margin: 14px 0;
  }
  .metric-chart svg {
    width: 100%;
    background: #111318;
    min-height: 80px;
  }
  .metric-chart small {
    font-size: 10px;
  }
  .chart-choice {
    display: grid;
    gap: 8px;
  }
  .chart-choice p {
    display: grid;
    gap: 2px;
    margin: 0;
    font-size: 11px;
  }
  .chart-choice p strong {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .chart-choice p span {
    color: var(--muted);
  }
  .sample-table {
    max-height: 200px;
    overflow: auto;
  }
  table {
    width: 100%;
    font-size: 11px;
  }
  th {
    text-align: left;
  }
</style>
