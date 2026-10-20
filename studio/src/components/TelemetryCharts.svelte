<script lang="ts">
  import type { MetricSeries, Presentation } from '../telemetryHistory';
  import { chartPath } from '../telemetryHistory';
  import { exportJSON } from '../files';
  let { history, presentation }: { history: MetricSeries[]; presentation?: Presentation } =
    $props();
  let selected = $state(''),
    point = $state(-1);
  const series = $derived(history.find((series) => series.id === selected) ?? history[0]);
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
  {#if history.length}<label
      >Host and metric<select
        value={series?.id ?? ''}
        onchange={(event) => {
          selected = event.currentTarget.value;
          point = -1;
        }}
        >{#each history as item}<option value={item.id}>{item.host} · {item.metric.name}</option
          >{/each}</select
      ></label
    >
    {#if series}<p>
        <strong>{formatted(current?.value)}</strong> · {current
          ? new Date(current.time * 1000).toLocaleTimeString()
          : 'No sample'}
      </p>
      <svg
        viewBox="-4 -4 608 128"
        role="img"
        aria-label={`${series.metric.name} history; minimum ${formatted(known.length ? Math.min(...known) : null)}, maximum ${formatted(known.length ? Math.max(...known) : null)}`}
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
            <thead><tr><th>Time</th><th>{series.metric.name}</th></tr></thead><tbody
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
