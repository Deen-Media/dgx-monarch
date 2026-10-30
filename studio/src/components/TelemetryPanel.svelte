<script lang="ts">
  import type { Telemetry } from '../contracts';
  import type { FollowedJob } from '../jobTracker.svelte';
  import type { MetricSeries, Presentation } from '../telemetryHistory';
  import { formatTime } from '../model';
  import { hostSummary } from '../hostHealth';
  import { saveLayout, telemetryDetailsOpen } from '../initialWorkspace';
  import HostHealth from './HostHealth.svelte';
  import TelemetryCharts from './TelemetryCharts.svelte';
  let {
    telemetry,
    history,
    presentation,
    driverState,
    connectionState,
    serverTime = null,
    activeCount,
    job = null,
  }: {
    telemetry: Telemetry;
    history: MetricSeries[];
    presentation?: Presentation;
    driverState?: string;
    connectionState: string;
    /** The companion's time at the newest snapshot, which ages retained telemetry. */
    serverTime?: number | null;
    /** Active or unresolved jobs, as Queue counts them. */
    activeCount: number;
    job?: FollowedJob | null;
  } = $props();
  const live = $derived(connectionState === 'connected');
  // The driver's readiness report in one line, beside the jobs and the driver's state.
  const readiness = $derived(
    hostSummary(telemetry, presentation, { now: serverTime, live }).overall,
  );
  // The jobs, readiness and both hosts stay in view. The readiness rows, metric history, sample
  // inspection, export and raw sample open on demand, and the browser keeps that choice with the
  // other layout preferences.
  let detailed = $state(telemetryDetailsOpen());
  function toggleDetails(event: Event & { currentTarget: HTMLDetailsElement }) {
    if (event.target !== event.currentTarget) return;
    detailed = event.currentTarget.open;
    saveLayout({ telemetryDetails: detailed });
  }
</script>

<aside class="telemetry-panel" id="studio-telemetry">
  <div class="panel-title">
    <h2>Spark telemetry</h2>
  </div>
  <div class="sample-state">
    <span class="status-dot" class:healthy={live}></span>{telemetry?.state === 'connected'
      ? 'Latest sample'
      : telemetry?.observed_at
        ? 'Last known sample'
        : 'No telemetry sample'}<small
      >{telemetry?.observed_at ? formatTime(telemetry.observed_at) : 'Not sampled'} · {telemetry?.state ??
        'unknown'}</small
    >
  </div>
  <div class="telemetry-summary">
    <span>Active or unresolved jobs</span><strong>{activeCount}</strong><span>Driver</span><strong
      class="text-small">{driverState ?? 'unknown'}</strong
    >{#if readiness}<span>Readiness</span><strong class="text-small"
        >{#if readiness.mark.icon}<span
            class="status-icon"
            style:color={readiness.mark.hex}
            aria-hidden="true">{readiness.mark.icon}</span
          >{/if}{readiness.state}</strong
      >{#if readiness.layers.length}<small class="readiness-layers"
          >{readiness.layers.join(', ')}</small
        >{/if}{#if readiness.stale}<small class="readiness-layers stale">{readiness.stale}</small
        >{/if}{/if}
    <!-- This tab's job, never the driver's global render block. -->
    <span>{job && !job.own ? 'Newest active job' : "This tab's job"}</span><strong
      class="text-small">{job?.headline ?? 'None'}</strong
    >{#if job?.detail}<small class="job-stage">{job.detail}</small>{/if}
  </div>
  <HostHealth {telemetry} {presentation} now={serverTime} {live} heading="Hosts" part="hosts" />
  <details class="telemetry-details" open={detailed} ontoggle={toggleDetails}>
    <summary>Detailed telemetry</summary>
    <HostHealth {telemetry} {presentation} now={serverTime} {live} part="readiness" />
    <TelemetryCharts {history} {presentation} />
    <details>
      <summary>Authoritative host sample</summary>
      <pre>{JSON.stringify(telemetry ?? {}, null, 2)}</pre>
    </details>
    <div class="memory-legend">
      <span><i style="background:#22d3ee"></i>GPU</span><span
        ><i style="background:#34d399"></i>Slab</span
      ><span><i style="background:#fb923c"></i>Retained pool</span><span
        ><i style="background:#c084fc"></i>Anonymous</span
      ><span><i style="background:#64748b"></i>Other</span><span
        ><i style="background:#facc15"></i>Reclaimable cache</span
      >
    </div>
    <small
      >Unknown counters stay unknown. Overlapping memory categories are not added together.</small
    >
  </details>
</aside>

<style>
  .job-stage,
  .readiness-layers {
    grid-column: 1 / -1;
    text-align: right;
    font-size: 10px;
    color: var(--muted);
  }
  .readiness-layers {
    margin-top: -6px;
  }
  .readiness-layers.stale {
    color: var(--status-degraded);
  }
  .status-icon {
    margin-right: 4px;
  }
  /* One line opens the detail below the summaries; the open state survives a reload. */
  .telemetry-details {
    margin-top: 6px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }
  .telemetry-details > summary {
    font-size: 11px;
    color: #aab4c2;
    cursor: pointer;
  }
  .telemetry-details > small {
    display: block;
    font-size: 9px;
  }
  /* The two-column panel of narrower screens sets the summary beside the taller host list. */
  @media (max-width: 1050px) {
    .telemetry-summary {
      align-self: start;
    }
  }
</style>
