<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Artifact, Job, Seeds, Telemetry, ReuseArtifact } from '../contracts';
  import type { FollowedJob } from '../jobTracker.svelte';
  import { isJobOutput } from '../jobFollow';
  import { formatTime } from '../model';
  import TelemetryCharts from '../components/TelemetryCharts.svelte';
  import Thumbnail from '../components/Thumbnail.svelte';
  import type { MetricSeries, Presentation } from '../telemetryHistory';
  import Viewer from '../components/Viewer.svelte';
  let {
    client,
    selected,
    comparison,
    artifacts,
    telemetry,
    history,
    presentation,
    driverState,
    connectionState,
    activeCount,
    job = null,
    working = false,
    showTelemetry,
    jobs,
    onuseseed,
    onselect,
    oncompare,
    oncollapse,
    navigate,
    reuse,
  }: {
    client: StudioClient;
    selected: Artifact | null;
    comparison: Artifact | null;
    artifacts: Artifact[];
    telemetry: Telemetry;
    history: MetricSeries[];
    presentation?: Presentation;
    driverState?: string;
    connectionState: string;
    /** Active or unresolved jobs, as Queue counts them. */
    activeCount: number;
    job?: FollowedJob | null;
    working?: boolean;
    showTelemetry: boolean;
    jobs: Job[];
    onuseseed: (seeds: Seeds) => void;
    onselect: (artifact: Artifact) => void;
    oncompare: (artifact: Artifact | null) => void;
    oncollapse: () => void;
    navigate: (view: string) => void;
    reuse: ReuseArtifact;
  } = $props();
  // Uploads and masks have no job, and a live preview is a running job's approximate frame.
  const outputs = $derived((artifacts ?? []).filter(isJobOutput).slice(0, 8));
</script>

<div class="workspace-center">
  <Viewer
    {job}
    {working}
    artifact={selected}
    compare={comparison}
    {client}
    {jobs}
    {onuseseed}
    onreuse={reuse}
    onselect={(a) => oncompare(a)}
    onqueue={() => navigate('queue')}
  />
  <div class="recent-strip">
    <div>
      <span class="eyebrow">RECENT OUTPUTS</span><button onclick={() => navigate('gallery')}
        >View gallery →</button
      >
    </div>
    <div class="recent-list">
      {#each outputs as artifact (artifact.id)}<button
          class:selected={selected?.id === artifact.id}
          title={artifact.filename ?? artifact.id}
          onclick={() => onselect(artifact)}
          ><span class="recent-thumb" aria-hidden="true"><Thumbnail {artifact} {client} /></span
          ><span>{artifact.intermediate ? 'Preview · ' : ''}{artifact.filename ?? artifact.id}</span
          ></button
        >{/each}{#if !outputs.length}<span class="muted">Completed outputs will appear here.</span
        >{/if}
    </div>
  </div>
</div>
{#if showTelemetry}<aside class="telemetry-panel" id="studio-telemetry">
    <div class="panel-title">
      <h2>Spark telemetry</h2>
      <button onclick={() => oncollapse()} aria-label="Collapse telemetry">×</button>
    </div>
    <div class="sample-state">
      <span class="status-dot" class:healthy={connectionState === 'connected'}
      ></span>{telemetry?.state === 'connected'
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
      >
      <!-- This tab's job, never the driver's global render block. -->
      <span>{job && !job.own ? 'Newest active job' : "This tab's job"}</span><strong
        class="text-small">{job?.headline ?? 'None'}</strong
      >
    </div>
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
  </aside>{/if}
