<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Artifact, Telemetry, ReuseArtifact } from '../contracts';
  import { formatTime } from '../model';
  import TelemetryCharts from '../components/TelemetryCharts.svelte';
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
    showTelemetry,
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
    activeCount: number;
    showTelemetry: boolean;
    onselect: (artifact: Artifact) => void;
    oncompare: (artifact: Artifact | null) => void;
    oncollapse: () => void;
    navigate: (view: string) => void;
    reuse: ReuseArtifact;
  } = $props();
</script>

<div class="workspace-center">
  <Viewer
    artifact={selected}
    compare={comparison}
    {client}
    onreuse={reuse}
    onselect={(a) => oncompare(a)}
  />
  <div class="recent-strip">
    <div>
      <span class="eyebrow">RECENT OUTPUTS</span><button onclick={() => navigate('gallery')}
        >View gallery →</button
      >
    </div>
    <div class="recent-list">
      {#each (artifacts ?? []).slice(0, 8) as artifact}<button
          class:selected={selected?.id === artifact.id}
          onclick={() => onselect(artifact)}
          ><span aria-hidden="true">{artifact.kind === 'video' ? '▷' : '▧'}</span><span
            >{artifact.filename ?? artifact.id}</span
          ></button
        >{/each}{#if !(artifacts ?? []).length}<span class="muted"
          >Completed outputs will appear here.</span
        >{/if}
    </div>
  </div>
</div>
{#if showTelemetry}<aside class="telemetry-panel">
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
      <span>Active jobs</span><strong>{activeCount}</strong><span>Driver</span><strong
        class="text-small">{driverState ?? 'unknown'}</strong
      >
    </div>
    <TelemetryCharts {history} {presentation} />
    <details open>
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
