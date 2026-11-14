<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Artifact, Job, Seeds, Telemetry, ReuseArtifact } from '../contracts';
  import type { FollowedJob } from '../jobTracker.svelte';
  import { isJobOutput } from '../jobFollow';
  import { recentStep } from '../recentOutputs';
  import TelemetryPanel from '../components/TelemetryPanel.svelte';
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
    serverTime = null,
    activeCount,
    job = null,
    working = false,
    showTelemetry,
    jobs,
    onuseseed,
    onselect,
    oncompare,
    navigate,
    reuse,
    onconsent,
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
    /** The companion's time at the newest snapshot, which ages retained telemetry. */
    serverTime?: number | null;
    /** Active or unresolved jobs, as Queue counts them. */
    activeCount: number;
    job?: FollowedJob | null;
    working?: boolean;
    showTelemetry: boolean;
    jobs: Job[];
    onuseseed: (seeds: Seeds) => void;
    onselect: (artifact: Artifact) => void;
    oncompare: (artifact: Artifact | null) => void;
    navigate: (view: string) => void;
    reuse: ReuseArtifact;
    onconsent?: () => void;
  } = $props();
  // Uploads and masks have no job, and a live preview is a running job's approximate frame.
  const outputs = $derived((artifacts ?? []).filter(isJobOutput).slice(0, 8));
  /** Left and Right on a tile select the previous or next output, as a click does, and take focus. */
  function stepRecent(event: KeyboardEvent, index: number) {
    const target = recentStep(event, index, outputs.length);
    if (target === null) return;
    event.preventDefault();
    onselect(outputs[target]);
    const tiles = (event.currentTarget as HTMLElement).parentElement?.children;
    (tiles?.[target] as HTMLElement | undefined)?.focus();
  }
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
    {onconsent}
  />
  <div class="recent-strip">
    <div>
      <span class="eyebrow">RECENT OUTPUTS</span><button onclick={() => navigate('gallery')}
        >View gallery →</button
      >
    </div>
    <div class="recent-list">
      {#each outputs as artifact, index (artifact.id)}<button
          class:selected={selected?.id === artifact.id}
          aria-pressed={selected?.id === artifact.id}
          title={artifact.filename ?? artifact.id}
          onclick={() => onselect(artifact)}
          onkeydown={(event) => stepRecent(event, index)}
          ><span class="recent-thumb" aria-hidden="true"><Thumbnail {artifact} {client} /></span
          ><span>{artifact.intermediate ? 'Preview · ' : ''}{artifact.filename ?? artifact.id}</span
          ></button
        >{/each}{#if !outputs.length}<span class="muted">Completed outputs will appear here.</span
        >{/if}
    </div>
  </div>
</div>
{#if showTelemetry}<TelemetryPanel
    {telemetry}
    {history}
    {presentation}
    {driverState}
    {connectionState}
    {serverTime}
    {activeCount}
    {job}
  />{/if}
