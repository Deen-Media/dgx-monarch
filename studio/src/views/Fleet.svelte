<script lang="ts">
  import TelemetryCharts from '../components/TelemetryCharts.svelte';
  import type { MetricSeries, Presentation } from '../telemetryHistory';
  import type { Telemetry, ChangeDraft, Job } from '../contracts';
  import type { StudioClient } from '../api';
  import RuntimeEvidence from '../components/RuntimeEvidence.svelte';
  let {
    telemetry,
    history,
    presentation,
    changeDraft,
    navigate,
    client,
    jobs,
  }: {
    telemetry: Telemetry;
    history: MetricSeries[];
    presentation?: Presentation;
    changeDraft: ChangeDraft;
    navigate: (view: string) => void;
    client: StudioClient;
    jobs: Job[];
  } = $props();
  let selected = $state('');
  const fleetJobs = $derived(
    jobs.filter(
      (job) =>
        job.submission?.requested?.operation === 'fleet' ||
        Object.values(job.monarch?.renders ?? {}).some((render: any) => render.fleet_index != null),
    ),
  );
  const selectedJob = $derived(fleetJobs.find((job) => job.id === selected) ?? fleetJobs[0]);
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">WORKERS & EXECUTION</span>
      <h1>Fleet</h1>
    </div>
    <button
      onclick={() => {
        changeDraft({ operation: 'fleet' });
        navigate('workspace');
      }}>Configure Fleet workflow ↗</button
    >
  </div>
  <div class="surface">
    <h3>Independent sampling across ranks</h3>
    <p>
      DGXMonarchFleetKSampler distributes world-1 jobs round-robin, with one job per rank per wave.
      Jobs share model, latent, steps, CFG, sampler and scheduler. Job i uses base seed + i; outputs
      retain input order.
    </p>
    <p class="muted">
      Up to 1,024 jobs. Negative prompts must be one broadcast value or one per job. Individual
      in-flight pause, reorder and retry are unavailable at the graph-task boundary.
    </p>
  </div>
  <TelemetryCharts {history} {presentation} />
  <h2>Durable Fleet execution</h2>
  <label
    >Graph task<select bind:value={selected}
      >{#each fleetJobs as job}<option value={job.id}>{job.id} · {job.state}</option>{/each}</select
    ></label
  >
  {#if selectedJob}<RuntimeEvidence {client} job={selectedJob} />{:else}<p>
      No recorded Fleet graph task has been observed.
    </p>{/if}
  <h2>Reported Fleet state</h2>
  <pre>{JSON.stringify(
      telemetry?.sample?.render?.fleet ??
        telemetry?.sample?.fleet ?? {
          state: 'Unknown: no Fleet sample reported.',
        },
      null,
      2,
    )}</pre>
  <h2>Worker and topology sample</h2>
  <pre>{JSON.stringify(
      telemetry?.sample?.workers ?? telemetry?.sample?.hosts ?? telemetry ?? {},
      null,
      2,
    )}</pre>
  <p class="muted">
    A missing end timestamp does not establish success or failure. Durable queue records carry job
    outcomes.
  </p>
</section>
