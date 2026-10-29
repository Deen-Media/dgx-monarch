<script lang="ts">
  import TelemetryCharts from '../components/TelemetryCharts.svelte';
  import HostHealth from '../components/HostHealth.svelte';
  import type { MetricSeries, Presentation } from '../telemetryHistory';
  import type { Telemetry, ChangeDraft, InspectWorkflow, Job, WorkflowEntry } from '../contracts';
  import type { Draft } from '../model';
  import type { StudioClient } from '../api';
  import RuntimeEvidence from '../components/RuntimeEvidence.svelte';
  import { loadWorkflow } from '../workflowChoice';
  import { fleetNextStep, fleetPlan, fleetPlanText, hasFleetSampler } from '../fleetSetup';
  let {
    telemetry,
    history,
    presentation,
    draft,
    workflows = [],
    changeDraft,
    inspect,
    navigate,
    setNotice,
    client,
    jobs,
  }: {
    telemetry: Telemetry;
    history: MetricSeries[];
    presentation?: Presentation;
    draft: Draft;
    workflows?: WorkflowEntry[];
    changeDraft: ChangeDraft;
    inspect: InspectWorkflow;
    navigate: (view: string) => void;
    setNotice: (message: string) => void;
    client: StudioClient;
    jobs: Job[];
  } = $props();
  let selected = $state(''),
    loading = $state(false);
  const fleetJobs = $derived(
    jobs.filter(
      (job) =>
        job.submission?.requested?.operation === 'fleet' ||
        Object.values(job.monarch?.renders ?? {}).some((render: any) => render.fleet_index != null),
    ),
  );
  const selectedJob = $derived(fleetJobs.find((job) => job.id === selected) ?? fleetJobs[0]);
  const plan = $derived(fleetPlan(draft, workflows));
  /** Load the Fleet workflow unless the draft holds one, label the draft Fleet, then open Create. */
  async function setUp() {
    if (loading) return;
    const chosen = plan,
      previous = draft.operation;
    loading = true;
    try {
      if (previous !== 'fleet') changeDraft({ operation: 'fleet' });
      if (chosen.kind === 'load') {
        await loadWorkflow(chosen.entry, inspect);
        // A refused or superseded load leaves its error on this page and the draft as it was.
        if (!hasFleetSampler(draft.workflow)) {
          if (previous !== 'fleet' && draft.operation === 'fleet')
            changeDraft({ operation: previous });
          return;
        }
      }
      navigate('workspace');
      setNotice(fleetNextStep(chosen));
    } finally {
      loading = false;
    }
  }
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">WORKERS & EXECUTION</span>
      <h1>Fleet</h1>
    </div>
  </div>
  <div class="fleet-intro">
    <div class="surface">
      <h2>What Fleet does</h2>
      <p>
        Fleet renders many separate prompts at the same time. Each line of the prompts input becomes
        its own job, and the jobs go to the GPUs in turn, one job per GPU at a time. Every job uses
        the same model, size, steps, CFG, sampler and scheduler. Line 1 uses seed noise_seed, line 2
        noise_seed + 1 and so on, and the outputs come back in prompt order.
      </p>
    </div>
    <div class="surface">
      <h2>When to use it</h2>
      <p>
        Use Fleet when you want more pictures per hour, such as ten prompts on one model, rather
        than one picture sooner. With two GPUs, two prompts render at once; on one GPU Fleet gains
        nothing. To make one picture faster, use a normal Monarch workflow, which can split one
        render across the GPUs.
      </p>
    </div>
    <div class="surface">
      <h2>First step</h2>
      <p>{fleetPlanText(plan)}</p>
      <button class="primary" disabled={loading} onclick={setUp}
        >{loading ? 'Loading the Fleet workflow…' : 'Set up Fleet in Create'}</button
      >
    </div>
  </div>
  <p class="muted">
    A run takes up to 1,024 jobs. The negative prompt is one value for every job or exactly one per
    job. A run in flight is one ComfyUI task: no single job in it can be paused, reordered or
    retried, so Queue retries the whole run.
  </p>
  <h2>Hosts</h2>
  <HostHealth {telemetry} {presentation} />
  <TelemetryCharts {history} {presentation} />
  <h2>Durable Fleet execution</h2>
  {#if fleetJobs.length > 1}<label
      >Graph task<select
        value={selectedJob?.id ?? ''}
        onchange={(event) => (selected = event.currentTarget.value)}
        >{#each fleetJobs as job}<option value={job.id}>{job.id} · {job.state}</option
          >{/each}</select
      ></label
    >{:else if selectedJob}<p>
      Graph task <span class="mono">{selectedJob.id}</span> · {selectedJob.state}, the only one
      recorded.
    </p>{/if}
  {#if selectedJob}<RuntimeEvidence {client} job={selectedJob} />{:else}<p>
      No recorded Fleet graph task has been observed.
    </p>{/if}
  <details>
    <summary>Reported Fleet state</summary>
    <pre>{JSON.stringify(
        telemetry?.sample?.render?.fleet ??
          telemetry?.sample?.fleet ?? {
            state: 'Unknown: no Fleet sample reported.',
          },
        null,
        2,
      )}</pre>
  </details>
  <details>
    <summary>Worker and topology sample</summary>
    <pre>{JSON.stringify(
        telemetry?.sample?.workers ?? telemetry?.sample?.hosts ?? telemetry ?? {},
        null,
        2,
      )}</pre>
  </details>
  <p class="muted">
    A missing end timestamp does not establish success or failure. Durable queue records carry job
    outcomes.
  </p>
</section>

<style>
  .fleet-intro {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
  }
  .fleet-intro .surface {
    display: grid;
    align-content: start;
  }
  .fleet-intro .primary {
    justify-self: start;
  }
</style>
