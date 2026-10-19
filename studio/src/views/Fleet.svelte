<script lang="ts">
  import type { Telemetry, ChangeDraft } from '../contracts';
  let {
    telemetry,
    changeDraft,
    navigate,
  }: { telemetry: Telemetry; changeDraft: ChangeDraft; navigate: (view: string) => void } =
    $props();
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
