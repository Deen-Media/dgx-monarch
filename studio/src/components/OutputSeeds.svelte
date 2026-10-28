<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Job, Seeds } from '../contracts';
  import { jobSummary } from '../jobProvenance';
  import { keyedRecord } from '../keyedRecord.svelte';
  import { recordedSeeds, seedText } from '../recordedSettings';
  let {
    client,
    jobId,
    job,
    onuseseed,
  }: {
    client: StudioClient;
    jobId: string;
    /** The snapshot row of the output's job, when the snapshot holds it. */
    job?: Job;
    onuseseed?: (seeds: Seeds) => void;
  } = $props();
  const summarized = $derived(recordedSeeds(job));
  // A job outside the snapshot loads its own summary once: submissions are immutable, so it
  // never needs a reload, and unlike a full record it keeps a large seed's exact digits.
  const record = keyedRecord<Job>(
    (id) => jobSummary(client, id),
    () => ({ id: summarized ? '' : jobId }),
  );
  const seeds = $derived(summarized ?? recordedSeeds(record.value));
  const count = $derived(Object.keys(seeds ?? {}).length);
</script>

<span class="output-seeds">
  {#if count}<span class="mono">{seedText(seeds)}</span>{#if onuseseed}<button
        onclick={() => onuseseed(seeds!)}
        >{count === 1 ? 'Use this seed' : 'Use these seeds'}</button
      >{/if}
  {:else if seeds}<span class="muted">No seed input</span>
  {:else if record.error}<span class="muted">Seed unavailable: {record.error}</span>
  {:else}<span class="muted">Reading the recorded seed…</span>{/if}
</span>
