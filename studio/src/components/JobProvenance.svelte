<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Job } from '../contracts';
  import { fullJob } from '../jobProvenance';
  import { exportJSON } from '../files';
  import { keyedRecord } from '../keyedRecord.svelte';
  import { isActive } from '../queueRows';
  import RuntimeEvidence from './RuntimeEvidence.svelte';
  let { client, summary }: { client: StudioClient; summary: Job } = $props();
  // Polls replace the summary row every tick; only a new job id clears the loaded provenance.
  const record = keyedRecord<Job>(
    (id) => fullJob(client, id),
    () => ({
      id: summary.id,
      revision: `${summary.state}:${summary.updated_at ?? ''}`,
      live: isActive(summary),
    }),
  );
</script>

<button disabled={record.loading} onclick={() => record.refresh()}>Refresh full provenance</button>
{#if record.error}<p role="alert">{record.error}</p>{/if}
{#if record.loading && !record.value}<p role="status">Loading full job provenance…</p>{/if}
{#if record.value}
  {@const job = record.value}
  <button onclick={() => exportJSON(job, `${job.id}.json`)}>Export provenance</button>
  <pre>{JSON.stringify(job, null, 2)}</pre>
  <RuntimeEvidence {client} {job} />
{/if}
