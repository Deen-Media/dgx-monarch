<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Job } from '../contracts';
  import { exactJob } from '../jobProvenance';
  import { exportText } from '../files';
  import { writeExactJSON } from '../exactJson';
  import { keyedRecord } from '../keyedRecord.svelte';
  import { isActive } from '../queueRows';
  import RuntimeEvidence from './RuntimeEvidence.svelte';
  let { client, summary }: { client: StudioClient; summary: Job } = $props();
  // Polls replace the summary row every tick; only a new job id clears the loaded provenance.
  // The exact record shows and exports a seed above 2**53 - 1 with every digit.
  const record = keyedRecord<Job>(
    (id) => exactJob(client, id),
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
  {@const text = writeExactJSON(job)}
  <button onclick={() => exportText(text, `${job.id}.json`)}>Export provenance</button>
  <pre>{text}</pre>
  <RuntimeEvidence {client} {job} />
{/if}
