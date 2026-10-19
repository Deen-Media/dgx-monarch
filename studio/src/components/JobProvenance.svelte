<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Job } from '../contracts';
  import { fullJob } from '../jobProvenance';
  import { exportJSON } from '../files';
  import RuntimeEvidence from './RuntimeEvidence.svelte';
  let { client, summary }: { client: StudioClient; summary: Job } = $props();
  let job = $state<Job | null>(null),
    error = $state(''),
    loading = $state(false);
  let sequence = 0;
  async function load() {
    const ticket = ++sequence,
      id = summary.id;
    loading = true;
    error = '';
    try {
      const result = await fullJob(client, id);
      if (ticket === sequence && summary.id === id) job = result;
    } catch (cause) {
      if (ticket === sequence) error = String(cause);
    } finally {
      if (ticket === sequence) loading = false;
    }
  }
  $effect(() => {
    const identity = `${summary.id}:${summary.state}:${summary.updated_at ?? ''}`;
    void identity;
    job = null;
    void load();
    return () => {
      sequence++;
    };
  });
</script>

<button disabled={loading} onclick={() => void load()}>Refresh full provenance</button>
{#if error}<p role="alert">{error}</p>{/if}
{#if loading}<p role="status">Loading full job provenance…</p>{/if}
{#if job}
  <button
    onclick={() => {
      if (job) exportJSON(job, `${job.id}.json`);
    }}>Export provenance</button
  >
  <pre>{JSON.stringify(job, null, 2)}</pre>
  <RuntimeEvidence {client} {job} />
{/if}
