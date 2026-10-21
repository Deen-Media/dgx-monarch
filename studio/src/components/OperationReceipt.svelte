<script lang="ts">
  import type { StudioClient } from '../api';
  import type { OperationRecord } from '../contracts';
  import { exportJSON } from '../files';
  import { keyedRecord } from '../keyedRecord.svelte';
  let { client, operation }: { client: StudioClient; operation: OperationRecord } = $props();
  async function read(id: string) {
    const result = await client.request<OperationRecord>(`/operations/${encodeURIComponent(id)}`);
    if (result.id !== id || result.summary)
      throw new Error('The full operation receipt was not returned.');
    return result;
  }
  // Polls replace the summary row every tick; only a new operation id clears the receipt.
  const receipt = keyedRecord<OperationRecord>(read, () => ({
    id: operation.id,
    revision: `${operation.state ?? ''}:${operation.updated_at ?? ''}`,
    live: operation.state === 'applying',
  }));
</script>

<button disabled={receipt.loading} onclick={() => receipt.refresh()}>Refresh full receipt</button>
{#if receipt.error}<p role="alert">{receipt.error}</p>{/if}
{#if receipt.loading && !receipt.value}<p role="status">Loading operation receipt…</p>{/if}
{#if receipt.value}
  {@const value = receipt.value}
  <button onclick={() => exportJSON(value, `operation-${value.id}.json`)}>Export receipt</button>
  <pre>{JSON.stringify(value, null, 2)}</pre>
{/if}
