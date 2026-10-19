<script lang="ts">
  import type { StudioClient } from '../api';
  import type { OperationRecord } from '../contracts';
  import { exportJSON } from '../files';
  let { client, operation }: { client: StudioClient; operation: OperationRecord } = $props();
  let receipt = $state<OperationRecord | null>(null),
    error = $state(''),
    loading = $state(false);
  let sequence = 0;
  async function load() {
    const ticket = ++sequence,
      id = operation.id;
    loading = true;
    error = '';
    try {
      const result = await client.request<OperationRecord>(`/operations/${encodeURIComponent(id)}`);
      if (result.id !== id || result.summary)
        throw new Error('The full operation receipt was not returned.');
      if (ticket === sequence && operation.id === id) receipt = result;
    } catch (cause) {
      if (ticket === sequence) error = String(cause);
    } finally {
      if (ticket === sequence) loading = false;
    }
  }
  $effect(() => {
    const identity = `${operation.id}:${operation.state}:${operation.updated_at ?? ''}`;
    void identity;
    receipt = null;
    void load();
    return () => {
      sequence++;
    };
  });
</script>

<button disabled={loading} onclick={() => void load()}>Refresh full receipt</button>
{#if error}<p role="alert">{error}</p>{/if}
{#if loading}<p role="status">Loading operation receipt…</p>{/if}
{#if receipt}
  <button
    onclick={() => {
      if (receipt) exportJSON(receipt, `operation-${receipt.id}.json`);
    }}>Export receipt</button
  >
  <pre>{JSON.stringify(receipt, null, 2)}</pre>
{/if}
