<script lang="ts">
  import { onMount } from 'svelte';
  import type { StudioClient } from '../api';
  import type { RecordData, StoredRecord } from '../contracts';
  import { loadRecordCatalog } from '../recordCatalog';
  import { recordProjects } from '../recordPersistence';
  let {
    client,
    projectId,
    gridId,
    disabled = false,
    onload,
    onresume,
  }: {
    client: StudioClient;
    projectId: string;
    gridId?: string;
    disabled?: boolean;
    onload: (data: RecordData) => Promise<void>;
    onresume: (id: string) => Promise<void>;
  } = $props();
  let records = $state<StoredRecord[]>([]),
    selected = $state(''),
    name = $state(''),
    busy = $state(false),
    error = $state(''),
    notice = $state('');
  const visible = $derived(
    records.filter((row) => !projectId || recordProjects(row.data).includes(projectId)),
  );
  const record = $derived(records.find((row) => row.id === selected));
  async function refresh() {
    records = [];
    records = await loadRecordCatalog(client, 'experiments');
  }
  async function run(action: () => Promise<void>) {
    busy = true;
    error = '';
    try {
      await action();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      busy = false;
    }
  }
  async function save() {
    if (!gridId || !name.trim()) return;
    const value = await client.request<StoredRecord>(
      `/records/experiments/${crypto.randomUUID()}`,
      {
        revision: 0,
        data: { name: name.trim(), grid_id: gridId, project_ids: projectId ? [projectId] : [] },
      },
      'PUT',
    );
    await refresh();
    selected = value.id;
    name = '';
    notice = 'Saved the grid definition and observed outcomes in the project library.';
  }
  onMount(() => {
    selected = sessionStorage.getItem('dgxm.studio.experiment.selection') ?? '';
    sessionStorage.removeItem('dgxm.studio.experiment.selection');
    void run(refresh);
  });
</script>

<section class="surface" aria-label="Saved grid experiments">
  <h2>Project experiments</h2>
  <label
    >Saved definition<select bind:value={selected}
      ><option value="">Choose an experiment</option>{#each visible as row}<option value={row.id}
          >{row.data.name ?? row.id}</option
        >{/each}</select
    ></label
  >
  <button
    disabled={busy || disabled || !record?.data.grid_request}
    onclick={() =>
      void run(async () => {
        if (record) await onload(record.data);
      })}>Load definition for a new preview</button
  >
  <button
    disabled={busy || disabled || typeof record?.data.grid_id !== 'string' || !record?.data.grid_id}
    onclick={() =>
      void run(async () => {
        if (record) await onresume(String(record.data.grid_id));
      })}>Resume linked local run</button
  >
  <button disabled={busy} onclick={() => void run(refresh)}>Refresh experiments</button>
  <p class="muted">
    Loading replaces the editable draft and axes. A new preview is required before submission.
    Imported source runs may be unavailable; their frozen definitions remain editable.
  </p>
  {#if gridId}<label>Name this grid<input bind:value={name} /></label><button
      disabled={busy || disabled || !name.trim()}
      onclick={() => void run(save)}>Save grid to project library</button
    >{/if}
  {#if error}<p role="alert">{error}</p>{/if}{#if notice}<p role="status">{notice}</p>{/if}
</section>
