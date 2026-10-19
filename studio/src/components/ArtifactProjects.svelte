<script lang="ts">
  import { onMount } from 'svelte';
  import { loadRecordCatalog } from '../recordCatalog';
  import type { StudioClient } from '../api';
  import type { Artifact, StoredRecord } from '../contracts';
  let {
    client,
    artifact,
    onsave,
  }: { client: StudioClient; artifact: Artifact; onsave: (value: Artifact) => void } = $props();
  let projects = $state<StoredRecord[]>([]),
    selected = $state<string[]>([]),
    error = $state(''),
    busy = $state(false);
  $effect(() => {
    selected = Array.isArray(artifact.project_ids)
      ? (artifact.project_ids as string[])
      : artifact.project_id
        ? [artifact.project_id]
        : [];
  });
  onMount(() => {
    void loadRecordCatalog(client, 'projects')
      .then((result) => (projects = result))
      .catch((cause) => (error = String(cause)));
  });
  async function save() {
    busy = true;
    error = '';
    try {
      onsave(
        await client.request<Artifact>(
          `/artifacts/${encodeURIComponent(artifact.id)}/projects`,
          { project_ids: $state.snapshot(selected), revision: artifact.membership_revision ?? 0 },
          'PUT',
        ),
      );
    } catch (cause) {
      error = String(cause);
    } finally {
      busy = false;
    }
  }
</script>

<label
  >Output project membership<select multiple bind:value={selected}
    >{#each projects as project}<option value={project.id}>{project.data.name ?? project.id}</option
      >{/each}</select
  ></label
>
<small>Membership does not move the output or change its frozen submission.</small>
<button disabled={busy} onclick={save}>Save output membership</button>
{#if error}<p role="alert">{error}</p>{/if}
