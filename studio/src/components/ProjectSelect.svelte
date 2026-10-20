<script lang="ts">
  import type { StudioClient } from '../api';
  import type { StoredRecord } from '../contracts';
  import { projectCatalog } from '../projectCatalog';
  let {
    client,
    summaries,
    value,
    onchange,
    label = 'Project',
    compact = false,
  }: {
    client: StudioClient;
    summaries: StoredRecord[];
    value: string;
    onchange: (id: string) => void;
    label?: string;
    compact?: boolean;
  } = $props();
  let projects = $state<StoredRecord[]>([]),
    loading = $state(false),
    error = $state('');
  const revision = $derived(summaries.map((row) => `${row.id}:${row.revision}`).join(','));
  $effect(() => {
    const api = client,
      version = revision;
    let current = true;
    loading = true;
    error = '';
    projects = [];
    void projectCatalog(api, version)
      .then((rows) => {
        if (current) projects = rows;
      })
      .catch((cause) => {
        if (current)
          error = `Project catalog unavailable: ${cause instanceof Error ? cause.message : String(cause)}`;
      })
      .finally(() => {
        if (current) loading = false;
      });
    return () => {
      current = false;
    };
  });
</script>

<label>
  <span class:sr-only={compact}>{label}</span>
  <select
    {value}
    title={error || undefined}
    aria-busy={loading}
    onchange={(event) => onchange(event.currentTarget.value)}
  >
    <option value="">All projects</option>
    {#if value && !projects.some((project) => project.id === value)}
      <option {value}
        >{summaries.find((project) => project.id === value)?.data.name ?? value} · {loading
          ? 'loading'
          : 'not in catalog'}</option
      >
    {/if}
    {#each projects as project}<option value={project.id}>{project.data.name ?? project.id}</option
      >{/each}
  </select>
</label>
{#if error}<small role="status" class:sr-only={compact}>{error}</small>{/if}
