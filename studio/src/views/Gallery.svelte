<script lang="ts">
  import type { Artifact, Job, ReuseArtifact } from '../contracts';
  import type { StudioClient } from '../api';
  import { fullJob } from '../jobProvenance';
  import { recordedSettings } from '../recordedSettings';
  import { formatTime } from '../model';
  import { exportJSON } from '../files';
  import {
    GALLERY_KINDS,
    GALLERY_STATES,
    galleryEntries,
    galleryQuery,
    hasDimensions,
    virtualRows,
    type GalleryFilter,
    type GalleryEntry,
  } from '../gallery';
  import Thumbnail from '../components/Thumbnail.svelte';
  import ArtifactProjects from '../components/ArtifactProjects.svelte';
  import ProjectSelect from '../components/ProjectSelect.svelte';
  import type { StoredRecord } from '../contracts';
  let {
    projectSummaries,
    jobs,
    projectId,
    selected,
    navigate,
    onselect,
    oncompare,
    reuse,
    setNotice,
    client,
  }: {
    projectSummaries: StoredRecord[];
    jobs: Job[];
    projectId: string;
    selected: Artifact | null;
    client: StudioClient;
    navigate: (view: string) => void;
    onselect: (artifact: Artifact) => void;
    oncompare: (artifact: Artifact) => void;
    reuse: ReuseArtifact;
    setNotice: (message: string) => void;
  } = $props();
  let filter = $state<GalleryFilter>({
    query: '',
    kind: '',
    operation: '',
    model: '',
    status: '',
    project: '',
    from: '',
    until: '',
    sort: 'newest',
  });
  let width = $state(1000),
    height = $state(660),
    top = $state(0),
    currentProject = $state(true);
  let loaded = $state<Artifact[]>([]),
    nextCursor = $state<string | null>(null),
    loading = $state(true),
    loadError = $state('');
  let widthFilter = $state(''),
    heightFilter = $state(''),
    generation = 0;
  let openActions = $state<string[]>([]);
  let scroller: HTMLDivElement;
  const entries = $derived(galleryEntries(loaded, jobs));
  const artifacts = $derived(entries);
  const window = $derived(virtualRows(artifacts.length, width, height, top));
  const query = $derived(
    galleryQuery(filter, {
      projectId: currentProject ? projectId : filter.project,
      width: widthFilter,
      height: heightFilter,
    }),
  );
  // Only uploads carry dimensions; keep the controls while a size filter is set so it can be cleared.
  const sizeFilters = $derived(!!widthFilter || !!heightFilter || hasDimensions(loaded));
  async function loadPage(parameters = query, reset = false) {
    const ticket = reset ? ++generation : generation;
    loading = true;
    loadError = '';
    // Keep the current page on screen until the new one arrives, as Queue does.
    const cursor = reset ? null : nextCursor;
    try {
      const page = await client.request<{ items: Artifact[]; next_cursor: string | null }>(
        `/artifacts?${parameters}${cursor ? '&cursor=' + encodeURIComponent(cursor) : ''}`,
      );
      if (ticket !== generation) return;
      const replace = reset || loaded.length >= 2000;
      loaded = replace
        ? page.items
        : [
            ...loaded,
            ...page.items.filter((item) => !loaded.some((existing) => existing.id === item.id)),
          ];
      nextCursor = page.next_cursor;
      if (replace && scroller) scroller.scrollTop = 0;
    } catch (cause) {
      if (ticket === generation) loadError = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (ticket === generation) loading = false;
    }
  }
  $effect(() => {
    const parameters = query;
    const timer = setTimeout(() => void loadPage(parameters, true), 200);
    return () => clearTimeout(timer);
  });
  async function download(artifact: Artifact) {
    try {
      const blob = await client.media(artifact.id),
        url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = artifact.filename;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      setNotice(cause instanceof Error ? cause.message : String(cause));
    }
  }
  async function exportProvenance(entry: GalleryEntry) {
    try {
      const job = await fullJob(client, entry.artifact.job_id, entry.job);
      exportJSON(
        { artifact: entry.artifact, submission: job.submission },
        `${entry.artifact.id}.json`,
      );
    } catch (cause) {
      setNotice(String(cause));
    }
  }
  async function copy(entry: GalleryEntry, kind: 'prompt' | 'seed') {
    try {
      const job = await fullJob(client, entry.artifact.job_id, entry.job);
      await navigator.clipboard.writeText(recordedSettings(job, kind));
      setNotice(
        kind === 'prompt'
          ? 'Copied recorded API prompt.'
          : 'Copied named input values including recorded seeds.',
      );
    } catch (cause) {
      setNotice(cause instanceof Error ? cause.message : String(cause));
    }
  }
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">YOUR OUTPUT LIBRARY</span>
      <h1>Gallery <span class="muted">{artifacts.length}</span></h1>
    </div>
    <button onclick={() => navigate('workspace')}>Back to create ↗</button>
  </div>
  <div class="filter-bar">
    <input
      type="search"
      bind:value={filter.query}
      placeholder="Search provenance and metadata"
      aria-label="Search gallery"
    />
    <select bind:value={filter.kind} aria-label="Media kind"
      >{#each GALLERY_KINDS as option}<option value={option.value}>{option.label}</option
        >{/each}</select
    >
    <select bind:value={filter.sort} aria-label="Sort gallery"
      ><option value="newest">Newest first</option><option value="oldest">Oldest first</option
      ></select
    >
  </div>
  <details>
    <summary>Filter by operation, model, date, status or project</summary>
    <div class="filter-grid">
      <label
        >Operation<select bind:value={filter.operation}
          ><option value="">All operations</option
          >{#each ['image', 'video', 'edit', 'upscale', 'workflow', 'fleet'] as value}<option
              >{value}</option
            >{/each}</select
        ></label
      >
      <label>Model<input bind:value={filter.model} placeholder="Model filename" /></label>
      <label
        >Status<select bind:value={filter.status}
          >{#each GALLERY_STATES as option}<option value={option.value}>{option.label}</option
            >{/each}</select
        ></label
      >
      {#if sizeFilters}<label>Width<input type="number" min="1" bind:value={widthFilter} /></label
        ><label>Height<input type="number" min="1" bind:value={heightFilter} /></label>{:else}<p
          class="muted"
        >
          Width and height filters apply to uploaded inputs and masks. ComfyUI does not report the
          size of generated outputs.
        </p>{/if}
      <label>From<input type="date" bind:value={filter.from} /></label><label
        >Through<input type="date" bind:value={filter.until} /></label
      >
      <label><input type="checkbox" bind:checked={currentProject} /> Current project</label>
      {#if !currentProject}<ProjectSelect
          {client}
          summaries={projectSummaries}
          value={filter.project}
          onchange={(id) => (filter.project = id)}
        />{/if}
    </div>
  </details>
  {#if loadError}<p class="error" role="alert">{loadError}</p>{/if}
  <p class="muted">
    {loaded.length} loaded records (maximum 2,000 per window). Thumbnail browsing mounts only visible
    rows. Full media loads when selected.
  </p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard scrolling of the virtual library) -->
  <div
    class="virtual-gallery"
    bind:this={scroller}
    role="region"
    aria-label="Virtualized output library"
    tabindex="0"
    bind:clientWidth={width}
    bind:clientHeight={height}
    onscroll={(event) => (top = event.currentTarget.scrollTop)}
  >
    <div style:height={`${window.before}px`}></div>
    <div
      class="virtual-grid"
      style:grid-template-columns={`repeat(${window.columns},minmax(0,1fr))`}
    >
      {#each artifacts.slice(window.start, window.end) as entry (entry.artifact.id)}{@const artifact =
          entry.artifact}
        <article class="artifact-card" class:selected={selected?.id === artifact.id}>
          <button
            class="artifact-preview"
            onclick={() => {
              onselect(artifact);
              navigate('workspace');
            }}
            aria-label={`View ${artifact.filename}`}><Thumbnail {artifact} {client} /></button
          >
          <h3 title={artifact.filename}>{artifact.filename}</h3>
          <p>{formatTime(artifact.created_at)} · {entry.operation} · {entry.status}</p>
          <div class="workflow-actions">
            <button onclick={() => void reuse(artifact)}>Reuse</button><button
              onclick={() => {
                if (!selected) {
                  onselect(artifact);
                  setNotice('Choose the second comparison output.');
                } else {
                  oncompare(artifact);
                  navigate('workspace');
                }
              }}>Compare</button
            >
            <details
              ontoggle={(event) => {
                openActions = event.currentTarget.open
                  ? [...openActions, artifact.id]
                  : openActions.filter((id) => id !== artifact.id);
              }}
            >
              <summary>Actions</summary>
              <div class="actions-menu">
                {#if openActions.includes(artifact.id)}<ArtifactProjects
                    {client}
                    {artifact}
                    onsave={(value) => {
                      loaded = loaded.map((item) => (item.id === value.id ? value : item));
                      setNotice('Output project membership saved.');
                    }}
                  />{/if}
                <button onclick={() => void download(artifact)}>Download</button>
                <button onclick={() => void reuse(artifact, 'edit')}>Send to edit</button><button
                  onclick={() => void reuse(artifact, 'upscale')}>Upscale</button
                >
                <button onclick={() => void reuse(artifact, 'workflow')}>Load workflow</button>
                <button onclick={() => void copy(entry, 'prompt')}>Copy API prompt</button><button
                  onclick={() => void copy(entry, 'seed')}>Copy named inputs / seeds</button
                >
                <button onclick={() => void exportProvenance(entry)}>Export provenance</button>
              </div>
            </details>
          </div>
        </article>{/each}
    </div>
    <div style:height={`${window.after}px`}></div>
    {#if !artifacts.length && loading}<p class="muted" role="status">
        Loading outputs…
      </p>{:else if !artifacts.length && !loadError}<div class="empty-state">
        <h2>No matching outputs.</h2>
        <p>Change filters or complete a generation.</p>
      </div>{/if}
  </div>
  <button disabled={loading} onclick={() => void loadPage(query, true)}>Refresh library</button>
  {#if nextCursor}<button disabled={loading} onclick={() => void loadPage()}
      >{loading ? 'Loading…' : loaded.length >= 2000 ? 'Next window' : 'Load more results'}</button
    >{/if}
</section>

<style>
  .filter-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px;
    padding: 12px 0;
  }
  .virtual-gallery {
    height: 65vh;
    min-height: 330px;
    overflow: auto;
    overflow-anchor: none;
  }
  .virtual-grid {
    display: grid;
    gap: 0 12px;
  }
  .artifact-card {
    overflow: visible;
    position: relative;
    height: 318px;
    margin-bottom: 12px;
    box-sizing: border-box;
  }
  .artifact-preview {
    height: 165px;
  }
  .artifact-card h3 {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  .artifact-card p {
    font-size: 10px;
  }
  .artifact-card.selected {
    border-color: var(--brand);
  }
  .actions-menu {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    background: var(--panel);
    position: absolute;
    z-index: 3;
    max-width: 230px;
    bottom: 8px;
    right: 8px;
    max-height: 240px;
    overflow: auto;
    padding: 10px;
    border: 1px solid var(--border);
  }
</style>
