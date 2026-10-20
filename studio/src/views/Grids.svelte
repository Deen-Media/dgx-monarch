<script lang="ts">
  import { onMount } from 'svelte';
  import { restoreSession, validGrid } from '../sessionRecovery';
  import RecoveryNotice from '../components/RecoveryNotice.svelte';
  import type { Binding, Draft } from '../model';
  import { requireRecordIdentity, type StudioClient } from '../api';
  import GridCellDetails from '../components/GridCellDetails.svelte';
  import Viewer from '../components/Viewer.svelte';
  import GridDefinition from '../components/GridDefinition.svelte';
  import { GridReviewGuard } from '../gridReview';
  import ExperimentLibrary from '../components/ExperimentLibrary.svelte';
  import { experimentDefinition } from '../experimentRecords';
  import { submissionRefused } from '../submissionRefusal';
  import type { Artifact, RecordData, ReuseArtifact } from '../contracts';
  import { exportJSON } from '../files';
  import { gridIdentity, type Grid } from '../gridState';
  let {
    client,
    draft,
    bindings,
    reuse,
    restoreDraft,
  }: {
    client: StudioClient;
    draft: Draft;
    bindings: Binding[];
    reuse: ReuseArtifact;
    restoreDraft: (draft: Draft) => Promise<void>;
  } = $props();
  let axisA = $state(''),
    axisB = $state(''),
    valuesA = $state('[1,2]'),
    valuesB = $state('[1]');
  let seedPolicy = $state('fixed'),
    seedBinding = $state(''),
    baseSeed = $state(0),
    excluded = $state('');
  let preview = $state<Grid | null>(null),
    grid = $state<Grid | null>(null),
    error = $state(''),
    busy = $state(false),
    confirmed = $state(false);
  let selected = $state<number[]>([]),
    resumeId = $state('');
  let requestKey = '',
    retryKey = '',
    retryIndices = $state<number[]>([]);
  let loadGeneration = 0;
  let dispatchPending = $state(false),
    recoveryBlocked = $state(false);
  let retryPending = $state(false);
  let savedGrids = $state<Grid[]>([]),
    savedCursor = $state<string | null>(null),
    savedLoading = $state(false),
    cellPage = $state(0),
    output = $state<Artifact | null>(null),
    compare = $state<Artifact | null>(null);
  const mutationBlocked = $derived(recoveryBlocked || busy || dispatchPending || retryPending);
  const storage = 'dgxm.studio.grid.v1';
  const editable = $derived(bindings.filter((binding) => binding.resolved !== false));
  const definition = $derived(
    JSON.stringify({
      draft,
      axisA,
      axisB,
      valuesA,
      valuesB,
      seedPolicy,
      seedBinding,
      baseSeed,
      excluded,
    }),
  );
  const reviewGuard = new GridReviewGuard();
  $effect(() => {
    if (reviewGuard.invalidate(definition, dispatchPending) && preview) {
      preview = null;
      confirmed = false;
      requestKey = '';
      error = 'Grid definition changed. Validate every cell again before submission.';
      try {
        remember();
      } catch (cause) {
        fail(cause);
      }
    }
  });
  function remember() {
    if (recoveryBlocked) return;
    sessionStorage.setItem(
      storage,
      JSON.stringify({
        preview: gridIdentity(preview),
        grid: gridIdentity(grid),
        requestKey,
        retryKey,
        retryIndices,
        dispatchPending,
        retryPending,
      }),
    );
  }
  function fail(cause: unknown) {
    error = cause instanceof Error ? cause.message : String(cause);
  }
  async function loadExperiment(data: RecordData) {
    if (recoveryBlocked || dispatchPending || retryPending)
      throw new Error('Reconcile the pending grid dispatch before replacing its definition.');
    loadGeneration++;
    const saved = experimentDefinition(data);
    await restoreDraft(saved.draft);
    axisA = saved.axisA;
    valuesA = saved.valuesA;
    axisB = saved.axisB;
    valuesB = saved.valuesB;
    seedPolicy = saved.seedPolicy;
    seedBinding = saved.seedBinding;
    baseSeed = saved.baseSeed;
    excluded = saved.excluded;
    preview = null;
    grid = null;
    confirmed = false;
    selected = [];
    cellPage = 0;
    requestKey = '';
    retryKey = '';
    remember();
  }
  async function prepare() {
    if (recoveryBlocked || dispatchPending || retryPending) return;
    loadGeneration++;
    busy = true;
    error = '';
    confirmed = false;
    preview = null;
    grid = null;
    selected = [];
    cellPage = 0;
    const requestedDefinition = definition;
    try {
      const axes = [{ name: axisA, values: JSON.parse(valuesA) }];
      if (axisB) axes.push({ name: axisB, values: JSON.parse(valuesB) });
      const exclusions = excluded.trim()
        ? excluded.split(',').map((value) => Number(value.trim()) - 1)
        : [];
      if (exclusions.some((value) => !Number.isInteger(value) || value < 0))
        throw new Error('Excluded cell numbers must be positive integers.');
      const reviewed = await client.request<Grid>('/grids/preview?summary=true', {
        draft: { ...$state.snapshot(draft), seed_policy: 'fixed' },
        axes,
        seed_policy: seedPolicy,
        ...(seedPolicy !== 'fixed' ? { base_seed: baseSeed, seed_binding: seedBinding } : {}),
        excluded: exclusions,
      });
      reviewGuard.accept(requestedDefinition, definition);
      preview = reviewed;
      requestKey = crypto.randomUUID();
      remember();
    } catch (cause) {
      fail(cause);
    } finally {
      busy = false;
    }
  }
  async function dispatch() {
    if (recoveryBlocked || !preview || !confirmed) return;
    loadGeneration++;
    busy = true;
    error = '';
    try {
      const prior = dispatchPending;
      dispatchPending = true;
      try {
        remember();
      } catch (cause) {
        dispatchPending = prior;
        throw cause;
      }
      grid = requireRecordIdentity(
        await client.request<Grid>('/grids?summary=true', {
          preview_id: preview.id,
          idempotency_key: requestKey,
        }),
      );
      dispatchPending = false;
      remember();
    } catch (cause) {
      if (submissionRefused(cause, 'grid')) {
        dispatchPending = false;
        preview = null;
        confirmed = false;
        remember();
      }
      fail(cause);
    } finally {
      busy = false;
    }
  }
  async function refresh(id = grid?.id, passive = false) {
    if (!id) return;
    if (!passive && (dispatchPending || retryPending)) {
      error = 'Reconcile the pending grid action before opening another run.';
      return;
    }
    const ticket = passive ? loadGeneration : ++loadGeneration;
    try {
      const result = await client.request<Grid>(`/grids/${encodeURIComponent(id)}?summary=true`);
      if (ticket !== loadGeneration || (passive && grid?.id !== id)) return;
      if (grid?.id !== id) {
        selected = [];
        retryKey = '';
        retryIndices = [];
        cellPage = 0;
      }
      grid = result;
      remember();
    } catch (cause) {
      fail(cause);
    }
  }
  async function retry() {
    if (recoveryBlocked || !grid || (!retryPending && !selected.length)) return;
    loadGeneration++;
    busy = true;
    error = '';
    if (!retryPending && (!retryKey || JSON.stringify(retryIndices) !== JSON.stringify(selected))) {
      retryKey = crypto.randomUUID();
      retryIndices = [...selected];
    }
    try {
      const prior = retryPending;
      retryPending = true;
      try {
        remember();
      } catch (cause) {
        retryPending = prior;
        throw cause;
      }
      grid = requireRecordIdentity(
        await client.request<Grid>(`/grids/${grid.id}/retry?summary=true`, {
          indices: retryIndices,
          idempotency_key: retryKey,
          confirmed: true,
        }),
      );
      retryKey = '';
      retryPending = false;
      selected = [];
      remember();
    } catch (cause) {
      if (submissionRefused(cause, 'grid-retry')) {
        retryPending = false;
        retryKey = '';
        retryIndices = [];
        remember();
      }
      fail(cause);
    } finally {
      busy = false;
    }
  }
  async function exportGrid(value: Grid, isPreview = false) {
    try {
      const path = isPreview ? `/grids/previews/${value.id}` : `/grids/${value.id}`;
      exportJSON(await client.request(path), isPreview ? 'grid-preview.json' : 'grid-results.json');
    } catch (cause) {
      fail(cause);
    }
  }
  async function loadSavedGrids() {
    if (savedLoading) return;
    savedLoading = true;
    try {
      const result = await client.request<{ items: Grid[]; next_cursor: string | null }>(
        `/grids${savedCursor ? '?cursor=' + encodeURIComponent(savedCursor) : ''}`,
      );
      savedGrids = savedGrids.length >= 2000 ? result.items : [...savedGrids, ...result.items];
      savedCursor = result.next_cursor;
    } catch (cause) {
      fail(cause);
    } finally {
      savedLoading = false;
    }
  }
  onMount(() => {
    void loadSavedGrids();
    const restored = restoreSession(storage, validGrid);
    recoveryBlocked = restored.blocked;
    if (restored.value) {
      const saved = restored.value;
      preview = gridIdentity(saved.preview);
      grid = gridIdentity(saved.grid);
      ({ requestKey, retryKey, retryIndices, dispatchPending, retryPending } = saved);
      if (preview) reviewGuard.restore(definition);
    }
    if (preview) {
      const id = preview.id,
        ticket = loadGeneration;
      void client
        .request<Grid>(`/grids/previews/${encodeURIComponent(id)}?summary=true`)
        .then((value) => {
          if (ticket === loadGeneration && preview?.id === id) preview = value;
        })
        .catch(fail);
    }
    let stopped = false,
      timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      if (!document.hidden && !busy) await refresh(grid?.id, true);
      if (!stopped) timer = setTimeout(tick, 3000);
    };
    void tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  });
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">CONTROLLED EXPLORATION</span>
      <h1>Parameter grids</h1>
    </div>
    <span class="tag">{draft.workflowName || 'No workflow'}</span>
  </div>
  <p class="muted">
    Each cell passes normal backend admission. Comfy's queue controls execution; Studio does not run
    a second scheduler. More cells increase total compute and memory residency time.
  </p>
  {#if recoveryBlocked}<RecoveryNotice
      storageKey={storage}
      check="Load durable grid IDs and inspect Queue for accepted cells"
      onclear={() => (recoveryBlocked = false)}
    />{/if}
  <ExperimentLibrary
    {client}
    projectId={draft.project_id}
    gridId={grid?.id}
    disabled={mutationBlocked}
    onload={loadExperiment}
    onresume={async (id) => {
      cellPage = 0;
      await refresh(id);
    }}
  />
  <GridDefinition
    {client}
    {editable}
    bind:axisA
    bind:axisB
    bind:valuesA
    bind:valuesB
    bind:seedPolicy
    bind:seedBinding
    bind:baseSeed
    bind:excluded
  />
  {#if savedCursor}<button disabled={savedLoading} onclick={loadSavedGrids}
      >{savedGrids.length >= 2000 ? 'Next saved-grid window' : 'Load more saved grids'}</button
    >{/if}
  <p>
    Execution target and topology are inherited from the active workflow and shown in each cell's
    backend preview. Maximum 1,024 cells.
  </p>
  <button class="primary" disabled={mutationBlocked || !draft.workflow || !axisA} onclick={prepare}
    >{busy ? 'Working…' : 'Validate every cell'}</button
  >
  {#if error}<p class="error" role="alert">{error}</p>
    {#if dispatchPending || retryPending}<p class="muted">
        An uncertain dispatch retains its request ID. Retry the same action to reconcile; do not
        create a new preview merely to retry transport.
      </p>{/if}{/if}
  {#if dispatchPending}<p role="status">
      Grid dispatch outcome is pending. Retry the same reviewed grid to reconcile its retained
      request identity.
    </p>{/if}
  {#if retryPending}<p role="status">
      Retry outcome is pending for cells {retryIndices.map((index) => index + 1).join(', ')}. Retry
      the retained selection to reconcile the same request.
    </p>{/if}
  {#if preview}<div class="surface">
      <h2>
        {preview.total ?? preview.cells.length} cells · {preview.job_count ??
          preview.cells.filter((cell) => !cell.excluded).length} jobs
      </h2>
      <p>
        Preview {preview.id} · digest {preview.digest} · expires {preview.expires_at
          ? new Date(preview.expires_at * 1000).toLocaleString()
          : 'unknown'}
      </p>
      <label
        ><input type="checkbox" bind:checked={confirmed} /> I reviewed the cells, exclusions, seed policy
        and execution cost.</label
      >
      <button
        class="primary"
        disabled={recoveryBlocked || busy || !confirmed || !preview.ready}
        onclick={dispatch}>Submit reviewed grid</button
      >
      <button onclick={() => void exportGrid(preview!, true)}>Export complete preview</button>
    </div>{/if}
  <div class="filter-bar">
    <label
      >Saved experiments<select
        value=""
        disabled={busy || dispatchPending || retryPending}
        onchange={(event) => {
          resumeId = event.currentTarget.value;
          cellPage = 0;
          void refresh(resumeId);
        }}
        ><option value="">Choose a saved grid</option>{#each savedGrids as saved}<option
            value={saved.id}>{saved.id} · {saved.state}</option
          >{/each}</select
      ></label
    >
    <label>Resume durable grid ID<input bind:value={resumeId} /></label><button
      disabled={!resumeId || busy}
      onclick={() => void refresh(resumeId)}>Load grid</button
    >
  </div>
  {#if grid}<div class="surface">
      <h2>Grid {grid.id} · {grid.state}</h2>
      <button onclick={() => void refresh()}>Refresh status</button><button
        onclick={() => void exportGrid(grid!)}>Export manifest and outcomes</button
      ><button
        disabled={recoveryBlocked || busy || (!retryPending && !selected.length)}
        onclick={retry}
        >{retryPending
          ? 'Reconcile retained retry'
          : `Retry ${selected.length} selected failed/cancelled cells`}</button
      >
    </div>{/if}
  <div class="grid-cells">
    {#each ((grid ?? preview)?.cells ?? []).slice(cellPage * 40, (cellPage + 1) * 40) as cell (cell.job_id ?? cell.preview_id ?? `${(grid ?? preview)?.id}:${cell.index}`)}<article
        class="surface"
      >
        <h3>Cell {cell.index + 1} · {cell.excluded ? 'Excluded' : (cell.state ?? 'Preview')}</h3>
        <pre>{JSON.stringify(cell.coordinates, null, 2)}</pre>
        {#if cell.errors?.length}<p class="error">{cell.errors.join('; ')}</p>{/if}
        {#if grid && ['failed', 'cancelled'].includes(cell.state ?? '')}<label
            ><input
              type="checkbox"
              disabled={retryPending}
              checked={selected.includes(cell.index)}
              onchange={(event) =>
                (selected = event.currentTarget.checked
                  ? [...selected, cell.index]
                  : selected.filter((index) => index !== cell.index))}
            /> Select for retry</label
          >{/if}
        <GridCellDetails
          {client}
          previewId={cell.preview_id}
          jobId={cell.job_id}
          onview={(artifact) => (output = artifact)}
          oncompare={(artifact) => {
            if (output) compare = artifact;
            else output = artifact;
          }}
        />
        <details>
          <summary>Inspect attempts and output IDs</summary>
          <pre>{JSON.stringify(cell, null, 2)}</pre>
        </details>
      </article>{/each}
  </div>
  {#if (grid ?? preview)?.cells.length}<div class="pagination">
      <button disabled={cellPage === 0} onclick={() => cellPage--}>Previous cells</button><span
        >Cell page {cellPage + 1}</span
      ><button
        disabled={(cellPage + 1) * 40 >= ((grid ?? preview)?.cells.length ?? 0)}
        onclick={() => cellPage++}>Next cells</button
      >
    </div>{/if}
  {#if output}<Viewer
      artifact={output}
      {compare}
      {client}
      onreuse={reuse}
      onselect={(artifact) => (compare = artifact)}
    />{/if}
</section>
