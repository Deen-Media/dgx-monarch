<script lang="ts">
  import { onMount } from 'svelte';
  import type { Binding, Draft } from '../model';
  import { fieldLabel } from '../model';
  import type { StudioClient } from '../api';
  import GridCellDetails from '../components/GridCellDetails.svelte';
  import Viewer from '../components/Viewer.svelte';
  import type { Artifact, ReuseArtifact } from '../contracts';
  import { exportJSON } from '../files';
  interface Cell {
    index: number;
    coordinates: Record<string, unknown>;
    excluded?: boolean;
    errors?: string[];
    preview_id?: string;
    job_id?: string;
    state?: string;
    attempts?: string[];
    artifact_ids?: string[];
  }
  interface Grid {
    id: string;
    state?: string;
    digest?: string;
    cells: Cell[];
    ready?: boolean;
    total?: number;
    job_count?: number;
    expires_at?: number;
  }
  let {
    client,
    draft,
    bindings,
    reuse,
  }: { client: StudioClient; draft: Draft; bindings: Binding[]; reuse: ReuseArtifact } = $props();
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
    retryIndices: number[] = [];
  let savedGrids = $state<Grid[]>([]),
    cellPage = $state(0),
    output = $state<Artifact | null>(null),
    compare = $state<Artifact | null>(null);
  const storage = 'dgxm.studio.grid.v1';
  const editable = $derived(bindings.filter((binding) => binding.resolved !== false));
  function remember() {
    sessionStorage.setItem(
      storage,
      JSON.stringify({ preview, grid, requestKey, retryKey, retryIndices }),
    );
  }
  function fail(cause: unknown) {
    error = cause instanceof Error ? cause.message : String(cause);
  }
  async function prepare() {
    busy = true;
    error = '';
    confirmed = false;
    preview = null;
    grid = null;
    selected = [];
    cellPage = 0;
    try {
      const axes = [{ name: axisA, values: JSON.parse(valuesA) }];
      if (axisB) axes.push({ name: axisB, values: JSON.parse(valuesB) });
      const exclusions = excluded.trim()
        ? excluded.split(',').map((value) => Number(value.trim()) - 1)
        : [];
      if (exclusions.some((value) => !Number.isInteger(value) || value < 0))
        throw new Error('Excluded cell numbers must be positive integers.');
      preview = await client.request<Grid>('/grids/preview', {
        draft: { ...$state.snapshot(draft), seed_policy: 'fixed' },
        axes,
        seed_policy: seedPolicy,
        base_seed: baseSeed,
        ...(seedBinding ? { seed_binding: seedBinding } : {}),
        excluded: exclusions,
      });
      requestKey = crypto.randomUUID();
      remember();
    } catch (cause) {
      fail(cause);
    } finally {
      busy = false;
    }
  }
  async function dispatch() {
    if (!preview || !confirmed) return;
    busy = true;
    error = '';
    try {
      grid = await client.request<Grid>('/grids', {
        preview_id: preview.id,
        idempotency_key: requestKey,
      });
      remember();
    } catch (cause) {
      fail(cause);
    } finally {
      busy = false;
    }
  }
  async function refresh(id = grid?.id, passive = false) {
    if (!id) return;
    try {
      const result = await client.request<Grid>(`/grids/${encodeURIComponent(id)}`);
      if (passive && grid?.id !== id) return;
      grid = result;
      remember();
    } catch (cause) {
      fail(cause);
    }
  }
  async function retry() {
    if (!grid || !selected.length) return;
    busy = true;
    error = '';
    if (!retryKey || JSON.stringify(retryIndices) !== JSON.stringify(selected)) {
      retryKey = crypto.randomUUID();
      retryIndices = [...selected];
      remember();
    }
    try {
      grid = await client.request<Grid>(`/grids/${grid.id}/retry`, {
        indices: retryIndices,
        idempotency_key: retryKey,
        confirmed: true,
      });
      retryKey = '';
      selected = [];
      remember();
    } catch (cause) {
      fail(cause);
    } finally {
      busy = false;
    }
  }
  onMount(() => {
    void client
      .request<{ items: Grid[] }>('/grids')
      .then((result) => (savedGrids = result.items))
      .catch(fail);
    try {
      const saved = JSON.parse(sessionStorage.getItem(storage) ?? 'null');
      if (saved) {
        preview = saved.preview;
        grid = saved.grid;
        requestKey = saved.requestKey;
        retryKey = saved.retryKey ?? '';
        retryIndices = saved.retryIndices ?? [];
      }
    } catch {
      error = 'Saved grid could not be restored. Load a durable grid ID.';
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
  <div class="operation-grid">
    <div class="surface">
      <h3>Axis A</h3>
      <select bind:value={axisA} aria-label="First grid axis"
        ><option value="">Choose a typed input</option>{#each editable as binding}<option
            value={binding.id}>{fieldLabel(binding)}</option
          >{/each}</select
      ><label>Values as JSON array<textarea bind:value={valuesA} rows="3"></textarea></label>
    </div>
    <div class="surface">
      <h3>Axis B (optional)</h3>
      <select bind:value={axisB} aria-label="Second grid axis"
        ><option value="">No second axis</option>{#each editable as binding}<option
            value={binding.id}>{fieldLabel(binding)}</option
          >{/each}</select
      ><label>Values as JSON array<textarea bind:value={valuesB} rows="3"></textarea></label>
    </div>
  </div>
  <div class="filter-bar">
    <label
      >Seed policy<select bind:value={seedPolicy}
        ><option value="fixed">Fixed draft seeds (no randomization)</option><option
          value="increment">Increment explicit seed per cell</option
        ></select
      ></label
    >
    {#if seedPolicy === 'increment'}<label
        >Seed binding<select bind:value={seedBinding}
          ><option value="">Choose seed input</option
          >{#each editable.filter( (binding) => ['seed', 'noise_seed'].includes(binding.input_name ?? '') ) as binding}<option
              value={binding.id}>{fieldLabel(binding)}</option
            >{/each}</select
        ></label
      ><label>Base seed<input type="number" min="0" step="1" bind:value={baseSeed} /></label>{/if}
    <label
      >Exclude cell numbers (comma separated)<input
        bind:value={excluded}
        placeholder="Example: 2,5"
      /></label
    >
  </div>
  <p>
    Execution target and topology are inherited from the active workflow and shown in each cell's
    backend preview. Maximum 1,024 cells.
  </p>
  <button class="primary" disabled={busy || !draft.workflow || !axisA} onclick={prepare}
    >{busy ? 'Working…' : 'Validate every cell'}</button
  >
  {#if error}<p class="error" role="alert">{error}</p>
    <p class="muted">
      An uncertain dispatch retains its request ID. Retry the same action to reconcile; do not
      create a new preview merely to retry transport.
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
      <button class="primary" disabled={busy || !confirmed || !preview.ready} onclick={dispatch}
        >Submit reviewed grid</button
      >
      <button onclick={() => exportJSON(preview, 'grid-preview.json')}
        >Export complete preview</button
      >
    </div>{/if}
  <div class="filter-bar">
    <label
      >Saved experiments<select
        value=""
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
        onclick={() => exportJSON(grid, 'grid-results.json')}>Export manifest and outcomes</button
      ><button disabled={busy || !selected.length} onclick={retry}
        >Retry {selected.length} selected failed/cancelled cells</button
      >
    </div>{/if}
  <div class="grid-cells">
    {#each ((grid ?? preview)?.cells ?? []).slice(cellPage * 40, (cellPage + 1) * 40) as cell}<article
        class="surface"
      >
        <h3>Cell {cell.index + 1} · {cell.excluded ? 'Excluded' : (cell.state ?? 'Preview')}</h3>
        <pre>{JSON.stringify(cell.coordinates, null, 2)}</pre>
        {#if cell.errors?.length}<p class="error">{cell.errors.join('; ')}</p>{/if}
        {#if grid && ['failed', 'cancelled'].includes(cell.state ?? '')}<label
            ><input
              type="checkbox"
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
