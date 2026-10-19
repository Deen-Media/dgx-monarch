<script lang="ts">
  import type { Binding, Draft } from '../model';
  import { fieldLabel, bindingType, bindingError, gridCombinations } from '../model';
  import type { StudioClient } from '../api';
  import type { JobPreview, SaveRecord } from '../contracts';
  import { exportJSON } from '../files';
  let {
    client,
    draft,
    bindings,
    pending,
    saveRecord,
    onreview,
  }: {
    client: StudioClient;
    draft: Draft;
    bindings: Binding[];
    pending: boolean;
    saveRecord: SaveRecord;
    onreview: (preview: JobPreview, kind: string) => void;
  } = $props();
  let busy = $state(false),
    gridAxis1 = $state(''),
    gridAxis2 = $state(''),
    gridValues1 = $state('[1,2]'),
    gridValues2 = $state('[1]'),
    gridPreviews = $state<{ values: Record<string, unknown>; preview: JobPreview }[]>([]),
    gridError = $state('');
  async function prepareGrid() {
    busy = true;
    gridError = '';
    gridPreviews = [];
    try {
      const axes = [{ id: gridAxis1, values: JSON.parse(gridValues1) }];
      if (gridAxis2) axes.push({ id: gridAxis2, values: JSON.parse(gridValues2) });
      if (axes.some((a) => !Array.isArray(a.values)))
        throw new Error('Axis values must be JSON arrays.');
      const cells = gridCombinations(axes, 1024);
      const results = [];
      for (const values of cells) {
        for (const [id, value] of Object.entries(values)) {
          const field = bindings.find((b) => b.id === id);
          if (!field) throw new Error('Grid axis is not in the active schema.');
          const issue = bindingError(field, value);
          if (issue) throw new Error(`${fieldLabel(field)}: ${issue}`);
        }
        const result = await client.request<JobPreview>('/jobs/preview', {
          ...$state.snapshot(draft),
          values: { ...draft.values, ...values },
        });
        results.push({ values, preview: result });
      }
      gridPreviews = results;
      await saveRecord('experiments', {
        name: `Grid ${new Date().toISOString()}`,
        axes,
        seed_policy: draft.seed_policy,
        draft: $state.snapshot(draft),
        cells: results.map((r) => ({
          values: r.values,
          preview_id: r.preview.id,
          errors: r.preview.errors ?? [],
        })),
      });
    } catch (e: any) {
      gridError = e.message;
    } finally {
      busy = false;
    }
  }
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
    Choose typed workflow axes. Every combination receives an independent backend preview; each
    accepted cell is submitted through normal admission.
  </p>
  <div class="operation-grid">
    <div class="surface">
      <h3>Axis A</h3>
      <select bind:value={gridAxis1} aria-label="First grid axis"
        ><option value="">Choose an input</option>{#each bindings as binding}<option
            value={binding.id}>{fieldLabel(binding)} ({bindingType(binding)})</option
          >{/each}</select
      ><label for="grid-values-a">Values as a JSON array</label><textarea
        id="grid-values-a"
        bind:value={gridValues1}
        rows="3"></textarea>
    </div>
    <div class="surface">
      <h3>Axis B · optional</h3>
      <select bind:value={gridAxis2} aria-label="Second grid axis"
        ><option value="">No second axis</option>{#each bindings as binding}<option
            value={binding.id}>{fieldLabel(binding)} ({bindingType(binding)})</option
          >{/each}</select
      ><label for="grid-values-b">Values as a JSON array</label><textarea
        id="grid-values-b"
        bind:value={gridValues2}
        rows="3"></textarea>
    </div>
  </div>
  <p>
    Seed policy: <strong>{draft.seed_policy}</strong>. Resource cost grows with the number of cells.
    Backend scheduling and topology guards apply.
  </p>
  <button class="primary" disabled={busy || !draft.workflow || !gridAxis1} onclick={prepareGrid}
    >{busy ? 'Validating cells…' : 'Validate and save experiment'}</button
  >{#if gridError}<p class="error" role="alert">
      {gridError}
    </p>{/if}{#if gridPreviews.length}<h2>{gridPreviews.length} reviewed cells</h2>
    <button
      onclick={() =>
        exportJSON(
          { draft, seed_policy: draft.seed_policy, cells: gridPreviews },
          'monarch-grid.json',
        )}>Export grid manifest</button
    >
    <div class="grid-cells">
      {#each gridPreviews as cell, i}<div class="surface">
          <h3>Cell {i + 1}</h3>
          <pre>{JSON.stringify(cell.values, null, 2)}</pre>
          <p class:warning={cell.preview.errors?.length}>
            {cell.preview.errors?.join(', ') || 'Backend preview ready'}
          </p>
          <button
            disabled={!!cell.preview.errors?.length || pending}
            onclick={() => {
              onreview(cell.preview, 'grid cell');
            }}>Review & submit cell</button
          >
        </div>{/each}
    </div>{/if}
</section>
