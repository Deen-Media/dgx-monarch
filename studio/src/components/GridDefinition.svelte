<script lang="ts">
  import type { Binding } from '../model';
  import { fieldLabel } from '../model';
  import type { StudioClient } from '../api';
  import { axisExample, validateBlocker } from '../gridState';
  import StyleAxisCatalog from './StyleAxisCatalog.svelte';
  let {
    client,
    editable,
    axisA = $bindable(''),
    axisB = $bindable(''),
    valuesA = $bindable(''),
    valuesB = $bindable(''),
    seedPolicy = $bindable('fixed'),
    seedBinding = $bindable(''),
    baseSeed = $bindable(0),
    excluded = $bindable(''),
    hasWorkflow,
    busy,
    blocked,
    onvalidate,
  }: {
    client: StudioClient;
    editable: Binding[];
    axisA: string;
    axisB: string;
    valuesA: string;
    valuesB: string;
    seedPolicy: string;
    seedBinding: string;
    baseSeed: number;
    excluded: string;
    /** Whether the Create draft has a workflow for the grid to vary. */
    hasWorkflow: boolean;
    busy: boolean;
    /** Work in progress, or a pending dispatch, retry or recovery, holds the button. */
    blocked: boolean;
    onvalidate: () => void;
  } = $props();
  const reason = $derived(validateBlocker(hasWorkflow, { axisA, axisB, valuesA, valuesB }));
</script>

<div class="operation-grid">
  <div class="surface">
    <h3>Axis A</h3>
    <select bind:value={axisA} aria-label="First grid axis"
      ><option value="">Choose a typed input</option><option value="styles"
        >Named style stack</option
      >{#each editable as binding}<option value={binding.id}>{fieldLabel(binding)}</option
        >{/each}</select
    ><label
      >Values as JSON array<textarea
        bind:value={valuesA}
        rows="3"
        placeholder={`Example: ${axisExample(axisA)}`}></textarea></label
    >
  </div>
  <div class="surface">
    <h3>Axis B (optional)</h3>
    <select bind:value={axisB} aria-label="Second grid axis"
      ><option value="">No second axis</option><option value="styles">Named style stack</option
      >{#each editable as binding}<option value={binding.id}>{fieldLabel(binding)}</option
        >{/each}</select
    ><label
      >Values as JSON array<textarea
        bind:value={valuesB}
        rows="3"
        placeholder={`Example: ${axisExample(axisB)}`}></textarea></label
    >
  </div>
</div>
<StyleAxisCatalog
  {client}
  onchoose={(axis, id) => {
    if (axis === 'A') {
      axisA = 'styles';
      valuesA = JSON.stringify([[id]]);
    } else {
      axisB = 'styles';
      valuesB = JSON.stringify([[id]]);
    }
  }}
/>
<div class="filter-bar">
  <label
    >Seed policy<select bind:value={seedPolicy}
      ><option value="fixed">Fixed draft seeds (no randomization)</option><option value="increment"
        >Increment explicit seed per cell</option
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
<button
  class="primary"
  disabled={blocked || !!reason}
  aria-describedby={reason ? 'grid-validate-reason' : undefined}
  onclick={() => onvalidate()}>{busy ? 'Working…' : 'Validate every cell'}</button
>{#if reason}<small id="grid-validate-reason" class="warning validate-reason">{reason}</small>{/if}
