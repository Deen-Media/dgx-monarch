<script lang="ts">
  import type { Binding, Draft } from '../model';
  import { bindingError, isSeedBinding, NOT_RUNNING } from '../model';
  import type {
    Catalog,
    WorkflowEntry,
    ChangeDraft,
    InspectWorkflow,
    ReportError,
    GraphConnection,
  } from '../contracts';
  import { exportJSON } from '../files';
  import { checkImportFile, checkRequestBody, type StudioClient } from '../api';
  import { filterKeyAction, loadWorkflow, workflowChoices, workflowTitle } from '../workflowChoice';
  import { operationChoices, outputSummary, workflowTraits } from '../workflowTraits';
  import { durationGroups, inputMetadata, synchronized } from '../frameCounts';
  import AssetEditor from '../components/AssetEditor.svelte';
  import DurationControl from '../components/DurationControl.svelte';
  import Field from '../components/Field.svelte';
  import LoraStack from '../components/LoraStack.svelte';
  import PromptStyles from '../components/PromptStyles.svelte';
  import WorkflowLayout from '../components/WorkflowLayout.svelte';
  let {
    client,
    draft,
    bindings = $bindable(),
    rawWorkflow = $bindable(),
    controlTab = $bindable(),
    bindingErrors,
    catalog,
    connections,
    workflows,
    workflowId,
    requestLimit,
    dirty,
    busy,
    pending,
    driverState,
    confirmation = '',
    changeDraft,
    inspect,
    fail,
    undo,
    prepareJob,
    rebind,
    catalogError = '',
    refreshCatalog,
    workflowQuery = $bindable(''),
  }: {
    client: StudioClient;
    draft: Draft;
    bindings: Binding[];
    rawWorkflow: string;
    controlTab: string;
    bindingErrors: string[];
    catalog: Catalog;
    connections: GraphConnection[];
    workflows: WorkflowEntry[];
    /** The bundled workflow the draft came from, or '' for any other workflow. */
    workflowId: string;
    requestLimit: number;
    dirty: boolean;
    busy: boolean;
    pending: boolean;
    driverState?: string;
    /** The brief line after an accepted submit, or ''. */
    confirmation?: string;
    changeDraft: ChangeDraft;
    inspect: InspectWorkflow;
    fail: ReportError;
    undo: () => void;
    /** Generate starts a valid job at once; with `review` true it opens Review settings first. */
    prepareJob: (review?: boolean) => Promise<void>;
    /** Apply values that change a dynamic choice and rebuild the bindings from them. */
    rebind: (values: Draft['values']) => Promise<void>;
    catalogError?: string;
    refreshCatalog: () => Promise<void>;
    /** The workflow filter's text; it starts empty on each mount unless a parent binds it. */
    workflowQuery?: string;
  } = $props();
  let fieldErrors = $derived(
    bindings
      .filter((b) => b.resolved !== false)
      .map((b) => bindingError(b, draft.values[b.id]))
      .filter(Boolean),
  );
  let visibleBindings = $derived(
    (controlTab === 'advanced'
      ? [...bindings]
      : bindings.filter((b) => b.prominent || b.display?.prominent)
    ).sort((a, b) => Number(a.order ?? 0) - Number(b.order ?? 0)),
  );
  // What the graph loads, makes and saves decides the operations it offers and the output it
  // names. Until an inspection lists its inputs, nothing is known and every operation stays.
  let traits = $derived(
    draft.workflow && bindings.length
      ? workflowTraits(draft.workflow, bindings, draft.values, catalog.object_info)
      : null,
  );
  let operations = $derived(operationChoices(traits, draft.operation));
  let operationNeed = $derived(operations.find((o) => o.id === draft.operation)?.need ?? null);
  let metadata = $derived(inputMetadata(draft.workflow, bindings, catalog.object_info));
  let durations = $derived(durationGroups(draft.workflow, bindings).filter(synchronized));
  // Duration follows the last size field it shows, or ends the list when none shows.
  let durationAnchor = $derived(
    [...visibleBindings].reverse().find((b) => ['width', 'height'].includes(b.input_name ?? ''))
      ?.id ?? '',
  );
  // Under Random the companion replaces every seed input for each job and records the seeds.
  let randomSeeds = $derived(draft.seed_policy === 'random');
  // Nothing accepts a connection at the driver origin, so a job could not be queued.
  let stopped = $derived(driverState === NOT_RUNNING);
  let generateOff = $derived(busy || !draft.workflow || !!fieldErrors.length || pending || stopped);
  // The template being inspected shows until the inspection ends; then the draft's own.
  let choosing = $state('');
  $effect(() => {
    if (!busy) choosing = '';
  });
  let filterInput: HTMLInputElement;
  let current = $derived(choosing || workflowId);
  let choices = $derived(workflowChoices(workflows, workflowQuery, current));
  let chosen = $derived(workflows.find((w) => w.id === current));
  function chooseWorkflow(entry: WorkflowEntry) {
    choosing = entry.id;
    void loadWorkflow(entry, inspect);
  }
  function filterKeydown(event: KeyboardEvent) {
    // Enter that confirms an input-method candidate finishes the text, nothing more.
    if (event.isComposing) return;
    const action = filterKeyAction(event.key, current, choices.matches);
    if (!action) return;
    event.preventDefault();
    if (action === 'clear') workflowQuery = '';
    else chooseWorkflow(action);
  }
  function clearFilter() {
    workflowQuery = '';
    filterInput.focus();
  }
  function fieldChange(binding: Binding, value: unknown) {
    const values = { ...draft.values, [binding.id]: value };
    // A dynamic choice changes which child inputs exist, so the bindings are rebuilt.
    if (binding.schema?.dynamic) void rebind(values);
    else changeDraft({ values });
  }
  async function importWorkflow(file: File | undefined) {
    if (!file) return;
    try {
      checkImportFile(file.size, requestLimit, 'Workflow file');
      const workflow = JSON.parse(await file.text());
      checkRequestBody({ workflow }, requestLimit, 'This workflow');
      await inspect(workflow, file.name);
    } catch (e) {
      fail(e);
    }
  }
</script>

<aside class="controls-panel">
  <div class="panel-title">
    <div>
      <span class="eyebrow">GENERATION</span>
      <h1>Create</h1>
    </div>
    <span class="tag" class:lime={dirty}>{dirty ? 'Draft edited' : 'Draft'}</span>
  </div>
  <label for="workflow">Workflow</label>
  <div class="workflow-filter">
    <input
      bind:this={filterInput}
      bind:value={workflowQuery}
      onkeydown={filterKeydown}
      type="search"
      placeholder="Filter by title or file name"
      aria-label="Filter workflows"
    /><button disabled={!workflowQuery} onclick={clearFilter} aria-label="Clear workflow filter"
      >Clear</button
    >
  </div>
  {#if workflowQuery && !choices.matches.length}<p class="muted">
      No packaged workflow matches this filter.
    </p>{/if}
  <select
    id="workflow"
    value={current}
    onchange={(e) => {
      const w = workflows.find((w) => w.id === e.currentTarget.value);
      if (w) chooseWorkflow(w);
    }}
    ><option value="">{draft.workflowName || 'Choose a workflow'}</option
    >{#each choices.options as w (w.id)}<option value={w.id} title={w.name}
        >{workflowTitle(w)}</option
      >{/each}</select
  >{#if chosen}<small class="workflow-file mono">{chosen.name}</small>{/if}{#if traits}<small
      class="workflow-output">Output: {outputSummary(traits)}</small
    >{/if}
  <div class="workflow-actions">
    <label class="button import-label"
      >Import JSON<input
        type="file"
        accept=".json,application/json"
        onchange={async (event) => {
          const input = event.currentTarget;
          try {
            await importWorkflow(input.files?.[0]);
          } finally {
            input.value = '';
          }
        }}
      /></label
    ><button
      disabled={!draft.workflow}
      onclick={() => exportJSON(draft.workflow, `${draft.workflowName || 'workflow'}.json`)}
      >Export</button
    ><button onclick={undo} disabled={!dirty}>Undo</button>
  </div>
  <div class="input-row">
    <div>
      <label for="generation-operation">Operation</label><select
        id="generation-operation"
        value={draft.operation}
        aria-describedby={operationNeed ? 'operation-need' : undefined}
        onchange={(e) => changeDraft({ operation: e.currentTarget.value })}
        >{#each operations as operation (operation.id)}<option value={operation.id}
            >{operation.label}{operation.need ? ' (not supported)' : ''}</option
          >{/each}</select
      >
    </div>
    <div>
      <label for="seed-policy">Seed policy</label><select
        id="seed-policy"
        aria-describedby="seed-policy-note"
        value={draft.seed_policy}
        onchange={(e) => changeDraft({ seed_policy: e.currentTarget.value })}
        ><option value="fixed">Fixed</option><option value="random">Random</option></select
      >
    </div>
  </div>
  <small class="seed-policy-note" id="seed-policy-note"
    >{randomSeeds
      ? 'Random: each job gets new seeds. Review settings shows them before it starts.'
      : 'Fixed: jobs use the seed fields as they are.'}</small
  >{#if operationNeed}<small class="warning operation-need" id="operation-need"
      >{operationNeed}</small
    >{/if}
  <div class="tabs" aria-label="Generation sections">
    {#each [{ id: 'inputs', label: 'Inputs' }, { id: 'advanced', label: 'All inputs' }, { id: 'assets', label: 'Assets' }, { id: 'loras', label: 'LoRAs' }, { id: 'workflow', label: 'Layout' }] as tab}<button
        class:active={controlTab === tab.id}
        aria-pressed={controlTab === tab.id}
        onclick={() => (controlTab = tab.id)}>{tab.label}</button
      >{/each}
  </div>
  <div class="controls-scroll">
    {#if controlTab === 'inputs' || controlTab === 'advanced'}
      {#if !draft.workflow}<div class="empty-state">
          <span class="outline-icon">◇</span>
          <h3>Start with a workflow</h3>
          <p>
            Load a supported workflow to discover its typed model, prompt and sampling controls.
          </p>
        </div>{/if}
      {#if controlTab === 'inputs' && draft.workflow && !visibleBindings.length}<p class="muted">
          No prominent inputs configured. Use All inputs or customize Layout.
        </p>{/if}
      {#snippet durationControls()}{#each durations as group (group.family)}<DurationControl
            {group}
            values={draft.values}
            onchange={(values) => changeDraft({ values })}
          />{/each}{/snippet}
      {#each visibleBindings as binding (binding.id)}<Field
          {binding}
          value={draft.values[binding.id]}
          meta={metadata[binding.id]}
          note={randomSeeds && isSeedBinding(binding)
            ? 'Chosen for each job: the Random policy replaces this value.'
            : ''}
          onchange={(value) => fieldChange(binding, value)}
        />{#if binding.id === durationAnchor}{@render durationControls()}{/if}{/each}
      {#if draft.workflow && !durationAnchor}{@render durationControls()}{/if}
      {#if bindingErrors.length}<div class="validation-list" role="alert">
          <h3>Workflow needs attention</h3>
          {#each bindingErrors as issue}<p>{issue}</p>{/each}
        </div>{/if}
      <PromptStyles {draft} {bindings} {changeDraft} />
      {#if draft.parent_artifact_ids.length}<div class="surface">
          <h3>Input lineage</h3>
          <p class="mono">{draft.parent_artifact_ids.join(', ')}</p>
          <small>Lineage records ancestry. Workflow asset controls specify the actual input.</small
          ><button onclick={() => changeDraft({ parent_artifact_ids: [] })}>Clear lineage</button>
        </div>{/if}
    {:else if controlTab === 'assets'}<AssetEditor
        {client}
        {draft}
        bind:bindings
        {changeDraft}
      />{:else if controlTab === 'loras'}<LoraStack
        {draft}
        {catalog}
        {catalogError}
        {refreshCatalog}
        {connections}
        {changeDraft}
      />{:else if controlTab === 'workflow'}<WorkflowLayout
        {draft}
        bind:bindings
        bind:rawWorkflow
        {changeDraft}
        {inspect}
        {fail}
      />
    {/if}
  </div>
  <div class="submission-footer">
    <div class="target-line">
      <span class="status-dot" class:healthy={driverState === 'connected'}></span><span
        >Driver {driverState ?? 'unknown'}</span
      ><span class="muted">Backend admission</span>
    </div>
    <div class="generate-actions">
      <button class="review-settings" onclick={() => prepareJob(true)} disabled={generateOff}
        >Review settings</button
      ><button class="primary generate" onclick={() => prepareJob()} disabled={generateOff}
        >Generate <span>↗</span></button
      >
    </div>
    {#if stopped}<small class="warning">Start the driver first; see Driver not running above.</small
      >{:else if confirmation}<small class="submit-confirmation" aria-hidden="true"
        >{confirmation}</small
      >{:else if fieldErrors.length || bindingErrors.length}<small class="warning"
        >{fieldErrors.length
          ? `Resolve ${fieldErrors.length} field issue(s).`
          : 'Workflow inspection reported issues. Generate checks the edited inputs again.'}</small
      >{:else if !draft.workflow}<small>Choose or import a workflow to continue.</small
      >{:else}<small
        >Generate checks the draft and starts the job. Review settings shows it first.</small
      >{/if}
    <span class="sr-only" role="status">{confirmation}</span>
  </div>
</aside>
