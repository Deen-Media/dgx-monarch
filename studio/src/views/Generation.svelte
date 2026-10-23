<script lang="ts">
  import type { Binding, Draft } from '../model';
  import { fieldLabel, bindingType, bindingError, NOT_RUNNING } from '../model';
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
  import AssetEditor from '../components/AssetEditor.svelte';
  import Field from '../components/Field.svelte';
  import LoraStack from '../components/LoraStack.svelte';
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
    changeDraft,
    inspect,
    fail,
    undo,
    prepareJob,
    rebind,
    catalogError = '',
    refreshCatalog,
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
    changeDraft: ChangeDraft;
    inspect: InspectWorkflow;
    fail: ReportError;
    undo: () => void;
    prepareJob: () => Promise<void>;
    /** Apply values that change a dynamic choice and rebuild the bindings from them. */
    rebind: (values: Draft['values']) => Promise<void>;
    catalogError?: string;
    refreshCatalog: () => Promise<void>;
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
  // Nothing accepts a connection at the driver origin, so a review could not be queued.
  let stopped = $derived(driverState === NOT_RUNNING);
  // The template being inspected shows until the inspection ends; then the draft's own.
  let choosing = $state('');
  $effect(() => {
    if (!busy) choosing = '';
  });
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
  <label for="workflow">Workflow</label><select
    id="workflow"
    value={choosing || workflowId}
    onchange={(e) => {
      const w = workflows.find((w) => w.id === e.currentTarget.value);
      if (!w) return;
      choosing = w.id;
      void inspect(w.workflow, w.name, false, w.id);
    }}
    ><option value="">{draft.workflowName || 'Choose a workflow'}</option
    >{#each workflows as w}<option value={w.id}>{w.name}</option>{/each}</select
  >
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
        onchange={(e) => changeDraft({ operation: e.currentTarget.value })}
        ><option value="image">Image</option><option value="video">Video</option><option
          value="edit">Edit</option
        ><option value="upscale">Upscale</option><option value="workflow">Workflow</option><option
          value="fleet">Fleet</option
        ></select
      >
    </div>
    <div>
      <label for="seed-policy">Seed policy</label><select
        id="seed-policy"
        value={draft.seed_policy}
        onchange={(e) => changeDraft({ seed_policy: e.currentTarget.value })}
        ><option value="fixed">Fixed</option><option value="random">Random</option></select
      >
    </div>
  </div>
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
      {#each visibleBindings as binding (binding.id)}<Field
          {binding}
          value={draft.values[binding.id]}
          onchange={(value) => fieldChange(binding, value)}
        />{/each}
      {#if bindingErrors.length}<div class="validation-list" role="alert">
          <h3>Workflow needs attention</h3>
          {#each bindingErrors as issue}<p>{issue}</p>{/each}
        </div>{/if}
      {#if bindings.some((binding) => binding.resolved !== false && bindingType(binding) === 'STRING')}<div
          class="surface"
        >
          <h3>Ordered prompt styles</h3>
          <label for="style-positive">Prompt input</label><select
            id="style-positive"
            value={draft.style_bindings?.prompt ?? ''}
            onchange={(e) =>
              changeDraft({
                style_bindings: {
                  ...draft.style_bindings,
                  prompt: e.currentTarget.value,
                },
              })}
            ><option value="">Choose the prompt binding</option
            >{#each bindings.filter((b) => b.resolved !== false && bindingType(b) === 'STRING') as field}<option
                value={field.id}>{fieldLabel(field)} · {field.node_id}</option
              >{/each}</select
          ><label for="style-negative">Negative prompt input · optional</label><select
            id="style-negative"
            value={draft.style_bindings?.negative_prompt ?? ''}
            onchange={(e) =>
              changeDraft({
                style_bindings: Object.fromEntries(
                  Object.entries({
                    ...draft.style_bindings,
                    negative_prompt: e.currentTarget.value,
                  }).filter(([, value]) => !!value),
                ),
              })}
            ><option value="">No negative input</option
            >{#each bindings.filter((b) => b.resolved !== false && bindingType(b) === 'STRING') as field}<option
                value={field.id}>{fieldLabel(field)} · {field.node_id}</option
              >{/each}</select
          >{#each draft.styles as style, i}<div class="stack-row">
              <span>{(style as any).name ?? `Style ${i + 1}`}</span><button
                onclick={() =>
                  changeDraft({ styles: draft.styles.filter((_, index) => index !== i) })}
                aria-label="Remove style">×</button
              >
            </div>{/each}<small>Effective text is shown in submission review.</small>
        </div>{/if}
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
    <button
      class="primary generate"
      onclick={prepareJob}
      disabled={busy || !draft.workflow || !!fieldErrors.length || pending || stopped}
      >Review generation <span>↗</span></button
    >{#if stopped}<small class="warning"
        >Start the driver first; see Driver not running above.</small
      >{:else if fieldErrors.length || bindingErrors.length}<small class="warning"
        >{fieldErrors.length
          ? `Resolve ${fieldErrors.length} field issue(s).`
          : 'Workflow inspection reported issues. Review revalidates the edited inputs.'}</small
      >{:else if !draft.workflow}<small>Choose or import a workflow to continue.</small
      >{:else}<small>Review effective settings and warnings before submitting.</small>{/if}
  </div>
</aside>
