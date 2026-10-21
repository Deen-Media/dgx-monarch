<script lang="ts">
  import type { Binding, Draft } from '../model';
  import { fieldLabel } from '../model';
  import { layoutWorkflow, unchangedWorkflow } from '../workflowLayout';
  import { numberField } from '../numberField';
  import type { ChangeDraft, InspectWorkflow, ReportError } from '../contracts';
  let {
    draft,
    bindings = $bindable(),
    rawWorkflow = $bindable(),
    changeDraft,
    inspect,
    fail,
  }: {
    draft: Draft;
    bindings: Binding[];
    rawWorkflow: string;
    changeDraft: ChangeDraft;
    inspect: InspectWorkflow;
    fail: ReportError;
  } = $props();
  function configureBinding(id: string, change: Partial<Binding>) {
    bindings = bindings.map((b) => (b.id === id ? { ...b, ...change } : b));
    if (draft.workflow) {
      const updated = layoutWorkflow($state.snapshot(draft.workflow), $state.snapshot(bindings));
      rawWorkflow = updated.rawWorkflow;
      changeDraft({ workflow: updated.workflow });
    }
  }
  async function rebind(source: string, target: string) {
    if (!draft.workflow || !target) return;
    const workflow = structuredClone($state.snapshot(draft.workflow));
    workflow.extra ??= {};
    workflow.extra.dgxm_studio ??= {};
    workflow.extra.dgxm_studio.bindings ??= {};
    workflow.extra.dgxm_studio.bindings[source] = {
      ...(workflow.extra.dgxm_studio.bindings[source] ?? {}),
      target,
    };
    await inspect(workflow, draft.workflowName, true);
  }
</script>

<p class="muted">
  Choose prominent inputs and display labels. Stable binding identities are retained with the
  workflow.
</p>
{#each bindings as binding}<div class="layout-field">
    <label
      ><input
        type="checkbox"
        checked={!!binding.prominent}
        onchange={(e) => configureBinding(binding.id, { prominent: e.currentTarget.checked })}
      />Prominent</label
    ><input
      aria-label={`Label for ${binding.id}`}
      value={fieldLabel(binding)}
      onchange={(e) => configureBinding(binding.id, { label: e.currentTarget.value })}
    /><code>{binding.id}</code>
    <div class="input-row">
      <label
        >Group<input
          value={String(binding.group ?? '')}
          onchange={(e) => configureBinding(binding.id, { group: e.currentTarget.value })}
        /></label
      ><label
        >Order<input
          type="number"
          placeholder="0"
          use:numberField={{
            value: binding.order,
            commit: (order) => order !== null && configureBinding(binding.id, { order }),
            event: 'change',
          }}
        /></label
      >
    </div>
    {#if binding.resolved === false}<label
        >Rebind to a supported input<select
          value=""
          onchange={(e) => void rebind(binding.id, e.currentTarget.value)}
          ><option value="">Choose stable target identity</option
          >{#each bindings.filter((b) => b.resolved !== false) as target}<option value={target.id}
              >{fieldLabel(target)} · {target.id}</option
            >{/each}</select
        ></label
      >{/if}
  </div>{/each}
<details>
  <summary>Inspect / repair workflow JSON</summary><textarea
    rows="12"
    class="mono"
    bind:value={rawWorkflow}
    spellcheck="false"
    aria-label="Workflow JSON"></textarea><button
    onclick={() => {
      try {
        const edited = JSON.parse(rawWorkflow);
        void inspect(
          edited,
          draft.workflowName,
          unchangedWorkflow($state.snapshot(draft.workflow), edited),
        );
      } catch (e) {
        fail(e);
      }
    }}>Validate edited workflow</button
  >
</details>
