<script lang="ts">
  import type { Catalog, ChangeDraft, GraphConnection } from '../contracts';
  import type { Draft } from '../model';
  let {
    draft,
    catalog,
    connections,
    changeDraft,
  }: { draft: Draft; catalog: Catalog; connections: GraphConnection[]; changeDraft: ChangeDraft } =
    $props();
  const loaders = $derived(
    ['LoraLoader', 'LoraLoaderModelOnly', 'DGXMonarchLoraLoader'].filter(
      (name) => name in (catalog.object_info ?? {}),
    ),
  );
  const compatible = $derived(
    connections.filter((item) =>
      item.type === 'CLIP'
        ? draft.lora_insertion?.loader === 'LoraLoader'
        : item.type ===
          (draft.lora_insertion?.loader === 'DGXMonarchLoraLoader' ? 'DGXM_MODEL' : 'MODEL'),
    ),
  );
  function chooseLoader(loader: string) {
    changeDraft({
      lora_insertion: loader ? { loader, model_targets: [], clip_targets: [] } : undefined,
    });
  }
  function toggle(connection: GraphConnection, checked: boolean) {
    if (!draft.lora_insertion) return;
    const key = connection.type === 'CLIP' ? 'clip_targets' : 'model_targets';
    const rows = draft.lora_insertion[key].filter(
      (row) => row.node_id !== connection.node_id || row.input_name !== connection.input_name,
    );
    if (checked) rows.push({ node_id: connection.node_id, input_name: connection.input_name });
    changeDraft({ lora_insertion: { ...draft.lora_insertion, [key]: rows } });
  }
</script>

<details>
  <summary>Insertion boundary for a workflow without LoRA slots</summary>
  <p class="muted">
    Select the exact connected consumers that should receive the stack. Other branches stay
    unchanged. Existing single LoRA chains extend at their tail automatically.
  </p>
  <label
    >Loader type<select
      value={draft.lora_insertion?.loader ?? ''}
      onchange={(event) => chooseLoader(event.currentTarget.value)}
      ><option value="">Use existing chain</option>{#each loaders as loader}<option>{loader}</option
        >{/each}</select
    ></label
  >
  {#if draft.lora_insertion}
    {#each compatible as connection}
      {@const key = connection.type === 'CLIP' ? 'clip_targets' : 'model_targets'}
      <label class="check-line"
        ><input
          type="checkbox"
          checked={draft.lora_insertion[key].some(
            (row) => row.node_id === connection.node_id && row.input_name === connection.input_name,
          )}
          onchange={(event) => toggle(connection, event.currentTarget.checked)}
        />{connection.type}: {connection.node_id}.{connection.input_name} ← {connection.source.join(
          ':',
        )}</label
      >
    {/each}
    <p class="muted">
      Each selected input type must share one source. The backend validates model and text-encoder
      compatibility at preview and execution.
    </p>
  {/if}
</details>
