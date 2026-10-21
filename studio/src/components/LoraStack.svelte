<script lang="ts">
  import type { Draft, Lora } from '../model';
  import type { Catalog, CatalogLora, ChangeDraft, GraphConnection } from '../contracts';
  import LoraInsertion from './LoraInsertion.svelte';
  import { numberField } from '../numberField';
  let {
    draft,
    catalog,
    changeDraft,
    connections,
  }: { draft: Draft; catalog: Catalog; changeDraft: ChangeDraft; connections: GraphConnection[] } =
    $props();
  let loraQuery = $state('');
  let loras = $derived(
    (catalog.loras ?? [])
      .filter((l) => !draft.lora_insertion || l.source_node === draft.lora_insertion.loader)
      .filter((l) => JSON.stringify(l).toLowerCase().includes(loraQuery.toLowerCase()))
      .slice(0, 100),
  );
  function loraSchema(item: { source_node?: string }, name: string) {
    return catalog.object_info?.[item.source_node ?? '']?.input?.required?.[name]?.[1] ?? {};
  }
  function addLora(item: CatalogLora) {
    const name = typeof item === 'string' ? item : (item.name ?? item.id);
    changeDraft({
      lora_stack: [
        ...draft.lora_stack,
        {
          id: typeof item === 'string' ? item : (item.id ?? name),
          name,
          enabled: true,
          source_node: item.source_node,
          model_strength: loraSchema(item, 'strength_model').default ?? 1,
          ...(item.source_node === 'LoraLoader'
            ? { clip_strength: loraSchema(item, 'strength_clip').default ?? 1 }
            : {}),
        },
      ],
    });
  }
  function updateLora(index: number, change: Partial<Lora>) {
    changeDraft({
      lora_stack: draft.lora_stack.map((l, i) => (i === index ? { ...l, ...change } : l)),
    });
  }
  function moveLora(index: number, offset: number) {
    const stack = [...draft.lora_stack];
    const [item] = stack.splice(index, 1);
    stack.splice(index + offset, 0, item);
    changeDraft({ lora_stack: stack });
  }
</script>

<h3>Studio stack override <span class="muted">{draft.lora_stack.length}</span></h3>
<p class="muted">
  With no Studio override, LoRA loaders and strengths already in the workflow remain active.
  Removing the last override restores those original workflow settings; it does not disable its
  LoRAs. Review All inputs and the effective submission to see inherited loaders.
</p>
<LoraInsertion {draft} {catalog} {connections} {changeDraft} />
{#each draft.lora_stack as lora, i}<div class="lora-card">
    <div class="stack-row">
      <label
        ><input
          type="checkbox"
          checked={lora.enabled}
          onchange={(e) => updateLora(i, { enabled: e.currentTarget.checked })}
        />{lora.name}</label
      ><button
        onclick={() =>
          changeDraft({
            lora_stack: draft.lora_stack.filter((_, index) => index !== i),
          })}
        aria-label={`Remove ${lora.name}`}>×</button
      >
    </div>
    <div class="input-row">
      <label
        >Model strength<input
          type="number"
          min={loraSchema(lora, 'strength_model').min}
          max={loraSchema(lora, 'strength_model').max}
          step={loraSchema(lora, 'strength_model').step ?? 'any'}
          use:numberField={{
            value: lora.model_strength,
            commit: (value) => value !== null && updateLora(i, { model_strength: value }),
          }}
        /></label
      >{#if lora.clip_strength !== undefined}<label
          >Text strength<input
            type="number"
            min={loraSchema(lora, 'strength_clip').min}
            max={loraSchema(lora, 'strength_clip').max}
            step={loraSchema(lora, 'strength_clip').step ?? 'any'}
            use:numberField={{
              value: lora.clip_strength,
              commit: (value) => value !== null && updateLora(i, { clip_strength: value }),
            }}
          /></label
        >{/if}
    </div>
    <div class="workflow-actions">
      <button disabled={i === 0} onclick={() => moveLora(i, -1)}>Move up</button><button
        disabled={i === draft.lora_stack.length - 1}
        onclick={() => moveLora(i, 1)}>Move down</button
      >
    </div>
  </div>{/each}
<label for="lora-search">Available LoRAs</label><input
  id="lora-search"
  type="search"
  placeholder="Search names and metadata…"
  bind:value={loraQuery}
/><small>Compatibility is verified by the backend; a filename is not evidence.</small>
{#each loras as lora}<button class="catalog-item" onclick={() => addLora(lora)}
    ><span>{lora.name ?? lora.id}<small>{lora.source_node}</small></span><span>+</span></button
  >{/each}{#if !loras.length}<p class="muted">No matching LoRAs reported by the driver.</p>{/if}
