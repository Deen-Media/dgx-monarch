<script lang="ts">
  import { onMount } from 'svelte';
  import { loadRecordCatalog } from '../recordCatalog';
  import type { StudioClient } from '../api';
  import type { StoredRecord } from '../contracts';
  let {
    client,
    onchoose,
  }: { client: StudioClient; onchoose: (axis: 'A' | 'B', id: string) => void } = $props();
  let styles = $state<StoredRecord[]>([]),
    error = $state('');
  onMount(() => {
    void loadRecordCatalog(client, 'styles')
      .then((result) => (styles = result))
      .catch((cause) => (error = String(cause)));
  });
</script>

<details>
  <summary>Saved prompt-style axes</summary>
  <p>
    A style axis value is an ordered array of saved style IDs, for example [["ink"], ["ink",
    "night"]]. Each cell freezes its selected definitions and revisions. Set the prompt bindings in
    Create before validation.
  </p>
  {#each styles as style}<div class="stack-row">
      <span>{style.data.name ?? style.id} · {style.id}</span><button
        onclick={() => onchoose('A', style.id)}>Use as axis A</button
      ><button onclick={() => onchoose('B', style.id)}>Use as axis B</button>
    </div>{/each}
  {#if error}<p role="alert">{error}</p>{/if}
</details>

<style>
  /* A record id never breaks on its own, so the line wraps anywhere and the buttons stay in view. */
  .stack-row span {
    min-width: 0;
    overflow-wrap: anywhere;
  }
</style>
