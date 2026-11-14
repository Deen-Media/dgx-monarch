<script lang="ts">
  let {
    ids,
    clear,
  }: {
    /** The outputs and uploads this draft descends from, by id. */
    ids: string[];
    clear: () => void;
  } = $props();
  let shown: HTMLElement[] = $state([]);
  let status = $state('');
  async function copy(id: string, index: number) {
    try {
      await navigator.clipboard.writeText(id);
      status = 'Copied the whole id.';
    } catch {
      // The id stays selected for a manual copy.
      if (shown[index]) getSelection()?.selectAllChildren(shown[index]);
      status = 'Studio could not copy; the id is selected for a manual copy.';
    }
  }
</script>

<div class="surface input-lineage">
  <h3>Input lineage</h3>
  <ul>
    {#each ids as id, index}<li>
        <code bind:this={shown[index]}>{id}</code><button
          type="button"
          title={id}
          aria-label={`Copy lineage id ${id}`}
          onclick={() => copy(id, index)}>Copy</button
        >
      </li>{/each}
  </ul>
  <small>Lineage records ancestry. Workflow asset controls specify the actual input.</small>
  <button onclick={clear}>Clear lineage</button>
  <small class="lineage-status" role="status">{status}</small>
</div>

<style>
  ul {
    list-style: none;
    margin: 0 0 10px;
    padding: 0;
  }
  li {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 6px;
  }
  code {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
    color: #bbc4cf;
  }
  li button {
    flex-shrink: 0;
    font-size: 11px;
    padding: 3px 8px;
  }
  small {
    display: block;
    margin-bottom: 10px;
  }
  .lineage-status:empty {
    margin: 0;
  }
</style>
