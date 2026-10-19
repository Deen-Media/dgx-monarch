<script lang="ts">
  import type { Artifact, Job } from '../contracts';
  import type { StudioClient } from '../api';
  import { parameterDifferences } from '../gallery';
  let { left, right, client }: { left: Artifact; right: Artifact; client: StudioClient } = $props();
  let pair = $state<Job[]>([]),
    error = $state('');
  $effect(() => {
    const ids = [left.job_id, right.job_id];
    let current = true;
    pair = [];
    error = '';
    Promise.all(ids.map((id) => client.request<Job>(`/jobs/${encodeURIComponent(id)}`)))
      .then((result) => {
        if (current) pair = result;
      })
      .catch((cause) => {
        if (current) error = String(cause);
      });
    return () => {
      current = false;
    };
  });
  const differences = $derived(
    pair.length === 2 ? parameterDifferences(pair[0].submission, pair[1].submission) : [],
  );
  const text = (value: unknown) => (value === undefined ? 'Not recorded' : JSON.stringify(value));
</script>

<details class="comparison-details">
  <summary>Parameter differences ({differences.length})</summary>
  <p>
    Recorded submission fields include prompts, seeds, model names, LoRA stacks, workflow and
    topology. Paths are preserved so similarly named inputs remain distinct.
  </p>
  {#if error}<p class="error">{error}</p>{:else if pair.length < 2}<p>
      Loading immutable submissions…
    </p>{:else}
    <div class="comparison-table">
      <table>
        <thead><tr><th>Field</th><th>{left.filename}</th><th>{right.filename}</th></tr></thead
        ><tbody
          >{#each differences as row}<tr
              ><td>{row.key}</td><td>{text(row.left)}</td><td>{text(row.right)}</td></tr
            >{/each}</tbody
        >
      </table>
    </div>
    {#if !differences.length}<p>Recorded submissions match.</p>{/if}
  {/if}
</details>

<style>
  .comparison-details {
    margin: 10px 0;
    font-size: 11px;
  }
  .comparison-table {
    overflow: auto;
    max-height: 280px;
  }
  table {
    border-collapse: collapse;
    width: 100%;
  }
  th,
  td {
    text-align: left;
    padding: 7px;
    border-bottom: 1px solid var(--border);
    overflow-wrap: anywhere;
    min-width: 150px;
  }
</style>
