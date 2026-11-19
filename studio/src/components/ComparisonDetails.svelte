<script lang="ts">
  import type { Artifact, Job } from '../contracts';
  import type { StudioClient } from '../api';
  import { COMPARE_DEPTH, differenceText, parameterDifferences } from '../gallery';
  import { comparableSubmission, exactJob } from '../jobProvenance';
  import { keyedRecord } from '../keyedRecord.svelte';
  let { left, right, client }: { left: Artifact; right: Artifact; client: StudioClient } = $props();
  // Submissions are immutable, so the pair loads once per pair of job ids, not once per poll.
  // Exact records keep seeds above 2**53 - 1 apart, so adjacent seeds still show as a difference.
  const pair = keyedRecord<Job[]>(
    (key) => Promise.all(key.split('\n').map((id) => exactJob(client, id))),
    () => ({ id: `${left.job_id}\n${right.job_id}` }),
  );
  const differences = $derived(
    pair.value?.length === 2
      ? parameterDifferences(
          comparableSubmission(pair.value[0]),
          comparableSubmission(pair.value[1]),
        )
      : [],
  );
</script>

<details class="comparison-details">
  <summary>Parameter differences ({differences.length})</summary>
  <p>
    Recorded submission fields include prompts, seeds, model names, LoRA stacks, workflow and
    topology. Paths are preserved so similarly named inputs remain distinct. A retry shows the
    runtime and versions it recorded itself.
  </p>
  {#if pair.error}<p class="error">{pair.error}</p>{:else if pair.value?.length !== 2}<p>
      Loading immutable submissions…
    </p>{:else}
    <div class="comparison-table">
      <table>
        <thead><tr><th>Field</th><th>{left.filename}</th><th>{right.filename}</th></tr></thead
        ><tbody
          >{#each differences as row}<tr
              ><td>{row.key}{row.whole ? ' (compared as one value)' : ''}</td><td
                >{differenceText(row.left)}</td
              ><td>{differenceText(row.right)}</td></tr
            >{/each}</tbody
        >
      </table>
    </div>
    {#if differences.some((row) => row.whole)}<p>
        Fields nested deeper than {COMPARE_DEPTH} levels are compared as one value.
      </p>{/if}
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
