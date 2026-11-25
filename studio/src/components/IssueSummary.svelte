<script lang="ts">
  import { tick } from 'svelte';
  import { fieldTab, reveal, type Issue } from '../fieldIssues';
  let {
    issues,
    blocking,
    checking = false,
    controlTab = $bindable(),
  }: {
    /** The fields to fix first, then the workflow check's other issues. */
    issues: Issue[];
    /** How many of them keep Generate off: the fields the draft cannot submit as they are. */
    blocking: number;
    /** Create is checking the workflow again after an edit. */
    checking?: boolean;
    controlTab: string;
  } = $props();
  let first = $derived(issues[0]);
  // While Create checks the workflow again, the earlier list may be out of date.
  let quiet = $derived(checking && !blocking);
  let listOpen = $state(false);
  // A field to fix keeps Generate off; a workflow issue alone leaves it on, and Generate checks.
  let lead = $derived(
    quiet
      ? 'Checking the workflow again after your edit.'
      : `${blocking ? 'Fix' : 'Check'} ${first?.text ?? ''}`,
  );
  /**
   * Open the tab that lists the issue's field, then scroll to its control and focus it. The list
   * of issues closes first, so the fields get their room back.
   */
  async function show(issue: Issue) {
    const field = issue.field;
    listOpen = false;
    controlTab = field
      ? fieldTab(field, controlTab)
      : controlTab === 'advanced'
        ? 'advanced'
        : 'inputs';
    await tick();
    if (field && reveal(document, `field-${field.id}`)) return;
    if (field && controlTab !== 'advanced') {
      controlTab = 'advanced';
      await tick();
      if (reveal(document, `field-${field.id}`)) return;
    }
    reveal(document, 'inspection-issues');
  }
</script>

<div class="issue-summary" id="generate-issues">
  <p class="issue-line warning">
    <span>{lead}</span>{#if !quiet}<button onclick={() => show(first)}
        >{first.field ? 'Show field' : 'Show issues'}</button
      >{/if}
  </p>
  {#if issues.length > 1 && !quiet}<details bind:open={listOpen}>
      <summary>All {issues.length} issues</summary>
      <ul>
        {#each issues as issue, index (index)}<li>
            <span>{issue.text}</span><button onclick={() => show(issue)}
              >{issue.field ? 'Show field' : 'Show issues'}</button
            >
          </li>{/each}
      </ul>
    </details>{/if}
</div>

<style>
  .issue-summary {
    margin-top: 10px;
    font-size: 10px;
  }
  .issue-summary p {
    margin: 0 0 6px;
  }
  .issue-line,
  li {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }
  .issue-line span,
  li span {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  button {
    flex-shrink: 0;
    font-size: 10px;
    padding: 3px 8px;
  }
  ul {
    margin: 6px 0 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 6px;
    /* The fields keep some room above a long list. */
    max-height: 84px;
    overflow-y: auto;
  }
  summary {
    cursor: pointer;
  }
</style>
