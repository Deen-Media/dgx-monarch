<script lang="ts">
  import type { InspectWorkflow, OperationCapability, WorkflowEntry } from '../contracts';
  import { destinations } from '../navigation';
  import { loadWorkflow, paletteWorkflows, workflowTitle } from '../workflowChoice';
  let {
    operations,
    workflows,
    navigate,
    inspect,
    onoperation,
  }: {
    operations: OperationCapability[];
    workflows: WorkflowEntry[];
    navigate: (view: string) => void;
    inspect: InspectWorkflow;
    onoperation: (id: string) => void;
  } = $props();
  let paletteDialog: HTMLDialogElement,
    paletteInput: HTMLInputElement,
    paletteQuery = $state('');
  // Each utility tab is listed by name under its group, so Grids, Fleet and Operations stay one search away.
  let commands = $derived(
    destinations.filter((n) =>
      `${n.group ?? ''} ${n.label}`.toLowerCase().includes(paletteQuery.toLowerCase()),
    ),
  );
  let operationCommands = $derived(
    operations.filter((o) =>
      `${o.label} ${o.id}`.toLowerCase().includes(paletteQuery.toLowerCase()),
    ),
  );
  let workflowCommands = $derived(paletteWorkflows(workflows, paletteQuery));
  /** Open Create and load the workflow as choosing it in the workflow select does. */
  function loadWorkflowCommand(entry: WorkflowEntry) {
    navigate('workspace');
    void loadWorkflow(entry, inspect);
  }
  export function open() {
    paletteQuery = '';
    paletteDialog.showModal();
    requestAnimationFrame(() => paletteInput.focus());
  }
  export function close() {
    paletteDialog.close();
  }
</script>

<dialog bind:this={paletteDialog} class="command-dialog" aria-labelledby="palette-title">
  <h2 id="palette-title" class="sr-only">Command palette</h2>
  <div class="palette-search">
    <span aria-hidden="true">⌕</span><input
      bind:this={paletteInput}
      bind:value={paletteQuery}
      placeholder="Go to a view, operation or workflow…"
      aria-label="Search commands"
    /><button onclick={() => paletteDialog.close()} aria-label="Close command palette">Esc</button>
  </div>
  <p class="eyebrow">NAVIGATION</p>
  {#each commands as command}<button class="command-row" onclick={() => navigate(command.id)}
      ><span>{command.icon} {command.group ? `${command.group}: ` : ''}{command.label}</span><span
        >↵</span
      ></button
    >{/each}{#if !commands.length && !operationCommands.length && !workflowCommands.length}<p
      class="muted"
    >
      No matching commands.
    </p>{/if}{#if operationCommands.length}<p class="eyebrow">REVIEW AN OPERATION</p>
    {#each operationCommands as command}<button
        class="command-row"
        onclick={() => {
          onoperation(command.id);
        }}
        ><span
          >{command.label}<small
            >{command.available ? 'Review before applying' : command.reason}</small
          ></span
        ><span>↵</span></button
      >{/each}{/if}{#if workflowCommands.length || (workflows.length && !paletteQuery.trim())}<p
      class="eyebrow"
    >
      LOAD A WORKFLOW
    </p>
    {#if !workflowCommands.length}<p class="muted">
        Type a title or file name to find a packaged workflow.
      </p>{/if}
    {#each workflowCommands as entry (entry.id)}<button
        class="command-row"
        onclick={() => loadWorkflowCommand(entry)}
        ><span>Load workflow: {workflowTitle(entry)}<small>{entry.name}</small></span><span>↵</span
        ></button
      >{/each}{/if}
  <div class="palette-hint">Tab to choose · Enter to open · Esc to close</div>
</dialog>
