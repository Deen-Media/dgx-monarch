<script lang="ts">
  import type { Connection } from '../connection.svelte';
  import type { Artifact, WorkflowEntry } from '../contracts';
  import type { Binding, Draft } from '../model';
  import { workflowTitle } from '../workflowChoice';
  import {
    bindOutput,
    NEXT_STEP,
    SEVERAL,
    sendToEdit,
    type EditActions,
    type EditContext,
    type EditOffer,
  } from '../editHandoff';
  let {
    connection,
    draft,
    bindings,
    workflowId,
  }: {
    connection: Connection;
    draft: Draft;
    bindings: Binding[];
    /** The packaged workflow the draft came from, or ''. */
    workflowId: string;
  } = $props();
  let dialog: HTMLDialogElement;
  let first = $state<HTMLDivElement>();
  // The output the dialog is about, the workflows it offers, or why it offers none.
  let output = $state<Artifact | null>(null);
  let choices = $state<WorkflowEntry[]>([]);
  let reason = $state('');
  let actions: EditActions | null = null;
  let binding = false;

  /** The live state Send to edit reads; each read is current. */
  function context(): EditContext {
    return {
      client: connection.client,
      info: () => connection.catalog.object_info,
      workflows: () => connection.workflows,
      workflowId: () => workflowId,
      draft: () => draft,
      bindings: () => bindings,
    };
  }

  /** App's actions, with one binding at a time: a second would find the workflow changed. */
  function guarded(app: EditActions): EditActions {
    return {
      ...app,
      withBusy: async (action) => {
        if (binding)
          return app.setNotice('Send to edit is still binding an output. Try again in a moment.');
        binding = true;
        try {
          await app.withBusy(action);
        } finally {
          binding = false;
        }
      },
    };
  }

  function offer(artifact: Artifact, plan: EditOffer) {
    output = artifact;
    choices = plan.kind === 'choose' ? plan.choices : [];
    reason = plan.kind === 'none' ? plan.reason : '';
    if (!dialog.open) dialog.showModal();
    requestAnimationFrame(() => first?.querySelector('button')?.focus());
  }

  /** Send to edit from the viewer or the gallery; see sendToEdit. */
  export function send(artifact: Artifact, app: EditActions) {
    actions = guarded(app);
    return sendToEdit(artifact, context(), actions, (plan) => offer(artifact, plan));
  }

  function choose(entry: WorkflowEntry) {
    const artifact = output;
    dialog.close();
    const app = actions;
    if (artifact && app) void app.withBusy(() => bindOutput(artifact, entry, context(), app));
  }

  function openCreate() {
    dialog.close();
    actions?.navigate('workspace');
  }
</script>

<dialog
  bind:this={dialog}
  class="edit-dialog"
  aria-labelledby="edit-title"
  onclose={() => (output = null)}
>
  <div class="section-heading">
    <div>
      <span class="eyebrow">SEND TO EDIT</span>
      <h2 id="edit-title">
        {choices.length ? 'Choose an editing workflow' : 'No workflow can edit this output'}
      </h2>
    </div>
    <button onclick={() => dialog.close()} aria-label="Close Send to edit">×</button>
  </div>
  {#if choices.length}
    <p>{SEVERAL}</p>
    <div class="edit-choices" bind:this={first}>
      {#each choices as entry (entry.id)}<button class="command-row" onclick={() => choose(entry)}
          ><span>{workflowTitle(entry)}<small class="mono">{entry.name}</small></span><span
            aria-hidden="true">↵</span
          ></button
        >{/each}
    </div>
    <div class="dialog-actions"><button onclick={() => dialog.close()}>Cancel</button></div>
  {:else}
    <p class="review-status decide" role="status">{reason}</p>
    <p>{NEXT_STEP}</p>
    <div class="dialog-actions" bind:this={first}>
      <button class="primary" onclick={openCreate}>Open Create</button><button
        onclick={() => dialog.close()}>Close</button
      >
    </div>
  {/if}
</dialog>

<style>
  .edit-dialog {
    width: min(540px, calc(100vw - 30px));
  }
  .edit-dialog p {
    font-size: 12px;
    line-height: 1.6;
    overflow-wrap: anywhere;
  }
  .edit-choices {
    display: grid;
    max-height: min(26rem, 55vh);
    overflow: auto;
    border-top: 1px solid var(--border);
  }
  .edit-choices .command-row {
    border-bottom: 1px solid var(--border);
    gap: 10px;
  }
</style>
