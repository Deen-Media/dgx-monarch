<script lang="ts">
  import type { Binding } from '../model';
  import type { Submission } from '../submission.svelte';
  import { reviewSummary, startReviewClock, submitBlock } from '../reviewSummary';
  let {
    submission,
    bindings,
    busy,
    submitPreview,
  }: {
    submission: Pick<Submission, 'preview' | 'pending' | 'confirmed' | 'arrivedAt'>;
    /** The reviewed draft's bindings: a draft edit clears the review, so they match it. */
    bindings: Binding[];
    busy: boolean;
    submitPreview: () => Promise<void>;
  } = $props();
  let previewDialog: HTMLDialogElement;
  // The record of one review opens on request; a new review starts with it closed.
  let recordFor = $state('');
  let isOpen = $state(false);
  // The companion's clock while the dialog is open, so the review says when it has expired.
  let now = $state<number | null>(null);
  const preview = $derived(submission.preview);
  const pending = $derived(!!submission.pending);
  const summary = $derived(preview && reviewSummary(preview, bindings, pending, busy, now));
  const blocked = $derived(
    preview ? submitBlock(preview, { confirmed: submission.confirmed, busy, pending, now }) : '',
  );
  $effect(() => {
    if (!preview) previewDialog?.close();
  });
  $effect(() => {
    if (!isOpen || !preview) return;
    return startReviewClock(preview, submission.arrivedAt, (value) => (now = value));
  });
  export function open() {
    previewDialog.showModal();
    isOpen = true;
  }
  /** Every outcome closes the dialog, so a refusal or an unknown outcome shows its banner. */
  async function submit() {
    try {
      await submitPreview();
    } finally {
      previewDialog.close();
    }
  }
</script>

<dialog
  bind:this={previewDialog}
  class="preview-dialog"
  aria-labelledby="preview-title"
  onclose={() => (isOpen = false)}
>
  <div class="section-heading">
    <div>
      <span class="eyebrow">IMMUTABLE SUBMISSION</span>
      <h2 id="preview-title">Review generation</h2>
    </div>
    <button onclick={() => previewDialog.close()} aria-label="Close review">×</button>
  </div>
  {#if preview && summary}
    <p class="review-status" class:ready={summary.ready} role="status">{summary.status}</p>
    <dl class="review-summary">
      <dt>Workflow</dt>
      <dd>{summary.workflow}</dd>
      <dt>Model files</dt>
      <dd>
        {#each summary.models as model}<p>{model}</p>{:else}<p class="muted">
            No model file input.
          </p>{/each}
      </dd>
      <dt>Seeds</dt>
      <dd>
        <p class="mono">{summary.seeds}</p>
        {#if summary.seedSource}<p class="muted">{summary.seedSource}</p>{/if}
      </dd>
      <dt>Prompt text</dt>
      <dd>
        {#each summary.texts as entry}<p class="muted">{entry.label}</p>
          <p class="review-text">{entry.text || '(empty)'}</p>
          {#if entry.draft !== undefined}<p class="muted">Your draft text before review:</p>
            <p class="review-text">{entry.draft || '(empty)'}</p>{/if}
        {:else}<p class="muted">No text input.</p>{/each}
        {#if summary.styles}<p class="muted">{summary.styles}</p>{/if}
      </dd>
      <dt>Topology</dt>
      <dd>
        {#each summary.topology as entry}<p>{entry}</p>{:else}<p class="muted">
            No topology input.
          </p>{/each}
        <p class="muted">The backend sets the topology that runs.</p>
      </dd>
      {#if summary.warnings.length}<dt>Checks still to run</dt>
        <dd class="warning">
          {#each summary.warnings as warning}<p>{warning}</p>{/each}
        </dd>{/if}
      {#if summary.expiry}<dt>Expiry</dt>
        <dd>{summary.expiry}</dd>{/if}
    </dl>
    <details
      class="review-record"
      open={recordFor === preview.id}
      ontoggle={(event) => (recordFor = event.currentTarget.open ? preview.id : '')}
    >
      <summary>Immutable submission record (JSON)</summary>
      {#if recordFor === preview.id}<pre>{JSON.stringify(preview, null, 2)}</pre>{/if}
    </details>
    <label class="check-line"
      ><input type="checkbox" bind:checked={submission.confirmed} /> I reviewed the effective settings
      and resource warnings.</label
    >
    <div class="dialog-actions">
      <button onclick={() => previewDialog.close()}>Return to draft</button><button
        class="primary"
        disabled={!!blocked}
        aria-describedby={blocked ? 'submit-block' : undefined}
        onclick={submit}>{busy ? 'Submitting…' : 'Submit reviewed job'}</button
      >
    </div>
    {#if blocked}<p class="muted submit-block" id="submit-block">{blocked}</p>{/if}
  {/if}
</dialog>
