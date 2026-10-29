<script lang="ts">
  import type { Submission } from '../submission.svelte';
  import { adjustTarget } from '../draftRefusal';
  import { CHECKS, reviewSummary, startReviewClock, submitBlock } from '../reviewSummary';
  let {
    submission,
    busy,
    submitPreview,
    controlTab = $bindable(),
  }: {
    submission: Pick<
      Submission,
      'preview' | 'refusal' | 'reviewed' | 'pending' | 'acknowledged' | 'arrivedAt' | 'acknowledge'
    >;
    busy: boolean;
    submitPreview: () => Promise<void>;
    /** The Create tab; a refusal's button opens the one that holds what it names. */
    controlTab?: string;
  } = $props();
  let previewDialog: HTMLDialogElement;
  // The record of one review opens on request; a new review starts with it closed.
  let recordFor = $state('');
  let isOpen = $state(false);
  // The companion's clock while the dialog is open, so the review says when it has expired.
  let now = $state<number | null>(null);
  const preview = $derived(submission.preview);
  const refusal = $derived(submission.refusal);
  const target = $derived(refusal && adjustTarget(refusal));
  const pending = $derived(!!submission.pending);
  const acknowledged = $derived(submission.acknowledged);
  const summary = $derived(
    preview && reviewSummary(preview, submission.reviewed, pending, busy, now, acknowledged),
  );
  const blocked = $derived(
    preview ? submitBlock(preview, { acknowledged, busy, pending, now }) : '',
  );
  const title = $derived(
    refusal || preview?.errors?.length
      ? 'Fix the draft'
      : summary?.warnings.length
        ? 'Check before generating'
        : 'Review settings',
  );
  $effect(() => {
    if (!preview && !refusal) previewDialog?.close();
  });
  $effect(() => {
    if (!isOpen || !preview) return;
    return startReviewClock(preview, submission.arrivedAt, (value) => (now = value));
  });
  export function open() {
    if (!previewDialog.open) previewDialog.showModal();
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
  /** Open the Create tab that holds what the refusal names. */
  function adjust(tab: string) {
    controlTab = tab;
    previewDialog.close();
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
      <span class="eyebrow">{refusal ? 'NEEDS ATTENTION' : 'IMMUTABLE SUBMISSION'}</span>
      <h2 id="preview-title">{title}</h2>
    </div>
    <button onclick={() => previewDialog.close()} aria-label="Close review">×</button>
  </div>
  {#if refusal}
    <p class="review-status" role="status">
      The companion refused this draft, so no job started. Change what it names, then generate
      again.
    </p>
    <p class="review-refusal">{refusal.message}</p>
    <p class="muted">Error code {refusal.code}</p>
    <div class="dialog-actions">
      <button onclick={() => previewDialog.close()}>Return to draft</button>{#if target}<button
          class="primary"
          onclick={() => adjust(target.tab)}>{target.label}</button
        >{/if}
    </div>
  {:else if preview && summary}
    <p class="review-status {summary.tone}" role="status">{summary.status}</p>
    {#if summary.warnings.length}<fieldset class="review-decisions">
        <legend>Needs your decision</legend>
        <p class="muted">Tick a warning to generate with it, or adjust the draft.</p>
        {#each summary.warnings as warning, index}<label class="check-line"
            ><input
              type="checkbox"
              checked={acknowledged[index] ?? false}
              onchange={(event) => submission.acknowledge(index, event.currentTarget.checked)}
            />
            {warning.accept}</label
          >{/each}
      </fieldset>{/if}
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
      <dt>Checks</dt>
      <dd>{CHECKS}</dd>
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
    <div class="dialog-actions">
      <button onclick={() => previewDialog.close()}
        >{summary.warnings.length ? 'Adjust the draft' : 'Return to draft'}</button
      ><button
        class="primary"
        disabled={!!blocked}
        aria-describedby={blocked ? 'submit-block' : undefined}
        onclick={submit}>{busy ? 'Submitting…' : 'Generate'}</button
      >
    </div>
    {#if blocked}<p class="muted submit-block" id="submit-block">{blocked}</p>{/if}
  {/if}
</dialog>
