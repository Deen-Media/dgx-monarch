<script lang="ts">
  import type { JobPreview } from '../contracts';
  let {
    preview,
    previewKind,
    busy,
    pending,
    confirmed = $bindable(),
    submitPreview,
  }: {
    preview: JobPreview | null;
    previewKind: string;
    busy: boolean;
    pending: boolean;
    confirmed: boolean;
    submitPreview: () => Promise<void>;
  } = $props();
  let previewDialog: HTMLDialogElement;
  $effect(() => {
    if (!preview) previewDialog?.close();
  });
  export function open() {
    previewDialog.showModal();
  }
  export function close() {
    previewDialog.close();
  }
</script>

<dialog bind:this={previewDialog} class="preview-dialog" aria-labelledby="preview-title">
  <div class="section-heading">
    <div>
      <span class="eyebrow">IMMUTABLE SUBMISSION</span>
      <h2 id="preview-title">Review {previewKind}</h2>
    </div>
    <button onclick={() => previewDialog.close()} aria-label="Close review">×</button>
  </div>
  {#if preview}<p class="muted">
      Review requested and effective values, resolved prompt, target and backend warnings. This
      review applies to this exact draft snapshot.
    </p>
    {#if preview.errors?.length}<div class="validation-list">
        {#each preview.errors as issue}<p>{issue}</p>{/each}
      </div>{/if}{#if preview.warnings?.length}<div class="warning">
        {#each preview.warnings as warning}<p>
            {typeof warning === 'string' ? warning : JSON.stringify(warning)}
          </p>{/each}
      </div>{/if}
    <pre>{JSON.stringify(preview, null, 2)}</pre>
    <label class="check-line"
      ><input type="checkbox" bind:checked={confirmed} /> I reviewed the effective settings and resource
      warnings.</label
    >
    <div class="dialog-actions">
      <button onclick={() => previewDialog.close()}>Return to draft</button><button
        class="primary"
        disabled={!confirmed || busy || !!preview.errors?.length || pending}
        onclick={submitPreview}>{busy ? 'Submitting…' : 'Submit reviewed job'}</button
      >
    </div>{/if}
</dialog>
