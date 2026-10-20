<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Artifact } from '../contracts';
  import { exportJSON } from '../files';
  let {
    client,
    previewId,
    jobId,
    onview,
    oncompare,
  }: {
    client: StudioClient;
    previewId?: string;
    jobId?: string;
    onview: (artifact: Artifact) => void;
    oncompare: (artifact: Artifact) => void;
  } = $props();
  let detail = $state<unknown>(),
    error = $state(''),
    busy = $state(false),
    outputs = $state<Artifact[]>([]);
  let cursor = $state<string | null>(null),
    nextCursor = $state<string | null>(null),
    pageNumber = $state(1);
  async function loadOutputs(next: string | null, number: number) {
    if (!jobId) return;
    const page = await client.request<{ items: Artifact[]; next_cursor: string | null }>(
      `/artifacts?limit=100&job_id=${encodeURIComponent(jobId)}${next ? '&cursor=' + encodeURIComponent(next) : ''}`,
    );
    outputs = page.items;
    cursor = next;
    nextCursor = page.next_cursor;
    pageNumber = number;
  }
  async function outputPage(next: string | null, number: number) {
    if (busy) return;
    busy = true;
    error = '';
    try {
      await loadOutputs(next, number);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      busy = false;
    }
  }
  async function inspect() {
    if (!jobId && !previewId) return;
    busy = true;
    error = '';
    try {
      detail = await client.request(
        jobId
          ? `/jobs/${encodeURIComponent(jobId)}`
          : `/jobs/previews/${encodeURIComponent(previewId!)}`,
      );
      if (jobId) await loadOutputs(null, 1);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      busy = false;
    }
  }
</script>

<button disabled={busy || (!jobId && !previewId)} onclick={inspect}
  >{busy ? 'Loading…' : 'Inspect full submission'}</button
>
{#if error}<p class="error">{error}</p>{/if}
{#if detail}<details open>
    <summary>Requested and effective settings, topology, workflow and outcome</summary>
    <pre>{JSON.stringify(detail, null, 2)}</pre>
    <button onclick={() => exportJSON(detail, `grid-cell-${jobId ?? previewId}.json`)}
      >Export cell</button
    >
  </details>{/if}
{#each outputs as artifact}<div class="workflow-actions">
    <span>{artifact.filename}</span><button onclick={() => onview(artifact)}>View</button><button
      onclick={() => oncompare(artifact)}>Compare</button
    >
  </div>{/each}
{#if cursor || nextCursor}<div class="pagination">
    <button disabled={busy || !cursor} onclick={() => void outputPage(null, 1)}
      >First outputs</button
    >
    <span>Output page {pageNumber} · up to 100 outputs</span>
    <button
      disabled={busy || !nextCursor}
      onclick={() => void outputPage(nextCursor, pageNumber + 1)}>Next outputs</button
    >
  </div>{/if}
