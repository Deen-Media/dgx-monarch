<script lang="ts">
  import { onMount } from 'svelte';
  import type { StudioClient } from '../api';
  import type { Capabilities, OperationRecord } from '../contracts';
  import Field from './Field.svelte';
  import { formatTime } from '../model';
  let {
    client,
    capabilities,
    operations = [],
    initialAction = '',
    onrefresh,
  }: {
    client: StudioClient;
    capabilities: Capabilities;
    operations?: OperationRecord[];
    initialAction?: string;
    onrefresh: () => void;
  } = $props();
  let action = $state(''),
    parameters = $state<Record<string, unknown>>({}),
    preview = $state<OperationRecord | null>(null),
    error = $state(''),
    busy = $state(false),
    confirmed = $state(false);
  let options = $derived(capabilities.operations ?? []);
  let current = $derived(options.find((o) => o.id === action));
  let requestKey = '';
  const reviewStorageKey = 'dgxm.studio.operation-review.v1';
  function rememberReview() {
    try {
      sessionStorage.setItem(
        reviewStorageKey,
        JSON.stringify({ action, parameters, preview, requestKey }),
      );
    } catch {
      error =
        'Operation review could not be saved locally. Keep this view open until its result is known.';
    }
  }
  onMount(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(reviewStorageKey) ?? 'null');
      if (
        saved &&
        (!initialAction || initialAction === saved.action) &&
        options.some((o) => o.id === saved.action)
      ) {
        action = saved.action;
        parameters = saved.parameters;
        preview = saved.preview;
        requestKey = saved.requestKey;
      }
    } catch {
      error =
        'A saved operation review could not be restored. Inspect durable receipts before retrying.';
    }
  });
  $effect(() => {
    if (initialAction) {
      action = initialAction;
      parameters = Object.fromEntries(
        (options.find((o) => o.id === initialAction)?.parameters ?? []).map((field) => [
          field.name,
          field.default ?? '',
        ]),
      );
      preview = null;
    }
  });
  async function review() {
    busy = true;
    error = '';
    preview = null;
    confirmed = false;
    try {
      preview = await client.request('/operations/preview', { action, parameters });
      requestKey = crypto.randomUUID();
      rememberReview();
    } catch (e: any) {
      error = e.message;
    } finally {
      busy = false;
    }
  }
  async function apply() {
    if (!preview || (preview.requires_confirmation && !confirmed)) return;
    busy = true;
    error = '';
    try {
      await client.request(`/operations/${encodeURIComponent(preview.id)}/apply`, {
        digest: preview.digest,
        confirmed: true,
        idempotency_key: requestKey,
      });
      preview = null;
      sessionStorage.removeItem(reviewStorageKey);
      onrefresh();
    } catch (e: any) {
      error = `Operation result: ${e.message}. Inspect the receipts before retrying; the same request identity is retained.`;
      onrefresh();
    } finally {
      busy = false;
    }
  }
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">HOST MANAGEMENT</span>
      <h2>Operations & diagnostics</h2>
    </div>
    <span class="tag">Reviewed actions</span>
  </div>
  <p class="muted">
    Review the authoritative plan, guards and exact digest before applying a host operation. Driver
    and companion have separate lifecycles.
  </p>
  <div class="operation-grid">
    <div class="surface">
      <label for="operation">Supported operation</label>
      <select
        id="operation"
        bind:value={action}
        onchange={() => {
          preview = null;
          confirmed = false;
          parameters = Object.fromEntries(
            (options.find((o) => o.id === action)?.parameters ?? []).map((field) => [
              field.name,
              field.default ?? '',
            ]),
          );
        }}
        ><option value="">Choose an operation</option>{#each options as option}<option
            value={option.id}
            >{option.label ?? option.id}{option.available === false ? ' - unavailable' : ''}</option
          >{/each}</select
      >
      {#if current}
        <p class:warning={current.available === false}>
          {current.reason ??
            current.description ??
            'Availability and guards are checked during preview.'}
        </p>
        {#each current.parameters ?? [] as field}<Field
            binding={{
              id: `operation-${field.name}`,
              label: field.name,
              type: field.type,
              required: field.required,
            }}
            value={parameters[field.name]}
            onchange={(value) => {
              parameters = { ...parameters, [field.name]: value };
              preview = null;
            }}
          />{/each}
        {#if !(current.parameters ?? []).length}<small
            >This action has no editable parameters.</small
          >{/if}
        {#if current.implementation === 'missing'}<p class="warning">
            Adapter not implemented in this draft.
          </p>{/if}
      {/if}
      <button
        class="primary"
        disabled={!action || current?.available === false || busy}
        onclick={review}>{busy ? 'Checking…' : 'Review operation'}</button
      >
      {#if error}<p class="error" role="alert">{error}</p>{/if}
    </div>
    <div class="surface">
      {#if preview}
        <h3>{preview.description ?? action}</h3>
        <span class="tag">{preview.available === false ? 'Unavailable' : 'Plan ready'}</span>
        {#if preview.reason}<p class="warning">{preview.reason}</p>{/if}
        <pre>{JSON.stringify(preview, null, 2)}</pre>
        {#if preview.requires_confirmation}<label class="check-line"
            ><input type="checkbox" bind:checked={confirmed} /> I reviewed this exact plan and authorize
            its application.</label
          >{/if}
        <button
          class="primary"
          disabled={(preview.requires_confirmation && !confirmed) ||
            busy ||
            preview.available === false}
          onclick={apply}>Apply reviewed plan</button
        >
      {:else}<div class="empty-state">
          <h3>Every action starts with a review.</h3>
          <p>The plan, consent requirements and refusal reasons appear here.</p>
        </div>{/if}
    </div>
  </div>
  <h3>Durable operation receipts</h3>
  {#if operations.length}{#each [...operations].reverse() as operation}<details class="record">
        <summary
          ><span>{operation.action ?? operation.id}</span><span class="tag"
            >{operation.state ?? operation.status ?? 'Recorded'}</span
          ><span>{formatTime(operation.updated_at ?? operation.created_at)}</span></summary
        >
        <pre>{JSON.stringify(operation, null, 2)}</pre>
      </details>{/each}{:else}<p class="muted">No operations recorded by this companion.</p>{/if}
</section>
