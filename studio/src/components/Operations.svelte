<script lang="ts">
  import { onMount } from 'svelte';
  import { keepsReview, restoreSession, validOperation } from '../sessionRecovery';
  import RecoveryNotice from './RecoveryNotice.svelte';
  import { ApiError, requireRecordIdentity, type StudioClient } from '../api';
  import type { Capabilities, OperationRecord, OperationRequest } from '../contracts';
  import { CONSENT_HELP, CONSENT_OPERATION, consentMatch, type PendingCard } from '../jobFailure';
  import OperationReceipt from './OperationReceipt.svelte';
  import OperationField from './OperationField.svelte';
  import { formatTime } from '../model';
  let {
    client,
    capabilities,
    operations = [],
    request = null,
    onrefresh,
  }: {
    client: StudioClient;
    capabilities: Capabilities;
    operations?: OperationRecord[];
    /** The operation to open: a palette choice, or the consent a failed job asked for. */
    request?: OperationRequest | null;
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
  let pending = $state(false),
    recoveryBlocked = $state(false);
  let appliedId = '';
  let openReceipt = $state('');
  /** The consent the failed job asked for, while the consent operation it opened stays chosen. */
  let consent = $state<OperationRequest['consent'] | null>(null);
  const match = $derived(
    consent && preview?.action === CONSENT_OPERATION
      ? consentMatch(preview, consent.consentId)
      : null,
  );
  function defaults(id: string) {
    return Object.fromEntries(
      (options.find((o) => o.id === id)?.parameters ?? []).map((field) => [
        field.name,
        field.default ?? '',
      ]),
    );
  }
  /** Fill the key and id of the listed request that carries the job's consent id; review again. */
  function useCard(card: PendingCard) {
    parameters = { ...parameters, key: card.key, id: card.id };
    preview = null;
    confirmed = false;
  }
  const reviewStorageKey = 'dgxm.studio.operation-review.v1';
  function rememberReview() {
    try {
      sessionStorage.setItem(
        reviewStorageKey,
        JSON.stringify({ action, parameters, preview, requestKey, pending, request: appliedId }),
      );
    } catch {
      throw new Error(
        'Operation was not sent because its retry identity could not be saved locally.',
      );
    }
  }
  onMount(() => {
    const restored = restoreSession(reviewStorageKey, validOperation);
    recoveryBlocked = restored.blocked;
    const saved = restored.value;
    if (saved && keepsReview(saved, request, options)) {
      ({ action, parameters, preview, requestKey, pending } = saved);
      appliedId = request?.id ?? '';
      // The request that made this review brings its consent context back with it.
      if (request && saved.request === request.id) consent = request.consent ?? null;
    }
  });
  /** Choose the requested operation and fill its fields, once per request. */
  function open(next: OperationRequest) {
    appliedId = next.id;
    // Choosing the open operation again from the palette keeps its fields and review.
    if (next.action === action && !next.consent) return;
    action = next.action;
    parameters = { ...defaults(next.action), ...next.parameters };
    preview = null;
    confirmed = false;
    consent = next.consent ?? null;
  }
  $effect(() => {
    if (request && request.id !== appliedId && !pending) open(request);
  });
  async function review() {
    if (recoveryBlocked || pending || busy) return;
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
    if (recoveryBlocked || !preview || busy || (preview.requires_confirmation && !confirmed))
      return;
    busy = true;
    error = '';
    const wasPending = pending;
    try {
      pending = true;
      rememberReview();
    } catch (cause) {
      pending = wasPending;
      busy = false;
      error = String(cause);
      return;
    }
    try {
      requireRecordIdentity(
        await client.request(`/operations/${encodeURIComponent(preview.id)}/apply`, {
          digest: preview.digest,
          // The checkbox state, so the companion's own confirmation gate decides.
          confirmed,
          idempotency_key: requestKey,
        }),
      );
      pending = false;
      preview = null;
      sessionStorage.removeItem(reviewStorageKey);
      onrefresh();
    } catch (e: any) {
      if (
        e instanceof ApiError &&
        [
          'stale_operation',
          'config_changed',
          'service_changed',
          'confirmation',
          'operation_unavailable',
        ].includes(e.code ?? '')
      ) {
        pending = false;
        preview = null;
        sessionStorage.removeItem(reviewStorageKey);
        error = `${e.message}. The operation was not accepted; review a new plan.`;
        return;
      }
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
  {#if recoveryBlocked}<RecoveryNotice
      storageKey={reviewStorageKey}
      check="Inspect durable operation receipts and actual host state"
      onclear={() => (recoveryBlocked = false)}
    />{/if}
  <div class="operation-grid">
    <div class="surface">
      <fieldset disabled={recoveryBlocked || busy || pending}>
        <label for="operation">Supported operation</label>
        <select
          id="operation"
          bind:value={action}
          onchange={() => {
            preview = null;
            confirmed = false;
            parameters = defaults(action);
            if (action !== CONSENT_OPERATION) consent = null;
          }}
          ><option value="">Choose an operation</option>{#each options as option}<option
              value={option.id}
              >{option.label ?? option.id}{option.available === false
                ? ' - unavailable'
                : ''}</option
            >{/each}</select
        >
        {#if consent && action === CONSENT_OPERATION}<div class="consent-context" role="note">
            <strong>Job <span class="mono">{consent.jobId}</span> asked for consent</strong>
            <p>{consent.humanReason}</p>
            <p>
              The ComfyUI panel's button for it is “{consent.panelAction}”. Model
              <span class="mono">{consent.unetName}</span>, kind
              <span class="mono">{consent.kind}</span>, consent id
              <span class="mono">{consent.consentId}</span>.
            </p>
            {#if current?.available === false}<p>
                Studio cannot grant it here. Grant it in the ComfyUI panel, or read
                <a href={CONSENT_HELP} target="_blank" rel="noopener noreferrer"
                  >how to grant a consent from Studio</a
                >.
              </p>{:else}<p class="muted">
                Leave key and id empty: with one pending request the review selects it, and with
                several it lists each one with its consent id. After the grant succeeds, retry the
                job from Queue.
              </p>{/if}
          </div>{/if}
        {#if current}
          <p class:warning={current.available === false}>
            {current.reason ??
              current.description ??
              'Availability and guards are checked during preview.'}
          </p>
          {#each current.parameters ?? [] as field}<OperationField
              {field}
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
      </fieldset>
      <button
        class="primary"
        disabled={recoveryBlocked || pending || !action || current?.available === false || busy}
        onclick={review}>{busy ? 'Checking…' : 'Review operation'}</button
      >
      {#if pending}<p role="status">
          Operation outcome is pending. Retry the retained plan to reconcile the same request before
          choosing another operation.
        </p>{/if}
      {#if error}<p class="error" role="alert">{error}</p>{/if}
    </div>
    <div class="surface">
      {#if preview}
        <h3>{preview.description ?? action}</h3>
        <span class="tag">{preview.available === false ? 'Unavailable' : 'Plan ready'}</span>
        {#if preview.reason}<p class="warning">{preview.reason}</p>{/if}
        {#if consent && match}{#if 'chosen' in match}<p class:warning={!match.same}>
              {match.same
                ? `This is the request job ${consent.jobId} asked for.`
                : `This pending request does not carry the consent id of job ${consent.jobId}. Check its card before you apply it.`}
            </p>{:else if match.listed.length === 1}{@const card = match.listed[0]}
            <p>
              The driver lists the request job {consent.jobId} asked for.
              <button onclick={() => useCard(card)}>Use its key and id</button>
            </p>{:else if match.listed.length}<p class="warning">
              Several pending requests carry the consent id of job {consent.jobId}. Enter the key
              and id of the one to accept.
            </p>{:else}<p class="warning">
              No pending request carries the consent id of job {consent.jobId}. Retry the job to ask
              again.
            </p>{/if}{/if}
        <pre>{JSON.stringify(preview, null, 2)}</pre>
        {#if preview.requires_confirmation}<label class="check-line"
            ><input type="checkbox" bind:checked={confirmed} /> I reviewed this exact plan and authorize
            its application.</label
          >{/if}
        <button
          class="primary"
          disabled={recoveryBlocked ||
            (preview.requires_confirmation && !confirmed) ||
            busy ||
            preview.available === false}
          onclick={apply}>{pending ? 'Reconcile retained operation' : 'Apply reviewed plan'}</button
        >
      {:else}<div class="empty-state">
          <h3>Every action starts with a review.</h3>
          <p>The plan, consent requirements and refusal reasons appear here.</p>
        </div>{/if}
    </div>
  </div>
  <h3>Durable operation receipts</h3>
  {#if operations.length}{#each [...operations].reverse() as operation (operation.id)}<details
        class="record"
        open={openReceipt === operation.id}
        ontoggle={(event) => {
          if (event.currentTarget.open) openReceipt = operation.id;
          else if (openReceipt === operation.id) openReceipt = '';
        }}
      >
        <summary
          ><span>{operation.action ?? operation.id}</span><span class="tag"
            >{operation.state ?? operation.status ?? 'Recorded'}</span
          ><span>{formatTime(operation.updated_at ?? operation.created_at)}</span></summary
        >
        {#if openReceipt === operation.id}<OperationReceipt {client} {operation} />{/if}
      </details>{/each}{:else}<p class="muted">No operations recorded by this companion.</p>{/if}
</section>
