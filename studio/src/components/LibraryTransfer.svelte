<script lang="ts">
  import { onMount } from 'svelte';
  import { submissionRefused } from '../submissionRefusal';
  import { requireLibraryReceipt } from '../libraryImport';
  import type { StudioClient } from '../api';
  import { exportJSON } from '../files';
  import { restoreSession, validLibrary, type LibraryPlan as Plan } from '../sessionRecovery';
  import RecoveryNotice from './RecoveryNotice.svelte';
  let {
    client,
    projectId = '',
    onrefresh,
  }: { client: StudioClient; projectId?: string; onrefresh: () => Promise<void> } = $props();
  let document = $state<unknown>(null),
    name = $state(''),
    mode = $state('merge'),
    plan = $state<Plan | null>(null);
  let choices = $state<Record<string, { action: string; revision: number }>>({}),
    confirmed = $state(false),
    busy = $state(false),
    reading = $state(false),
    error = $state(''),
    notice = $state('');
  let key = '';
  let sourceGeneration = 0;
  let uncertain = $state(false),
    recoveryBlocked = $state(false);
  const storage = 'dgxm.studio.library.pending.v1';
  onMount(() => {
    const restored = restoreSession(storage, validLibrary);
    recoveryBlocked = restored.blocked;
    if (restored.value) {
      ({ plan, key, name } = restored.value);
      uncertain = true;
    }
  });
  async function read(file?: File) {
    if (recoveryBlocked || !file || uncertain) return;
    const ticket = ++sourceGeneration;
    reading = true;
    try {
      if (file.size > 4 * 1024 * 1024) throw new Error('Library imports are limited to 4 MiB.');
      const incoming = JSON.parse(await file.text());
      if (ticket !== sourceGeneration) return;
      document = incoming;
      name = file.name;
      plan = null;
      choices = {};
      confirmed = false;
      error = '';
    } catch (cause) {
      error = String(cause);
    } finally {
      if (ticket === sourceGeneration) reading = false;
    }
  }
  async function review() {
    if (recoveryBlocked || uncertain || busy || reading) return;
    const ticket = sourceGeneration;
    busy = true;
    error = '';
    confirmed = false;
    try {
      const reviewed = await client.request<Plan>('/library/import/preview', {
        library: $state.snapshot(document),
        mode,
        choices: $state.snapshot(choices),
      });
      if (ticket !== sourceGeneration) return;
      plan = reviewed;
      key = crypto.randomUUID();
    } catch (cause) {
      error = String(cause);
    } finally {
      busy = false;
    }
  }
  async function apply() {
    if (recoveryBlocked || reading || !plan || !confirmed || !plan.ready) return;
    busy = true;
    error = '';
    try {
      sessionStorage.setItem(storage, JSON.stringify({ plan: $state.snapshot(plan), key, name }));
    } catch {
      busy = false;
      error = 'The import was not sent because its retry identity could not be saved locally.';
      return;
    }
    uncertain = true;
    try {
      const result = await client.request<{ imported: number; repair_list: unknown[] }>(
        '/library/import',
        { preview_id: plan.id, digest: plan.digest, confirmed: true, idempotency_key: key },
      );
      requireLibraryReceipt(result);
      notice = `${result.imported} records imported; ${result.repair_list.length} unresolved source references. No jobs were dispatched.`;
      sessionStorage.removeItem(storage);
      uncertain = false;
      plan = null;
      document = null;
      name = '';
      await onrefresh();
    } catch (cause) {
      if (submissionRefused(cause, 'library')) {
        uncertain = false;
        sessionStorage.removeItem(storage);
      }
      error = `${String(cause)} ${uncertain ? 'Retry uses the same import identity.' : 'Review the import again before retrying.'}`;
    } finally {
      busy = false;
    }
  }
  async function download(scoped: boolean) {
    try {
      exportJSON(
        await client.request(
          `/library/export${scoped && projectId ? `?project_id=${encodeURIComponent(projectId)}` : ''}`,
        ),
        scoped ? 'monarch-project.json' : 'monarch-library.json',
      );
    } catch (cause) {
      error = String(cause);
    }
  }
</script>

<section class="surface" aria-label="Library import and export">
  <h2>Portable library</h2>
  {#if recoveryBlocked}<RecoveryNotice
      storageKey={storage}
      check="Inspect the project library and import receipts"
      onclear={() => (recoveryBlocked = false)}
    />{/if}
  <div class="workflow-actions">
    <button onclick={() => void download(false)}>Export complete library</button>
    <button disabled={!projectId} onclick={() => void download(true)}>Export active project</button>
    <label class="button import-label"
      >Import library<input
        type="file"
        accept=".json,application/json"
        disabled={recoveryBlocked || uncertain || busy || reading}
        onchange={(event) => void read(event.currentTarget.files?.[0])}
      /></label
    >
  </div>
  <p class="muted">
    Exports include record data and source references. Imported references do not copy media or
    dispatch jobs.
  </p>
  {#if document}
    <p>{name}</p>
    <label
      >Import identities<select
        bind:value={mode}
        disabled={recoveryBlocked || uncertain || busy || reading}
        onchange={() => {
          plan = null;
          choices = {};
        }}
        ><option value="merge">Preserve IDs and review conflicts</option><option value="copy"
          >Create copies with remapped project IDs</option
        ></select
      ></label
    >
    <button disabled={recoveryBlocked || busy || uncertain || reading} onclick={review}
      >Review library import</button
    >
  {/if}
  {#if plan}
    <p>
      {plan.changes.length} records reviewed · {plan.conflicts.length} unresolved conflicts · {plan
        .repair_list.length} source repairs
    </p>
    {#each plan.conflicts as conflict}
      <div class="surface">
        <h3>{conflict.kind} · {conflict.id}</h3>
        <label
          >Resolve {conflict.key}<select
            value={choices[conflict.key]?.action ?? ''}
            disabled={recoveryBlocked || busy || uncertain || reading}
            onchange={(event) => {
              choices = {
                ...choices,
                [conflict.key]: { action: event.currentTarget.value, revision: conflict.revision },
              };
              confirmed = false;
            }}
            ><option value="">Choose a resolution</option><option value="skip">Keep existing</option
            ><option value="replace">Replace reviewed revision {conflict.revision}</option></select
          ></label
        >
        <details>
          <summary>Compare record data</summary>
          <pre>{JSON.stringify(
              { current: conflict.current, incoming: conflict.incoming },
              null,
              2,
            )}</pre>
        </details>
      </div>
    {/each}
    {#if plan.conflicts.length}<button
        disabled={recoveryBlocked || uncertain || busy || reading}
        onclick={review}>Review conflict resolutions</button
      >{/if}
    {#if plan.repair_list.length}<details open>
        <summary>Source references requiring repair</summary>
        <pre>{JSON.stringify(plan.repair_list, null, 2)}</pre>
      </details>{/if}
    <label class="check-line"
      ><input type="checkbox" bind:checked={confirmed} disabled={!plan.ready} />I reviewed all
      changes and unresolved source references.</label
    >
    <button
      class="primary"
      disabled={recoveryBlocked || busy || reading || !confirmed || !plan.ready}
      onclick={apply}>Import reviewed library</button
    >
  {/if}
  {#if error}<p role="alert">{error}</p>{/if}{#if notice}<p role="status">{notice}</p>{/if}
</section>
