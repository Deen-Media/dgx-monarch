<script lang="ts">
  import type { Connection } from '../connection.svelte';
  import type { Submission } from '../submission.svelte';
  import type { DraftAutosave } from '../draftAutosave.svelte';
  import type { Draft } from '../model';
  let {
    messages,
    connection,
    submission,
    autosave,
    draft,
    busy,
    ondismiss,
    onload,
    connect,
    reconcilePending,
    retryPending,
  }: {
    messages: { notice: string; error: string; localError: string };
    connection: Connection;
    submission: Submission;
    autosave: DraftAutosave;
    draft: Draft;
    busy: boolean;
    ondismiss: (kind: 'notice' | 'error') => void;
    onload: () => void;
    connect: () => Promise<void>;
    reconcilePending: () => Promise<void>;
    retryPending: () => Promise<void>;
  } = $props();
</script>

{#if messages.notice}<div class="notice" role="status">
    <span>{messages.notice}</span><button
      onclick={() => ondismiss('notice')}
      aria-label="Dismiss notification">×</button
    >
  </div>{/if}
{#if messages.error}<div class="notice error" role="alert">
    <span>{messages.error}</span><button
      onclick={() => ondismiss('error')}
      aria-label="Dismiss error">×</button
    >
  </div>{/if}
{#if messages.localError}<div class="notice warning" role="alert">
    <span>{messages.localError}</span>
  </div>{/if}
{#if autosave.retained}<div class="notice warning" role="alert">
    <span
      >The saved draft in this browser could not be restored. Automatic saves are paused so it is
      not overwritten.</span
    ><button onclick={() => autosave.exportSaved()}>Export saved record</button><button
      onclick={() => autosave.resume()}>Allow saving over it</button
    >
  </div>{/if}
{#if autosave.storageError}<div class="notice warning" role="alert">
    <span>{autosave.storageError}</span><button onclick={() => autosave.exportDraft(draft)}
      >Export this draft</button
    >
  </div>{/if}
{#if autosave.conflict}<div class="notice warning" role="alert">
    <span>Another tab saved a newer draft. Automatic saves are paused.</span><button
      onclick={() => autosave.keep(draft)}>Export this tab, then keep it</button
    ><button onclick={onload}>Load newer draft</button>
  </div>{/if}
{#if connection.connected && connection.state !== 'connected'}<div class="notice warning">
    <span>Connection stale. Last sample {connection.lastSeen || 'unknown'}. {connection.error}</span
    ><button onclick={() => void connect()}>Reconnect</button>
  </div>{/if}
{#if connection.connected && connection.snapshot.driver?.history?.state === 'unavailable'}<div
    class="notice warning"
    role="status"
  >
    Recent driver history is unavailable. Queue observations and saved records remain visible;
    completion and new output discovery may be delayed.
  </div>{/if}
{#if submission.pending}<div class="notice warning" role="alert">
    <span
      >{submission.outcome || 'A submission has an unresolved outcome.'} Its request identity is retained.</span
    ><button disabled={busy} onclick={reconcilePending}>Reconcile request</button
    >{#if submission.confirmedMissing}<button disabled={busy} onclick={retryPending}
        >Retry retained request</button
      >{/if}
  </div>{/if}
{#if submission.recoveryError}<div class="notice warning" role="alert">
    <span>{submission.recoveryError}</span><button
      disabled={busy}
      onclick={() => submission.clearUnreadableRequest()}
      >I checked Queue; clear unreadable request</button
    >
  </div>{/if}
