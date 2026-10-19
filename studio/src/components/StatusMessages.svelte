<script lang="ts">
  import type { Connection } from '../connection.svelte';
  import type { Submission } from '../submission.svelte';
  let {
    messages,
    connection,
    submission,
    busy,
    ondismiss,
    onexport,
    onkeep,
    onload,
    connect,
    reconcilePending,
    retryPending,
  }: {
    messages: { notice: string; error: string; localError: string; draftConflict: boolean };
    connection: Connection;
    submission: Submission;
    busy: boolean;
    ondismiss: (kind: 'notice' | 'error') => void;
    onexport: () => void;
    onkeep: () => void;
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
    {messages.localError}<button onclick={() => onexport()}>Export draft</button>
  </div>{/if}
{#if messages.draftConflict}<div class="notice warning" role="alert">
    <span>Another tab saved a newer draft. Automatic saves are paused.</span><button
      onclick={onkeep}>Export this tab, then keep it</button
    ><button onclick={onload}>Load newer draft</button>
  </div>{/if}
{#if connection.connected && connection.state !== 'connected'}<div class="notice warning">
    <span>Connection stale. Last sample {connection.lastSeen || 'unknown'}. {connection.error}</span
    ><button onclick={() => void connect()}>Reconnect</button>
  </div>{/if}
{#if submission.pending}<div class="notice warning" role="alert">
    <span>A submission has an unresolved outcome. Its request identity is retained.</span><button
      disabled={busy}
      onclick={reconcilePending}>Reconcile request</button
    >{#if submission.confirmedMissing}<button disabled={busy} onclick={retryPending}
        >Retry retained request</button
      >{/if}
  </div>{/if}
