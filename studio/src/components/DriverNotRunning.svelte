<script lang="ts">
  import type { Connection } from '../connection.svelte';
  import { sourceText } from '../firstRun';
  let { connection, connect }: { connection: Connection; connect: () => Promise<void> } = $props();
  const waiting = $derived(connection.connected && connection.driverNotRunning);
  const origin = $derived(connection.capabilities?.driver_origin || 'the driver origin');
  const start = $derived(connection.capabilities?.driver_start);
  const fromOperations = $derived(
    !!connection.capabilities?.operations?.some((op) => op.id === 'driver.start' && op.available),
  );
  let command = $state<HTMLElement>();
  // Once shown, connect again when the driver answers: that reads the catalog and inspects the
  // restored draft. The panel stays mounted, so it sees the snapshot that hides it.
  let shown = false;
  $effect(() => {
    if (!connection.connected) shown = false;
    else if (waiting) shown = true;
    else if (shown && connection.snapshot.driver?.state === 'connected') {
      shown = false;
      void connect();
    }
  });
  async function copy() {
    if (command) getSelection()?.selectAllChildren(command);
    try {
      await navigator.clipboard.writeText(start?.command ?? '');
    } catch {
      // The command stays selected for a manual copy.
    }
  }
</script>

{#if waiting}<section class="notice driver-setup" role="status" aria-live="polite">
    <h2>Driver not running</h2>
    <p>Studio is connected, but nothing accepts a connection at {origin}.</p>
    <p>
      If ComfyUI already runs at another address, set driver in the [studio] table of cluster.toml
      to that address and restart Studio. These records belong to {origin}; a new address needs
      fresh records, as the Studio guide explains under Configure Studio in cluster.toml.
    </p>
    {#if start?.command}<p>Start it on the Studio host:</p>
      <div class="driver-command">
        <code bind:this={command}>{start.command}</code><button onclick={copy}>Copy command</button>
      </div>
      <p class="muted">{sourceText(start.source, start.config, start.command)}</p>
    {:else}<p>
        Start your ComfyUI driver so it listens at {origin}. To show your own start command here,
        set driver_command in the [studio] table of cluster.toml.
      </p>{/if}
    {#if fromOperations}<p>You can also start it from Operations.</p>{/if}
    <p class="muted">
      Studio checks every few seconds and connects by itself when the driver answers.
    </p>
  </section>{/if}
