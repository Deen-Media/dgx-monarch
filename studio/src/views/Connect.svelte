<script lang="ts">
  let {
    token = $bindable(),
    busy,
    connect,
  }: { token: string; busy: boolean; connect: () => Promise<void> } = $props();
</script>

<section class="connect-screen">
  <div class="connect-art">
    <span class="eyebrow">YOUR MODELS. YOUR WORKSPACE.</span>
    <h1>Ideas take<br /><span>flight here.</span></h1>
    <p>
      Image, video and distributed generation.<br />One workspace for your Monarch host.
    </p>
  </div>
  <form
    class="connect-card"
    onsubmit={(e) => {
      e.preventDefault();
      void connect();
    }}
  >
    <span class="eyebrow">CONNECT TO COMPANION</span>
    <h2>Welcome to Studio</h2>
    <p class="muted">
      Use the launch token printed by your Monarch Studio companion. It stays in this browser tab's
      session.
    </p>
    <label for="token">Launch token</label><input
      id="token"
      type="password"
      bind:value={token}
      autocomplete="off"
      required
      placeholder="Paste your launch token"
    /><button class="primary" disabled={busy || !token.trim()}
      >{busy ? 'Connecting…' : 'Open workspace →'}</button
    ><small
      >Host: {location.host}<br />If the companion is unavailable, start its service on the host. An
      offline page cannot restart its own server.</small
    >
  </form>
</section>
