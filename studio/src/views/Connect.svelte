<script lang="ts">
  let {
    token = $bindable(),
    busy,
    error = '',
    connect,
  }: { token: string; busy: boolean; error?: string; connect: () => Promise<void> } = $props();
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
      Open Studio with the sign-in link that <code>dgxm studio</code> prints or opens; to get a new
      one, run <code>dgxm studio open</code> on the Studio host. You can also paste the link or the
      token here; a Studio started with <code>--allow-management</code> takes its token only here. It
      stays in this browser tab's session.
    </p>
    <label for="token">Launch token</label><input
      id="token"
      type="password"
      bind:value={token}
      autocomplete="off"
      required
      placeholder="Paste your launch token"
    />{#if error}<p class="error" role="alert">{error}</p>{/if}<button
      class="primary"
      disabled={busy || !token.trim()}>{busy ? 'Connecting…' : 'Open workspace →'}</button
    ><small
      >Host: {location.host}<br />If Studio is not running, start it on the host with
      <code>dgxm studio</code>, or keep it running with <code>dgxm studio up</code>. Each start
      makes a new token. This page cannot start Studio.</small
    >
  </form>
</section>
