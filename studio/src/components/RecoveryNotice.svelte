<script lang="ts">
  let { storageKey, check, onclear }: { storageKey: string; check: string; onclear: () => void } =
    $props();
  let error = $state('');
  function clear() {
    try {
      sessionStorage.removeItem(storageKey);
      onclear();
    } catch {
      error = 'Browser storage is unavailable. The saved identity remains protected.';
    }
  }
</script>

<div class="surface" role="alert">
  <p>
    A saved request could not be restored. {check} before clearing it. New requests are blocked until
    you check.
  </p>
  <button onclick={clear}>I checked the durable records; clear unreadable request</button>
  {#if error}<p class="error">{error}</p>{/if}
</div>
