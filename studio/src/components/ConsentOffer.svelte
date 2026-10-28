<script lang="ts">
  import { CONSENT_HELP, type ConsentAction, type ConsentRequest } from '../jobFailure';
  let {
    consent,
    action = null,
    onconsent,
  }: {
    consent: ConsentRequest;
    action?: ConsentAction | null;
    /** Open Operations with the consent operation chosen. */
    onconsent?: () => void;
  } = $props();
</script>

<span class="consent-reason">Consent needed: {consent.humanReason}</span><span class="consent-panel"
  >The ComfyUI panel's button for it is “{consent.panelAction}”.</span
>{#if action && onconsent}<button
    class="job-link"
    title="Opens Operations with the consent operation chosen. You review the exact plan before anything applies."
    onclick={onconsent}>{action.label}</button
  >{/if}{#if action && !action.available}<span class="consent-unavailable"
    >{action.reason}
    <a href={CONSENT_HELP} target="_blank" rel="noopener noreferrer"
      >How to grant a consent from Studio</a
    ></span
  >{/if}
