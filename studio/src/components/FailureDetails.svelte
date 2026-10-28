<script lang="ts">
  import {
    CLASS_TITLES,
    TROUBLESHOOTING,
    type ConsentAction,
    type JobFailure,
  } from '../jobFailure';
  import ConsentOffer from './ConsentOffer.svelte';
  let {
    failure,
    consent = null,
    onconsent,
  }: {
    failure: JobFailure;
    consent?: ConsentAction | null;
    onconsent?: () => void;
  } = $props();
</script>

<section class="failure-details" aria-label="Failure reason">
  <span class="eyebrow">FAILURE</span>
  {#if failure.refusalClass || failure.code}<dl class="failure-facts">
      {#if failure.refusalClass}<dt>Refusal</dt>
        <dd>
          class {failure.refusalClass}, {CLASS_TITLES[failure.refusalClass]}{#if failure.guard},
            guard
            <span class="mono">{failure.guard}</span>{failure.waivable ? ', waivable' : ''}{/if}
        </dd>{/if}
      {#if failure.code}<dt>Code</dt>
        <dd class="mono">{failure.code}</dd>{/if}
    </dl>{/if}
  {#if failure.text}<p class="failure-text">{failure.text}</p>{/if}
  {#if failure.troubleshooting !== null}<p>
      <a
        href={TROUBLESHOOTING}
        target="_blank"
        rel="noopener noreferrer"
        title="Opens docs/TROUBLESHOOTING.md; find entry {failure.troubleshooting} there"
        >Troubleshooting #{failure.troubleshooting}</a
      >
    </p>{/if}
  {#if failure.consent}<div class="failure-consent">
      <p><ConsentOffer consent={failure.consent} action={consent} {onconsent} /></p>
      <dl class="failure-facts">
        <dt>Consent kind</dt>
        <dd class="mono">{failure.consent.kind}</dd>
        <dt>Model</dt>
        <dd class="mono">{failure.consent.unetName}</dd>
        <dt>Consent id</dt>
        <dd class="mono">{failure.consent.consentId}</dd>
      </dl>
    </div>{/if}
</section>
