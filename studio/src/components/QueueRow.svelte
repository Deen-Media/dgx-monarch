<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Artifact, Job, OperationCapability } from '../contracts';
  import { formatTime } from '../model';
  import { recordedSeeds, seedText } from '../recordedSettings';
  import { consentAction, jobFailure } from '../jobFailure';
  import FailureDetails from './FailureDetails.svelte';
  import JobProvenance from './JobProvenance.svelte';
  import Thumbnail from './Thumbnail.svelte';
  let {
    client,
    job,
    output = null,
    open = false,
    busy,
    operations = [],
    ontoggle,
    jobAction,
    onconsent,
  }: {
    client: StudioClient;
    job: Job;
    /** The job's newest final output in the snapshot, or null; the row shows its thumbnail. */
    output?: Artifact | null;
    open?: boolean;
    busy: boolean;
    /** The companion's operation capabilities; consent.apply decides the consent control. */
    operations?: OperationCapability[];
    ontoggle: (open: boolean) => void;
    jobAction: (job: Job, action: 'cancel' | 'retry') => Promise<void>;
    onconsent?: (job: Job) => void;
  } = $props();
  const seeds = $derived(seedText(recordedSeeds(job)));
  // Only a failed or unresolved job's error explains its outcome.
  const failure = $derived(
    job.state === 'failed' || job.state === 'unknown' ? jobFailure(job.error) : null,
  );
  const consent = $derived(consentAction(failure, operations));
</script>

<details class="record job-row" {open} ontoggle={(e) => ontoggle(e.currentTarget.open)}>
  <summary
    >{#if output}<span class="job-thumb"><Thumbnail artifact={output} {client} /></span>{/if}<span
      class="status-label"
      class:running={job.state === 'running'}>{job.state ?? 'unknown'}</span
    ><strong>{job.submission?.requested?.operation ?? 'workflow'}</strong><span class="mono"
      >{job.id}</span
    >{#if seeds}<span class="mono">{seeds}</span>{/if}<span class="job-time"
      >{formatTime(job.created_at)}</span
    >{#if failure?.summary}<small class="job-reason">{failure.summary}</small>{/if}</summary
  >
  <div class="job-details">
    <div class="workflow-actions">
      <button
        disabled={busy || !['queued', 'running'].includes(job.state)}
        onclick={() => void jobAction(job, 'cancel')}>Request cancel</button
      ><button
        disabled={busy || !['failed', 'cancelled'].includes(job.state)}
        onclick={() => void jobAction(job, 'retry')}>Retry as new job</button
      >
    </div>
    <p class="muted">
      Cancellation is requested first; the backend confirms the final outcome. Retry creates a
      linked job.
    </p>
    {#if failure}<FailureDetails
        {failure}
        {consent}
        onconsent={onconsent ? () => onconsent(job) : undefined}
      />{/if}
    {#if job.progress?.max}<progress max={job.progress.max} value={job.progress.value ?? 0}
      ></progress><span
        >{job.progress.value ?? 0} / {job.progress.max} · node {job.progress.node ??
          'unknown'}</span
      >{:else if job.state === 'running'}<p>Running · completion percentage unavailable</p>{/if}
    {#if open}<JobProvenance {client} summary={job} />{/if}
  </div>
</details>
