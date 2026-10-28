<script lang="ts">
  import type { FollowedJob } from '../jobTracker.svelte';
  let { job = null, onqueue }: { job?: FollowedJob | null; onqueue?: () => void } = $props();
  const share = $derived(
    job?.progress && job.progress !== 'indeterminate'
      ? (100 * job.progress.value) / job.progress.max
      : 0,
  );
</script>

<div class="job-status" class:empty={!job} style:--job-color={job?.color}>
  {#if job}<p
      title={job.own
        ? 'The job this tab submitted'
        : 'The newest active job; this tab has submitted none'}
    >
      <span class="job-dot" aria-hidden="true"></span><strong>{job.headline}</strong
      >{#if job.detail}<span aria-hidden="true">·</span><span class="job-detail">{job.detail}</span
        >{/if}{#if job.link && onqueue}<button class="job-link" onclick={onqueue}>{job.link}</button
        >{/if}
    </p>
    {#if job.progress === 'indeterminate'}<div
        class="job-progress indeterminate"
        role="progressbar"
        aria-label="Job progress, no step count reported"
      >
        <span></span>
      </div>
    {:else if job.progress}<div
        class="job-progress"
        role="progressbar"
        aria-label="Steps of the node now reporting"
        aria-valuemin={0}
        aria-valuemax={job.progress.max}
        aria-valuenow={job.progress.value}
      >
        <span style:width={`${share}%`}></span>
      </div>{/if}{/if}
  <span class="sr-only" role="status">{job?.announcement ?? ''}</span>
</div>
