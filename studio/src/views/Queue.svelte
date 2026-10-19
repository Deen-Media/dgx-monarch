<script lang="ts">
  import type { Job } from '../contracts';
  import { formatTime } from '../model';
  import { exportJSON } from '../files';
  let {
    jobs,
    busy,
    refresh,
    fail,
    jobAction,
  }: {
    jobs: Job[];
    busy: boolean;
    refresh: () => Promise<void>;
    fail: (error: unknown) => void;
    jobAction: (job: Job, action: 'cancel' | 'retry') => Promise<void>;
  } = $props();
  let selectedJob = $state<Job | null>(null);
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">AUTHORITATIVE EXECUTION</span>
      <h1>Queue & history</h1>
    </div>
    <button onclick={() => void refresh().catch(fail)}>Refresh</button>
  </div>
  {#each [...jobs].reverse() as job}<details
      class="record"
      open={selectedJob?.id === job.id}
      ontoggle={(e) => {
        if (e.currentTarget.open) selectedJob = job;
      }}
    >
      <summary
        ><span class="status-label" class:running={job.state === 'running'}
          >{job.state ?? 'unknown'}</span
        ><strong>{job.submission?.requested?.operation ?? 'workflow'}</strong><span class="mono"
          >{job.id}</span
        ><span>{formatTime(job.created_at)}</span></summary
      >
      <div class="job-details">
        <div class="workflow-actions">
          <button
            disabled={busy || !['queued'].includes(job.state)}
            onclick={() => void jobAction(job, 'cancel')}>Request cancel</button
          ><button
            disabled={busy || !['failed', 'cancelled'].includes(job.state)}
            onclick={() => void jobAction(job, 'retry')}>Retry as new job</button
          ><button onclick={() => exportJSON(job, `${job.id}.json`)}>Export provenance</button>
        </div>
        <p class="muted">
          Cancellation is requested first; the backend confirms the final outcome. Retry creates a
          linked job.
        </p>
        {#if job.progress?.total}<progress
            max={job.progress.total}
            value={job.progress.completed ?? job.progress.step ?? 0}
          ></progress><span
            >{job.progress.completed ?? job.progress.step ?? 0} / {job.progress.total}</span
          >{:else if job.state === 'running'}<p>Running · completion percentage unavailable</p>{/if}
        <pre>{JSON.stringify(job, null, 2)}</pre>
      </div>
    </details>{/each}{#if !jobs.length}<div class="empty-state">
      <h2>No recorded jobs yet.</h2>
      <p>Review and submit a workflow to create the first immutable job record.</p>
    </div>{/if}
</section>
