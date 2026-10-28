<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Job } from '../contracts';
  import { untrack } from 'svelte';
  import { formatTime } from '../model';
  import { recordedSeeds, seedText } from '../recordedSettings';
  import { emptyQueueText, isActive, keptPage, settledJobs, type SnapshotIds } from '../queueRows';
  import JobProvenance from '../components/JobProvenance.svelte';
  let {
    client,
    jobs,
    truncated = false,
    busy,
    refresh,
    fail,
    jobAction,
  }: {
    client: StudioClient;
    jobs: Job[];
    truncated?: boolean;
    busy: boolean;
    refresh: () => Promise<void>;
    fail: (error: unknown) => void;
    jobAction: (job: Job, action: 'cancel' | 'retry') => Promise<void>;
  } = $props();
  let selectedJob = $state<Job | null>(null);
  let history = $state<Job[]>([]),
    cursor = $state<string | null>(null),
    loading = $state(true),
    query = $state(''),
    status = $state(''),
    page = $state(0);
  let revision = 0;
  const active = $derived(jobs.filter(isActive));
  const terminal = $derived(
    history
      .map((job) => jobs.find((current) => current.id === job.id) ?? job)
      .filter((job) => !active.some((current) => current.id === job.id)),
  );
  const shown = $derived([...active, ...terminal.slice(page * 40, (page + 1) * 40)]);
  async function load(reset = false, keepPosition = false) {
    const ticket = reset ? ++revision : revision;
    loading = true;
    const params = new URLSearchParams({ limit: '100', q: query, status });
    if (!reset && cursor) params.set('cursor', cursor);
    try {
      const result = await client.request<{ items: Job[]; next_cursor: string | null }>(
        `/jobs?${params}`,
      );
      if (ticket !== revision) return;
      const replace = reset || history.length >= 2000;
      history = replace ? result.items : [...history, ...result.items];
      cursor = result.next_cursor;
      if (replace) page = keepPosition ? keptPage(page, history.length) : 0;
    } catch (cause) {
      fail(cause);
    } finally {
      if (ticket === revision) loading = false;
    }
  }
  let observed: SnapshotIds | null = null;
  $effect(() => {
    // A job that settles after the history load must come back through history.
    const { next, settled } = settledJobs(observed, jobs);
    observed = next;
    if (settled.length) untrack(() => void load(true, true));
  });
  const empty = $derived(emptyQueueText(query, status, loading));
  $effect(() => {
    const search = query + status;
    const timer = setTimeout(() => {
      if (search !== undefined) void load(true);
    }, 200);
    return () => clearTimeout(timer);
  });
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">AUTHORITATIVE EXECUTION</span>
      <h1>Queue & history</h1>
    </div>
    <button onclick={() => void refresh().catch(fail)}>Refresh</button>
  </div>
  <div class="filter-bar">
    <input
      type="search"
      bind:value={query}
      placeholder="Search durable job provenance"
      aria-label="Search job history"
    /><select bind:value={status} aria-label="Filter job history state"
      ><option value="">All history states</option
      >{#each ['succeeded', 'failed', 'cancelled', 'unknown'] as state}<option>{state}</option
        >{/each}</select
    ><button disabled={loading} onclick={() => void load(true)}>Refresh history</button>
  </div>
  {#if truncated}<p class="warning" role="status">
      The live snapshot is limited; additional jobs may exist. Search or load more durable history
      to find jobs outside this snapshot.
    </p>{/if}
  <p class="muted">
    {active.length} active or unresolved jobs remain visible above {history.length} loaded history records
    (maximum 2,000 per window).
  </p>
  {#each shown as job (job.id)}{@const seeds = seedText(recordedSeeds(job))}
    <details
      class="record"
      open={selectedJob?.id === job.id}
      ontoggle={(e) => {
        if (e.currentTarget.open) selectedJob = job;
        else if (selectedJob?.id === job.id) selectedJob = null;
      }}
    >
      <summary
        ><span class="status-label" class:running={job.state === 'running'}
          >{job.state ?? 'unknown'}</span
        ><strong>{job.submission?.requested?.operation ?? 'workflow'}</strong><span class="mono"
          >{job.id}</span
        >{#if seeds}<span class="mono">{seeds}</span>{/if}<span>{formatTime(job.created_at)}</span
        ></summary
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
        {#if job.progress?.max}<progress max={job.progress.max} value={job.progress.value ?? 0}
          ></progress><span
            >{job.progress.value ?? 0} / {job.progress.max} · node {job.progress.node ??
              'unknown'}</span
          >{:else if job.state === 'running'}<p>Running · completion percentage unavailable</p>{/if}
        {#if selectedJob?.id === job.id}<JobProvenance {client} summary={job} />{/if}
      </div>
    </details>{/each}{#if !shown.length}<div class="empty-state">
      <h2>{empty.title}</h2>
      <p>{empty.detail}</p>
    </div>{/if}
  <div class="pagination">
    <button disabled={page === 0} onclick={() => page--}>Previous history page</button><span
      >History page {page + 1}</span
    ><button disabled={(page + 1) * 40 >= terminal.length} onclick={() => page++}
      >Next history page</button
    >
  </div>
  {#if cursor}<button disabled={loading} onclick={() => void load()}
      >{history.length >= 2000 ? 'Next history window' : 'Load more history'}</button
    >{/if}
</section>
