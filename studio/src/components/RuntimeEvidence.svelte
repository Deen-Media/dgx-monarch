<script lang="ts">
  import type { StudioClient } from '../api';
  import type { Job } from '../contracts';
  let { client, job }: { client: StudioClient; job: Job } = $props();
  let items = $state<Record<string, any>[]>([]),
    cursor = $state<string | null>(null),
    kind = $state('renders'),
    error = $state(''),
    busy = $state(false),
    loaded = $state(false);
  let currentId = '';
  let sequence = 0;
  $effect(() => {
    if (job.id !== currentId) {
      currentId = job.id;
      sequence++;
      busy = false;
      items = [];
      cursor = null;
      loaded = false;
    }
  });
  async function load(more = false) {
    const ticket = ++sequence,
      jobId = job.id,
      selectedKind = kind;
    busy = true;
    error = '';
    try {
      const page = await client.request<{
        items: Record<string, any>[];
        next_cursor: string | null;
      }>(
        `/jobs/${encodeURIComponent(job.id)}/runtime?kind=${kind}&limit=40${more && cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''}`,
      );
      if (ticket !== sequence || job.id !== jobId || kind !== selectedKind) return;
      items = page.items;
      cursor = page.next_cursor;
      loaded = true;
    } catch (cause) {
      if (ticket === sequence) error = String(cause);
    } finally {
      if (ticket === sequence) busy = false;
    }
  }
</script>

<details>
  <summary>Monarch admission, Worker acknowledgements and Fleet results</summary>
  <p>
    Evidence joins this exact Comfy prompt and render identity. Output validation, native owner ACK
    and graph completion are separate facts.
  </p>
  {#if job.monarch}
    <p>
      {job.monarch.render_count ?? Object.keys(job.monarch.renders ?? {}).length} renders · {job
        .monarch.event_count} observed events
    </p>
    <pre>{JSON.stringify(job.monarch.render_status_counts ?? {}, null, 2)}</pre>
  {:else}<p>
      No prompt-bound Monarch evidence has been observed. This is unknown, not a failed or
      successful render.
    </p>{/if}
  <label
    >Evidence records<select
      bind:value={kind}
      onchange={() => {
        sequence++;
        busy = false;
        loaded = false;
        cursor = null;
        items = [];
      }}
      ><option value="renders">Per-render outcomes</option><option value="events"
        >Exact runtime events</option
      ></select
    ></label
  >
  <button disabled={busy} onclick={() => void load()}>Refresh evidence</button>
  {#each items as item}
    <article class="surface">
      <h4>
        {item.fleet_index == null ? 'Render' : `Fleet item ${item.fleet_index}`} · {item.status ??
          item.event?.kind ??
          'unknown'}
      </h4>
      <p class="mono">{item.render_id ?? item.event?.render_id}</p>
      {#if item.effective_topology}<p>
          Effective topology: {JSON.stringify(item.effective_topology)}
        </p>{/if}
      {#if item.worker_result_authority}<p>Worker result: {item.worker_result_authority}</p>{/if}
      <details>
        <summary>Complete recorded facts</summary>
        <pre>{JSON.stringify(item, null, 2)}</pre>
      </details>
    </article>
  {/each}
  {#if loaded && !items.length}<p>No records on this page.</p>{/if}
  {#if cursor}<button disabled={busy} onclick={() => void load(true)}>Next evidence page</button
    >{/if}
  {#if error}<p role="alert">{error}</p>{/if}
</details>
