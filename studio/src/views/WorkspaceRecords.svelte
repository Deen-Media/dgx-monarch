<script lang="ts">
  import type { StudioClient } from '../api';
  import LibraryTransfer from '../components/LibraryTransfer.svelte';
  import { retainRecordPage } from '../recordRetention';
  import { RecordSession, recordProjects } from '../recordPersistence';
  import { recordContent } from '../recordContent';
  import { loadRecordCatalog } from '../recordCatalog';
  import type { Draft } from '../model';
  import type {
    Artifact,
    Job,
    Layout,
    Records,
    RecordKind,
    StoredRecord,
    RecordData,
    SaveRecord,
    ReportError,
  } from '../contracts';
  import { exportJSON } from '../files';
  let {
    allRecords,
    draft,
    selected,
    activeJobs,
    layout,
    saveRecord,
    applyRecord,
    fail,
    client,
    onrefresh,
  }: {
    allRecords: Records;
    draft: Draft;
    selected: Artifact | null;
    activeJobs: Job[];
    layout: Layout;
    saveRecord: SaveRecord;
    applyRecord: (record: StoredRecord, kind: RecordKind) => Promise<void>;
    fail: ReportError;
    client: StudioClient;
    onrefresh: () => Promise<void>;
  } = $props();
  let busy = $state(false),
    recordKind = $state<RecordKind>('projects'),
    recordName = $state(''),
    recordText = $state(''),
    negativeText = $state(''),
    styleMode = $state('append'),
    gridId = $state(''),
    selectedRecord = $state<StoredRecord | null>(null);
  let projectOptions = $state<StoredRecord[]>([]);
  const projectRevision = $derived(
    (allRecords.projects ?? []).map((row) => `${row.id}:${row.revision}`).join(','),
  );
  $effect(() => {
    projectRevision;
    void loadRecordCatalog(client, 'projects')
      .then((rows) => (projectOptions = rows))
      .catch((cause) => {
        projectOptions = [];
        fail(cause);
      });
  });
  const kinds: RecordKind[] = [
    'projects',
    'presets',
    'styles',
    'sessions',
    'workflows',
    'experiments',
  ];
  let records = $state<StoredRecord[]>([]),
    projectFilter = $state(''),
    membership = $state<string[]>([]);
  let nextOffset = $state<number | null>(null),
    loading = $state(false),
    loadError = $state('');
  let sequence = 0,
    loadedFor = '';
  // A new record keeps one identity per form session until a save is confirmed.
  const session = new RecordSession();
  let windowNotice = $state('');
  const revisionKey = $derived(
    (allRecords[recordKind] ?? []).map((row) => `${row.id}:${row.revision}`).join(','),
  );
  async function load(append = false) {
    const ticket = ++sequence,
      listing = `${recordKind}|${projectFilter}`;
    loading = true;
    loadError = '';
    // Keep the current rows until the new page arrives; clear only for another kind or filter.
    if (!append && loadedFor !== listing) records = [];
    try {
      const params = new URLSearchParams({
        limit: '100',
        offset: String(append ? (nextOffset ?? 0) : 0),
      });
      if (projectFilter && recordKind !== 'projects') params.set('project_id', projectFilter);
      const result = await client.request<{ records: StoredRecord[]; next_offset: number | null }>(
        `/records/${recordKind}?${params}`,
      );
      if (ticket !== sequence) return;
      const retained = retainRecordPage(append ? records : [], result.records);
      records = retained.records;
      loadedFor = listing;
      windowNotice = retained.replaced
        ? 'Started the next record window. Earlier rows were released to stay within 8 MiB and 2,000 records.'
        : append
          ? windowNotice
          : '';
      nextOffset = result.next_offset;
    } catch (cause) {
      if (ticket === sequence) {
        loadError = cause instanceof Error ? cause.message : String(cause);
        fail(cause);
      }
    } finally {
      if (ticket === sequence) loading = false;
    }
  }
  $effect(() => {
    revisionKey;
    recordKind;
    projectFilter;
    void load();
  });
  $effect(() => {
    projectFilter = draft.project_id;
  });
  $effect(() => {
    if (!selectedRecord) membership = draft.project_id ? [draft.project_id] : [];
  });
  async function createRecord() {
    busy = true;
    try {
      let data: RecordData = {
        ...recordContent(recordKind, selectedRecord?.data ?? null, {
          draft: $state.snapshot(draft),
          selectedArtifactId: selected?.id,
          activeJobIds: activeJobs.map((job) => job.id),
          layout: $state.snapshot(layout),
        }),
        name: recordName.trim(),
      };
      data.project_ids = recordKind === 'projects' ? [] : membership;
      if (!data.name) throw new Error('Enter a record name.');
      if (recordKind === 'styles')
        data = { ...data, prompt: recordText, negative_prompt: negativeText, mode: styleMode };
      else if (recordKind === 'experiments') data = { ...data, notes: recordText, grid_id: gridId };
      else if (recordKind === 'projects') data.notes = recordText;
      const editing = selectedRecord;
      await saveRecord(
        recordKind,
        data,
        editing?.id ?? session.id(recordKind),
        editing?.revision ?? 0,
      );
      if (!editing) session.settle();
      selectedRecord = null;
      recordName = '';
      recordText = '';
      negativeText = '';
      styleMode = 'append';
      gridId = '';
      await load();
    } catch (e) {
      session.failed(e);
      fail(e);
    } finally {
      busy = false;
    }
  }
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">REUSABLE WORK</span>
      <h1>Project library</h1>
    </div>
  </div>
  <LibraryTransfer
    {client}
    projectId={draft.project_id}
    onrefresh={async () => {
      await onrefresh();
      await load();
    }}
  />
  <label
    >Project filter<select bind:value={projectFilter}
      ><option value="">All projects and shared records</option
      >{#each projectOptions as project}<option value={project.id}
          >{project.data.name ?? project.id}</option
        >{/each}</select
    ></label
  >
  <div class="tabs">
    {#each kinds as kind}<button
        disabled={busy}
        class:active={recordKind === kind}
        aria-pressed={recordKind === kind}
        onclick={() => {
          recordKind = kind;
          selectedRecord = null;
          recordName = '';
          recordText = '';
          negativeText = '';
          styleMode = 'append';
          gridId = '';
          membership = draft.project_id ? [draft.project_id] : [];
        }}>{kind}</button
      >{/each}
  </div>
  <div class="operation-grid">
    <div>
      <h2>Saved {recordKind}</h2>
      <p class="muted">
        Up to 8 MiB and 2,000 records per window. Load more continues from the last row.
      </p>
      {#if windowNotice}<p role="status">{windowNotice}</p>{/if}
      {#each records as record}<div class="surface library-record">
          <div>
            <h3>{record.data?.name ?? record.id}</h3>
            <small>Revision {record.revision} · {record.id}</small>
          </div>
          <div class="workflow-actions">
            <button disabled={busy || loading} onclick={() => void applyRecord(record, recordKind)}
              >{recordKind === 'experiments'
                ? 'Open in Grids'
                : recordKind === 'styles'
                  ? 'Append style'
                  : recordKind === 'projects'
                    ? 'Select project'
                    : 'Load into draft'}</button
            ><button
              disabled={busy || loading}
              onclick={() => exportJSON(record, `${record.id}.json`)}>Export</button
            ><button
              disabled={busy || loading}
              onclick={() => {
                selectedRecord = record;
                recordName = record.data?.name ?? '';
                recordText = record.data?.notes ?? record.data?.prompt ?? '';
                negativeText = record.data.negative_prompt ?? '';
                styleMode = String(record.data.mode ?? 'append');
                gridId = typeof record.data.grid_id === 'string' ? record.data.grid_id : '';
                membership = recordProjects(record.data);
              }}>Edit</button
            >
          </div>
          <details>
            <summary>Record content</summary>
            <pre>{JSON.stringify(record.data, null, 2)}</pre>
          </details>
        </div>{/each}{#if loading && !records.length}<p role="status">
          Loading saved {recordKind}…
        </p>{:else if !records.length && !loadError}<p class="muted">No saved {recordKind}.</p>{/if}
      {#if nextOffset !== null}<button disabled={loading} onclick={() => void load(true)}
          >Load more records</button
        >{/if}
    </div>
    <div class="surface">
      <h2>{selectedRecord ? 'Update record' : 'Save a record'}</h2>
      <label for="record-name">Name</label><input
        id="record-name"
        disabled={busy}
        bind:value={recordName}
      />{#if ['projects', 'styles', 'experiments'].includes(recordKind)}<label for="record-text"
          >{recordKind === 'styles' ? 'Prompt text to append' : 'Notes'}</label
        ><textarea id="record-text" rows="6" disabled={busy} bind:value={recordText}
        ></textarea>{/if}
      {#if recordKind === 'styles'}
        <label
          >Negative prompt<textarea rows="3" disabled={busy} bind:value={negativeText}
          ></textarea></label
        >
        <label
          >Style mode<select disabled={busy} bind:value={styleMode}
            ><option value="append">Append</option><option value="template">Template</option
            ></select
          ></label
        >
        <p class="muted">
          Use {'{prompt}'} to insert the existing text. Without it, text is appended in stack order.
        </p>
      {/if}
      {#if recordKind === 'experiments'}<label
          >Linked durable grid ID<input disabled={busy} bind:value={gridId} /></label
        >
        <p class="muted">
          A local grid freezes its definition and observed outcomes on save. Imported definitions
          remain available when their original run is absent.
        </p>{/if}
      {#if recordKind !== 'projects'}<label
          >Project membership<select
            multiple
            disabled={busy}
            bind:value={membership}
            aria-label="Record project membership"
            >{#each projectOptions as project}<option value={project.id}
                >{project.data.name ?? project.id}</option
              >{/each}</select
          ></label
        >
        <p class="muted">
          No selected project means a shared record. Membership does not move source assets.
        </p>{/if}
      <p class="muted">
        {selectedRecord && ['presets', 'sessions', 'workflows'].includes(recordKind)
          ? 'Editing the name and membership preserves this record’s saved content. Save a new record to capture the current workspace.'
          : recordKind === 'presets'
            ? 'Saves the complete current generation draft. Loading replaces its fields and supports undo.'
            : recordKind === 'sessions'
              ? 'Saves the draft, selected output, active job references and panel layout. Jobs are reconciled before control actions.'
              : recordKind === 'workflows'
                ? 'Saves the original workflow and Studio display configuration.'
                : 'Records are owned by this companion and use revision checks to prevent stale writes.'}
      </p>
      <button class="primary" disabled={busy || !recordName.trim()} onclick={createRecord}
        >Save {recordKind}</button
      >{#if selectedRecord}<button
          disabled={busy}
          onclick={() => {
            selectedRecord = null;
            recordName = '';
            recordText = '';
            negativeText = '';
            styleMode = 'append';
            gridId = '';
          }}>Cancel edit</button
        >{/if}
    </div>
  </div>
</section>
