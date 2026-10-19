<script lang="ts">
  import type { StudioClient } from '../api';
  import LibraryTransfer from '../components/LibraryTransfer.svelte';
  import { recordProjects } from '../recordPersistence';
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
      .catch(fail);
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
    loading = $state(false);
  let sequence = 0;
  const revisionKey = $derived(
    (allRecords[recordKind] ?? []).map((row) => `${row.id}:${row.revision}`).join(','),
  );
  async function load(append = false) {
    const ticket = ++sequence;
    loading = true;
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
      records = append ? [...records, ...result.records].slice(-2000) : result.records;
      nextOffset = result.next_offset;
    } catch (cause) {
      fail(cause);
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
      let data: RecordData = { ...(selectedRecord?.data ?? {}), name: recordName.trim() };
      data.project_ids = recordKind === 'projects' ? [] : membership;
      if (!data.name) throw new Error('Enter a record name.');
      if (recordKind === 'presets') data.draft = $state.snapshot(draft);
      else if (recordKind === 'sessions')
        data = {
          ...data,
          draft: $state.snapshot(draft),
          selected_artifact_id: selected?.id,
          active_job_ids: activeJobs.map((j) => j.id),
          layout,
        };
      else if (recordKind === 'workflows') data.workflow = draft.workflow ?? undefined;
      else if (recordKind === 'styles')
        data = { ...data, prompt: recordText, negative_prompt: negativeText, mode: styleMode };
      else if (recordKind === 'experiments') data = { ...data, notes: recordText, grid_id: gridId };
      else data.notes = recordText;
      await saveRecord(recordKind, data, selectedRecord?.id, selectedRecord?.revision ?? 0);
      await load();
      selectedRecord = null;
      recordName = '';
      recordText = '';
      negativeText = '';
      styleMode = 'append';
      gridId = '';
    } catch (e) {
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
        class:active={recordKind === kind}
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
      {#each records as record}<div class="surface library-record">
          <div>
            <h3>{record.data?.name ?? record.id}</h3>
            <small>Revision {record.revision} · {record.id}</small>
          </div>
          <div class="workflow-actions">
            <button onclick={() => void applyRecord(record, recordKind)}
              >{recordKind === 'experiments'
                ? 'Open in Grids'
                : recordKind === 'styles'
                  ? 'Append style'
                  : recordKind === 'projects'
                    ? 'Select project'
                    : 'Load into draft'}</button
            ><button onclick={() => exportJSON(record, `${record.id}.json`)}>Export</button><button
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
        </div>{/each}{#if !records.length}<p class="muted">No saved {recordKind}.</p>{/if}
      {#if nextOffset !== null}<button disabled={loading} onclick={() => void load(true)}
          >Load more records</button
        >{/if}
    </div>
    <div class="surface">
      <h2>{selectedRecord ? 'Update record' : 'Save a record'}</h2>
      <label for="record-name">Name</label><input
        id="record-name"
        bind:value={recordName}
      />{#if ['projects', 'styles', 'experiments'].includes(recordKind)}<label for="record-text"
          >{recordKind === 'styles' ? 'Prompt text to append' : 'Notes'}</label
        ><textarea id="record-text" rows="6" bind:value={recordText}></textarea>{/if}
      {#if recordKind === 'styles'}
        <label>Negative prompt<textarea rows="3" bind:value={negativeText}></textarea></label>
        <label
          >Style mode<select bind:value={styleMode}
            ><option value="append">Append</option><option value="template">Template</option
            ></select
          ></label
        >
        <p class="muted">
          Use {'{prompt}'} to insert the existing text. Without it, text is appended in stack order.
        </p>
      {/if}
      {#if recordKind === 'experiments'}<label
          >Linked durable grid ID<input bind:value={gridId} /></label
        >
        <p class="muted">
          A local grid freezes its definition and observed outcomes on save. Imported definitions
          remain available when their original run is absent.
        </p>{/if}
      {#if recordKind !== 'projects'}<label
          >Project membership<select
            multiple
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
        {recordKind === 'presets'
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
