<script lang="ts">
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
  }: {
    allRecords: Records;
    draft: Draft;
    selected: Artifact | null;
    activeJobs: Job[];
    layout: Layout;
    saveRecord: SaveRecord;
    applyRecord: (record: StoredRecord, kind: RecordKind) => Promise<void>;
    fail: ReportError;
  } = $props();
  let busy = $state(false),
    recordKind = $state<RecordKind>('projects'),
    recordName = $state(''),
    recordText = $state(''),
    selectedRecord = $state<StoredRecord | null>(null);
  const kinds: RecordKind[] = [
    'projects',
    'presets',
    'styles',
    'sessions',
    'workflows',
    'experiments',
  ];
  let records = $derived(allRecords[recordKind] ?? []);
  async function createRecord() {
    busy = true;
    try {
      let data: RecordData = { ...(selectedRecord?.data ?? {}), name: recordName.trim() };
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
        data = { ...data, prompt: recordText, negative_prompt: '', mode: 'append' };
      else data.notes = recordText;
      await saveRecord(recordKind, data, selectedRecord?.id, selectedRecord?.revision ?? 0);
      selectedRecord = null;
      recordName = '';
      recordText = '';
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
    <button
      onclick={() =>
        exportJSON({ schema_version: '1.0', records: allRecords }, 'monarch-library.json')}
      >Export library</button
    >
  </div>
  <div class="tabs">
    {#each kinds as kind}<button
        class:active={recordKind === kind}
        onclick={() => {
          recordKind = kind;
          selectedRecord = null;
          recordName = '';
          recordText = '';
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
            {#if recordKind !== 'experiments'}<button
                onclick={() => void applyRecord(record, recordKind)}
                >{recordKind === 'styles'
                  ? 'Append style'
                  : recordKind === 'projects'
                    ? 'Select project'
                    : 'Load into draft'}</button
              >{/if}<button onclick={() => exportJSON(record, `${record.id}.json`)}>Export</button
            ><button
              onclick={() => {
                selectedRecord = record;
                recordName = record.data?.name ?? '';
                recordText = record.data?.notes ?? record.data?.prompt ?? '';
              }}>Edit</button
            >
          </div>
          <details>
            <summary>Record content</summary>
            <pre>{JSON.stringify(record.data, null, 2)}</pre>
          </details>
        </div>{/each}{#if !records.length}<p class="muted">No saved {recordKind}.</p>{/if}
    </div>
    <div class="surface">
      <h2>{selectedRecord ? 'Update record' : 'Save a record'}</h2>
      <label for="record-name">Name</label><input
        id="record-name"
        bind:value={recordName}
      />{#if ['projects', 'styles', 'experiments'].includes(recordKind)}<label for="record-text"
          >{recordKind === 'styles' ? 'Prompt text to append' : 'Notes'}</label
        ><textarea id="record-text" rows="6" bind:value={recordText}></textarea>{/if}
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
          }}>Cancel edit</button
        >{/if}
    </div>
  </div>
</section>
