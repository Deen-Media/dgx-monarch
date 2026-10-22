<script lang="ts">
  import { onMount } from 'svelte';
  import { fullJob } from './jobProvenance';
  import { initialWorkspace } from './initialWorkspace';
  import { Submission } from './submission.svelte';
  import { CATALOG_NOTICE, Connection } from './connection.svelte';
  import { emptyDraft, restoreDraft, driverStatus } from './model';
  import { DraftAutosave } from './draftAutosave.svelte';
  import { InspectionGuard, droppedInspection, DROPPED_REVIEW } from './inspectionGuard';
  import { requestLimit } from './api';
  import type { Binding, Draft } from './model';
  import type {
    Artifact,
    Job,
    Workflow,
    StoredRecord,
    RecordKind,
    RecordData,
    GraphConnection,
  } from './contracts';
  import { isActiveJob, requestJobAction } from './jobActions';
  import { persistRecord } from './recordPersistence';
  import { artifactDraft, liveArtifact } from './artifactDraft';
  import { inspectDraft, rebuildBindings } from './workflowInspector';
  import StatusMessages from './components/StatusMessages.svelte';
  import AppShell from './components/AppShell.svelte';
  import CommandPalette from './components/CommandPalette.svelte';
  import SubmissionReview from './components/SubmissionReview.svelte';
  import Viewer from './components/Viewer.svelte';
  import Operations from './components/Operations.svelte';
  import Generation from './views/Generation.svelte';
  import WorkspaceOutput from './views/WorkspaceOutput.svelte';
  import Gallery from './views/Gallery.svelte';
  import Queue from './views/Queue.svelte';
  import Grids from './views/Grids.svelte';
  import Fleet from './views/Fleet.svelte';
  import WorkspaceRecords from './views/WorkspaceRecords.svelte';
  import Settings from './views/Settings.svelte';
  import Connect from './views/Connect.svelte';
  const connection = new Connection();
  const submission = new Submission(connection);
  const autosave = new DraftAutosave(),
    guard = new InspectionGuard();
  const LAYOUT_KEY = 'dgxm.studio.layout.v1';
  let draft = $state<Draft>(emptyDraft()),
    undoDraft: Draft | null = null;
  let bindings = $state<Binding[]>([]),
    bindingErrors = $state<string[]>([]),
    rawWorkflow = $state(''),
    controlTab = $state('inputs');
  let connections = $state<GraphConnection[]>([]);
  let view = $state('workspace'),
    selected = $state<Artifact | null>(null),
    comparison = $state<Artifact | null>(null),
    selectedOperation = $state('');
  let busy = $state(false),
    notice = $state(''),
    error = $state(''),
    dirty = $state(false),
    localError = $state('');
  let showTelemetry = $state(true),
    showConsole = $state(false),
    presentation = $state(false),
    width = $state(340);
  let pendingActions = 0,
    workflowId = $state('');
  let palettePanel: CommandPalette, previewPanel: SubmissionReview;
  let jobs = $derived(connection.snapshot.jobs),
    activeJobs = $derived(jobs.filter(isActiveJob));
  const displayedArtifact = $derived(liveArtifact(connection.snapshot.artifacts, selected)),
    displayedComparison = $derived(liveArtifact(connection.snapshot.artifacts, comparison));
  const refresh = () => connection.refresh();
  function navigate(id: string) {
    view = id;
    palettePanel?.close();
    requestAnimationFrame(() => document.getElementById('main-heading')?.focus());
  }
  function openPalette() {
    palettePanel.open();
  }
  function onKey(event: KeyboardEvent) {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      openPalette();
    }
    if (event.key === 'Escape') presentation = false;
  }
  async function connect() {
    busy = true;
    error = '';
    try {
      await connection.connect();
      const degraded = connection.degradedNotice();
      if (degraded) setNotice(degraded);
      if (draft.workflow) await inspect(draft.workflow, draft.workflowName, true);
    } catch (e) {
      fail(e);
    } finally {
      busy = false;
    }
  }
  async function importDraft(value: Draft) {
    const previous = structuredClone($state.snapshot(draft));
    changeDraft(value);
    draft = value;
    bindings = [];
    connections = [];
    bindingErrors = [];
    rawWorkflow = '';
    if (draft.workflow) await inspect(draft.workflow, draft.workflowName, true);
    undoDraft = previous;
  }

  function setNotice(message: string) {
    notice = message;
    error = '';
  }
  function fail(e: unknown) {
    error = e instanceof Error ? e.message : String(e);
  }

  function changeDraft(change: Partial<Draft>) {
    guard.edit(change);
    if (change.workflow !== undefined) workflowId = '';
    undoDraft = structuredClone($state.snapshot(draft));
    draft = { ...draft, ...change };
    dirty = true;
    submission.preview = null;
  }
  async function undo() {
    if (undoDraft) {
      await importDraft(undoDraft);
      if (!error) setNotice('Draft change undone.');
    }
  }
  async function inspect(
    workflow: Workflow,
    name: string,
    preserveValues = false,
    templateId = '',
  ) {
    const ticket = guard.begin(connection.client);
    const values = preserveValues ? structuredClone($state.snapshot(draft.values)) : undefined;
    await withBusy(async () => {
      const result = await inspectDraft(connection.client, workflow, name, values);
      const state = guard.inspected(ticket, connection.client);
      if (state !== 'current')
        return void (state !== 'newer' && fail(droppedInspection(name, state)));
      ({ bindings, connections, errors: bindingErrors } = result);
      changeDraft(result.patch);
      workflowId = templateId;
      rawWorkflow = result.rawWorkflow;
      controlTab = result.controlTab;
    });
  }

  /** A dynamic choice changes which inputs exist, so inspect again with the edited values. */
  async function rebind(values: Draft['values']) {
    changeDraft({ values });
    await withBusy(async () => {
      const rebuilt = await rebuildBindings(connection, guard, () => draft, values);
      if (rebuilt) ({ bindings, connections, errors: bindingErrors, draft } = rebuilt);
    });
  }

  async function jobAction(job: Job, action: 'cancel' | 'retry') {
    await withBusy(async () => {
      await requestJobAction(connection.client, job, action, setNotice);
      await refresh();
    });
  }
  async function reuse(artifact: Artifact, operation?: string) {
    await withBusy(() => loadArtifact(artifact, operation));
  }
  async function loadArtifact(artifact: Artifact, operation?: string) {
    const api = connection.client,
      ticket = guard.begin(api);
    const job = await fullJob(
      api,
      artifact.job_id,
      jobs.find((j) => j.id === artifact.job_id),
    );
    if (guard.inspected(ticket, connection.client) !== 'current') return;
    if (!job?.submission?.workflow && !job?.submission?.prompt) {
      error = 'Source workflow is unavailable. Restore the job provenance before reuse.';
      return;
    }
    await importDraft(artifactDraft(artifact, job, operation));
    if (error) return;
    navigate('workspace');
    dirty = true;
    setNotice(
      operation
        ? `Output lineage attached. Set the corresponding workflow asset input before reviewing ${operation}.`
        : 'Recorded settings loaded into a new editable draft.',
    );
  }

  async function saveRecord(kind: RecordKind, data: RecordData, id?: string, revision = 0) {
    return persistRecord(connection.client, refresh, setNotice, kind, data, id, revision);
  }

  async function applyRecord(record: StoredRecord, kind: RecordKind) {
    try {
      if (kind === 'experiments') {
        sessionStorage.setItem('dgxm.studio.experiment.selection', record.id);
        navigate('grid');
      }
      if (kind === 'projects') {
        changeDraft({ project_id: record.id });
        setNotice(`Project: ${record.data.name}`);
      }
      if (kind === 'styles') {
        changeDraft({ styles: [...draft.styles, { ...record.data, id: record.id }] });
        setNotice('Style appended. Review expanded prompts before submission.');
      }
      if (kind === 'presets' || kind === 'sessions') {
        await importDraft(restoreDraft(record.data.draft));
        if (error) return;
        if (kind === 'sessions') {
          await refresh();
          selected =
            (connection.snapshot.artifacts ?? []).find(
              (a) => a.id === record.data.selected_artifact_id,
            ) ?? null;
          const layout = record.data.layout ?? {};
          width = Math.min(520, Math.max(280, layout.width ?? 340));
          showTelemetry = layout.showTelemetry ?? true;
          showConsole = layout.showConsole ?? false;
        }
        navigate('workspace');
        setNotice('Draft replaced from saved record. Undo is available.');
      }
      if (kind === 'workflows') await inspect(record.data.workflow!, record.data.name ?? record.id);
    } catch (e) {
      fail(e);
    }
  }

  onMount(() => {
    const restored = initialWorkspace();
    localError = restored.error;
    submission.recoveryError = restored.pendingError;
    if (restored.draft) draft = restored.draft;
    autosave.restore(restored.saved, restored.draft);
    if (restored.layout) ({ width, showTelemetry, showConsole } = restored.layout);
    submission.pending = restored.pending;
    const storage = (e: StorageEvent) => autosave.observe(e.key, e.newValue);
    window.addEventListener('storage', storage);
    const stopConnection = connection.observeVisibility();
    if (connection.token) void connect();
    return () => {
      stopConnection();
      window.removeEventListener('storage', storage);
    };
  });
  $effect(() => autosave.save($state.snapshot(draft), dirty));
  $effect(() => {
    try {
      localStorage.setItem(LAYOUT_KEY, JSON.stringify({ width, showTelemetry, showConsole }));
    } catch {
      /* Settings remain usable in this tab. */
    }
  });

  $effect(() => {
    if (submission.observe(jobs)) setNotice('Submission reconciled.');
  });
  $effect(() => {
    // The catalog was read again after the connect notice reported it unread.
    if (!connection.catalogError && notice.startsWith(CATALOG_NOTICE)) notice = '';
  });
  async function withBusy(action: () => Promise<void>) {
    pendingActions++;
    busy = true;
    error = '';
    try {
      await action();
    } catch (e) {
      fail(e);
    } finally {
      busy = --pendingActions > 0;
    }
  }
  async function prepareJob() {
    const ticket = guard.snapshot(connection.client);
    await withBusy(async () => {
      const current = () => guard.previewed(ticket, connection.client);
      if (await submission.prepare($state.snapshot(draft), current)) previewPanel.open();
      else fail(DROPPED_REVIEW);
    });
  }
  async function submitPreview() {
    await withBusy(async () => {
      const job = await submission.submit();
      previewPanel.close();
      dirty = false;
      setNotice(`Submitted ${job.id}.`);
    });
  }
  async function reconcilePending() {
    await withBusy(async () => {
      const found = await submission.reconcile();
      setNotice(
        found
          ? 'Submission reconciled.'
          : 'The companion confirms no accepted request for this identity. You can retry the retained request.',
      );
    });
  }
  async function retryPending() {
    await withBusy(() => submission.retryRetained());
  }
</script>

<svelte:window onkeydown={onKey} />
<AppShell
  {connection}
  {draft}
  activeCount={activeJobs.length}
  {view}
  bind:presentation
  {width}
  bind:showTelemetry
  bind:showConsole
  {navigate}
  {changeDraft}
  {openPalette}
>
  <StatusMessages
    messages={{ notice, error, localError }}
    {connection}
    {submission}
    {autosave}
    {draft}
    {busy}
    {connect}
    {reconcilePending}
    {retryPending}
    ondismiss={(kind) => {
      if (kind === 'notice') notice = '';
      else error = '';
    }}
    onload={() => {
      try {
        void importDraft(autosave.loadNewer());
      } catch (e) {
        fail(e);
      }
    }}
  />
  <main id="main-heading" tabindex="-1">
    {#if !connection.connected}<Connect
        bind:token={connection.token}
        {busy}
        error={connection.error}
        {connect}
      />
    {:else if presentation}<Viewer
        activeCount={activeJobs.length}
        artifact={displayedArtifact}
        compare={displayedComparison}
        client={connection.client}
        onreuse={reuse}
        onselect={(a) => (comparison = a)}
      />
    {:else if view === 'workspace'}<div class="workspace">
        <Generation
          client={connection.client}
          {draft}
          bind:bindings
          bind:rawWorkflow
          bind:controlTab
          {bindingErrors}
          catalog={connection.catalog}
          {connections}
          workflows={connection.workflows}
          {workflowId}
          requestLimit={requestLimit(connection.capabilities)}
          {dirty}
          {busy}
          pending={!!submission.pending || !!submission.recoveryError}
          driverState={driverStatus(connection.snapshot.driver)}
          {changeDraft}
          {inspect}
          {fail}
          {undo}
          {prepareJob}
          {rebind}
          catalogError={connection.catalogError}
          refreshCatalog={() => connection.refreshCatalog()}
        /><WorkspaceOutput
          client={connection.client}
          selected={displayedArtifact}
          comparison={displayedComparison}
          artifacts={connection.snapshot.artifacts}
          telemetry={connection.snapshot.telemetry}
          history={connection.metricHistory}
          presentation={connection.capabilities.presentation}
          driverState={connection.snapshot.driver.state}
          connectionState={connection.state}
          activeCount={activeJobs.length}
          {showTelemetry}
          onselect={(a) => (selected = a)}
          oncompare={(a) => (comparison = a)}
          oncollapse={() => (showTelemetry = false)}
          {navigate}
          {reuse}
        />
      </div>
    {:else if view === 'gallery'}<Gallery
        client={connection.client}
        projectSummaries={connection.snapshot.records.projects ?? []}
        {jobs}
        projectId={draft.project_id}
        {selected}
        {navigate}
        onselect={(a) => (selected = a)}
        oncompare={(a) => (comparison = a)}
        {reuse}
        {setNotice}
      />
    {:else if view === 'queue'}<Queue
        truncated={connection.snapshot.jobs_truncated}
        client={connection.client}
        {jobs}
        {busy}
        {refresh}
        {fail}
        {jobAction}
      />
    {:else if view === 'grid'}<Grids
        {reuse}
        client={connection.client}
        {draft}
        {bindings}
        restoreDraft={importDraft}
      />
    {:else if view === 'fleet'}<Fleet
        client={connection.client}
        {jobs}
        history={connection.metricHistory}
        presentation={connection.capabilities.presentation}
        telemetry={connection.snapshot.telemetry}
        {changeDraft}
        {navigate}
      />
    {:else if view === 'operations'}<Operations
        client={connection.client}
        capabilities={connection.capabilities}
        initialAction={selectedOperation}
        operations={connection.snapshot.operations}
        onrefresh={() => void refresh().catch(fail)}
      />
    {:else if view === 'projects'}<WorkspaceRecords
        client={connection.client}
        onrefresh={refresh}
        allRecords={connection.snapshot.records}
        {draft}
        {selected}
        {activeJobs}
        layout={{ width, showTelemetry, showConsole, view }}
        {saveRecord}
        {applyRecord}
        {fail}
      />
    {:else if view === 'settings'}<Settings
        {draft}
        bind:width
        bind:showTelemetry
        bind:showConsole
        capabilities={connection.capabilities}
        driverState={driverStatus(connection.snapshot.driver)}
        onimport={importDraft}
        disconnect={() => connection.disconnect()}
        {fail}
      />{/if}
  </main>
</AppShell>
<CommandPalette
  bind:this={palettePanel}
  operations={connection.capabilities.operations ?? []}
  {navigate}
  onoperation={(id) => {
    selectedOperation = id;
    navigate('operations');
  }}
/>
<SubmissionReview
  bind:this={previewPanel}
  preview={submission.preview}
  previewKind={submission.kind}
  {busy}
  pending={!!submission.pending}
  bind:confirmed={submission.confirmed}
  {submitPreview}
/>
