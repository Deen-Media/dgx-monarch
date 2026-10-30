<script lang="ts">
  import { onMount } from 'svelte';
  import { exactJob } from './jobProvenance';
  import { initialWorkspace, saveLayout } from './initialWorkspace';
  import { Submission } from './submission.svelte';
  import { JobTracker } from './jobTracker.svelte';
  import { CATALOG_NOTICE, Connection, errorText } from './connection.svelte';
  import { emptyDraft, restoreDraft, driverStatus, seedChange } from './model';
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
  import { OperationOpener } from './operationOpener.svelte';
  import { persistRecord } from './recordPersistence';
  import { artifactDraft, reuseNotice } from './artifactDraft';
  import { inspectDraft, rebuildBindings } from './workflowInspector';
  import StatusMessages from './components/StatusMessages.svelte';
  import AppShell from './components/AppShell.svelte';
  import CommandPalette from './components/CommandPalette.svelte';
  import SubmissionReview from './components/SubmissionReview.svelte';
  import Viewer from './components/Viewer.svelte';
  import EditHandoff from './components/EditHandoff.svelte';
  import Generation from './views/Generation.svelte';
  import WorkspaceOutput from './views/WorkspaceOutput.svelte';
  import Gallery from './views/Gallery.svelte';
  import Queue from './views/Queue.svelte';
  import Utilities from './views/Utilities.svelte';
  import { isUtility, railView, resolveView } from './navigation';
  import WorkspaceRecords from './views/WorkspaceRecords.svelte';
  import Settings from './views/Settings.svelte';
  import Connect from './views/Connect.svelte';
  const connection = new Connection();
  const submission = new Submission(connection);
  const tracker = new JobTracker(connection, submission);
  const autosave = new DraftAutosave(),
    guard = new InspectionGuard(),
    opener = new OperationOpener(navigate);
  let draft = $state<Draft>(emptyDraft()),
    undoDraft: Draft | null = null;
  let bindings = $state<Binding[]>([]),
    bindingErrors = $state<string[]>([]),
    rawWorkflow = $state(''),
    controlTab = $state('inputs');
  let connections = $state<GraphConnection[]>([]);
  let view = $state('workspace');
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
  let palettePanel: CommandPalette, previewPanel: SubmissionReview, handoff: EditHandoff;
  let jobs = $derived(connection.snapshot.jobs),
    activeJobs = $derived(jobs.filter(isActiveJob));
  const refresh = () => connection.refresh();
  function navigate(id: string) {
    presentation = false; // a view that opens must show, not stay behind the presentation
    view = resolveView(id);
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
  async function importDraft(value: Draft, templateId = '') {
    const previous = structuredClone($state.snapshot(draft));
    changeDraft(value);
    draft = value;
    bindings = [];
    connections = [];
    bindingErrors = [];
    rawWorkflow = '';
    if (draft.workflow) await inspect(draft.workflow, draft.workflowName, true, templateId);
    undoDraft = previous;
  }
  function setNotice(message: string) {
    notice = message;
    error = '';
  }
  function fail(e: unknown) {
    error = errorText(e, connection.fromLink);
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
  /** Put an output's recorded seeds into the draft with the Fixed policy; Undo reverts both. */
  function useSeed(seeds: Record<string, number | string>) {
    const result = seedChange(draft, bindings, seeds);
    if ('error' in result) return void (error = result.error);
    changeDraft(result.change);
    setNotice(result.notice);
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
      const fit = { operation: () => draft.operation, info: connection.catalog.object_info };
      const result = await inspectDraft(connection.client, workflow, name, values, fit);
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
      const record = await requestJobAction(connection.client, job, action, setNotice);
      if (action === 'retry' && record) tracker.own(record.id);
      await refresh();
    });
  }
  async function reuse(artifact: Artifact, operation?: string) {
    if (operation === 'edit')
      return handoff.send(artifact, { withBusy, importDraft, navigate, setNotice });
    await withBusy(() => loadArtifact(artifact, operation));
  }
  async function loadArtifact(artifact: Artifact, operation?: string) {
    const api = connection.client,
      ticket = guard.begin(api);
    // The exact record, so a seed above 2**53 - 1 is refused by name rather than loaded rounded.
    const job = await exactJob(api, artifact.job_id);
    if (guard.inspected(ticket, connection.client) !== 'current') return;
    if (!job?.submission?.workflow && !job?.submission?.prompt) {
      error = 'Source workflow is unavailable. Restore the job provenance before reuse.';
      return;
    }
    await importDraft(artifactDraft(artifact, job, operation));
    if (error) return;
    navigate('workspace');
    dirty = true;
    setNotice(reuseNotice(operation));
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
        setNotice('Style appended. Review settings shows the expanded prompts.');
      }
      if (kind === 'presets' || kind === 'sessions') {
        await importDraft(restoreDraft(record.data.draft));
        if (error) return;
        if (kind === 'sessions') {
          await refresh();
          tracker.selected =
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
  $effect(() => saveLayout({ width, showTelemetry, showConsole }));

  $effect(() => {
    if (submission.observe(jobs)) setNotice('Submission reconciled.');
  });
  $effect(() => tracker.observe());
  $effect(() => tracker.run(connection.hidden));
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
  /** Generate starts a valid job at once; Review settings, a warning or a refusal open the review. */
  async function prepareJob(review = false) {
    const ticket = guard.snapshot(connection.client);
    await withBusy(async () => {
      const current = () => guard.previewed(ticket, connection.client);
      const next = await submission.prepare($state.snapshot(draft), current, review, bindings);
      if (next === 'ready') await submitPreview();
      else if (next === 'review') previewPanel.open();
      else fail(DROPPED_REVIEW);
    });
  }
  async function submitPreview() {
    await withBusy(async () => {
      tracker.own((await submission.submit()).id);
      dirty = false;
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
  working={tracker.working}
  view={railView(view)}
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
        job={tracker.status}
        working={tracker.working}
        artifact={tracker.shown}
        compare={tracker.shownComparison}
        client={connection.client}
        onreuse={reuse}
        onselect={(a) => (tracker.comparison = a)}
        onqueue={() => navigate('queue')}
        onconsent={() => opener.consent(tracker.job)}
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
          confirmation={submission.confirmation}
          {changeDraft}
          {inspect}
          {fail}
          {undo}
          {prepareJob}
          {rebind}
          issueRefresh={{ guard, apply: (next) => ({ errors: bindingErrors, connections } = next) }}
          catalogError={connection.catalogError}
          refreshCatalog={() => connection.refreshCatalog()}
        /><WorkspaceOutput
          client={connection.client}
          selected={tracker.shown}
          comparison={tracker.shownComparison}
          artifacts={connection.snapshot.artifacts}
          telemetry={connection.snapshot.telemetry}
          history={connection.metricHistory}
          presentation={connection.capabilities.presentation}
          driverState={connection.snapshot.driver.state}
          connectionState={connection.state}
          serverTime={connection.snapshot.server_time}
          activeCount={activeJobs.length}
          job={tracker.status}
          working={tracker.working}
          {showTelemetry}
          {jobs}
          onuseseed={useSeed}
          onselect={(a) => (tracker.selected = a)}
          oncompare={(a) => (tracker.comparison = a)}
          {navigate}
          {reuse}
          onconsent={() => opener.consent(tracker.job)}
        />
      </div>
    {:else if view === 'gallery'}<Gallery
        client={connection.client}
        projectSummaries={connection.snapshot.records.projects ?? []}
        {jobs}
        projectId={draft.project_id}
        selected={tracker.selected}
        {navigate}
        onselect={(a) => (tracker.selected = a)}
        oncompare={(a) => (tracker.comparison = a)}
        {reuse}
        {setNotice}
      />
    {:else if view === 'queue'}<Queue
        {connection}
        {busy}
        {refresh}
        {fail}
        {jobAction}
        onconsent={opener.consent}
      />
    {:else if isUtility(view)}<Utilities
        {view}
        {connection}
        {draft}
        {bindings}
        {reuse}
        restoreDraft={importDraft}
        {changeDraft}
        {inspect}
        {navigate}
        {setNotice}
        request={opener.request}
        onrefresh={() => void refresh().catch(fail)}
      />
    {:else if view === 'projects'}<WorkspaceRecords
        client={connection.client}
        onrefresh={refresh}
        allRecords={connection.snapshot.records}
        {draft}
        selected={tracker.selected}
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
  workflows={connection.workflows}
  {navigate}
  {inspect}
  onoperation={opener.operation}
/>
<SubmissionReview bind:this={previewPanel} {submission} {busy} {submitPreview} bind:controlTab />
<EditHandoff bind:this={handoff} {connection} {draft} {bindings} {workflowId} />
