<script lang="ts">
  import type { Draft } from '../model';
  import { restoreDraft } from '../model';
  import { exportJSON } from '../files';
  import type { Capabilities, ReportError } from '../contracts';
  let {
    draft,
    width = $bindable(),
    showTelemetry = $bindable(),
    showConsole = $bindable(),
    capabilities,
    driverState,
    onimport,
    disconnect,
    fail,
  }: {
    draft: Draft;
    width: number;
    showTelemetry: boolean;
    showConsole: boolean;
    capabilities: Capabilities;
    driverState?: string;
    onimport: (draft: Draft) => Promise<void>;
    disconnect: () => void;
    fail: ReportError;
  } = $props();
</script>

<section class="page-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">WORKSPACE PREFERENCES</span>
      <h1>Settings</h1>
    </div>
  </div>
  <div class="operation-grid">
    <div class="surface">
      <h2>Layout</h2>
      <label for="panel-width">Control panel width · {width}px</label><input
        id="panel-width"
        type="range"
        min="280"
        max="520"
        step="10"
        bind:value={width}
      /><label class="check-line"
        ><input type="checkbox" bind:checked={showTelemetry} /> Spark telemetry panel</label
      ><label class="check-line"
        ><input type="checkbox" bind:checked={showConsole} /> Event console</label
      ><button onclick={() => exportJSON(draft, 'monarch-draft.json')}
        >Export recoverable draft</button
      ><label class="button import-label"
        >Import draft<input
          type="file"
          accept=".json"
          onchange={async (e) => {
            try {
              const file = e.currentTarget.files?.[0];
              if (!file) return;
              if (file.size > 8 * 1024 * 1024) throw new Error('Draft exceeds 8 MiB.');
              await onimport(restoreDraft(JSON.parse(await file.text())));
            } catch (error) {
              fail(error);
            }
          }}
        /></label
      >
    </div>
    <div class="surface">
      <h2>Connection & retention</h2>
      <p>Host: {location.host}</p>
      <p>
        Schema: {capabilities.schema_version ?? 'unknown'} · Driver: {driverState ?? 'unknown'}
      </p>
      <p class="muted">
        Drafts and layout are local to this browser. Shared records live in the companion. Tokens
        are limited to tab session storage.
      </p>
      <p class="muted">
        Event console: 400 lines / 128 KiB. Gallery: 40 records per page. Viewer: selected media
        only, 64 MiB per file. Polling backs off to 30 seconds and pauses while hidden.
      </p>
      <button onclick={disconnect}>Disconnect and forget token</button>
      <details>
        <summary>Capability contract</summary>
        <pre>{JSON.stringify(capabilities, null, 2)}</pre>
      </details>
    </div>
  </div>
</section>
