<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { Connection } from '../connection.svelte';
  import type { Draft } from '../model';
  import type { ChangeDraft } from '../contracts';
  import { navigation } from '../navigation';
  let {
    connection,
    draft,
    activeCount,
    view,
    presentation = $bindable(),
    width,
    showTelemetry = $bindable(),
    showConsole = $bindable(),
    navigate,
    changeDraft,
    openPalette,
    children,
  }: {
    connection: Connection;
    draft: Draft;
    activeCount: number;
    view: string;
    presentation: boolean;
    width: number;
    showTelemetry: boolean;
    showConsole: boolean;
    navigate: (view: string) => void;
    changeDraft: ChangeDraft;
    openPalette: () => void;
    children: Snippet;
  } = $props();
</script>

<div class="app-shell" class:presentation style:--control-width={`${width}px`}>
  {#if !presentation}<aside class="nav-rail" aria-label="Primary navigation">
      <a
        class="brand-mark"
        href="#workspace"
        aria-label="Monarch Studio home"
        onclick={(e) => {
          e.preventDefault();
          navigate('workspace');
        }}
        ><svg
          viewBox="0 0 40 40"
          aria-hidden="true"
          class:flapping={activeCount > 0 && !connection.hidden}
          ><path class="wing left" d="M20 20C8 1 0 7 6 20c3 6 9 7 14 1-12 9-6 17 0 4Z" /><path
            class="wing right"
            d="M20 20C32 1 40 7 34 20c-3 6-9 7-14 1 12 9 6 17 0 4Z"
          /></svg
        ></a
      >
      <span class="rail-divider"></span>
      {#each navigation as item}<button
          class:current={view === item.id}
          onclick={() => navigate(item.id)}
          title={item.label}
          aria-label={item.label}
          aria-current={view === item.id ? 'page' : undefined}
          ><span aria-hidden="true">{item.icon}</span><small>{item.label}</small
          >{#if item.id === 'queue' && activeCount}<b>{activeCount}</b>{/if}</button
        >{/each}
      <button
        class="palette-launch"
        onclick={openPalette}
        title="Command palette (Ctrl+K)"
        aria-label="Command palette"><span>⌕</span><small>Search</small></button
      >
    </aside>{/if}
  <div class="main-shell">
    <header class="app-header">
      <div class="identity">
        <span class="wordmark">MONARCH <em>STUDIO</em></span>
        <div class="host-line">
          <span class="status-dot" class:healthy={connection.state === 'connected'}></span><span
            >{connection.capabilities.host ?? location.host}</span
          ><span class="muted">/</span><span
            class="connection"
            title={connection.lastSeen ? `Last sample ${connection.lastSeen}` : ''}
            >{connection.state}</span
          >
        </div>
      </div>
      {#if !presentation}<div class="context-select">
          <label for="active-project" class="sr-only">Active project</label><select
            id="active-project"
            value={draft.project_id}
            onchange={(e) => changeDraft({ project_id: e.currentTarget.value })}
            ><option value="">All projects</option
            >{#each connection.snapshot.records.projects ?? [] as project}<option value={project.id}
                >{project.data?.name ?? project.id}</option
              >{/each}</select
          >
        </div>{/if}
      <div class="header-actions">
        <span class="activity-pill">{activeCount ? `${activeCount} active` : 'No active jobs'}</span
        ><button class="icon-button" onclick={() => (presentation = !presentation)}
          >{presentation ? 'Exit presentation' : 'Present'}</button
        >{#if !presentation}<button
            class="icon-button"
            onclick={openPalette}
            aria-label="Open command palette">⌘ K</button
          >{/if}
      </div>
    </header>

    {@render children()}

    {#if connection.connected && !presentation}<footer class="app-footer">
        <span
          ><span class="status-dot" class:healthy={connection.state === 'connected'}></span>
          Companion {connection.state}</span
        ><span class="muted">{draft.workflowName || 'No workflow'} · {draft.operation}</span>
        <div>
          <button onclick={() => (showTelemetry = !showTelemetry)}
            >Telemetry {showTelemetry ? '−' : '+'}</button
          ><button onclick={() => (showConsole = !showConsole)}
            >Console {showConsole ? '−' : '+'}</button
          >
        </div>
      </footer>{/if}
    {#if showConsole && connection.connected && !presentation}<section
        class="console-panel"
        aria-label="Companion event console"
      >
        <div class="panel-title">
          <h2>Event console</h2>
          <button onclick={() => (connection.logs = [])}>Clear view</button><button
            onclick={() => (showConsole = false)}
            aria-label="Close console">×</button
          >
        </div>
        <pre>{connection.logs.join('\n') || 'No events received in this session.'}</pre>
      </section>{/if}
  </div>
</div>
