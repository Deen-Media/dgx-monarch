<script lang="ts">
  import type { Connection } from '../connection.svelte';
  import type { Binding, Draft } from '../model';
  import type { ChangeDraft, InspectWorkflow, OperationRequest, ReuseArtifact } from '../contracts';
  import { utilities } from '../navigation';
  import Grids from './Grids.svelte';
  import Fleet from './Fleet.svelte';
  import Operations from '../components/Operations.svelte';
  let {
    view,
    connection,
    draft,
    bindings,
    reuse,
    restoreDraft,
    changeDraft,
    inspect,
    navigate,
    setNotice,
    request = null,
    onrefresh,
  }: {
    /** The open tab's route id: grid, fleet or operations. */
    view: string;
    connection: Connection;
    draft: Draft;
    bindings: Binding[];
    reuse: ReuseArtifact;
    restoreDraft: (draft: Draft) => Promise<void>;
    changeDraft: ChangeDraft;
    inspect: InspectWorkflow;
    navigate: (view: string) => void;
    setNotice: (message: string) => void;
    /** The operation Operations opens next, from the palette or a failed job's consent. */
    request?: OperationRequest | null;
    onrefresh: () => void;
  } = $props();
  const tab = $derived(utilities.find((item) => item.id === view) ?? utilities[0]);
</script>

<nav class="utility-tabs" aria-label="Utilities">
  <span class="eyebrow">UTILITIES</span>
  <div class="tabs">
    {#each utilities as item (item.id)}<button
        class:active={item.id === tab.id}
        aria-current={item.id === tab.id ? 'page' : undefined}
        onclick={() => navigate(item.id)}>{item.label}</button
      >{/each}
  </div>
  <p class="muted">{tab.summary}</p>
</nav>
{#if tab.id === 'grid'}<Grids
    {reuse}
    client={connection.client}
    {draft}
    {bindings}
    {restoreDraft}
  />{:else if tab.id === 'fleet'}<Fleet
    client={connection.client}
    jobs={connection.snapshot.jobs}
    history={connection.metricHistory}
    presentation={connection.capabilities.presentation}
    telemetry={connection.snapshot.telemetry}
    workflows={connection.workflows}
    {draft}
    {changeDraft}
    {inspect}
    {navigate}
    {setNotice}
  />{:else}<Operations
    client={connection.client}
    capabilities={connection.capabilities}
    {request}
    operations={connection.snapshot.operations}
    {onrefresh}
  />{/if}

<style>
  .utility-tabs {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 40px 0;
  }
  .utility-tabs .tabs {
    margin: 6px 0 0;
  }
  .utility-tabs p {
    margin: 10px 0 0;
    font-size: 12px;
  }
  .utility-tabs + :global(.page-section) {
    padding-top: 22px;
  }
  @media (max-width: 1050px) {
    .utility-tabs {
      padding: 22px 25px 0;
    }
  }
  @media (max-width: 760px) {
    .utility-tabs {
      padding: 18px 18px 0;
    }
  }
</style>
