<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import type { StudioClient } from '../api';
  import type { Artifact, ReuseArtifact } from '../contracts';
  import ComparisonDetails from './ComparisonDetails.svelte';
  import { displayUnknown } from '../model';
  let {
    artifact = null,
    compare = null,
    client,
    onreuse,
    onselect,
  }: {
    artifact?: Artifact | null;
    compare?: Artifact | null;
    client: StudioClient;
    onreuse: ReuseArtifact;
    onselect: (a: Artifact | null) => void;
  } = $props();
  let url = $state(''),
    compareUrl = $state(''),
    error = $state(''),
    loading = $state(false);
  let zoom = $state(1),
    mode = $state('fit'),
    wipe = $state(50),
    comparison = $state('side');
  let metadata = $state(false),
    viewport: HTMLDivElement;
  let video = $state<HTMLVideoElement>(),
    secondVideo = $state<HTMLVideoElement>();
  let muted = $state(true),
    loop = $state(true),
    sync = $state(false);
  let frameRate = $derived(artifact?.metadata?.frame_rate ?? artifact?.metadata?.fps);
  const mediaKey = $derived(
    JSON.stringify([
      artifact?.id,
      artifact?.intermediate ? artifact.metadata?.sha256 : null,
      compare?.id,
      compare?.intermediate ? compare.metadata?.sha256 : null,
    ]),
  );
  let generation = 0;
  let pan = { x: 0, y: 0, left: 0, top: 0, active: false };
  $effect(() => {
    const key = mediaKey;
    const [selected, secondary] = untrack(() => [artifact, compare]);
    const api = client;
    if (!key) return;
    const ticket = ++generation;
    const controller = new AbortController();
    untrack(() => {
      if (url) URL.revokeObjectURL(url);
      if (compareUrl) URL.revokeObjectURL(compareUrl);
    });
    url = '';
    compareUrl = '';
    error = '';
    zoom = 1;
    mode = 'fit';
    if (!selected) {
      loading = false;
      return;
    }
    loading = true;
    Promise.all([
      api.media(selected.id, controller.signal),
      secondary ? api.media(secondary.id, controller.signal) : Promise.resolve(null),
    ])
      .then(([first, second]) => {
        if (ticket !== generation) return;
        if (first.size > 64 * 1024 * 1024 || (second?.size ?? 0) > 64 * 1024 * 1024)
          throw new Error(
            'Media exceeds the 64 MiB viewer limit. Download from the source output directory.',
          );
        url = URL.createObjectURL(first);
        if (second) compareUrl = URL.createObjectURL(second);
      })
      .catch((e) => {
        if (ticket === generation) error = String(e.message ?? e);
      })
      .finally(() => {
        if (ticket === generation) loading = false;
      });
    return () => controller.abort();
  });
  onDestroy(() => {
    generation++;
    if (url) URL.revokeObjectURL(url);
    if (compareUrl) URL.revokeObjectURL(compareUrl);
  });
  function fullscreen() {
    viewport?.requestFullscreen().catch((e) => {
      error = e.message;
    });
  }
  function syncPlayback() {
    if (!sync || !video || !secondVideo) return;
    const duration = Number.isFinite(secondVideo.duration)
      ? secondVideo.duration
      : video.currentTime;
    const target = Math.min(video.currentTime, duration);
    if (Math.abs(secondVideo.currentTime - target) > 0.08) secondVideo.currentTime = target;
    secondVideo.playbackRate = video.playbackRate;
    if (video.paused || target >= duration) secondVideo.pause();
    else if (secondVideo.paused) void secondVideo.play().catch(() => {});
  }
  function stepFrame(direction: number) {
    if (!video || !frameRate) return;
    video.pause();
    video.currentTime = Math.max(
      0,
      Math.min(video.duration, video.currentTime + direction / frameRate),
    );
    syncPlayback();
  }
  $effect(() => {
    if (sync) syncPlayback();
  });
</script>

<section class="viewer">
  <div class="section-heading">
    <div>
      <span class="eyebrow">OUTPUT VIEWER</span>
      <h2>
        {compare ? 'Compare results' : artifact ? artifact.filename : 'A space for your next idea'}
      </h2>
    </div>
    <span class="tag">{artifact ? (artifact.kind ?? 'artifact') : 'No output selected'}</span>
  </div>
  <div class="viewer-toolbar" aria-label="Viewer controls">
    <button
      disabled={!url}
      class:active={mode === 'fit'}
      onclick={() => {
        mode = 'fit';
        zoom = 1;
      }}>Fit</button
    >
    <button
      disabled={!url}
      class:active={mode === 'actual'}
      onclick={() => {
        mode = 'actual';
        zoom = 1;
      }}>1:1</button
    >
    <button
      disabled={!url}
      onclick={() => {
        mode = 'actual';
        zoom = Math.max(0.1, zoom - 0.25);
      }}
      aria-label="Zoom out">−</button
    >
    <span class="mono">{Math.round(zoom * 100)}%</span>
    <button
      disabled={!url}
      onclick={() => {
        mode = 'actual';
        zoom = Math.min(8, zoom + 0.25);
      }}
      aria-label="Zoom in">+</button
    >
    <button disabled={!url} onclick={fullscreen}>Fullscreen</button>
    <button disabled={!artifact} class:active={metadata} onclick={() => (metadata = !metadata)}
      >Details</button
    >
    {#if url && artifact}<a class="button" href={url} download={artifact.filename ?? artifact.id}
        >Download</a
      >{/if}
  </div>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (the scrollable viewport supports keyboard panning) -->
  <div
    class="viewport"
    bind:this={viewport}
    class:empty={!artifact}
    class:actual={mode === 'actual'}
    role="region"
    aria-label="Output image or video"
    tabindex="0"
    onpointerdown={(e) => {
      if (mode !== 'actual' || (e.target as HTMLElement).tagName === 'VIDEO') return;
      pan = {
        x: e.clientX,
        y: e.clientY,
        left: viewport.scrollLeft,
        top: viewport.scrollTop,
        active: true,
      };
      viewport.setPointerCapture(e.pointerId);
    }}
    onpointermove={(e) => {
      if (pan.active) {
        viewport.scrollLeft = pan.left - (e.clientX - pan.x);
        viewport.scrollTop = pan.top - (e.clientY - pan.y);
      }
    }}
    onpointerup={() => (pan.active = false)}
    onpointercancel={() => (pan.active = false)}
  >
    {#if !artifact}
      <div class="empty-art">
        <svg viewBox="0 0 120 100" aria-hidden="true"
          ><path
            d="M60 53C27 8 4 14 18 51c7 18 27 18 42 7C26 78 36 99 58 66M60 53c33-45 56-39 42-2-7 18-27 18-42 7 34 20 24 41 2 8"
          /><path d="M60 46v29m0-28-6-9m6 9 6-9" /></svg
        >
        <h3>Make something remarkable.</h3>
        <p>Choose a workflow, shape your inputs,<br />then review a submission to begin.</p>
      </div>
    {:else if loading}<p role="status" class="muted">Loading selected media…</p>
    {:else if error}<p role="alert" class="error">{error}</p>
    {:else if url}
      <div
        class="media-stage"
        class:side={!!compareUrl && comparison === 'side'}
        class:wipe={!!compareUrl && comparison === 'wipe'}
        style:--zoom={zoom}
      >
        {#if artifact.kind === 'video'}<video
            bind:this={video}
            src={url}
            controls
            {muted}
            {loop}
            onplay={syncPlayback}
            onpause={syncPlayback}
            onseeked={syncPlayback}
            ontimeupdate={syncPlayback}
            onratechange={syncPlayback}><track kind="captions" /></video
          >
        {:else}<img
            src={url}
            alt={artifact.filename ?? 'Selected generated output'}
            draggable="false"
          />{/if}
        {#if compareUrl && compare}
          <div
            class="comparison-media"
            style:clip-path={comparison === 'wipe' ? `inset(0 0 0 ${wipe}%)` : undefined}
          >
            {#if compare.kind === 'video'}<video
                bind:this={secondVideo}
                src={compareUrl}
                controls
                {muted}
                {loop}><track kind="captions" /></video
              >
            {:else}<img
                src={compareUrl}
                alt={compare.filename ?? 'Comparison output'}
                draggable="false"
              />{/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
  {#if compare}<div class="viewer-toolbar">
      <button class:active={comparison === 'side'} onclick={() => (comparison = 'side')}
        >Side by side</button
      ><button
        disabled={artifact?.kind === 'video' || compare?.kind === 'video'}
        class:active={comparison === 'wipe'}
        onclick={() => (comparison = 'wipe')}>Image wipe</button
      >{#if comparison === 'wipe'}<label
          >Wipe <input type="range" min="0" max="100" bind:value={wipe} /></label
        >{/if}<button onclick={() => onselect(null)}>Exit comparison</button>
    </div>{/if}
  {#if artifact?.kind === 'video'}<div class="viewer-toolbar">
      <button disabled={!frameRate} onclick={() => stepFrame(-1)}>Previous frame</button><button
        disabled={!frameRate}
        onclick={() => stepFrame(1)}>Next frame</button
      >
      <small
        >{frameRate
          ? `${frameRate} fps step`
          : 'Frame rate unknown; use native seek controls'}</small
      >
      <label><input type="checkbox" bind:checked={muted} /> Mute</label><label
        ><input type="checkbox" bind:checked={loop} /> Loop</label
      >{#if compare?.kind === 'video'}<label
          ><input type="checkbox" bind:checked={sync} /> Synchronize playback time</label
        ><small
          >First video is the playback leader. The shorter video holds at its end; timestamps
          synchronize, not source frames.</small
        >{/if}
    </div>{/if}
  {#if artifact}<div class="viewer-footer">
      <span class="muted"
        >{artifact.metadata?.width ?? '?'} × {artifact.metadata?.height ?? '?'} · {artifact.id} · {artifact.created_at ??
          'Time unknown'}</span
      >
      <div>
        <button onclick={() => onreuse(artifact)}>Reuse settings</button><button
          onclick={() => onreuse(artifact, 'edit')}>Send to edit</button
        ><button onclick={() => onreuse(artifact, 'upscale')}>Upscale</button>
      </div>
    </div>{/if}
  {#if artifact && compare}<ComparisonDetails left={artifact} right={compare} {client} />{/if}
  {#if metadata && artifact}<div class="metadata-panel">
      <dl>
        <dt>Job</dt>
        <dd>{displayUnknown(artifact.job_id)}</dd>
        <dt>Output state</dt>
        <dd>
          {artifact.intermediate
            ? 'Intermediate preview'
            : displayUnknown(artifact.state ?? artifact.status ?? 'Recorded output')}
        </dd>
        <dt>Duration / frame rate</dt>
        <dd>
          {displayUnknown(artifact.metadata?.duration)} s / {displayUnknown(
            artifact.metadata?.frame_rate ?? artifact.metadata?.fps,
          )} fps
        </dd>
      </dl>
      <pre>{JSON.stringify({ artifact, comparison: compare }, null, 2)}</pre>
    </div>{/if}
</section>
