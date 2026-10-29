<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import type { StudioClient } from '../api';
  import type { Artifact, Job, ReuseArtifact, Seeds } from '../contracts';
  import type { FollowedJob } from '../jobTracker.svelte';
  import ComparisonDetails from './ComparisonDetails.svelte';
  import JobStatusLine from './JobStatusLine.svelte';
  import OutputSeeds from './OutputSeeds.svelte';
  import { displayUnknown, formatTime } from '../model';
  import { frameTarget } from '../mediaPlayback';
  import { pendingView } from '../jobFollow';
  import {
    NO_MEDIA,
    currentMedia,
    decodeFailed,
    failLoad,
    finishLoad,
    startLoad,
    type ViewerMedia,
  } from '../viewerMedia';
  let {
    artifact = null,
    compare = null,
    job = null,
    working = false,
    client,
    jobs,
    onuseseed,
    onreuse,
    onselect,
    onqueue,
    onconsent,
  }: {
    artifact?: Artifact | null;
    compare?: Artifact | null;
    /** The job the viewer follows, or null. */
    job?: FollowedJob | null;
    /** Whether any job is in flight; the frame's activity border follows it. */
    working?: boolean;
    client: StudioClient;
    /** Snapshot job rows; without them, as in presentation mode, the footer shows no seed. */
    jobs?: Job[];
    onuseseed?: (seeds: Seeds) => void;
    onreuse: ReuseArtifact;
    onselect: (a: Artifact | null) => void;
    onqueue?: () => void;
    /** Open Operations for the consent the followed job's failure asks for. */
    onconsent?: () => void;
  } = $props();
  // The loaded URLs with their media kinds; `shown` is what belongs to the current selection.
  let media = $state.raw<ViewerMedia>(NO_MEDIA);
  let zoom = $state(1),
    mode = $state('fit'),
    wipe = $state(50),
    comparison = $state('side');
  let metadata = $state(false),
    viewport: HTMLDivElement;
  let mediaDuration = $state(NaN);
  let video = $state<HTMLVideoElement>(),
    secondVideo = $state<HTMLVideoElement>();
  let muted = $state(true),
    loop = $state(true),
    sync = $state(false);
  let frameRate = $derived(artifact?.metadata?.frame_rate ?? artifact?.metadata?.fps);
  let canStep = $derived(frameTarget(0, mediaDuration, frameRate, 1) !== null);
  const mediaKey = $derived(
    JSON.stringify([
      artifact?.id,
      artifact?.intermediate ? artifact.metadata?.sha256 : null,
      compare?.id,
      compare?.intermediate ? compare.metadata?.sha256 : null,
    ]),
  );
  const shown = $derived(currentMedia(media, mediaKey));
  // With no output shown, the followed job's workflow and state stand in for one.
  const pending = $derived(!artifact && job ? pendingView(job) : null);
  let alive = true;
  let pan = { x: 0, y: 0, left: 0, top: 0, active: false };
  function revoke(old: ViewerMedia) {
    if (old.url) URL.revokeObjectURL(old.url);
    if (old.compareUrl) URL.revokeObjectURL(old.compareUrl);
  }
  $effect(() => {
    const key = mediaKey;
    const [selected, secondary] = untrack(() => [artifact, compare]);
    const api = client;
    // The effect writes the media state but never reads it outside untrack: a read would make
    // each finished load start the next one.
    const previous = untrack(() => media);
    revoke(previous);
    const next = startLoad(previous, key, selected, secondary);
    media = next;
    const ticket = next.ticket;
    mediaDuration = NaN;
    zoom = 1;
    mode = 'fit';
    if (selected?.kind === 'video' || secondary?.kind === 'video') comparison = 'side';
    if (!selected) return;
    const controller = new AbortController();
    Promise.all([
      api.media(selected.id, controller.signal),
      secondary ? api.media(secondary.id, controller.signal) : Promise.resolve(null),
    ])
      .then(([first, second]) => {
        if (!alive || media.ticket !== ticket) return;
        if (first.size > 64 * 1024 * 1024 || (second?.size ?? 0) > 64 * 1024 * 1024)
          throw new Error(
            'Media exceeds the 64 MiB viewer limit. Download from the source output directory.',
          );
        const secondUrl = second ? URL.createObjectURL(second) : '';
        media = finishLoad(media, ticket, URL.createObjectURL(first), secondUrl);
      })
      .catch((e) => {
        if (alive) media = failLoad(media, ticket, String(e.message ?? e));
      });
    return () => controller.abort();
  });
  onDestroy(() => {
    alive = false;
    revoke(media);
  });
  function fullscreen() {
    viewport?.requestFullscreen().catch((e) => {
      media = { ...media, error: e.message };
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
    if (!video) return;
    const target = frameTarget(video.currentTime, video.duration, frameRate, direction);
    if (target === null) return;
    video.pause();
    video.currentTime = target;
    syncPlayback();
  }
  /** An element could not decode its source; only the element showing the current URL counts. */
  function decode(element: Element, comparing = false) {
    media = decodeFailed(media, element.getAttribute('src'), comparing);
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
        {compare
          ? 'Compare results'
          : artifact
            ? artifact.filename
            : (pending?.title ?? 'A space for your next idea')}
      </h2>
    </div>
    <span class="tag"
      >{artifact ? (artifact.kind ?? 'artifact') : (pending?.tag ?? 'No output selected')}</span
    >
  </div>
  <JobStatusLine {job} {onqueue} {onconsent} />
  <div class="viewer-toolbar" aria-label="Viewer controls">
    <button
      disabled={!shown.url}
      class:active={mode === 'fit'}
      aria-pressed={mode === 'fit'}
      onclick={() => {
        mode = 'fit';
        zoom = 1;
      }}>Fit</button
    >
    <button
      disabled={!shown.url}
      class:active={mode === 'actual'}
      aria-pressed={mode === 'actual'}
      onclick={() => {
        mode = 'actual';
        zoom = 1;
      }}>1:1</button
    >
    <button
      disabled={!shown.url}
      onclick={() => {
        mode = 'actual';
        zoom = Math.max(0.1, zoom - 0.25);
      }}
      aria-label="Zoom out">−</button
    >
    <span class="mono">{Math.round(zoom * 100)}%</span>
    <button
      disabled={!shown.url}
      onclick={() => {
        mode = 'actual';
        zoom = Math.min(8, zoom + 0.25);
      }}
      aria-label="Zoom in">+</button
    >
    <button disabled={!shown.url} onclick={fullscreen}>Fullscreen</button>
    <button
      disabled={!artifact}
      class:active={metadata}
      aria-expanded={metadata}
      aria-controls="media-metadata"
      onclick={() => (metadata = !metadata)}>Details</button
    >
    {#if shown.url && artifact}<a
        class="button"
        href={shown.url}
        download={artifact.filename ?? artifact.id}>Download</a
      >{/if}
  </div>
  <div class="viewer-frame" class:working>
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
      {#if pending}<div class="empty-art">
          <h3>{pending.heading}</h3>
          <p>{pending.note}</p>
        </div>
      {:else if !artifact}
        <div class="empty-art">
          <svg viewBox="0 0 120 100" aria-hidden="true"
            ><path
              d="M60 53C27 8 4 14 18 51c7 18 27 18 42 7C26 78 36 99 58 66M60 53c33-45 56-39 42-2-7 18-27 18-42 7 34 20 24 41 2 8"
            /><path d="M60 46v29m0-28-6-9m6 9 6-9" /></svg
          >
          <h3>Make something remarkable.</h3>
          <p>Choose a workflow, shape your inputs,<br />then press Generate to begin.</p>
        </div>
      {:else if shown.loading}<p role="status" class="muted">Loading selected media…</p>
      {:else if shown.error}<p role="alert" class="error">{shown.error}</p>
      {:else if shown.url}
        <!-- Each element takes its kind from the media it loaded, never from a newer selection. -->
        <div
          class="media-stage"
          class:side={!!shown.compareUrl && comparison === 'side'}
          class:wipe={!!shown.compareUrl && comparison === 'wipe'}
          style:--zoom={zoom}
        >
          {#if shown.kind === 'video'}<video
              bind:this={video}
              src={shown.url}
              controls
              {muted}
              {loop}
              onloadedmetadata={() => (mediaDuration = video?.duration ?? NaN)}
              ondurationchange={() => (mediaDuration = video?.duration ?? NaN)}
              onerror={(e) => decode(e.currentTarget)}
              onplay={syncPlayback}
              onpause={syncPlayback}
              onseeked={syncPlayback}
              ontimeupdate={syncPlayback}
              onratechange={syncPlayback}><track kind="captions" /></video
            >
          {:else}<img
              src={shown.url}
              alt={artifact.filename ?? 'Selected generated output'}
              draggable="false"
              onerror={(e) => decode(e.currentTarget)}
            />{/if}
          {#if shown.compareUrl && compare}
            <div
              class="comparison-media"
              style:clip-path={comparison === 'wipe' ? `inset(0 0 0 ${wipe}%)` : undefined}
            >
              {#if shown.compareKind === 'video'}<video
                  bind:this={secondVideo}
                  src={shown.compareUrl}
                  controls
                  {muted}
                  {loop}
                  onerror={(e) => decode(e.currentTarget, true)}><track kind="captions" /></video
                >
              {:else}<img
                  src={shown.compareUrl}
                  alt={compare.filename ?? 'Comparison output'}
                  draggable="false"
                  onerror={(e) => decode(e.currentTarget, true)}
                />{/if}
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </div>
  {#if compare}<div class="viewer-toolbar">
      <button
        class:active={comparison === 'side'}
        aria-pressed={comparison === 'side'}
        onclick={() => (comparison = 'side')}>Side by side</button
      ><button
        disabled={artifact?.kind === 'video' || compare?.kind === 'video'}
        class:active={comparison === 'wipe'}
        aria-pressed={comparison === 'wipe'}
        onclick={() => (comparison = 'wipe')}>Image wipe</button
      >{#if comparison === 'wipe'}<label
          >Wipe <input type="range" min="0" max="100" bind:value={wipe} /></label
        >{/if}<button onclick={() => onselect(null)}>Exit comparison</button>
    </div>{/if}
  {#if artifact?.kind === 'video'}<div class="viewer-toolbar">
      <button disabled={!canStep} onclick={() => stepFrame(-1)}>Previous frame</button><button
        disabled={!canStep}
        onclick={() => stepFrame(1)}>Next frame</button
      >
      <small
        >{canStep
          ? `${frameRate} fps step`
          : 'Frame stepping needs valid duration and frame rate; use native seek controls'}</small
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
        >{artifact.metadata?.width ?? '?'} × {artifact.metadata?.height ?? '?'} · {artifact.id} · {formatTime(
          artifact.created_at,
        )}</span
      >
      {#if jobs && artifact.job_id}<OutputSeeds
          {client}
          jobId={artifact.job_id}
          job={jobs.find((row) => row.id === artifact.job_id)}
          {onuseseed}
        />{/if}
      <div>
        <button onclick={() => onreuse(artifact)}>Reuse settings</button><button
          onclick={() => onreuse(artifact, 'edit')}>Send to edit</button
        ><button onclick={() => onreuse(artifact, 'upscale')}>Upscale</button>
      </div>
    </div>{/if}
  {#if artifact && compare}<ComparisonDetails left={artifact} right={compare} {client} />{/if}
  {#if metadata && artifact}<div class="metadata-panel" id="media-metadata">
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
