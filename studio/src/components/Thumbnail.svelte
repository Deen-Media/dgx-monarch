<script lang="ts">
  import type { Artifact } from '../contracts';
  import type { StudioClient } from '../api';
  import { whenNearViewport } from '../nearViewport';
  import { NO_THUMBNAIL, formatDuration, posterNote, videoPoster } from '../videoPoster';
  let { artifact, client }: { artifact: Artifact; client: StudioClient } = $props();
  let url = $state(''),
    status = $state('Loading preview'),
    // A video's length, from its own metadata, shown on its poster.
    length = $state(''),
    // The placeholder that stands until the preview replaces it.
    placeholder = $state<HTMLElement | null>(null),
    // The preview downloads once the thumbnail nears the visible area, so a long Queue page or
    // outputs strip does not start every fetch at once.
    near = $state(false);
  // Each snapshot poll replaces every artifact object. Fetch again only when the identity, or
  // the digest of an intermediate preview, changes.
  const id = $derived(artifact.id);
  const digest = $derived(artifact.intermediate ? artifact.metadata?.sha256 : undefined);
  // The companion makes thumbnails of images only; a video's poster is its first frame, drawn
  // in the browser, with a length the driver recorded as the fallback to the video's own.
  const video = $derived(artifact.kind === 'video');
  const recorded = $derived(Number(artifact.metadata?.duration) || null);
  $effect(() => {
    const element = placeholder;
    if (near || !element) return;
    return whenNearViewport(element, () => (near = true));
  });
  $effect(() => {
    if (!near) return;
    const api = client;
    void digest;
    url = '';
    status = 'Loading preview';
    length = '';
    const controller = new AbortController();
    let active = true,
      blobUrl = '';
    const show = (blob: Blob) => {
      blobUrl = URL.createObjectURL(blob);
      url = blobUrl;
    };
    if (video)
      videoPoster(api, id, digest, controller.signal)
        .then((result) => {
          if (!active) return;
          const shown = { ...result, duration: result.duration ?? recorded };
          length = formatDuration(shown.duration);
          if (shown.poster) show(shown.poster);
          else status = posterNote(shown);
        })
        .catch(() => {
          if (active) status = NO_THUMBNAIL;
        });
    else
      api
        .thumbnail(id, controller.signal)
        .then((blob) => {
          if (!active) return;
          show(blob);
        })
        .catch(() => {
          if (active) status = 'Preview unavailable';
        });
    return () => {
      active = false;
      controller.abort();
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  });
</script>

{#if url && video}<span class="poster"
    ><img
      src={url}
      alt={artifact.filename}
      decoding="async"
      onerror={() => {
        url = '';
        status = NO_THUMBNAIL;
      }}
    /><span class="length">▶{length ? ` ${length}` : ''}</span></span
  >{:else if url}<img
    src={url}
    alt={artifact.filename}
    loading="lazy"
    decoding="async"
    onerror={() => {
      url = '';
      status = 'Preview unavailable';
    }}
  />{:else}<span bind:this={placeholder}>{video ? '▷' : '▧'}</span><small>{status}</small>{/if}

<style>
  img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  small {
    font-size: 10px;
  }
  /* The poster fills the tile; its length sits in the corner, as video players show it. */
  .poster {
    position: relative;
    display: block;
    width: 100%;
    height: 100%;
    min-height: 0;
  }
  .poster img {
    display: block;
  }
  .length {
    position: absolute;
    right: 3px;
    bottom: 3px;
    padding: 1px 4px;
    border-radius: 3px;
    background: rgb(0 0 0 / 72%);
    color: #e6ebf1;
    font-size: 9px;
    font-weight: 600;
    line-height: 1.4;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
</style>
