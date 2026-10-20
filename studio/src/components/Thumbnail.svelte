<script lang="ts">
  import type { Artifact } from '../contracts';
  import type { StudioClient } from '../api';
  let { artifact, client }: { artifact: Artifact; client: StudioClient } = $props();
  let url = $state(''),
    status = $state('Loading preview');
  $effect(() => {
    const id = artifact.id;
    const digest = artifact.intermediate ? artifact.metadata?.sha256 : undefined;
    const api = client;
    void digest;
    url = '';
    status = 'Loading preview';
    const controller = new AbortController();
    let active = true,
      blobUrl = '';
    api
      .thumbnail(id, controller.signal)
      .then((blob) => {
        if (!active) return;
        blobUrl = URL.createObjectURL(blob);
        url = blobUrl;
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

{#if url}<img
    src={url}
    alt={artifact.filename}
    loading="lazy"
    decoding="async"
    onerror={() => {
      url = '';
      status = 'Preview unavailable';
    }}
  />{:else}<span>{artifact.kind === 'video' ? '▷' : '▧'}</span><small>{status}</small>{/if}

<style>
  img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  small {
    font-size: 10px;
  }
</style>
