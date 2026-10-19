<script lang="ts">
  import { encodeMaskAlpha } from '../mask';
  import { onDestroy } from 'svelte';
  import type { StudioClient } from '../api';
  import type { Binding, Draft } from '../model';
  import { bindingType, fieldLabel } from '../model';
  import type { ChangeDraft } from '../contracts';
  let {
    client,
    draft,
    bindings = $bindable(),
    changeDraft,
  }: {
    client: StudioClient;
    draft: Draft;
    bindings: Binding[];
    changeDraft: ChangeDraft;
  } = $props();
  let canvas = $state<HTMLCanvasElement>(null!);
  let width = $state(512),
    height = $state(512),
    brush = $state(32),
    erase = $state(false);
  let target = $state(''),
    source = $state(''),
    name = $state(''),
    error = $state('');
  let busy = $state(false),
    drawing = false,
    mode = $state('input');
  let file = $state<File | undefined>();
  let receipt = $state<{ id: string; width: number; height: number; binding_value: string } | null>(
    null,
  );
  const limit = 2 * 1024 * 1024;
  const choices = $derived(
    bindings.filter((b) => b.resolved !== false && ['STRING', 'COMBO'].includes(bindingType(b))),
  );
  function clear() {
    sourceGeneration++;
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    receipt = null;
  }
  let sourceGeneration = 0;
  function reset() {
    sourceGeneration++;
    if (source) URL.revokeObjectURL(source);
    source = '';
    file = undefined;
    name = '';
    receipt = null;
    clear();
  }
  async function replace(next: File | undefined) {
    if (!next) return;
    error = '';
    if (next.size > limit || !['image/png', 'image/jpeg', 'image/webp'].includes(next.type)) {
      error = 'Choose a PNG, JPEG or WebP image no larger than 2 MiB.';
      return;
    }
    const ticket = ++sourceGeneration;
    const nextUrl = URL.createObjectURL(next);
    const image = new Image();
    image.src = nextUrl;
    try {
      await image.decode();
      if (ticket !== sourceGeneration) {
        URL.revokeObjectURL(nextUrl);
        return;
      }
      if (image.naturalWidth > 4096 || image.naturalHeight > 4096)
        throw new Error('Editor dimensions are limited to 4096 pixels per side.');
      reset();
      source = nextUrl;
      file = next;
      name = next.name;
      width = image.naturalWidth;
      height = image.naturalHeight;
    } catch (cause) {
      URL.revokeObjectURL(nextUrl);
      error = String(cause);
    }
  }
  let previous: { x: number; y: number } | null = null;
  function paint(event: PointerEvent) {
    if (!drawing) return;
    sourceGeneration++;
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) * canvas.width) / rect.width;
    const y = ((event.clientY - rect.top) * canvas.height) / rect.height;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
    context.fillStyle = '#87f900';
    if (previous) {
      context.strokeStyle = '#87f900';
      context.lineWidth = brush;
      context.lineCap = 'round';
      context.beginPath();
      context.moveTo(previous.x, previous.y);
      context.lineTo(x, y);
      context.stroke();
    }
    previous = { x, y };
    context.beginPath();
    context.arc(x, y, brush / 2, 0, Math.PI * 2);
    context.fill();
    receipt = null;
  }
  async function maskBlob(): Promise<Blob> {
    const output = document.createElement('canvas');
    output.width = canvas.width;
    output.height = canvas.height;
    const context = output.getContext('2d')!;
    const pixels = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
    pixels.data.set(encodeMaskAlpha(pixels.data));
    context.putImageData(pixels, 0, 0);
    return new Promise((resolve, reject) =>
      output.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed.'))),
        'image/png',
      ),
    );
  }
  async function upload() {
    busy = true;
    error = '';
    const originalWorkflow = draft.workflow,
      originalTarget = target,
      originalMode = mode;
    const originalSource = sourceGeneration,
      originalName = name,
      originalWidth = width,
      originalHeight = height;
    try {
      const blob = originalMode === 'mask' ? await maskBlob() : file;
      if (!blob) throw new Error('Choose an input image first.');
      if (!originalTarget) throw new Error('Choose the actual workflow input binding.');
      if (blob.size > limit) throw new Error('Encoded asset exceeds the 2 MiB upload limit.');
      const data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1]);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });
      const result = await client.request<{
        id: string;
        width: number;
        height: number;
        binding_value: string;
      }>('/assets', {
        filename: originalMode === 'mask' ? 'studio-mask.png' : originalName,
        media_type: blob.type,
        data_base64: data,
        kind: originalMode,
      });
      if (
        draft.workflow !== originalWorkflow ||
        target !== originalTarget ||
        mode !== originalMode ||
        sourceGeneration !== originalSource ||
        width !== originalWidth ||
        height !== originalHeight
      )
        throw new Error(
          `Asset ${result.binding_value} uploaded, but the draft or editor changed. It was not attached; select the intended input and upload again.`,
        );
      receipt = result;
      bindings = bindings.map((binding) =>
        binding.id === target && bindingType(binding) === 'COMBO'
          ? {
              ...binding,
              schema: {
                ...binding.schema,
                choices: [
                  ...new Set([
                    ...(binding.schema?.choices ?? binding.options ?? []),
                    result.binding_value,
                  ]),
                ],
              },
            }
          : binding,
      );
      changeDraft({
        values: { ...draft.values, [target]: result.binding_value },
        parent_artifact_ids: [...new Set([...draft.parent_artifact_ids, result.id])],
      });
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      busy = false;
    }
  }
  onDestroy(() => {
    if (source) URL.revokeObjectURL(source);
  });
</script>

<div class="asset-editor">
  <h3>Input image and mask</h3>
  <label
    >Asset type<select bind:value={mode}
      ><option value="input">Input image</option><option value="mask">Paint mask</option></select
    ></label
  >
  <label
    >Actual workflow input<select bind:value={target}
      ><option value="">Choose a binding</option>{#each choices as binding}<option
          value={binding.id}>{fieldLabel(binding)} · {binding.node_id}</option
        >{/each}</select
    ></label
  >
  <label class="button import-label"
    >{source ? 'Replace source image' : 'Choose source image'}<input
      type="file"
      accept="image/png,image/jpeg,image/webp"
      onchange={(event) => void replace(event.currentTarget.files?.[0])}
    /></label
  >
  <button onclick={reset}>Clear editor</button>
  <p>
    {name || 'Blank canvas'} · {width} × {height} · {width === height
      ? 'Square'
      : width > height
        ? 'Landscape'
        : 'Portrait'}
  </p>
  {#if mode === 'mask'}
    <div class="input-row">
      <label>Width<input type="number" min="1" max="4096" bind:value={width} /></label><label
        >Height<input type="number" min="1" max="4096" bind:value={height} /></label
      >
    </div>
    <label
      >Brush diameter {brush}px<input type="range" min="1" max="256" bind:value={brush} /></label
    >
    <label><input type="checkbox" bind:checked={erase} /> Erase paint</label>
    <button onclick={clear}>Clear mask</button><button
      onclick={() => {
        sourceGeneration++;
        const c = canvas.getContext('2d');
        if (c) {
          c.globalCompositeOperation = 'source-over';
          c.fillStyle = '#87f900';
          c.fillRect(0, 0, canvas.width, canvas.height);
        }
        receipt = null;
      }}>Mask all</button
    >
    <p class="muted">
      Green marks the masked region. Uploaded PNG uses transparent alpha for masked pixels (Comfy
      LoadImage mask output). Changing dimensions clears paint.
    </p>
  {/if}
  <div class="canvas-wrap" style:aspect-ratio={`${width}/${height}`}>
    {#if source}<img src={source} alt="Actual source selected for upload" />{/if}
    {#if mode === 'mask'}<canvas
        bind:this={canvas}
        width={Math.min(4096, Math.max(1, width || 1))}
        height={Math.min(4096, Math.max(1, height || 1))}
        aria-label="Paint mask with pointer; Mask all and Clear mask buttons are keyboard accessible"
        onpointerdown={(event) => {
          drawing = true;
          previous = null;
          canvas.setPointerCapture(event.pointerId);
          paint(event);
        }}
        onpointermove={paint}
        onpointerup={() => (drawing = false)}
        onpointercancel={() => (drawing = false)}
      ></canvas>{/if}
  </div>
  <button class="primary" disabled={busy || !target || (mode === 'input' && !file)} onclick={upload}
    >{busy ? 'Uploading…' : 'Upload and bind asset'}</button
  >
  {#if receipt}<p role="status">
      Submitted asset: {receipt.binding_value} · {receipt.width} × {receipt.height} · ID {receipt.id}
    </p>{/if}
  {#if target && draft.values[target]}<p class="mono">
      Current bound value: {String(draft.values[target])}
    </p>
    <button
      onclick={() => {
        changeDraft({ values: { ...draft.values, [target]: '' } });
        receipt = null;
      }}>Clear bound input</button
    >{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</div>

<style>
  .asset-editor {
    display: grid;
    gap: 10px;
  }
  .canvas-wrap {
    position: relative;
    width: 100%;
    max-height: 460px;
    overflow: hidden;
    background: repeating-conic-gradient(#333 0% 25%, #20242a 0% 50%) 50% / 20px 20px;
  }
  .canvas-wrap img,
  .canvas-wrap canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: fill;
  }
  .canvas-wrap canvas {
    opacity: 0.6;
    touch-action: none;
    cursor: crosshair;
  }
</style>
