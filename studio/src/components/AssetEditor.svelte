<script lang="ts">
  import { encodeMaskAlpha, maskGeometry, maskIsEmpty, maskPoint } from '../mask';
  import { assetEditor as editor, maskCanvas, type AssetReceipt } from '../assetEditorState.svelte';
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
  // Source, paint, target and mode live in assetEditorState, so tab and view switches keep them.
  let canvas = $state<HTMLCanvasElement>(null!);
  let drawing = false;
  const geometry = $derived(maskGeometry(editor.width, editor.height));
  const limit = 2 * 1024 * 1024;
  const choices = $derived(
    bindings.filter((b) => b.resolved !== false && ['STRING', 'COMBO'].includes(bindingType(b))),
  );
  const targetAvailable = $derived(choices.some((binding) => binding.id === editor.target));
  function wipe() {
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  }
  function clear() {
    wipe();
    editor.clearMask();
  }
  function reset() {
    editor.reset();
    wipe();
  }
  function dimensionsChanged() {
    editor.clearMask();
    previous = null;
  }
  async function replace(next: File | undefined) {
    if (!next) return;
    editor.error = '';
    if (next.size > limit || !['image/png', 'image/jpeg', 'image/webp'].includes(next.type)) {
      editor.error = 'Choose a PNG, JPEG or WebP image no larger than 2 MiB.';
      return;
    }
    const ticket = ++editor.generation;
    const nextUrl = URL.createObjectURL(next);
    const image = new Image();
    image.src = nextUrl;
    try {
      await image.decode();
      if (ticket !== editor.generation) {
        URL.revokeObjectURL(nextUrl);
        return;
      }
      if (image.naturalWidth > 4096 || image.naturalHeight > 4096)
        throw new Error('Editor dimensions are limited to 4096 pixels per side.');
      editor.useSource(nextUrl, next, image.naturalWidth, image.naturalHeight);
      wipe();
    } catch (cause) {
      URL.revokeObjectURL(nextUrl);
      editor.error = String(cause);
    }
  }
  let previous: { x: number; y: number } | null = null;
  function paint(event: PointerEvent) {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const point = maskPoint(event.clientX, event.clientY, rect, canvas.width, canvas.height);
    if (!point) return;
    const { x, y } = point;
    const context = canvas.getContext('2d');
    if (!context) return;
    editor.changed();
    context.globalCompositeOperation = editor.erase ? 'destination-out' : 'source-over';
    context.fillStyle = '#87f900';
    if (previous) {
      context.strokeStyle = '#87f900';
      context.lineWidth = editor.brush;
      context.lineCap = 'round';
      context.beginPath();
      context.moveTo(previous.x, previous.y);
      context.lineTo(x, y);
      context.stroke();
    }
    previous = { x, y };
    context.beginPath();
    context.arc(x, y, editor.brush / 2, 0, Math.PI * 2);
    context.fill();
  }
  async function maskBlob(): Promise<Blob> {
    const pixels = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
    if (maskIsEmpty(pixels.data))
      throw new Error(
        'The mask is empty, so it would mask nothing. Paint the area to change, or use Mask all, before uploading.',
      );
    const output = document.createElement('canvas');
    output.width = canvas.width;
    output.height = canvas.height;
    const context = output.getContext('2d')!;
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
    editor.busy = true;
    editor.error = '';
    const originalWorkflow = draft.workflow,
      originalTarget = editor.target,
      originalMode = editor.mode;
    const originalSource = editor.generation,
      originalName = editor.name,
      originalWidth = editor.width,
      originalHeight = editor.height;
    try {
      if (!geometry.valid) throw new Error('Enter whole dimensions between 1 and 4096 pixels.');
      if (!originalTarget || !targetAvailable)
        throw new Error('Choose the actual workflow input binding.');
      const blob = originalMode === 'mask' ? await maskBlob() : editor.file;
      if (!blob) throw new Error('Choose an input image first.');
      if (blob.size > limit) throw new Error('Encoded asset exceeds the 2 MiB upload limit.');
      const data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1]);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });
      const result = await client.request<AssetReceipt>('/assets', {
        filename: originalMode === 'mask' ? 'studio-mask.png' : originalName,
        media_type: blob.type,
        data_base64: data,
        kind: originalMode,
      });
      if (
        draft.workflow !== originalWorkflow ||
        editor.target !== originalTarget ||
        editor.mode !== originalMode ||
        editor.generation !== originalSource ||
        editor.width !== originalWidth ||
        editor.height !== originalHeight
      )
        throw new Error(
          `Asset ${result.binding_value} uploaded, but the draft or editor changed. It was not attached; select the intended input and upload again.`,
        );
      editor.receipt = result;
      bindings = bindings.map((binding) =>
        binding.id === originalTarget && bindingType(binding) === 'COMBO'
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
        values: { ...draft.values, [originalTarget]: result.binding_value },
        parent_artifact_ids: [...new Set([...draft.parent_artifact_ids, result.id])],
      });
    } catch (cause) {
      editor.error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      editor.busy = false;
    }
  }
</script>

<div class="asset-editor">
  <h3>Input image and mask</h3>
  <label
    >Asset type<select bind:value={editor.mode}
      ><option value="input">Input image</option><option value="mask">Paint mask</option></select
    ></label
  >
  <label
    >Actual workflow input<select bind:value={editor.target}
      ><option value="">Choose a binding</option>{#each choices as binding}<option
          value={binding.id}>{fieldLabel(binding)} · {binding.node_id}</option
        >{/each}</select
    ></label
  >
  <label class="button import-label"
    >{editor.source ? 'Replace source image' : 'Choose source image'}<input
      type="file"
      accept="image/png,image/jpeg,image/webp"
      onchange={(event) => {
        const selected = event.currentTarget.files?.[0];
        event.currentTarget.value = '';
        void replace(selected);
      }}
    /></label
  >
  <button onclick={reset}>Clear editor</button>
  <p class="muted">
    Choose a PNG, JPEG or WebP file of at most 2 MiB and 4096 pixels a side. The companion
    accepts any image within these limits; it refuses images above 16 megapixels.
  </p>
  <p>
    {editor.name || 'Blank canvas'} · {editor.width} × {editor.height} · {editor.width ===
    editor.height
      ? 'Square'
      : editor.width > editor.height
        ? 'Landscape'
        : 'Portrait'}
  </p>
  {#if editor.mode === 'mask'}
    <div class="input-row">
      <label
        >Width<input
          type="number"
          min="1"
          max="4096"
          step="1"
          disabled={!!editor.source}
          bind:value={editor.width}
          oninput={dimensionsChanged}
        /></label
      ><label
        >Height<input
          type="number"
          min="1"
          max="4096"
          step="1"
          disabled={!!editor.source}
          bind:value={editor.height}
          oninput={dimensionsChanged}
        /></label
      >
    </div>
    <label
      >Brush diameter {editor.brush}px<input
        type="range"
        min="1"
        max="256"
        bind:value={editor.brush}
      /></label
    >
    <label><input type="checkbox" bind:checked={editor.erase} /> Erase paint</label>
    <button onclick={clear}>Clear mask</button><button
      onclick={() => {
        editor.changed();
        const c = canvas.getContext('2d');
        if (c) {
          c.globalCompositeOperation = 'source-over';
          c.fillStyle = '#87f900';
          c.fillRect(0, 0, canvas.width, canvas.height);
        }
      }}>Mask all</button
    >
    <p class="muted">
      Green marks the masked region. Uploaded PNG uses transparent alpha for masked pixels (Comfy
      LoadImage mask output). Paint stays while you switch tabs or views. {editor.source
        ? 'Mask dimensions match the source image. Clear the editor to create a different canvas.'
        : 'Changing blank-canvas dimensions clears paint.'}
    </p>
  {/if}
  <p class="muted">
    The complete image is centered and scaled to fit for editing. Upload preserves its pixel
    dimensions; this editor does not crop or stretch the source.
  </p>
  <div
    class="canvas-wrap"
    style:aspect-ratio={`${geometry.width}/${geometry.height}`}
    style:max-width={`${geometry.maxStageWidth}px`}
  >
    {#if editor.source}<img src={editor.source} alt="Actual source selected for upload" />{/if}
    {#if editor.mode === 'mask'}<canvas
        bind:this={canvas}
        use:maskCanvas={editor}
        width={geometry.width}
        height={geometry.height}
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
  {#if !geometry.valid}<p class="error" role="alert">
      Enter whole dimensions between 1 and 4096 pixels.
    </p>{/if}
  <button
    class="primary"
    disabled={editor.busy ||
      !geometry.valid ||
      !targetAvailable ||
      (editor.mode === 'input' && !editor.file)}
    onclick={upload}>{editor.busy ? 'Uploading…' : 'Upload and bind asset'}</button
  >
  {#if editor.receipt}<p role="status">
      Submitted asset: {editor.receipt.binding_value} · {editor.receipt.width} × {editor.receipt
        .height} · ID {editor.receipt.id}
    </p>{/if}
  {#if targetAvailable && draft.values[editor.target]}<p class="mono">
      Current bound value: {String(draft.values[editor.target])}
    </p>
    <button
      onclick={() => {
        changeDraft({ values: { ...draft.values, [editor.target]: '' } });
        editor.receipt = null;
      }}>Clear bound input</button
    >{/if}
  {#if editor.error}<p class="error" role="alert">{editor.error}</p>{/if}
</div>

<style>
  .asset-editor {
    display: grid;
    gap: 10px;
  }
  .canvas-wrap {
    position: relative;
    width: 100%;
    margin-inline: auto;
    overflow: hidden;
    background: repeating-conic-gradient(#333 0% 25%, #20242a 0% 50%) 50% / 20px 20px;
  }
  .canvas-wrap img,
  .canvas-wrap canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .canvas-wrap canvas {
    opacity: 0.6;
    touch-action: none;
    cursor: crosshair;
  }
</style>
