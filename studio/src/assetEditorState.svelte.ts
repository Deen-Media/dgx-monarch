/** What the companion returned for an uploaded asset. */
export interface AssetReceipt {
  id: string;
  width: number;
  height: number;
  binding_value: string;
}

interface Pixels {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

interface CanvasLike {
  width: number;
  height: number;
  getContext(kind: '2d'): {
    getImageData(x: number, y: number, width: number, height: number): Pixels;
    putImageData(pixels: Pixels, x: number, y: number): void;
  } | null;
}

/**
 * The asset editor's source image, painted mask, target and mode. It lives
 * outside the editor component, because the Assets tab and the workspace view
 * remount the editor, and a remount must not discard the user's work.
 */
export class AssetEditorState {
  mode = $state<'input' | 'mask'>('input');
  target = $state('');
  width = $state(512);
  height = $state(512);
  brush = $state(32);
  erase = $state(false);
  /** Object URL of the chosen source image, or '' for a blank canvas. */
  source = $state('');
  name = $state('');
  file = $state<File | undefined>();
  receipt = $state<AssetReceipt | null>(null);
  busy = $state(false);
  error = $state('');
  /** Bumps on every edit, so an upload can tell that the editor changed under it. */
  generation = 0;
  private mask: Pixels | null = null;
  private revoke: (url: string) => void;

  constructor(revoke: (url: string) => void = (url) => URL.revokeObjectURL(url)) {
    this.revoke = revoke;
  }

  /** Record an edit: a later upload result no longer describes the editor. */
  changed() {
    this.generation++;
    this.receipt = null;
  }

  /** Save the painted pixels, for example when the mask canvas leaves the page. */
  capture(canvas: CanvasLike) {
    const context = canvas.getContext('2d');
    if (context) this.mask = context.getImageData(0, 0, canvas.width, canvas.height);
  }

  /** Paint saved pixels back onto a remounted canvas of the same size. */
  restore(canvas: CanvasLike) {
    const mask = this.mask;
    if (!mask || mask.width !== canvas.width || mask.height !== canvas.height) return;
    canvas.getContext('2d')?.putImageData(mask, 0, 0);
  }

  /** Forget saved paint, as Clear mask and a dimension change do. */
  clearMask() {
    this.mask = null;
    this.changed();
  }

  /** Use a decoded source image; its dimensions become the canvas dimensions. */
  useSource(url: string, file: File, width: number, height: number) {
    this.reset();
    this.source = url;
    this.file = file;
    this.name = file.name;
    this.width = width;
    this.height = height;
  }

  /** Clear the source, the paint and the receipt; the mode and target stay. */
  reset() {
    if (this.source) this.revoke(this.source);
    this.source = '';
    this.file = undefined;
    this.name = '';
    this.clearMask();
  }
}

/** Svelte action for the mask canvas: restore paint on mount and save it on removal. */
export function maskCanvas(node: CanvasLike, state: AssetEditorState) {
  state.restore(node);
  return {
    destroy() {
      state.capture(node);
    },
  };
}

/** One editor state per browser tab. */
export const assetEditor = new AssetEditorState();
