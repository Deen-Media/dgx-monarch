import type { StudioClient } from './api';

/** What a video tile shows in place of the image thumbnail the companion makes for images. */
export interface VideoPoster {
  /** A JPEG of the video's first frame, or null when none could be drawn. */
  poster: Blob | null;
  /** The length in seconds from the video's own metadata, or null when it gives none. */
  duration: number | null;
  /**
   * '' with a poster. 'missing' when no frame could be drawn: the download failed, the frame did
   * not load in time or the video holds no picture. 'unplayable' only when the browser reports
   * that it cannot decode the video.
   */
  problem: '' | 'missing' | 'unplayable';
}

/** The tile's words when no frame could be drawn, and when the browser cannot play the video. */
export const NO_THUMBNAIL = 'No thumbnail';
export const UNPLAYABLE = 'Cannot play in this browser';

/** The longest side of a drawn poster, as for the companion's image thumbnails. */
export const POSTER_SIZE = 320;
/** How long a video may take to show its first frame before the tile gives up on it. */
export const FRAME_TIMEOUT_MS = 15_000;
/** Videos that download and decode at once, so a page of videos does not start them all. */
const AT_ONCE = 2;
/** Posters kept for the page's life, the least recently used dropped first; each is a small JPEG. */
const KEPT = 64;

/** A length as m:ss, or h:mm:ss from an hour; a clip under a second reads 0:01; none reads ''. */
export function formatDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0) return '';
  const total = Math.max(1, Math.round(seconds));
  const hours = Math.floor(total / 3600),
    minutes = Math.floor((total % 3600) / 60);
  const rest = `${String(total % 60).padStart(2, '0')}`;
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${rest}` : `${minutes}:${rest}`;
}

/** MediaError codes: the browser could not decode the video, or cannot play its format. */
const DECODE_ERROR = 3,
  NOT_SUPPORTED = 4;
/** Only the browser's own decode or format verdict means the video cannot play here. */
export function failureOf(error: { code: number } | null | undefined): 'missing' | 'unplayable' {
  return error?.code === DECODE_ERROR || error?.code === NOT_SUPPORTED ? 'unplayable' : 'missing';
}

/** The tile's line without a poster: why it has none, then the length when the video gave one. */
export function posterNote(result: VideoPoster): string {
  const why = result.problem === 'unplayable' ? UNPLAYABLE : NO_THUMBNAIL;
  const length = formatDuration(result.duration);
  return length ? `${why} · ${length}` : why;
}

/** The browser pieces a poster needs; a test hands in stand-ins. */
export interface Decoder {
  video: () => HTMLVideoElement;
  canvas: () => HTMLCanvasElement;
  createURL: (blob: Blob) => string;
  revokeURL: (url: string) => void;
  timeoutMs: number;
}
const browser = (): Decoder => ({
  video: () => document.createElement('video'),
  canvas: () => document.createElement('canvas'),
  createURL: (blob) => URL.createObjectURL(blob),
  revokeURL: (url) => URL.revokeObjectURL(url),
  timeoutMs: FRAME_TIMEOUT_MS,
});

/** A drawn result, and whether a later look would give the same one, so it may be kept. */
export interface Drawn extends VideoPoster {
  lasting: boolean;
}

/** Seconds, or null for a missing, endless or zero length. */
const seconds = (value: number) => (Number.isFinite(value) && value > 0 ? value : null);

/**
 * Decode a downloaded video in a muted element that is never shown, and draw its first frame
 * into a canvas no wider or taller than POSTER_SIZE. The element lets go of the video, and its
 * object URL is revoked, once the frame is drawn, the video fails, the signal aborts or the
 * timeout passes. A timeout or an abort is not lasting: the next look tries again.
 */
export function drawPoster(
  blob: Blob,
  signal?: AbortSignal,
  decoder: Decoder = browser(),
): Promise<Drawn> {
  return new Promise((resolve) => {
    const video = decoder.video(),
      url = decoder.createURL(blob);
    let duration: number | null = null,
      done = false;
    const finish = (poster: Blob | null, problem: VideoPoster['problem'], lasting = true) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      video.removeAttribute('src');
      video.load();
      decoder.revokeURL(url);
      resolve({ poster, duration, problem, lasting });
    };
    const abort = () => finish(null, 'missing', false);
    const timer = setTimeout(abort, decoder.timeoutMs);
    if (signal?.aborted) return abort();
    signal?.addEventListener('abort', abort);
    video.muted = true;
    video.preload = 'auto';
    video.addEventListener('loadedmetadata', () => (duration = seconds(video.duration)));
    video.addEventListener('error', () => finish(null, failureOf(video.error)));
    video.addEventListener('loadeddata', () => {
      duration ??= seconds(video.duration);
      const { videoWidth: width, videoHeight: height } = video;
      // An audio-only file plays, but it has no picture to draw.
      if (!width || !height) return finish(null, 'missing');
      const scale = Math.min(1, POSTER_SIZE / Math.max(width, height));
      const canvas = decoder.canvas();
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      try {
        const context = canvas.getContext('2d');
        if (!context) return finish(null, 'missing');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((poster) => finish(poster, poster ? '' : 'missing'), 'image/jpeg', 0.8);
      } catch {
        finish(null, 'missing');
      }
    });
    video.src = url;
  });
}

const kept = new Map<string, VideoPoster>();
let running = 0;
const waiting: (() => void)[] = [];

/** Wait for one of the AT_ONCE places; the function it gives back frees the place. */
function place(signal: AbortSignal): Promise<() => void> {
  return new Promise((resolve, reject) => {
    const take = () => {
      running++;
      let freed = false;
      resolve(() => {
        if (freed) return;
        freed = true;
        running--;
        waiting.shift()?.();
      });
    };
    if (running < AT_ONCE) return take();
    const leave = () => {
      const index = waiting.indexOf(turn);
      if (index >= 0) waiting.splice(index, 1);
      reject(signal.reason);
    };
    const turn = () => {
      signal.removeEventListener('abort', leave);
      take();
    };
    waiting.push(turn);
    signal.addEventListener('abort', leave, { once: true });
  });
}

/**
 * The poster of one video output: kept from an earlier look, or downloaded through the
 * companion and drawn now. A failed download leaves the thumbnail missing and is tried again
 * next time. It rejects only when `signal` aborts while the video waits for its turn.
 */
export async function videoPoster(
  client: Pick<StudioClient, 'media'>,
  id: string,
  digest: unknown,
  signal: AbortSignal,
  draw: typeof drawPoster = drawPoster,
): Promise<VideoPoster> {
  const key = JSON.stringify([id, digest ?? null]);
  const known = kept.get(key);
  if (known) {
    kept.delete(key);
    kept.set(key, known);
    return known;
  }
  const free = await place(signal);
  try {
    let blob: Blob;
    try {
      blob = await client.media(id, signal);
    } catch {
      return { poster: null, duration: null, problem: 'missing' };
    }
    const { lasting, ...result } = await draw(blob, signal);
    if (lasting) {
      kept.set(key, result);
      if (kept.size > KEPT) kept.delete(kept.keys().next().value!);
    }
    return result;
  } finally {
    free();
  }
}
