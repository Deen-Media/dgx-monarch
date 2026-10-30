import test from 'node:test';
import assert from 'node:assert/strict';
import {
  NO_THUMBNAIL,
  UNPLAYABLE,
  drawPoster,
  failureOf,
  formatDuration,
  posterNote,
  videoPoster,
  type Decoder,
  type Drawn,
} from './videoPoster.ts';
import { renderComponent, visibleText } from './renderHarness.js';

/** A stand-in for a detached video element: the test fires the events a browser would. */
class FakeVideo extends EventTarget {
  muted = false;
  preload = '';
  src = '';
  duration = NaN;
  videoWidth = 0;
  videoHeight = 0;
  error: { code: number } | null = null;
  loads = 0;
  removeAttribute(name: string) {
    if (name === 'src') this.src = '';
  }
  load() {
    this.loads++;
  }
  fire(name: string) {
    this.dispatchEvent(new Event(name));
  }
}
interface FakeCanvas {
  width: number;
  height: number;
  drawn: unknown[][];
}
function decoder(options: { blob?: Blob | null; draw?: () => void; timeoutMs?: number } = {}) {
  const video = new FakeVideo(),
    revoked: string[] = [];
  const canvas: FakeCanvas & Record<string, unknown> = {
    width: 0,
    height: 0,
    drawn: [],
    getContext: () => ({
      drawImage: (...args: unknown[]) => {
        options.draw?.();
        canvas.drawn.push(args);
      },
    }),
    toBlob: (done: (blob: Blob | null) => void, type: string) =>
      done(options.blob === undefined ? new Blob(['frame'], { type }) : options.blob),
  };
  const pieces: Decoder = {
    video: () => video as unknown as HTMLVideoElement,
    canvas: () => canvas as unknown as HTMLCanvasElement,
    createURL: () => 'blob:clip',
    revokeURL: (url) => revoked.push(url),
    timeoutMs: options.timeoutMs ?? 60_000,
  };
  return { video, canvas, revoked, pieces };
}
const clip = new Blob(['not really a video'], { type: 'video/mp4' });

test('the length reads as m:ss, h:mm:ss from an hour, and a short clip as 0:01', () => {
  assert.equal(formatDuration(2.04), '0:02');
  assert.equal(formatDuration(5.5), '0:06');
  assert.equal(formatDuration(0.4), '0:01');
  assert.equal(formatDuration(75), '1:15');
  assert.equal(formatDuration(3725), '1:02:05');
  for (const none of [null, undefined, NaN, Infinity, 0, -3])
    assert.equal(formatDuration(none), '');
});

test('a decodable video gives a poster of its first frame and the length from its metadata', async () => {
  const { video, canvas, revoked, pieces } = decoder();
  const drawn = drawPoster(clip, undefined, pieces);
  assert.equal(video.src, 'blob:clip');
  assert.equal(video.muted, true, 'the element never makes a sound');
  video.duration = 2.04;
  video.fire('loadedmetadata');
  [video.videoWidth, video.videoHeight] = [640, 360];
  video.fire('loadeddata');
  const result = await drawn;
  assert.equal(result.problem, '');
  assert.equal(result.duration, 2.04);
  assert.equal(result.poster?.type, 'image/jpeg');
  assert.equal(result.lasting, true);
  assert.deepEqual([canvas.width, canvas.height], [320, 180], 'drawn at thumbnail size');
  assert.equal(canvas.drawn[0][0], video, 'the frame comes from the video element');
  assert.deepEqual(revoked, ['blob:clip'], 'the object URL goes once the frame is drawn');
  assert.deepEqual([video.src, video.loads], ['', 1], 'the element lets go of the video');
  assert.equal(
    posterNote(result),
    NO_THUMBNAIL + ' · 0:02',
    'the note is used only without a poster',
  );
});

test('only a decode or format failure says the video cannot play', async () => {
  for (const [code, problem] of [
    [3, 'unplayable'],
    [4, 'unplayable'],
    [2, 'missing'],
    [1, 'missing'],
  ] as const) {
    const { video, revoked, pieces } = decoder();
    const drawn = drawPoster(clip, undefined, pieces);
    video.error = { code };
    video.fire('error');
    const result = await drawn;
    assert.equal(result.problem, problem, `MediaError code ${code}`);
    assert.equal(result.poster, null);
    assert.deepEqual(revoked, ['blob:clip']);
  }
  assert.equal(failureOf(null), 'missing');
  assert.equal(posterNote({ poster: null, duration: null, problem: 'unplayable' }), UNPLAYABLE);
});

test('a playable video without a drawable frame says the thumbnail is missing, with its length', async () => {
  const cases: [string, Parameters<typeof decoder>[0], number, number][] = [
    ['an audio-only file', {}, 0, 0],
    ['a canvas that returns no image', { blob: null }, 320, 240],
    [
      'a draw that throws',
      {
        draw: () => {
          throw new Error('tainted');
        },
      },
      320,
      240,
    ],
  ];
  for (const [name, options, width, height] of cases) {
    const { video, pieces } = decoder(options);
    const drawn = drawPoster(clip, undefined, pieces);
    video.duration = 4;
    video.fire('loadedmetadata');
    [video.videoWidth, video.videoHeight] = [width, height];
    video.fire('loadeddata');
    const result = await drawn;
    assert.equal(result.problem, 'missing', name);
    assert.equal(result.poster, null, name);
    assert.equal(posterNote(result), `${NO_THUMBNAIL} · 0:04`, name);
    assert.notEqual(posterNote(result), UNPLAYABLE, name);
  }
});

test('a frame that never loads, or a tile that goes away, leaves nothing to keep', async () => {
  const slow = decoder({ timeoutMs: 5 });
  const late = await drawPoster(clip, undefined, slow.pieces);
  assert.deepEqual(
    { problem: late.problem, lasting: late.lasting, revoked: slow.revoked },
    { problem: 'missing', lasting: false, revoked: ['blob:clip'] },
  );
  const gone = decoder(),
    controller = new AbortController();
  const drawn = drawPoster(clip, controller.signal, gone.pieces);
  controller.abort();
  assert.equal((await drawn).lasting, false);
  gone.video.fire('error');
  assert.deepEqual(gone.revoked, ['blob:clip'], 'a late event changes nothing');
});

/** A companion stand-in whose downloads the test answers one at a time. */
function companion() {
  const asked: string[] = [],
    answers: ((blob: Blob | Error) => void)[] = [];
  return {
    asked,
    answer: (index: number, value: Blob | Error) => answers[index](value),
    media: (id: string) =>
      new Promise<Blob>((resolve, reject) => {
        asked.push(id);
        answers.push((value) => (value instanceof Error ? reject(value) : resolve(value)));
      }),
  };
}
const drawn = (problem: Drawn['problem'] = ''): Drawn => ({
  poster: problem ? null : new Blob(['frame'], { type: 'image/jpeg' }),
  duration: 3,
  problem,
  lasting: true,
});
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test('a drawn poster is kept, and a failed download is tried again next time', async () => {
  const client = companion(),
    signal = new AbortController().signal;
  const first = videoPoster(client, 'kept-1', undefined, signal, async () => drawn());
  await settle();
  client.answer(0, clip);
  assert.equal((await first).problem, '');
  assert.equal((await videoPoster(client, 'kept-1', undefined, signal)).problem, '');
  assert.deepEqual(client.asked, ['kept-1'], 'the kept poster needs no second download');
  const failed = videoPoster(client, 'kept-2', undefined, signal, async () => drawn());
  await settle();
  client.answer(1, new Error('HTTP 502'));
  assert.deepEqual(await failed, { poster: null, duration: null, problem: 'missing' });
  const retried = videoPoster(client, 'kept-2', undefined, signal, async () => drawn());
  await settle();
  client.answer(2, clip);
  assert.equal((await retried).problem, '');
  assert.deepEqual(client.asked, ['kept-1', 'kept-2', 'kept-2']);
});

test('two videos download at a time, and a tile that goes away leaves the queue', async () => {
  const client = companion(),
    signal = new AbortController().signal,
    leaving = new AbortController();
  const posters = ['queue-1', 'queue-2', 'queue-3', 'queue-4'].map((id) =>
    videoPoster(client, id, undefined, id === 'queue-3' ? leaving.signal : signal, async () =>
      drawn(),
    ),
  );
  const left = posters[2].then(
    () => 'drawn',
    () => 'left',
  );
  await settle();
  assert.deepEqual(client.asked, ['queue-1', 'queue-2']);
  leaving.abort();
  assert.equal(await left, 'left');
  client.answer(0, clip);
  await settle();
  assert.deepEqual(
    client.asked,
    ['queue-1', 'queue-2', 'queue-4'],
    'the next waiting video starts',
  );
  client.answer(1, clip);
  client.answer(2, clip);
  for (const index of [0, 1, 3]) assert.equal((await posters[index]).problem, '');
});

test('a video tile starts with the video mark until its poster is drawn', async () => {
  const body = await renderComponent('./components/Thumbnail.svelte', {
    artifact: { id: 'v', job_id: 'j', kind: 'video', filename: 'clip.mp4', created_at: 1 },
    client: {},
  });
  assert.equal(visibleText(body), '▷ Loading preview');
  const image = await renderComponent('./components/Thumbnail.svelte', {
    artifact: { id: 'i', job_id: 'j', kind: 'image', filename: 'still.png', created_at: 1 },
    client: {},
  });
  assert.equal(visibleText(image), '▧ Loading preview');
});
