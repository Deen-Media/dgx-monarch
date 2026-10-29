import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import type { Artifact } from './contracts.ts';
import {
  NO_MEDIA,
  currentMedia,
  decodeFailed,
  failLoad,
  finishLoad,
  startLoad,
  type ViewerMedia,
} from './viewerMedia.ts';
import { componentFunctions } from './renderHarness.js';

const output = (id: string, kind: 'image' | 'video') =>
  ({ id, job_id: 'j', kind, filename: `${id}.${kind === 'video' ? 'mp4' : 'png'}` }) as Artifact;
const DECODE = /^Selected media cannot be decoded by this browser\./;

// Viewer.svelte's handler, run as shipped: an element reports the URL it was given.
const decodeCode = await componentFunctions('./components/Viewer.svelte', ['decode']);

/**
 * The viewer's media state driven the way Viewer.svelte drives it: a new selection begins a
 * load, the fetch answers with an object URL, and an element that fails reports its `src`.
 */
function viewer() {
  const state = { media: NO_MEDIA as ViewerMedia };
  const handler = new Function(
    'decodeFailed',
    'state',
    `'use strict'; let media; ${decodeCode}
    return (element, comparing) => { media = state.media; decode(element, comparing); state.media = media; };`,
  )(decodeFailed, state);
  const key = (artifact: Artifact) => JSON.stringify([artifact.id, null, null, null]);
  let made = 0;
  return {
    get media() {
      return state.media;
    },
    /** Select an output; the returned function answers its fetch and gives the object URL. */
    select(artifact: Artifact) {
      state.media = startLoad(state.media, key(artifact), artifact, null);
      const ticket = state.media.ticket;
      return () => {
        const url = `blob:studio/${artifact.id}-${++made}`;
        state.media = finishLoad(state.media, ticket, url);
        return url;
      };
    },
    /** An element given `src` failed to decode it. */
    failed(src: string, comparing = false) {
      handler({ getAttribute: (name: string) => (name === 'src' ? src : null) }, comparing);
    },
    shown(artifact: Artifact) {
      const { loading, error, url, kind } = currentMedia(state.media, key(artifact));
      return { loading, error, url, kind };
    },
  };
}

test('video, image, other image, back, video, image: every image shows (issue #107)', () => {
  const view = viewer();
  const clip = output('clip', 'video'),
    still = output('still', 'image'),
    other = output('other', 'image');
  const clipUrl = view.select(clip)();
  assert.deepEqual(view.shown(clip), { loading: false, error: '', url: clipUrl, kind: 'video' });
  // The moment the selection changes, before its load begins, the frame shows loading. It never
  // pairs the image's kind with the video's URL, which made an <img> try to decode the MP4.
  assert.deepEqual(view.shown(still), { loading: true, error: '', url: '', kind: '' });
  const stillLoaded = view.select(still);
  // The removed element that held the video still fires its error event, while the image loads
  // and again after it has loaded; neither belongs to the output on screen.
  view.failed(clipUrl);
  const stillUrl = stillLoaded();
  view.failed(clipUrl);
  assert.deepEqual(view.shown(still), { loading: false, error: '', url: stillUrl, kind: 'image' });
  const otherUrl = view.select(other)();
  view.failed(stillUrl);
  assert.deepEqual(view.shown(other), { loading: false, error: '', url: otherUrl, kind: 'image' });
  const backUrl = view.select(still)();
  assert.deepEqual(view.shown(still), { loading: false, error: '', url: backUrl, kind: 'image' });
  const againUrl = view.select(clip)();
  view.failed(backUrl);
  assert.deepEqual(view.shown(clip), { loading: false, error: '', url: againUrl, kind: 'video' });
  const lastLoaded = view.select(still);
  view.failed(againUrl);
  const lastUrl = lastLoaded();
  view.failed(againUrl);
  assert.deepEqual(view.shown(still), { loading: false, error: '', url: lastUrl, kind: 'image' });
});

test('a decode failure of the media on screen still says so', () => {
  const view = viewer();
  const clip = output('clip', 'video');
  const url = view.select(clip)();
  view.failed(url);
  assert.match(view.shown(clip).error, DECODE);
  // The next selection starts without the earlier failure.
  const still = output('still', 'image');
  const stillUrl = view.select(still)();
  assert.deepEqual(view.shown(still), { loading: false, error: '', url: stillUrl, kind: 'image' });
});

test('the comparison side reports its own failures and ignores a stale one', () => {
  const left = output('left', 'image'),
    right = output('right', 'video');
  let media = startLoad(NO_MEDIA, 'pair', left, right);
  assert.equal(media.kind, 'image');
  assert.equal(media.compareKind, 'video');
  media = finishLoad(media, media.ticket, 'blob:left', 'blob:right');
  assert.equal(
    decodeFailed(media, 'blob:old-right', true),
    media,
    'an earlier element changes nothing',
  );
  assert.equal(decodeFailed(media, 'blob:right', false), media, 'the sides do not mix');
  assert.match(
    decodeFailed(media, 'blob:right', true).error,
    /^Comparison media cannot be decoded/,
  );
  assert.equal(decodeFailed(NO_MEDIA, '', false), NO_MEDIA, 'no URL on screen, nothing to fail');
});

test('an answer or a failure for an earlier selection changes nothing', () => {
  const first = startLoad(NO_MEDIA, 'first', output('a', 'image'), null);
  const second = startLoad(first, 'second', output('b', 'video'), null);
  assert.equal(finishLoad(second, first.ticket, 'blob:a'), second);
  assert.equal(failLoad(second, first.ticket, 'Network down'), second);
  const failed = failLoad(second, second.ticket, 'Media exceeds the 64 MiB viewer limit.');
  assert.deepEqual(
    [failed.loading, failed.error],
    [false, 'Media exceeds the 64 MiB viewer limit.'],
  );
  // With no output selected nothing loads, so the frame shows no loading state.
  assert.equal(startLoad(second, 'none', null, null).loading, false);
});

test('the load effect never subscribes to the media state it writes', async () => {
  // A read of `media` in the effect's synchronous part would rerun it after every finished load,
  // so the viewer would revoke and fetch the same output forever and stay on Loading.
  const source = await readFile(new URL('./components/Viewer.svelte', import.meta.url), 'utf8');
  const start = source.indexOf('const key = mediaKey;');
  const body = source.slice(start, source.indexOf('Promise.all(', start));
  assert.ok(start > 0 && body.includes('startLoad('), 'the effect that begins each load');
  const reads = body
    .replace(/untrack\(\(\) => media\)/g, '')
    .replace(/\/\/.*$/gm, '')
    .match(/\bmedia\b(?!\s*=[^=])/g);
  assert.equal(reads, null, 'only untrack() reads the media state there');
});

test('the viewer picks each element from the loaded media and passes the URL that failed', async () => {
  const source = await readFile(new URL('./components/Viewer.svelte', import.meta.url), 'utf8');
  assert.match(source, /\{#if shown\.kind === 'video'\}<video/);
  assert.match(source, /\{#if shown\.compareKind === 'video'\}<video/);
  assert.doesNotMatch(source, /\{#if (artifact|compare)\??\.kind === 'video'\}<video/);
  assert.equal(
    (source.match(/onerror=\{\(e\) => decode\(e\.currentTarget(, true)?\)\}/g) ?? []).length,
    4,
  );
  assert.match(source, /const shown = \$derived\(currentMedia\(media, mediaKey\)\);/);
});
