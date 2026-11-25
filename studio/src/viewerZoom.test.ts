import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_ZOOM,
  MIN_ZOOM,
  anchorAt,
  contentBox,
  scrollToAnchor,
  shownScale,
  wheelZoom,
  type Box,
} from './viewerZoom.ts';
import { componentFunctions } from './renderHarness.js';

const near = (actual: number, expected: number, message?: string) =>
  assert.ok(Math.abs(actual - expected) < 1e-9, message ?? `${actual} is not ${expected}`);

test('the wheel zooms in going up and out going down, by the distance scrolled', () => {
  near(wheelZoom(1, 0, -100)!, 2 ** 0.25);
  near(wheelZoom(1, 0, 100)!, 2 ** -0.25);
  near(wheelZoom(2, 0, -4)!, 2 * 2 ** 0.01, 'a trackpad step zooms a little');
  // Firefox reports a mouse wheel in lines and some wheels in pages.
  near(wheelZoom(1, 0, -3, 1)!, 2 ** 0.3);
  near(wheelZoom(1, 0, 1, 2)!, 0.5, 'one page halves the zoom, no further');
  near(wheelZoom(1, 0, -5000)!, 2, 'no single event zooms past a doubling');
});

test('a sideways scroll is no zoom, so the viewer keeps scrolling', () => {
  assert.equal(wheelZoom(1, 100, 0), null);
  assert.equal(wheelZoom(1, -120, 40), null, 'Shift and the wheel, or a trackpad swipe');
  assert.equal(wheelZoom(1, 0, 0), null);
});

test('the zoom stays between the button limits and never moves against the wheel', () => {
  assert.equal(wheelZoom(MAX_ZOOM, 0, -100), MAX_ZOOM);
  assert.equal(wheelZoom(MIN_ZOOM, 0, 100), MIN_ZOOM);
  near(wheelZoom(7.9, 0, -100)!, MAX_ZOOM);
  // A huge image in Fit shows below the smallest zoom: down keeps it, up goes to the limit.
  assert.equal(wheelZoom(0.04, 0, 100), 0.04);
  assert.equal(wheelZoom(0.04, 0, -100), MIN_ZOOM);
});

test('Fit shows an image at the smaller of its two ratios, centered in its box', () => {
  const natural = { width: 200, height: 100 };
  assert.equal(shownScale({ width: 100, height: 100 }, natural), 0.5);
  assert.equal(shownScale({ width: 400, height: 100 }, natural), 1);
  assert.deepEqual(contentBox({ left: 0, top: 0, width: 100, height: 100 }, natural), {
    left: 0,
    top: 25,
    width: 100,
    height: 50,
  });
  assert.equal(shownScale({ width: 10, height: 10 }, { width: 0, height: 0 }), 1);
});

/**
 * A viewport at (40, 30) on screen, 300 by 200 pixels, scrolled to `scroll`, holding an image of
 * `natural` size at `zoom`, laid out as the viewer lays it out: centered along an axis where it is
 * smaller than the viewport, and at the scroll origin where it is larger.
 */
function layout(
  natural: { width: number; height: number },
  zoom: number,
  scroll: { x: number; y: number },
): Box {
  const width = natural.width * zoom,
    height = natural.height * zoom;
  return {
    left: 40 + Math.max(0, (300 - width) / 2) - scroll.x,
    top: 30 + Math.max(0, (200 - height) / 2) - scroll.y,
    width,
    height,
  };
}
/** The browser keeps a scroll position within the content. */
const clampScroll = (value: number, content: number, view: number) =>
  Math.min(Math.max(0, content - view), Math.max(0, value));

test('the point under the pointer stays put when the zoomed image fills the viewport', () => {
  const natural = { width: 640, height: 480 };
  const pointer = { x: 40 + 210, y: 30 + 70 };
  let zoom = 1,
    scroll = { x: 120, y: 90 };
  for (const deltaY of [-100, -100, -240, 100, 60, -30]) {
    const before = layout(natural, zoom, scroll);
    const anchor = anchorAt(before, natural, pointer);
    zoom = wheelZoom(zoom, 0, deltaY)!;
    const shift = scrollToAnchor(anchor, layout(natural, zoom, scroll), natural, pointer);
    scroll = {
      x: clampScroll(scroll.x + shift.x, natural.width * zoom, 300),
      y: clampScroll(scroll.y + shift.y, natural.height * zoom, 200),
    };
    const after = anchorAt(layout(natural, zoom, scroll), natural, pointer);
    near(after.x, anchor.x, `x after ${deltaY}`);
    near(after.y, anchor.y, `y after ${deltaY}`);
  }
});

test('an image smaller than the viewport stays centered while it grows', () => {
  const natural = { width: 100, height: 50 };
  const pointer = { x: 40 + 150 + 20, y: 30 + 100 };
  const before = layout(natural, 1, { x: 0, y: 0 });
  const anchor = anchorAt(before, natural, pointer);
  const shift = scrollToAnchor(anchor, layout(natural, 1.5, { x: 0, y: 0 }), natural, pointer);
  // The wanted shift is positive, but the content is no wider than the viewport: no scroll range.
  assert.ok(shift.x > 0);
  assert.equal(clampScroll(shift.x, 150, 300), 0);
});

/** A stand-in for the browser's image element; `instanceof` tells it from other targets. */
class FakeImage {
  naturalWidth = 640;
  naturalHeight = 480;
  rect: () => Box = () => ({ left: 0, top: 0, width: 0, height: 0 });
  getBoundingClientRect() {
    return this.rect();
  }
  get natural() {
    return { width: this.naturalWidth, height: this.naturalHeight };
  }
}
const wheelCode = await componentFunctions('./components/Viewer.svelte', ['wheel']);
/**
 * The viewer's own wheel handler, run against the layout above: Fit shows the image at the
 * smaller of its two ratios, centered, and the viewport keeps its scroll within the content.
 */
function viewer(start: { mode: string; zoom: number }) {
  const image = new FakeImage();
  let left = 0,
    top = 0,
    current = () => start;
  const scale = () =>
    current().mode === 'fit'
      ? shownScale({ width: 300, height: 200 }, image.natural)
      : current().zoom;
  const viewport = {
    get scrollLeft() {
      return left;
    },
    set scrollLeft(value: number) {
      left = clampScroll(value, image.naturalWidth * scale(), 300);
    },
    get scrollTop() {
      return top;
    },
    set scrollTop(value: number) {
      top = clampScroll(value, image.naturalHeight * scale(), 200);
    },
  };
  image.rect = () => layout(image.natural, scale(), { x: left, y: top });
  const handler = new Function(
    'state',
    'viewport',
    'HTMLImageElement',
    'flushSync',
    'anchorAt',
    'scrollToAnchor',
    'shownScale',
    'wheelZoom',
    `'use strict'; let { mode, zoom } = state; ${wheelCode}
     return { wheel, state: () => ({ mode, zoom }) };`,
  )(
    start,
    viewport,
    FakeImage,
    (change: () => void) => change(),
    anchorAt,
    scrollToAnchor,
    shownScale,
    wheelZoom,
  );
  current = handler.state;
  const spin = (target: unknown, pointer: { x: number; y: number }, deltaY: number, deltaX = 0) => {
    let prevented = false;
    handler.wheel({
      target,
      deltaX,
      deltaY,
      deltaMode: 0,
      clientX: pointer.x,
      clientY: pointer.y,
      preventDefault: () => (prevented = true),
    });
    return prevented;
  };
  return { viewport, image, spin, state: handler.state };
}

test('the handler zooms over an image from Fit at its shown scale and scrolls to the anchor', () => {
  const view = viewer({ mode: 'fit', zoom: 1 });
  const fit = shownScale({ width: 300, height: 200 }, view.image.natural);
  let pointer = { x: 200, y: 120 },
    anchor = anchorAt(view.image.rect(), view.image.natural, pointer);
  assert.equal(view.spin(view.image, pointer, -100), true, 'the page does not scroll');
  assert.equal(view.state().mode, 'actual');
  near(view.state().zoom, fit * 2 ** 0.25, 'the first step starts from the Fit scale');
  assert.ok(view.viewport.scrollLeft > 0 && view.viewport.scrollTop > 0, 'the viewport scrolled');
  let after = anchorAt(view.image.rect(), view.image.natural, pointer);
  near(after.x, anchor.x);
  near(after.y, anchor.y);
  // Further steps in and out around another point keep that point under the pointer too.
  pointer = { x: 90, y: 180 };
  for (const deltaY of [-100, -100, 50, -200]) {
    anchor = anchorAt(view.image.rect(), view.image.natural, pointer);
    assert.equal(view.spin(view.image, pointer, deltaY), true);
    after = anchorAt(view.image.rect(), view.image.natural, pointer);
    near(after.x, anchor.x, `x after ${deltaY}`);
    near(after.y, anchor.y, `y after ${deltaY}`);
  }
});

test('the handler leaves the wheel alone off the image, over a video and sideways', () => {
  const view = viewer({ mode: 'actual', zoom: 1 });
  const pointer = { x: 100, y: 100 };
  for (const target of [{ tagName: 'DIV' }, { tagName: 'VIDEO' }, null]) {
    assert.equal(
      view.spin(target, pointer, -100),
      false,
      'the viewer and the page scroll as before',
    );
    assert.deepEqual(view.state(), { mode: 'actual', zoom: 1 });
  }
  assert.equal(view.spin(view.image, pointer, 0, 80), false, 'a sideways scroll pans');
  assert.deepEqual(view.state(), { mode: 'actual', zoom: 1 });
  view.image.naturalWidth = 0;
  assert.equal(view.spin(view.image, pointer, -100), false, 'an image not yet loaded is no target');
});

test('at the zoom limit the wheel over the image still never scrolls the page', () => {
  const view = viewer({ mode: 'actual', zoom: MAX_ZOOM });
  view.viewport.scrollLeft = 500;
  assert.equal(view.spin(view.image, { x: 100, y: 100 }, -100), true);
  assert.deepEqual(view.state(), { mode: 'actual', zoom: MAX_ZOOM });
  assert.deepEqual([view.viewport.scrollLeft, view.viewport.scrollTop], [500, 0]);
});
