/** The zoom range the viewer's buttons and the wheel share: a tenth to eight times actual size. */
export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 8;

/** A wheel scroll of this many pixels doubles or halves the zoom; one notch of 100 is about 19%. */
const PIXELS_PER_DOUBLING = 400;
/** Pixels per line and per page, for wheels that report lines or pages (deltaMode 1 and 2). */
const LINE_PIXELS = 40;
const PAGE_PIXELS = 800;

export interface Point {
  x: number;
  y: number;
}
export interface Size {
  width: number;
  height: number;
}
/** A box on screen, as getBoundingClientRect reports it. */
export interface Box extends Size {
  left: number;
  top: number;
}

export const clampZoom = (zoom: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));

/**
 * The zoom after one wheel event, or null when the event is no zoom: a sideways scroll, such as
 * Shift and the wheel or a trackpad swipe, keeps scrolling the viewer. Up zooms in and down zooms
 * out, by a factor that follows the distance scrolled, at most a doubling or a halving per event.
 * The result stays between MIN_ZOOM and MAX_ZOOM and never moves against the wheel, so a step
 * at a limit, or down from a Fit view already below MIN_ZOOM, returns `zoom` unchanged.
 */
export function wheelZoom(zoom: number, deltaX: number, deltaY: number, deltaMode = 0) {
  if (!deltaY || Math.abs(deltaX) > Math.abs(deltaY)) return null;
  const unit = deltaMode === 1 ? LINE_PIXELS : deltaMode === 2 ? PAGE_PIXELS : 1;
  const pixels = Math.max(-PIXELS_PER_DOUBLING, Math.min(PIXELS_PER_DOUBLING, deltaY * unit));
  const next = clampZoom(zoom * 2 ** (-pixels / PIXELS_PER_DOUBLING));
  return pixels > 0 ? Math.min(zoom, next) : Math.max(zoom, next);
}

/**
 * The scale an image shows at inside its element box. The viewer's media use object-fit:
 * contain, so it is the smaller of the two ratios; at a zoom the box takes the image's own shape
 * and both ratios equal the zoom.
 */
export function shownScale(box: Size, natural: Size): number {
  if (!natural.width || !natural.height) return 1;
  return Math.min(box.width / natural.width, box.height / natural.height);
}

/** Where the image's pixels sit inside its element box: centered, at the shown scale. */
export function contentBox(box: Box, natural: Size): Box {
  const scale = shownScale(box, natural);
  const width = natural.width * scale,
    height = natural.height * scale;
  return {
    left: box.left + (box.width - width) / 2,
    top: box.top + (box.height - height) / 2,
    width,
    height,
  };
}

/** The point of the image under the pointer, as fractions of the image's width and height. */
export function anchorAt(box: Box, natural: Size, pointer: Point): Point {
  const shown = contentBox(box, natural);
  return {
    x: shown.width ? (pointer.x - shown.left) / shown.width : 0.5,
    y: shown.height ? (pointer.y - shown.top) / shown.height : 0.5,
  };
}

/**
 * How far to scroll so the anchored point of the image, laid out again in `box` after the zoom,
 * comes back under the pointer. Add it to the viewport's scrollLeft and scrollTop. The browser
 * stops scrolling at the ends of the range, so along an axis where the image is smaller than the
 * viewport it stays where the layout centers it.
 */
export function scrollToAnchor(anchor: Point, box: Box, natural: Size, pointer: Point): Point {
  const shown = contentBox(box, natural);
  return {
    x: shown.left + anchor.x * shown.width - pointer.x,
    y: shown.top + anchor.y * shown.height - pointer.y,
  };
}
