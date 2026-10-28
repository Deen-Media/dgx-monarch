/** The margin around the visible area within which a preview starts to download. */
export const NEAR_MARGIN = '200px';

/**
 * Call `onNear` once, when `element` comes within NEAR_MARGIN of the visible area, and return a
 * function that stops watching. Without an IntersectionObserver (an old browser, or a test
 * runtime) it calls `onNear` at once, as the thumbnails did before this gate.
 */
export function whenNearViewport(
  element: Element,
  onNear: () => void,
  Observer: typeof IntersectionObserver | undefined = globalThis.IntersectionObserver,
): () => void {
  if (Observer === undefined) {
    onNear();
    return () => {};
  }
  let done = false;
  const observer = new Observer(
    (entries) => {
      if (done || !entries.some((entry) => entry.isIntersecting)) return;
      done = true;
      observer.disconnect();
      onNear();
    },
    { rootMargin: NEAR_MARGIN },
  );
  observer.observe(element);
  return () => {
    done = true;
    observer.disconnect();
  };
}
