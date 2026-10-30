/**
 * Svelte action for Create's pinned footer: keeps `--footer-room` on the pane at the footer's
 * height, so a field brought into view, by a focus or a scroll, stops above the footer instead
 * of under it.
 */
export function footerRoom(node: HTMLElement) {
  const pane = node.parentElement;
  if (!pane || typeof ResizeObserver === 'undefined') return {};
  const measure = () => pane.style.setProperty('--footer-room', `${node.offsetHeight}px`);
  const observer = new ResizeObserver(measure);
  observer.observe(node);
  measure();
  return {
    destroy() {
      observer.disconnect();
      pane.style.removeProperty('--footer-room');
    },
  };
}
