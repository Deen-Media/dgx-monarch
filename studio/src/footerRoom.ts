/**
 * Svelte action for Create's pinned footer. It keeps `--footer-room` on the pane at the footer's
 * height, which the pane's scroll padding uses, so a field brought into view stops above the
 * footer. A browser's own scroll on focus can ignore that padding, so once a focus has settled, a
 * control left under the footer is brought into view the same way. Anything else stays put.
 */
export function footerRoom(node: HTMLElement) {
  const pane = node.parentElement;
  if (!pane || typeof ResizeObserver === 'undefined') return {};
  const measure = () => pane.style.setProperty('--footer-room', `${node.offsetHeight}px`);
  const clear = (event: Event) => {
    const target = event.target as HTMLElement | null;
    if (!target || node.contains(target) || typeof target.scrollIntoView !== 'function') return;
    requestAnimationFrame(() => {
      const footer = node.getBoundingClientRect();
      const control = target.getBoundingClientRect();
      if (control.bottom > footer.top && control.top < footer.bottom)
        target.scrollIntoView({ block: 'nearest' });
    });
  };
  const observer = new ResizeObserver(measure);
  observer.observe(node);
  pane.addEventListener('focusin', clear);
  measure();
  return {
    destroy() {
      observer.disconnect();
      pane.removeEventListener('focusin', clear);
      pane.style.removeProperty('--footer-room');
    },
  };
}
