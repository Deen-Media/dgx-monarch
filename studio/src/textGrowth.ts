/** A text box that grows with its text, as Create's main prompt does. */

/**
 * The height that shows all of a text box's text: its scroll height plus its border and any
 * scroll bar, never below its height when empty and never above its CSS ceiling.
 */
export function fittedHeight(scrollHeight: number, frame: number, floor: number, ceiling: number) {
  return Math.min(Math.max(scrollHeight + frame, floor), ceiling);
}

/**
 * Svelte action: while `grow` holds, fit a text area to its text on each keystroke and each
 * change of its value, such as Undo. Its rows set the floor and its max-height the ceiling, past
 * which it scrolls. Without `grow` it leaves the box alone, so a box resized by hand keeps its size.
 */
export function growText(node: HTMLTextAreaElement, params: { value: unknown; grow: boolean }) {
  let grow = params.grow;
  const fit = () => {
    if (!grow) return;
    node.style.height = 'auto';
    const ceiling = parseFloat(getComputedStyle(node).maxHeight);
    const frame = node.offsetHeight - node.clientHeight;
    const height = fittedHeight(
      node.scrollHeight,
      frame,
      node.offsetHeight,
      Number.isFinite(ceiling) ? ceiling : Infinity,
    );
    node.style.height = `${height}px`;
  };
  fit();
  node.addEventListener('input', fit);
  return {
    update(next: { value: unknown; grow: boolean }) {
      if (grow && !next.grow) node.style.height = '';
      grow = next.grow;
      fit();
    },
    destroy: () => node.removeEventListener('input', fit),
  };
}
