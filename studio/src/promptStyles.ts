/** The ordered prompt styles a draft applies, as Create lists and reorders them. */

/** The list with one style moved `offset` places, or null when the move would leave the list. */
export function moveStyle<T>(styles: readonly T[], index: number, offset: number): T[] | null {
  const target = index + offset;
  const inside = (position: number) =>
    Number.isInteger(position) && position >= 0 && position < styles.length;
  if (offset === 0 || !inside(index) || !inside(target)) return null;
  const moved = [...styles];
  const [style] = moved.splice(index, 1);
  moved.splice(target, 0, style);
  return moved;
}

/** The saved name of an applied style, or its place in the list when it has none. */
export function styleName(style: unknown, index: number): string {
  const name = (style as { name?: unknown } | null)?.name;
  return typeof name === 'string' && name ? name : `Style ${index + 1}`;
}

/** Why a style's Move up or Move down is off, or '' when both work. */
export function styleEndNote(index: number, count: number): string {
  if (count === 1) return 'Only style';
  if (index === 0) return 'Already first';
  return index === count - 1 ? 'Already last' : '';
}
