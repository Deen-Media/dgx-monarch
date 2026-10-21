/** What the text in a number box means right now. */
export type NumberText =
  { kind: 'number'; value: number } | { kind: 'empty' } | { kind: 'partial' };

const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/**
 * Read a number box. A browser reports "" for both a cleared box and a partial
 * entry such as "-" or "1e"; only `badInput` tells them apart, so a partial
 * entry is never read as empty and never as 0.
 */
export function readNumberText(text: string, badInput = false): NumberText {
  if (badInput) return { kind: 'partial' };
  const trimmed = text.trim();
  if (trimmed === '') return { kind: 'empty' };
  if (!NUMBER.test(trimmed)) return { kind: 'partial' };
  const value = Number(trimmed);
  return Number.isFinite(value) ? { kind: 'number', value } : { kind: 'partial' };
}

/** Show the stored value, or nothing when no finite number is stored; never a default. */
export function displayNumber(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : '';
}

export interface NumberFieldOptions {
  /** The stored value the box shows whenever it does not have focus. */
  value: unknown;
  /** Receives each valid number, or null when a nullable box is cleared. */
  commit: (value: number | null) => void;
  /** Commit null for a cleared box; otherwise a cleared box shows the stored value again on blur. */
  nullable?: boolean;
  /** Commit only whole numbers. */
  integer?: boolean;
  min?: number;
  max?: number;
  /** Commit on each keystroke (default) or only when the box reports a change. */
  event?: 'input' | 'change';
}

/** Decide what a box commits for its current text, or undefined when it commits nothing. */
export function committedValue(
  text: NumberText,
  options: Pick<NumberFieldOptions, 'value' | 'nullable' | 'integer' | 'min' | 'max'>,
): number | null | undefined {
  if (text.kind === 'empty') return options.nullable && options.value !== null ? null : undefined;
  if (text.kind !== 'number') return undefined;
  const { value } = text;
  if (options.integer && !Number.isSafeInteger(value)) return undefined;
  if (options.min !== undefined && value < options.min) return undefined;
  if (options.max !== undefined && value > options.max) return undefined;
  return Object.is(value, options.value) ? undefined : value;
}

interface NumberInputElement {
  value: string;
  validity?: { badInput?: boolean };
  ownerDocument?: { activeElement?: unknown } | null;
  addEventListener(type: string, listener: () => void): void;
  removeEventListener(type: string, listener: () => void): void;
}

/**
 * Svelte action for a number box. While the box has focus it keeps the typed
 * text, so the stored value never rewrites a partial entry such as "-0." and
 * the sign or decimal point is never lost; it shows the stored value again
 * when focus leaves.
 */
export function numberField(node: NumberInputElement, options: NumberFieldOptions) {
  let current = options;
  let focused = node.ownerDocument?.activeElement === node;
  const show = () => {
    if (!focused) node.value = displayNumber(current.value);
  };
  const read = () => {
    const next = committedValue(readNumberText(node.value, node.validity?.badInput), current);
    if (next !== undefined) current.commit(next);
  };
  const focus = () => {
    focused = true;
  };
  const blur = () => {
    focused = false;
    show();
  };
  const event = options.event ?? 'input';
  node.addEventListener(event, read);
  node.addEventListener('focus', focus);
  node.addEventListener('blur', blur);
  show();
  return {
    update(next: NumberFieldOptions) {
      current = next;
      show();
    },
    destroy() {
      node.removeEventListener(event, read);
      node.removeEventListener('focus', focus);
      node.removeEventListener('blur', blur);
    },
  };
}
