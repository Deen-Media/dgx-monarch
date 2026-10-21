import test from 'node:test';
import assert from 'node:assert/strict';
import {
  committedValue,
  displayNumber,
  numberField,
  readNumberText,
  type NumberFieldOptions,
} from './numberField.ts';

/** A stand-in number box: `type` sets the text a browser would show and report. */
function box() {
  const listeners = new Map<string, () => void>();
  const node = {
    value: '',
    validity: { badInput: false },
    ownerDocument: { activeElement: null as unknown },
    addEventListener(type: string, listener: () => void) {
      listeners.set(type, listener);
    },
    removeEventListener(type: string) {
      listeners.delete(type);
    },
  };
  const fire = (type: string) => listeners.get(type)?.();
  return {
    node,
    listeners,
    focus: () => fire('focus'),
    blur: () => fire('blur'),
    type(text: string, valid = true) {
      // A browser reports "" with badInput for a partial number entry.
      node.value = valid ? text : '';
      node.validity.badInput = !valid;
      fire('input');
    },
  };
}

function field(options: Partial<NumberFieldOptions> & { value: unknown }) {
  const committed: (number | null)[] = [];
  const control = box();
  let current: NumberFieldOptions = {
    ...options,
    commit: (value) => {
      committed.push(value);
      current = { ...current, value };
      action.update(current);
    },
  };
  const action = numberField(control.node, current);
  return { ...control, committed, action, options: () => current };
}

test('partial entries are never read as empty or as zero', () => {
  assert.deepEqual(readNumberText('', true), { kind: 'partial' });
  assert.deepEqual(readNumberText(''), { kind: 'empty' });
  for (const text of ['-', '+', '.', '1e', '1e+', '--1', 'abc'])
    assert.deepEqual(readNumberText(text), { kind: 'partial' }, text);
  assert.deepEqual(readNumberText('.5'), { kind: 'number', value: 0.5 });
  assert.deepEqual(readNumberText('-0.5'), { kind: 'number', value: -0.5 });
  assert.deepEqual(readNumberText('1.'), { kind: 'number', value: 1 });
  assert.deepEqual(readNumberText('1e400'), { kind: 'partial' });
});

test('the box shows the stored value and nothing for an empty one, never a default', () => {
  assert.equal(displayNumber(42), '42');
  assert.equal(displayNumber(-0.25), '-0.25');
  for (const value of [null, undefined, '', Number.NaN, Infinity, '20'])
    assert.equal(displayNumber(value), '');
});

test('typing a negative strength over a positive one keeps the sign', () => {
  const control = field({ value: 1 });
  assert.equal(control.node.value, '1');
  control.focus();
  control.type('-', false);
  control.type('-0');
  control.type('-0.', false);
  control.type('-0.5');
  // The stored value never rewrote the typed text while the box had focus.
  assert.equal(control.node.value, '-0.5');
  assert.deepEqual(control.committed, [-0, -0.5]);
  assert.ok(
    !control.committed.some((value) => Object.is(value, 0) || value === 1 || value === 0.5),
  );
  control.blur();
  assert.equal(control.node.value, '-0.5');
});

test('a leading decimal point commits the fraction, not the digits', () => {
  const control = field({ value: 1 });
  control.focus();
  control.type('.', false);
  control.type('.5');
  assert.deepEqual(control.committed, [0.5]);
});

test('clearing a required box commits nothing and shows the stored value again on blur', () => {
  const control = field({ value: 0.8 });
  control.focus();
  control.type('');
  assert.deepEqual(control.committed, []);
  assert.equal(control.node.value, '');
  control.blur();
  assert.equal(control.node.value, '0.8');
});

test('clearing a nullable box stores null and the next digits are not appended to a default', () => {
  const control = field({ value: 42, nullable: true });
  control.focus();
  control.type('');
  control.type('3');
  control.type('30');
  assert.deepEqual(control.committed, [null, 3, 30]);
  assert.equal(control.node.value, '30');
  control.blur();
  assert.equal(control.node.value, '30');
  const cleared = field({ value: 42, nullable: true });
  cleared.focus();
  cleared.type('');
  cleared.blur();
  assert.equal(cleared.node.value, '');
});

test('a stored change made elsewhere shows only while the box has no focus', () => {
  const control = field({ value: 5 });
  control.focus();
  control.type('7');
  control.action.update({ ...control.options(), value: 9 });
  assert.equal(control.node.value, '7');
  control.blur();
  assert.equal(control.node.value, '9');
  control.action.update({ ...control.options(), value: 11 });
  assert.equal(control.node.value, '11');
});

test('integer and range bounds keep invalid counts out of the stored value', () => {
  const options = { value: 1, integer: true, min: 1 };
  assert.equal(committedValue({ kind: 'number', value: 0 }, options), undefined);
  assert.equal(committedValue({ kind: 'number', value: 1.5 }, options), undefined);
  assert.equal(committedValue({ kind: 'number', value: 2 }, options), 2);
  assert.equal(committedValue({ kind: 'number', value: 1 }, options), undefined);
  assert.equal(committedValue({ kind: 'empty' }, options), undefined);
  const control = field({ value: 1, integer: true, min: 1 });
  control.focus();
  control.type('');
  control.type('0');
  control.blur();
  assert.deepEqual(control.committed, []);
  assert.equal(control.node.value, '1');
});

test('a change-event box commits only on change and releases its listeners', () => {
  const control = field({ value: 0, event: 'change' });
  control.focus();
  control.type('4');
  assert.deepEqual(control.committed, []);
  control.node.value = '4';
  control.listeners.get('change')?.();
  assert.deepEqual(control.committed, [4]);
  control.action.destroy();
  assert.equal(control.listeners.size, 0);
});
