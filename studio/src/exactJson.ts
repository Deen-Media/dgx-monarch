/**
 * JSON that keeps whole numbers above 2**53 - 1 exact. JSON.parse reads every number as a double,
 * which holds whole numbers exactly only up to Number.MAX_SAFE_INTEGER, while ComfyUI seeds reach
 * 2**64 - 1. The companion writes each record with Python's json module, so the text of a record
 * holds every digit; parsing it here keeps them, as a BigInt.
 */

/** A whole number above 2**53 - 1 read as a double, which has already lost digits. */
export function roundedNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && !Number.isSafeInteger(value);
}

/**
 * Parse JSON text as JSON.parse does, except that a whole number beyond 2**53 - 1 reads as a
 * BigInt: an integer literal with every digit it has, and a literal such as 1e21 with the exact
 * value of its double. A Number this large therefore only comes from a parse that rounded it.
 * Text JSON.parse refuses is refused with the same error.
 */
export function parseExactJSON(text: string): unknown {
  JSON.parse(text);
  // Valid JSON is whitespace around strings, punctuation and bare literals.
  const token = /\s*("[^"\\]*(?:\\.[^"\\]*)*"|[{}[\],:]|[^\s{}[\],:"]+)/y;
  const next = () => (token.exec(text) as RegExpExecArray)[1];
  const read = (head: string): unknown => {
    if (head === '{') {
      const object: Record<string, unknown> = {};
      for (let key = next(); key !== '}'; key = next()) {
        if (key === ',') key = next();
        next();
        // A data property, as JSON.parse makes, so a "__proto__" key stays an ordinary key.
        Object.defineProperty(object, JSON.parse(key), {
          value: read(next()),
          enumerable: true,
          writable: true,
          configurable: true,
        });
      }
      return object;
    }
    if (head === '[') {
      const list: unknown[] = [];
      for (let item = next(); item !== ']'; item = next())
        list.push(read(item === ',' ? next() : item));
      return list;
    }
    const value = JSON.parse(head);
    if (!roundedNumber(value)) return value;
    return BigInt(/^-?\d+$/.test(head) ? head : value);
  };
  return read(next());
}

/**
 * Read a record's text exactly, or as JSON.parse does when the exact walk cannot, such as for
 * nesting deeper than the browser's call stack allows. The fallback keeps the action working as it
 * did before exact reads; a number it rounded is still never written or loaded, since
 * writeExactJSON and Reuse refuse it by name. Text JSON.parse refuses still throws its error.
 */
export function readExactJSON(text: string): unknown {
  try {
    return parseExactJSON(text);
  } catch {
    return JSON.parse(text);
  }
}

/** The refusal for a rounded number at `path`, so no copy or export ever holds one. */
function roundedError(path: string): Error {
  return new Error(
    `${path} is a whole number above ${Number.MAX_SAFE_INTEGER} that this browser holds rounded, so Studio wrote nothing.`,
  );
}

/**
 * The text JSON.stringify(value, null, 2) writes, with each BigInt written as its digits. A
 * rounded number stops the write, so the text never holds a value the record does not. Nesting
 * deeper than this walk's call stack writes with JSON.stringify itself, which reaches further;
 * it still refuses a rounded number, and it cannot write a BigInt that deep.
 */
export function writeExactJSON(value: unknown): string {
  try {
    return writeNested(value);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
  }
  // A loop, not a replacer, so the check reaches as deep as JSON.stringify does.
  const stack: [unknown, string][] = [[value, '']];
  for (let entry = stack.pop(); entry; entry = stack.pop()) {
    const [item, key] = entry;
    if (roundedNumber(item)) throw roundedError(key);
    if (typeof item === 'bigint')
      throw new Error(
        `${key} holds ${item} too deep in the record to write exactly, so Studio wrote nothing.`,
      );
    if (item !== null && typeof item === 'object')
      for (const [name, child] of Object.entries(item)) stack.push([child, name]);
  }
  return JSON.stringify(value, null, 2);
}

function writeNested(value: unknown): string {
  const path: string[] = [];
  const write = (item: unknown, indent: string): string => {
    if (typeof item === 'bigint') return item.toString();
    if (roundedNumber(item)) throw roundedError(path.join('.'));
    if (item === null || typeof item !== 'object') return JSON.stringify(item);
    const inner = `${indent}  `,
      list = Array.isArray(item),
      parts: string[] = [];
    for (const [key, child] of Object.entries(item)) {
      // As JSON.stringify does: an object leaves these out, and a list writes null.
      if (child === undefined || typeof child === 'function' || typeof child === 'symbol') {
        if (list) parts.push('null');
        continue;
      }
      path.push(key);
      const text = write(child, inner);
      path.pop();
      parts.push(list ? text : `${JSON.stringify(key)}: ${text}`);
    }
    const [open, close] = list ? ['[', ']'] : ['{', '}'];
    return parts.length
      ? `${open}\n${inner}${parts.join(`,\n${inner}`)}\n${indent}${close}`
      : open + close;
  };
  return write(value, '');
}
