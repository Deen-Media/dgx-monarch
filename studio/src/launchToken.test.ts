import test from 'node:test';
import assert from 'node:assert/strict';
import { launchToken, takeLaunchToken, tokenFromInput } from './launchToken.ts';

// Fake tokens of the companion's shape: 43 characters of [A-Za-z0-9_-].
const TOKEN = 'fakeLaunchToken_0123456789-abcdefghijklmnop';
const OTHER = 'otherLaunchToken_123456789-abcdefghijklmnop';

function page(hash: string, search = '') {
  const calls: unknown[][] = [];
  const location = { hash, pathname: '/', search };
  const history = {
    state: { kept: true },
    replaceState: (...args: unknown[]) => {
      calls.push(args);
    },
  };
  return { location, history, calls };
}

test('a sign-in link hands its token over once and leaves the URL without the fragment', () => {
  const { location, history, calls } = page(`#token=${TOKEN}`, '?view=1');
  takeLaunchToken(location, history as unknown as History);
  assert.deepEqual(calls, [[{ kept: true }, '', '/?view=1']]);
  assert.equal(launchToken(), TOKEN);
  assert.equal(launchToken(), '');
});

test('an invalid token fragment is stripped and gives nothing', () => {
  for (const hash of ['#token=short', `#token=${TOKEN}!`, `#token=${TOKEN}&x=1`, '#token=']) {
    const { location, history, calls } = page(hash);
    takeLaunchToken(location, history as unknown as History);
    assert.deepEqual(calls, [[{ kept: true }, '', '/']], hash);
    assert.equal(launchToken(), '', hash);
  }
  // A later invalid link also drops a token an earlier link left untaken.
  takeLaunchToken(page(`#token=${TOKEN}`).location, page('').history as unknown as History);
  const invalid = page('#token=cut-short');
  takeLaunchToken(invalid.location, invalid.history as unknown as History);
  assert.equal(launchToken(), '');
});

test('other fragments and an empty hash are left alone', () => {
  for (const hash of ['', '#workspace', `#tokens=${TOKEN}`, `#x#token=${TOKEN}`]) {
    const { location, history, calls } = page(hash);
    takeLaunchToken(location, history as unknown as History);
    assert.deepEqual(calls, [], hash);
    assert.equal(launchToken(), '', hash);
  }
});

test('a refused history update still hands the token over', () => {
  const location = { hash: `#token=${TOKEN}`, pathname: '/', search: '' };
  const history = {
    state: null,
    replaceState: () => {
      throw new Error('SecurityError');
    },
  };
  takeLaunchToken(location, history as unknown as History);
  assert.equal(launchToken(), TOKEN);
});

test('taking a launch token never touches storage or the console', () => {
  const touched: string[] = [];
  const trap = (name: string) =>
    new Proxy(
      {},
      {
        get(_target, key) {
          touched.push(`${name}.${String(key)}`);
          return () => {};
        },
      },
    );
  const saved = {
    sessionStorage: globalThis.sessionStorage,
    localStorage: globalThis.localStorage,
    console: globalThis.console,
  };
  Object.assign(globalThis, {
    sessionStorage: trap('sessionStorage'),
    localStorage: trap('localStorage'),
    console: trap('console'),
  });
  let token = '';
  try {
    const { location, history } = page(`#token=${TOKEN}`);
    takeLaunchToken(location, history as unknown as History);
    token = launchToken();
    tokenFromInput(`http://127.0.0.1:8220/#token=${OTHER}`);
  } finally {
    Object.assign(globalThis, saved);
  }
  assert.equal(token, TOKEN);
  assert.deepEqual(touched, []);
});

test('the token form takes a bare token or a whole sign-in link', () => {
  assert.equal(tokenFromInput(TOKEN), TOKEN);
  assert.equal(tokenFromInput(`  ${TOKEN}\n`), TOKEN);
  assert.equal(tokenFromInput(`http://127.0.0.1:8220/#token=${TOKEN}`), TOKEN);
  assert.equal(tokenFromInput(`  http://[::1]:8220/#token=${TOKEN} \n`), TOKEN);
  assert.equal(tokenFromInput(`127.0.0.1:8220/#token=${TOKEN}`), TOKEN);
  // Anything else goes to the companion unchanged, which refuses it.
  for (const text of [
    'manual-token',
    `http://127.0.0.1:8220/#token=${TOKEN.slice(0, 20)}`,
    `http://127.0.0.1:8220/?token=${TOKEN}`,
    `see http://127.0.0.1:8220/#token=${TOKEN}`,
    '',
  ])
    assert.equal(tokenFromInput(text), text);
});
