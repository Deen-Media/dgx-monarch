import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { compileModule } from 'svelte/compiler';
import { ApiError } from './api.ts';
import { takeLaunchToken } from './launchToken.ts';
import { STALE_LINK, unreachableText } from './firstRun.ts';

// Compile the actual rune module, as connection.test.ts does. Relative imports resolve to the
// same .ts files this test imports, so both share one launch token.
async function connectionModule() {
  const source = await readFile(new URL('./connection.svelte.ts', import.meta.url), 'utf8');
  const plain = ts.transpile(source, {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
  });
  const compiled = compileModule(plain, { filename: 'connection.svelte.js', generate: 'client' }).js
    .code;
  const resolved = compiled.replace(/from ['"]([^'"]+)['"]/g, (_match, specifier) => {
    const url = specifier.startsWith('.')
      ? new URL(`${specifier}.ts`, import.meta.url).href
      : import.meta.resolve(specifier);
    return `from ${JSON.stringify(url)}`;
  });
  return import(`data:text/javascript;base64,${Buffer.from(resolved).toString('base64')}`);
}
const { Connection, errorText } = await connectionModule();

// Fake tokens of the companion's shape: 43 characters of [A-Za-z0-9_-].
const LINK = 'fakeLinkToken_0123456789-abcdefghijklmnopqr';
const STORED = 'fakeStoredToken_0123456789-abcdefghijklmnop';
const FRESH = 'fakeFreshToken_0123456789-abcdefghijklmnopq';
const KEY = 'dgxm.studio.token.v1';
const REFUSED = "Enter this companion's owner token.";

/** Session storage with these entries, or one whose every call throws. */
function setup(entries: Record<string, string> = {}, broken = false) {
  const store = new Map(Object.entries(entries));
  const fail = () => {
    throw new Error('SecurityError');
  };
  globalThis.sessionStorage = (
    broken
      ? { getItem: fail, setItem: fail, removeItem: fail }
      : {
          getItem: (key: string) => store.get(key) ?? null,
          setItem: (key: string, value: string) => void store.set(key, value),
          removeItem: (key: string) => void store.delete(key),
        }
  ) as Storage;
  globalThis.document = { hidden: false } as Document;
  globalThis.location = { host: '127.0.0.1:8765' } as Location;
  return store;
}
/** Open the page from a sign-in link, as main.ts does before the app mounts. */
function openLink(token: string) {
  const history = { state: null, replaceState() {} };
  takeLaunchToken({ hash: `#token=${token}`, pathname: '/', search: '' }, history);
}

interface Fixture {
  token: string;
  catalog: string;
  driver: Record<string, unknown>;
}
/** A companion that accepts one token; `catalog` is 'ok' or the code its catalog read refuses with. */
async function withCompanion(fixture: Fixture, body: () => Promise<void>) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
    const auth = new Headers(init?.headers).get('Authorization');
    if (auth !== `Bearer ${fixture.token}`)
      return json({ error: { code: 'authentication', message: REFUSED } }, 401);
    if (url.endsWith('/capabilities'))
      return json({ schema_version: '1.0', driver_origin: 'http://127.0.0.1:8191' });
    if (url.endsWith('/workflows')) return json({ workflows: [] });
    if (url.endsWith('/catalog'))
      return fixture.catalog === 'ok'
        ? json({ loras: [] })
        : json({ error: { code: fixture.catalog, message: 'Driver catalog refused.' } }, 503);
    return json({
      instance_id: 'fixture',
      server_time: 10,
      jobs: [],
      artifacts: [],
      operations: [],
      records: {},
      telemetry: {},
      driver: fixture.driver,
    });
  };
  try {
    await body();
  } finally {
    globalThis.fetch = originalFetch;
  }
}
const up = { state: 'connected' };
const notRunning = { state: 'unavailable', code: 'driver_not_running', since: 5 };

test('a sign-in link token beats the stored one and is stored once the companion accepts it', async () => {
  const store = setup({ [KEY]: STORED });
  openLink(LINK);
  const connection = new Connection();
  assert.equal(connection.token, LINK);
  assert.equal(store.get(KEY), STORED);
  await withCompanion({ token: LINK, catalog: 'ok', driver: up }, async () => {
    try {
      await connection.connect();
      assert.equal(connection.connected, true);
      assert.equal(connection.client.token, LINK);
      assert.equal(store.get(KEY), LINK);
    } finally {
      connection.stop();
    }
  });
  // The link is handed over once; a later connection starts from storage.
  const later = new Connection();
  assert.equal(later.fromLink, false);
  assert.equal(later.token, LINK);
});

test('storage that throws still connects with the link token', async () => {
  setup({}, true);
  openLink(LINK);
  const connection = new Connection();
  assert.equal(connection.token, LINK);
  await withCompanion({ token: LINK, catalog: 'ok', driver: up }, async () => {
    try {
      await connection.connect();
      assert.equal(connection.connected, true);
      assert.equal(connection.client.token, LINK);
    } finally {
      connection.stop();
    }
  });
});

test('a refused link token does not replace the stored token and names the link', async () => {
  const store = setup({ [KEY]: STORED });
  openLink(LINK);
  const connection = new Connection();
  await withCompanion({ token: STORED, catalog: 'ok', driver: up }, async () => {
    try {
      const error = await connection.connect().catch((e: unknown) => e);
      assert.ok(error instanceof ApiError && error.status === 401);
      assert.equal(error.message, REFUSED);
      assert.equal(connection.connected, false);
      assert.equal(connection.fromLink, true);
      assert.equal(errorText(error, connection.fromLink), STALE_LINK);
      assert.equal(store.get(KEY), STORED);
      // A token the user then types is judged on its own, not blamed on the link.
      connection.token = FRESH;
      const typed = await connection.connect().catch((e: unknown) => e);
      assert.equal(connection.fromLink, false);
      assert.equal(errorText(typed, connection.fromLink), REFUSED);
      // The stored token still signs in.
      connection.token = STORED;
      await connection.connect();
      assert.equal(connection.connected, true);
    } finally {
      connection.stop();
    }
  });
});

test('fromLink clears after a successful connect', async () => {
  setup();
  openLink(LINK);
  const connection = new Connection();
  assert.equal(connection.fromLink, true);
  await withCompanion({ token: LINK, catalog: 'ok', driver: up }, async () => {
    try {
      await connection.connect();
      assert.equal(connection.fromLink, false);
    } finally {
      connection.stop();
    }
  });
  // The same token refused later means a restart, not a bad link.
  await withCompanion({ token: FRESH, catalog: 'ok', driver: up }, async () => {
    try {
      const error = await connection.connect().catch((e: unknown) => e);
      assert.equal(connection.fromLink, false);
      assert.match(errorText(error, connection.fromLink), /may have restarted with a new owner/);
    } finally {
      connection.stop();
    }
  });
});

test('a whole sign-in link pasted into the form signs in with its token', async () => {
  const store = setup();
  const connection = new Connection();
  assert.equal(connection.token, '');
  await withCompanion({ token: LINK, catalog: 'ok', driver: up }, async () => {
    try {
      connection.token = `  http://127.0.0.1:8220/#token=${LINK} `;
      await connection.connect();
      assert.equal(connection.client.token, LINK);
      assert.equal(store.get(KEY), LINK);
      connection.disconnect();
      // A pasted link that the companion refuses gets the link sentence too.
      connection.token = `http://127.0.0.1:8220/#token=${STORED}`;
      const error = await connection.connect().catch((e: unknown) => e);
      assert.equal(errorText(error, connection.fromLink), STALE_LINK);
    } finally {
      connection.stop();
    }
  });
});

test('a 401 at connect with a stored token gives the restart sentence', async () => {
  setup({ [KEY]: STORED });
  const connection = new Connection();
  assert.equal(connection.token, STORED);
  await withCompanion({ token: FRESH, catalog: 'ok', driver: up }, async () => {
    try {
      const error = await connection.connect().catch((e: unknown) => e);
      assert.ok(error instanceof ApiError && error.status === 401);
      assert.equal(
        errorText(error, connection.fromLink),
        `${REFUSED} The companion may have restarted with a new owner token. Enter the current token to reconnect, or run dgxm studio open on the Studio host for a new sign-in link.`,
      );
      // A token typed for the first time gets the companion's own refusal.
      connection.token = LINK;
      const typed = await connection.connect().catch((e: unknown) => e);
      assert.equal(errorText(typed, connection.fromLink), REFUSED);
    } finally {
      connection.stop();
    }
  });
});

test('a polling failure that is not an ApiError names the host and the tunnel', async () => {
  setup();
  const connection = new Connection();
  connection.client.token = STORED;
  connection.connected = true;
  connection.state = 'connected';
  connection.client.request = async () => {
    throw new TypeError('NetworkError when attempting to fetch resource.');
  };
  try {
    connection.stopped = false;
    await connection.poll();
    assert.equal(connection.connected, true);
    assert.equal(connection.state, 'stale');
    assert.equal(connection.error, unreachableText('127.0.0.1:8765'));
    // A companion that answers keeps its own message.
    connection.client.request = async () => {
      throw new ApiError(502, 'The companion or a proxy in front of it did not answer (HTTP 502).');
    };
    connection.stopped = false;
    await connection.poll();
    assert.match(connection.error, /did not answer \(HTTP 502\)/);
  } finally {
    connection.stop();
  }
});

test('the connect notice is empty while the driver is not running', async () => {
  setup();
  const connection = new Connection();
  const fixture = { token: LINK, catalog: 'driver_not_running', driver: notRunning };
  await withCompanion(fixture, async () => {
    try {
      connection.token = LINK;
      await connection.connect();
      assert.equal(connection.connected, true);
      assert.equal(connection.driverNotRunning, true);
      assert.equal(connection.degradedNotice(), '');
      // Before the first observation the snapshot has no code; the catalog code decides.
      connection.snapshot = { ...connection.snapshot, driver: { state: 'unavailable' } };
      assert.equal(connection.driverNotRunning, true);
      assert.equal(connection.degradedNotice(), '');
      // Any other driver fault keeps the notice.
      connection.snapshot = {
        ...connection.snapshot,
        driver: { state: 'unavailable', code: 'driver_unknown' },
      };
      assert.equal(connection.driverNotRunning, false);
      assert.match(connection.degradedNotice(), /could not be read: Driver catalog refused\./);
    } finally {
      connection.stop();
    }
  });
});

test('catalogCode follows the catalog read', async () => {
  setup();
  const connection = new Connection();
  const fixture = { token: LINK, catalog: 'driver_not_running', driver: notRunning };
  await withCompanion(fixture, async () => {
    try {
      connection.token = LINK;
      await connection.connect();
      assert.equal(connection.catalogCode, 'driver_not_running');
      fixture.catalog = 'ok';
      fixture.driver = up;
      await connection.refreshCatalog();
      assert.equal(connection.catalogCode, '');
      assert.equal(connection.catalogError, '');
      fixture.catalog = 'driver_unknown';
      await connection.refreshCatalog();
      assert.equal(connection.catalogCode, 'driver_unknown');
      // A read that fails without a refusal carries no code.
      const originalFetch = globalThis.fetch;
      globalThis.fetch = async () => {
        throw new TypeError('NetworkError when attempting to fetch resource.');
      };
      try {
        await connection.refreshCatalog();
      } finally {
        globalThis.fetch = originalFetch;
      }
      assert.equal(connection.catalogCode, '');
      fixture.catalog = 'driver_not_running';
      await connection.refreshCatalog();
      assert.equal(connection.catalogCode, 'driver_not_running');
      connection.disconnect();
      assert.equal(connection.catalogCode, '');
      assert.equal(connection.driverNotRunning, false);
    } finally {
      connection.stop();
    }
  });
});
