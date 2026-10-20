import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { compileModule } from 'svelte/compiler';

// Compile the actual rune module to exercise lifecycle behavior with delayed responses.
async function connectionClass() {
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
  return (await import(`data:text/javascript;base64,${Buffer.from(resolved).toString('base64')}`))
    .Connection;
}
const Connection = await connectionClass();
function setup() {
  const store = new Map<string, string>();
  globalThis.sessionStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  } as Storage;
  globalThis.document = { hidden: false } as Document;
  return new Connection();
}
function deferred<T>() {
  let resolve!: (result: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
test('snapshot refresh preserves unconsumed event pages', async () => {
  const connection = setup();
  connection.client.token = 'token';
  let snapshotTime = 0;
  const paths: string[] = [];
  connection.client.request = async (path: string) => {
    paths.push(path);
    if (path === '/snapshot')
      return {
        instance_id: 'same',
        server_time: ++snapshotTime,
        cursor: snapshotTime === 1 ? 0 : 400,
        telemetry: {},
      };
    const after = Number(path.split('=')[1]);
    return {
      cursor: after + 200,
      latest_cursor: 400,
      events: Array.from({ length: 200 }, (_, i) => ({
        cursor: after + i + 1,
        kind: 'job',
        data: { n: after + i + 1 },
      })),
    };
  };
  try {
    await connection.refresh();
    connection.stopped = false;
    await connection.poll();
    connection.stop();
    assert.equal(connection.cursor, 200);
    connection.stopped = false;
    await connection.poll();
    assert.equal(connection.cursor, 400);
    assert.equal(connection.logs.length, 400);
    assert.ok(paths.includes('/events?after=200'));
  } finally {
    connection.stop();
  }
});

test('replay reset accepts the server cursor and visibly records the gap', async () => {
  const connection = setup();
  connection.client.token = 'token';
  connection.client.request = async (path: string) =>
    path === '/snapshot'
      ? { instance_id: 'same', server_time: 1, cursor: 50, telemetry: {} }
      : { reset: true, cursor: 3, events: [{ cursor: 3, kind: 'recovered' }] };
  try {
    await connection.refresh();
    connection.stopped = false;
    await connection.poll();
    assert.equal(connection.cursor, 3);
    assert.match(connection.logs[0], /replay gap/);
    assert.match(connection.logs[1], /recovered/);
  } finally {
    connection.stop();
  }
});
test('disconnect invalidates an in-flight snapshot and clears retained host state', async () => {
  const connection = setup();
  const response = deferred<unknown>();
  connection.client.token = 'old';
  connection.client.request = () => response.promise;
  connection.logs = ['private old host log'];
  const refresh = connection.refresh();
  connection.disconnect();
  response.resolve({ instance_id: 'old', server_time: 10, jobs: [{ id: 'old' }] });
  await refresh;
  assert.equal(connection.connected, false);
  assert.equal(connection.state, 'unavailable');
  assert.deepEqual(connection.snapshot.jobs, []);
  assert.deepEqual(connection.logs, []);
});
test('disconnect invalidates negotiation before it can restore the token', async () => {
  const connection = setup();
  const response = deferred<Response>();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = () => response.promise;
  try {
    connection.token = 'secret';
    const connecting = connection.connect();
    connection.disconnect();
    response.resolve(new Response(JSON.stringify({ schema_version: '1.0' })));
    await connecting;
    assert.equal(connection.connected, false);
    assert.equal(sessionStorage.getItem('dgxm.studio.token.v1'), null);
    assert.deepEqual(connection.capabilities, {});
  } finally {
    globalThis.fetch = originalFetch;
    connection.stop();
  }
});
test('an older in-flight snapshot cannot undo a companion restart', async () => {
  const connection = setup();
  const older = deferred<unknown>(),
    newer = deferred<unknown>();
  let calls = 0;
  connection.client.token = 'token';
  connection.client.request = () => (++calls === 1 ? older.promise : newer.promise);
  const oldRequest = connection.refresh();
  const newRequest = connection.refresh();
  newer.resolve({ instance_id: 'new', server_time: 20, jobs: [], telemetry: {}, cursor: 2 });
  await newRequest;
  older.resolve({ instance_id: 'old', server_time: 10, jobs: [], telemetry: {}, cursor: 99 });
  await oldRequest;
  assert.equal(connection.snapshot.instance_id, 'new');
  assert.equal(connection.cursor, 2);
  connection.stop();
});
test('storage denial still permits manual authentication and disconnect', async () => {
  setup();
  for (const method of ['getItem', 'setItem', 'removeItem'] as const)
    sessionStorage[method] = () => {
      throw new Error('SecurityError');
    };
  const connection = new Connection();
  assert.equal(connection.token, '');
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    const body = url.endsWith('/capabilities')
      ? { schema_version: '1.0' }
      : url.endsWith('/workflows')
        ? { workflows: [] }
        : url.endsWith('/snapshot')
          ? {
              instance_id: 'synthetic',
              server_time: 10,
              jobs: [],
              artifacts: [],
              operations: [],
              records: {},
              telemetry: {},
              driver: {},
            }
          : {};
    return new Response(JSON.stringify(body));
  };
  try {
    connection.token = 'manual-token';
    await connection.connect();
    assert.equal(connection.connected, true);
    assert.equal(connection.client.token, 'manual-token');
    connection.disconnect();
    assert.equal(connection.connected, false);
    assert.equal(connection.client.token, '');
  } finally {
    connection.stop();
    globalThis.fetch = originalFetch;
  }
});
