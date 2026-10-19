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
