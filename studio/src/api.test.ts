import test from 'node:test';
import assert from 'node:assert/strict';
import {
  StudioClient,
  ApiError,
  checkImportFile,
  checkRequestBody,
  requestLimit,
  DEFAULT_REQUEST_BYTES,
} from './api.ts';
const realFetch = globalThis.fetch;
test('requests bind auth and schema version to same-origin API', async () => {
  globalThis.fetch = async (input, options) => {
    assert.equal(input, '/api/v1/jobs');
    assert.equal(options?.method, 'POST');
    assert.deepEqual(options?.headers, {
      Authorization: 'Bearer test-token',
      'X-DGXM-Schema': '1.0',
      'Content-Type': 'application/json',
    });
    assert.equal(options?.body, '{"preview_id":"preview","idempotency_key":"retained"}');
    return new Response('{"id":"job"}', { status: 202 });
  };
  try {
    assert.deepEqual(
      await new StudioClient('test-token').request('/jobs', {
        preview_id: 'preview',
        idempotency_key: 'retained',
      }),
      { id: 'job' },
    );
  } finally {
    globalThis.fetch = realFetch;
  }
});
test('backend refusals remain errors with their authoritative message', async () => {
  globalThis.fetch = async () =>
    new Response('{"error":{"code":"stale_operation","message":"Configuration changed"}}', {
      status: 409,
    });
  try {
    await assert.rejects(
      new StudioClient('test-token').request('/operations/p/apply', {}),
      (e: unknown) =>
        e instanceof ApiError && e.status === 409 && e.message === 'Configuration changed',
    );
  } finally {
    globalThis.fetch = realFetch;
  }
});
test('oversized media is cancelled before reading its body', async () => {
  let cancelled = false;
  globalThis.fetch = async () =>
    new Response(
      new ReadableStream({
        cancel() {
          cancelled = true;
        },
      }),
      { headers: { 'Content-Length': String(65 * 1024 * 1024) } },
    );
  try {
    await assert.rejects(new StudioClient('test-token').media('output'), /64 MiB/);
    assert.equal(cancelled, true);
  } finally {
    globalThis.fetch = realFetch;
  }
});
test('thumbnails use an authenticated bounded endpoint independently of full media', async () => {
  let cancelled = false;
  globalThis.fetch = async (input, options) => {
    assert.equal(input, '/api/v1/artifacts/a%2Fb/thumbnail');
    assert.deepEqual(options?.headers, { Authorization: 'Bearer test-token' });
    return new Response(
      new ReadableStream({
        cancel() {
          cancelled = true;
        },
      }),
      { headers: { 'Content-Length': String(513 * 1024) } },
    );
  };
  try {
    await assert.rejects(new StudioClient('test-token').thumbnail('a/b'), /512 KiB/);
    assert.equal(cancelled, true);
  } finally {
    globalThis.fetch = realFetch;
  }
});
for (const body of ['', '{"id":', '<html>Proxy error</html>']) {
  test(`malformed successful JSON stays an uncertain failure: ${body}`, async () => {
    globalThis.fetch = async () => new Response(body, { status: 202 });
    try {
      await assert.rejects(new StudioClient('token').request('/jobs', {}), SyntaxError);
    } finally {
      globalThis.fetch = realFetch;
    }
  });
}
test('non-JSON HTTP refusal retains its HTTP status', async () => {
  globalThis.fetch = async () => new Response('Proxy unavailable', { status: 502 });
  try {
    await assert.rejects(
      new StudioClient('token').request('/jobs', {}),
      (error: unknown) =>
        error instanceof ApiError &&
        error.status === 502 &&
        /did not answer \(HTTP 502\)/.test(error.message),
    );
  } finally {
    globalThis.fetch = realFetch;
  }
});
for (const [status, body, expected] of [
  [413, '413: Request Entity Too Large', /larger than the companion accepts \(HTTP 413\)/],
  [500, '500 Internal Server Error\n\nServer got itself in trouble', /failed while handling/],
  [500, '{}', /failed while handling the request \(HTTP 500\)/],
  [401, '{"error":{}}', /refused the owner token \(HTTP 401\)/],
  [400, '{"error":{"code":"lora_binding","message":""}}', /refused the request \(HTTP 400\)/],
] as const) {
  test(`an HTTP ${status} body without a readable message gets a status-aware message: ${body}`, async () => {
    globalThis.fetch = async () =>
      new Response(body, { status, headers: { 'Content-Type': 'text/plain' } });
    try {
      await assert.rejects(new StudioClient('token').request('/workflows/inspect', {}), (error) => {
        assert.ok(error instanceof ApiError);
        assert.equal(error.status, status);
        assert.match(error.message, expected);
        assert.notEqual(error.message, '{}');
        return true;
      });
    } finally {
      globalThis.fetch = realFetch;
    }
  });
}
test('the JSON error envelope keeps its own message and code', async () => {
  globalThis.fetch = async () =>
    new Response('{"error":{"code":"request_limit","message":"Request body exceeds 4 MiB."}}', {
      status: 413,
    });
  try {
    await assert.rejects(
      new StudioClient('token').request('/workflows/inspect', {}),
      (error: unknown) =>
        error instanceof ApiError &&
        error.code === 'request_limit' &&
        error.message === 'Request body exceeds 4 MiB.',
    );
  } finally {
    globalThis.fetch = realFetch;
  }
});
test('the import limit comes from the published request ceiling', () => {
  assert.equal(requestLimit({ limits: { request_bytes: 1024 } }), 1024);
  for (const limits of [
    undefined,
    {},
    { request_bytes: 0 },
    { request_bytes: '4' },
    { request_bytes: 1.5 },
  ])
    assert.equal(requestLimit({ limits } as never), DEFAULT_REQUEST_BYTES);
  const limit = 4 * 1024 * 1024;
  // An 8 MiB file may still compact under a 4 MiB request; a larger one is not read.
  assert.doesNotThrow(() => checkImportFile(8 * 1024 * 1024, limit, 'Workflow file'));
  assert.throws(
    () => checkImportFile(8 * 1024 * 1024 + 1, limit, 'Workflow file'),
    /Workflow file is 8 MiB; files above 8 MiB are not read, because the companion accepts requests up to 4 MiB\./,
  );
  const workflow = { text: 'x'.repeat(limit) };
  assert.throws(
    () => checkRequestBody({ workflow }, limit, 'The workflow'),
    /The workflow needs a 4 MiB request; the companion accepts requests up to 4 MiB\./,
  );
  assert.doesNotThrow(() =>
    checkRequestBody({ workflow: { text: 'x'.repeat(limit - 64) } }, limit, 'W'),
  );
  // The limit counts encoded bytes, not characters.
  assert.throws(() => checkRequestBody({ text: 'é'.repeat(600) }, 1024, 'W'), /needs a/);
});
