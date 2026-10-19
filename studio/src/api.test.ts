import test from 'node:test';
import assert from 'node:assert/strict';
import { StudioClient, ApiError } from './api.ts';
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
