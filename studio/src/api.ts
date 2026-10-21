export const API_VERSION = '1';
/** The companion's request ceiling when its capabilities do not publish one. */
export const DEFAULT_REQUEST_BYTES = 4 * 1024 * 1024;
export class ApiError extends Error {
  status: number;
  code?: string;
  constructor(status: number, message: string, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}
/** An acknowledgement without its durable identity cannot settle a mutation. */
export function requireRecordIdentity<T>(value: T): T {
  if (
    !value ||
    typeof value !== 'object' ||
    !('id' in value) ||
    typeof value.id !== 'string' ||
    !value.id
  )
    throw new Error(
      'The companion response has no durable record identity. Reconcile the retained request.',
    );
  return value;
}
export class StudioClient {
  token: string;
  constructor(token: string) {
    this.token = token;
  }
  async request<T = any>(path: string, body?: unknown, method?: string): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(`/api/v1${path}`, {
        method: method ?? (body === undefined ? 'GET' : 'POST'),
        headers: {
          Authorization: `Bearer ${this.token}`,
          'X-DGXM-Schema': `${API_VERSION}.0`,
          ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
        credentials: 'same-origin',
        cache: 'no-store',
      });
      const data = response.ok ? await response.json() : await response.json().catch(() => null);
      if (!response.ok) throw refusal(response.status, data);
      return data as T;
    } finally {
      clearTimeout(timeout);
    }
  }
  async thumbnail(id: string, signal?: AbortSignal): Promise<Blob> {
    return this.media(id, signal, true);
  }
  async media(id: string, signal?: AbortSignal, thumbnail = false): Promise<Blob> {
    const response = await fetch(
      `/api/v1/artifacts/${encodeURIComponent(id)}/${thumbnail ? 'thumbnail' : 'file'}`,
      {
        headers: { Authorization: `Bearer ${this.token}` },
        cache: 'no-store',
        signal,
      },
    );
    if (!response.ok) throw new ApiError(response.status, `Media unavailable (${response.status})`);
    const limit = thumbnail ? 512 * 1024 : 64 * 1024 * 1024;
    const size = Number(response.headers.get('Content-Length') ?? 0);
    if (size > limit) {
      await response.body?.cancel();
      throw new Error(
        thumbnail
          ? 'Thumbnail exceeds the 512 KiB display budget.'
          : 'Media exceeds the 64 MiB display budget.',
      );
    }
    if (!response.body) throw new Error('Media response has no body.');
    const reader = response.body.getReader();
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    let received = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      received += value.byteLength;
      if (received > limit) {
        await reader.cancel();
        throw new Error(
          thumbnail
            ? 'Thumbnail exceeds the 512 KiB display budget.'
            : 'Media exceeds the 64 MiB display budget.',
        );
      }
      chunks.push(value as Uint8Array<ArrayBuffer>);
    }
    return new Blob(chunks, {
      type: response.headers.get('Content-Type') ?? 'application/octet-stream',
    });
  }
}

/** Say what an HTTP status means when the refusal carries no readable message. */
export function statusMessage(status: number): string {
  if (status === 401) return 'The companion refused the owner token (HTTP 401).';
  if (status === 403) return 'The companion refused this request (HTTP 403).';
  if (status === 404) return 'The companion has no such record or route (HTTP 404).';
  if (status === 413) return 'The request is larger than the companion accepts (HTTP 413).';
  if (status === 502 || status === 503 || status === 504)
    return `The companion or a proxy in front of it did not answer (HTTP ${status}).`;
  if (status >= 500) return `The companion failed while handling the request (HTTP ${status}).`;
  return `The companion refused the request (HTTP ${status}).`;
}

/** Build the error for a refused request, from the JSON envelope when one is readable. */
export function refusal(status: number, data: unknown): ApiError {
  const body = data && typeof data === 'object' ? (data as Record<string, any>) : {};
  const error = body.error;
  const text = (value: unknown) => (typeof value === 'string' && value.trim() ? value : undefined);
  const message =
    text(error) ?? text(error?.message) ?? text(body.message) ?? statusMessage(status);
  return new ApiError(status, message, typeof error?.code === 'string' ? error.code : undefined);
}

/** The request ceiling the companion publishes in `capabilities.limits.request_bytes`. */
export function requestLimit(capabilities: { limits?: { request_bytes?: unknown } }): number {
  const value = capabilities.limits?.request_bytes;
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
    ? value
    : DEFAULT_REQUEST_BYTES;
}

const mib = (bytes: number) => `${Number((bytes / 1048576).toFixed(1))} MiB`;

/**
 * Refuse a file before reading it. Indentation is dropped when a workflow is
 * sent, so a file may be up to twice the request limit; the encoded request is
 * then checked against the exact limit.
 */
export function checkImportFile(size: number, limit: number, label: string) {
  if (size > 2 * limit)
    throw new Error(
      `${label} is ${mib(size)}; files above ${mib(2 * limit)} are not read, because the companion accepts requests up to ${mib(limit)}.`,
    );
}

/** Refuse a request body the companion would refuse with HTTP 413. */
export function checkRequestBody(body: unknown, limit: number, label: string) {
  const size = new TextEncoder().encode(JSON.stringify(body)).length;
  if (size > limit)
    throw new Error(
      `${label} needs a ${mib(size)} request; the companion accepts requests up to ${mib(limit)}.`,
    );
}
