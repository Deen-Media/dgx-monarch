export const API_VERSION = '1';
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
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
      const data = await response.json().catch(() => ({}));
      if (!response.ok)
        throw new ApiError(
          response.status,
          typeof data.error === 'string'
            ? data.error
            : (data.error?.message ?? data.message ?? JSON.stringify(data.error ?? data)),
        );
      return data as T;
    } finally {
      clearTimeout(timeout);
    }
  }
  async media(id: string, signal?: AbortSignal): Promise<Blob> {
    const response = await fetch(`/api/v1/artifacts/${encodeURIComponent(id)}/file`, {
      headers: { Authorization: `Bearer ${this.token}` },
      cache: 'no-store',
      signal,
    });
    if (!response.ok) throw new ApiError(response.status, `Media unavailable (${response.status})`);
    const limit = 64 * 1024 * 1024;
    const size = Number(response.headers.get('Content-Length') ?? 0);
    if (size > limit) {
      await response.body?.cancel();
      throw new Error('Media exceeds the 64 MiB display budget.');
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
        throw new Error('Media exceeds the 64 MiB display budget.');
      }
      chunks.push(value as Uint8Array<ArrayBuffer>);
    }
    return new Blob(chunks, {
      type: response.headers.get('Content-Type') ?? 'application/octet-stream',
    });
  }
}
