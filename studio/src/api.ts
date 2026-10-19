export const API_VERSION = '1';
export class ApiError extends Error {
  status: number;
  code?: string;
  constructor(status: number, message: string, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
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
          typeof data.error?.code === 'string' ? data.error.code : undefined,
        );
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
