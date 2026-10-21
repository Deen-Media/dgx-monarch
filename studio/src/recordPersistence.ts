import { ApiError, type StudioClient } from './api.ts';
import type { RecordData, RecordKind, StoredRecord } from './contracts';
import { findRecord } from './recordCatalog.ts';

/** Client code for a kept identity that an earlier attempt already saved with other content. */
export const RECORD_SAVED_EARLIER = 'record_saved_earlier';

const canonical = (value: unknown): string =>
  JSON.stringify(value, (_key, item) =>
    item && typeof item === 'object' && !Array.isArray(item)
      ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b)))
      : item,
  );
/** The companion keeps every field the client sends and may add its own, such as grid snapshots. */
const holds = (stored: RecordData, sent: RecordData) =>
  Object.entries(sent).every(([key, value]) => canonical(stored[key]) === canonical(value));
const message = (cause: unknown) => (cause instanceof Error ? cause.message : String(cause));

/**
 * PUT a record under a kept identity. A revision conflict on a new record can mean that an
 * earlier attempt with this identity committed and its reply was lost, so read the record back
 * and treat it as saved when it holds what this attempt sent.
 */
export async function putRecord(
  client: StudioClient,
  kind: RecordKind,
  data: RecordData,
  id: string,
  revision = 0,
): Promise<{ record: StoredRecord; replayed: boolean }> {
  try {
    const record = await client.request<StoredRecord>(
      `/records/${kind}/${encodeURIComponent(id)}`,
      { revision, data },
      'PUT',
    );
    return { record, replayed: false };
  } catch (cause) {
    if (
      revision !== 0 ||
      !(cause instanceof ApiError) ||
      cause.status !== 409 ||
      cause.code !== 'revision_conflict'
    )
      throw cause;
    const stored = await findRecord(client, kind, id);
    if (!stored) throw cause;
    if (!holds(stored.data, data))
      throw new ApiError(
        409,
        'An earlier attempt already saved this record with different content. It is in the library now; edit that record to apply these changes.',
        RECORD_SAVED_EARLIER,
      );
    return { record: stored, replayed: true };
  }
}

export async function persistRecord(
  client: StudioClient,
  refresh: () => Promise<void>,
  notice: (text: string) => void,
  kind: RecordKind,
  data: RecordData,
  id: string = crypto.randomUUID(),
  revision = 0,
): Promise<StoredRecord> {
  const { record, replayed } = await putRecord(client, kind, data, id, revision);
  const saved = replayed
    ? `${kind} record was already saved by an earlier attempt.`
    : `${kind} record saved.`;
  try {
    await refresh();
  } catch (cause) {
    // The PUT committed; a failed refresh must not read as a failed save.
    notice(`${saved} The view did not refresh: ${message(cause)}`);
    return record;
  }
  notice(saved);
  return record;
}

/** One record identity per form session, kept until a save is confirmed or the identity is used. */
export class RecordSession {
  #scope = '';
  #id = '';
  #mint: () => string;
  constructor(mint: () => string = () => crypto.randomUUID()) {
    this.#mint = mint;
  }
  id(scope = ''): string {
    if (!this.#id || scope !== this.#scope) {
      this.#scope = scope;
      this.#id = this.#mint();
    }
    return this.#id;
  }
  /** Start a new form session after a confirmed save. */
  settle() {
    this.#id = '';
  }
  /** Keep the identity when the outcome is unknown, so a retry reconciles the same record. */
  failed(cause: unknown) {
    if (cause instanceof ApiError && cause.code === RECORD_SAVED_EARLIER) this.settle();
  }
}

export function recordProjects(data: RecordData): string[] {
  if (Array.isArray(data.project_ids))
    return data.project_ids.filter((id): id is string => typeof id === 'string');
  const legacy = data.project_id ?? data.draft?.project_id;
  return typeof legacy === 'string' && legacy ? [legacy] : [];
}
