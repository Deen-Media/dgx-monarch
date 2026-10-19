import type { StudioClient } from './api';
import type { RecordKind, StoredRecord } from './contracts';
import { RECORD_BYTE_LIMIT, RECORD_ROW_LIMIT, recordBytes } from './recordRetention.ts';

/** The durable library permits 2,000 records per kind; fetch every bounded page. */
export async function loadRecordCatalog(
  client: StudioClient,
  kind: RecordKind,
): Promise<StoredRecord[]> {
  const records = new Map<string, StoredRecord>();
  let offset: number | null = 0;
  let read = 0,
    bytes = 2;
  while (offset !== null && read < RECORD_ROW_LIMIT) {
    const result: { records: StoredRecord[]; next_offset: number | null } = await client.request(
      `/records/${kind}?limit=200&offset=${offset}`,
    );
    read += result.records.length;
    if (read > RECORD_ROW_LIMIT)
      throw new Error(
        'The catalog exceeds the 2,000-record browser limit. Browse it by window in the Project library.',
      );
    for (const record of result.records) {
      const previous = records.get(record.id);
      bytes += recordBytes(record) - (previous ? recordBytes(previous) : 0);
      if (bytes > RECORD_BYTE_LIMIT)
        throw new Error(
          'This catalog exceeds the 8 MiB browser limit. Browse records by window in the Project library. No partial catalog was loaded.',
        );
      if (records.size < RECORD_ROW_LIMIT || previous) records.set(record.id, record);
    }
    if (
      result.next_offset !== null &&
      (!Number.isInteger(result.next_offset) ||
        result.next_offset <= offset ||
        !result.records.length)
    )
      throw new Error('The companion returned a non-advancing library page.');
    offset = result.next_offset;
  }
  if (offset !== null)
    throw new Error(
      'This catalog exceeds the 2,000-record browser limit. Browse it by window in the Project library.',
    );
  return [...records.values()];
}
