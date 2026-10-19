import type { StudioClient } from './api';
import type { RecordKind, StoredRecord } from './contracts';

/** The durable library permits 2,000 records per kind; fetch every bounded page. */
export async function loadRecordCatalog(
  client: StudioClient,
  kind: RecordKind,
): Promise<StoredRecord[]> {
  const records = new Map<string, StoredRecord>();
  let offset: number | null = 0;
  for (let page = 0; offset !== null && page < 10; page++) {
    const result: { records: StoredRecord[]; next_offset: number | null } = await client.request(
      `/records/${kind}?limit=200&offset=${offset}`,
    );
    for (const record of result.records) records.set(record.id, record);
    offset = result.next_offset;
  }
  return [...records.values()];
}
