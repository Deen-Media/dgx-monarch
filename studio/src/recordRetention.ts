import type { StoredRecord } from './contracts';

export const RECORD_BYTE_LIMIT = 8 * 1024 * 1024;
export const RECORD_ROW_LIMIT = 2000;
export function recordBytes(record: StoredRecord): number {
  return new TextEncoder().encode(JSON.stringify(record)).byteLength + 1;
}

/** A page can replace a full window; it never silently drops rows from that page. */
export function retainRecordPage(current: StoredRecord[], page: StoredRecord[]) {
  const pageBytes = page.reduce((total, record) => total + recordBytes(record), 2);
  if (pageBytes > RECORD_BYTE_LIMIT || page.length > RECORD_ROW_LIMIT)
    throw new Error('The record page exceeds the 8 MiB / 2,000-record display limit.');
  const currentBytes = current.reduce((total, record) => total + recordBytes(record), 2);
  const replaced =
    current.length > 0 &&
    (currentBytes + pageBytes > RECORD_BYTE_LIMIT ||
      current.length + page.length > RECORD_ROW_LIMIT);
  return { records: replaced ? page : [...current, ...page], replaced };
}
