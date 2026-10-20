import type { StudioClient } from './api';
import type { StoredRecord } from './contracts';
import { loadRecordCatalog } from './recordCatalog.ts';

const cache = new WeakMap<StudioClient, { revision: string; pending: Promise<StoredRecord[]> }>();

/** Reuse one bounded full catalog per client until its compact revision signal changes. */
export function projectCatalog(client: StudioClient, revision: string): Promise<StoredRecord[]> {
  const previous = cache.get(client);
  if (previous?.revision === revision) return previous.pending;
  const pending = loadRecordCatalog(client, 'projects').catch((error) => {
    if (cache.get(client)?.pending === pending) cache.delete(client);
    throw error;
  });
  cache.set(client, { revision, pending });
  return pending;
}
