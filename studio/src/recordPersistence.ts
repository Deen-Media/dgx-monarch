import type { StudioClient } from './api';
import type { RecordData, RecordKind, StoredRecord } from './contracts';

export async function persistRecord(
  client: StudioClient,
  refresh: () => Promise<void>,
  notice: (text: string) => void,
  kind: RecordKind,
  data: RecordData,
  id: string = crypto.randomUUID(),
  revision = 0,
): Promise<StoredRecord> {
  const result = await client.request<StoredRecord>(
    `/records/${kind}/${encodeURIComponent(id)}`,
    { revision, data },
    'PUT',
  );
  await refresh();
  notice(`${kind} record saved.`);
  return result;
}

export function recordProjects(data: RecordData): string[] {
  if (Array.isArray(data.project_ids))
    return data.project_ids.filter((id): id is string => typeof id === 'string');
  const legacy = data.project_id ?? data.draft?.project_id;
  return typeof legacy === 'string' && legacy ? [legacy] : [];
}
