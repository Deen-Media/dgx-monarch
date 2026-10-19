import type { Artifact, Job } from './contracts';

export interface GalleryFilter {
  query: string;
  kind: string;
  operation: string;
  model: string;
  status: string;
  project: string;
  from: string;
  until: string;
  sort: string;
}
export interface GalleryEntry {
  artifact: Artifact;
  job?: Job;
  operation: string;
  models: string[];
  status: string;
  project: string;
  searchable: string;
}
export function scalarLeaves(value: unknown, prefix = '', depth = 0): Record<string, unknown> {
  if (depth > 12) return {};
  if (value === null || typeof value !== 'object') return { [prefix]: value };
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, item]) =>
      Object.entries(scalarLeaves(item, prefix ? `${prefix}.${key}` : key, depth + 1)),
    ),
  );
}
export function galleryEntries(artifacts: Artifact[], jobs: Job[]): GalleryEntry[] {
  const byId = new Map(jobs.map((job) => [job.id, job]));
  return artifacts.map((artifact) => {
    const job = byId.get(artifact.job_id);
    const leaves = scalarLeaves(job?.submission ?? artifact.metadata ?? {});
    const models = [
      ...new Set(
        Object.entries(leaves)
          .filter(
            ([key, value]) => /model|ckpt|unet|checkpoint/i.test(key) && typeof value === 'string',
          )
          .map(([, value]) => String(value)),
      ),
    ];
    return {
      artifact,
      job,
      models,
      operation: String(
        job?.submission.requested?.operation ??
          artifact.operation ??
          artifact.metadata?.operation ??
          'unknown',
      ),
      status: job?.state ?? artifact.status ?? artifact.state ?? 'unknown',
      project: String(artifact.project_id ?? job?.submission.project_id ?? ''),
      searchable: JSON.stringify({ artifact, submission: job?.submission }).toLowerCase(),
    };
  });
}
export function filterGallery(entries: GalleryEntry[], filter: GalleryFilter): GalleryEntry[] {
  const from = filter.from ? new Date(`${filter.from}T00:00:00`).getTime() / 1000 : -Infinity;
  const until = filter.until ? new Date(`${filter.until}T23:59:59.999`).getTime() / 1000 : Infinity;
  return entries
    .filter(
      (entry) =>
        (!filter.project || entry.project === filter.project) &&
        (!filter.query || entry.searchable.includes(filter.query.toLowerCase())) &&
        (!filter.kind || entry.artifact.kind === filter.kind) &&
        (!filter.operation || entry.operation === filter.operation) &&
        (!filter.status || entry.status === filter.status) &&
        (!filter.model ||
          entry.models.some((model) => model.toLowerCase().includes(filter.model.toLowerCase()))) &&
        entry.artifact.created_at >= from &&
        entry.artifact.created_at <= until,
    )
    .sort((a, b) =>
      filter.sort === 'name'
        ? a.artifact.filename.localeCompare(b.artifact.filename)
        : filter.sort === 'oldest'
          ? a.artifact.created_at - b.artifact.created_at
          : b.artifact.created_at - a.artifact.created_at,
    );
}
export function virtualRows(
  count: number,
  width: number,
  height: number,
  top: number,
  rowHeight = 330,
) {
  const columns = Math.max(1, Math.floor(width / 250));
  const rows = Math.ceil(count / columns);
  const startRow = Math.max(0, Math.min(rows, Math.floor(top / rowHeight) - 1));
  const endRow = Math.min(rows, Math.ceil((top + height) / rowHeight) + 1);
  return {
    columns,
    start: startRow * columns,
    end: Math.min(count, endRow * columns),
    before: startRow * rowHeight,
    after: Math.max(0, (rows - endRow) * rowHeight),
  };
}
export function parameterDifferences(left: unknown, right: unknown) {
  const a = scalarLeaves(left),
    b = scalarLeaves(right);
  return [...new Set([...Object.keys(a), ...Object.keys(b)])]
    .sort()
    .filter((key) => JSON.stringify(a[key]) !== JSON.stringify(b[key]))
    .map((key) => ({ key, left: a[key], right: b[key] }));
}
