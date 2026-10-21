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
/** Media kinds the companion records; uploads carry their own kinds. */
export const GALLERY_KINDS = [
  { value: '', label: 'All media' },
  { value: 'image', label: 'Images' },
  { value: 'video', label: 'Video' },
  { value: 'input', label: 'Uploaded inputs' },
  { value: 'mask', label: 'Uploaded masks' },
];
/** Output states. Intermediate previews are a flag on the artifact, not a state. */
export const GALLERY_STATES = [
  { value: '', label: 'All states' },
  { value: 'succeeded', label: 'succeeded' },
  { value: 'failed', label: 'failed' },
  { value: 'cancelled', label: 'cancelled' },
  { value: 'intermediate', label: 'Intermediate previews' },
];

/** Build the companion query for the gallery filters; every filter maps to a stored field. */
export function galleryQuery(
  filter: GalleryFilter,
  scope: { projectId: string; width?: string | number | null; height?: string | number | null },
): string {
  const params = new URLSearchParams({
    limit: '100',
    sort: filter.sort === 'oldest' ? 'oldest' : 'newest',
  });
  const day = (value: string, time: string) =>
    value ? String(new Date(`${value}T${time}`).getTime() / 1000) : '';
  const options: Record<string, string> = {
    q: filter.query,
    kind: filter.kind,
    operation: filter.operation,
    model: filter.model,
    status: filter.status === 'intermediate' ? '' : filter.status,
    intermediate: filter.status === 'intermediate' ? 'true' : '',
    project_id: scope.projectId,
    width: scope.width ? String(scope.width) : '',
    height: scope.height ? String(scope.height) : '',
    after: day(filter.from, '00:00:00'),
    before: day(filter.until, '23:59:59.999'),
  };
  for (const [key, value] of Object.entries(options)) if (value) params.set(key, value);
  return params.toString();
}

/**
 * ComfyUI does not report the size of generated outputs, so only uploads carry dimensions.
 * The size filters are useful only when a loaded row has them.
 */
export function hasDimensions(artifacts: Artifact[]): boolean {
  return artifacts.some((artifact) =>
    [artifact.metadata?.width, artifact.metadata?.height].some(
      (value) => typeof value === 'number' && Number.isFinite(value),
    ),
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
