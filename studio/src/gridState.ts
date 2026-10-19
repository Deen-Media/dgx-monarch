export interface Cell {
  index: number;
  coordinates: Record<string, unknown>;
  excluded?: boolean;
  errors?: string[];
  preview_id?: string;
  job_id?: string;
  state?: string;
  attempts?: string[];
  artifact_ids?: string[];
}
export interface Grid {
  id: string;
  state?: string;
  digest?: string;
  cells: Cell[];
  ready?: boolean;
  total?: number;
  job_count?: number;
  expires_at?: number;
}

/** Persist enough to reconcile a request, never repeated workflow/cell payloads. */
export function gridIdentity(grid: Grid | null): Grid | null {
  if (!grid) return null;
  const { id, state, digest, ready, total, job_count, expires_at } = grid;
  return { id, state, digest, ready, total, job_count, expires_at, cells: [] };
}
