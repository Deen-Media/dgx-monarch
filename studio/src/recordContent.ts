import type { Draft } from './model';
import type { Layout, RecordData, RecordKind } from './contracts';

interface Workspace {
  draft: Draft;
  selectedArtifactId?: string;
  activeJobIds: string[];
  layout: Layout;
}

/** Editing metadata preserves the captured content; only a new record captures the workspace. */
export function recordContent(
  kind: RecordKind,
  existing: RecordData | null,
  workspace: Workspace,
): RecordData {
  if (existing) return { ...existing };
  if (kind === 'presets') return { draft: workspace.draft };
  if (kind === 'sessions')
    return {
      draft: workspace.draft,
      selected_artifact_id: workspace.selectedArtifactId,
      active_job_ids: workspace.activeJobIds,
      layout: workspace.layout,
    };
  if (kind === 'workflows') return { workflow: workspace.draft.workflow ?? undefined };
  return {};
}
