import { restoreDraft } from './model.ts';
import type { Draft } from './model.ts';

/** Restore request identity independently of optional draft and layout preferences. */
export function initialWorkspace() {
  let pending: { key: string; preview_id: string } | null = null;
  let pendingError = '';
  const errors: string[] = [];
  try {
    const value = JSON.parse(sessionStorage.getItem('dgxm.studio.pending') ?? 'null');
    if (value !== null) {
      if (
        !value ||
        typeof value !== 'object' ||
        typeof value.key !== 'string' ||
        !value.key ||
        typeof value.preview_id !== 'string' ||
        !value.preview_id
      )
        throw new Error('Invalid pending submission identity.');
      pending = { key: value.key, preview_id: value.preview_id };
    }
  } catch {
    pendingError =
      'Saved submission identity is unreadable. Check Queue for accepted work before clearing it.';
  }
  let saved = '',
    draft: Draft | null = null;
  let layout: { width: number; showTelemetry: boolean; showConsole: boolean } | null = null;
  try {
    saved = localStorage.getItem('dgxm.studio.draft.v1') ?? '';
  } catch {
    errors.push('Browser storage is unavailable, so no saved draft was read.');
  }
  try {
    draft = saved ? restoreDraft(JSON.parse(saved).draft) : null;
  } catch {
    // The caller retains `saved` and pauses autosave until the user exports or replaces it.
    draft = null;
  }
  try {
    const value = JSON.parse(localStorage.getItem('dgxm.studio.layout.v1') ?? 'null');
    if (value)
      layout = {
        width: Number.isFinite(value.width) ? Math.min(520, Math.max(280, value.width)) : 340,
        showTelemetry: typeof value.showTelemetry === 'boolean' ? value.showTelemetry : true,
        showConsole: typeof value.showConsole === 'boolean' ? value.showConsole : false,
      };
  } catch {
    errors.push('Local layout could not be restored.');
  }
  return { saved, draft, layout, pending, pendingError, error: errors.join(' ') };
}
