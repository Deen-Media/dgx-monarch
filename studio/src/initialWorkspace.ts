import { restoreDraft } from './model';

/** Read all local state before applying it, so malformed JSON cannot partially restore the UI. */
export function initialWorkspace() {
  const saved = localStorage.getItem('dgxm.studio.draft.v1');
  const layout = JSON.parse(localStorage.getItem('dgxm.studio.layout.v1') ?? 'null');
  const pending = JSON.parse(sessionStorage.getItem('dgxm.studio.pending') ?? 'null');
  if (pending && (typeof pending.key !== 'string' || typeof pending.preview_id !== 'string'))
    throw new Error('Invalid pending submission identity.');
  return {
    saved: saved ?? '',
    draft: saved ? restoreDraft(JSON.parse(saved).draft) : null,
    layout: layout
      ? {
          width: Number.isFinite(layout.width) ? Math.min(520, Math.max(280, layout.width)) : 340,
          showTelemetry: typeof layout.showTelemetry === 'boolean' ? layout.showTelemetry : true,
          showConsole: typeof layout.showConsole === 'boolean' ? layout.showConsole : false,
        }
      : null,
    pending,
  };
}
