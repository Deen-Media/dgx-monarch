import { restoreDraft } from './model.ts';
import type { Draft } from './model.ts';

const LAYOUT_KEY = 'dgxm.studio.layout.v1';

/** The layout preferences this browser keeps; a panel may add one of its own. */
interface LayoutPreferences {
  width: number;
  showTelemetry: boolean;
  showConsole: boolean;
  /** Whether the telemetry panel's detailed telemetry is open; closed until opened. */
  telemetryDetails: boolean;
}

/** The stored preferences, or none when storage fails or holds no readable object. */
function storedLayout(): Record<string, unknown> {
  try {
    const value = JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? 'null');
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

/**
 * Keep layout preferences for the next visit, merged into the stored ones, so the workspace's
 * save and a panel's save never drop each other's; the settings still work in this tab when
 * storage fails.
 */
export function saveLayout(layout: Partial<LayoutPreferences>) {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify({ ...storedLayout(), ...layout }));
  } catch {
    /* Settings remain usable in this tab. */
  }
}

/** Whether the telemetry panel's detailed telemetry was left open; closed by default. */
export function telemetryDetailsOpen(): boolean {
  return storedLayout().telemetryDetails === true;
}

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
    const value = JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? 'null');
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
