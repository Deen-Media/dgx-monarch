import { restoreDraft } from './model';
import type { Draft } from './model';
import { exportText } from './files';

export const DRAFT_KEY = 'dgxm.studio.draft.v1';
export const STORAGE_UNAVAILABLE =
  'Browser storage is full or blocked, so this draft is not being saved in this browser. Export it to keep your changes; Studio tries again on your next edit.';

/** Sorted-key JSON of a draft, so two tabs holding the same draft compare equal. */
export function draftPayload(draft: unknown): string {
  return JSON.stringify(draft, (_key, value) =>
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(
          Object.keys(value)
            .sort()
            .map((key) => [key, value[key]]),
        )
      : value,
  );
}

/**
 * Saves the unsent draft to this browser. Saving pauses while another tab's
 * newer draft is unresolved, and while a stored draft that failed to restore is
 * kept for export, so neither is overwritten without the user choosing to.
 */
export class DraftAutosave {
  /** Another tab saved a different draft. */
  conflict = $state(false);
  /** The stored draft could not be restored and is kept until the user acts. */
  retained = $state(false);
  /** The last write failed; '' after a write succeeds. */
  storageError = $state('');
  private saved = '';
  private payload = '';
  /** The exact stored text that failed to restore. */
  private kept = '';
  private storage: () => Storage;
  private download: (text: string, name: string) => void;

  constructor(
    storage: () => Storage = () => localStorage,
    download: (text: string, name: string) => void = exportText,
  ) {
    this.storage = storage;
    this.download = download;
  }

  /** Adopt the stored text and the draft restored from it at startup. */
  restore(saved: string, draft: Draft | null) {
    this.saved = saved;
    this.payload = draft ? draftPayload(draft) : '';
    this.retained = draft === null && saved !== '';
    this.kept = this.retained ? saved : '';
  }

  /** Write an edited draft, unless saving is paused or it equals what this tab last saved. */
  save(draft: Draft, dirty: boolean) {
    if (!dirty || this.conflict || this.retained) return;
    const payload = draftPayload(draft);
    if (payload === this.payload) return;
    const value = JSON.stringify({
      schema_version: '1.0',
      draft,
      updated_at: new Date().toISOString(),
    });
    try {
      this.storage().setItem(DRAFT_KEY, value);
    } catch {
      this.storageError = STORAGE_UNAVAILABLE;
      return;
    }
    this.saved = value;
    this.payload = payload;
    this.storageError = '';
  }

  /** React to another tab's write; the same draft under a newer timestamp is not a conflict. */
  observe(key: string | null, value: string | null) {
    if (key !== DRAFT_KEY && key !== null) return;
    if (value === this.saved) return;
    if (value === null) {
      // The record was removed; the next edit writes a new one.
      this.saved = '';
      this.payload = '';
      return;
    }
    let payload = '';
    try {
      payload = draftPayload(restoreDraft(JSON.parse(value).draft));
    } catch {
      // An unreadable record from another tab is still a change to resolve.
    }
    if (payload && payload === this.payload) {
      this.saved = value;
      return;
    }
    this.conflict = true;
  }

  /** Read the other tab's draft for import; afterwards that draft counts as saved here. */
  loadNewer(): Draft {
    const raw = this.storage().getItem(DRAFT_KEY);
    if (!raw) throw new Error('No saved draft is stored in this browser.');
    let draft: Draft;
    try {
      draft = restoreDraft(JSON.parse(raw).draft);
    } catch (error) {
      throw new Error(
        `The newer draft could not be restored: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    this.saved = raw;
    this.payload = draftPayload(draft);
    this.conflict = false;
    return draft;
  }

  /** Export this tab's draft, then make it the stored draft. */
  keep(draft: Draft) {
    this.download(JSON.stringify(draft, null, 2), 'monarch-this-tab.json');
    this.conflict = false;
    this.payload = '';
    this.save(draft, true);
  }

  /** Download the retained text exactly as found, even when it does not parse. */
  exportSaved() {
    this.download(this.kept, 'monarch-saved-draft.json');
  }

  /** Download this tab's draft when it cannot be saved here. */
  exportDraft(draft: Draft) {
    this.download(JSON.stringify(draft, null, 2), 'monarch-draft.json');
  }

  /** Let the next edit replace the retained record. */
  resume() {
    this.retained = false;
    this.kept = '';
    this.payload = '';
  }
}
