import { untrack } from 'svelte';

/** A live record reloads at most once in this period; the last change always loads. */
export const LIVE_RELOAD_MS = 5000;

export interface RecordKey {
  id: string;
  /** Changes when the summary row shows that the full record changed. */
  revision?: string;
  /** True while the record can still change, which throttles reloads. */
  live?: boolean;
}
export interface KeyedRecordOptions {
  throttleMs?: number;
  now?: () => number;
  schedule?: (run: () => void, delay: number) => unknown;
  cancel?: (handle: unknown) => void;
}

/**
 * Holds the full record behind a summary row. Snapshot polls replace every row with a new
 * object, so only a new id clears the record; a new revision of the same id reloads in place.
 */
export class KeyedRecord<T> {
  value = $state<T | null>(null);
  error = $state('');
  loading = $state(false);
  #read: (id: string) => Promise<T>;
  #throttle: number;
  #now: () => number;
  #schedule: (run: () => void, delay: number) => unknown;
  #cancel: (handle: unknown) => void;
  #id = '';
  #revision = '';
  #ticket = 0;
  #started = -Infinity;
  #timer: unknown = null;

  constructor(read: (id: string) => Promise<T>, options: KeyedRecordOptions = {}) {
    this.#read = read;
    this.#throttle = options.throttleMs ?? LIVE_RELOAD_MS;
    this.#now = options.now ?? (() => Date.now());
    this.#schedule = options.schedule ?? ((run, delay) => setTimeout(run, delay));
    this.#cancel =
      options.cancel ?? ((handle) => clearTimeout(handle as ReturnType<typeof setTimeout>));
  }

  /** Follow the current key. Identical keys do nothing, so idle polls never refetch. */
  follow(id: string, revision = '', live = false) {
    untrack(() => {
      if (id !== this.#id) {
        this.#id = id;
        this.#revision = revision;
        this.#ticket++;
        this.#stopTimer();
        this.value = null;
        this.error = '';
        this.loading = false;
        if (id) this.#fetch();
        return;
      }
      if (!id || revision === this.#revision) return;
      this.#revision = revision;
      const wait = this.#started + this.#throttle - this.#now();
      if (!live || wait <= 0) {
        this.#stopTimer();
        this.#fetch();
      } else if (this.#timer === null)
        this.#timer = this.#schedule(() => {
          this.#timer = null;
          this.#fetch();
        }, wait);
    });
  }

  /** Reload the current record now and keep it on screen while the request runs. */
  refresh() {
    untrack(() => {
      this.#stopTimer();
      if (this.#id) this.#fetch();
    });
  }

  /** Drop pending work when the owner unmounts. */
  dispose() {
    this.#ticket++;
    this.#stopTimer();
  }

  #stopTimer() {
    if (this.#timer !== null) this.#cancel(this.#timer);
    this.#timer = null;
  }

  async #fetch() {
    const ticket = ++this.#ticket,
      id = this.#id;
    this.#started = this.#now();
    this.loading = true;
    try {
      const result = await this.#read(id);
      if (ticket !== this.#ticket) return;
      this.value = result;
      this.error = '';
    } catch (cause) {
      if (ticket === this.#ticket)
        this.error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (ticket === this.#ticket) this.loading = false;
    }
  }
}

/** Create a record during component setup and follow a reactive key. */
export function keyedRecord<T>(
  read: (id: string) => Promise<T>,
  key: () => RecordKey,
  options: KeyedRecordOptions = {},
): KeyedRecord<T> {
  const record = new KeyedRecord(read, options);
  // Derived primitives stop identical snapshot rows from re-running the effect.
  const id = $derived(key().id);
  const revision = $derived(key().revision ?? '');
  const live = $derived(key().live ?? false);
  $effect(() => record.follow(id, revision, live));
  $effect(() => () => record.dispose());
  return record;
}
