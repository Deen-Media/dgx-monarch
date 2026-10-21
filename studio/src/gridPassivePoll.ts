import { ApiError } from './api.ts';
import type { Grid } from './gridState';

export const PASSIVE_INTERVAL_MS = 3000;
export const PASSIVE_MAX_INTERVAL_MS = 60000;

export interface PassivePollOptions {
  /** Read the summary of one durable grid. */
  read: (id: string) => Promise<Grid>;
  /** The grid on screen now. */
  target: () => string | undefined;
  /** The view's load generation; any user action advances it. */
  ticket: () => number;
  /** True while the page is hidden or an action is running. */
  paused: () => boolean;
  apply: (id: string, grid: Grid) => void;
  /** The companion has no such grid; nothing against it can have been accepted here. */
  missing: (id: string) => void;
  /** Passive status text, kept apart from action errors; an empty string clears it. */
  report: (text: string) => void;
  interval?: number;
  maxInterval?: number;
  schedule?: (run: () => void, delay: number) => unknown;
  cancel?: (handle: unknown) => void;
}

/**
 * A grid dispatch answered with not_found settles its receipt. The companion looks up the request
 * ID before it reads the preview, so this reply means it accepted nothing under that ID.
 */
export function dispatchAbsent(cause: unknown): string | null {
  return cause instanceof ApiError && cause.status === 404 && cause.code === 'not_found'
    ? 'This companion has no grid accepted under this request ID and no longer holds the reviewed preview, so it submitted nothing. If you used another state directory, check for the grid there. Validate every cell again to submit here.'
    : null;
}

/**
 * Background status refresh for the grid on screen. Failures back off, a reply for another grid
 * or an older load generation is dropped on both paths, and a 404 settles the grid as absent
 * because the companion never deletes grid runs.
 */
export class GridPassivePoll {
  #options: PassivePollOptions;
  #failures = 0;
  #timer: unknown = null;
  #stopped = true;

  constructor(options: PassivePollOptions) {
    this.#options = options;
  }

  get delay(): number {
    const base = this.#options.interval ?? PASSIVE_INTERVAL_MS;
    return Math.min(
      this.#options.maxInterval ?? PASSIVE_MAX_INTERVAL_MS,
      base * 2 ** this.#failures,
    );
  }

  start() {
    this.#stopped = false;
    void this.tick();
  }

  stop() {
    this.#stopped = true;
    if (this.#timer !== null)
      (this.#options.cancel ?? ((handle) => clearTimeout(handle as number)))(this.#timer);
    this.#timer = null;
  }

  async tick() {
    this.#timer = null;
    const options = this.#options,
      id = options.target();
    if (id && !options.paused()) {
      const ticket = options.ticket();
      const current = () =>
        !this.#stopped && ticket === options.ticket() && options.target() === id;
      try {
        const grid = await options.read(id);
        if (current()) {
          this.#failures = 0;
          options.report('');
          options.apply(id, grid);
        }
      } catch (cause) {
        if (!current()) {
          // A failure for another grid or an earlier generation says nothing about this one.
        } else if (cause instanceof ApiError && cause.status === 404) {
          this.#failures = 0;
          options.missing(id);
        } else {
          this.#failures++;
          const reason = cause instanceof Error ? cause.message : String(cause);
          options.report(
            `Background grid status refresh failed, so Studio tries again in ${Math.round(this.delay / 1000)} s: ${reason}`,
          );
        }
      }
    }
    if (!this.#stopped)
      this.#timer = (this.#options.schedule ?? ((run, delay) => setTimeout(run, delay)))(
        () => void this.tick(),
        this.delay,
      );
  }
}
