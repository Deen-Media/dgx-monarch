import { ApiError, StudioClient } from './api';
import { savedToken } from './sessionRecovery';
import { launchToken, tokenFromInput } from './launchToken';
import { driverWaiting, unreachableText } from './firstRun';
import { consumeEvents, boundedLog, formatTime, driverStatus } from './model';
import type { Capabilities, Catalog, Snapshot, WorkflowEntry } from './contracts';

import { appendTelemetry, type MetricSeries } from './telemetryHistory';

export { errorText } from './firstRun';

const TOKEN_KEY = 'dgxm.studio.token.v1';
/** The connect notice for a driver catalog that could not be read starts with this text. */
export const CATALOG_NOTICE = 'Companion connected. The driver catalog could not be read:';
/** A token this tab used before was refused: the companion most likely restarted. */
const restarted = (message: string) =>
  `${message} The companion may have restarted with a new owner token. Enter the current token to reconnect, or run dgxm studio open on the Studio host for a new sign-in link.`;

/** Owns companion transport and reconciled host state, never unsent draft edits. */
export class Connection {
  /** The token of the sign-in link this tab opened with, until a connect succeeds. */
  private link = launchToken();
  /** The token this tab restored or last saw accepted; a 401 for it means a restart. */
  private accepted = savedToken(TOKEN_KEY);
  /** A link token wins over an older stored one; storage is written only after it works. */
  token = $state(this.link || this.accepted);
  /** Whether the token in use came from a sign-in link; cleared after a successful connect. */
  fromLink = !!this.link;
  client = $state(new StudioClient(this.token));
  connected = $state(false);
  state = $state('unavailable');
  lastSeen = $state('');
  error = $state('');
  capabilities = $state<Capabilities>({});
  workflows = $state<WorkflowEntry[]>([]);
  catalog = $state<Catalog>({});
  /** Why the last driver catalog read failed, or '' when it succeeded. */
  catalogError = $state('');
  /** The error code of the last failed catalog read, or '' when it succeeded. */
  catalogCode = $state('');
  /** The catalog read in flight; a refresh joins it rather than starting another. */
  catalogRead?: Promise<void>;
  snapshot = $state<Snapshot>({
    jobs: [],
    artifacts: [],
    operations: [],
    records: {},
    telemetry: {},
    driver: {},
  });
  logs = $state<string[]>([]);
  metricHistory = $state<MetricSeries[]>([]);
  hidden = $state(document.hidden);
  cursor = 0;
  private instance?: string;
  private timer?: ReturnType<typeof setTimeout>;
  private stopped = false;
  private backoff = 1500;
  private polling = false;
  private generation = 0;
  private snapshotSequence = 0;
  /** Whether the last snapshot showed the driver connected; undefined until one arrives. */
  private driverUp?: boolean;
  /** Whether the catalog was already read again during this driver connection. */
  private catalogRetried = false;

  async refresh() {
    if (!this.client.token) return;
    const client = this.client,
      generation = this.generation,
      sequence = ++this.snapshotSequence;
    const result = await client.request<Snapshot>('/snapshot');
    // The request sequence orders responses; the host clock may step backward.
    if (generation !== this.generation || sequence !== this.snapshotSequence) return;
    const newInstance = result.instance_id !== this.instance;
    const restarted = !!this.instance && newInstance;
    if (restarted) {
      this.cursor = 0;
      this.logs = [];
      this.metricHistory = [];
      const capabilities = await client.request<Capabilities>('/capabilities');
      if (generation !== this.generation || sequence !== this.snapshotSequence) return;
      this.capabilities = capabilities;
    }
    this.instance = result.instance_id;
    this.snapshot = result;
    this.metricHistory = appendTelemetry(
      this.metricHistory,
      result.telemetry,
      this.capabilities.presentation?.metrics ?? [],
    );
    // A snapshot reconciles state, but does not consume the event pages between
    // the last replay cursor and its latest cursor.
    if (newInstance) this.cursor = result.cursor ?? 0;
    this.lastSeen = formatTime(result.server_time);
    this.state = 'connected';
    this.connected = true;
    this.error = '';
    this.watchCatalog(restarted, result.driver?.state === 'connected');
  }

  /**
   * Read the catalog again when the companion restarted or the driver connected again, since
   * either may serve other files, and once while connected after a failed read. A read that
   * keeps failing on a connected driver waits for the next connection or an explicit refresh.
   */
  private watchCatalog(restarted: boolean, up: boolean) {
    const cameUp = up && this.driverUp === false;
    this.driverUp = up;
    if (!up) this.catalogRetried = false;
    else if (restarted || cameUp || (this.catalogError && !this.catalogRetried)) {
      this.catalogRetried = true;
      void this.refreshCatalog();
    }
  }

  /** Read the driver catalog again: its LoRA files and the node schemas LoRA controls use. */
  refreshCatalog(): Promise<void> {
    if (this.catalogRead) return this.catalogRead;
    const read = this.readCatalog(this.client, this.generation, false).finally(() => {
      if (this.catalogRead === read) this.catalogRead = undefined;
    });
    this.catalogRead = read;
    return read;
  }

  private async readCatalog(client: StudioClient, generation: number, clear: boolean) {
    try {
      const catalog = await client.request<Catalog>('/catalog');
      if (generation !== this.generation) return;
      this.catalog = catalog;
      this.catalogError = '';
      this.catalogCode = '';
    } catch (error) {
      if (generation !== this.generation) return;
      // At connect nothing earlier applies; a failed refresh keeps the last list it read.
      if (clear) this.catalog = {};
      this.catalogError = error instanceof Error ? error.message : String(error);
      this.catalogCode = error instanceof ApiError ? (error.code ?? '') : '';
    }
  }

  /** Whether nothing accepts a connection at the driver origin, so the page says how to start it. */
  get driverNotRunning(): boolean {
    return driverWaiting(this.snapshot.driver, this.catalogCode);
  }

  async connect() {
    this.stop();
    const generation = this.generation;
    this.state = 'reconnecting';
    this.error = '';
    // The form also takes a whole sign-in link, pasted.
    const typed = this.token.trim(),
      token = tokenFromInput(typed);
    this.fromLink = typed.includes('#token=') || (!!this.link && token === this.link);
    const client = new StudioClient(token);
    this.client = client;
    try {
      const cap = await client.request<Capabilities>('/capabilities');
      if (generation !== this.generation) return;
      if (String(cap.schema_version ?? '').split('.')[0] !== '1')
        throw new Error(
          `Incompatible companion schema ${cap.schema_version ?? 'unknown'}; Studio requires 1.x.`,
        );
      this.capabilities = cap;
      const workflows = await client.request<{ workflows: WorkflowEntry[] }>('/workflows');
      if (generation !== this.generation) return;
      this.workflows = workflows.workflows ?? [];
      [this.driverUp, this.catalogRetried, this.catalogRead] = [undefined, false, undefined];
      await this.readCatalog(client, generation, true);
      if (generation !== this.generation) return;
      try {
        sessionStorage.setItem(TOKEN_KEY, this.client.token);
      } catch {
        // Authentication works in memory; mutation owners separately require durable identities.
      }
      [this.accepted, this.link, this.fromLink] = [token, '', false];
      await this.refresh();
      if (generation !== this.generation) return;
      this.backoff = 1500;
      this.stopped = false;
      clearTimeout(this.timer);
      this.timer = setTimeout(() => void this.poll(), this.backoff);
    } catch (error) {
      if (generation !== this.generation) return;
      this.connected = false;
      this.state = 'unavailable';
      if (
        error instanceof ApiError &&
        error.status === 401 &&
        !this.fromLink &&
        token === this.accepted
      )
        throw new ApiError(401, restarted(error.message), error.code);
      throw error;
    }
  }

  private async poll() {
    if (this.stopped || this.polling) return;
    this.polling = true;
    const generation = this.generation,
      client = this.client;
    if (!this.hidden && this.client.token) {
      try {
        const result = await client.request('/events?after=' + this.cursor);
        if (generation !== this.generation) return;
        const accepted = consumeEvents(result.reset ? 0 : this.cursor, result.events ?? []);
        this.cursor = Math.max(accepted.cursor, Number(result.cursor) || 0);
        this.logs = boundedLog([
          ...this.logs,
          ...(result.reset
            ? ['Event replay gap: some earlier events are no longer retained.']
            : []),
          ...accepted.events.map(
            (event) =>
              `${event.timestamp ?? event.server_time ?? ''} ${event.kind ?? event.type ?? 'event'} ${JSON.stringify(event.data ?? event.payload ?? event)}`,
          ),
        ]);
        await this.refresh();
        if (generation !== this.generation) return;
        this.backoff = 1500;
      } catch (error) {
        if (generation !== this.generation) return;
        if (error instanceof ApiError && error.status === 401) {
          // A restarted companion issues a new token; show the token form again.
          this.stop();
          this.connected = false;
          this.state = 'unavailable';
          this.error = restarted(error.message);
          return;
        }
        this.state = 'stale';
        // Nothing answered: the companion stopped, or the tunnel to it closed.
        this.error = error instanceof ApiError ? error.message : unreachableText(location.host);
        this.backoff = Math.min(30000, this.backoff * 2);
      }
    }
    this.polling = false;
    if (!this.stopped) this.timer = setTimeout(() => void this.poll(), this.backoff);
  }

  /** Explain a connection that works but cannot drive generation yet, or ''. */
  degradedNotice(): string {
    // The driver-not-running panel replaces this notice.
    if (this.driverNotRunning) return '';
    const driver = this.snapshot.driver;
    const reason = /[.!?]$/.test(this.catalogError) ? this.catalogError : `${this.catalogError}.`;
    if (this.catalogError)
      return `${CATALOG_NOTICE} ${reason} Management controls remain available.`;
    return driver.state === 'connected'
      ? ''
      : `Companion connected. The driver is ${driverStatus(driver)}. Management controls remain available.`;
  }

  observeVisibility() {
    const listener = () => {
      this.hidden = document.hidden;
      if (!this.hidden && this.connected) {
        clearTimeout(this.timer);
        void this.poll();
      }
    };
    document.addEventListener('visibilitychange', listener);
    return () => {
      document.removeEventListener('visibilitychange', listener);
      this.stop();
    };
  }

  stop() {
    this.generation++;
    this.polling = false;
    this.stopped = true;
    clearTimeout(this.timer);
  }
  disconnect() {
    this.stop();
    try {
      sessionStorage.removeItem(TOKEN_KEY);
    } catch {
      // Always disconnect this tab even if token persistence is unavailable.
    }
    this.token = '';
    this.client = new StudioClient('');
    this.connected = false;
    this.state = 'unavailable';
    this.instance = undefined;
    this.cursor = 0;
    this.logs = [];
    this.metricHistory = [];
    this.capabilities = {};
    this.catalog = {};
    this.catalogError = '';
    this.catalogCode = '';
    [this.accepted, this.link, this.fromLink] = ['', '', false];
    [this.driverUp, this.catalogRetried, this.catalogRead] = [undefined, false, undefined];
    this.error = '';
    this.workflows = [];
    this.snapshot = {
      jobs: [],
      artifacts: [],
      operations: [],
      records: {},
      telemetry: {},
      driver: {},
    };
  }
}
