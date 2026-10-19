import { StudioClient } from './api';
import { consumeEvents, boundedLog, formatTime } from './model';
import type { Capabilities, Catalog, Snapshot, WorkflowEntry } from './contracts';

const TOKEN_KEY = 'dgxm.studio.token.v1';

/** Owns companion transport and reconciled host state, never unsent draft edits. */
export class Connection {
  token = $state(sessionStorage.getItem(TOKEN_KEY) ?? '');
  client = $state(new StudioClient(this.token));
  connected = $state(false);
  state = $state('unavailable');
  lastSeen = $state('');
  error = $state('');
  capabilities = $state<Capabilities>({});
  workflows = $state<WorkflowEntry[]>([]);
  catalog = $state<Catalog>({});
  snapshot = $state<Snapshot>({
    jobs: [],
    artifacts: [],
    operations: [],
    records: {},
    telemetry: {},
    driver: {},
  });
  logs = $state<string[]>([]);
  hidden = $state(document.hidden);
  cursor = 0;
  private instance?: string;
  private timer?: ReturnType<typeof setTimeout>;
  private stopped = false;
  private backoff = 1500;
  private polling = false;

  async refresh() {
    if (!this.client.token) return;
    const result = await this.client.request<Snapshot>('/snapshot');
    if (
      result.instance_id === this.instance &&
      Number(result.server_time) < Number(this.snapshot.server_time)
    )
      return;
    if (this.instance && result.instance_id !== this.instance) {
      this.cursor = 0;
      this.logs = [];
      this.capabilities = await this.client.request('/capabilities');
    }
    this.instance = result.instance_id;
    this.snapshot = result;
    this.cursor = Math.max(this.cursor, result.cursor ?? 0);
    this.lastSeen = formatTime(result.server_time);
    this.state = 'connected';
    this.connected = true;
    this.error = '';
  }

  async connect() {
    this.state = 'reconnecting';
    this.client = new StudioClient(this.token.trim());
    try {
      const cap = await this.client.request<Capabilities>('/capabilities');
      if (String(cap.schema_version ?? '').split('.')[0] !== '1')
        throw new Error(
          `Incompatible companion schema ${cap.schema_version ?? 'unknown'}; Studio requires 1.x.`,
        );
      this.capabilities = cap;
      this.workflows =
        (await this.client.request<{ workflows: WorkflowEntry[] }>('/workflows')).workflows ?? [];
      try {
        this.catalog = await this.client.request('/catalog');
      } catch {
        this.catalog = {};
      }
      sessionStorage.setItem(TOKEN_KEY, this.client.token);
      await this.refresh();
      this.backoff = 1500;
      this.stopped = false;
      clearTimeout(this.timer);
      this.timer = setTimeout(() => void this.poll(), this.backoff);
    } catch (error) {
      this.connected = false;
      this.state = 'unavailable';
      throw error;
    }
  }

  private async poll() {
    if (this.stopped || this.polling) return;
    this.polling = true;
    if (!this.hidden && this.client.token) {
      try {
        const result = await this.client.request('/events?after=' + this.cursor);
        const accepted = consumeEvents(this.cursor, result.events ?? []);
        this.cursor = Math.max(accepted.cursor, Number(result.cursor) || 0);
        this.logs = boundedLog([
          ...this.logs,
          ...accepted.events.map(
            (event) =>
              `${event.timestamp ?? event.server_time ?? ''} ${event.kind ?? event.type ?? 'event'} ${JSON.stringify(event.data ?? event.payload ?? event)}`,
          ),
        ]);
        await this.refresh();
        this.backoff = 1500;
      } catch (error) {
        this.state = 'stale';
        this.error = error instanceof Error ? error.message : String(error);
        this.backoff = Math.min(30000, this.backoff * 2);
      }
    }
    this.polling = false;
    if (!this.stopped) this.timer = setTimeout(() => void this.poll(), this.backoff);
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
    this.stopped = true;
    clearTimeout(this.timer);
  }
  disconnect() {
    this.stop();
    sessionStorage.removeItem(TOKEN_KEY);
    this.token = '';
    this.client = new StudioClient('');
    this.connected = false;
    this.state = 'unavailable';
  }
}
