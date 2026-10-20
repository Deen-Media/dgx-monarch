import { StudioClient } from './api';
import { savedToken } from './sessionRecovery';
import { consumeEvents, boundedLog, formatTime } from './model';
import type { Capabilities, Catalog, Snapshot, WorkflowEntry } from './contracts';

import { appendTelemetry, type MetricSeries } from './telemetryHistory';

const TOKEN_KEY = 'dgxm.studio.token.v1';

/** Owns companion transport and reconciled host state, never unsent draft edits. */
export class Connection {
  token = $state(savedToken(TOKEN_KEY));
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

  async refresh() {
    if (!this.client.token) return;
    const client = this.client,
      generation = this.generation,
      sequence = ++this.snapshotSequence;
    const result = await client.request<Snapshot>('/snapshot');
    if (generation !== this.generation || sequence !== this.snapshotSequence) return;
    if (
      result.instance_id === this.instance &&
      Number(result.server_time) < Number(this.snapshot.server_time)
    )
      return;
    const newInstance = result.instance_id !== this.instance;
    if (this.instance && result.instance_id !== this.instance) {
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
  }

  async connect() {
    this.stop();
    const generation = this.generation;
    this.state = 'reconnecting';
    const client = new StudioClient(this.token.trim());
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
      try {
        const catalog = await client.request<Catalog>('/catalog');
        if (generation !== this.generation) return;
        this.catalog = catalog;
      } catch {
        if (generation !== this.generation) return;
        this.catalog = {};
      }
      try {
        sessionStorage.setItem(TOKEN_KEY, this.client.token);
      } catch {
        // Authentication works in memory; mutation owners separately require durable identities.
      }
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
