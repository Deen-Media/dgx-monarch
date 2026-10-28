import type { Binding, Draft } from './model';

/** JSON records are open to preserve driver/plugin metadata on round-trip. */
export type Workflow = Record<string, any>;
export interface WorkflowEntry {
  id: string;
  /** The file stem; drafts, exports and records keep it. */
  name: string;
  /** The template's Note title, or its humanized stem when it has none. */
  title: string;
  workflow: Workflow;
}
export interface Inspection {
  workflow: Workflow;
  bindings: Binding[];
  errors: string[];
  connections?: GraphConnection[];
}
export interface GraphConnection {
  node_id: string;
  input_name: string;
  type: 'MODEL' | 'DGXM_MODEL' | 'CLIP';
  source: [string, number];
}
/**
 * One recorded seed. A browser reads a JSON number as a double, exact only up to
 * Number.MAX_SAFE_INTEGER (2**53 - 1), while ComfyUI seeds reach 2**64 - 1. A job summary sends a
 * seed up to that limit as a JSON number and a larger one as its exact decimal digits, a string;
 * wire_records.browser_seeds in the companion applies the rule.
 */
export type SeedValue = number | string;
/** Seeds by execution identity, `node id.input name`, as the companion records them. */
export type Seeds = Record<string, SeedValue>;
export interface JobPreview {
  id: string;
  digest: string;
  errors: string[];
  warnings: unknown[];
  requested: Draft;
  effective: {
    /**
     * The seeds the prompt sends: the draft's under Fixed, those chosen at review under Random.
     * A review's seeds are safe integers: Random picks at most 2**53 - 1, and a Fixed draft
     * field refuses a larger value before review.
     */
    seeds?: Seeds;
    style_bindings?: Record<string, string>;
    [key: string]: unknown;
  };
  prompt: Workflow;
  workflow: Workflow;
  model_identities?: Array<{ node_id: string; input: string; name: string }>;
  requested_topology?: Array<{
    node_id: string;
    class_type: string;
    parameters: Record<string, unknown>;
  }>;
  /** Seconds since the epoch; a review can be submitted until it expires. */
  created_at?: number;
  expires_at?: number;
  [key: string]: unknown;
}
export interface Job {
  summary?: boolean;
  id: string;
  state: string;
  created_at: number;
  updated_at?: number;
  idempotency_key?: string;
  submission: {
    requested?: Partial<Draft>;
    /**
     * A summary row carries only the seeds here, by the SeedValue rule, and an imported job's
     * seeds come from its prompt. A full record carries the frozen numbers, which a browser
     * rounds above 2**53 - 1.
     */
    effective?: { seeds?: Seeds | null; [key: string]: unknown };
    workflow?: Workflow;
    project_id?: string;
    [key: string]: unknown;
  };
  progress?: { max?: number; value?: number; node?: string | number } | null;
  /** Driver-clock times from ComfyUI's start and end events, in seconds; null when not reported. */
  started_at?: number | null;
  ended_at?: number | null;
  /** The companion's own clock, like `server_time`, when it first saw the job running. */
  running_observed_at?: number | null;
  error?: { code?: string; message?: string; [key: string]: unknown } | null;
  monarch?: {
    schema_version: number;
    event_count: number;
    render_count?: number;
    render_status_counts?: Record<string, number>;
    renders?: Record<string, RuntimeRender>;
    omitted_render_count?: number;
  };
  [key: string]: unknown;
}
export interface RuntimeRender {
  render_id: string;
  producer_id: string;
  node_id: string;
  fleet_index?: number | null;
  status: string;
  effective_topology?: unknown;
  [key: string]: unknown;
}
export interface Artifact {
  id: string;
  job_id: string;
  kind: 'image' | 'video';
  filename: string;
  created_at: number;
  project_id?: string;
  intermediate?: boolean;
  /** `live_preview` marks the approximate frame a running job streams. */
  source?: string;
  state?: string;
  status?: string;
  metadata?: {
    width?: number;
    height?: number;
    duration?: number;
    frame_rate?: number;
    fps?: number;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}
export interface Layout {
  width: number;
  showTelemetry: boolean;
  showConsole: boolean;
  view?: string;
}
export interface RecordData {
  name?: string;
  notes?: string;
  prompt?: string;
  negative_prompt?: string;
  draft?: Draft;
  workflow?: Workflow;
  selected_artifact_id?: string;
  active_job_ids?: string[];
  layout?: Partial<Layout>;
  [key: string]: unknown;
}
export interface StoredRecord {
  id: string;
  revision: number;
  data: RecordData;
}
export type RecordKind =
  'projects' | 'presets' | 'styles' | 'sessions' | 'workflows' | 'experiments';
export type Records = Partial<Record<RecordKind, StoredRecord[]>>;
export interface OperationParameter {
  choices?: unknown[];
  min?: number;
  max?: number;
  name: string;
  type: string;
  default?: unknown;
  required?: boolean;
}
export interface OperationCapability {
  id: string;
  label: string;
  available: boolean;
  reason?: string;
  description?: string;
  parameters: OperationParameter[];
  requires_confirmation?: boolean;
  implementation?: string;
}
export interface OperationRecord {
  id: string;
  action: string;
  state?: string;
  status?: string;
  created_at: number;
  updated_at?: number;
  digest: string;
  description?: string;
  available: boolean;
  reason?: string;
  requires_confirmation: boolean;
  [key: string]: unknown;
}
/** An operation to open in Operations: from the palette, or a consent a failed job asked for. */
export interface OperationRequest {
  /**
   * A new identity for each request, unique across page loads: asking again applies again, and a
   * review saved in this tab comes back only for the request that made it.
   */
  id: string;
  action: string;
  parameters?: Record<string, unknown>;
  consent?: { jobId: string } & import('./jobFailure').ConsentRequest;
}
export interface Capabilities {
  presentation?: import('./telemetryHistory').Presentation;
  schema_version?: string;
  host?: string;
  instance_id?: string;
  operations?: OperationCapability[];
  limits?: { request_bytes?: number; [key: string]: unknown };
  driver_origin?: string;
  /** The command the page shows when no driver accepts a connection; Studio never runs it. */
  driver_start?: {
    command: string | null;
    source: 'driver_service' | 'driver_command' | 'launcher' | null;
    config: string | null;
  };
  [key: string]: unknown;
}
export interface Telemetry {
  state?: string;
  observed_at?: number;
  sample?: Record<string, any>;
  [key: string]: unknown;
}
export interface Snapshot {
  schema_version?: string;
  instance_id?: string;
  server_time?: number;
  cursor?: number;
  jobs: Job[];
  jobs_truncated?: boolean;
  artifacts: Artifact[];
  operations: OperationRecord[];
  records: Records;
  telemetry: Telemetry;
  driver: {
    state?: string;
    code?: string;
    since?: number;
    timestamp?: number;
    history?: { state: string; code?: string };
  };
}
export interface NumberSchema {
  default?: number;
  min?: number;
  max?: number;
  step?: number;
}
export interface CatalogLora {
  id: string;
  name: string;
  source_node: string;
  compatibility?: string;
}
export interface Catalog {
  loras?: CatalogLora[];
  object_info?: Record<string, { input?: { required?: Record<string, [unknown, NumberSchema?]> } }>;
}
export type InspectWorkflow = (
  workflow: Workflow,
  name: string,
  preserveValues?: boolean,
  /** The bundled workflow id the select shows once this inspection applies. */
  templateId?: string,
) => Promise<void>;
export type ChangeDraft = (changes: Partial<Draft>) => void;
export type ReuseArtifact = (artifact: Artifact, operation?: string) => Promise<void>;
export type SaveRecord = (
  kind: RecordKind,
  data: RecordData,
  id?: string,
  revision?: number,
) => Promise<StoredRecord>;
export type ReportError = (error: unknown) => void;
