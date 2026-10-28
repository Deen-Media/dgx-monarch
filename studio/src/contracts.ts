import type { Binding, Draft } from './model';

/** JSON records are open to preserve driver/plugin metadata on round-trip. */
export type Workflow = Record<string, any>;
export interface WorkflowEntry {
  id: string;
  name: string;
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
export interface JobPreview {
  id: string;
  digest: string;
  errors: string[];
  warnings: unknown[];
  requested: Draft;
  effective: Record<string, unknown>;
  prompt: Workflow;
  workflow: Workflow;
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
