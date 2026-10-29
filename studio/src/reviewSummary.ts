import { bindingType, fieldLabel, type Binding } from './model.ts';
import { seedText } from './recordedSettings.ts';
import type { JobPreview } from './contracts';

export interface ReviewText {
  label: string;
  /** The text the prompt sends. */
  text: string;
  /** The draft's own text, shown only when the review changed it, as styles do. */
  draft?: string;
}

/** A warning the backend reports for one job, and the words of the box that accepts it. */
export interface ReviewWarning {
  text: string;
  accept: string;
}

/** How the status line reads: ready to generate, waiting for a decision, or blocked. */
export type ReviewTone = 'ready' | 'decide' | 'blocked';

export interface ReviewSummary {
  tone: ReviewTone;
  status: string;
  workflow: string;
  models: string[];
  seeds: string;
  seedSource: string;
  texts: ReviewText[];
  styles: string;
  topology: string[];
  warnings: ReviewWarning[];
  expiry: string;
}

/** What Studio checked before the job, and what the backend checks only when the job runs. */
export const CHECKS =
  "Studio checked these values against the driver's inputs. The backend checks memory, topology, identity and consent when the job runs.";

const clockTime = (seconds: number) => new Date(seconds * 1000).toLocaleTimeString();

/**
 * The warnings a person must decide on before the job runs, each with its own box. A warning
 * that brings an `acknowledgement` sentence uses it; any other is accepted in its own words.
 */
export function reviewWarnings(preview: Pick<JobPreview, 'warnings'>): ReviewWarning[] {
  return (preview.warnings ?? []).map((warning) => {
    const fields = (warning ?? {}) as { message?: unknown; acknowledgement?: unknown };
    const text =
      typeof warning === 'string'
        ? warning
        : typeof fields.message === 'string' && fields.message
          ? fields.message
          : JSON.stringify(warning);
    const own = typeof fields.acknowledgement === 'string' ? fields.acknowledgement : '';
    return { text, accept: own ? `${text} ${own}` : `Generate anyway: ${text}` };
  });
}

/** How many warnings still wait for their box. */
function undecided(preview: Pick<JobPreview, 'warnings'>, acknowledged: boolean[]): number {
  return (preview.warnings ?? []).filter((_, index) => !acknowledged[index]).length;
}

/** Whether nothing in the review needs a person: no error to fix and no warning left open. */
export function reviewDecided(
  preview: Pick<JobPreview, 'errors' | 'warnings'>,
  acknowledged: boolean[] = [],
): boolean {
  return !preview.errors?.length && !undecided(preview, acknowledged);
}

/**
 * Whether the companion's clock `now` has reached the review's expiry. A clock not yet read is
 * null and never expires a review; the companion's own check at submit is authoritative.
 */
export function reviewExpired(preview: JobPreview, now: number | null): boolean {
  return now !== null && typeof preview.expires_at === 'number' && now >= preview.expires_at;
}

/** Whether the review can start the job, in the words the dialog shows above the summary. */
export function reviewStatus(
  preview: JobPreview,
  pending: boolean,
  busy: boolean,
  now: number | null = null,
  acknowledged: boolean[] = [],
): { tone: ReviewTone; status: string } {
  if (preview.errors?.length)
    return { tone: 'blocked', status: `Blocked: ${preview.errors.join(' ')}` };
  if (pending)
    return {
      tone: 'blocked',
      status: busy
        ? 'Submitting. Wait for the outcome.'
        : 'Blocked: an earlier submission is unresolved.',
    };
  if (reviewExpired(preview, now))
    return {
      tone: 'blocked',
      status: `This review expired at ${clockTime(preview.expires_at!)}. Close it and generate again for a fresh one.`,
    };
  const open = undecided(preview, acknowledged);
  if (open)
    return {
      tone: 'decide',
      status:
        open === 1
          ? 'One warning needs your decision before this job runs.'
          : `${open} warnings need your decision before this job runs.`,
    };
  return { tone: 'ready', status: 'Ready to generate.' };
}

/** Why Generate is off, or '' when it is on. The most basic block comes first. */
export function submitBlock(
  preview: JobPreview,
  state: { acknowledged?: boolean[]; busy: boolean; pending: boolean; now?: number | null },
): string {
  if (preview.errors?.length)
    return 'Generate is off until the problems above are fixed. Return to the draft and correct them.';
  if (state.pending)
    return state.busy
      ? 'Generate is off while this submission is sent.'
      : 'Generate is off until the earlier submission is reconciled. Use Reconcile request in the banner.';
  if (reviewExpired(preview, state.now ?? null))
    return 'Generate is off for an expired review. Close it and generate again.';
  if (state.busy) return 'Generate is off while Studio finishes the current request.';
  if (undecided(preview, state.acknowledged ?? []))
    return 'Generate is off until you tick each warning above, or adjust the draft.';
  return '';
}

/** The timers the review clock uses; a test passes fakes. */
export interface ReviewTimers {
  now: () => number;
  every: (run: () => void, ms: number) => unknown;
  after: (run: () => void, ms: number) => unknown;
  cancel: (every: unknown, after: unknown) => void;
}
const BROWSER_TIMERS: ReviewTimers = {
  now: () => performance.now(),
  every: (run, ms) => setInterval(run, ms),
  after: (run, ms) => setTimeout(run, ms),
  cancel: (every, after) => {
    clearInterval(every as ReturnType<typeof setInterval>);
    clearTimeout(after as ReturnType<typeof setTimeout>);
  },
};

/**
 * Keep `set` on the companion's clock while a review is open: the review's created_at, the
 * companion's clock when it made the review, advanced by the browser's monotonic timer since the
 * review arrived at `arrived`. The browser's wall clock never enters. The clock reads at once,
 * then every minute and exactly at expires_at. A reply's travel time leaves it slightly behind
 * the companion, whose own expiry check at submit is authoritative. Returns the stop function.
 */
export function startReviewClock(
  review: Pick<JobPreview, 'created_at' | 'expires_at'>,
  arrived: number,
  set: (now: number) => void,
  timers: ReviewTimers = BROWSER_TIMERS,
): () => void {
  const { created_at: created, expires_at: expires } = review;
  if (typeof created !== 'number' || typeof expires !== 'number') return () => {};
  const read = () => created + (timers.now() - arrived) / 1000;
  set(read());
  const every = timers.every(() => set(read()), 60_000);
  // At the expiry moment the clock reads expires_at at least, so rounding cannot hold it back.
  const after = timers.after(
    () => set(Math.max(read(), expires)),
    Math.max(0, (expires - read()) * 1000),
  );
  return () => timers.cancel(every, after);
}

/**
 * The text inputs a reader checks before submitting: the multi-line ones, such as prompts, and
 * any input a style rewrote. A workflow with no multi-line text input shows every text input.
 */
function reviewTexts(preview: JobPreview, bindings: Binding[]): ReviewText[] {
  const styled = new Set(Object.values(preview.effective?.style_bindings ?? {}));
  const strings = bindings.filter((b) => b.resolved !== false && bindingType(b) === 'STRING');
  const prompts = strings.filter((b) => b.schema?.multiline || styled.has(b.id));
  return (prompts.length ? prompts : strings).flatMap((b) => {
    const text = preview.prompt?.[b.execution_node_id ?? b.node_id ?? '']?.inputs?.[b.input_name!];
    if (typeof text !== 'string') return [];
    const draft = preview.requested?.values?.[b.id] ?? b.value;
    return [
      {
        label: `${fieldLabel(b)} · node ${b.node_id}`,
        text,
        ...(typeof draft === 'string' && draft !== text ? { draft } : {}),
      },
    ];
  });
}

export function reviewSummary(
  preview: JobPreview,
  bindings: Binding[],
  pending: boolean,
  busy: boolean,
  now: number | null = null,
  acknowledged: boolean[] = [],
): ReviewSummary {
  const requested = preview.requested ?? ({} as Partial<JobPreview['requested']>);
  const seeds = preview.effective?.seeds ?? {};
  const random = requested.seed_policy === 'random';
  const order = (preview.effective?.styles as { style_order?: string[] } | undefined)?.style_order;
  const names = new Map(
    (requested.styles ?? []).map((style) => {
      const { id, name } = style as { id?: string; name?: string };
      return [id, name ?? id];
    }),
  );
  const minutes =
    typeof preview.expires_at === 'number' && typeof preview.created_at === 'number'
      ? Math.round((preview.expires_at - preview.created_at) / 60)
      : null;
  return {
    ...reviewStatus(preview, pending, busy, now, acknowledged),
    workflow: [requested.workflowName || 'Unnamed workflow', requested.operation]
      .filter(Boolean)
      .join(' · '),
    models: (preview.model_identities ?? []).map(
      (model) => `${model.name} (${model.input}, node ${model.node_id})`,
    ),
    seeds: seedText(seeds) || 'This workflow has no seed input.',
    seedSource: !Object.keys(seeds).length
      ? ''
      : random
        ? 'Chosen for this job by the Random policy. Your seed fields keep their values.'
        : 'Taken from your seed fields by the Fixed policy.',
    texts: reviewTexts(preview, bindings),
    styles: order?.length
      ? `Styles applied in this order: ${order.map((id) => names.get(id) ?? id).join(', ')}.`
      : '',
    topology: (preview.requested_topology ?? [])
      .filter((entry) => Object.keys(entry.parameters ?? {}).length)
      .map(
        (entry) =>
          `${entry.class_type} (node ${entry.node_id}): ${Object.entries(entry.parameters)
            .map(
              ([key, value]) =>
                `${key} ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`,
            )
            .join(', ')}`,
      ),
    warnings: reviewWarnings(preview),
    expiry:
      minutes === null
        ? ''
        : `This review ${reviewExpired(preview, now) ? 'expired' : 'expires'} at ${clockTime(preview.expires_at!)}, ${minutes} minutes after Studio made it, because it records the driver's runtime as it was then.`,
  };
}
