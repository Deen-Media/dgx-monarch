import type { Job, OperationCapability, OperationRecord, OperationRequest } from './contracts';

/**
 * Read a failed job's `error` for display. Every field comes from the recorded text: the class
 * tag that dgx_monarch.refusal writes, the remedy sentence and the consent block that the
 * companion's bounded_refusal keeps, and the first sentence. Nothing here guesses a cause, and a
 * message without that structure shows whole up to `TEXT_LIMIT`. The Python suite runs this
 * module on real refusals, so a change on either side fails a test.
 */

/** The refusal classes and the titles dgx_monarch.refusal.CLASS_TITLES gives them. */
export const CLASS_TITLES = {
  P: 'physics',
  C: 'capacity-protective',
  U: 'unproven correctness',
  K: 'known-wrong math',
} as const;
export type RefusalClass = keyof typeof CLASS_TITLES;

/** What a consent descriptor asks for, in the fields Studio shows. */
export interface ConsentRequest {
  kind: string;
  consentId: string;
  /** The model the request names, relative to ComfyUI's model folder. */
  unetName: string;
  humanReason: string;
  /** The label of the ComfyUI panel button that grants it. */
  panelAction: string;
}

export interface JobFailure {
  code: string;
  refusalClass: RefusalClass | null;
  guard: string | null;
  waivable: boolean;
  /** The first sentence after the class tag, or of the message when it has no tag. */
  headline: string;
  /** One line for the viewer status line and the Queue row. */
  summary: string;
  /** The message without its machine-readable consent block, up to `TEXT_LIMIT`. */
  text: string;
  /** The driver's panel and headless instructions, or ''. */
  remedy: string;
  /** The docs/TROUBLESHOOTING.md entry the message cites, or null. */
  troubleshooting: number | null;
  consent: ConsentRequest | null;
}

export const TEXT_LIMIT = 4000;
export const HEADLINE_LIMIT = 200;
export const CONSENT_OPERATION = 'consent.apply';
export const DOCS = 'https://github.com/Deen-Media/dgx-monarch/blob/main/docs';
export const CONSENT_HELP = `${DOCS}/STUDIO.md#read-a-failure-and-grant-a-consent`;
/** docs/TROUBLESHOOTING.md; its sections have no anchor by number, so a link opens the page. */
export const TROUBLESHOOTING = `${DOCS}/TROUBLESHOOTING.md`;
/** The markers of dgx_monarch.consent_descriptor, and the bound on the JSON between them. */
export const DESCRIPTOR_BEGIN = '<<<DGXM-CONSENT-V1 ';
export const DESCRIPTOR_END = ' DGXM-CONSENT-END>>>';
const DESCRIPTOR_BYTES = 4096;
const REQUIRED = [
  'kind',
  'consent_id',
  'unet_name',
  'file_identity',
  'context_fingerprint',
  'human_reason',
  'evidence',
  'panel_action',
  'env_fallback',
];
// dgx_monarch.consent_descriptor.valid_kind: the rescue kind, or a class K kind whose short name
// is letters, digits, "-", "_" or ".".
const KIND = /^(?:rescue-slab|waive-known-wrong:[\p{L}\p{N}._-]{1,48})$/u;
// dgx_monarch.refusal._TAG_RE; a guard and a waivable flag come together or not at all.
const TAG = /\[dgxm:([PCUK])(?: guard=([A-Za-z0-9_:.-]{1,64}))?(?: waivable=([01]))?\]/g;
// The remedy sentences dgx_monarch.refusal writes, as job_observations._REMEDY reads them.
const REMEDY =
  / Open the DGX Monarch panel and click "[^"\n]{1,200}" on the card for this model, then queue the render again\. Headless: set [A-Za-z0-9_]{1,64}=\S{1,64} on the driver and run again\.(?: See docs\/TROUBLESHOOTING\.md #(\d{1,4})\.)?| See docs\/TROUBLESHOOTING\.md #(\d{1,4})\./g;
const CITATION = /TROUBLESHOOTING\.md #(\d{1,4})(?!\d)/g;
const NO_MANAGEMENT =
  'This companion was started without --allow-management, so Studio cannot grant this consent. Grant it in the ComfyUI panel, or start Studio again with --allow-management.';

/** The first sentence of a driver message, or its first line when no sentence ends there. */
export function firstSentence(text: unknown, limit = HEADLINE_LIMIT): string {
  if (typeof text !== 'string') return '';
  const line = text.trim().split('\n', 1)[0].trim();
  const sentence = /^.*?[.!?](?=\s|$)/.exec(line)?.[0] ?? line;
  return bounded(sentence, limit);
}

function bounded(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit - 1).trimEnd()}…` : text;
}

function count(text: string, part: string): number {
  return text.split(part).length - 1;
}

/** One class tag anywhere in the text, as dgx_monarch.refusal.parse_refusal_tag reads it. */
function classTag(text: string) {
  const matches = [...text.matchAll(TAG)];
  if (matches.length !== 1) return null;
  const [whole, letter, guard, waivable] = matches[0];
  if ((guard === undefined) !== (waivable === undefined)) return null;
  return {
    refusalClass: letter as RefusalClass,
    guard: guard ?? null,
    waivable: waivable === '1',
    end: (matches[0].index ?? 0) + whole.length,
  };
}

function flatAndFinite(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  return Object.values(value).every(
    (item) =>
      item === null ||
      typeof item === 'string' ||
      typeof item === 'boolean' ||
      (typeof item === 'number' && Number.isFinite(item)),
  );
}

/**
 * The one consent block in the text, read as dgx_monarch.consent_descriptor.parse reads it, and
 * the text without it. A missing, repeated, oversized or malformed block yields no request. One
 * difference: JSON.parse keeps the last of two equal keys, where the Python parse refuses.
 */
function descriptor(text: string): { consent: ConsentRequest; rest: string } | null {
  if (count(text, DESCRIPTOR_BEGIN) !== 1) return null;
  const start = text.indexOf(DESCRIPTOR_BEGIN);
  const tail = text.slice(start + DESCRIPTOR_BEGIN.length);
  if (count(tail, DESCRIPTOR_END) !== 1) return null;
  const payload = tail.slice(0, tail.indexOf(DESCRIPTOR_END));
  if (!payload || /\p{Surrogate}/u.test(payload)) return null;
  if (new TextEncoder().encode(payload).length > DESCRIPTOR_BYTES) return null;
  let body: Record<string, unknown>;
  try {
    body = JSON.parse(payload);
  } catch {
    return null;
  }
  if (!body || typeof body !== 'object' || Array.isArray(body) || body.version !== 1) return null;
  if (typeof body.kind !== 'string' || !KIND.test(body.kind)) return null;
  if (!REQUIRED.every((name) => typeof body[name] === 'string' && body[name] !== '')) return null;
  if (body.family_hint != null && typeof body.family_hint !== 'string') return null;
  if (!flatAndFinite(body.measured)) return null;
  const end = start + DESCRIPTOR_BEGIN.length + payload.length + DESCRIPTOR_END.length;
  return {
    consent: {
      kind: body.kind,
      consentId: body.consent_id as string,
      unetName: body.unet_name as string,
      humanReason: body.human_reason as string,
      panelAction: body.panel_action as string,
    },
    rest: `${text.slice(0, start)}${text.slice(end)}`,
  };
}

/** The last remedy the message carries and the troubleshooting entry it cites. */
function remedyOf(text: string): { remedy: string; troubleshooting: number | null } {
  const found = [...text.matchAll(REMEDY)].at(-1);
  const cited = found?.[1] ?? found?.[2] ?? [...text.matchAll(CITATION)].at(-1)?.[1];
  const panel = found && found[2] === undefined ? found[0] : '';
  return {
    remedy: panel.replace(/ See docs\/TROUBLESHOOTING\.md #\d{1,4}\.$/, '').trim(),
    troubleshooting: cited === undefined ? null : Number(cited),
  };
}

/** A readable failure from a job's recorded error, or null when the job recorded none. */
export function jobFailure(error: Job['error']): JobFailure | null {
  if (!error || typeof error !== 'object') return null;
  const code = typeof error.code === 'string' ? error.code : '';
  const message = typeof error.message === 'string' ? error.message : '';
  if (!code && !message) return null;
  const block = descriptor(message);
  const shown = (block ? block.rest : message).trimEnd();
  const tag = classTag(message);
  const headline = (tag && firstSentence(message.slice(tag.end))) || firstSentence(shown) || '';
  const lead = tag
    ? `Refused, class ${tag.refusalClass} (${CLASS_TITLES[tag.refusalClass]})`
    : code;
  return {
    code,
    refusalClass: tag?.refusalClass ?? null,
    guard: tag?.guard ?? null,
    waivable: tag?.waivable ?? false,
    headline,
    summary: [lead, headline].filter(Boolean).join(': '),
    text: bounded(shown, TEXT_LIMIT),
    ...remedyOf(shown),
    consent: block?.consent ?? null,
  };
}

export interface ConsentAction {
  label: 'Grant and retry' | 'Review consent';
  /** Whether this companion can review and apply the consent operation. */
  available: boolean;
  /** Why it cannot, in plain words, or ''. */
  reason: string;
}

/**
 * The consent control for a failure that asks for consent. Only the companion's capability row
 * for consent.apply decides: it is unavailable exactly when the companion was started without
 * --allow-management. With no such row there is no control.
 */
export function consentAction(
  failure: JobFailure | null,
  operations: OperationCapability[] | undefined,
): ConsentAction | null {
  if (!failure?.consent) return null;
  const row = operations?.find((operation) => operation.id === CONSENT_OPERATION);
  if (!row) return null;
  if (row.available) return { label: 'Grant and retry', available: true, reason: '' };
  const reason = row.reason?.includes('--allow-management') ? NO_MANAGEMENT : (row.reason ?? '');
  return { label: 'Review consent', available: false, reason };
}

/**
 * Open the consent operation for a job that asked for consent. The descriptor carries no pending
 * key or id, so both stay empty: the review then selects the one pending request, or lists every
 * pending request with its consent id.
 */
export function consentRequest(job: Job, id: string): OperationRequest | null {
  const consent = jobFailure(job.error)?.consent;
  if (!consent) return null;
  return {
    id,
    action: CONSENT_OPERATION,
    parameters: { key: '', id: '' },
    consent: { jobId: job.id, ...consent },
  };
}

export interface PendingCard {
  key: string;
  id: string;
  consent_id?: unknown;
  title?: unknown;
}

function cards(value: unknown): PendingCard[] {
  return Array.isArray(value)
    ? value.filter(
        (card): card is PendingCard =>
          !!card && typeof card.key === 'string' && typeof card.id === 'string',
      )
    : [];
}

/**
 * How a consent review's plan relates to the job's request. A ready plan names the card it chose,
 * and `same` says whether that card carries the job's consent id. A blocked plan lists every
 * pending card, and `listed` holds those that carry it. A plan that read no pending list, as
 * without --allow-management, yields null.
 */
export function consentMatch(
  preview: OperationRecord | null,
  consentId: string,
): { chosen: PendingCard; same: boolean } | { listed: PendingCard[] } | null {
  const plan = preview?.adapter_plan as { card?: unknown; pending?: unknown } | null | undefined;
  const [chosen] = cards(plan?.card ? [plan.card] : []);
  if (chosen) return { chosen, same: chosen.consent_id === consentId };
  if (!Array.isArray(plan?.pending)) return null;
  return { listed: cards(plan.pending).filter((card) => card.consent_id === consentId) };
}
