import type { StudioClient } from './api';
import type { GraphConnection, Inspection } from './contracts';
import type { InspectionGuard, Ticket } from './inspectionGuard.ts';
import type { Draft } from './model';

/** App's inspection guard, and the way to show a new answer's issues and model connections. */
export interface IssueRefresh {
  guard: InspectionGuard<unknown>;
  apply: (next: { errors: string[]; connections: GraphConnection[] }) => void;
}
/** The current companion connection and draft, read when a refresh starts and when it answers. */
export interface Current {
  client: () => StudioClient;
  draft: () => Draft;
}

/** Create waits this long after the last edit, so typing in a field sends one inspection. */
export const REFRESH_DELAY_MS = 400;

/** The guard's state when Create first showed each list that carries no inspection ticket. */
const shown = new WeakMap<string[], Ticket<unknown>>();
/** Where a list keeps the ticket of the inspection that produced it. */
const MADE = Symbol('inspection ticket');

/**
 * Keep the ticket of the inspection that produced `errors` on the list itself, so the list is
 * dated by that inspection and not by when Create first shows it. A rebuild that kept a field
 * edited while it ran applies errors from before that edit, and its ticket marks them stale at
 * once. The property is not writable, so Svelte's state proxy reads it from the list unchanged.
 */
export function stampIssues(errors: string[], ticket: Ticket<unknown>): string[] {
  if (!(MADE in errors)) Object.defineProperty(errors, MADE, { value: ticket });
  return errors;
}

/**
 * Whether the shown workflow issues may no longer fit the draft. The guard decides: a workflow or
 * value edit, a later inspection or a new companion connection since the inspection that produced
 * the list. A list with no ticket, such as one a workflow load showed, which applies only while
 * current, is dated by Create's first look at it, so it keeps its age when Create closes and opens.
 */
export function issuesStale(errors: string[], guard: InspectionGuard<unknown>, client: unknown) {
  if (!errors.length) return false;
  let ticket = (errors as { [MADE]?: Ticket<unknown> })[MADE] ?? shown.get(errors);
  if (!ticket) shown.set(errors, (ticket = guard.snapshot(client)));
  return guard.inspected(ticket, client) !== 'current';
}

/**
 * Inspect the draft again with its values and show the answer. It never supersedes an inspection
 * in flight, and it drops its answer without a notice when the guard no longer calls it current:
 * an edit meanwhile asks again, and an inspection Studio started brings its own answer.
 */
export async function refreshIssues(refresh: IssueRefresh, current: Current): Promise<boolean> {
  const client = current.client();
  const { workflow, values } = current.draft();
  if (!workflow) return false;
  const ticket = refresh.guard.snapshot(client);
  const result = await client.request<Inspection>('/workflows/inspect', { workflow, values });
  if (refresh.guard.inspected(ticket, current.client()) !== 'current') return false;
  refresh.apply({
    errors: stampIssues(result.errors ?? [], ticket),
    connections: result.connections ?? [],
  });
  return true;
}

/**
 * Body of Create's effect. It reads the connection and the draft's workflow and values, so each
 * edit runs it again, and while the shown issues are stale it inspects the draft again once the
 * edits settle. Returns the cleanup that cancels a refresh not yet sent. `checking` says whether a
 * refresh waits or runs; one that fails leaves the shown issues as they were.
 */
export function followIssues(
  refresh: IssueRefresh | undefined,
  errors: string[],
  current: Current,
  checking: (on: boolean) => void,
  delay = REFRESH_DELAY_MS,
) {
  const client = current.client();
  const draft = current.draft();
  void [draft.workflow, draft.values];
  if (!refresh || !issuesStale(errors, refresh.guard, client)) return void checking(false);
  checking(true);
  let live = true;
  const timer = setTimeout(() => {
    void refreshIssues(refresh, current)
      .catch(() => false)
      .finally(() => live && checking(false));
  }, delay);
  return () => {
    live = false;
    clearTimeout(timer);
  };
}
