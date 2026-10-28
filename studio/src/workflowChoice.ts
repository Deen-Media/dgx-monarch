import type { InspectWorkflow, WorkflowEntry } from './contracts';

/** The text Studio lists a packaged workflow by: its catalog title, or its file stem without one. */
export function workflowTitle(entry: WorkflowEntry): string {
  return entry.title || entry.name;
}

const collator = new Intl.Collator('en', { sensitivity: 'base', numeric: true });

/** A copy of the catalog in title order, ignoring case; the file stem breaks a tie. */
export function byTitle(entries: readonly WorkflowEntry[]): WorkflowEntry[] {
  return [...entries].sort(
    (a, b) =>
      collator.compare(workflowTitle(a), workflowTitle(b)) || collator.compare(a.name, b.name),
  );
}

/** True when every word of the query appears, in any case, in the title or the file stem. */
export function matchesWorkflow(entry: WorkflowEntry, query: string): boolean {
  const text = `${workflowTitle(entry)} ${entry.name}`.toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((word) => text.includes(word));
}

/**
 * The workflow select's choices in title order. `matches` are the entries the query matches;
 * `options` also keeps the current choice when the query hides it, so the select still shows it.
 */
export function workflowChoices(entries: readonly WorkflowEntry[], query: string, current: string) {
  const sorted = byTitle(entries);
  const matches = sorted.filter((entry) => matchesWorkflow(entry, query));
  const options = sorted.filter((entry) => entry.id === current || matches.includes(entry));
  return { matches, options };
}

/**
 * What a key does in the filter box: Escape clears the filter, and Enter loads the first match
 * while no packaged workflow is chosen. Any other key, or Enter with a choice, does nothing.
 */
export function filterKeyAction(
  key: string,
  current: string,
  matches: readonly WorkflowEntry[],
): 'clear' | WorkflowEntry | undefined {
  if (key === 'Escape') return 'clear';
  if (key === 'Enter' && !current) return matches[0];
  return undefined;
}

/** The palette's workflow commands: none until the query has a word, then its matches by title. */
export function paletteWorkflows(
  entries: readonly WorkflowEntry[],
  query: string,
): WorkflowEntry[] {
  return query.trim() ? byTitle(entries).filter((entry) => matchesWorkflow(entry, query)) : [];
}

/** Load a packaged workflow into the draft; the select and the command palette both use this. */
export function loadWorkflow(entry: WorkflowEntry, inspect: InspectWorkflow): Promise<void> {
  return inspect(entry.workflow, entry.name, false, entry.id);
}
