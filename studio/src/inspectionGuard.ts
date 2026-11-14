/** A request started against one draft and one companion connection. */
export interface Ticket<Client> {
  inspection: number;
  revision: number;
  client: Client;
}

/** Why a result may no longer apply: `newer` means a later inspection replaced it. */
export type TicketState = 'current' | 'newer' | 'edited' | 'reconnected';

/** Only workflow and value edits make an inspection result stale. */
export function supersedesInspection(change: object): boolean {
  const edit = change as { workflow?: unknown; values?: unknown };
  return edit.workflow !== undefined || edit.values !== undefined;
}

/**
 * Orders workflow inspections and job previews against draft edits and
 * reconnects. An inspection is stale after a workflow or value edit, a newer
 * inspection or a reconnect; a preview is stale after any draft edit or a
 * reconnect.
 */
export class InspectionGuard<Client = unknown> {
  private inspection = 0;
  private latest = 0;
  private revision = 0;

  /** Start an inspection; any earlier one in flight becomes `newer`. */
  begin(client: Client): Ticket<Client> {
    this.latest = ++this.inspection;
    return { inspection: this.inspection, revision: this.revision, client };
  }

  /**
   * Capture the draft revision before a preview, or before Create checks its shown issues again,
   * without superseding an inspection.
   */
  snapshot(client: Client): Ticket<Client> {
    return { inspection: this.inspection, revision: this.revision, client };
  }

  /** Record a draft edit. */
  edit(change: object) {
    this.revision++;
    if (supersedesInspection(change)) this.inspection++;
  }

  /** Decide whether an inspection result still applies. */
  inspected(ticket: Ticket<Client>, client: Client): TicketState {
    if (ticket.client !== client) return 'reconnected';
    if (this.latest > ticket.inspection) return 'newer';
    return this.inspection === ticket.inspection ? 'current' : 'edited';
  }

  /** Decide whether a preview still matches the draft and connection it was made from. */
  previewed(ticket: Ticket<Client>, client: Client): boolean {
    return ticket.client === client && ticket.revision === this.revision;
  }
}

/** Tell the user why an inspection result was not applied, or '' when nothing needs saying. */
export function droppedInspection(name: string, state: TicketState): string {
  const subject = name ? `The inspection of ${name}` : 'The workflow inspection';
  if (state === 'edited')
    return `${subject} was discarded because the workflow or its values changed while it ran. Choose the workflow again to load it.`;
  if (state === 'reconnected')
    return `${subject} was discarded because the companion connection changed while it ran. Choose the workflow again to load it.`;
  return '';
}

export const DROPPED_REVIEW =
  'Studio dropped this check because the draft or the companion connection changed while it ran, so no job started. Generate again.';
