import type { Job, OperationRequest } from './contracts';
import { consentRequest } from './jobFailure';

/**
 * What Operations opens next: an operation the palette chose, or the consent a failed job asks
 * for. Each request gets a new random identity, so asking again applies again and a review saved
 * before a reload never answers a later request. Opening never reviews or applies anything; the
 * user still reviews the exact plan in Operations.
 */
export class OperationOpener {
  request = $state<OperationRequest | null>(null);
  #navigate: (view: string) => void;
  constructor(navigate: (view: string) => void) {
    this.#navigate = navigate;
  }

  /** Open an operation chosen in the palette. */
  operation = (action: string) => {
    this.request = { id: crypto.randomUUID(), action };
    this.#navigate('operations');
  };

  /** Open consent.apply for a job whose failure carries a consent request; others open nothing. */
  consent = (job: Job | undefined) => {
    const request = job ? consentRequest(job, crypto.randomUUID()) : null;
    if (!request) return;
    this.request = request;
    this.#navigate('operations');
  };
}
