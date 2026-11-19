import type { Artifact } from './contracts';

/**
 * What the output viewer loaded for one selection: the object URLs of the selected output and
 * its comparison, with the media kinds those URLs hold. A URL and its kind change together, so
 * the viewer never hands one output's bytes to an element made for another kind of media.
 */
export interface ViewerMedia {
  /** The selection these URLs belong to: the viewer's media key, or '' before any load. */
  key: string;
  /** Counts loads; an answer for an earlier load changes nothing. */
  ticket: number;
  loading: boolean;
  error: string;
  url: string;
  /** The kind of the output `url` holds, recorded when its load began. */
  kind: string;
  compareUrl: string;
  compareKind: string;
}

export const NO_MEDIA: ViewerMedia = {
  key: '',
  ticket: 0,
  loading: false,
  error: '',
  url: '',
  kind: '',
  compareUrl: '',
  compareKind: '',
};

/**
 * Begin loading a new selection. The earlier URLs, kinds and error belong to the earlier
 * selection and go with it; the caller revokes those URLs. With no output selected nothing loads.
 */
export function startLoad(
  media: ViewerMedia,
  key: string,
  selected: Artifact | null,
  compare: Artifact | null,
): ViewerMedia {
  return {
    ...NO_MEDIA,
    key,
    ticket: media.ticket + 1,
    loading: !!selected,
    kind: selected?.kind ?? '',
    compareKind: compare?.kind ?? '',
  };
}

/** The fetched media of load `ticket` arrived as object URLs. */
export function finishLoad(
  media: ViewerMedia,
  ticket: number,
  url: string,
  compareUrl = '',
): ViewerMedia {
  if (ticket !== media.ticket) return media;
  return { ...media, loading: false, url, compareUrl };
}

/** Load `ticket` failed with `message`. */
export function failLoad(media: ViewerMedia, ticket: number, message: string): ViewerMedia {
  if (ticket !== media.ticket) return media;
  return { ...media, loading: false, error: message };
}

/**
 * A media element could not decode `source`, the URL it was given. Only the element that shows
 * the current URL counts: a removed image from an earlier selection still fires its error event
 * after the viewer has moved on, and that failure belongs to no output on screen.
 */
export function decodeFailed(
  media: ViewerMedia,
  source: string | null,
  comparing = false,
): ViewerMedia {
  const current = comparing ? media.compareUrl : media.url;
  if (!current || source !== current) return media;
  const which = comparing ? 'Comparison media' : 'Selected media';
  return {
    ...media,
    error: `${which} cannot be decoded by this browser. Download the file to inspect it or open it in a compatible viewer.`,
  };
}

/**
 * The media the viewer frame shows for the selection `key`. Until a load for that selection has
 * begun, which happens just after the selection changes, it shows as loading, never with the
 * URLs of the selection before it.
 */
export function currentMedia(media: ViewerMedia, key: string): ViewerMedia {
  return media.key === key ? media : { ...NO_MEDIA, key, ticket: media.ticket, loading: true };
}
