/** A Studio view: its route id, the label and icon it shows, and for a tab its one-line purpose. */
export interface Destination {
  id: string;
  label: string;
  icon: string;
  /** The rail entry that holds this view as a tab, when the rail does not list it. */
  group?: string;
  summary?: string;
}

/**
 * The views grouped under Utilities, in tab order. Their route ids stay the ones links and
 * buttons already open, such as Grant and retry opening `operations`.
 */
export const utilities: Destination[] = [
  {
    id: 'grid',
    label: 'Grids',
    icon: '⊞',
    group: 'Utilities',
    summary:
      'Run the Create draft several times with one or two inputs changed, then compare the results side by side.',
  },
  {
    id: 'fleet',
    label: 'Fleet',
    icon: '◇',
    group: 'Utilities',
    summary:
      'Render many separate prompts at once, one job per GPU, when you want more pictures rather than one picture faster.',
  },
  {
    id: 'operations',
    label: 'Operations',
    icon: '⌘',
    group: 'Utilities',
    summary:
      'Check and look after the hosts: run Doctor, start the driver, grant a consent or update. Each action shows its exact plan first.',
  },
];

/** The rail entry for the utility tabs; navigating to it opens the tab used last. */
export const UTILITIES = 'utilities';

/** The navigation rail, top to bottom. */
export const navigation: Destination[] = [
  { id: 'workspace', label: 'Create', icon: '◈' },
  { id: 'gallery', label: 'Gallery', icon: '▦' },
  { id: 'queue', label: 'Queue', icon: '≡' },
  { id: UTILITIES, label: 'Utilities', icon: '⊞' },
  { id: 'projects', label: 'Library', icon: '▤' },
  { id: 'settings', label: 'Settings', icon: '⚙' },
];

/** Every view the command palette opens: the rail's views, with each utility tab in its place. */
export const destinations: Destination[] = navigation.flatMap((item) =>
  item.id === UTILITIES ? utilities : [item],
);

export const isUtility = (view: string) => utilities.some((item) => item.id === view);

/** The rail entry that shows as current while a view is open. */
export const railView = (view: string) => (isUtility(view) ? UTILITIES : view);

let lastUtility = utilities[0].id;
/** The view a navigation opens. The Utilities alias opens the utility tab opened last, Grids at first. */
export function resolveView(id: string): string {
  if (id === UTILITIES) return lastUtility;
  if (isUtility(id)) lastUtility = id;
  return id;
}
