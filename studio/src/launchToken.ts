const LAUNCH = /^#token=([A-Za-z0-9_-]{32,128})$/;
const TOKEN = /^[A-Za-z0-9_-]{32,128}$/;
/** A sign-in link, or any text that ends in its fragment. */
const LINK = /^\S*#token=([A-Za-z0-9_-]{32,128})$/;
let taken = '';

/** Move a launch token out of the URL fragment. Runs once, before the app mounts. */
export function takeLaunchToken(
  location: Pick<Location, 'hash' | 'pathname' | 'search'>,
  history: Pick<History, 'replaceState' | 'state'>,
): void {
  if (!location.hash.startsWith('#token=')) return;
  const match = LAUNCH.exec(location.hash);
  try {
    history.replaceState(history.state, '', location.pathname + location.search);
  } catch {
    // The fragment stays visible; the token still reaches this tab in memory.
  }
  taken = match ? match[1] : '';
}

/** The launch token, handed over once; later calls return ''. */
export function launchToken(): string {
  const token = taken;
  taken = '';
  return token;
}

/** A typed or pasted value: a bare token, or a whole sign-in link. Other text comes back unchanged. */
export function tokenFromInput(text: string): string {
  const value = text.trim();
  if (TOKEN.test(value)) return value;
  const link = LINK.exec(value);
  return link ? link[1] : text;
}
