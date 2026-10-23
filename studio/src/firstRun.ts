import { ApiError } from './api.ts';

/** The companion's code when nothing accepts a connection at the driver origin. */
export const DRIVER_NOT_RUNNING = 'driver_not_running';

/** Shown when the companion refuses the token a sign-in link carried. */
export const STALE_LINK =
  'This sign-in link was not accepted: it is from an earlier start of Studio, or it was cut short when copied. Run dgxm studio open on the Studio host for a new link.';

/**
 * Whether the page should say how to start the driver. Before the companion's first
 * observation the snapshot has no code yet, so a catalog read refused for the same
 * reason counts too.
 */
export function driverWaiting(
  driver: { state?: string; code?: string } | undefined,
  catalogCode: string,
): boolean {
  if (driver?.state === 'connected') return false;
  if (driver?.code) return driver.code === DRIVER_NOT_RUNNING;
  return catalogCode === DRIVER_NOT_RUNNING;
}

/**
 * The text for a failed action, or '' when the driver-not-running panel already says
 * it. A refused sign-in link gets its own sentence.
 */
export function errorText(e: unknown, fromLink: boolean): string {
  if (e instanceof ApiError && e.code === DRIVER_NOT_RUNNING) return '';
  if (fromLink && e instanceof ApiError && e.status === 401) return STALE_LINK;
  return e instanceof Error ? e.message : String(e);
}

/** Where the start command came from, naming the config file it was read from. */
export function sourceText(
  source: string | null | undefined,
  config: string | null | undefined,
  command: string | null | undefined,
): string {
  if (source === 'driver_service') {
    const unit = (command ?? '').split(' ').pop();
    const from = config
      ? `From the driver_service setting in ${config}.`
      : 'From the --driver-service option this Studio started with.';
    return `${from} If it fails, read: journalctl --user -u ${unit} -n 50`;
  }
  if (source === 'driver_command')
    return `From the driver_command setting in ${config ?? 'cluster.toml'}. Studio reads it when it starts, so restart Studio after an edit, and edit cluster.toml when no render runs.`;
  if (source === 'launcher')
    return 'The repository launcher for this driver origin, with the Python that runs Studio; run it in a terminal you keep open. If you start ComfyUI another way, set driver_command in the [studio] table of cluster.toml.';
  return '';
}

/** The stale banner when a poll gets no answer at all from the companion. */
export function unreachableText(host: string): string {
  return `Studio does not answer at ${host}. If you reach it through SSH, check that the ssh -L tunnel is still open; if Studio restarted, run dgxm studio open on the Studio host for a new link.`;
}
