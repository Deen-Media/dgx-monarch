import test from 'node:test';
import assert from 'node:assert/strict';
import { ApiError } from './api.ts';
import {
  DRIVER_NOT_RUNNING,
  STALE_LINK,
  driverWaiting,
  errorText,
  sourceText,
  unreachableText,
} from './firstRun.ts';

const CONFIG = '/home/user/.config/dgx-monarch/cluster.toml';

test('the driver panel shows only while nothing accepts a connection at the driver origin', () => {
  assert.equal(DRIVER_NOT_RUNNING, 'driver_not_running');
  const cases: [Record<string, unknown> | undefined, string, boolean][] = [
    [{ state: 'connected' }, '', false],
    [{ state: 'connected' }, 'driver_not_running', false],
    [{ state: 'unavailable', code: 'driver_not_running', since: 5 }, '', true],
    [{ state: 'unavailable', code: 'driver_not_running', since: 5 }, 'driver_unknown', true],
    // Before the companion's first observation the snapshot has no code yet.
    [{ state: 'unavailable', timestamp: null }, 'driver_not_running', true],
    [{}, 'driver_not_running', true],
    [undefined, 'driver_not_running', true],
    [{ state: 'unavailable', timestamp: null }, '', false],
    [{ state: 'unavailable', code: 'driver_unknown' }, '', false],
    [{ state: 'unavailable', code: 'driver_unknown' }, 'driver_not_running', false],
    [{ state: 'unavailable', code: 'driver_unreachable' }, '', false],
    [{ state: 'unavailable', code: 'driver_unreachable' }, 'driver_not_running', false],
    [{ state: 'unavailable' }, 'driver_unknown', false],
  ];
  for (const [driver, catalogCode, expected] of cases)
    assert.equal(
      driverWaiting(driver, catalogCode),
      expected,
      JSON.stringify([driver, catalogCode]),
    );
});

test('errorText hides only the not-running refusal and names a refused link', () => {
  const notRunning = new ApiError(
    503,
    'No driver accepted a connection at the driver origin; start the driver.',
    'driver_not_running',
  );
  assert.equal(errorText(notRunning, false), '');
  assert.equal(errorText(notRunning, true), '');
  const refused = new ApiError(401, "Enter this companion's owner token.", 'authentication');
  assert.equal(errorText(refused, true), STALE_LINK);
  assert.equal(errorText(refused, false), "Enter this companion's owner token.");
  assert.equal(
    STALE_LINK,
    'This sign-in link was not accepted: it is from an earlier start of Studio, or it was cut short when copied. Run dgxm studio open on the Studio host for a new link.',
  );
  const unknown = new ApiError(
    503,
    'The driver did not return a complete response.',
    'driver_unknown',
  );
  assert.equal(errorText(unknown, true), 'The driver did not return a complete response.');
  assert.equal(
    errorText(new Error('Workflow file is too large.'), true),
    'Workflow file is too large.',
  );
  assert.equal(errorText('plain text', false), 'plain text');
});

test('sourceText names the setting and the config file the command came from', () => {
  assert.equal(
    sourceText('driver_service', CONFIG, 'systemctl --user start comfyui-driver.service'),
    `From the driver_service setting in ${CONFIG}. If it fails, read: journalctl --user -u comfyui-driver.service -n 50`,
  );
  // A companion started by hand with --driver-service has no config to name.
  assert.equal(
    sourceText('driver_service', null, 'systemctl --user start comfyui-alt.service'),
    'From the --driver-service option this Studio started with. If it fails, read: journalctl --user -u comfyui-alt.service -n 50',
  );
  assert.equal(
    sourceText('driver_command', CONFIG, 'bash /home/user/bin/start-driver.sh'),
    `From the driver_command setting in ${CONFIG}. Studio reads it when it starts, so restart Studio after an edit, and edit cluster.toml when no render runs.`,
  );
  assert.equal(
    sourceText('launcher', null, 'bash comfy-driver.sh'),
    'The repository launcher for this driver origin, with the Python that runs Studio; run it in a terminal you keep open. If you start ComfyUI another way, set driver_command in the [studio] table of cluster.toml.',
  );
  assert.equal(sourceText(null, null, null), '');
  assert.equal(sourceText('something_new', CONFIG, 'x'), '');
});

test('unreachableText names the host and the tunnel', () => {
  assert.equal(
    unreachableText('127.0.0.1:8220'),
    'Studio does not answer at 127.0.0.1:8220. If you reach it through SSH, check that the ssh -L tunnel is still open; if Studio restarted, run dgxm studio open on the Studio host for a new link.',
  );
});
