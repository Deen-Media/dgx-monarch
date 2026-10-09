"""Journaled remote release switch primitives."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterable

from ..config import ClusterConfig, HostConfig
from .lifecycle_lock import locked_script
from .update_types import Certainty

_TOKEN = re.compile(r"u-[0-9a-f]{12}-[0-9a-f]{16}")
_BASE = ".local/share/dgx-monarch"


HostRunner = Callable[
    [ClusterConfig, HostConfig, str, int], subprocess.CompletedProcess[str]
]


def lifecycle_host_runner(host_runner: HostRunner) -> HostRunner:
    """Wrap release mutations in the shared per-user lifecycle lock."""
    def run(
        config: ClusterConfig, host: HostConfig, script: str, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        return host_runner(config, host, locked_script(config.python_bin, script), timeout)
    return run


def _token(value: str) -> str:
    if _TOKEN.fullmatch(value) is None:
        raise ValueError("release token is invalid")
    return value


def activation_script(token: str) -> str:
    """Build one host's journaled switch, run under the lifecycle lock; on failure it prints the probe's verdict."""
    token = _token(token)
    return f"""
if (
set -eu
BASE="$HOME/{_BASE}"
NEW="$BASE/releases/{token}/site"
LIVE="$BASE/src"
TXN="$BASE/transactions/{token}"
BACKUP="$BASE/backups/{token}/src"
test -d "$NEW/dgx_monarch"
umask 077
mkdir -p "$TXN" "$BASE/backups/{token}"
if [ -L "$LIVE" ]; then
  PRIOR=$(readlink -f -- "$LIVE")
  case "$PRIOR" in "$BASE"/releases/*/site) ;; *) exit 40;; esac
  printf %s symlink > "$TXN/kind"
  printf %s "$PRIOR" > "$TXN/prior"
elif [ -d "$LIVE" ]; then
  test ! -e "$BACKUP"
  printf %s directory > "$TXN/kind"
  : > "$TXN/prior"
elif [ ! -e "$LIVE" ]; then
  printf %s absent > "$TXN/kind"
  : > "$TXN/prior"
else
  exit 41
fi
sync "$TXN/kind" "$TXN/prior"
if [ "$(cat "$TXN/kind")" = directory ]; then mv -- "$LIVE" "$BACKUP"; fi
NEXT="$BASE/.src-next-{token}"
test ! -e "$NEXT"
ln -s -- "$NEW" "$NEXT"
mv -T -- "$NEXT" "$LIVE"
test "$(readlink -f -- "$LIVE")" = "$NEW"
echo ACTIVATED
); then
  :
else
  if DGXM_ACTIVATION_PROBE=$( {_activation_probe_body(token)} ); then
    case "$DGXM_ACTIVATION_PROBE" in
      NEW) echo ACTIVATION_PROBE_NEW;;
      PRIOR) echo ACTIVATION_PROBE_PRIOR;;
      *) echo ACTIVATION_PROBE_UNKNOWN;;
    esac
  else
    echo ACTIVATION_PROBE_UNKNOWN
  fi
fi
"""


def _activation_probe_body(token: str) -> str:
    """Print NEW or PRIOR, or exit 2 when unsure; it calls exit, so run it in its own process or inside $( )."""
    token = _token(token)
    return f"""
BASE="$HOME/{_BASE}"
LIVE="$BASE/src"
NEW="$BASE/releases/{token}/site"
TXN="$BASE/transactions/{token}"
if [ -L "$LIVE" ] && [ "$(readlink -f -- "$LIVE" 2>/dev/null)" = "$NEW" ]; then echo NEW; exit 0; fi
if [ ! -f "$TXN/kind" ]; then [ ! -e "$TXN" ] && echo PRIOR && exit 0; exit 2; fi
KIND=$(cat "$TXN/kind" 2>/dev/null) || exit 2
if [ "$KIND" = symlink ]; then
  PRIOR=$(cat "$TXN/prior" 2>/dev/null) || exit 2
  [ -L "$LIVE" ] && [ "$(readlink -f -- "$LIVE" 2>/dev/null)" = "$PRIOR" ] && echo PRIOR && exit 0
elif [ "$KIND" = directory ]; then
  {{ [ -d "$LIVE" ] && [ ! -L "$LIVE" ]; }} || {{ [ ! -e "$LIVE" ] && [ -d "$BASE/backups/{token}/src" ]; }} || exit 2
  echo PRIOR
  exit 0
elif [ "$KIND" = absent ]; then
  [ ! -e "$LIVE" ] && echo PRIOR && exit 0
fi
exit 2
"""


def probe_activation(
    config: ClusterConfig,
    host: HostConfig,
    token: str,
    host_runner: HostRunner,
) -> Certainty:
    """Classify a failed switch only from an exact readback of the live link and its journal."""
    token = _token(token)
    script = _activation_probe_body(token)
    try:
        result = host_runner(config, host, script, 30)
    except (OSError, subprocess.TimeoutExpired):
        return Certainty.UNKNOWN
    if result.returncode != 0:
        return Certainty.UNKNOWN
    if "NEW" in result.stdout:
        return Certainty.SUCCEEDED
    return Certainty.FAILED if "PRIOR" in result.stdout else Certainty.UNKNOWN


def activate_hosts(
    config: ClusterConfig,
    hosts: Iterable[HostConfig],
    token: str,
    host_runner: HostRunner,
) -> Certainty:
    """Switch and classify every host inside one lock interval per host."""
    results: list[Certainty] = []
    run = lifecycle_host_runner(host_runner)
    for host in hosts:
        try:
            result = run(config, host, activation_script(token), 60)
        except (OSError, subprocess.TimeoutExpired):
            results.append(Certainty.UNKNOWN)
            continue
        if result.returncode == 0 and (
            "ACTIVATED" in result.stdout or "ACTIVATION_PROBE_NEW" in result.stdout
        ):
            results.append(Certainty.SUCCEEDED)
        elif result.returncode == 0 and "ACTIVATION_PROBE_PRIOR" in result.stdout:
            results.append(Certainty.FAILED)
        else:
            results.append(Certainty.UNKNOWN)
    if all(result == Certainty.SUCCEEDED for result in results):
        return Certainty.SUCCEEDED
    if any(result == Certainty.UNKNOWN for result in results):
        return Certainty.UNKNOWN
    return Certainty.FAILED
