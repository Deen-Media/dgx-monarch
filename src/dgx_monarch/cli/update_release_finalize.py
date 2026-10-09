"""Finalize a verified worker release on every host; one host's failure does not stop the rest."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence

from ..config import ClusterConfig, HostConfig
from .update_cancellation import stronger_cancellation
from .update_transaction import Certainty

_TOKEN = re.compile(r"u-[0-9a-f]{12}-[0-9a-f]{16}")
_BASE = ".local/share/dgx-monarch"
HostRunner = Callable[
    [ClusterConfig, HostConfig, str, int], subprocess.CompletedProcess[str]
]


def finalize_release(
    config: ClusterConfig,
    hosts: Sequence[HostConfig],
    token: str,
    host_runner: HostRunner,
) -> Certainty:
    if _TOKEN.fullmatch(token) is None:
        raise ValueError("release token is invalid")
    target = f"$HOME/{_BASE}/releases/{token}/site"
    certainties: list[Certainty] = []
    pending: BaseException | None = None
    for host in hosts:
        script = f"""
set -eu
BASE="$HOME/{_BASE}"
TXN="$BASE/transactions/{token}"
BACKUP="$BASE/backups/{token}"
TARGET="{target}"
test "$(readlink -f -- "$BASE/src")" = "$TARGET"
if [ -f "$TXN/kind" ] && [ "$(cat "$TXN/kind")" = symlink ]; then
  PRIOR=$(cat "$TXN/prior")
  case "$PRIOR" in "$BASE"/releases/*/site) ;; *) exit 70;; esac
  test "$PRIOR" != "$TARGET"
  PRIOR_ROOT=${{PRIOR%/site}}
  test "$PRIOR_ROOT" != "$BASE/releases/{token}"
  rm -rf -- "$PRIOR_ROOT"
fi
if [ -d "$BACKUP" ]; then rm -rf -- "$BACKUP"; fi
rm -rf -- "$TXN"
echo FINALIZED
"""
        try:
            result = host_runner(config, host, script, 60)
        except (OSError, subprocess.TimeoutExpired):
            certainties.append(Certainty.UNKNOWN)
            continue
        except BaseException as error:
            certainties.append(Certainty.UNKNOWN)
            pending = stronger_cancellation(pending, error)
            continue
        certainties.append(
            Certainty.SUCCEEDED
            if result.returncode == 0 and "FINALIZED" in result.stdout
            else Certainty.FAILED
        )
    if pending is not None:
        raise pending
    if all(value == Certainty.SUCCEEDED for value in certainties):
        return Certainty.SUCCEEDED
    return Certainty.UNKNOWN if Certainty.UNKNOWN in certainties else Certainty.FAILED
