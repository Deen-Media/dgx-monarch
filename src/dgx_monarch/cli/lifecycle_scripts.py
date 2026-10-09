"""Lock-held Worker start/stop payload builders."""
from __future__ import annotations

import shlex

from ..config import ClusterConfig, HostConfig
from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX
from . import actor_sweep
from .lifecycle_generation import (
    invalidate_generation_shell,
    require_generation_shell,
    settle_generation_shell,
    start_generation_shell,
    systemd_generation_invalidator,
)
from .lifecycle_host import owned_listener_probe_shell
from .lifecycle_managed import managed_start_guard


def build_start_script(
    config: ClusterConfig,
    host: HostConfig,
    *,
    pattern: str,
    pythonpath: str,
    generation: str | None,
    unit_name: str,
    log_path: str,
    pybin_shell: str,
    loop_args: str,
) -> str:
    """Build a start, exact listener proof, generation record, and sweep."""
    unit_path = f"$HOME/.config/systemd/user/{unit_name}"
    fenced_unit = ""
    if generation is not None:
        directive = shlex.quote(systemd_generation_invalidator())
        fenced_unit = f'''\
UNIT="{unit_path}"
if [ -f "$UNIT" ] && ! grep -Fqx -- {directive} "$UNIT"; then
  echo UNKNOWN_UNFENCED_SYSTEMD
  exit 75
fi
'''
    script = managed_start_guard(config, host, pythonpath) + fenced_unit + start_generation_shell(
        generation, pattern, host.address, unit_name
    ) + owned_listener_probe_shell(host, pattern) + f"""
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX={shlex.quote(SOURCE_ONLY_PYCACHE_PREFIX)}
UNIT="{unit_path}"
if [ -f "$UNIT" ]; then
  systemctl --user set-environment PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX={shlex.quote(SOURCE_ONLY_PYCACHE_PREFIX)}
  systemctl --user daemon-reload
  systemctl --user stop {unit_name} 2>/dev/null || true
  pkill -f -- {shlex.quote(pattern)} 2>/dev/null || true
  sleep 0.5
  systemctl --user start {unit_name}
  sleep 1.5
  if [ "$(systemctl --user is-active {unit_name} 2>/dev/null)" = active ] && \
      dgxm_owned_listener_ready; then
    DGXM_START_MODE=systemd
    if dgxm_record_generation; then echo STARTED_SYSTEMD
    else DGXM_RECORD_STATUS=$?; [ "$DGXM_RECORD_STATUS" -eq 10 ] || exit "$DGXM_RECORD_STATUS"; fi
  else
    if dgxm_quarantine_failed_start systemd; then echo FAILED_SYSTEMD
    else DGXM_SETTLE_STATUS=$?; exit "$DGXM_SETTLE_STATUS"; fi
    journalctl --user -u {unit_name} -n 5 --no-pager 2>/dev/null || true
  fi
else
  umask 077
  mkdir -p "$(dirname "{log_path}")"
  chmod 700 "$(dirname "{log_path}")"
  chmod 600 "{log_path}" "{log_path}.1" 2>/dev/null || true
  if [ -f "{log_path}" ] && [ "$(wc -c < "{log_path}" 2>/dev/null || echo 0)" -gt 52428800 ]; then
    mv -f "{log_path}" "{log_path}.1"
    chmod 600 "{log_path}.1"
  fi
  : >> "{log_path}"
  chmod 600 "{log_path}"
  pkill -f -- {shlex.quote(pattern)} 2>/dev/null || true
  sleep 0.5
  export PYTHONPATH={pythonpath}
  if [ "$DGXM_SOURCE_KIND" = MANAGED ]; then export PYTHONPATH="$DGXM_START_SOURCE"; fi
  {pybin_shell}
  nohup "$PYBIN" {loop_args} >> "{log_path}" 2>&1 < /dev/null &
  disown
  sleep 1.5
  if dgxm_owned_listener_ready; then
    DGXM_START_MODE=nohup
    if dgxm_record_generation; then echo STARTED_NOHUP
    else DGXM_RECORD_STATUS=$?; [ "$DGXM_RECORD_STATUS" -eq 10 ] || exit "$DGXM_RECORD_STATUS"; fi
  else
    if dgxm_quarantine_failed_start nohup; then echo FAILED_NOHUP
    else DGXM_SETTLE_STATUS=$?; exit "$DGXM_SETTLE_STATUS"; fi
    echo "--- log tail ---"
    tail -n 5 "{log_path}" 2>/dev/null || true
  fi
fi
"""
    return script + actor_sweep.sweep_script(config, host)


def build_stop_script(
    config: ClusterConfig,
    host: HostConfig,
    *,
    pattern: str,
    required_generation: str | None,
    unit_name: str,
) -> str:
    """Build a generic invalidating or generation-bound stop plus sweep."""
    guard = (
        invalidate_generation_shell()
        if required_generation is None
        else require_generation_shell(required_generation, pattern)
    )
    settled = "" if required_generation is None else settle_generation_shell()
    script = guard + f"""
UNIT="$HOME/.config/systemd/user/{unit_name}"
if [ -f "$UNIT" ]; then
  if ! systemctl --user stop {unit_name}; then
    echo FAILED_SYSTEMD_STOP
    exit 1
  fi
  if [ "$(systemctl --user is-active {unit_name} 2>/dev/null)" = active ]; then
    echo FAILED_SYSTEMD_ACTIVE
    exit 1
  fi
  pkill -f -- {shlex.quote(pattern)} 2>/dev/null || true
  sleep 0.5
  if pgrep -f -- {shlex.quote(pattern)} >/dev/null; then
    echo FAILED_STRAY_ACTIVE
    exit 1
  fi
  {settled}
  echo DONE_SYSTEMD
else
  pkill -f -- {shlex.quote(pattern)} 2>/dev/null || true
  sleep 0.5
  if pgrep -f -- {shlex.quote(pattern)} >/dev/null; then
    echo FAILED_NOHUP_ACTIVE
    exit 1
  fi
  {settled}
  echo DONE_NOHUP
fi
"""
    return script + actor_sweep.sweep_script(config, host, all_loop_children=True)
