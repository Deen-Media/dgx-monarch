"""Run host-local actor sweeps for ``dgxm down``, ``dgxm up``, and ``dgxm reap``.

``actor_reaper`` owns the policy and runs on each target as a stdlib-only
module. This module builds its invocation and parses its summary into a
per-host verdict. The standalone ``reap`` command also covers local-mode
hosts, which have no configured workers for ``down`` or ``up`` to visit.
"""
from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import Any

from . import actor_reaper

SWEEP_TIMEOUT_S = 45


def sweep_script(
    config: Any,
    host: Any,
    *,
    all_loop_children: bool = False,
    grace_s: float = 0.0,
    dry_run: bool = False,
) -> str:
    """Build the host invocation using lifecycle_scripts' nohup environment setup.

    ``set -u``, ``_pybin_shell`` and exported PYTHONPATH are required because a
    systemd unit's Environment= does not reach a separate ``bash -s`` script.
    Quote every interpolated value.
    """
    from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX
    from . import lifecycle

    pythonpath = (
        shlex.quote(str(lifecycle._pkg_src_dir())) if lifecycle._is_local(host)
        else f'"$HOME/{lifecycle._MANAGED_SRC_REL}"'
    )
    flags = ["--sweep", "--loop-address", shlex.quote(host.address), "--grace", f"{grace_s:g}"]
    if all_loop_children:
        flags.append("--all-loop-children")
    if dry_run:
        flags.append("--dry-run")
    return f"""
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX={shlex.quote(SOURCE_ONLY_PYCACHE_PREFIX)}
export PYTHONPATH={pythonpath}
{lifecycle._pybin_shell(config.python_bin)}
"$PYBIN" -m dgx_monarch.cli.actor_reaper {" ".join(flags)}
"""


def sweep_host(
    config: Any,
    host: Any,
    *,
    runner: Callable[..., Any],
    all_loop_children: bool = False,
    grace_s: float = 0.0,
    dry_run: bool = False,
) -> bool:
    """Sweep one host; return False on failure or surviving targets.

    Two compatibility cases are non-fatal: a missing module and exit 0 without a
    summary. Every other missing summary is a failure because the summary is
    emitted only after the SIGTERM/SIGKILL sequence completes. A nonzero exit or
    timeout therefore leaves the survivor count unknown. The signal waits total
    ``term_wait_s + kill_wait_s``, below ``SWEEP_TIMEOUT_S``; reaching that timeout
    can indicate an unresponsive host or uninterruptible target.
    """
    import subprocess

    script = sweep_script(
        config, host, all_loop_children=all_loop_children,
        grace_s=grace_s, dry_run=dry_run)
    try:
        result = runner(config, host, script, timeout=SWEEP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"    {host.name}: actor sweep did not finish within {SWEEP_TIMEOUT_S}s, so "
              "stray actor processes on this host are unaccounted for")
        return False
    except OSError as exc:
        print(f"    {host.name}: actor sweep unavailable ({exc})")
        return False
    return report_result(host, result, dry_run=dry_run)


def report_result(host: Any, result: Any, *, dry_run: bool = False) -> bool:
    """Print one host's sweep result; False when it failed or, outside a dry run, left a survivor.

    `lifecycle` also calls it on the sweep that `lifecycle_scripts` appends to
    each stop and start script.
    """
    stdout = getattr(result, "stdout", "") or ""
    stderr = (getattr(result, "stderr", "") or "").strip()
    code = int(getattr(result, "returncode", 0) or 0)
    try:
        summary = actor_reaper.parse_summary(stdout)
    except ValueError as exc:
        lines = stderr.splitlines() or stdout.splitlines()
        detail = lines[-1] if lines else "no output"
        print(f"    {host.name}: actor sweep FAILED ({exc}; exit {code}; "
              f"last output: {detail}), so stray actor processes on this host "
              "are unaccounted for")
        return False
    if summary is None:
        note = stderr.splitlines()[-1] if stderr else "no summary line"
        if "No module named" in stderr:
            # A mixed-version fleet during rollout must not fail `dgxm down`.
            note = "host package predates the actor sweep; `dgxm up` syncs it"
        elif code:
            print(f"    {host.name}: actor sweep FAILED (exit {code}: {note}), so stray "
                  "actor processes on this host are unaccounted for")
            return False
        print(f"    {host.name}: actor sweep skipped ({note})")
        return True
    print(f"    {host.name}: actor sweep {actor_reaper.summary_fields(summary)}")
    if summary["left"] and not dry_run:
        print(f"    {host.name}: {summary['left']} actor process(es) survived SIGKILL, so they "
              "are stuck in the kernel; a reboot is the only remedy")
    if code:
        note = stderr.splitlines()[-1] if stderr else "non-zero exit after the summary"
        print(f"    {host.name}: actor sweep FAILED (exit {code}: {note}), so stray "
              "actor processes on this host are unaccounted for")
    return not code and (dry_run or not summary["left"])


def reap_command(config: Any, *, dry_run: bool = False, grace_s: float = 0.0) -> bool:
    """Sweep every configured host and the local host once.

    A driver may not be a configured worker, and local-mode configurations have no
    workers. Skip the separate local pass only if a configured host is local.
    """
    from . import lifecycle

    hosts = tuple(config.hosts) if config is not None else ()
    ok = True
    for host in hosts:
        ok &= sweep_host(
            config, host, runner=lifecycle.run_on_host,
            grace_s=grace_s, dry_run=dry_run)
    if not any(lifecycle._is_local(host) for host in hosts):
        summary = actor_reaper.run_sweep(grace_s=grace_s, dry_run=dry_run)
        print(f"  local: actor sweep {actor_reaper.summary_fields(summary)}")
        ok = ok and (dry_run or not summary["left"])
    return ok
