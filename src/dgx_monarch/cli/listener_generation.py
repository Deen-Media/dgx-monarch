"""Identify the process holding a worker's listener and its generation.

A LISTEN row alone does not show whether a worker restarted. Hash the boot ID,
owner PID and process start time into a short generation token that changes
with listener ownership without disclosing host identity.

Read procfs, the generation marker and ``systemctl --user show`` fields
passively; never open the worker's protocol socket.
"""
from __future__ import annotations

import subprocess
from collections.abc import Mapping
from typing import Any

from ..config import ClusterConfig, HostConfig
from .lifecycle_generation import GENERATION_FORMAT, GENERATION_MARKER_REL
from .probe_certainty import OK, UNOBSERVED, WARN
from .worker_health import _UNIT_NAME, _proc_listener_target

GEN_FIELD = "listener_gen"
LINE_PREFIX = GEN_FIELD + "="
ENTRY = "docs/TROUBLESHOOTING.md #101"
NAME_SUFFIX = "listener generation"

_BODY = '''
import hashlib as _gh, os as _go, pathlib as _gp, subprocess as _gs

def _gen_owner(proc_path, endpoint):
    inodes = set()
    for _row in _gp.Path(proc_path).read_text().splitlines()[1:]:
        _parts = _row.split()
        if len(_parts) > 9 and _parts[1] == endpoint and _parts[3] == "0A":
            inodes.add(_parts[9])
    if not inodes:
        return ""
    for _entry in _gp.Path("/proc").iterdir():
        if not _entry.name.isdigit():
            continue
        try:
            _items = list((_entry / "fd").iterdir())
        except OSError:
            continue
        _owned = set()
        for _item in _items:
            try:
                _link = _go.readlink(_item)
            except OSError:
                continue
            if _link.startswith("socket:["):
                _owned.add(_link[8:-1])
        if inodes & _owned:
            return _entry.name
    return None

def _gen_unit():
    try:
        _row = _gs.run(
            ["systemctl", "--user", "show", _gen_unit_name, "--no-pager",
             "--property=NRestarts", "--property=ExecMainStartTimestamp",
             "--property=ActiveState"],
            capture_output=True, text=True, timeout=5)
    except Exception:
        return "unknown", "unknown", "unknown"
    if _row.returncode:
        return "absent", "absent", "absent"
    _fields = {}
    for _entry in _row.stdout.splitlines():
        _key, _sep, _value = _entry.partition("=")
        if _sep:
            _fields[_key] = "_".join(_value.split()) or "none"
    return (_fields.get("ActiveState", "unknown"),
            _fields.get("NRestarts", "unknown"),
            _fields.get("ExecMainStartTimestamp", "unknown"))

def _gen_marker(identity):
    try:
        _lines = (_gp.Path.home() / _gen_marker_rel).read_text().splitlines()
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unreadable"
    if len(_lines) != 4 or _lines[0] != _gen_marker_format:
        return "malformed"
    return "match" if _lines[3] == identity else "stale"

def _gen_report():
    _pid = _gen_owner(_gen_proc, _gen_endpoint)
    _state, _restarts, _started = _gen_unit()
    _tail = f"unit_state={_state} unit_restarts={_restarts} unit_started={_started}"
    if _pid == "":
        return f"{_gen_prefix}none loop_pid=none age_s=none listening=false {_tail} marker=none"
    if _pid is None:
        return f"{_gen_prefix}unknown loop_pid=unknown age_s=unknown listening=true {_tail} marker=unknown"
    _born = (_gp.Path("/proc") / _pid / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    _boot = _gp.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    _up = float(_gp.Path("/proc/uptime").read_text().split()[0])
    _age = _up - int(_born) / _go.sysconf("SC_CLK_TCK")
    _gen = _gh.sha256(f"{_boot}|{_pid}|{_born}".encode()).hexdigest()[:12]
    return (f"{_gen_prefix}{_gen} loop_pid={_pid} age_s={_age:.0f} listening=true "
            f"{_tail} marker={_gen_marker(_pid + ':' + _born)}")

try:
    print(_gen_report())
except Exception as _exc:
    print(f"{_gen_prefix}unknown error={type(_exc).__name__}")
'''


def probe_source(address: str) -> str:
    """The stdlib-only snippet one host runs, with its endpoint set in the header."""
    proc_path, endpoint = _proc_listener_target(address)
    header = (
        f"_gen_proc = {proc_path!r}\n"
        f"_gen_endpoint = {endpoint!r}\n"
        f"_gen_prefix = {LINE_PREFIX!r}\n"
        f"_gen_unit_name = {_UNIT_NAME!r}\n"
        f"_gen_marker_rel = {GENERATION_MARKER_REL!r}\n"
        f"_gen_marker_format = {GENERATION_FORMAT!r}\n"
    )
    return header + _BODY


def parse(text: str) -> dict[str, str]:
    """Read the probe's line into fields; an empty dict means no line.

    The leading field is renamed to `gen`, which is what the attach trace and
    the doctor row both call it. The wire keeps the longer spelling so the
    prefix stays unique among the several lines one host script prints.
    """
    if not text.startswith(LINE_PREFIX):
        return {}
    fields: dict[str, str] = {}
    for part in text.split():
        key, separator, value = part.partition("=")
        if separator and key:
            fields.setdefault("gen" if key == GEN_FIELD else key, value)
    return fields


def select(lines: list[str]) -> dict[str, str]:
    """Pick this probe's line out of the several a per-host script prints."""
    for text in lines:
        if text.startswith(LINE_PREFIX):
            return parse(text)
    return {}


def probe(
    config: ClusterConfig,
    host: HostConfig,
    *,
    runner: Any,
    timeout: float = 3.0,
) -> dict[str, str]:
    """Probe one host; return transport and execution faults as result fields.

    Attach recovery calls this probe, so an unreachable host must add only a
    bounded wait and must not raise another error.
    """
    from .lifecycle import _pybin_shell

    script = (f"{_pybin_shell(config.python_bin)}\n\"$PYBIN\" - <<'DGXMGEN'\n"
              f"{probe_source(host.address)}\nDGXMGEN\n")
    try:
        result = runner(config, host, script, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"gen": "unknown", "error": type(exc).__name__}
    if result.returncode != 0:
        return {"gen": "unknown", "error": f"exit_{result.returncode}"}
    fields = select(result.stdout.splitlines())
    return fields or {"gen": "unknown", "error": "no_line"}


def fleet(config: ClusterConfig, *, runner: Any,
          timeout: float = 3.0) -> dict[str, dict[str, str]]:
    """One generation reading per configured loop, keyed by worker address."""
    return {host.address: probe(config, host, runner=runner, timeout=timeout)
            for host in config.hosts}


def unobserved_row(host_name: str,
                   reason: str) -> tuple[str, str, str, str, bool]:
    """Build the generation row for a failed or unanswered host check.

    Include failed hosts so doctor reports their unknown generation explicitly.
    Reuse the probe's fault fields for ``doctor_row``.
    """
    return doctor_row(host_name, {"gen": "unknown", "error": reason})


def doctor_row(host_name: str,
               fields: Mapping[str, str]) -> tuple[str, str, str, str, bool]:
    """Return (status, name, detail, reason, critical) for listener generation.

    This advisory row identifies the listener and its age. The Worker service row
    already fails when the loop is not listening.
    """
    name = f"{host_name} {NAME_SUFFIX}"
    generation = str(fields.get("gen", ""))
    if not fields or generation in ("", "unknown"):
        reason = str(fields.get("error", "the probe printed no reading"))
        return (WARN, name,
                f"unobserved: {reason}; the live listener generation is unknown "
                f"({ENTRY})", UNOBSERVED, False)
    if generation == "none":
        return (WARN, name,
                f"no process holds the worker LISTEN socket ({ENTRY})", "", False)
    detail = (
        f"gen={generation} pid={fields.get('loop_pid', 'unknown')} "
        f"age_s={fields.get('age_s', 'unknown')} "
        f"restarts={fields.get('unit_restarts', 'unknown')} "
        f"started={fields.get('unit_started', 'unknown')} "
        f"marker={fields.get('marker', 'unknown')}"
    )
    return OK, name, detail, "", False
