"""Doctor's xFuser rows: the build here, the build each Worker host runs, and whether they match.

The allowed builds and their words live in ``xfuser_build``. Each row reads
package metadata, never imports xFuser, so doctor runs beside live renders and
registers no torch operator. A FAIL here is a build Worker setup refuses; the
patched build 1.4.0 installed WARNs with the one-line move to stock 0.8.0.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence

from .. import xfuser_build
from .probe_certainty import FAIL, OK, WARN

NAME = "xfuser"
MATCH_NAME = "xfuser build match"
# Must match the probe's output prefix so doctor can select its line.
LINE_PREFIX = "xfuser_build="
UNKNOWN = "unknown"

# Runs under each host's `config.python_bin` inside doctor's per-host heredoc;
# self-contained, with names no other probe uses.
PROBE_SOURCE = """try:
    import importlib.metadata as _xmd
    _xbuild = _xmd.version("xfuser")
except Exception as _xerr:
    _xbuild = "missing" if type(_xerr).__name__ == "PackageNotFoundError" else "unknown"
print(f"xfuser_build={_xbuild}")"""

_STATUS = {xfuser_build.OK: OK, xfuser_build.WARN: WARN, xfuser_build.FAIL: FAIL}


def select(lines: Sequence[str]) -> str:
    """The build a host's probe printed, or ``unknown`` when it printed none."""
    line = next((ln for ln in lines if ln.startswith(LINE_PREFIX)), "")
    return line[len(LINE_PREFIX):].strip() or UNKNOWN


def driver_row(version: str | None = None, python: str | None = None) -> tuple[str, str, str]:
    """The (status, name, detail) row for this interpreter, the driver's."""
    version = xfuser_build.installed_version() if version is None else version
    state, detail = xfuser_build.verdict(version, python or sys.executable)
    return _STATUS[state], NAME, detail


def host_row(host: str, version: str, python: str) -> tuple[str, str, str]:
    """The row for one Worker host, from its probe's reading."""
    name = f"{host} {NAME}"
    if version == UNKNOWN:
        return (WARN, name, f"unobserved: the xFuser probe did not answer on {host}, so the build "
                            f"its Worker setup will meet is unknown ({xfuser_build.ENTRY})")
    state, detail = xfuser_build.verdict(version, python, driver=False)
    return _STATUS[state], name, detail


def match_row(driver: str, hosts: Mapping[str, str]) -> tuple[str, str, str] | None:
    """WARN when the driver and the hosts it read run two builds; None when no host answered."""
    seen = {name: build for name, build in hosts.items() if build != UNKNOWN}
    if not seen:
        return None
    builds = {driver, *seen.values()}
    if len(builds) == 1:
        return OK, MATCH_NAME, driver
    each = ", ".join(f"{name}={build}" for name, build in [("driver", driver), *sorted(seen.items())])
    return (WARN, MATCH_NAME,
            f"two builds: {each}. A Gate PASS names the build it was proven under, so a risky "
            f"render on these boxes refuses until they match; install xFuser {xfuser_build.STOCK} "
            f"on each (docs/TROUBLESHOOTING.md #8)")
