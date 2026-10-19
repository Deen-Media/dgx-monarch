"""Read where each host's Worker source link points, and the package version there.

Guided setup puts remote Workers in ``releases/s-...`` slots and a verified
update puts every Worker in a ``releases/u-...`` slot. Plain ``dgxm update``
never changes a slot, so it must refuse before it fetches, and Doctor must name
a slot whose package version differs from the driver's (TROUBLESHOOTING #118).

The probe reads only what every install has had since guided setup first wrote
a slot: the ``~/.local/share/dgx-monarch/src`` link and the slot's
``dgx_monarch/__init__.py``, parsed rather than imported, as the verified
update's attestation reads it. It needs no config key, so a newer Doctor
spawned by an older updater can run it against Workers the older code wrote.
"""
from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable

from ..config import ClusterConfig, HostConfig

RELEASE_LINE = "dgxm_worker_release="
TROUBLESHOOTING = "docs/TROUBLESHOOTING.md #118"

# Printed as one line: PLAIN (no link), OUTSIDE (a link that does not resolve
# into releases/, such as guided setup's link to a colocated checkout), or
# SLOT <token> <version>, where an unreadable version reads UNREADABLE.
PROBE_SOURCE = r'''
import ast,pathlib,re
base=pathlib.Path.home()/'.local/share/dgx-monarch'
src=base/'src'
def version(site):
 try: tree=ast.parse((site/'dgx_monarch'/'__init__.py').read_text('utf-8'))
 except (OSError,ValueError,SyntaxError): return 'UNREADABLE'
 found=[node.value.value for node in tree.body if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='__version__' and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)]
 return found[0] if len(found)==1 and re.fullmatch(r'[0-9A-Za-z.+!_-]{1,64}',found[0]) else 'UNREADABLE'
if not src.is_symlink(): print('dgxm_worker_release=PLAIN')
else:
 target=src.resolve(); releases=(base/'releases').resolve()
 if not target.is_relative_to(releases): print('dgxm_worker_release=OUTSIDE')
 else:
  parts=target.relative_to(releases).parts
  token=parts[0] if parts and re.fullmatch(r'[us]-[0-9a-f]+-[0-9a-f]+',parts[0]) else 'UNKNOWN'
  print('dgxm_worker_release=SLOT '+token+' '+version(target))
'''

# `-I` ignores the PYTHONPATH a caller exported for its own probe and `-S`
# skips site packages, so no installed dgx_monarch code is imported.
PROBE_COMMAND = f'"$PYBIN" -I -S -B -c {shlex.quote(PROBE_SOURCE)}'

Runner = Callable[..., subprocess.CompletedProcess[str]]


def parse(lines: list[str]) -> tuple[str, str, str] | None:
    """(kind, token, version) from the probe's line, or None when it printed none."""
    line = next((line for line in lines if line.startswith(RELEASE_LINE)), None)
    if line is None:
        return None
    fields = line[len(RELEASE_LINE):].split()
    if fields in (["PLAIN"], ["OUTSIDE"]):
        return fields[0], "", ""
    if len(fields) == 3 and fields[0] == "SLOT":
        return "SLOT", fields[1], fields[2]
    return None


def release_slot_hosts(
    config: ClusterConfig, runner: Runner | None = None,
) -> tuple[list[tuple[HostConfig, str]], list[HostConfig]]:
    """Hosts whose source link resolves into releases/ (with the slot token), and hosts unobserved.

    Read-only, and run without the lifecycle lock, as Doctor runs its probes.
    A host that did not answer, or answered without a verdict, stays apart
    from one that answered "no slot".
    """
    from .lifecycle import _pybin_shell, run_on_host

    run = runner or run_on_host
    slots: list[tuple[HostConfig, str]] = []
    unobserved: list[HostConfig] = []
    for host in config.hosts:
        try:
            result = run(config, host, f"{_pybin_shell(config.python_bin)}\n{PROBE_COMMAND}\n", timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            unobserved.append(host)
            continue
        verdict = parse(result.stdout.splitlines()) if result.returncode == 0 else None
        if verdict is None:
            unobserved.append(host)
        elif verdict[0] == "SLOT":
            slots.append((host, verdict[1]))
    return slots, unobserved
