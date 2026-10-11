"""Bounded passive idle guard; no raw process arguments or environments escape."""
from __future__ import annotations

import inspect


class IdleMaintenanceError(RuntimeError):
    """A fixed, non-sensitive maintenance refusal reason."""


def require_idle(proc_root: str = "/proc") -> None:
    """Inspect bounded metadata, then only Python/Monarch/Comfy candidates."""
    import os
    import subprocess
    from pathlib import Path

    def read(path: Path, limit: int) -> bytes:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise OSError("bounded process observation exceeded")
        return data

    unknown = False
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.geteuid():
                continue
            state = read(entry / "stat", 4096).rsplit(b")", 1)[1].split()[0]
            if state in {b"Z", b"X", b"x"}:
                continue
            comm = read(entry / "comm", 256).strip().lower()
            if not comm:
                raise OSError("missing process name")
            argv = None
            try:
                executable = os.path.basename(os.readlink(entry / "exe")).lower().encode()
                names: tuple[bytes, ...] = (comm, executable)
            except PermissionError:
                # Ordinary non-dumpable user daemons may hide exe. Use bounded
                # argv signatures, never a daemon allowlist or a skipped error.
                raw = read(entry / "cmdline", 65536)
                if not raw or not raw.endswith(b"\0") or not raw.split(b"\0", 1)[0]:
                    raise OSError("malformed process arguments") from None
                argv = raw.split(b"\0")
                names = (comm, os.path.basename(argv[0]).lower(), *(arg.lower() for arg in argv[1:]
                         if arg.startswith((b"monarch.", b"dgx_monarch."))
                         or arg.endswith(b"main.py")))
            if not any(any(label in name for label in (b"python", b"monarch", b"hyperactor", b"comfy", b"main.py"))
                       for name in names):
                continue
            if argv is None:
                raw = read(entry / "cmdline", 65536)
                if not raw or not raw.endswith(b"\0") or not raw.split(b"\0", 1)[0]:
                    raise OSError("malformed process arguments") from None
                argv = raw.split(b"\0")
            if any(argv[i:i + 2] == [b"-m", b"dgx_monarch.cli.worker_loop"] for i in range(len(argv) - 1)):
                continue  # Persistent coordinator; actors are inspected independently.
            is_actor = any(argv[i:i + 2] == [b"-m", b"monarch._src.actor.bootstrap_main"]
                           for i in range(len(argv) - 1))
            is_comfy = any(part.endswith(b"/main.py") or part == b"main.py" for part in argv)
            is_comfy |= b"comfyui" in argv or b"comfy" in argv
            if is_actor or is_comfy:
                raise IdleMaintenanceError("active_comfy_or_actor")
            env = read(entry / "environ", 131072).split(b"\0")
            if any(item.startswith(b"HYPERACTOR_MESH_BOOTSTRAP_MODE=") for item in env):
                raise IdleMaintenanceError("active_comfy_or_actor")
        except FileNotFoundError:
            if entry.exists():
                unknown = True
        except (OSError, IndexError):
            unknown = True
    if unknown:
        raise IdleMaintenanceError("process_observation_unknown")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise IdleMaintenanceError("gpu_observation_unknown")
    if result.stdout.strip():
        raise IdleMaintenanceError("gpu_client_active")


PROBE_SOURCE = inspect.getsource(IdleMaintenanceError) + "\n" + inspect.getsource(require_idle)
