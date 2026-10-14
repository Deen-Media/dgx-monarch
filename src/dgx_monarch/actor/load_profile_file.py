"""Write Init ``load_profile`` reports to a bounded directory.

With ``load_profile`` on, each ``load_models_gpu`` call that takes 2 s or
more leaves one cProfile report. The reports live in
``~/.cache/dgx-monarch/profiles/``, beside the pack's other cache files, and
the directory keeps the newest ``PROFILE_KEEP``: each write removes the
oldest reports past that count. Every worker process on a host shares the
directory, so the cap is per host and user, not per process.

This module imports no comfy code; ``comfy_bridge`` calls it after a load.
"""
from __future__ import annotations

import io
import os
import pstats
import tempfile
from typing import Any

PROFILE_DIR = "~/.cache/dgx-monarch/profiles"
PROFILE_KEEP = 20
PROFILE_PREFIX = "dgxm-load-profile."
PROFILE_SUFFIX = ".txt"


def profile_dir() -> str:
    """Return the report directory, read at call time so HOME applies."""
    return os.path.expanduser(PROFILE_DIR)


def write_load_profile(profiler: Any, took: float, t0: float) -> str:
    """Write one report, prune the oldest past the cap and return its path.

    The file is private and exclusively created, as ``NamedTemporaryFile``
    makes it. A failed write raises ``OSError``; a failed prune does not,
    because the report it follows is already on disk.
    """
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(30)
    directory = profile_dir()
    os.makedirs(directory, mode=0o700, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=directory,
        prefix=f"{PROFILE_PREFIX}{os.getpid()}.{int(t0)}.",
        suffix=PROFILE_SUFFIX,
        delete=False,
    ) as output_file:
        output_path = output_file.name
        output_file.write(f"load_models_gpu took {took:.1f}s\n")
        output_file.write(buf.getvalue())
    prune_load_profiles(directory, keep=PROFILE_KEEP, spare=output_path)
    return output_path


def prune_load_profiles(directory: str, keep: int, spare: str = "") -> None:
    """Remove the oldest reports so at most *keep* remain, *spare* among them.

    Only regular files with the report prefix and suffix count. Other worker
    processes may prune the same directory at once, so a file that is already
    gone is skipped, and any other error ends the prune quietly.
    """
    spare_name = os.path.basename(spare)
    reports = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                name = entry.name
                if (not name.startswith(PROFILE_PREFIX)
                        or not name.endswith(PROFILE_SUFFIX)
                        or name == spare_name
                        or not entry.is_file(follow_symlinks=False)):
                    continue
                try:
                    stamp = entry.stat(follow_symlinks=False).st_mtime_ns
                except FileNotFoundError:
                    continue
                reports.append((stamp, name, entry.path))
    except OSError:
        return
    reports.sort()
    excess = len(reports) - max(keep - (1 if spare else 0), 0)
    for _stamp, _name, path in reports[:max(excess, 0)]:
        try:
            os.unlink(path)
        except FileNotFoundError:
            continue
        except OSError:
            return
