"""Worker setup's xFuser check, run before setup changes any state or imports xFuser.

The installed build is fixed for the life of the actor process. Stock 0.7.0
would fail deep in attention setup on an NPU helper released yunchang lacks,
and no render has been checked on any other build, so setup refuses each with
the line that installs stock 0.8.0 (``xfuser_build``).
"""
from __future__ import annotations

import sys

from .. import xfuser_build
from ..adapters.base import UnsupportedModelError
from ..refusal import RefusalClass, refusal


def refuse_unsupported_xfuser() -> None:
    """Raise a class P refusal when this process's xFuser is not one this release runs on.

    A build that runs is pinned for the process, so the Gate identity this
    actor reports stays the build it imports after this check.
    """
    version = xfuser_build.installed_version()
    blocked = xfuser_build.setup_refusal(version, sys.executable)
    if blocked is not None:
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, (
            f"{blocked} Use xFuser {xfuser_build.STOCK} on that box: while no render runs, "
            f"`{xfuser_build.move_command(sys.executable, version)}`, {xfuser_build.RESTART} "
            f"({xfuser_build.ENTRY}).")))
    xfuser_build.pin_for_process(version)
