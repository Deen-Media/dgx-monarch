"""How a test's rank process ends once its work is done.

A test that runs ranks with a Gloo process group in subprocesses ends each
rank, after its report is written, and after its group is destroyed where the
script destroys it, through the ending here: flush both streams, then leave
through ``os._exit(0)``. Interpreter
finalization then never runs on that success path, and finalization is where
the abort of PyTorch issue 179238 lives. Gloo's run-loop threads outlive
``destroy_process_group``, since the Gloo backend has no shutdown step, and
one that is still releasing a finished collective's work needs the GIL; when
it meets the finalizing interpreter instead, the process ends with "terminate
called without an active exception" (SIGABRT, return code -6) after the
rank's work has passed. A failure before the ending still raises, so its
return code stays non-zero.

``ENDING`` is the source a rank script run by ``python -c`` or from a written
file ends with; ``leave`` is the same ending for a rank function that
``torch.multiprocessing.spawn`` runs. ``tests/test_rank_exit_scan.py`` fails
when a rank script that starts a Gloo group ends any other way.
"""
from __future__ import annotations

import os
import sys
from typing import NoReturn

ENDING = """
import os as _rank_exit_os, sys as _rank_exit_sys
_rank_exit_sys.stdout.flush()
_rank_exit_sys.stderr.flush()
_rank_exit_os._exit(0)
"""


def leave() -> NoReturn:
    """End a spawned rank function's process at once, its streams flushed."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
