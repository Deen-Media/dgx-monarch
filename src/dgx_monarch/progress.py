"""Per-step progress streamed from the result leader actors to the ComfyUI
progress bar of a cluster render (DESIGN.md §5.8).
"""
from __future__ import annotations

import threading
import time
from typing import Any, cast

from .error_utils import prefer_error, raise_with_distinct_cause, reconcile_error
from .log import get_logger

log = get_logger(__name__)


def _thread_started(thread: threading.Thread) -> bool:
    """Whether this thread got far enough through start() to join.

    ident alone misses a start() that an exception interrupted after the OS
    thread existed. CPython's private _started event covers that case.
    typeshed does not declare _started, so the read goes through a cast.
    """
    return thread.ident is not None or cast(Any, thread)._started.is_set()


class ProgressReceiver:
    """Driver side: open a Monarch channel whose port the caller passes to the
    sample call as progress_port, and pump step messages into a comfy ProgressBar
    on a background thread.

    Each result leader (one per data-parallel group, actor/sampling.py
    is_result_leader) sends {"step": int, "total": int} per denoise step, then
    {"done": True}. A leader given a one-element custom sigma schedule returns
    before sampling and sends neither. The pump stops at the first done.
    """

    def __init__(self, total_steps: int, on_step=None, on_cancel=None, *,
                 track_render: bool = False, step_token: object | None = None):
        # Channel.open waits for __enter__ so a caller can publish this receiver
        # into its cleanup guard before the first channel or thread side effect.
        self.port = None
        self._receiver: Any = None
        self._total = total_steps
        # Fleet submits outside render_submit/PendingRender, so it opts in
        # to receiver-owned telemetry lifetime. Other callers keep theirs.
        if track_render and step_token is not None:
            raise ValueError("a progress receiver owns its telemetry token or borrows one, not both")
        self._telemetry_token = object() if track_render else None
        # A borrowed token (the normal render guard's) attributes steps only:
        # its owner starts and retires it, and telemetry ignores a step that
        # arrives after retirement instead of reviving an ownerless render.
        self._step_token = self._telemetry_token if track_render else step_token
        # Optional aggregate hook: when set, each step message goes to
        # on_step(msg) instead of a per-render ProgressBar, so the pipeline and
        # fleet nodes can fold several concurrent renders into one bar.
        self._on_step = on_step
        self._on_cancel = on_cancel
        self._thread: threading.Thread | None = None
        self._cancel_thread: threading.Thread | None = None
        self._closed = threading.Event()
        # Monotonic stamp of the last leader message. The driver's sample wait
        # reads it through activity() to tell a slow step from a dead fleet
        # (mesh_lease.collect_with_liveness, docs/TROUBLESHOOTING.md #86).
        self._last_activity = time.monotonic()

    def activity(self) -> float:
        """Monotonic time of the last progress message (or of construction)."""
        return self._last_activity

    def __enter__(self):
        try:  # first-render notice tap: never allowed to break the bar
            from .first_render import note_proof_render

            note_proof_render()
        except Exception:
            pass
        if self._telemetry_token is not None:
            from .telemetry import render_progress

            render_progress.start(self._total, token=self._telemetry_token)
        from monarch.actor import Channel

        self.port, self._receiver = Channel.open()
        pbar = None
        if self._on_step is None:
            try:
                from comfy.utils import ProgressBar

                pbar = ProgressBar(self._total)
            except Exception:
                pbar = None

        def pump():
            while True:
                try:
                    msg = self._receiver.recv().get(timeout=3600)
                except Exception as exc:
                    log.warning("progress stream ended abnormally: %r", exc)
                    return
                self._last_activity = time.monotonic()
                if not isinstance(msg, dict) or msg.get("done"):
                    return
                try:  # telemetry tap (dgxm top): never allowed to break the bar
                    from .telemetry import render_progress

                    render_progress.step(
                        msg.get("step", 0), msg.get("total", self._total),
                        token=self._step_token)
                except Exception:
                    pass
                if self._on_step is not None:
                    self._on_step(msg)
                elif pbar is not None:
                    pbar.update_absolute(msg.get("step", 0), msg.get("total", self._total))

        self._thread = threading.Thread(target=pump, name="dgxm-progress", daemon=True)
        self._thread.start()
        if self._on_cancel is not None:
            def monitor_cancel():
                try:
                    import comfy.model_management as mm
                except Exception as exc:
                    log.warning("render cancel monitor stopped; comfy.model_management did not import: %r", exc)
                    return
                while not self._closed.wait(0.1):
                    try:
                        if mm.processing_interrupted():
                            self._on_cancel()
                            return
                    except Exception as exc:
                        log.warning("render cancel monitor stopped; a cancel may not reach the workers: %r", exc)
                        return

            self._cancel_thread = threading.Thread(
                target=monitor_cancel, name="dgxm-cancel", daemon=True)
            self._cancel_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # A leader's {"done": True} ends the pump; a leader that dies, hangs, fails
        # or returns before sampling (a zero-step custom schedule) or cannot send
        # never delivers it. The driver holds the port too, so send the sentinel
        # here, or each such render leaks a thread blocked up to the 3600 s recv
        # timeout. Never block teardown.
        cleanup_error: BaseException | None = None
        try:
            self._closed.set()
            if self.port is not None:
                try:
                    self.port.send({"done": True})
                except Exception:
                    pass
            if self._thread is not None and _thread_started(self._thread):
                self._thread.join(timeout=2.0)
            if self._cancel_thread is not None and _thread_started(self._cancel_thread):
                self._cancel_thread.join(timeout=2.0)
        except BaseException as close_error:
            cleanup_error = close_error
        if self._telemetry_token is not None:
            try:
                from .telemetry import render_progress

                render_progress.finish(self._telemetry_token)
            except BaseException as telemetry_error:
                cleanup_error = prefer_error(
                    cleanup_error, telemetry_error, "progress telemetry retirement failed")
        if cleanup_error is not None:
            strongest, cause = reconcile_error(
                exc, cleanup_error, "progress receiver cleanup failed")
            raise_with_distinct_cause(strongest, cause)
        return False
