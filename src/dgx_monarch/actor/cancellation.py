"""Out-of-band cancellation for active and actor-mailbox-queued renders."""
from __future__ import annotations

import socket
import threading
from typing import Any

from monarch.actor import concurrent_endpoint, endpoint

from ..loader_options import validate_sample_request_weight_dtypes


def raise_if_sample_cancelled(cancel_event: threading.Event | None) -> None:
    """Stop pre-denoise work at explicit, bounded cancellation boundaries."""
    if cancel_event is not None and cancel_event.is_set():
        from .sampling import RenderCancelledError

        raise RenderCancelledError(
            "distributed render was cancelled before denoising")


class CancellableSampleMixin:
    _gpu_lock: Any
    _on_gpu: Any
    _sample_impl: Any
    rank: int | None

    def _init_cancellation(self) -> None:
        self._cancel_lock = threading.Lock()
        self._active_cancel: tuple[str, threading.Event] | None = None
        self._pending_cancels: dict[str, None] = {}

    @concurrent_endpoint
    async def sample(self, request: dict, progress_port=None) -> dict:
        """Run on the GPU thread while the actor loop remains cancellable."""
        validate_sample_request_weight_dtypes(request)
        async with self._gpu_lock:
            render_id = str(request.get("_dgxm_render_id", ""))
            cancel_event = threading.Event()
            with self._cancel_lock:
                if render_id in self._pending_cancels:
                    cancel_event.set()
                    self._pending_cancels.pop(render_id, None)
                self._active_cancel = (render_id, cancel_event)
            try:
                return await self._on_gpu(
                    self._sample_impl, request, progress_port, cancel_event)
            finally:
                with self._cancel_lock:
                    if (self._active_cancel is not None
                            and self._active_cancel[1] is cancel_event):
                        self._active_cancel = None
                    self._pending_cancels.pop(render_id, None)

    @endpoint
    async def cancel_sample(self, render_id: str) -> dict:
        """Record cancellation even when the target is still queued."""
        render_id = str(render_id)
        with self._cancel_lock:
            self._pending_cancels[render_id] = None
            while len(self._pending_cancels) > 256:
                self._pending_cancels.pop(next(iter(self._pending_cancels)))
            active = self._active_cancel
            if active is not None and active[0] == render_id:
                active[1].set()
        return {"host": socket.gethostname(), "rank": self.rank, "cancelled": True}
