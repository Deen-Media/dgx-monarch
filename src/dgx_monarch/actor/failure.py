"""Failure-path cleanup for actor endpoints.

Any state mutation that can fail cleans up after itself (docs/DESIGN.md
section 5.2). A sampler endpoint that dies mid-denoise would otherwise leave
the actor holding LoRA-patched weights plus a full allocator cache, and
nothing reclaims that memory until a later render succeeds, so the reclaim
runs on the exception path.
"""
from __future__ import annotations

import functools
import gc

from ..log import get_logger
from ..transfer_utils import (
    failure_summary,
    raise_with_distinct_cause,
    reconcile_error,
)

log = get_logger(__name__)


class CudaContextPoisonedError(RuntimeError):
    """This worker's CUDA context died on a sticky device fault.

    It must stay untagged (no refusal class): the decision is host-local, so
    a mixed fleet can leave the healthy rank inside a collective, and the
    lease doctrine's rank-symmetry rule says such arms abandon rather than
    retire consumed. One Recycle respawns clean workers.
    """


_POISON_TEXT = (
    "this worker's CUDA context is poisoned: an earlier render died on a "
    "sticky device fault (a misaligned address or illegal access class "
    "error), and every later kernel in this process would fail or return corrupt "
    "results. Nothing was run. Recycle the mesh to respawn the workers "
    "(docs/TROUBLESHOOTING.md #86)."
)


def _cuda_context_dead() -> bool:
    """Probe whether this process's CUDA context still accepts work.

    Sticky faults poison the context permanently; benign failures (typed
    refusals, OOM) leave it serviceable, so the probe stays False for them.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        return False
    except Exception:
        return True


def _note_poison_if_context_dead(worker) -> None:
    try:
        if getattr(worker, "_cuda_context_poisoned", False):
            return
        if _cuda_context_dead():
            worker._cuda_context_poisoned = True
            log.error(
                "CUDA context poisoned by a sticky device fault; this worker "
                "refuses further GPU endpoints until the mesh is recycled "
                "(docs/TROUBLESHOOTING.md #86)")
    except BaseException:
        pass


def cleanup_on_failure(func):
    """Decorator for GPUWorker sampler/loader endpoints.

    On failure: clean up the active patchers, empty the allocator cache, and
    run gc. Ordinary cleanup errors stay secondary; operational cancellation
    wins with exact identity so the driver can stop promptly.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_cuda_context_poisoned", False):
            raise CudaContextPoisonedError(_POISON_TEXT)
        try:
            return func(self, *args, **kwargs)
        except BaseException as primary:
            failure: BaseException = primary
            failure_cause: BaseException | None = None
            try:
                import comfy.model_management as mm

                store = getattr(self, "store", None)
                if store is not None:
                    store.cleanup_active()
                mm.soft_empty_cache()
            except BaseException as cleanup_exc:
                failure, failure_cause = reconcile_error(
                    failure,
                    cleanup_exc,
                    "post-failure cleanup also failed",
                )
                try:
                    log.warning(
                        "post-failure cleanup itself failed: %s",
                        failure_summary(cleanup_exc),
                    )
                except BaseException:
                    pass
            try:
                gc.collect()
            except BaseException as gc_exc:
                failure, failure_cause = reconcile_error(
                    failure,
                    gc_exc,
                    "post-failure garbage collection also failed",
                )
            # A sticky device fault outlives this exception: probe the
            # context once per failure so the next endpoint refuses typed
            # instead of computing on a dead context or blocking its peer.
            _note_poison_if_context_dead(self)
            if failure is primary and failure_cause is None:
                raise
            raise_with_distinct_cause(failure, failure_cause)

    return wrapper
