"""Driver-side runtime primitives shared by the mesh lifecycle.

These helpers cover environment discovery, bounded creation waits, worker-loop
attachment and the event-loop bridge. They own no mesh lifecycle state and no
supervision reconciliation.
"""
from __future__ import annotations

import hashlib
import math
import os
import re

from .config_schema import MAX_GPUS_PER_HOST, MAX_WORLD_SIZE

_MESH_CREATION_TIMEOUT_ENV = "DGXM_MESH_CREATION_TIMEOUT"
_DEFAULT_MESH_CREATION_TIMEOUT_S = 180.0
_MAX_MESH_CREATION_TIMEOUT_S = 3600.0

# The variable monarch reads for its own config-push budget. dgx_monarch's
# package import setdefaults it, so it is always set by the time this module
# loads; an operator who exports a larger one before launch owns it.
ATTACH_CONFIG_TIMEOUT_ENV = "HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT"
_ATTACH_INIT_FLOOR_S = 70
_ATTACH_INIT_MARGIN_S = 10
# monarch parses this budget with humantime, which spells minutes `3m` and
# milliseconds `500ms`. A reader of bare seconds would drop a real three minute
# budget to the floor and put the wait back inside monarch's own window in
# silence. This reader takes one number and one unit; a compound value such as
# `2m30s` keeps the floor.
_DURATION = re.compile(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)$")
_DURATION_UNITS = {
    "": 1.0, "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 0.001, "msec": 0.001, "msecs": 0.001, "millis": 0.001,
    "millisecond": 0.001, "milliseconds": 0.001,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
}


def _budget_seconds(named: str) -> float | None:
    """One humantime duration as seconds, or None when this reader cannot read it."""
    match = _DURATION.match(named.strip().lower())
    if match is None or match.group(2) not in _DURATION_UNITS:
        return None
    return float(match.group(1)) * _DURATION_UNITS[match.group(2)]


def _attach_init_wait() -> int:
    """Return a wait bound above the configured Monarch config-push budget.

    The per-host Monarch timeout should surface before a generic join timeout
    (docs/TROUBLESHOOTING.md #1). Read the active budget so operator overrides
    also extend this wait. An unreadable value retains the default floor.
    """
    budget = _budget_seconds(os.environ.get(ATTACH_CONFIG_TIMEOUT_ENV, ""))
    if budget is None or not math.isfinite(budget) or budget <= 0:
        return _ATTACH_INIT_FLOOR_S
    return max(_ATTACH_INIT_FLOOR_S, math.ceil(budget) + _ATTACH_INIT_MARGIN_S)


ATTACH_INIT_WAIT_S = _attach_init_wait()

# The scratch-thread join must outlast the future's own deadline. When both
# bounds are equal, which one fires is a race, and a join TimeoutError would
# replace the future's typed one while a live thread still drives the call
# (transfer_utils.rdma_outer_timeout adds transfer.RDMA_READ_OUTER_MARGIN_S to
# the RDMA read's scratch-thread join for the same reason).
FUTURE_GET_JOIN_MARGIN_S = 15.0


def config_fingerprint(path: str | os.PathLike[str] | None) -> str:
    """Content identity for a small cluster TOML (mtime alone is lossy)."""
    if not path:
        return "local"
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unreadable"


def local_worker_args_fingerprint(worker_args: dict) -> str:
    """Content identity for the only part of a config a local mesh consumes.

    A local mesh takes `[worker_args]` and nothing else, so fingerprinting the
    whole file would rebuild a live local fleet over a `[cluster]` edit that
    cannot reach it. This changes exactly when the knobs the workers run with
    change, whichever file supplied them.

    No knobs reads as the plain `local` the no-config path stores: an empty
    table hands the workers what no file hands them, and a different digest
    would recycle a live fleet over a difference the workers cannot see.
    """
    if not worker_args:
        return "local"
    payload = repr(sorted((str(key), repr(value)) for key, value in worker_args.items()))
    return "local:" + hashlib.sha256(payload.encode()).hexdigest()


def mesh_cache_key(source: str, comfy_dir: str, cluster: bool) -> tuple:
    """Stable per source; mutable host/GPU counts must replace the old fleet."""
    if cluster:
        return ("cluster", os.path.realpath(source))
    return ("local", "local")


def detect_comfy_dir(explicit: str = "") -> str:
    if explicit:
        return explicit
    try:
        import folder_paths

        return os.path.dirname(os.path.abspath(folder_paths.__file__))
    except Exception:
        return os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))


def src_pythonpath() -> str:
    # .../dgx-monarch/src/dgx_monarch/mesh_runtime.py -> .../dgx-monarch/src
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bootstrap_proc() -> None:
    """Make the driver-advertised dgx_monarch source importable on a worker."""
    import sys

    for path in filter(None, os.environ.get("DGXM_PYTHONPATH", "").split(os.pathsep)):
        if path not in sys.path:
            sys.path.insert(0, path)
    # Arm the actor-side reaper here, the one piece of dgx-monarch that already
    # runs inside every spawned proc. Imported function-locally and after the
    # path insertion above so the driver-advertised source wins, and from a
    # stdlib-only leaf so SetupActor never pulls torch or comfy in ahead of
    # CUDA init. It stays inert until a client lease arrives.
    from . import actor_lifetime

    actor_lifetime.install()


def attach_once(addresses: list[str]):
    """Attach the worker loops, timing each half of the handshake."""
    from monarch.actor import attach_to_workers

    from . import attach_trace
    from .log import get_logger

    log = get_logger(__name__)
    with attach_trace.half(log, "attach_to_workers", hosts=len(addresses)):
        hosts = attach_to_workers(
            ca="trust_all_connections", workers=addresses, name="dgxm")
    with attach_trace.half(log, "initialized_get", bound_s=ATTACH_INIT_WAIT_S):
        get_off_loop(hosts.initialized, ATTACH_INIT_WAIT_S, "dgxm-attach-init")
    return hosts


def visible_gpu_count() -> int:
    try:
        import torch

        return max(torch.cuda.device_count(), 1)
    except Exception:
        return 1


def validated_local_gpu_count(requested: object, detected: int) -> int:
    value = detected if requested == 0 and not isinstance(requested, bool) else requested
    if (isinstance(value, bool) or not isinstance(value, int) or value < 1
            or value > MAX_GPUS_PER_HOST or value > MAX_WORLD_SIZE):
        raise ValueError(
            f"local gpus_per_host must be an integer in 1..{min(MAX_GPUS_PER_HOST, MAX_WORLD_SIZE)} "
            f"after auto-detection (got {value!r})")
    return value


def mesh_creation_timeout_s(environ: dict[str, str] | None = None) -> float:
    """Return the finite, positive same-key single-flight wait bound."""
    source = os.environ if environ is None else environ
    raw = source.get(
        _MESH_CREATION_TIMEOUT_ENV, str(_DEFAULT_MESH_CREATION_TIMEOUT_S))
    try:
        timeout_s = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_MESH_CREATION_TIMEOUT_ENV} must be a number in "
            f"(0, {_MAX_MESH_CREATION_TIMEOUT_S:g}]") from exc
    if (not math.isfinite(timeout_s) or timeout_s <= 0
            or timeout_s > _MAX_MESH_CREATION_TIMEOUT_S):
        raise ValueError(
            f"{_MESH_CREATION_TIMEOUT_ENV} must be a number in "
            f"(0, {_MAX_MESH_CREATION_TIMEOUT_S:g}]")
    return timeout_s


def wait_for_mesh_creation(condition, creating: set[tuple], key: tuple,
                           error_type: type[RuntimeError]) -> None:
    """Bound one same-key creation wait while the caller holds *condition*."""
    try:
        timeout_s = mesh_creation_timeout_s()
    except ValueError as exc:
        raise error_type(str(exc)) from exc
    if condition.wait_for(lambda: key not in creating, timeout=timeout_s):
        return
    raise error_type(
        f"timed out after {timeout_s:g}s waiting for another caller to finish "
        "creating this mesh; inspect the driver log and retry after it finishes, "
        "or restart ComfyUI if that creator is wedged")


def run_blocking_off_loop(fn, timeout_s: float, thread_name: str):
    """Run a blocking monarch call safely from any context.

    Comfy executes nodes on its event loop, where a blocking monarch
    ``Future.get`` logs a WARNING per call (``get_off_loop`` says why).
    Off-loop, call inline, with no deadline; on-loop, block in a scratch
    thread. A join that times out with the thread alive raises
    ``TimeoutError``: callers must never treat it as success.
    """
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    holder: list = []

    def _capture():
        try:
            holder.append((fn(), None))
        except BaseException as exc:  # re-raised on the calling thread
            holder.append((None, exc))

    worker = threading.Thread(target=_capture, name=thread_name)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive() or not holder:
        raise TimeoutError(f"{thread_name} did not complete within {timeout_s:.0f}s")
    result, exc = holder[0]
    if exc is not None:
        raise exc
    return result


def get_off_loop(future, timeout_s: float, thread_name: str = "dgxm-await"):
    """Wait for a Monarch future off the event-loop thread.

    Monarch warns when ``Future.get`` runs on an asyncio event-loop thread.
    ComfyUI calls from that context use a scratch thread; other callers run
    inline. The caller blocks in either case.

    Preserve the future's timeout and add ``FUTURE_GET_JOIN_MARGIN_S`` to the
    thread join, so the join cannot replace the future's typed timeout. RPC
    callers treat that timeout as an ambiguous mutation.
    """
    return run_blocking_off_loop(
        lambda: future.get(timeout=timeout_s), timeout_s + FUTURE_GET_JOIN_MARGIN_S,
        thread_name)
