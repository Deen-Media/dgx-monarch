"""Route ComfyUI's stock VAE decodes over the Workers (DESIGN.md section 5.6).

``[worker_args] worker_vae_decode = "auto"`` in cluster.toml turns this on;
absent or ``"off"`` keeps every decode on the driver. The Init node arms it
for the mesh it attaches (``arm``). Armed, keyed wrappers on ComfyUI's
``VAE.decode`` and ``VAE.decode_tiled`` see each decode the stock VAE Decode
and VAE Decode (Tiled) nodes make, and a keyed wrapper on
``comfy.utils.tiled_scale_multidim`` sees the tile loop inside the tiled one.
Nothing is copied from ComfyUI: its own loops run once, on the driver, with
only the per-unit decode swapped (vae_decode_plan.py).

The exactness rule: the reference is ComfyUI's one-host tiled decode for
that VAE (vae_decode_families.py), and the output must be byte-identical to
it. The driver decodes its own share with the stock function and accepts a
Worker's unit only for the exact input the stock loop asks for, built by the
same VAE file in the same dtype under the same attention and backend flags
(vae_decode_env.py). Fail-open: any error in the split path decodes the unit
here, with one WARNING, so a failure costs time and never a wrong image.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from .log import get_logger
from .vae_decode_families import DECODE, DECODE_TILED, H3_CHUNK, MULTIDIM, Family, classify, small_still
from .vae_decode_schedule import UnitBoard, pump
from .vae_decode_telemetry import DecodeRecord, publish

log = get_logger("dgx_monarch.vae_decode")

KNOB = "worker_vae_decode"
_KEY = "_dgxm_vae_decode_route"
SESSION_CLAIM_S = 5.0
_JOIN_S = 5.0
_STOCK_FUNCTIONS = frozenset({"VAE.decode_tiled_3d.<locals>.<lambda>", "VAE.decode_tiled_.<locals>.<lambda>"})
_SPLIT_ARGUMENTS = ("samples", "function", "tile", "overlap", "upscale_amount", "out_channels",
                    "output_device", "downscale", "index_formulas", "pbar")
_ACTIVE: contextvars.ContextVar[_Route | None] = contextvars.ContextVar("dgxm_vae_decode", default=None)
_LOCK = threading.Lock()
_INSTALL_LOCK = threading.Lock()
# Process lifetime, one slot: the mesh the last Init node armed, by weak
# reference, and its config path, which finds a respawned handle.
_ARMED: tuple[weakref.ref, str] | None = None


def knob(handle: Any) -> str:
    """The ``worker_vae_decode`` setting of the handle's cluster.toml; absent is off."""
    value = dict(getattr(getattr(handle, "config", None), "worker_args", None) or {}).get(KNOB, "off")
    return value if value in ("off", "auto") else "off"


def arm(handle: Any) -> None:
    """Called by the Init node for the mesh it attached; never raises into Init."""
    global _ARMED
    try:
        on = knob(handle) == "auto"
        with _LOCK:
            previous = _ARMED
            _ARMED = (weakref.ref(handle), str(getattr(handle.config, "source", "") or "")) if on else None
        if on:
            install()
            if previous is None:
                log.info("Worker-assisted VAE decode is on ([worker_args] %s = \"auto\"): stock VAE "
                         "decodes split over the Workers where the family allows (docs/MODELS.md)", KNOB)
        elif previous is not None:
            log.info("Worker-assisted VAE decode is off ([worker_args] %s); decodes stay on the driver", KNOB)
    except Exception as exc:
        log.warning("Worker-assisted VAE decode could not arm, so decodes stay on the driver: %r", exc)


def _live(handle: Any) -> bool:
    return (handle is not None and not getattr(handle, "defunct", False)
            and not getattr(handle, "teardown_complete", False)
            and getattr(handle, "setup_cleanup_state", None) is None)


def armed_handle() -> Any:
    """The armed mesh, or a live respawn of it on the same config, or None."""
    with _LOCK:
        armed = _ARMED
    if armed is None:
        return None
    handle = armed[0]()
    if _live(handle):
        return handle
    from . import mesh

    try:
        # A copy of the registry without its lock: a concurrent attach may
        # change it, which reads as no live mesh rather than failing a decode.
        candidates = list(mesh._MESHES.values())
        for candidate in candidates:
            if _live(candidate) and str(getattr(candidate.config, "source", "") or "") == armed[1]:
                return candidate
    except Exception:
        return None
    return None


def install() -> None:
    """Install the keyed wrappers once; an installed wrapper is never stacked."""
    import comfy.sd
    import comfy.utils

    with _INSTALL_LOCK:
        for entry in (DECODE, DECODE_TILED):
            current = getattr(comfy.sd.VAE, entry)
            if not getattr(current, _KEY, False):
                setattr(comfy.sd.VAE, entry, _entry_wrapper(current, entry))
        current = comfy.utils.tiled_scale_multidim
        if not getattr(current, _KEY, False):
            comfy.utils.tiled_scale_multidim = _tiled_wrapper(current)


def _entry_wrapper(original: Callable[..., Any], entry: str) -> Callable[..., Any]:
    @functools.wraps(original)
    def decode(self: Any, samples: Any, *args: Any, **kwargs: Any) -> Any:
        if _ACTIVE.get() is not None or _ARMED is None:
            return original(self, samples, *args, **kwargs)
        route = _open(self, samples, entry)
        if route is None:
            return original(self, samples, *args, **kwargs)
        return route.run(lambda: original(self, samples, *args, **kwargs))

    setattr(decode, _KEY, True)
    return decode


def _tiled_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(original)

    @functools.wraps(original)
    def tiled_scale_multidim(*args: Any, **kwargs: Any) -> Any:
        route = _ACTIVE.get()
        if route is None or route.family.kind != MULTIDIM:
            return original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return original(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        function = arguments.get("function")
        if (set(arguments) != set(_SPLIT_ARGUMENTS) or arguments.get("downscale")
                or getattr(function, "__module__", "") != "comfy.sd"
                or getattr(function, "__qualname__", "") not in _STOCK_FUNCTIONS):
            route.note("this tile loop is not the stock VAE decode loop the split replays")
            return original(*args, **kwargs)
        return route.split_multidim(original, arguments)

    setattr(tiled_scale_multidim, _KEY, True)
    return tiled_scale_multidim


def _open(vae: Any, samples: Any, entry: str) -> _Route | None:
    """A route for this decode, or None after logging why it stays here."""
    family, why = classify(vae)
    record = DecodeRecord(family=family.name if family else _class_name(vae), entry=entry)
    reason = why
    if family is not None and entry not in family.entries:
        reason = ("VAE Decode runs the whole latent through the model at once on this family, a "
                  "different result from the tiled decode; use VAE Decode (Tiled) to split it")
    elif family is not None and (pixels := small_still(vae, samples)) is not None:
        reason = (f"a {pixels / 1e6:.2f} MP still decodes in about a second here; stills at or "
                  "under 1.2 MP keep the one-host decode")
    elif family is not None and family.kind == H3_CHUNK and _frames(samples) == 1:
        reason = "a MiniMax H3 still is one temporal chunk, so there is nothing to split"
    handle = armed_handle() if not reason else None
    if not reason and handle is None:
        reason = "no live DGX Monarch mesh: the Init node attaches one and the first render sets it up"
    elif not reason and getattr(handle, "setup_key", None) is None:
        reason = "the mesh has not finished setup; the first render on it sets it up"
    ranks: list[int] = []
    source: dict[str, Any] | str = ""
    if not reason:
        from .vae_decode_remote import remote_ranks, vae_source

        ranks = remote_ranks(handle)
        source = vae_source(vae)
        if not ranks:
            reason = "no Worker rank sits on a GPU other than this driver's"
        elif isinstance(source, str):
            reason = source
    if reason or family is None or not isinstance(source, dict):
        record.reason = reason
        publish(record)
        return None
    from .vae_decode_env import describe

    return _Route(handle, family, vae, ranks, source, describe(vae), record)


def _frames(samples: Any) -> int:
    shape = tuple(getattr(samples, "shape", ()))
    return int(shape[2]) if len(shape) == 5 else 1


def _class_name(vae: Any) -> str:
    fsm = getattr(vae, "first_stage_model", None)
    return type(fsm).__name__ if fsm is not None else "unknown"


def _interrupt() -> None:
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()


class _Route:
    """One stock decode call routed over the Workers."""

    def __init__(self, handle: Any, family: Family, vae: Any, ranks: list[int],
                 source: dict[str, Any], env: dict[str, Any], record: DecodeRecord) -> None:
        self.handle = handle
        self.family = family
        self.vae = vae
        self.ranks = ranks
        self.source = source
        self.env = env
        self.record = record
        self.token: Any = None
        self._retired: set[int] = set()
        self._notes: list[str] = []

    def note(self, why: str) -> None:
        if why not in self._notes:
            self._notes.append(why)

    def run(self, call: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            with ExitStack() as stack:
                from .mesh_session import mutation_render_session
                from .mesh_setup import current_setup_token

                try:
                    stack.enter_context(mutation_render_session(self.handle, timeout_s=SESSION_CLAIM_S))
                    with self.handle.lock:
                        self.token = current_setup_token(self.handle)
                except Exception as exc:
                    self.note(f"the mesh is busy or not ready ({type(exc).__name__}: {exc})")
                    stack.close()
                    return call()
                marker = _ACTIVE.set(self)
                stack.callback(_ACTIVE.reset, marker)
                if self.family.kind == H3_CHUNK:
                    stack.enter_context(self._h3_shadow())
                return call()
        finally:
            self.record.seconds = time.perf_counter() - started
            self.record.split = any(self.record.remote.values())
            if not self.record.split and not self.record.reason:
                self.record.reason = (self._notes[0] if self._notes else
                                      "every unit was decoded here" if self.record.units else
                                      "the stock decode asked for no unit the split can take")
            publish(self.record)

    @staticmethod
    def _board(shapes: list[tuple[int, ...]], dtype: Any) -> UnitBoard:
        """A board whose budget charges each unit the bytes of its expected reply."""
        import math

        import torch

        from .vae_decode_remote import REPLY_BUDGET_BYTES

        size = torch.empty((), dtype=dtype).element_size()
        return UnitBoard(len(shapes), unit_bytes=[math.prod(shape) * size for shape in shapes],
                         budget_bytes=REPLY_BUDGET_BYTES)

    @contextmanager
    def _pumps(self, board: UnitBoard, kind: str, inputs: list[Any], output_dtype: Any,
               expected: Callable[[int], tuple[int | None, ...]], reply_dtype: Any) -> Iterator[None]:
        from . import vae_decode_remote as remote

        base = remote.request_base(kind, self.source, self.env, output_dtype)
        send = remote.sender(self.handle, self.token, base, inputs)
        wait = remote.waiter(self.handle)
        accept = remote.acceptor(reply_dtype, expected)
        threads = []
        for rank in self.ranks:
            if rank in self._retired:
                continue
            thread = threading.Thread(
                target=contextvars.copy_context().run,
                args=(pump, board, rank, send, wait),
                kwargs={"first_timeout_s": remote.FIRST_UNIT_TIMEOUT_S, "unit_timeout_s": remote.UNIT_TIMEOUT_S,
                        "accept": accept, "unwrap": lambda reply: reply["output"]},
                name=f"dgxm-vae-decode-rank{rank}", daemon=True)
            thread.start()
            threads.append(thread)
        try:
            yield
        finally:
            board.close()
            for thread in threads:
                thread.join(timeout=_JOIN_S)
            self._absorb(board)

    def _absorb(self, board: UnitBoard) -> None:
        stats = board.stats
        self.record.units += stats.units
        self.record.local += stats.local
        self.record.fallback += stats.fallback
        self.record.budget_waits += stats.budget_waits
        self.record.peak_reply_bytes = max(self.record.peak_reply_bytes, stats.peak_bytes)
        for rank, count in stats.remote.items():
            self.record.remote[rank] = self.record.remote.get(rank, 0) + count
        for failure in stats.failures:
            self._retired.add(failure.rank)
            self.record.failures.append(f"rank {failure.rank}: {failure.cause}")

    def _wait_s(self) -> float:
        from .vae_decode_remote import FIRST_UNIT_TIMEOUT_S

        return FIRST_UNIT_TIMEOUT_S + 60.0

    def split_multidim(self, original: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        from .vae_decode_plan import (
            MultidimReplay,
            PlanError,
            multidim_input,
            multidim_output_shapes,
            multidim_plan,
        )

        samples = arguments["samples"]
        try:
            plan = multidim_plan(
                original, samples, tile=arguments["tile"], overlap=arguments["overlap"],
                upscale_amount=arguments["upscale_amount"], out_channels=arguments["out_channels"],
                downscale=False, index_formulas=arguments["index_formulas"])
        except PlanError as exc:
            self.note(str(exc))
            return original(**arguments)
        if len(plan) < 2:
            self.note("the tiled decode is one tile here, so there is nothing to split")
            return original(**arguments)
        inputs = [multidim_input(samples, geometry, self.vae.vae_dtype) for geometry in plan]
        # Every dimension of a reply is checked against what the stock decode
        # returns for that tile, so a wrong-size reply never reaches the blend.
        shapes = multidim_output_shapes(plan, arguments["upscale_amount"], int(arguments["out_channels"]))
        output_dtype = self.vae.vae_output_dtype()
        board = self._board(shapes, output_dtype)
        replay = MultidimReplay(board, plan, samples, arguments["function"], arguments["output_device"],
                                max_wait_s=self._wait_s(), check_interrupt=_interrupt)
        with self._pumps(board, MULTIDIM, inputs, output_dtype, shapes.__getitem__, output_dtype):
            result = original(**{**arguments, "function": replay})
        self._broken(replay)
        return result

    @contextmanager
    def _h3_shadow(self) -> Iterator[None]:
        fsm = self.vae.first_stage_model
        fsm.decode_temporal = functools.partial(self.split_h3, fsm)
        try:
            yield
        finally:
            fsm.__dict__.pop("decode_temporal", None)

    def split_h3(self, fsm: Any, z: Any, output_buffer: Any = None) -> Any:
        from .vae_decode_plan import H3Replay, h3_canvas_shape, h3_plan

        original = functools.partial(type(fsm).decode_temporal, fsm)
        try:
            plan = h3_plan(fsm, z)
        except Exception as exc:
            self.note(f"the H3 chunk plan could not be read ({type(exc).__name__}: {exc})")
            return original(z, output_buffer)
        if len(plan) < 2:
            self.note("the H3 decode is one temporal chunk here, so there is nothing to split")
            return original(z, output_buffer)
        if output_buffer is not None and output_buffer.device.type == "cpu":
            output_buffer.zero_()  # pre-touch from the CPU before the GPU writes it
        inputs = [window.contiguous().cpu() for window in plan]
        shapes = [h3_canvas_shape(fsm, window) for window in plan]
        board = self._board(shapes, z.dtype)
        replay = H3Replay(board, plan, fsm, max_wait_s=self._wait_s(), check_interrupt=_interrupt)
        fsm._adaptive_decode = replay
        try:
            # The canvas dtype is the decoder's, which a quantized decoder may
            # not share with the latent, so the replay checks it against the
            # first stock canvas instead of here.
            with self._pumps(board, H3_CHUNK, inputs, None, shapes.__getitem__, None):
                return original(z, output_buffer)
        finally:
            fsm.__dict__.pop("_adaptive_decode", None)
            self._broken(replay)

    def _broken(self, replay: Any) -> None:
        if replay.broken is not None:
            self.record.local += replay.unplanned
            self.record.fallback += replay.unplanned
            self.record.failures.append(f"the stock loop left the plan ({replay.broken}); "
                                        "this driver decoded the rest")
