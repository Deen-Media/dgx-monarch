"""Unit plans and verified replays for Worker-assisted VAE decode.

The reference is ComfyUI's own single-host tiled decode, and nothing here
reimplements it. Two kinds of decode loop are split:

* ``comfy.utils.tiled_scale_multidim``, which ``VAE.decode_tiled_3d`` and
  ``VAE.decode_tiled_`` call with a per-tile decode function. The plan is read
  by running that stock function once on meta tensors with a recorder in place
  of the decode function: no memory, no math, and the tile order, positions and
  sizes are the stock code's own. The replay then runs the stock function on
  the real latent with one change: the decode function returns a Worker's
  result for a tile whose input view matches the plan exactly, and calls the
  stock function here for every other tile. Blending, feathering, division,
  output dtype and device all stay the stock code's.
* MiniMax H3's temporal chunks (``MiniMaxH3VideoVAE.decode_temporal``), which
  the model tiles itself (``handles_tiling``). The plan names each chunk's
  latent window; the replay stands in for ``_adaptive_decode`` during the stock
  ``decode_temporal`` and accepts a Worker's canvas only when the chunk the
  stock loop hands it equals the planned window exactly (``torch.equal``).

A unit whose input differs from the plan is decoded here, and so is every unit
after it. That costs speed and never changes an output.
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .vae_decode_schedule import UnitBoard


class PlanError(RuntimeError):
    """The stock loop's plan could not be read; the decode runs unsplit."""


@dataclass(frozen=True)
class Geometry:
    """One tile's input view, relative to the latent it was cut from."""

    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


def view_geometry(view: Any, base: Any) -> Geometry | None:
    """The view's placement in ``base``'s storage, or None when it is not a view of it."""
    if view.device != base.device:
        return None
    if view.device.type != "meta" and (
            view.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()):
        return None
    return Geometry(int(view.storage_offset() - base.storage_offset()),
                    tuple(int(n) for n in view.shape), tuple(int(n) for n in view.stride()))


def pretouched_cpu(tensor: Any) -> Any:
    """Copy ``tensor`` into CPU pages the CPU has already written.

    On GB10 a device-to-host copy into freshly allocated pageable memory runs
    at about 0.1 GB/s until the CPU has touched the pages (issue #93, item 3),
    so the destination is zero-filled from the CPU first. The copy is exact.
    """
    import torch

    out = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu")
    out.zero_()
    out.copy_(tensor)
    return out


def _scaled(size: int, factor: Any, downscale: bool) -> int:
    if callable(factor):
        return round(factor(size))
    return round(size / factor if downscale else factor * size)


def _factors(upscale_amount: Any, dims: int) -> list[Any]:
    return list(upscale_amount) if isinstance(upscale_amount, (tuple, list)) else [upscale_amount] * dims


def multidim_output_shapes(plan: list[Geometry], upscale_amount: Any, out_channels: int) -> list[tuple[int, ...]]:
    """The shape the stock decode function returns for each planned tile.

    The scale rule is the one the stock loop sizes its own output buffer with
    (``upscale_amount`` per axis, a callable or a factor), applied to each
    tile's own sizes, so a Worker reply of any other shape is refused before
    the stock blend sees it.
    """
    shapes = []
    for geometry in plan:
        spatial = geometry.size[2:]
        factors = _factors(upscale_amount, len(spatial))
        shapes.append((geometry.size[0], int(out_channels),
                       *(_scaled(size, factor, False) for size, factor in zip(spatial, factors, strict=True))))
    return shapes


def multidim_plan(original: Callable[..., Any], samples: Any, *, tile: Sequence[int],
                  overlap: Any, upscale_amount: Any, out_channels: int, downscale: bool,
                  index_formulas: Any) -> list[Geometry]:
    """Every tile the stock loop will ask for, in its order, from a meta dry run."""
    import torch

    dims = len(tile)
    factors = _factors(upscale_amount, dims)
    meta = torch.empty_strided(tuple(samples.shape), tuple(samples.stride()),
                               dtype=samples.dtype, device="meta")
    units: list[Geometry] = []

    def record(s_in: Any) -> Any:
        geometry = view_geometry(s_in, meta)
        if geometry is None:
            raise PlanError("the stock loop handed the decode a tensor that is not a view of the latent")
        units.append(geometry)
        sizes = [_scaled(int(s_in.shape[d + 2]), factors[d], downscale) for d in range(dims)]
        return torch.empty([int(s_in.shape[0]), int(out_channels), *sizes], device="meta")

    try:
        original(meta, record, tile=tuple(tile), overlap=overlap, upscale_amount=upscale_amount,
                 out_channels=out_channels, output_device="meta", downscale=downscale,
                 index_formulas=index_formulas, pbar=None)
    except PlanError:
        raise
    except Exception as exc:
        raise PlanError(f"the tile plan could not be read: {type(exc).__name__}: {exc}") from exc
    return units


def multidim_input(samples: Any, geometry: Geometry, dtype: Any) -> Any:
    """The exact tile the stock loop will cut, cast as its decode function casts it first."""
    view = samples.as_strided(geometry.size, geometry.stride,
                              samples.storage_offset() + geometry.offset)
    # The stock decode function starts with ``a.to(vae_dtype)`` on the
    # latent's own device; doing that here first keeps the cast identical and
    # halves the bytes on the wire.
    return view.to(dtype).contiguous().cpu()


class _Replay:
    """Shared order and fallback bookkeeping for one replayed loop."""

    def __init__(self, board: UnitBoard, count: int, *, max_wait_s: float,
                 check_interrupt: Callable[[], None] | None) -> None:
        self._board = board
        self._count = count
        self._next = 0
        self._max_wait_s = max_wait_s
        self._check_interrupt = check_interrupt
        self.broken: str | None = None
        self.unplanned = 0  # units decoded here after the plan broke, outside the board
        # Unit 0 is always decoded here (UnitBoard's lookahead), so every
        # remote result meets the dtype and channel count of a stock one.
        self._reference: tuple[Any, int] | None = None

    def _turn(self, matches: Callable[[int], bool]) -> tuple[int, bool, Any]:
        unit = self._next
        self._next += 1
        if self.broken is None:
            if unit >= self._count:
                self._break(f"the stock loop asked for unit {unit + 1} of a {self._count}-unit plan")
            elif not matches(unit):
                self._break(f"unit {unit} differs from the plan")
        if self.broken is not None:
            self.unplanned += 1
            return unit, False, None
        remote, value = self._board.take(unit, max_wait_s=self._max_wait_s,
                                         check_interrupt=self._check_interrupt)
        return unit, remote, value

    def _break(self, why: str) -> None:
        self.broken = why
        self._board.close()

    def _local_result(self, out: Any) -> Any:
        if self._reference is None and hasattr(out, "dtype") and getattr(out, "ndim", 0) >= 2:
            self._reference = (out.dtype, int(out.shape[1]))
        return out

    def _usable(self, unit: int, value: Any) -> bool:
        """Whether a remote result matches the stock results in dtype and channels."""
        if self._reference is None:
            return True
        dtype, channels = self._reference
        if value.dtype == dtype and int(value.shape[1]) == channels:
            return True
        self._board.reject(unit, f"unit {unit} came back as {value.dtype} with {int(value.shape[1])} "
                                 f"channels where the stock decode gives {dtype} with {channels}")
        return False

    @property
    def consumed(self) -> int:
        return self._next


class MultidimReplay(_Replay):
    """Stands in for the stock per-tile decode function during the real loop."""

    def __init__(self, board: UnitBoard, plan: list[Geometry], samples: Any,
                 function: Callable[[Any], Any], output_device: Any, *, max_wait_s: float,
                 check_interrupt: Callable[[], None] | None = None) -> None:
        super().__init__(board, len(plan), max_wait_s=max_wait_s, check_interrupt=check_interrupt)
        self._plan = plan
        self._samples = samples
        self._function = function
        self._cpu_output = getattr(output_device, "type", str(output_device)) == "cpu"

    def __call__(self, s_in: Any) -> Any:
        unit, remote, value = self._turn(
            lambda unit: view_geometry(s_in, self._samples) == self._plan[unit])
        if remote and self._usable(unit, value):
            # Already on the CPU; the stock ``.to(output_device)`` moves it if needed.
            return value
        out = self._local_result(self._function(s_in))
        if self._cpu_output and getattr(out, "device", None) is not None and out.device.type != "cpu":
            # The stock loop's ``.to(output_device)`` would land in fresh pages.
            return pretouched_cpu(out)
        return out


def h3_plan(fsm: Any, z: Any) -> list[Any]:
    """Each temporal chunk ``MiniMaxH3VideoVAE.decode_temporal`` will hand ``_adaptive_decode``.

    Read off the model's own chunk helper and attributes. The windows are
    checked against the stock loop's chunks one by one in the replay, so a
    change upstream makes this decode run here, never a wrong frame.
    """
    import torch

    pad_tokens, chunks = fsm._decode_temporal_chunks(int(z.shape[2]))
    padded = z
    if pad_tokens > 0:
        padded = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    step = int(fsm.tokens_chunk_size)
    span = step + int(fsm.token_overlap)
    return [padded[:, :, index * step:index * step + span, :, :] for index in range(int(chunks))]


def h3_canvas_shape(fsm: Any, window: Any) -> tuple[int, ...]:
    """The canvas ``_adaptive_decode`` returns for one window, from the model's ratios."""
    b, _c, t, h, w = (int(n) for n in window.shape)
    channels = int(getattr(getattr(fsm, "decoder", None), "out_channels", 3))
    return (b, channels, t * int(fsm.vae_ratio_t), h * int(fsm.vae_ratio), w * int(fsm.vae_ratio))


class H3Replay(_Replay):
    """Stands in for ``MiniMaxH3VideoVAE._adaptive_decode`` during the stock ``decode_temporal``."""

    def __init__(self, board: UnitBoard, plan: list[Any], fsm: Any, *, max_wait_s: float,
                 check_interrupt: Callable[[], None] | None = None) -> None:
        super().__init__(board, len(plan), max_wait_s=max_wait_s, check_interrupt=check_interrupt)
        self._plan = plan
        self._original = functools.partial(type(fsm)._adaptive_decode, fsm)

    def __call__(self, z: Any, frames: slice = slice(None)) -> Any:
        import torch

        def matches(unit: int) -> bool:
            planned = self._plan[unit]
            return (frames == slice(None) and z.shape == planned.shape
                    and z.dtype == planned.dtype and bool(torch.equal(z, planned)))

        unit, remote, value = self._turn(matches)
        if remote and self._usable(unit, value):
            # The stock canvas lives on the decode device; so must this one.
            return value.to(device=z.device)
        return self._local_result(self._original(z, frames))
