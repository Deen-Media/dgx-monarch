"""Only real rows enter the shared NVFP4 activation scale.

Without a static ``input_scale`` an NVFP4 Linear quantizes its input against
``amax(abs(x)) / (448 * 6)``, and under sequence parallelism the shared scale
reduces each rank's amax across the group (adapters/quant_activation_scale.py).
A rank's input also holds the zero rows ``shard_seq`` added so a stream
divides the shard. After a block's norm and modulation those rows are not
zero, and whenever one of them held the largest value of a call, the whole
group would quantize against a scale one GPU never computes.

A sharded forward therefore names, around its block loop, which of its
Linears see which sharded stream, and how many of that stream's local rows are
real on this rank. The scale wrapper then reduces the amax over the real rows
alone. Roles go by module path, never by shape: a replicated text context, a
per-batch modulation vector or a full-row text projection can have the same
row count as a sharded stream by chance. A Linear named for a stream whose
input does not have that stream's rows is a dgx-monarch fault and refuses,
because reducing over it as it is would put the pads back in silently.

The real rows of a stream are the leading rows of each of its segments: a
joint stream ``[text_local, image_local]`` has two, each padded at its own
tail, and Anima's H-row shard has one per frame. Each span is a view, so the
reduction copies nothing beyond the ``abs`` the whole-tensor amax took
already. A rank whose rows are all padding contributes zero, so it still
issues every reduction its peers issue.

Mage-Flow keeps its own scope (adapters/mage_nvfp4_scale.py); a family whose
forward sets neither is listed in tests/test_scale_rows_coverage.py with the
reason it needs none.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch

from ..refusal import RefusalClass, refusal
from . import base
from .base import UnsupportedModelError


@dataclass(frozen=True)
class Stream:
    """One sharded stream as this rank holds it.

    ``segments`` lists ``(local rows, real rows)`` along ``axis`` in the order
    the stream concatenates them; the real rows lead each segment.
    """

    segments: tuple[tuple[int, int], ...]
    axis: int = -2

    @property
    def rows(self) -> int:
        return sum(local for local, _real in self.segments)


@dataclass(frozen=True)
class _Scope:
    role: Callable[[str], str | None]
    streams: Mapping[str, Stream]


_SCOPE: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar(
    "dgxm_scale_rows", default=None)
# Process state: which (role table, stream names, model paths) triples already
# passed ``check_streams_read``, so the check walks a model's paths once per loop
# kind. Keyed on the objects, not their ids, so a model loaded after another was
# freed cannot inherit its answer; cleared past a bound, as a long session loads
# many models.
_ANSWERED: set[tuple[Callable[[str], str | None], frozenset[str], frozenset[str]]] = set()
_ANSWERED_BOUND = 64


def check_streams_read(model_paths: frozenset[str]) -> None:
    """Refuse, class P, a named stream that no layer of the running model reads.

    A ComfyUI that renames a module would leave the role table matching none
    of the layers that read a stream, and those layers would then take their
    whole input, pads included, into the scale. Each wrapped Linear carries its
    own model's module paths and calls this before it reduces, so the check
    reads the model that runs, and a mixed-precision file whose readers of a
    stream are not quantized still matches by path. Every rank holds the same
    model and names the same streams, so every rank refuses at the same layer.
    """
    scope = _SCOPE.get()
    if scope is None or not model_paths:
        return
    key = (scope.role, frozenset(scope.streams), model_paths)
    if key in _ANSWERED:
        return
    if len(_ANSWERED) >= _ANSWERED_BOUND:
        _ANSWERED.clear()
    for name in scope.streams:
        if not any(scope.role(path) == name for path in model_paths):
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"no layer of this model matches the role table for sharded stream {name!r}, "
                "so the layers that read it would take its pad rows into the shared nvfp4 "
                "activation scale. A ComfyUI that renamed one of the model's modules is the "
                "likeliest cause; this is a dgx-monarch fault, not a workflow setting. Use "
                "topology 'single' (mode=local with gpus_per_host=1), or a pure Ulysses "
                "render of an fp8 or bf16 file, until it is fixed.",
            ))
    _ANSWERED.add(key)


def real_count(original: int, local: int, rank: int | None = None,
               world: int | None = None) -> int:
    """How many of a stream's ``local`` rows on ``rank`` are real.

    ``shard_seq`` pads ``original`` rows up to the degree and gives rank r the
    chunk starting at ``r * local``; only rows below ``original`` are real. A
    stream that divides the degree has no pad, on any rank. ``rank`` and
    ``world`` default to the sequence-parallel group's; a forward that reads its
    own accessors passes them.
    """
    if rank is None:
        if original >= local * (base.sp_world() if world is None else world):
            return local
        rank = base.sp_rank()
    return max(0, min(local, original - rank * local))


def stream(*segments: tuple[int, int], rank: int | None = None, world: int | None = None,
           axis: int = -2) -> Stream:
    """A stream from ``(original, local)`` pairs, the way ``shard_seq`` reports them."""
    return Stream(tuple((local, real_count(original, local, rank, world))
                        for original, local in segments), axis)


@contextlib.contextmanager
def real_rows(role: Callable[[str], str | None],
              streams: Mapping[str, Stream]) -> Iterator[None]:
    """Count only real rows in the shared scale of every Linear ``role`` names.

    Every named stream must be read by some layer of the running model; each
    wrapped Linear checks that before it reduces (``check_streams_read``).
    """
    token = _SCOPE.set(_Scope(role, dict(streams)))
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextlib.contextmanager
def whole_rows() -> Iterator[None]:
    """Count every row while a Linear runs on a stream's gathered, unpadded rows.

    A full-row text projection (adapters/chroma_text.full_row_text_projections)
    gathers its stream and trims the pad before it calls the Linear, so the
    input holds real rows only. Mage-Flow's own row scope steps aside too.

    No production file reaches this today. The full-row projections select a
    ``torch.nn.Linear``, and Boogu's selector also an fp8 mixed-precision
    Linear whose input scale is a constant, and Mage-Flow's an unquantized
    one; an NVFP4 layer is none of these, so no Linear the shared scale wraps
    runs inside this scope. It guards a file that would bring the two
    together, which none on record does.
    """
    from . import mage_nvfp4_scale

    token = _SCOPE.set(None)
    with mage_nvfp4_scale.whole_rows():
        try:
            yield
        finally:
            _SCOPE.reset(token)


def _refuse(path: str, detail: str) -> None:
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"the nvfp4 layer {path} met an input the shared activation scale cannot read "
        f"its real rows from ({detail}). This is a dgx-monarch fault, not a workflow "
        "setting: report the shapes. Use topology 'single' (mode=local with "
        "gpus_per_host=1) until it is fixed.",
    ))


def spans(path: str, value: torch.Tensor) -> Sequence[torch.Tensor]:
    """The real-row views of ``value`` that the shared scale reduces over.

    With no scope set, or for a Linear the scope gives no stream, the whole
    input: nothing it holds is a pad row.
    """
    scope = _SCOPE.get()
    if scope is None:
        return (value,)
    name = scope.role(path)
    if name is None:
        return (value,)
    found = scope.streams.get(name)
    if found is None:
        _refuse(path, f"no stream named {name!r} in this block loop")
        raise AssertionError("unreachable")
    if value.ndim < 2:
        _refuse(path, f"a {value.ndim}-dimensional input for stream {name!r}")
    axis = found.axis % value.ndim
    if value.shape[axis] != found.rows:
        _refuse(path, f"{value.shape[axis]} rows on axis {axis} where stream {name!r} "
                      f"holds {found.rows}")
    out = []
    offset = 0
    for local, real in found.segments:
        if real:
            out.append(value.narrow(axis, offset, real))
        offset += local
    return tuple(out)


def by_pattern(rules: Sequence[tuple[str, str | None]]) -> Callable[[str], str | None]:
    """A role function from ``(regular expression, stream)`` rules over module
    paths, the first full match winning.

    A rule's stream None leaves those Linears unscoped (a replicated input or a
    per-batch vector); a path no rule matches is unscoped too.
    """
    compiled = [(re.compile(pattern), name) for pattern, name in rules]

    def role(path: str) -> str | None:
        for pattern, name in compiled:
            if pattern.fullmatch(path):
                return name
        return None

    return role
