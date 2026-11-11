"""The memory a Worker returns when it drops the model a load replaces.

A load into a slot that holds another model is a swap: the store drops the
slot before it loads (``actor/store_load.fresh_load``), and the drop of a stock
resident returns its GPU copy in-process. On 2026-11-05 the head rose from
58.7 to 88.5 GiB as the store dropped a 26.6 GiB cast DiT, and on 2026-11-11 a
swap from a 37.5 GiB BF16 MiniMax-H3-Image DiT to its 11.7 GiB W4A8 file never
took the head below the 32.9 GiB it read before the swap.

The cross-rank agreement credits that copy in the Worker's own price
(``capacity_lora_bake.credit_stock_swap``). The driver host's own prices did
not: the loader preflight and the render-site checks for a pending load priced
the new DiT beside the old one, and a user who changed model in one session
met a refusal, or the Old model in the way card, with room on the box. Row 5 of
the 2026-11 Studio study (flux1-dev held, then LTX 2.3: 80.8 GiB needed, 67.5
available, 22.2 GiB of flux1-dev about to be dropped) and the MiniMax-H3-Image
INT8 load of 2026-11-11 (46.0 GiB needed, 34.1 available, the 37.5 GiB BF16
DiT in the same slot) are that fault.

This module takes the credit from the answers the Workers on the driver host
give to the agreement's own quote, and it takes only what a Worker's own price
credited, so the driver never admits on a drop the agreement would not count:

- the Worker's row for this slot says the load is a swap, and its stock price
  names the model it drops and the bytes it credited for it (``dropped`` and
  ``credit_bytes`` in the row's measured block, written by
  ``credit_stock_swap``). That rule covers a stock resident only, at the
  file's size or the lower bound of a cast's copy. A slab or comfy-managed
  resident, an FSDP shard, a slot whose load or cleanup is in flight, a reuse,
  and a LoRA change that keeps its base carry no such pair and give nothing;
- a row priced on a slab rung gives nothing: the agreement prices that load
  with the old model held;
- only the slot the load goes into. A model in another slot stays, and a load
  whose slot is not known gets no credit;
- one Worker's worth. The driver's prices charge one Worker's weights however
  many ranks share the box, so the credit is the smallest any of them reports,
  and nothing when one of them reports none.

And the credit goes only onto a price that charges the DiT's own load. A price
that reads the weights as already resident charges nothing for them, so it has
no drop to count: the callers apply the credit only beside a weight charge.

Pure: no ComfyUI, no mesh call and no file read. ``nodes/loader_relief.py``
asks the Workers and applies the credit at the loader; ``applying`` carries it
to the render-site prices that charge a pending load.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, overload

_GIB = 2 ** 30
_SWAP = "swap"


@dataclass(frozen=True, slots=True)
class SwapCredit:
    """What the driver host gets back before a load, and the model that returns it."""

    bytes: int = 0
    # The checkpoint the Worker drops, as the store names it, and the box it is on.
    dropped: str = ""
    box: str = ""

    def __bool__(self) -> bool:
        return self.bytes > 0

    @property
    def gib(self) -> str:
        return f"{self.bytes / _GIB:.1f}"


NONE = SwapCredit()


def _row_credit(quote: Any, slot: str) -> tuple[int, str]:
    """What one Worker's own price credited for a load into ``slot``, and the model it drops."""
    if (getattr(quote, "slot", None) != slot or getattr(quote, "would_load", None) != _SWAP
            or getattr(quote, "use_slab", False)):
        return 0, ""
    block = getattr(quote, "measured", None) or {}
    credit, dropped = block.get("credit_bytes"), block.get("dropped")
    if not dropped or not isinstance(credit, int) or isinstance(credit, bool) or credit <= 0:
        return 0, ""
    return credit, str(dropped)


def from_answers(answers: Sequence[Any], slot: str, *, box: str = "") -> SwapCredit:
    """The credit for a load into ``slot``, or ``NONE``.

    ``answers`` are the agreement's answers from the ranks that share the
    driver host, each with its price rows for this request.
    """
    if not slot or not answers:
        return NONE
    credits: list[tuple[int, str]] = []
    for answer in answers:
        rows = [row for row in getattr(answer, "quotes", ()) or () if getattr(row, "slot", None) == slot]
        credit = _row_credit(rows[0], slot) if rows else (0, "")
        if credit[0] <= 0:
            return NONE
        credits.append(credit)
    size, name = min(credits)
    return SwapCredit(size, name, box)


# Render lifetime: the credit the render-site rescue found, read through
# ``credited`` by the prices that charge a pending load on the driver host, and
# by each only on the branch where it charges that load (nodes/render_preflight,
# h3_activation, mesh_safety's reference and pose price). None outside ``applying``.
@dataclass(slots=True)
class Applied:
    """A credit in force, and whether any price that charges a load took it."""

    bytes: int
    used: bool = False


_PENDING: ContextVar[Applied | None] = ContextVar("dgxm_swap_credit", default=None)


@contextmanager
def applying(credit: SwapCredit) -> Iterator[Applied]:
    """Count ``credit`` as available in the pending-load prices inside the block."""
    applied = Applied(max(int(credit.bytes), 0))
    token = _PENDING.set(applied)
    try:
        yield applied
    finally:
        _PENDING.reset(token)


@overload
def credited(mem_available: int) -> int: ...
@overload
def credited(mem_available: None) -> None: ...
def credited(mem_available: int | None) -> int | None:
    """A reading for a pending load: plus what the drop returns first, and the
    credit in force is marked as taken, so its rescue says only what it counted."""
    applied = _PENDING.get()
    if mem_available is None or applied is None or not applied.bytes:
        return mem_available
    applied.used = True
    return mem_available + applied.bytes
