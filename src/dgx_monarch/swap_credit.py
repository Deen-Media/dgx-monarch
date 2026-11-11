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

This module reads the credit from the answers the Workers on the driver host
give to the agreement's own quote, by ``credit_stock_swap``'s rule:

- the Worker's row for this slot says the load is a swap;
- the slot holds a stock resident, whose drop was measured to return its copy.
  A slab or comfy-managed resident, an FSDP shard and a slot whose load or
  cleanup is in flight give nothing;
- the size is the one the Worker reports for that copy: the figure its own
  price credited where its row carries one, else its snapshot's figure, which
  for a cast is a lower bound;
- only the slot the load goes into. A model in another slot stays, and a load
  whose slot is not known gets no credit;
- another file. A swap onto the file the slot already holds is a LoRA or cast
  change, where the store may keep the base, so only a drop the Worker's own
  price named counts there;
- one Worker's worth. The driver's prices charge one Worker's weights however
  many ranks share the box, so the credit is the smallest any of them reports,
  and nothing when one of them reports none.

A reuse gets no credit and needs none here: the loader's quote names no LoRA
stack, so it cannot tell a reuse from a stack change.

Pure: no ComfyUI, no mesh call and no file read. ``nodes/loader_relief.py``
asks the Workers and applies the credit at the loader; ``applying`` carries it
to the render-site prices that charge a pending load.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_GIB = 2 ** 30
CREDITED_RESIDENCY = "stock"
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


def _row_credit(quote: Any, slot: str, unet_name: str) -> tuple[int, str]:
    """One Worker's credit for a load of ``unet_name`` into ``slot``, and the model it drops."""
    if getattr(quote, "slot", None) != slot or getattr(quote, "would_load", None) != _SWAP:
        return 0, ""
    if (getattr(quote, "holds_resident", None) or {}).get(slot) != CREDITED_RESIDENCY:
        return 0, ""
    name = str((getattr(quote, "resident_names", None) or {}).get(slot) or "")
    held = (getattr(quote, "resident_bytes", None) or {}).get(slot)
    block = getattr(quote, "measured", None) or {}
    # The Worker's own swap credit, where its stock price applied one and named the drop.
    own = block.get("credit_bytes") if block.get("dropped") and not getattr(quote, "use_slab", False) else None
    # A swap onto the file the slot already holds is a stack or cast change. The
    # store may keep the base and re-apply the stack, so only a drop the Worker's
    # own price named is credited there, never the snapshot's figure.
    for figure in (own,) if not name or name == unet_name else (own, held):
        if isinstance(figure, int) and not isinstance(figure, bool) and figure > 0:
            return figure, name
    return 0, ""


def from_answers(answers: Sequence[Any], slot: str, unet_name: str, *, box: str = "") -> SwapCredit:
    """The credit for a load of ``unet_name`` into ``slot``, or ``NONE``.

    ``answers`` are the agreement's answers from the ranks that share the
    driver host, each with its price rows for this request.
    """
    if not slot or not answers:
        return NONE
    credits: list[tuple[int, str]] = []
    for answer in answers:
        rows = [row for row in getattr(answer, "quotes", ()) or () if getattr(row, "slot", None) == slot]
        credit = _row_credit(rows[0], slot, unet_name) if rows else (0, "")
        if credit[0] <= 0:
            return NONE
        credits.append(credit)
    size, name = min(credits)
    return SwapCredit(size, name, box)


# Render lifetime: the credit the render-site rescue found, read by the prices
# that charge a pending load on the driver host, through
# ``mesh_safety.pending_load_available_bytes``. Zero outside ``applying``.
_PENDING: ContextVar[int] = ContextVar("dgxm_swap_credit", default=0)


@contextmanager
def applying(credit: SwapCredit) -> Iterator[SwapCredit]:
    """Count ``credit`` as available in the pending-load prices inside the block."""
    token = _PENDING.set(max(int(credit.bytes), 0))
    try:
        yield credit
    finally:
        _PENDING.reset(token)


def credited(mem_available: int | None) -> int | None:
    """MemAvailable for a pending load: the reading plus what the drop returns first."""
    return None if mem_available is None else mem_available + _PENDING.get()
