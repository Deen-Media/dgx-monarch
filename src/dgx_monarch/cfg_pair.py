"""Whether a render's prompt pair runs as two calls on one GPU, read on the driver.

A cfg record measured only on prompt pairs of different text rows
(topology_cfg_evidence, ``pairs="unequal"``) grants its row only when this
answer is true. ComfyUI batches cond and uncond into one call when its concat
rule allows (comfy/samplers.py ``can_concat_cond``). cfg2 runs that pair as two
batch-1 calls, which one GPU does not, and on the families measured the two
did not match (docs/VALIDATION.md, Campaign K). A pair that cannot fold runs as
two batch-1 calls on one GPU too, and cfg2's per-cond dispatch
(adapters/cfg_dispatch.py) reproduces them.

Anima and Boogu, the families whose records carry the scope, wrap the text as
``CONDRegular``, which concatenates only at equal shape: Anima's Qwen3 rows on
a Worker, which samples outside inference mode and so keeps comfy's 512-row pad
inside the forward, and Boogu's Qwen3-VL rows beside its ``num_tokens``
constant. Different row counts therefore never fold, and that is the one fact
read here. Equal rows may fold or may not, and anything this cannot read
answers false too: that costs speed, never fidelity. A family whose text is a
``CONDCrossAttn`` folds some unequal lengths as well
(``cfg_dispatch.cross_attn_pair_folds``), so its record needs that rule before
it can take the scope.
"""
from __future__ import annotations

from .adapters.cfg_dispatch import _UNMODELLED_COND_KEYS

# Present on either side, these change how comfy groups the pair, and the
# per-cond dispatch keeps such a pair on the slice path; read by value, as
# cfg_dispatch reads them, because the key can hold None.
_GROUPING_VALUES = ("hooks", "control")


def _text_rows(conditioning: object) -> int | None:
    """The text rows of a one-entry conditioning list, or None when unread."""
    if not isinstance(conditioning, list | tuple) or len(conditioning) != 1:
        return None
    entry = conditioning[0]
    if not isinstance(entry, list | tuple) or len(entry) != 2:
        return None
    tensor, options = entry
    if not isinstance(options, dict):
        return None
    if any(key in options for key in _UNMODELLED_COND_KEYS) \
            or any(options.get(key) is not None for key in _GROUPING_VALUES):
        return None
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) != 3:  # [batch, rows, width]
        return None
    return int(shape[1])


def text_rows_differ(positive: object, negative: object) -> bool:
    """True only when each side is one plain entry and their text rows differ."""
    pos, neg = _text_rows(positive), _text_rows(negative)
    return pos is not None and neg is not None and pos != neg
