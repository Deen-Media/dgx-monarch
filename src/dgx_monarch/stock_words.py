"""The words a stock price prints for what it charges, and for a credit.

``capacity_fit`` re-exports these, so every caller keeps reading them there.
A credited LoRA change names what the Worker already holds: the file's own copy
(``capacity_lora_bake.credit_hot_swap``) and, when the agreement credited it,
ComfyUI's host copy of the weights the held stack patches (its backup). The
figures come from ``capacity_fit.gib``, the one rounding every block and card
prints, read at call time because that module imports this one.
"""
from __future__ import annotations


def _gib(value: int) -> float:
    from .capacity_fit import gib

    return gib(value)


def _beside(weights: float, backup: float, stack: float = 0.0) -> str:
    """What a credited LoRA change is priced beside, and what it still charges."""
    if not backup:
        return (f"the {weights} GiB copy of this file the Worker already holds, room for a host "
                "copy of the weights its LoRA stack changes and the floor")
    files = f"the {stack} GiB of LoRA files the new stack reads, " if stack else ""
    return (f"the {weights} GiB copy of this file and the {backup} GiB host copy of its patched "
            f"weights (ComfyUI's backup) the Worker already holds, room for {files}the weights "
            "that backup does not hold and the floor")


def stock_terms(size_gib: float, credit_bytes: int = 0, backup_bytes: int = 0,
                stack_bytes: int = 0) -> str:
    """What a stock price charges, as the rescue card prints it after "under stock residency".

    A credit is the copy the slot holds, which a cast makes other than the file,
    plus the held backup (``backup_bytes``) net of the new stack's own files
    (``stack_bytes``), which stay charged.
    """
    if credit_bytes:
        backup, files = max(int(backup_bytes), 0), max(int(stack_bytes), 0)
        held = _gib(int(credit_bytes) - backup + files)
        return f"beside {_beside(held, _gib(backup), _gib(files))},"
    return f"(a {size_gib} GiB file plus the host copy it is placed from)"


def stock_second_term(measured: dict) -> str:
    """The fleet card's words after "a N GiB file": the host copy, and a cast's own copy."""
    cast = measured.get("cast_gib")
    bake = int(measured.get("bake_bytes") or 0) if measured.get("bake") == "classic" else 0
    return ((f"cast to a {cast} GiB copy, " if cast else "") + "plus the host copy it is placed from"
            + (f" and {_gib(bake)} GiB for the LoRA stack's classic bake" if bake else ""))


def bake_sentences(measured: dict, *, name_the_term: bool = False) -> str:
    """What a refusal adds about a LoRA stack's bake: the classic term, and the in-place bake not taken.

    ``name_the_term`` is for a sentence that has not already named the classic
    bake's share (the credited one). The note is ``capacity_classic_bake``'s,
    which starts "; baking it in place would need".
    """
    bake = int(measured.get("bake_bytes") or 0) if measured.get("bake") == "classic" else 0
    words = (f" That includes {_gib(bake)} GiB for the LoRA stack's classic bake."
             if bake and name_the_term else "")
    note = str(measured.get("in_place_note") or "")
    return words + (f" B{note[3:]}." if note.startswith("; b") else "")


def hot_swap_sentence(where: str, unet_name: str, needed: float, weights: float,
                      avail: float, short: float, backup: float = 0.0, stack: float = 0.0) -> str:
    """The fleet card's sentence for a credited price that still falls short.

    The figure leaves out the held file, and the held backup when it was
    credited, so the sentence says it is room beside them rather than adding
    them into the sum.
    """
    return (f"{where} priced a LoRA change on {unet_name}: the change needs {needed} GiB "
            f"beside {_beside(weights, backup, stack)}. Only {avail} GiB of unified memory is "
            f"available, {short} GiB short.")


__all__ = ["bake_sentences", "hot_swap_sentence", "stock_second_term", "stock_terms"]
