"""Price what a LoRA stack's bake adds beside the weights on unified memory.

ComfyUI's classic bake (``lora_low_rss`` off) moves each patched weight from
the host onto the device among that key's temporaries, and the process's GPU
figure ends above the weights by an amount that follows how many weights the
stack patches, not the stack's size. Measured on one GPU of a DGX Spark under
its clock cap (max 2100 MHz), as the rise over the same model's load with no
stack (docs/DESIGN.md section 5.9 holds the table and the dates):

- LTX 2.3 scaled FP8, 1660 patched weights (18.07 GiB of them): 12.34 GiB;
- the same model, 480 patched weights (12.0 GiB): 5.03 GiB;
- MiniMax H3 INT8 ConvRot, 208 patched weights: 1.21 GiB;
- Qwen Image plain FP8, 720 patched weights: 0.84 GiB.

``classic_bake_bytes`` charges 2 GiB plus 7.5 MiB a patched weight, which
covers every point (5.52 and 14.16 GiB for the two LTX stacks). A patched bias
counts as a weight, as it does in the count the Worker logs. The stock price
(``capacity_fit``) had no term for any of it, and a 7.1 GiB stack took a load
8.7 GiB past its price.

What is measured and what is not:

- Quantized FP8, INT8 ConvRot and plain FP8 weights have the points above. A
  quantized FP8 weight larger than the largest one under those points (64
  MiB) is past them, and a stack that patches one is charged the larger of
  the line and 0.75 of its patched quantized FP8 bytes (the larger share the
  two LTX points gave, 0.68, rounded up).
- Full-precision weights (BF16, FP16, FP32) have no measured point. They are
  charged by the line, as the lead ruled on 2026-11-12, until the hardware
  block's two BF16 cells are in (a Klein BF16 stack cut at the latent, and the
  BF16 Wan-Animate 2 template's first stack change). A share of their bytes
  with no cell behind it priced a recovery that worked on hardware (clear and
  retry on that template, 2026-11-05) as a refusal. ``GUARDED_BY_BYTES`` is
  the one place to change when those cells are read.
- A stack whose LoRA file cannot be read or mapped is charged for every weight
  of its checkpoint. A checkpoint whose own header cannot be read gives no
  weights to count, so its stack is charged the line for
  ``UNREADABLE_STACK_WEIGHTS``, the largest count measured. That is a stand-in
  and not a bound: a larger model may patch more.

The stack's own files are charged on every classic price: the Worker reads
them into host memory for the bake, and a clone that read them earlier has
gone by then.

The in-place bake (what ``lora_low_rss`` does on a loaded model, and what the
store falls back to when the classic price does not fit and the identity gate
has passed the combination) leaves almost none of the classic bake's memory:
it rose 8.28 GiB for a 7.08 GiB stack staged on the device and 2.25 for a
0.305 GiB one, the stack's own bytes plus about 2 GiB of working memory. The
fallback runs with ``lora_low_rss`` off, where the stack stays on the host, so
``in_place_bake_bytes`` charges one copy of the stack and that working memory.

No torch and no ComfyUI import: the driver's quote and the Worker's walls
call the same functions with the same arguments, so one load has one price.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from .capacity_floor import ABSOLUTE_HOST_FLOOR_BYTES
from .capacity_lora_bake import _base_index, _find_base_key, _lora_targets, stack_file_bytes
from .safetensors_header import TensorDescriptor, read_safetensors_header

GIB = 1 << 30
MIB = 1 << 20

# A cover of the four measured rises: 5.52 GiB for the 5.03 measured, 14.16 for the 12.34.
CLASSIC_BAKE_FIXED_BYTES = 2 * GIB
CLASSIC_BAKE_BYTES_PER_WEIGHT = int(7.5 * MIB)
# The larger share of the patched bytes measured (0.68), rounded up.
CLASSIC_BAKE_GUARD_RATIO = 0.75
FP8_QUANTIZED = "fp8_quantized"
FULL_PRECISION = "full_precision"
OTHER = "other"
# The formats whose patched bytes are guarded by ``CLASSIC_BAKE_GUARD_RATIO`` past their
# measured points. FULL_PRECISION is not in it: no BF16, FP16 or FP32 cell exists yet, and
# those weights are charged by the line until the hardware block's two BF16 cells are read.
# Adding it here (with or without a row below) is the whole change if they call for it.
GUARDED_BY_BYTES: tuple[str, ...] = (FP8_QUANTIZED,)
# The largest patched weight under each guarded format's measured points. A stack whose
# largest patched weight of that format is no larger is charged by the line alone; a guarded
# format with no row here is always guarded. The FP8 row is read from the LTX 2.3 scaled FP8
# file's header (2026-11-12): the largest weight either measured stack patches is 64.0 MiB.
MEASURED_LARGEST_WEIGHT_BYTES = {FP8_QUANTIZED: 64 * MIB}
# What a stack is counted as when its checkpoint's header cannot be read: the largest count
# of patched weights measured (LTX 2.3's own stack). A stand-in, not a bound: a larger model
# whose header cannot be read may patch more.
UNREADABLE_STACK_WEIGHTS = 1660
# The bake's working memory beside the stack's own bytes (1.9 and 1.2 GiB measured).
IN_PLACE_BAKE_FIXED_BYTES = 2 * GIB

_FP8_DTYPES = frozenset({"F8_E4M3", "F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FNUZ"})
_FULL_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_QUANT_MARKERS = (".weight_scale", ".scale_weight", ".comfy_quant")
_WEIGHT_SUFFIXES = (".weight", ".bias")

# The sample request's field for the driver's permission (nodes/in_place_bake_permission.py).
# Only the value True grants it; absent, an older driver, a requeue or a recycle all read as no.
IN_PLACE_BAKE_PROVEN = "_dgxm_in_place_bake_proven"

# Why an in-place bake was not priced. Each is a whole clause; a refusal prints it after the
# classic price's terms (``StockFit.in_place_note``, ``stock_words.bake_sentences``).
GATE_HAS_NOT_PASSED = ("baking it in place was not tried: no identity-gate PASS for this combination's in-place "
                "bake came with this load; render it once with lora_low_rss on so the gate can prove it")


def weight_class(key: str, tensors: Mapping[str, TensorDescriptor]) -> str:
    """How the classic bake's guard reads one checkpoint weight, from the header alone."""
    dtype = tensors[key].dtype
    if dtype in _FULL_DTYPES:
        return FULL_PRECISION
    stem = key.rsplit(".", 1)[0]
    if dtype in _FP8_DTYPES and any(stem + marker in tensors for marker in _QUANT_MARKERS):
        return FP8_QUANTIZED
    return OTHER


def term_from_weights(weights: Iterable[tuple[str, int]], *, guard: bool = True) -> int:
    """The classic bake's charge for patched weights given as (class, stored bytes).

    ``guard=False`` charges the line alone. The partial-load route asks for
    that: its BF16 cell ran on hardware under the rule without this term, and
    the bytes guard has no measurement behind it.
    """
    count = 0
    by_class: dict[str, list[int]] = {}
    for kind, nbytes in weights:
        count += 1
        by_class.setdefault(kind, []).append(max(int(nbytes), 0))
    if not count:
        return 0
    line = CLASSIC_BAKE_FIXED_BYTES + CLASSIC_BAKE_BYTES_PER_WEIGHT * count
    if not guard:
        return line
    guarded = 0
    for kind in GUARDED_BY_BYTES:
        sizes = by_class.get(kind)
        if not sizes:
            continue
        measured = MEASURED_LARGEST_WEIGHT_BYTES.get(kind)
        if measured is None or max(sizes) > measured:
            guarded += sum(sizes)
    return max(line, int(CLASSIC_BAKE_GUARD_RATIO * guarded))


def _every_weight(tensors: Mapping[str, TensorDescriptor]) -> set[str]:
    return {key for key in tensors if key.endswith(_WEIGHT_SUFFIXES)}


def patched_weights(tensors: Mapping[str, TensorDescriptor], lora_stack: list | None,
                    resolve_lora_path: Callable[[str], str]) -> set[str]:
    """The checkpoint keys this stack patches; every weight when a file or a target cannot be read."""
    index = _base_index(tensors)
    keys: set[str] = set()
    for entry in lora_stack or ():
        name = entry.get("name") if isinstance(entry, dict) else None
        if not name:
            continue
        try:
            targets = _lora_targets(read_safetensors_header(resolve_lora_path(str(name))).tensors)
        except Exception:
            return _every_weight(tensors)
        if targets is None:
            return _every_weight(tensors)
        for stem, suffix in targets:
            found = _find_base_key(stem, suffix, index)
            if found is None:
                return _every_weight(tensors)
            keys.add(found)
    return keys


def classic_bake_bytes(checkpoint_path: str, lora_stack: list | None,
                       resolve_lora_path: Callable[[str], str]) -> int:
    """What the classic bake of ``lora_stack`` adds on the device beside this checkpoint's weights.

    A checkpoint whose header cannot be read gives no weights to count or to
    class, so the stack is charged the line for ``UNREADABLE_STACK_WEIGHTS``.
    """
    if not lora_stack:
        return 0
    try:
        tensors = read_safetensors_header(checkpoint_path).tensors
    except Exception:
        return term_from_weights((OTHER, 0) for _ in range(UNREADABLE_STACK_WEIGHTS))
    keys = patched_weights(tensors, lora_stack, resolve_lora_path)
    return term_from_weights((weight_class(key, tensors), tensors[key].nbytes) for key in keys)


def stack_bytes(lora_stack: list | None, resolve_lora_path: Callable[[str], str]) -> int:
    """The stack's LoRA files on disk; a file that cannot be sized counts as nothing."""
    return int(stack_file_bytes((lora_stack, resolve_lora_path)) or 0) if lora_stack else 0


def in_place_bake_bytes(stack_files: int) -> int:
    """What baking a stack in place on a loaded model takes with ``lora_low_rss`` off.

    One copy of the stack, read into host memory for the bake, and the bake's
    working memory. Nothing is staged on the device in that mode.
    """
    return max(int(stack_files), 0) + IN_PLACE_BAKE_FIXED_BYTES


def _gib(value: int) -> float:
    return round(value / GIB, 1)


def priced(fit: Any, checkpoint_path: str, lora_stack: list | None, resolve_lora_path: Callable[[str], str],
           *, held_bytes: int = 0, in_place: str | bool = "", bare_load: bool = False, charged: int = 0) -> Any:
    """``fit`` (a ``capacity_fit.StockFit``) with the classic bake charged, or re-priced as an in-place bake.

    Every site that prices one stack change calls this with the same
    arguments, so the quote, the store and the Worker's walls agree.

    ``held_bytes`` is what dropping the slot's resident returns: the classic
    change is compared with the reading plus that, and so is an in-place bake
    that must first reload the base bare. ``in_place`` is ``True`` when the
    store may bake this stack in place, else the clause that says why not (""
    when the question does not arise). ``bare_load`` says the slot does not
    hold this base bare: it is dropped, the base loads bare at the stock
    price, and the bake then needs its room beside the loaded copy.

    The in-place price is taken only when the classic one, so credited, does
    not fit and the in-place one does. A classic price that misses carries
    what the in-place bake would have needed and why it was not taken.

    ``charged`` is what this request's earlier slots already take out of the
    reading (the quote prices a pair's second model after its first). The
    bake is chosen against what is left, as the store chooses it when that
    slot's load runs beside the first model. The fit returned keeps the
    reading as read: ``capacity_quote._charge`` takes the charge off once.
    """
    if not lora_stack or not getattr(fit, "applies", False) or getattr(fit, "avail_bytes", None) is None:
        return fit
    read, spent = int(fit.avail_bytes), max(int(charged), 0)
    fit = replace(fit, avail_bytes=read - spent)
    return replace(_chosen(fit, checkpoint_path, lora_stack, resolve_lora_path, max(int(held_bytes), 0),
                           in_place, bare_load, "was left beside this request's other model" if spent else "was read"),
                   avail_bytes=read)


def _chosen(fit: Any, checkpoint_path: str, lora_stack: list, resolve_lora_path: Callable[[str], str],
            held: int, in_place: str | bool, bare_load: bool, reading_words: str) -> Any:
    """``priced``'s choice between the two bakes, against ``fit.avail_bytes`` as given."""
    files = stack_bytes(lora_stack, resolve_lora_path)
    classic = replace(fit, bake_bytes=classic_bake_bytes(checkpoint_path, lora_stack, resolve_lora_path) + files)
    classic = replace(classic, fits=classic.required_bytes <= classic.avail_bytes)
    if classic.required_bytes - held <= classic.avail_bytes or not in_place:
        return classic
    if in_place is not True:
        return replace(classic, in_place_note="; " + str(in_place))
    need = in_place_bake_bytes(files)
    if bare_load:
        need = max(int(fit.required_bytes) - ABSOLUTE_HOST_FLOOR_BYTES, int(fit.resident_bytes) + need)
        need = max(need - held, IN_PLACE_BAKE_FIXED_BYTES)  # the drop returns the resident first
    taken = replace(classic, in_place_bytes=need)
    if taken.required_bytes <= taken.avail_bytes:
        return replace(taken, fits=True)
    beside = "" if bare_load else f" beside the {_gib(held)} GiB loaded copy"
    return replace(classic, in_place_note=f"; baking it in place would need {taken.required_gib} GiB{beside}, "
                   f"and {taken.avail_gib} GiB {reading_words}")


def probe(lora_stack: list | None, resolve_lora_path: Callable[[str], str], *, held_bytes: int = 0,
          in_place: str | bool = "", bare_load: bool = False, charged: int = 0,
          stock_load_fit: Callable[[str, dict], Any] | None = None) -> Callable[[str, dict], Any]:
    """A stock probe (``capacity_quote.price``'s ``fit_probe``) that charges this load's classic bake."""
    def fit_probe(path: str, model_options: dict) -> Any:
        from . import capacity_fit

        return priced((stock_load_fit or capacity_fit.stock_load_fit)(path, model_options), path, lora_stack,
                      resolve_lora_path, held_bytes=held_bytes, in_place=in_place, bare_load=bare_load,
                      charged=charged)

    return fit_probe


__all__ = ["CLASSIC_BAKE_BYTES_PER_WEIGHT", "CLASSIC_BAKE_FIXED_BYTES", "CLASSIC_BAKE_GUARD_RATIO",
           "FP8_QUANTIZED", "FULL_PRECISION", "GATE_HAS_NOT_PASSED", "GUARDED_BY_BYTES",
           "IN_PLACE_BAKE_FIXED_BYTES", "IN_PLACE_BAKE_PROVEN", "MEASURED_LARGEST_WEIGHT_BYTES", "OTHER",
           "UNREADABLE_STACK_WEIGHTS", "classic_bake_bytes", "in_place_bake_bytes", "patched_weights", "priced",
           "probe", "stack_bytes", "term_from_weights", "weight_class"]
