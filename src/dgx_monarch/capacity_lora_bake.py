"""Price temporary device allocations from a slab-resident LoRA bake.

Under ``lora_low_rss``, ComfyUI allocates patched weights outside the slab
before ``WeightSlab.reabsorb`` reclaims them. This charge supplements
``slab_load_fit``'s file-size and floor terms. Header-only matching uses known
LoRA suffixes, normalized prefixes, and same-path fused aliases; it never
guesses a structural rename across model families. Callers supply parsed
checkpoint headers and the LoRA path resolver, avoiding torch/Comfy imports.

Mapped quantized weights retain their stored width; plain weights cost two
bytes per element. An unmapped target, unreadable LoRA, or resolution failure
charges the whole checkpoint file as a conservative bound. A LoRA that holds
only per-block attention configs patches no weight and charges nothing.

See docs/VALIDATION.md, "The LoRA bake's unpriced slab stray" (2026-09-28),
for the derivation and checks against campaign LoRA/checkpoint pairs.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from .safetensors_header import DTYPE_BITS, TensorDescriptor, read_safetensors_header

_NARROW_DTYPE_BITS = 8      # a quantized stray never gets upcast: stored width
_COMPUTE_DTYPE_BYTES = 2    # a plain key's compute dtype, regardless of storage

# One stem per matrix pair or delta, and the base suffix it maps to.
_LORA_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".lora_down.weight", ".weight"),
    (".lora_up.weight", ".weight"),
    (".lora_A.weight", ".weight"),
    (".lora_B.weight", ".weight"),
    (".diff", ".weight"),
    (".diff_b", ".bias"),
)

# A per-block attention pin a LoRA can carry; it patches no weight.
_ATTENTION_CONFIG_SUFFIX = ".comfy_attention.config"

_TEXT_ENCODER_MARKERS = (   # never reaches the diffusion checkpoint priced here
    "lora_te", "text_encoder", "text_encoders", "clip_l", "clip_g",
    "t5xxl", "t5_xxl", "conditioner",
)

# Stripped as one more candidate from either side (a checkpoint or a LoRA
# stem carries at most one); the combined chain covers a doubled
# ``model.diffusion_model.`` export prefix in one strip.
_PREFIX_CHAINS = ("modeldiffusionmodel", "diffusionmodel", "transformer",
                  "unet", "loraunet", "model")

# Tried once more, against a sibling fused/renamed key, when a diffusers-style
# projection stem misses directly.
_PROJECTION_ALIASES: tuple[tuple[str, str], ...] = (
    (".to_out.0", ".out"),
    (".to_out", ".out"),
    (".out_proj", ".out"),
    (".to_q", ".qkv"),
    (".to_k", ".qkv"),
    (".to_v", ".qkv"),
    (".q_proj", ".qkv"),
    (".k_proj", ".qkv"),
    (".v_proj", ".qkv"),
)

_NON_ALNUM = re.compile(r"[^0-9a-z]")


def _normalize(name: str) -> str:
    return _NON_ALNUM.sub("", name.lower())


def _prefix_variants(name: str) -> set[str]:
    """``name``, normalized, and once more with a known prefix chain stripped."""
    norm = _normalize(name)
    variants = {norm}
    for chain in _PREFIX_CHAINS:
        if norm.startswith(chain) and len(norm) > len(chain):
            variants.add(norm[len(chain):])
    return variants


def _is_text_encoder_key(raw_key: str) -> bool:
    low = raw_key.lower()
    return any(marker in low for marker in _TEXT_ENCODER_MARKERS)


def _lora_targets(lora_keys: Mapping[str, object]) -> set[tuple[str, str]] | None:
    """This LoRA's own (module stem, base suffix) pairs, or ``None`` when not
    one key named a recognized suffix (an unrecognized container, priced like
    any unmapped target, never as a free empty stack). A file whose recognized
    keys are all text-encoder ones returns a real empty set, and so does a file
    of per-block attention configs alone: ComfyUI stages each as an object
    patch, a new ``ComfyAttention`` module that holds no weight, so it bakes
    nothing (attention_pins.py)."""
    targets: set[tuple[str, str]] = set()
    saw_recognized_suffix = False
    if lora_keys and all(key.endswith(_ATTENTION_CONFIG_SUFFIX) for key in lora_keys):
        return targets
    for key in lora_keys:
        for suffix, base_suffix in _LORA_SUFFIXES:
            if key.endswith(suffix):
                saw_recognized_suffix = True
                if not _is_text_encoder_key(key):
                    targets.add((key[: -len(suffix)], base_suffix))
                break
    if not targets and not saw_recognized_suffix:
        return None
    return targets


def _base_index(tensors: Mapping[str, TensorDescriptor]) -> dict[tuple[str, str], str]:
    """(normalized stem variant, suffix) -> the checkpoint's own key, indexed
    under every prefix variant of that key too, not just the LoRA's."""
    index: dict[tuple[str, str], str] = {}
    for key in tensors:
        for suffix in (".weight", ".bias"):
            if key.endswith(suffix):
                for variant in _prefix_variants(key[: -len(suffix)]):
                    index.setdefault((variant, suffix), key)
                break
    return index


def _find_base_key(stem: str, suffix: str, index: dict[tuple[str, str], str]) -> str | None:
    """The checkpoint key this stem answers, direct or via a fused alias."""
    for variant in _prefix_variants(stem):
        found = index.get((variant, suffix))
        if found is not None:
            return found
    for alias_suffix, replacement in _PROJECTION_ALIASES:
        if not stem.endswith(alias_suffix):
            continue
        aliased = stem[: -len(alias_suffix)] + replacement
        for variant in _prefix_variants(aliased):
            found = index.get((variant, suffix))
            if found is not None:
                return found
    return None


def _priced_bytes(descriptor: TensorDescriptor) -> int:
    bits = DTYPE_BITS.get(descriptor.dtype)
    if bits is not None and bits <= _NARROW_DTYPE_BITS:
        return descriptor.nbytes
    numel = 1
    for dim in descriptor.shape:
        numel *= dim
    return numel * _COMPUTE_DTYPE_BYTES


def lora_bake_bytes(
    checkpoint_path: str,
    checkpoint_tensors: Mapping[str, TensorDescriptor],
    lora_stack: list[dict] | None,
    resolve_lora_path: Callable[[str], str],
) -> int:
    """The bake's stray transient for this stack, in bytes.

    ``checkpoint_tensors`` is the base checkpoint's own header, already
    parsed by the caller (``slab_load_fit`` reads it anyway to check the
    stored dtype); this never opens that file a second time. Any resolution
    failure, unreadable container, or unmapped target prices the whole
    checkpoint instead, per the module docstring's fallback derivation.
    """
    if not lora_stack:
        return 0
    size_bytes = (os.path.getsize(checkpoint_path)
                  if os.path.exists(checkpoint_path) else 0)
    index = _base_index(checkpoint_tensors)
    mapped: dict[str, int] = {}
    for entry in lora_stack:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not name:
            continue
        try:
            lora_path = resolve_lora_path(name)
            targets = _lora_targets(read_safetensors_header(lora_path).tensors)
        except Exception:
            # ``resolve_lora_path`` is caller-injected and untrusted here: a
            # missing file, an unbootstrapped comfy in a test host, or any
            # other resolution failure prices the checkpoint whole rather
            # than failing the capacity estimator.
            return size_bytes
        if targets is None:
            return size_bytes
        for stem, suffix in targets:
            found = _find_base_key(stem, suffix, index)
            if found is None:
                return size_bytes
            mapped.setdefault(found, _priced_bytes(checkpoint_tensors[found]))
    return sum(mapped.values())


def patched_base_keys(checkpoint_tensors: Mapping[str, TensorDescriptor], lora_paths: list[str]) -> set[str] | None:
    """The checkpoint keys a stack of LoRA files patches, or None when a file is unreadable."""
    index, keys = _base_index(checkpoint_tensors), set()
    for path in lora_paths:
        try:
            targets = _lora_targets(read_safetensors_header(path).tensors)
        except Exception:
            return None
        keys |= {key for stem, suffix in targets or () if (key := _find_base_key(stem, suffix, index))}
    return keys


def swap_credit_bytes(held: object, request_key: str, size_bytes: int, context: dict | None = None,
                      slab_weights: object = "auto") -> int:
    """One file of credit against the stray term, for a stock resident of this checkpoint.

    Under slab auto the store adopts such a resident and hot-swaps in place
    (``store_residency.policy_keeps_resident``), so no stray transient occurs;
    the quote still priced it as a swap. Explicit slab_weights=on and a pending
    slab retry reload, so they get none. Evidence: docs/VALIDATION.md, the LoRA bake section.
    """
    if (context or {}).get("would_load") != "swap" or slab_weights != "auto":  # WOULD_LOAD_SWAP
        return 0
    if not _same_base_stock_resident(held, request_key):
        return 0
    return max(int(size_bytes), 0)


def _keys(held: object, request_key: str) -> tuple[tuple, tuple] | None:
    """This request's key and the slot's, parsed, for a stock resident; else None."""
    import ast

    if not isinstance(held, dict) or held.get("residency") != "stock":
        return None
    try:
        mine, theirs = ast.literal_eval(request_key), ast.literal_eval(held.get("request_key") or "")
    except (ValueError, SyntaxError):
        return None
    ok = isinstance(mine, tuple) and isinstance(theirs, tuple) and len(mine) > 2 and len(theirs) > 2
    return (mine, theirs) if ok else None


def _same_base_stock_resident(held: object, request_key: str) -> bool:
    """Whether the slot holds a stock copy of this request's base, whatever its stack."""
    keys = _keys(held, request_key)
    # (*base_key, lora_signature, quant): the same base checkpoint, any stack.
    return keys is not None and keys[0][:-2] == keys[1][:-2] and keys[0][-1] == keys[1][-1]


def _names(signature: object) -> set[str] | None:
    """The LoRA files a stack signature ``((name, strength), ...)`` names, or None."""
    if not isinstance(signature, tuple) or not all(
            isinstance(entry, tuple) and entry and isinstance(entry[0], str) for entry in signature):
        return None
    return {entry[0] for entry in signature}


def held_backup_credit(held: object, request_key: str, size_bytes: int) -> int:
    """The host copy ComfyUI's backup holds that this LoRA change re-applies from, or 0.

    The Worker reports the backup's host bytes, a lower bound, only for the state
    measured on 2026-11-05 (``actor/worker_capacity._backup_bytes``). The held stack
    patched every weight a stack of its own files patches, so a change to such a
    stack (a strength, an order, a LoRA removed) swaps in place from that copy and
    places again only what the backup lacks: E3 grew the actor by 0.8 GiB. A file the
    held stack lacks gets nothing, since the store may reload, which keeps the backup
    through the fresh load. Never more than the host copy the stock price charges.
    """
    from .capacity_fit import STOCK_PLACEMENT_RATIO

    reported = held.get("backup_bytes") if isinstance(held, dict) else None
    keys = _keys(held, request_key)
    if (keys is None or not isinstance(reported, int) or isinstance(reported, bool)
            or reported <= 0):
        return 0
    mine, theirs = _names(keys[0][-2]), _names(keys[1][-2])
    if mine is None or not theirs or not mine <= theirs:
        return 0
    return min(reported, int(STOCK_PLACEMENT_RATIO * max(int(size_bytes), 0)))


def credit_hot_swap(fit: Any, held: object, request_key: str, context: dict | None,
                    slab_weights: object) -> Any:
    """The stock price of a LoRA change the store makes in place, with the held copy credited.

    The quote reads such a change as a swap and would price a whole stock load
    beside the base the slot already holds. The store does not load it again:
    ``store_bake.swaps_in_place`` says it keeps the base and re-applies the
    stack, which places at most the host copy of the weights the stack patches.
    So the file's own copy is credited and the host copy and the floor stay
    charged. Measured on 2026-11-04 on the head, the INT8 Wan-Animate 2 hot swap
    on a freshly loaded base took about 16 GiB of the box's free memory, 22.1 GiB
    priced this way, and its one-step sample about 10 GiB more; the BF16 file was
    refused at 69.1 GiB against 61.8 GiB available, where this price is 38.6 GiB.
    The Worker's own ladder never applies the credit (``store_residency.resolve``
    passes no snapshot), so a change that falls back to a fresh load is priced
    whole there. An explicit ``slab_weights=on`` drops a stock
    resident before it loads, so it gets no credit. A change the store makes by
    dropping this copy and loading fresh (``store_bake.copies_off_the_gpu``)
    keeps it: the drop returns the copy, and its wall prices the load whole. A
    change to a stack of the held stack's own files is also credited the backup
    the Worker reports (``held_backup_credit``); ``credit_bytes`` carries both, so a
    clear is still weighed against the price a cleared box pays (``old_model``).
    """
    from dataclasses import replace

    if ((context or {}).get("would_load") != "swap" or slab_weights is True  # WOULD_LOAD_SWAP
            or not getattr(fit, "applies", False) or getattr(fit, "avail_bytes", None) is None
            or not isinstance(held, dict) or held.get("swaps_in_place") is not True
            or not _same_base_stock_resident(held, request_key)):
        return fit
    # The copy the slot holds: the file's size, or a cast's own. The store may drop
    # it and load fresh (store_bake.copies_off_the_gpu), so a cast is credited no
    # more than the lower bound the Worker reports for it (credit_stock_swap's
    # figure), and nothing when it reports none; the price's cast figure is an
    # upper bound.
    held_bytes = int(getattr(fit, "resident_bytes", fit.size_bytes))
    if int(getattr(fit, "cast_bytes", 0) or 0) > 0:
        reported = held.get("resident_bytes")
        if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
            held_bytes = min(held_bytes, reported)
        else:
            held_bytes = 0
    if held_bytes <= 0:
        return fit
    # ComfyUI's backup of the held stack's weights, where this change re-applies from it.
    backup = 0 if int(getattr(fit, "cast_bytes", 0) or 0) > 0 else held_backup_credit(
        held, request_key, fit.size_bytes)
    credited = replace(fit, credit_bytes=held_bytes + backup, backup_bytes=backup)
    return replace(credited, fits=credited.required_bytes <= credited.avail_bytes)


def credit_stock_swap(fit: Any, held: object, context: dict | None) -> Any:
    """The stock price of a load that replaces the slot's own stock resident.

    The store never holds both copies. A stack change on the same file keeps
    the base (``credit_hot_swap``, the LoRA case of this credit), and every
    other swap drops the slot before it loads (``actor/store_load.fresh_load``).
    A drop of a stock resident returns its GPU copy in-process: on 2026-11-05
    the head rose from 58.7 to 88.5 GiB, and from 57.9 to 88.5, as the store
    dropped a 26.6 GiB bf16-cast Wan 2.2 DiT. So that copy is credited, the file
    or a lower bound of a cast's own, as the quote endpoint reports it in the
    slot's snapshot entry (``actor/worker_capacity._resident_bytes``) when the
    entry still describes the resident those figures were read from; the arena
    a stock load retains is not. A slab, comfy-managed or FSDP resident, and a slot whose
    load or cleanup is in flight, are priced as held.
    """
    from dataclasses import replace

    held_bytes = held.get("resident_bytes") if isinstance(held, dict) else None
    if (getattr(fit, "credit_bytes", 0) > 0 or (context or {}).get("would_load") != "swap"
            or not getattr(fit, "applies", False) or getattr(fit, "avail_bytes", None) is None
            or not isinstance(held, dict) or held.get("residency") != "stock"
            or not isinstance(held_bytes, int) or isinstance(held_bytes, bool) or held_bytes <= 0):
        return fit
    credited = replace(fit, credit_bytes=held_bytes, dropped=str(held.get("resident_name") or "a model"))
    return replace(credited, fits=credited.required_bytes <= credited.avail_bytes)


__all__ = ["credit_hot_swap", "credit_stock_swap", "held_backup_credit", "lora_bake_bytes",
           "patched_base_keys", "swap_credit_bytes"]
