"""Price temporary device allocations from a slab-resident LoRA bake.

Under ``lora_low_rss``, ComfyUI allocates patched weights outside the slab
before ``WeightSlab.reabsorb`` reclaims them. This charge supplements
``slab_load_fit``'s file-size and floor terms. Header-only matching uses known
LoRA suffixes, normalized prefixes, and same-path fused aliases; it never
guesses a structural rename across model families. Callers supply parsed
checkpoint headers and the LoRA path resolver, avoiding torch/Comfy imports.

Mapped quantized weights retain their stored width; plain weights cost two
bytes per element. An unmapped target, unreadable LoRA, or resolution failure
charges the whole checkpoint file as a conservative bound.

See docs/VALIDATION.md, "The LoRA bake's unpriced slab stray" (2026-09-28),
for the derivation and checks against campaign LoRA/checkpoint pairs.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping

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
    keys are all text-encoder ones returns a real empty set."""
    targets: set[tuple[str, str]] = set()
    saw_recognized_suffix = False
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
    import ast

    if (context or {}).get("would_load") != "swap" or slab_weights != "auto":  # WOULD_LOAD_SWAP
        return 0
    if not isinstance(held, dict) or held.get("residency") != "stock":
        return 0
    try:
        mine, theirs = ast.literal_eval(request_key), ast.literal_eval(held.get("request_key") or "")
    except (ValueError, SyntaxError):
        return 0
    # (*base_key, lora_signature, quant): the same base checkpoint, any stack.
    if not (isinstance(mine, tuple) and isinstance(theirs, tuple) and len(mine) > 2
            and mine[:-2] == theirs[:-2] and mine[-1] == theirs[-1]):
        return 0
    return max(int(size_bytes), 0)


__all__ = ["lora_bake_bytes", "patched_base_keys", "swap_credit_bytes"]
