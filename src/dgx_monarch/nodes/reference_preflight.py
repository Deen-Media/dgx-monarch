"""Driver reads of what a render's conditioning carries that a topology refuses.

Krea2's reference latents refuse every sharded topology (the preflight below).
And an equal prompt pair leaves a pair-scoped cfg2 row under ``auto`` only for
the Ulysses row it falls back to, which must be one this very render can run:
the sampler nodes ask ``sequence_split_refused`` whether that row's sharded
forward refuses what the pair carries where cfg2's stock forward runs it, and
tag the latent with the answer (topology_cfg_evidence.CfgEvidence.permits).
"""
from __future__ import annotations

from typing import Any

from .. import family_select

# The families with a pair-scoped cfg2 row, and the flux family Chroma's
# forward comes from, whose sharded forwards refuse a mask or
# index_timestep_zero references (adapters/qwen_image.py, flux_family.py,
# boogu.py), each read as ComfyUI hands it to the forward (comfy/model_base.py
# extra_conds). Qwen-Image passes on any mask. The flux family passes a mask
# only beside ``attention_mask_img_shape`` (a regional mask, resized to the
# latent's tokens), and its forward refuses one that is not a no-op; a plain
# text keep-mask never reaches it. Boogu passes a mask with a zero, and its
# forward refuses where the kept rows (num_tokens) differ from the caption's.
# Krea2's references refuse cfg2 too, so they keep no pair on cfg2.
_REFUSES_A_PUBLISHED_MASK = frozenset({"chroma", "flux", "flux2", "longcat"})
_REFUSES_A_MASKED_ROW = frozenset({"boogu"})
_REFUSES_ANY_MASK = frozenset({"qwen_image"})
_REFUSES_TIMESTEP_ZERO_REFERENCES = frozenset({"qwen_image", "chroma", "flux", "flux2",
                                               "longcat"})
TIMESTEP_ZERO = "index_timestep_zero"


def krea2_ref_preflight_summary(positive, negative=None) -> tuple[bool, str | None]:
    """Combine positive+negative raw CONDITIONING lists into one
    (has_reference_latents, resolved_method) summary for
    preflight_krea2_reference_latents. Call before conditioning_for_wire
    packs either list."""
    from ..adapters.krea2 import krea2_reference_latents_summary

    has_p, method_p = krea2_reference_latents_summary(positive)
    has_n, method_n = krea2_reference_latents_summary(negative)
    return has_p or has_n, method_p if method_p is not None else method_n


def preflight_krea2_reference_latents(
    model: Any, ref_summary: tuple[bool, str | None] | None, topo: Any
) -> None:
    """Refuse unsupported Krea2 reference latents before worker dispatch.

    The shared adapter predicate keeps this driver mirror aligned with the
    worker guard. Sniff failures leave that guard as the backstop. Model option
    patches remain worker-only because no DGX node serializes them.
    """
    if ref_summary is None:
        return
    has_refs, method = ref_summary
    from ..adapters.krea2 import (
        KREA2_REF_LATENTS_REFUSAL_MESSAGE,
        krea2_ref_latents_would_reject,
    )

    if not krea2_ref_latents_would_reject(has_refs, method):
        return
    if topo.sequence_parallel <= 1 and topo.cfg <= 1:
        return
    try:
        import folder_paths

        from ..adapters.detect import sniff_checkpoint

        path = folder_paths.get_full_path("diffusion_models", model.unet_name)
        if path is None:
            return
        family = family_select.effective_family(
            sniff_checkpoint(path)[0], family_select.override_for_model(model))
    except Exception:
        return  # fail open: the mid-forward guard is the backstop
    if family != "krea2":
        return
    from ..adapters.base import UnsupportedModelError

    raise UnsupportedModelError(KREA2_REF_LATENTS_REFUSAL_MESSAGE)


def _entries(*conditionings: object):
    for conditioning in conditionings:
        for entry in conditioning if isinstance(conditioning, list | tuple) else ():
            if isinstance(entry, list | tuple) and len(entry) == 2 and isinstance(entry[1], dict):
                yield entry[1]


def _masks_a_row(mask: Any) -> bool:
    try:
        return bool((mask == 0).any())
    except Exception:
        return True  # unreadable: assume it masks, which keeps cfg2


def _published_and_effective(entry: dict) -> bool:
    """Whether the flux family's forward receives this entry's mask and refuses it.

    Resizing to the latent's tokens keeps an all-visible mask all visible, so
    the mask as the conditioning holds it answers for the resized one.
    """
    if entry.get("attention_mask_img_shape") is None:
        return False
    try:
        from ..adapters.base import _mask_is_noop

        return not _mask_is_noop(entry["attention_mask"])
    except Exception:
        return True  # unreadable: assume it masks, which keeps cfg2


def _default_ref_method(path: str) -> str | None:
    """The method a Qwen-Image file defaults to, as comfy's model detection
    reads it: Edit 2511 carries an ``__index_timestep_zero__`` key."""
    from ..adapters.detect import read_safetensors_keys

    keys, _header = read_safetensors_keys(path)
    return TIMESTEP_ZERO if any(key.endswith("__index_timestep_zero__") for key in keys) else None


def sequence_split_refused(model: Any, positive: object, negative: object) -> bool:
    """Whether this family's sharded forward refuses what the pair carries,
    where cfg2's stock forward runs it.

    Qwen-Image refuses any text attention mask; Chroma and the flux family a
    regional mask ComfyUI passes them that hides a key; Boogu a mask that masks
    a row, whose num_tokens then differs from its caption rows. Qwen-Image and
    the flux family refuse references whose
    method is index_timestep_zero, named in the conditioning or, with none
    named, the default an Edit 2511 file carries. A header that cannot be read
    answers True where the pair carries a mask or references: cfg2 runs every
    one of these, so keeping it costs a notice, never a render.
    """
    entries = list(_entries(positive, negative))
    masks = [entry for entry in entries if entry.get("attention_mask") is not None]
    refs = [entry for entry in entries if entry.get("reference_latents")]
    if not masks and not refs:
        return False
    try:
        import folder_paths

        from ..adapters.detect import sniff_checkpoint

        path = folder_paths.get_full_path("diffusion_models", model.unet_name)
        family = family_select.effective_family(
            sniff_checkpoint(path)[0], family_select.override_for_model(model))
        if masks and (
                family in _REFUSES_ANY_MASK
                or (family in _REFUSES_A_PUBLISHED_MASK
                    and any(map(_published_and_effective, masks)))
                or (family in _REFUSES_A_MASKED_ROW
                    and any(_masks_a_row(entry["attention_mask"]) for entry in masks))):
            return True
        if refs and family in _REFUSES_TIMESTEP_ZERO_REFERENCES:
            methods = {entry.get("reference_latents_method") for entry in refs}
            if None in methods and family == "qwen_image":
                methods = (methods - {None}) | {_default_ref_method(path)}
            return TIMESTEP_ZERO in methods
    except Exception:
        return True
    return False
