"""MiniMax H3 request features the packed sequence-parallel forward cannot honor.

Each check runs before any ComfyUI import or collective in the H3 forward
(adapters/minimax_h3.py) and refuses with what the feature needs and where it
runs instead: a missing or partial carried-audio payload, a batch above one, an
attention mask, a 'dit' block replacement, and a latent noise mask.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError


def _has_current_audio_carry_outer(forward: Any) -> bool:
    """Fingerprint the outer audio-carry contract required by this inner return.

    The payload and model modules can be version-skewed while method signatures
    remain identical, so code constants distinguish the required conversion.
    """
    function = getattr(forward, "__func__", forward)
    code = getattr(function, "__code__", None)
    if code is None:
        return False
    constants = frozenset(value for value in code.co_consts if isinstance(value, str))
    return {
        "audio_scale",
        "minimax_h3_sigma_shift_video",
        "minimax_h3_sigma_shift_audio",
    } <= constants and "time_shift_sigma" in code.co_names


def _reject_latent_noise_masks(denoise_mask: Any, audio_denoise_mask: Any) -> None:
    """Refuse a latent noise mask instead of dropping it.

    Comfy ff6c8a8 added the two mask parameters. The message below says what
    H3 does with a mask, in the block stream and the output head alike, and
    why this forward, which plans from the unmasked stream times, cannot
    honor it. The values are request-derived and identical on every rank
    before the first collective, so class P retires the sample lease consumed
    and the operator re-queues on the same fleet. The exception is a first-use
    gate ceremony, which proves the operator's own frozen latent: a refusal
    there aborts the ceremony and costs one Recycle, the contract
    docs/TROUBLESHOOTING.md #49 records for a mid-ceremony refusal.
    """
    for name, mask in (("denoise_mask", denoise_mask),
                       ("audio_denoise_mask", audio_denoise_mask)):
        if mask is None:
            continue
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"minimax_h3: this render carries a '{name}'. H3 runs masked rows at their "
            "own timestep: it shifts the stream's modulation row when the kept rows "
            "share one value, and turns that row into a per-token index tensor when "
            "they do not. This distributed forward plans its modulation rows from the "
            "unmasked stream times alone, so it would renoise the rows the mask "
            "preserves. Clear the latent mask, or take a true single-GPU reference "
            "(mode=local, gpus_per_host=1) (docs/MODELS.md).",
        ))


def _reject_unsupported_h3_features(
    x: Any,
    transformer_options: dict[str, Any],
    minimax_payload: Any,
    kwargs: dict[str, Any],
    outer_forward: Any,
) -> None:
    """Refuse unsupported inputs before ComfyUI imports or collectives."""
    # Request-derived payload failures are identical on every rank before the
    # first collective, so class-P handling retires the sample lease consumed.
    payload_error: str | None = None
    if not minimax_payload:
        payload_error = refusal(
            RefusalClass.PHYSICS,
            "minimax_h3: this render carries no minimax_payload content. Stock comfy always "
            "publishes at least one key (the payload holds the packed layout, the condition "
            "latents, the presentation token tags and the sampler seed), so an absent or "
            "empty payload means the conditioning never reached this worker. Rendering "
            "anyway would turn a keyframe or reference-clip render into plain "
            "text-to-video with no error. Re-run the workflow, or take a true single-GPU "
            "reference (mode=local, gpus_per_host=1) (docs/MODELS.md).",
        )
    elif not isinstance(minimax_payload, dict) or "audio_scale" not in minimax_payload:
        payload_error = refusal(
            RefusalClass.PHYSICS,
            "minimax_h3: minimax_payload is missing the required 'audio_scale' marker for "
            "ComfyUI's carried-audio outer-forward contract. This payload came from an "
            "older or partial ComfyUI path; this distributed inner forward returns raw "
            "audio velocity, so running without the matching outer conversion would "
            "silently integrate the wrong audio ODE. Update ComfyUI consistently on every "
            "host, or take a true single-GPU reference (mode=local, gpus_per_host=1) "
            "(docs/MODELS.md).",
            troubleshooting=68,
        )
    else:
        audio_scale = minimax_payload["audio_scale"]
        audio_scale_value: float | None = None
        if not isinstance(audio_scale, bool) and isinstance(audio_scale, Real):
            try:
                audio_scale_value = float(audio_scale)
            except Exception:
                # A custom Real may fail conversion; keep it inside the typed gate.
                pass
        if (
            audio_scale_value is None
            or not math.isfinite(audio_scale_value)
            or audio_scale_value <= 0.0
        ):
            payload_error = refusal(
                RefusalClass.PHYSICS,
                "minimax_h3: minimax_payload['audio_scale'] must be a finite positive "
                "numeric value from ComfyUI's carried-audio contract; got an invalid "
                f"{type(audio_scale).__name__} value. "
                "Update ComfyUI consistently on every host, or take a true single-GPU "
                "reference (mode=local, gpus_per_host=1) (docs/MODELS.md).",
                troubleshooting=68,
            )
        elif not _has_current_audio_carry_outer(outer_forward):
            # Must stay untagged: host code can differ across ranks, so one rank
            # may refuse while another enters a collective. The lease retires
            # abandoned and needs one Recycle before redispatch
            # (docs/TROUBLESHOOTING.md #68).
            payload_error = (
                "minimax_h3: minimax_payload carries 'audio_scale', but the bound stock "
                "MiniMaxH3Model.forward does not carry ComfyUI's matching audio unscale "
                "and velocity-conversion contract. This is a partial or mixed ComfyUI "
                "update; distributed execution would silently integrate the wrong audio "
                "ODE. Update ComfyUI consistently on every host, then reset the Attached "
                "mesh; or take a true single-GPU reference (mode=local, gpus_per_host=1) "
                "(docs/MODELS.md, docs/TROUBLESHOOTING.md #68)."
            )
    if payload_error is not None:
        raise UnsupportedModelError(payload_error)
    video = x[0] if isinstance(x, (list, tuple)) and len(x) else None
    shape = getattr(video, "shape", None)
    if shape is not None and len(shape) and int(shape[0]) != 1:
        from .minimax_h3 import MINIMAX_H3_BATCH_CAPPED_MESSAGE

        raise UnsupportedModelError(
            f"minimax_h3: the packed forward received batch {int(shape[0])}, and this "
            "family runs batch 1 only. " + MINIMAX_H3_BATCH_CAPPED_MESSAGE
        )
    for name in ("attention_mask", "attn_mask"):
        if kwargs.get(name) is not None:
            raise UnsupportedModelError(
                f"minimax_h3: this render carries a '{name}', but H3's stock attention is "
                "maskless and fully bidirectional, so the sharded kernel cannot apply "
                "one. Run masked workflows on a true single-GPU reference (mode=local, "
                "gpus_per_host=1) (docs/MODELS.md)."
            )
    if transformer_options.get("patches_replace", {}).get("dit", {}):
        raise UnsupportedModelError(
            "minimax_h3: a 'dit' block replacement receives the token stream, the rotation "
            "table and the modulation segment table, all three of which are rank-local under "
            "sequence parallelism, so the patch would see a partial sequence and rank-local "
            "coordinates. Run this workflow on a true single-GPU reference (mode=local, "
            "gpus_per_host=1) (docs/MODELS.md)."
        )
