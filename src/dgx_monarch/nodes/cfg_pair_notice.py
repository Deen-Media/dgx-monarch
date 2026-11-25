"""A log line a render, and a browser notice a prompt, when cfg2 runs an equal pair.

One GPU folds such a pair into one call of two entries; cfg2 makes two calls
of one entry each. Those two calls equal stock's own separate calls bit for
bit, and stock's batched call reads apart from them on the families measured
(docs/VALIDATION.md, the equal-pair records). `auto` leaves cfg2 for such a
pair only where the row it falls to is bytes equal to one GPU
(topology_cfg_evidence), so a render still on cfg2 with an equal pair either
chose cfg2 by hand or holds a file with no such row (Krea2 FP8, Chroma FP8 from
1.2 to 1.64 MP, any MXFP8 or NVFP4 file, a full-precision file cast to fp8 at
load). Nothing refuses: the notice says what differs, that cfg2 there equals
stock's separate calls, and which layout is bytes equal for the file where one
is recorded. A hand-chosen cfg2 shows it as a browser notice, once
per queued prompt (a Pipeline sweep is one prompt), and every render logs it;
one `auto` kept on cfg2 logs it at INFO only. It is not a mark on the node.

The sampler nodes, and the CFG guiders for Sampler Custom, read the pair before
the wire (cfg_pair.text_rows_equal) and put the answer in the request under
``REQUEST_KEY``; the render submission takes it off the request and holds it
across a healed retry, so it never reaches a Worker.
"""
from __future__ import annotations

from typing import Any

from ..log import get_logger

log = get_logger(__name__)

REQUEST_KEY = "_dgxm_cfg_pair_equal"

# What cfg2 read against one GPU's batched call on an equal pair, by family;
# read-only after import.
_READINGS = {
    "krea2": "one-step NRMS 0.0104 to 0.040 on Krea2 RAW BF16, zeroed and real negatives "
             "alike (2026-11-24)",
    "qwen_image21": "one-step NRMS 0.0004 to 0.0019 on Qwen-Image 2.1 BF16 (2026-11-24)",
    "chroma": "NRMS 0.034 on Chroma BF16 (2026-11-07)",
    "boogu": "one-step NRMS 0.014 to 0.025 on Boogu FP8 (2026-11-07)",
}
# Files whose pure Ulysses render on the Init node's Torch Flash kernel has a
# dated reading of bytes equal against one GPU, by (family, kind as auto reads
# it), each with the date and the reference it was read against: a one-GPU
# render, or row 0 of a same-build dp2 render, the one-GPU reference (each dp2
# rank renders its latent alone, with one GPU's cond and uncond batching). An
# fp8 kind here is FP8 storage: a block-scaled MXFP8 or NVFP4 file, and a
# full-precision file cast to fp8 at load, read apart
# (topology_cfg_evidence.pair_kind) and are never claimed.
ONE_GPU = "one GPU"
DP2_ROW_0 = "row 0 of a same-build dp2 render, the one-GPU reference"
# Read-only after import.
_BYTES_EQUAL_ULYSSES = {
    # Row 12: 2026-10-06 RAW BF16 at 1536 and 1448; 2026-11-24 at 1280, both zeroed forms.
    ("krea2", "bf16"): ("2026-10-06 and 2026-11-24", ONE_GPU),
    # Row 11: 2026-10-06 Turbo FP8 on TORCH_FLASH at 1536.
    ("krea2", "fp8"): ("2026-10-06", ONE_GPU),
    # Row 12: 2026-10-06 RAW INT8 at 1536.
    ("krea2", "int8"): ("2026-10-06", ONE_GPU),
    # Row 71: 2026-10-06 base 2512 FP8 at 1328 and 1024.
    ("qwen_image", "fp8"): ("2026-10-06", ONE_GPU),
    # Row 22: 2026-10-02 Chroma1-HD BF16, exact decoded pixels against a
    # same-build one-GPU render at 1024 and 1536 on the 44/44 and 100/28 pairs,
    # tensor bytes equal in the instrumented 1024 run.
    ("chroma", "bf16"): ("2026-10-02", ONE_GPU),
    # Campaign K, 2026-10-14: Chroma1-HD FP8-mixed and INT8 ConvRot at 1152 to
    # 1280 square, uly2 on Torch Flash bit-identical in every cell. auto keeps an
    # FP8 pair on cfg2 all the same, since its uly2 there runs SAGE (row 21).
    ("chroma", "fp8"): ("2026-10-14", DP2_ROW_0),
    ("chroma", "int8"): ("2026-10-14", DP2_ROW_0),
    # 2026-11-24 BF16 at 1024 under auto (uly2), with a zeroed negative.
    ("qwen_image21", "bf16"): ("2026-11-24", ONE_GPU),
    # Row 78: 2026-10-06 fp8_scaled file, uly2 bit-identical to dp2.
    ("boogu", "fp8"): ("2026-10-06", DP2_ROW_0),
}
# Driver state of process lifetime, one queued prompt at a time: the prompt
# whose browser notice already went out. ComfyUI runs one prompt at a time, and
# a Pipeline sweep is one prompt of several renders, so it gets one browser
# notice and a log line for every render.
_shown_for_prompt: str | None = None


def text(family: str, quant: str) -> str:
    """The notice for one render of ``family`` at ``quant``."""
    reading = _READINGS.get(family)
    differs = (f"differ from its batched call ({reading})" if reading is not None
               else "may differ from its batched call (not measured for this family)")
    dated = _BYTES_EQUAL_ULYSSES.get((family, quant))
    layout = (f"uly2 on the Torch Flash kernel renders this file bytes equal to one GPU "
              f"(read {dated[0]} against {dated[1]})." if dated is not None
              else "No layout is recorded bytes equal to one GPU's batched call for this file.")
    return (f"cfg2 is running a prompt and a negative of equal text rows. One GPU folds "
            f"such a pair into one call of two entries; cfg2 makes two calls of one, which "
            f"equal stock ComfyUI's own separate calls and {differs}. {layout}")


def announce(model: Any, topology: Any, equal_pair: bool) -> bool:
    """Publish the notice for this render if it is cfg2 on an equal pair.

    Every such render logs one line. A hand-chosen cfg2 also shows a browser
    notice, once per queued prompt; one `auto` kept on cfg2 logs at INFO only.
    Returns whether the render logged its line. Never raises into the render.
    """
    global _shown_for_prompt
    try:
        if not equal_pair or int(getattr(topology, "cfg", 1)) <= 1 \
                or int(getattr(topology, "sequence_parallel", 1)) > 1:
            return False
        family, quant = _family(model)
        chosen = getattr(getattr(model, "mesh", None), "topology_preset", "auto") != "auto"
        message = text(family, quant)
        prompt = _prompt_id()
        if not chosen or (prompt is not None and prompt == _shown_for_prompt):
            log.info("%s", message)
            return True
        from .. import first_render

        # first_render.notice writes the log line itself, beside the browser notice.
        first_render.notice("cfg2-equal-pair", message, f"cfg2 equal pair: {family} {quant}",
                            severity="warn", summary="DGX Monarch: cfg2 on an equal pair")
        _shown_for_prompt = prompt
        return True
    except Exception as exc:  # a notice must never fail a render
        log.debug("cfg2 equal-pair notice skipped: %r", exc)
        return False


def _prompt_id() -> str | None:
    """The id of the prompt ComfyUI is executing, or None outside one."""
    try:
        from comfy_execution.utils import get_executing_context

        prompt = getattr(get_executing_context(), "prompt_id", None)
    except Exception:
        return None
    return prompt if isinstance(prompt, str) and prompt else None


def _family(model: Any) -> tuple[str, str]:
    """The family and the kind auto reads for this file.

    The header's quant kind, read as fp8 under the loader's fp8 cast as auto
    reads it (topology_cfg_evidence.loaded_quant_kind), with a block-scaled or a
    cast fp8 file named apart (topology_cfg_evidence.pair_kind).
    """
    import folder_paths

    from .. import family_select
    from ..adapters.detect import sniff_checkpoint_for_topology
    from ..topology_cfg_evidence import loaded_quant_kind, pair_kind

    path = folder_paths.get_full_path("diffusion_models", model.unet_name)
    sniffed, quant = sniff_checkpoint_for_topology(path)
    weight = (getattr(model, "options", None) or {}).get("weight_dtype", "default")
    quant = loaded_quant_kind(quant, weight)
    family = family_select.effective_family(sniffed, family_select.override_for_model(model))
    return family, pair_kind(path, quant)
