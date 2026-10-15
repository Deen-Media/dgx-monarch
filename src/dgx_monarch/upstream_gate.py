"""Header-only gate for artifacts this release has no model forward for, or
whose weight scales ComfyUI would drop.

World size, topology, budget, and consent cannot change this refusal. The
driver checks it at the loader before capacity, mesh healing, setup, or RPC;
Init may already have created the fleet. The worker uses the same message at
its load entry point.

Only standard-library modules load eagerly. Detection and the header reader
import at call time so headless callers can import the gate.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

TROUBLESHOOTING = 91
DROPPED_SCALE_TROUBLESHOOTING = 120


# The body only: the class tag is built at the raise, never here. An
# import-time tag binds to a stale refusal module in a long-lived process
# (see the comment above actor/partial_load_guard.REFUSAL_TEXT).
ZIMAGE_L2P_REFUSAL = (
    "this Z-Image checkpoint is the L2P pixel-space surface (local_decoder "
    "weights, no dec_net head), and this release carries no L2P forward. "
    "Current ComfyUI has no L2P model contract, so it reads the header as "
    "latent Z-Image with a memory_usage_factor of 2.8, asks for about 129 GiB "
    "of sample-time memory at render size on a 121 GiB box, and streams every "
    "block on every step until the host runs out. Even if memory held, the latent forward "
    "would run on pixel-space weights and the image would be wrong. Nothing "
    "was loaded and nothing was quarantined. No consent can supply a missing "
    "forward, so no card will appear and there is no waiver for this refusal. "
    "Use the latent Z-Image checkpoint or the DCT PixelSpace checkpoint "
    "instead; both are hardware validated. This refusal retires when ComfyUI "
    "carries the L2P model contract (upstream draft Comfy-Org/ComfyUI#14055) "
    "and the L2P forward lands (issue #279 section 1)."
)

DROPPED_SCALE_REFUSAL = (
    "this checkpoint stores {count} quantized layer(s), first {first}, whose "
    "weight scale ComfyUI does not read: the layer has no comfy_quant key "
    "and no _quantization_metadata entry that ComfyUI matches to it, and its "
    "scale is not a scale_weight that ComfyUI's legacy scaled_fp8 conversion "
    "renames. ComfyUI drops those scales as unexpected keys. An fp8 or int8 "
    "file then loads without an error and renders with its scales ignored, "
    "so the image is wrong; a packed 4-bit file fails at load or at the "
    "first step. Nothing was loaded and nothing was quarantined. No consent "
    "can make ComfyUI read these scales, so no card will appear and there is "
    "no waiver for this refusal. Use an export of the same model in "
    "ComfyUI's quantized layout (per-layer comfy_quant keys or "
    "_quantization_metadata), or its bf16 file."
)

# Weight dtypes whose scale sidecar ComfyUI must be told about. F8 (every
# variant), F4 and I8 follow adapters.detect._has_scale_beside_quantized_weight.
# U8 holds packed 4-bit weights: under DynamicVRAM ComfyUI's lazy Linear load
# assigns the packed tensor with no shape check and drops the scale, and the
# render fails only at the first matmul, so U8 is examined too
# (tests/test_dropped_scale_gate.py holds both load paths).
SCALED_WEIGHT_DTYPES = ("I8", "F4", "U8")

# ComfyUI's quantized ops read <layer>.weight_scale. A legacy scaled_fp8 file
# stores <layer>.scale_weight, which only the legacy conversion renames.
_SCALE_SUFFIXES = (".weight_scale", ".scale_weight")

# comfy.model_detection.unet_prefix_from_state_dict's candidates, in its order.
_UNET_PREFIXES = ("model.diffusion_model.", "model.model.", "net.")
_SENSENOVA_KEYS = (
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
    "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
)


def _scaled_weight_dtype(dtype: str) -> bool:
    return dtype.startswith("F8") or dtype in SCALED_WEIGHT_DTYPES


def uncovered_scaled_layers(path: str) -> tuple[str, ...]:
    """Layers whose weight scales ComfyUI would drop, sorted; raises on a bad header.

    The gate replays, on key names alone, what
    ``comfy.sd.load_diffusion_model_state_dict`` does before the model loads
    (``comfy_load_keys``). It then names every layer whose ``weight_scale``
    sits beside an F8, F4, I8 or U8 weight with no ``comfy_quant`` record, and
    every ``scale_weight`` the legacy conversion left unrenamed: a plain
    Linear drops both as unexpected keys. Layer names are the ones the model
    sees, after ComfyUI's prefix strip. An L2P layout answers none, because
    #91 refuses it first.
    """
    return _read_by_identity(*_identity(path))[1]


def _convert_old_quants(keys: dict[str, str], metadata: dict[str, str]) -> dict[str, str]:
    """``comfy.utils.convert_old_quants(sd, "", metadata)`` on names and dtypes.

    With ``_quantization_metadata`` in the header, every layer its ``layers``
    table names gains a ``<layer>.comfy_quant`` key, spelled as the table
    spells it, and nothing else changes. Without it, a bare ``scaled_fp8`` key
    marks the legacy layout: the marker goes, each ``<layer>.scale_weight``
    becomes ``<layer>.weight_scale`` and its layer gains a ``comfy_quant``
    record, and each ``scale_input`` becomes ``input_scale``. ComfyUI also
    drops an ``input_scale`` of exactly 1.0, a value a header cannot show; no
    scale the gate examines depends on it.
    """
    out = dict(keys)
    if "_quantization_metadata" in metadata:
        for layer in json.loads(metadata["_quantization_metadata"])["layers"].keys():
            out[f"{layer}.comfy_quant"] = "U8"
        return out
    if "scaled_fp8" not in keys:
        return out
    out.clear()
    recorded: list[str] = []
    for key, dtype in keys.items():
        if key == "scaled_fp8":
            continue
        if key.endswith(".scale_weight"):
            recorded.append(key[: -len(".scale_weight")])
            key = f"{recorded[-1]}.weight_scale"
        elif key.endswith(".scale_input"):
            key = key[: -len(".scale_input")] + ".input_scale"
        out[key] = dtype
    for layer in recorded:
        out[f"{layer}.comfy_quant"] = "U8"
    return out


def _unet_prefix(keys: dict[str, str]) -> str:
    """``comfy.model_detection.unet_prefix_from_state_dict`` on key names."""
    if (any(key.startswith("detector.") for key in keys)
            and any(key.startswith("tracker.") for key in keys)):
        return ""
    if all(key in keys for key in _SENSENOVA_KEYS):
        return ""
    counts = dict.fromkeys(_UNET_PREFIXES, 0)
    for key in keys:
        for prefix in _UNET_PREFIXES:
            if key.startswith(prefix):
                counts[prefix] += 1
                break
    top = max(counts, key=counts.__getitem__)
    return top if counts[top] > 5 else "model."


def comfy_load_keys(dtypes: dict[str, str], metadata: dict[str, str]) -> dict[str, str]:
    """The keys, with their dtypes, ComfyUI's diffusion-model load hands the model.

    The steps of ``comfy.sd.load_diffusion_model_state_dict`` (ComfyUI
    0.41.0): a conversion, one prefix strip that filters out every key
    outside the prefix, and, when any key survives, a second conversion on the
    survivors. So a ``_quantization_metadata`` name covers a layer only when it
    is the layer's stripped name or its full name under that prefix, and a
    ``comfy_quant`` key or ``scaled_fp8`` marker outside the prefix reaches
    nothing. tests/test_dropped_scale_gate.py checks the result against
    ComfyUI's own load, key for key.
    """
    keys = _convert_old_quants(dtypes, metadata)
    prefix = _unet_prefix(keys)
    stripped = {key[len(prefix):]: dtype for key, dtype in keys.items()
                if key.startswith(prefix)}
    return _convert_old_quants(stripped, metadata) if stripped else keys


def _dropped_scale_layers(keys: dict[str, str]) -> tuple[str, ...]:
    layers: set[str] = set()
    for key in keys:
        suffix = next((s for s in _SCALE_SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            continue
        layer = key[: -len(suffix)]
        if not _scaled_weight_dtype(keys.get(f"{layer}.weight", "")):
            continue
        if suffix == ".weight_scale" and f"{layer}.comfy_quant" in keys:
            continue
        layers.add(layer)
    return tuple(sorted(layers))


def _identity(path: str) -> tuple[str, int, int, int]:
    stat = Path(path).stat()
    return str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


@lru_cache(maxsize=16)
def _read_by_identity(
    path: str, _mtime_ns: int, _ctime_ns: int, _size: int
) -> tuple[bool, tuple[str, ...]]:
    """One header read per file identity: (L2P key layout, uncovered layers).

    An L2P layout returns before the metadata is parsed, so a malformed
    ``_quantization_metadata`` cannot turn the #91 refusal off.
    """
    from .adapters.detect import is_zimage_l2p_from_keys
    from .safetensors_header import read_safetensors_header

    header = read_safetensors_header(path)
    if is_zimage_l2p_from_keys(list(header.tensors)):
        return True, ()
    uncovered = _dropped_scale_layers(comfy_load_keys(
        {key: info.dtype for key, info in header.tensors.items()}, header.metadata))
    if uncovered:
        from .log import get_logger

        # Once per file identity, because the cache holds the answer.
        get_logger(__name__).info(
            "upstream gate: %s carries %d weight scale(s) ComfyUI would drop (%s)",
            path, len(uncovered), ", ".join(uncovered[:4]))
    return False, uncovered


def _verdict(path: str | None) -> tuple[int, tuple[str, ...]] | None:
    """(TROUBLESHOOTING number, uncovered layers) for a refused path, or None.

    Total: an unnamed, unreadable or non-safetensors path answers None, so a
    failed read degrades to a normal load rather than to a refusal. One header
    read answers both questions; only an L2P layout reads again, through
    ``detect.sniff_zimage_l2p``, which logs the matched key paths.
    """
    try:
        if not path or not str(path).lower().endswith((".safetensors", ".sft")):
            return None
        l2p_layout, layers = _read_by_identity(*_identity(str(path)))
        if l2p_layout:
            from .adapters import detect

            if detect.sniff_zimage_l2p(str(path)):
                return TROUBLESHOOTING, ()
        return (DROPPED_SCALE_TROUBLESHOOTING, layers) if layers else None
    except Exception:  # a broken read never refuses a load
        return None


def _scale_fields(layers: tuple[str, ...]) -> dict[str, object]:
    # A layer name comes from the file: cap it so the body stays short.
    return {"count": len(layers), "first": layers[0][:160]}


def refusal_for(path: str | None) -> tuple[str, int] | None:
    """The refusal body and its TROUBLESHOOTING number for ``path``, or None.

    Total, as ``_verdict`` is. The sweep matrix and the tests read it;
    the load funnels raise through ``raise_refusal``.
    """
    verdict = _verdict(path)
    if verdict is None:
        return None
    number, layers = verdict
    if number == TROUBLESHOOTING:
        return ZIMAGE_L2P_REFUSAL, number
    return DROPPED_SCALE_REFUSAL.format(**_scale_fields(layers)), number


def raise_refusal(path: str | None) -> None:
    """Raise the class P refusal ``refusal_for`` names, or return.

    The driver's loader preflight and the Worker's load funnel both call this,
    so the two raise the same text and number. The tag is built here, at the
    raise, with each body named where the refusal ledger can read it.
    """
    verdict = _verdict(path)
    if verdict is None:
        return
    from .adapters.base import UnsupportedModelError
    from .refusal import RefusalClass, refusal

    number, layers = verdict
    if number == TROUBLESHOOTING:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, ZIMAGE_L2P_REFUSAL, troubleshooting=TROUBLESHOOTING))
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS, DROPPED_SCALE_REFUSAL.format(**_scale_fields(layers)),
        troubleshooting=DROPPED_SCALE_TROUBLESHOOTING))
