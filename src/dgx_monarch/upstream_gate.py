"""Header-only gate for artifacts this release has no model forward for, whose
weight scales ComfyUI would drop, or whose metadata ComfyUI cannot build from.

World size, topology, budget, and consent cannot change this refusal. The
driver checks it at the loader before capacity, mesh healing, setup, or RPC;
Init may already have created the fleet. The worker uses the same message at
its load entry point.

Only standard-library modules load eagerly. Detection and the header reader
import at call time so headless callers can import the gate.
"""
from __future__ import annotations

import json
import os
import struct
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .safetensors_header import SafetensorsHeader

TROUBLESHOOTING = 91
DROPPED_SCALE_TROUBLESHOOTING = 120
H3_IMAGE_GRID_TROUBLESHOOTING = 149
H3_IMAGE_QUANT_METADATA_TROUBLESHOOTING = 150


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

# MiniMax-H3-Image (adapters/minimax_h3_image_header.py). Two faults of the file,
# each decided before any load: no sigma grid, then quantization metadata that
# cannot be read.
H3_IMAGE_GRID_REFUSAL = (
    "this MiniMax-H3-Image checkpoint's header has no usable transformer.pdd_sigmas (a "
    "list of numbers) in its config metadata. ComfyUI builds this model from that sigma "
    "grid and has no default for it, so stock ComfyUI cannot build it from this file (a "
    "file without the grid raises a TypeError), and dgx-monarch has nothing to render. "
    "Nothing was loaded and nothing was quarantined. No consent "
    "can supply the grid, so no card will appear and there is no waiver for this "
    "refusal. Use the file as Comfy-Org/MiniMax-H3-Image publishes it, whose config "
    "carries the grid, or restore the config metadata of the export it came from."
)

H3_IMAGE_QUANT_METADATA_REFUSAL = (
    "this MiniMax-H3-Image checkpoint's _quantization_metadata cannot be read: it is not "
    "JSON, or it holds no layers map. ComfyUI reads that metadata to load the quantized "
    "weights and fails to parse this one too (comfy.utils.convert_old_quants), so stock "
    "ComfyUI cannot load this file either, and dgx-monarch cannot tell which weight "
    "scales it would keep. Nothing was loaded and nothing was quarantined. No consent "
    "can repair the metadata, so no card will appear and there is no waiver for this "
    "refusal. Use the file as Comfy-Org/MiniMax-H3-Image publishes it, or restore the "
    "metadata of the export it came from."
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

# The closed default. A header neither reader below could read, with no proof
# that stock ComfyUI cannot read it, would otherwise load past every refusal above.
UNREAD_HEADER_TROUBLESHOOTING = 152
UNREAD_HEADER_REFUSAL = (
    "the upstream gate could not read this checkpoint's safetensors header, and "
    "the failure does not show that ComfyUI could not read it either. The gate "
    "decides from the header alone whether a file is one this release must not "
    "load (an artifact with no model forward here, weight scales ComfyUI would "
    "drop, metadata ComfyUI cannot build from), so a header it cannot read is "
    "refused, not loaded unchecked. Nothing was loaded and nothing was "
    "quarantined. No consent can read the header, so no card will appear and "
    "there is no waiver for this refusal. Use stock ComfyUI on one GPU instead, "
    "through ComfyUI's own Load Diffusion Model node rather than the DGX Monarch "
    "loader, or restore the file from the export it came from."
)


class QuantMetadataError(ValueError):
    """``_quantization_metadata`` that ComfyUI's ``convert_old_quants`` cannot parse either."""


class _Dtype(NamedTuple):
    dtype: str


class _LenientHeader(NamedTuple):
    """A header the strict reader refused, as far as the gate's predicates read it."""

    tensors: dict[str, _Dtype]
    metadata: dict[str, str]


def _lenient_header(path: str) -> _LenientHeader | None:
    """Read a header the strict reader refused, as ComfyUI's own reader reads it.

    ``comfy.utils.load_safetensors``, the laxest stock reader, takes the 8-byte
    length, decodes UTF-8 and calls plain ``json.loads``: the last copy of a
    key written twice wins, and no name, byte range or metadata type is
    checked. safetensors' own reader accepts a subset of that. This read hands
    the predicates the same names: every key but ``__metadata__`` with its
    dtype string, and the string entries of ``__metadata__`` (ComfyUI cannot
    parse any other entry either). It validates nothing else and feeds no load.

    None means no stock reader can read the file either: it does not open, its
    length field is missing, past the file or past the bound (both stock
    readers stop at a smaller one), its header is not UTF-8 JSON, or the JSON
    is not an object. Any other failure raises, and ``_verdict`` refuses on it.
    """
    from .safetensors_header import MAX_HEADER_BYTES

    try:
        handle = open(path, "rb")
    except OSError:
        return None
    with handle:
        size = os.fstat(handle.fileno()).st_size
        prefix = handle.read(8)
        length = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else None
        if length is None or length > min(MAX_HEADER_BYTES, size - 8):
            return None
        raw = handle.read(length)
    if len(raw) != length:  # the file changed under the read: no verdict on what it holds
        raise OSError(f"{Path(path).name}: the header read came short")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except ValueError:  # not UTF-8 or not JSON: both stock readers refuse it too
        return None
    if not isinstance(parsed, dict):
        return None
    tensors = {}
    for key, info in parsed.items():
        if key != "__metadata__":
            dtype = info.get("dtype") if isinstance(info, dict) else None
            tensors[key] = _Dtype(dtype if isinstance(dtype, str) else "")
    metadata = parsed.get("__metadata__")
    strings = ({key: value for key, value in metadata.items() if isinstance(value, str)}
               if isinstance(metadata, dict) else {})
    return _LenientHeader(tensors, strings)


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
    spells it, and nothing else changes; a table that is JSON ``null`` names
    no layer, and ComfyUI loads on. Without it, a bare ``scaled_fp8`` key
    marks the legacy layout: the marker goes, each ``<layer>.scale_weight``
    becomes ``<layer>.weight_scale`` and its layer gains a ``comfy_quant``
    record, and each ``scale_input`` becomes ``input_scale``. ComfyUI also
    drops an ``input_scale`` of exactly 1.0, a value a header cannot show; no
    scale the gate examines depends on it.
    """
    out = dict(keys)
    if "_quantization_metadata" in metadata:
        try:
            table = json.loads(metadata["_quantization_metadata"])
            layers = [] if table is None else list(table["layers"].keys())
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            # Not JSON, no ``layers``, or ``layers`` not a map: ComfyUI raises on the
            # same text, so its load fails and no scale drops silently.
            raise QuantMetadataError(f"_quantization_metadata cannot be parsed ({exc!r})") from exc
        for layer in layers:
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
) -> tuple[bool, tuple[str, ...], int | None]:
    """One header read per file identity: (L2P key layout, uncovered layers,
    a MiniMax-H3-Image refusal number or None).

    An L2P layout returns before the scale check, and its test never raises on
    the metadata, so a malformed ``_quantization_metadata`` cannot turn #91 off.

    The strict reader refuses headers that stock ComfyUI reads (a key written
    twice, an empty tensor name, shared byte ranges, metadata that is not all
    strings), so its failure is no verdict: the header is read again through
    ``_lenient_header`` and the same predicates answer on what ComfyUI would
    load. A damaged header therefore gets every refusal an intact one gets.
    """
    from .adapters import gate_keys
    from .adapters import minimax_h3_image_header as h3_image_header
    from .adapters.detect import is_zimage_l2p_from_keys
    from .log import get_logger
    from .safetensors_header import read_safetensors_header

    header: SafetensorsHeader | _LenientHeader
    try:
        header = read_safetensors_header(path)
    except Exception as exc:
        lenient = _lenient_header(path)
        if lenient is None:
            raise  # no stock reader reads it either: the strict reader's error stands
        header = lenient
        get_logger(__name__).info(
            "upstream gate: %s failed the strict header read (%s); its header was "
            "read again as ComfyUI reads it", path, exc)
    # Both family tests read ComfyUI's exact keys first (adapters/gate_keys.py),
    # so an extra tensor that moves the header detector's reading cancels neither.
    names = list(header.tensors)
    comfy_keys = gate_keys.comfy_detection_keys(
        {key: info.dtype for key, info in header.tensors.items()}, header.metadata)
    if gate_keys.reads_zimage_l2p(names, comfy_keys):
        if isinstance(header, _LenientHeader) or not is_zimage_l2p_from_keys(names):
            # detect.sniff_zimage_l2p reads strictly and by its own needles, so it
            # cannot log this file.
            get_logger(__name__).info(
                "upstream gate: %s carries the Z-Image L2P decoder (%s) and no dec_net head",
                path, ", ".join(sorted(key for key in names if key.startswith(
                    "local_decoder.") or ".local_decoder." in key)[:4]))
        return True, (), None
    # Decided from keys and the config metadata alone, before any quantization
    # metadata is parsed, so a damaged ``_quantization_metadata`` cannot cancel it.
    h3_image = gate_keys.reads_minimax_h3_image(names, comfy_keys)
    h3_image_number = (H3_IMAGE_GRID_TROUBLESHOOTING
                       if h3_image and h3_image_header.grid_missing(header.metadata) else None)
    try:
        uncovered = _dropped_scale_layers(comfy_load_keys(
            {key: info.dtype for key, info in header.tensors.items()}, header.metadata))
    except QuantMetadataError:
        if not h3_image:
            # No verdict. Stock ComfyUI fails to parse the same metadata at load
            # (comfy.utils.convert_old_quants), so no scale is silently dropped.
            raise
        # This family fails closed before any load: the grid refusal stands, or
        # the metadata refusal answers for metadata ComfyUI cannot parse either.
        uncovered = ()
        if h3_image_number is None:
            h3_image_number = H3_IMAGE_QUANT_METADATA_TROUBLESHOOTING
    if uncovered:
        # Once per file identity, because the cache holds the answer.
        get_logger(__name__).info(
            "upstream gate: %s carries %d weight scale(s) ComfyUI would drop (%s)",
            path, len(uncovered), ", ".join(uncovered[:4]))
    return False, uncovered, h3_image_number


def _verdict(path: str | None) -> tuple[int, tuple[str, ...]] | None:
    """(TROUBLESHOOTING number, uncovered layers) for a refused path, or None.

    Total. An unnamed, missing or non-safetensors path answers None. So does a
    file no stock reader can read either, or one whose quantization metadata
    ComfyUI cannot parse (``_stock_fails_too``): that load fails by itself.
    Any other failed read refuses (#152), because a load that skipped the gate
    could render wrong. One header read answers every question; an L2P layout
    is read once more, for the log line alone, through
    ``detect.sniff_zimage_l2p``, which logs the matched key paths. A
    MiniMax-H3-Image file without its grid answers before dropped scales; one whose
    quantization metadata cannot be read answers in their place, since the scale
    check could not finish.
    """
    if not path or not str(path).lower().endswith((".safetensors", ".sft")):
        return None
    try:
        if not os.path.isfile(str(path)):  # missing, or a folder: the load's own error stands
            return None
        identity = _identity(str(path))
    except (OSError, ValueError):
        return None
    try:
        l2p_layout, layers, h3_image_number = _read_by_identity(*identity)
    except Exception as exc:
        if _stock_fails_too(exc):
            return None
        from .log import get_logger

        get_logger(__name__).warning(
            "upstream gate: %s: the header read failed with no verdict on the file (%r)",
            path, exc)
        return UNREAD_HEADER_TROUBLESHOOTING, ()
    if l2p_layout:
        from .adapters import detect

        detect.sniff_zimage_l2p(str(path))  # the log line; the layout read above is the verdict
        return TROUBLESHOOTING, ()
    if h3_image_number == H3_IMAGE_GRID_TROUBLESHOOTING:
        return h3_image_number, ()
    if layers:
        return DROPPED_SCALE_TROUBLESHOOTING, layers
    return (h3_image_number, ()) if h3_image_number is not None else None


def _stock_fails_too(exc: Exception) -> bool:
    """Whether a failed read needs no refusal, because the load fails by itself.

    True for the strict reader's error, which ``_read_by_identity`` lets out
    only when no stock reader can read the header either; for
    ``_quantization_metadata`` ComfyUI cannot parse either; and for a process
    that cannot import detection at all: a headless caller loads nothing.
    """
    from .safetensors_header import SafetensorsHeaderError

    return isinstance(exc, (SafetensorsHeaderError, QuantMetadataError, ImportError))


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
    if number == UNREAD_HEADER_TROUBLESHOOTING:
        return UNREAD_HEADER_REFUSAL, number
    if number == TROUBLESHOOTING:
        return ZIMAGE_L2P_REFUSAL, number
    if number == H3_IMAGE_GRID_TROUBLESHOOTING:
        return H3_IMAGE_GRID_REFUSAL, number
    if number == H3_IMAGE_QUANT_METADATA_TROUBLESHOOTING:
        return H3_IMAGE_QUANT_METADATA_REFUSAL, number
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
    if number == UNREAD_HEADER_TROUBLESHOOTING:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, UNREAD_HEADER_REFUSAL, troubleshooting=number))
    if number == TROUBLESHOOTING:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, ZIMAGE_L2P_REFUSAL, troubleshooting=TROUBLESHOOTING))
    if number == H3_IMAGE_GRID_TROUBLESHOOTING:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, H3_IMAGE_GRID_REFUSAL, troubleshooting=number))
    if number == H3_IMAGE_QUANT_METADATA_TROUBLESHOOTING:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, H3_IMAGE_QUANT_METADATA_REFUSAL, troubleshooting=number))
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS, DROPPED_SCALE_REFUSAL.format(**_scale_fields(layers)),
        troubleshooting=DROPPED_SCALE_TROUBLESHOOTING))
