"""Detect checkpoint family and quantization from safetensors headers.

Auto topology reads architecture-specific parameter paths without tensors."""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..log import get_logger
from ..safetensors_header import (
    SafetensorsFileIdentity,
    SafetensorsHeaderError,
    read_safetensors_header,
)

log = get_logger(__name__)
# Every listed module-path segment must appear. `patchify_proj.` separates LTX
# from other layouts carrying `adaln_single` and `transformer_blocks`.
_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("krea2", ("txtfusion.",)),
    ("ideogram4", ("llm_cond_proj.",)),
    ("pixeldit_comfy", ("pixel_embedder.", "pixel_blocks.")),  # comfy PixDiT_T2I + PiD
    # Qwen 2.1's one-stream block-causal layout precedes legacy qwen_image.
    ("qwen_image21", ("modulation.1.", "transformer_blocks.", "img_in.")),
    ("chroma", ("distilled_guidance_layer.",)),  # ChromaRadiance sniffs as chroma too
    # Hunyuan precedes Flux because it reuses Flux block paths. Its token
    # refiner plus one variant marker excludes unsupported HunyuanVideo 1.0.
    ("hunyuan", ("txt_in.individual_token_refiner.", "cond_type_embedding.")),
    ("hunyuan", ("txt_in.individual_token_refiner.", "byt5_in.")),
    ("hunyuan", ("txt_in.individual_token_refiner.", "time_r_in.")),
    # Order matters: Flux2 has unique modulation keys; Flux requires
    # `vector_in`; LongCat is the remaining shared block layout.
    ("flux2", ("double_stream_modulation_img.",)),
    ("flux", ("double_blocks.", "single_blocks.", "img_attn.qkv.", "vector_in.")),
    ("longcat", ("double_blocks.", "single_blocks.", "img_attn.qkv.")),
    # Lens precedes Qwen because they share common transformer/text/image paths.
    ("lens", ("transformer_blocks.", "attn.norm_added_q.", "img_mlp.w1.")),
    ("qwen_image", ("transformer_blocks.", "txt_norm.", "img_in.")),
    ("ltx", ("adaln_single.", "transformer_blocks.", "patchify_proj.")),
    ("kandinsky5", ("visual_transformer_blocks.", "text_transformer_blocks.")),
    # H3's two projection paths are unique; its refiner and block paths differ
    # from Hunyuan, Omnigen2, and Ernie.
    ("minimax_h3", ("video_patch_proj.", "audio_patch_proj.")),
    ("ernie", ("adaLN_sa_ln.", "mlp.linear_fc2.")),
    # One architecture, two spellings: ComfyUI's CogVideoX flattens the time
    # embedding to `time_embedding_linear_1`, a diffusers export keeps
    # `time_embedding.linear_1`. `patch_embed.text_proj.` is CogVideoX alone, so
    # either time path beside it names the family. The second row only names it:
    # ComfyUI's detector keys on the flattened `blocks.0.norm1.linear.weight` and
    # has no diffusers conversion for this family, so a diffusers-layout file must
    # be converted before any loader can build it (docs/TROUBLESHOOTING.md #102).
    ("cogvideo", ("patch_embed.text_proj.", "time_embedding_linear_1.")),
    ("cogvideo", ("patch_embed.text_proj.", "time_embedding.linear_1.")),
    ("zimage", ("cap_embedder.", "x_pad_token")),  # x_pad_token distinguishes from Lumina 2.0
    ("anima", ("llm_adapter.", "mlp.layer1.")),  # cosmos MiniTrainDIT + anima-only llm_adapter
    # Boogu precedes Omnigen2 because both live in the Omnigen layout.
    ("boogu", ("double_stream_layers.", "img_instruct_attn.")),
    ("omnigen2", ("time_caption_embed.timestep_embedder.", "layers.")),
    # Animate2 uses config metadata; other Wan variants precede plain Wan.
    ("wan_scail", ("patch_embedding_pose.", "text_embedding.")),
    ("wan_dancer", ("patch_embedding_global.", "text_embedding.")),
    ("wan", ("text_embedding.", "self_attn.")),
)

# Z-Image ships three surfaces on one backbone, and the `zimage` row above
# matches all three. The L2P pixel-space checkpoint holds `local_decoder.*`
# (beside `all_x_embedder.16-1.*`) and the DCT PixelSpace one holds `dec_net.*`,
# which ComfyUI reads as pixel space; ComfyUI 3216c62e reads L2P as latent
# Z-Image (docs/TROUBLESHOOTING.md #91). One present and one absent needle
# name L2P apart from both. This reading retires when the L2P forward lands.
_ZIMAGE_L2P_PRESENT = ("local_decoder.",)
_ZIMAGE_L2P_ABSENT = ("dec_net.",)

# Mage Flow reads as the qwen_image row by key. ComfyUI names it apart after
# Lens's block keys, by a 2560-wide `txt_norm` and a 128-row `proj_out` (Qwen
# Image: 3584, 64), first dimensions only, because 4-bit exports halve the second.
_MAGE_FLOW_WIDTHS = {"txt_norm.weight": 2560, "proj_out.weight": 128}

_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.model.",
    "net.",
    "model.",
)
def model_base_touchpoints(adapters: Iterable[object]) -> tuple[str, ...]:
    """Return broad and exact `comfy.model_base` names without a registry cycle."""
    names: dict[str, None] = {}
    for adapter in adapters:
        for field in ("model_base_classes", "exact_model_base_classes"):
            for name in getattr(adapter, field, ()):
                names.setdefault(name, None)
    return tuple(names)


def model_detection_touchpoints(adapters: Iterable[object]) -> tuple[str, ...]:
    """Return Comfy-owned instance paths consulted by adapter selection."""
    paths: dict[str, None] = {}
    for adapter in adapters:
        for path in getattr(adapter, "model_detection_attrs", ()):
            paths.setdefault(path, None)
    return tuple(paths)


class CheckpointSniffError(ValueError):
    """The file is not a readable .safetensors checkpoint."""


@dataclass(frozen=True, slots=True)
class FsdpLaunchQuantProof:
    """FSDP precision result bound to the inode inspected by the parser."""

    quant_kind: str | None
    file_identity: SafetensorsFileIdentity | None


@dataclass(frozen=True, slots=True)
class SignatureNearMiss:
    """One signature row a key list partly satisfies, and what it lacks."""

    family: str
    present: int
    total: int
    missing: tuple[str, ...]


def _read_safetensors_keys_with_identity(
    path: str | Path,
) -> tuple[list[str], dict, SafetensorsFileIdentity]:
    path = Path(path)
    try:
        parsed = read_safetensors_header(path)
    except SafetensorsHeaderError as exc:
        raise CheckpointSniffError(
            f"{exc}. Auto topology "
            "needs a .safetensors checkpoint; pick an explicit topology preset for other formats."
        ) from exc
    tensors = {
        name: descriptor.as_header_dict()
        for name, descriptor in parsed.tensors.items()
    }
    return (
        list(tensors),
        {**parsed.metadata, "__tensors__": tensors},
        parsed.file_identity,
    )


def read_safetensors_keys(path: str | Path) -> tuple[list[str], dict]:
    """Return (tensor keys, __metadata__) from a .safetensors header."""
    keys, header, _identity = _read_safetensors_keys_with_identity(path)
    return keys, header


def _strip_prefix(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _has_segment(keys: list[str], needle: str) -> bool:
    """Match `needle` only at dotted module-path boundaries.

    A trailing dot already supplies the boundary. Dotless leaf names must match
    a complete segment so, for example, `x_pad_token` cannot match a tokenizer.
    """
    if needle.endswith("."):
        return any(k.startswith(needle) or f".{needle}" in k for k in keys)
    return any(
        k == needle or k.startswith(f"{needle}.") or k.endswith(f".{needle}") or f".{needle}." in k
        for k in keys
    )


def detect_family_from_keys(keys: list[str]) -> str:
    stripped = [_strip_prefix(k) for k in keys]
    for family, needles in _SIGNATURES:
        if all(_has_segment(stripped, needle) for needle in needles):
            return family
    return "unknown"


def detect_family_from_header(keys: list[str], header: dict) -> str:
    """Refine a Qwen-Image match while preserving Lens and earlier signature matches."""
    family = detect_family_from_keys(keys)
    tensors = header.get("__tensors__")
    if family != "qwen_image" or not isinstance(tensors, dict):
        return family
    dims = {_strip_prefix(str(key)): (info.get("shape") or [None])[0]
            for key, info in tensors.items() if isinstance(info, dict)}
    mage = all(dims.get(name) == width for name, width in _MAGE_FLOW_WIDTHS.items())
    return "mage_flow" if mage else family


def nearest_signatures(keys: list[str]) -> tuple[SignatureNearMiss, ...]:
    """Rank the rows this key list only partly satisfies, closest first.

    A row appears only when at least one of its key paths is present and at
    least one is absent, so a row that matched outright is never a near miss.
    The order prefers the most paths present, then the fewest missing, then the
    table's own order. This names nothing: two rows can sit at the same
    distance, and the table is written as specific-to-general chains where a
    partial match on an earlier row is the normal outcome of a later one
    matching. Read it only where no row matched at all.
    """
    stripped = [_strip_prefix(k) for k in keys]
    ranked = []
    for family, needles in _SIGNATURES:
        missing = tuple(n for n in needles if not _has_segment(stripped, n))
        present = len(needles) - len(missing)
        if present and missing:
            ranked.append(SignatureNearMiss(family, present, len(needles), missing))
    ranked.sort(key=lambda near: (-near.present, len(near.missing)))
    return tuple(ranked)


def detect_quant_from_header(keys: list[str], header: dict) -> str:
    """Return the checkpoint kind used by auto topology.

    Quantized markers win over full-precision dtypes. Any F16 tensor yields
    `fp16` so the BF16-only FSDP gate cannot grant it; unmarked headers retain
    the `bf16` compatibility default.
    """
    stripped = [_strip_prefix(k) for k in keys]
    if any(k == "scaled_fp8" or k.endswith(".scaled_fp8") for k in stripped):
        return "fp8"
    tensors = header.get("__tensors__", {})
    dtypes = {info.get("dtype", "") for info in tensors.values() if isinstance(info, dict)}
    if any(dt.startswith("F8") or dt == "F4" for dt in dtypes):
        return "fp8"
    if "I8" in dtypes:
        return "int8"
    if "F16" in dtypes:
        return "fp16"
    return "bf16"


def _bytes_by_dtype(header: dict) -> dict[str, int]:
    """Total tensor bytes per safetensors dtype name, from a parsed header."""
    tensors = header.get("__tensors__", {})
    out: dict[str, int] = {}
    for info in tensors.values():
        if not isinstance(info, dict):
            continue
        offsets = info.get("data_offsets") or (0, 0)
        size = int(offsets[1]) - int(offsets[0]) if len(offsets) == 2 else 0
        name = str(info.get("dtype", ""))
        out[name] = out.get(name, 0) + max(size, 0)
    return out


def sniff_fsdp_launch_quant_proof(path: str | Path) -> FsdpLaunchQuantProof:
    """Return strict on-disk precision bound to the parsed inode.

    A uniform BF16 descriptor set answers `bf16`, a uniform F16 set `fp16`,
    and a BF16 core whose only other dtype is F32 inside the fp32-islands
    ceiling (`fsdp_islands.header_core_kind`) still answers `bf16`. Every
    other format answers its own kind, and a file that cannot be read or whose
    identity changed under the parse answers `unknown` with no identity. FSDP
    refuses anything but the admitted kinds before materialization.
    """
    path = Path(path)
    if path.suffix.lower() not in {".safetensors", ".sft"}:
        return FsdpLaunchQuantProof(None, None)
    try:
        keys, header, identity = _read_safetensors_keys_with_identity(path)
    except CheckpointSniffError:
        return FsdpLaunchQuantProof("unknown", None)

    generic_kind = detect_quant_from_header(keys, header)
    if generic_kind not in ("bf16", "fp16"):
        quant_kind = generic_kind
    else:
        from .fsdp_islands import header_core_kind

        quant_kind = header_core_kind(_bytes_by_dtype(header))

    # Recheck the path after parsing to catch replacement or symlink retargeting
    # between descriptor validation and proof publication.
    try:
        current = SafetensorsFileIdentity.from_stat(os.stat(path))
    except OSError:
        return FsdpLaunchQuantProof("unknown", None)
    if current != identity:
        return FsdpLaunchQuantProof("unknown", None)
    return FsdpLaunchQuantProof(quant_kind, identity)


def sniff_fsdp_launch_quant(path: str | Path) -> str | None:
    """Return the strict on-disk precision used by the FSDP launch gate."""
    return sniff_fsdp_launch_quant_proof(path).quant_kind


def _has_scale_beside_quantized_weight(keys: list[str], tensors: dict) -> bool:
    """Detect a comfy-kitchen scale sidecar beside an F8, F4, or I8 weight.

    A plain FP8 checkpoint without ``*.weight_scale``, such as Qwen-Image,
    does not match.
    """
    for key in keys:
        if not key.endswith(".weight_scale"):
            continue
        weight_dtype = tensors.get(key[: -len(".weight_scale")] + ".weight", {}).get("dtype", "")
        if weight_dtype.startswith("F8") or weight_dtype in ("I8", "F4"):
            return True
    return False


def fsdp_lora_admission_property(path: str | Path) -> str | None:
    """Return the header property that excludes FSDP LoRA, or ``None``.

    Comfy-kitchen quantized weights cannot use ``actor.fsdp_lora``. Recognized
    markers are ``scaled_fp8``, any I8 tensor, ``*.comfy_quant``, embedded
    ``_quantization_metadata``, and a scale sidecar beside an F8, F4, or I8 weight.
    Plain FP8 files without these markers remain eligible.

    FP32 islands require a live check: the header cannot tell whether the
    architecture retains or casts them. ``fsdp_lora._file_tensor`` refuses a
    live/file dtype mismatch before baking.
    """
    path = Path(path)
    if path.suffix.lower() not in {".safetensors", ".sft"}:
        return None
    try:
        keys, header = read_safetensors_keys(path)
    except CheckpointSniffError:
        return None
    stripped = [_strip_prefix(k) for k in keys]
    tensors = header.get("__tensors__", {})
    dtypes = set(_bytes_by_dtype(header))
    metadata_keys = {k for k in header if k != "__tensors__"}
    if (
        any(k == "scaled_fp8" or k.endswith(".scaled_fp8") for k in stripped)
        or "I8" in dtypes
        or any(k.endswith(".comfy_quant") for k in stripped)
        or "_quantization_metadata" in metadata_keys
        or _has_scale_beside_quantized_weight(keys, tensors)
    ):
        return "quantized_shards"
    return None


def _without_bf16_default(quant_kind: str, header: dict) -> str:
    """Name an all-FP32 file apart from the header's BF16 compatibility default.

    ``detect_quant_from_header`` answers ``bf16`` for any header carrying no
    FP8, INT8 or F16 marker, an all-FP32 one included, which is why
    ``sniff_fsdp_launch_quant_proof`` re-reads the dtype set before it grants
    anything. This draws the same line the live detector draws
    (``actor.store_detect.BF16_CORE_FP32_AUXILIARY_PROFILE``): a BF16 core
    carrying auxiliary FP32 tensors is a BF16 export and keeps that reading.
    Only a file whose tensors are all F32 or F64 reads FP32; every other header
    keeps the default.
    """
    if quant_kind != "bf16":
        return quant_kind
    tensors = header.get("__tensors__", {})
    dtypes = {info.get("dtype", "") for info in tensors.values() if isinstance(info, dict)}
    if "BF16" in dtypes or not dtypes:
        return quant_kind
    return "fp32" if dtypes <= {"F32", "F64"} else quant_kind


_SniffResult = tuple[str, str, str, tuple[str, ...]]


def _read_sniff(path: str | Path) -> _SniffResult:
    """Read one header into `(family, quant_kind, storage_kind, metadata keys)`.

    Both the cached path and the stat-failure path go through here, so they
    cannot answer different shapes.
    """
    keys, header = read_safetensors_keys(path)
    quant_kind = detect_quant_from_header(keys, header)
    metadata_keys = tuple(sorted(k for k in header if k != "__tensors__"))
    from .wan_animate2 import is_wan_animate2_metadata

    family = ("wan_animate2" if is_wan_animate2_metadata(header)
              else detect_family_from_header(keys, header))
    return (family, quant_kind,
            _without_bf16_default(quant_kind, header), metadata_keys)


@lru_cache(maxsize=64)
def _sniff_checkpoint_by_identity(
    path: str, _mtime_ns: int, _ctime_ns: int, _size: int
) -> _SniffResult:
    return _read_sniff(path)


def _sniff(path: str | Path) -> _SniffResult:
    """`(family, quant_kind, storage_kind, metadata keys)`, cached by path metadata.

    The key includes mtime, ctime, and size because preserved mtimes alone do
    not identify writes or replacements. Exceptions are not cached, so a failed
    read is retried.
    """
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return _read_sniff(path)
    return _sniff_checkpoint_by_identity(
        str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
    )


@lru_cache(maxsize=16)
def _nearest_signatures_by_identity(
    path: str, _mtime_ns: int, _ctime_ns: int, _size: int
) -> tuple[SignatureNearMiss, ...]:
    return nearest_signatures(read_safetensors_keys(path)[0])


def sniff_nearest_signatures(path: str | Path) -> tuple[SignatureNearMiss, ...]:
    """Rank the rows a checkpoint only partly satisfies, cached by path metadata."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return nearest_signatures(read_safetensors_keys(path)[0])
    return _nearest_signatures_by_identity(
        str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
    )


def is_zimage_l2p_from_keys(keys: list[str]) -> bool:
    """Name the Z-Image L2P layout from the header keys alone.

    Pure and total: a family word this table already answers, plus one decoder
    path present and the other absent. It reads the raw key list on purpose,
    never a forced family, because it guards the host rather than the choice of
    adapter.
    """
    if detect_family_from_keys(keys) != "zimage":
        return False
    stripped = [_strip_prefix(k) for k in keys]
    if not all(_has_segment(stripped, needle) for needle in _ZIMAGE_L2P_PRESENT):
        return False
    return not any(_has_segment(stripped, needle) for needle in _ZIMAGE_L2P_ABSENT)


@lru_cache(maxsize=16)
def _zimage_l2p_by_identity(
    path: str, _mtime_ns: int, _ctime_ns: int, _size: int
) -> bool:
    keys = read_safetensors_keys(path)[0]
    answer = is_zimage_l2p_from_keys(keys)
    if answer:
        # The same boundary rule the predicate above used. A plain
        # ``startswith`` would drop a nested spelling and log an empty list,
        # which is the one line the hardware acceptance reads.
        matched = sorted({
            key for key in (_strip_prefix(k) for k in keys)
            if any(key.startswith(needle) or f".{needle}" in key
                   for needle in _ZIMAGE_L2P_PRESENT)
        })
        # Once per file identity, because the cache holds the answer: the
        # acceptance record needs the spelling this run matched.
        log.info("upstream gate: %s carries the Z-Image L2P decoder (%s) and no "
                 "%s head", path, ", ".join(matched[:4]),
                 _ZIMAGE_L2P_ABSENT[0].rstrip("."))
    return answer


def sniff_zimage_l2p(path: str | Path) -> bool:
    """Return false for unreadable or non-L2P headers."""
    try:
        resolved = Path(path)
        stat = resolved.stat()
        return _zimage_l2p_by_identity(
            str(resolved), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    except Exception as exc:
        log.debug("upstream gate stood down for %s: %r", path, exc)
        return False


def sniff_metadata_keys(path: str | Path) -> tuple[str, ...]:
    """Return sorted header metadata names."""
    return _sniff(path)[3]


def sniff_checkpoint(path: str | Path) -> tuple[str, str]:
    """Return `(family, quant_kind)` as the header reads it, cached by path."""
    family, quant_kind, _storage, _metadata_keys = _sniff(path)
    return family, quant_kind


def sniff_checkpoint_for_topology(path: str | Path) -> tuple[str, str]:
    """Return the family and storage kind used by auto topology."""
    family, _quant_kind, storage_kind, _metadata_keys = _sniff(path)
    return family, storage_kind
