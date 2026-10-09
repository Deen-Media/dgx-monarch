"""FP32 islands and auxiliary modules under FSDP capacity mode.

An "island" is a full-precision parameter inside an otherwise bf16 denoising
core (an fp16 core is admitted uniform only): LTX's scale-shift tables,
MiniMax H3's audio patch and head projections, the modulation projection some
Krea2 exports keep in fp32, and stock Comfy Wan's input patch convolution.
FSDP2 could shard them (the uniform-dtype assertion covers trainable
parameters only, and every parameter here is frozen), but replicating them is
simpler: they are passed to ``fully_shard`` as ``ignored_params``, stay plain
tensors on the load device, and the forward math is untouched. The ceiling
below bounds the cost: a rank holds 1/world of the core plus at most this
share in islands.

The same mechanism keeps auxiliary modules that run outside the denoise path
out of every collective: their parameters are ignored rather than wrapped, so
they can never deadlock an all-gather, and a weight they read is a plain
tensor when ComfyUI runs them ahead of the first pre-forward hook.
"""
from __future__ import annotations

from typing import Any

FP32_ISLANDS_PROFILE = "bf16_core_fp32_islands_v1"
ALL_FP16_PROFILE = "all_fp16"
# Header measurements put LTX 2.5 islands at 0.045% of bytes, MiniMax H3 at 0.1%,
# and Krea2 RAW at 4.9% (including an 864 MiB modulation projection). A 5% cap
# admits these layouts while bounding the replicated cost.
FP32_ISLANDS_MAX_FRACTION = 0.05
# Absolute cap for cases without a total byte count, such as ledger admission.
# 4 GiB covers 5% of the measured 62 GiB checkpoint class (about 3.1 GiB).
FP32_ISLANDS_MAX_BYTES = 4 * 1024 * 1024 * 1024
# Safetensors header dtype names for the two admitted cores and the islands.
CORE_HEADER_DTYPES = {"BF16": "bf16", "F16": "fp16"}
ISLAND_HEADER_DTYPE = "F32"
# Modules a family runs outside its diffusion model's forward. ComfyUI calls
# ``BaseModel.extra_conds`` once per sampling, before the denoise loop, and
# three supported families reach into the diffusion model from there. No FSDP2
# pre-forward hook has gathered anything at that point, so a sharded weight is
# a DTensor meeting a plain text state, which fails at the first addmm. Key the
# table by the ComfyUI model_base class whose extra_conds makes the call; the
# canary checks it against the installed ComfyUI.
OUTSIDE_FORWARD_MODULE_ATTRS: dict[str, tuple[str, ...]] = {
    # Anima.extra_conds -> preprocess_text_embeds -> llm_adapter.
    "Anima": ("llm_adapter",),
    # MiniMaxH3.extra_conds -> preprocess_text_embeds -> condition_proj, then
    # token_refiner. 19 tensors, 1.487 GiB on the 61.7 GiB bf16 file. The
    # packed forward runs the same two on the shorter path, so replicating
    # them buys the extra_conds call and costs nothing else.
    "MiniMaxH3": ("condition_proj", "token_refiner"),
    # LTXAV.extra_conds -> preprocess_text_embeds -> both caption projections
    # where the export builds them, then both connectors. 258 tensors, 3.755
    # GiB on the LTX 2.5 bf16 transformer, which carries the connectors and no
    # caption projection. The method returns early only when the context
    # already arrives at the processed width; the text encoder marks its output
    # unprocessed on both non-compat routes.
    "LTXAV": ("caption_projection", "audio_caption_projection",
              "video_embeddings_connector", "audio_embeddings_connector"),
}
# One flat census for the wrap, built by attribute name off the live diffusion
# model. A name the family lacks reads as None below and costs nothing.
AUXILIARY_MODULE_ATTRS = tuple(dict.fromkeys(
    attr for attrs in OUTSIDE_FORWARD_MODULE_ATTRS.values() for attr in attrs))
# The names above are not all unique to the family that declares them, and a
# flat census alone would replicate a module another family runs inside its
# forward. LTXV builds `caption_projection` in the LTX base both it and LTXAV
# share and calls it from `_prepare_context`, inside the forward, where a
# sharded weight is gathered and a replicated one only costs a rank the whole
# module. LTXV never reaches it from `extra_conds`, so it is subtracted for
# that family: the wrap reads the comfy class off `type(patcher.model)`, and
# the canary seam holds this table the way it holds the one above. A family
# the table does not name keeps the whole flat census, the safe answer for a
# class this repository has not read.
FORWARD_PATH_NAME_OWNERS: dict[str, tuple[str, ...]] = {
    "LTXV": ("caption_projection",),
}


def islands_within_ceiling(fp32_bytes: int, total_bytes: int) -> bool:
    """Whether ``fp32_bytes`` of islands fit the share the profile admits."""
    return (
        0 < fp32_bytes <= FP32_ISLANDS_MAX_BYTES
        and total_bytes > 0
        and fp32_bytes / total_bytes <= FP32_ISLANDS_MAX_FRACTION
    )


def classify_fp32_islands(
    fp32_parameters: dict[str, Any],
    total_parameter_bytes: int,
) -> tuple[str, int, int] | None:
    """Adapter-owned evidence for a bf16 core carrying fp32 islands, or None."""
    if not fp32_parameters:
        return None
    fp32_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in fp32_parameters.values()
    )
    if not islands_within_ceiling(fp32_bytes, total_parameter_bytes):
        return None
    return FP32_ISLANDS_PROFILE, len(fp32_parameters), fp32_bytes


def header_core_kind(bytes_by_dtype: dict[str, int]) -> str:
    """Strict on-disk kind from a safetensors header's bytes per dtype.

    ``bf16`` or ``fp16`` for a uniform core, and ``bf16`` for a bf16 core whose
    only other dtype is fp32 inside the islands ceiling. ``fp32`` for full
    precision no profile admits: fp32 (alone or with F64) and no core, fp32
    beside both cores, or one core whose only other dtype is fp32 when that
    core is fp16 or the fp32 share passes the ceiling. ``unknown`` for every
    other mix.
    """
    dtypes = {name for name, size in bytes_by_dtype.items() if size or name}
    cores = [name for name in dtypes if name in CORE_HEADER_DTYPES]
    if len(cores) != 1:
        if ISLAND_HEADER_DTYPE in dtypes and dtypes <= {ISLAND_HEADER_DTYPE, "F64"}:
            return "fp32"
        return "fp32" if cores and ISLAND_HEADER_DTYPE in dtypes else "unknown"
    core = cores[0]
    others = dtypes - {core}
    if not others:
        return CORE_HEADER_DTYPES[core]
    if others != {ISLAND_HEADER_DTYPE}:
        return "unknown"
    if core != "BF16":
        # fp16 cores are admitted uniform only; fp16 plus fp32 is a mix the
        # live detector reads as mixed fp16, which no profile admits.
        return "fp32"
    total = sum(bytes_by_dtype.values())
    if islands_within_ceiling(bytes_by_dtype.get(ISLAND_HEADER_DTYPE, 0), total):
        return "bf16"
    return "fp32"


def island_parameters(diffusion_model) -> set:
    """Every fp32 parameter of the model: replicated, never sharded."""
    import torch

    return {
        parameter
        for parameter in diffusion_model.parameters()
        if parameter.dtype == torch.float32
    }


def outside_forward_module_attrs(model=None) -> tuple[str, ...]:
    """The census names to read off a diffusion model, for this comfy family.

    ``model`` is the comfy ``model_base`` instance the wrap holds. Its class
    chain names the family, which is how a name another family owns on its own
    forward path is subtracted. A name any class in that chain declares as an
    outside-forward module is never subtracted, so a family that inherits from
    one of the owners above and does reach the module keeps it replicated.
    Without a model, or for a class neither table names, the whole flat census
    applies.
    """
    if model is None:
        return AUXILIARY_MODULE_ATTRS
    owned: set = set()
    declared: set = set()
    for klass in type(model).__mro__:
        owned.update(FORWARD_PATH_NAME_OWNERS.get(klass.__name__, ()))
        declared.update(OUTSIDE_FORWARD_MODULE_ATTRS.get(klass.__name__, ()))
    subtract = owned - declared
    if not subtract:
        return AUXILIARY_MODULE_ATTRS
    return tuple(attr for attr in AUXILIARY_MODULE_ATTRS if attr not in subtract)


def auxiliary_module_parameters(diffusion_model, model=None) -> set:
    """Parameters of modules that run outside the denoise path."""
    ignored: set = set()
    for attr in outside_forward_module_attrs(model):
        module = getattr(diffusion_model, attr, None)
        parameters = getattr(module, "parameters", None)
        if module is not None and callable(parameters):
            ignored.update(parameters())
    return ignored
