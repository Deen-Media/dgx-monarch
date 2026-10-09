"""Executable inventory of ComfyUI Python compatibility touchpoints.

tests/test_adapter_imports_are_manifested.py AST-walks the adapters package for
comfy imports and fails when one is missing here. Keep entries grouped by the
owning Comfy module and name the full dotted path: the daily ComfyUI-master
canary names the first incompatible surface by that path.

This covers importable functions, classes, constants, generic ModelPatcher
state, and adapter-selection class names. For model-family forwards it checks
signatures only (comfy_forward_contracts.py); their behavior is covered by the
per-family adapter tests, the stock-equivalence tests the canary runs on tiny
real comfy models, and the hardware identity gates. Checkpoint family detection
reads safetensors headers in ``adapters.detect`` and imports nothing from comfy.

Comfy names that only the canary scripts and benchmark/ use are not listed
here; each canary exercises its own. Browser extension and HTTP behavior are
covered by the node and web snapshot tests and the hardware workflows.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable, Iterable
from typing import cast

from .comfy_forward_contracts import (
    ADDITIONAL_REBOUND_MODEL_BASE_TOUCHPOINTS,
    BROAD_ADAPTER_SUBCLASS_CONTRACTS,
    FAMILY_FORWARD_TOUCHPOINTS,
    REBOUND_METHOD_CONTRACTS,
)
from .comfy_rebound_validation import (
    broad_adapter_subclass_incompatibility,
    rebound_signature_incompatibility,
)
from .comfy_touchpoint import Touchpoint, _attr, _call

__all__ = ["TOUCHPOINTS", "Touchpoint", "assert_comfy_surface"]


# Hard runtime dependencies. Reads with a default value are listed in
# OPTIONAL_TOUCHPOINTS below.
TOUCHPOINTS: tuple[Touchpoint, ...] = (
    # Core weight loading, progress, and tensor helpers.
    _call("comfy.utils", "load_torch_file", "safe_load", "device", "return_metadata", positional=("ckpt",)),
    _call(
        "comfy.utils", "convert_old_quants", "model_prefix", "metadata",
        positional=("state_dict",),
    ),
    _call("comfy.utils", "ProgressBar", positional=("total",)),
    _call("comfy.utils", "ProgressBar.update_absolute", positional=("self", "value", "total")),
    _call("comfy.utils", "repeat_to_batch_size", positional=("tensor", "batch_size")),
    _call("comfy.utils", "unpack_latents", positional=("combined_latent", "latent_shapes")),
    _call("comfy.utils", "string_to_seed", positional=("data",)),
    _call("comfy.utils", "copy_to_param", positional=("obj", "attr", "value")),
    _call("comfy.model_base", "BaseModel.load_model_weights", "unet_prefix", "assign", positional=("self", "sd")),
    _call("comfy.model_base", "BaseModel.process_latent_out", positional=("self", "latent")),
    # Memory policy. The globals are hard requirements even where their first
    # read has a fallback: a write to a renamed knob fails silently and leaves
    # workers off the configured memory settings.
    _call("comfy.model_management", "get_free_memory", "dev", "torch_free_too"),
    _call("comfy.model_management", "get_torch_device"),
    _call("comfy.model_management", "load_models_gpu", "force_full_load", positional=("models",)),
    _call("comfy.model_management", "unload_all_models"),
    _call("comfy.model_management", "soft_empty_cache"),
    _call("comfy.model_management", "cast_to_device", "copy", positional=("tensor", "device", "dtype")),
    _call("comfy.model_management", "lora_compute_dtype", positional=("device",)),
    _call("comfy.model_management", "cast_to", "dtype", "device", positional=("weight",)),
    _call("comfy.model_management", "intermediate_device"),
    _call("comfy.model_management", "processing_interrupted"),
    _call("comfy.model_management", "throw_exception_if_processing_interrupted"),
    _attr("comfy.model_management", "EXTRA_RESERVED_VRAM"),
    _attr("comfy.model_management", "NUM_STREAMS"),
    _attr("comfy.model_management", "DISABLE_SMART_MEMORY"),
    _attr("comfy.model_management", "MAX_PINNED_MEMORY"),
    # Model and LoRA loaders, ModelPatcher, bake, and quant entry points.
    _call("comfy.sd", "load_diffusion_model", "model_options", positional=("unet_path",)),
    _call("comfy.sd", "load_diffusion_model_state_dict", "model_options", positional=("sd",)),
    _call("comfy.sd", "load_lora_for_models", positional=("model", "clip", "lora", "strength_model", "strength_clip")),
    _call("comfy.sd", "CLIP.tokenize", positional=("self", "text")),
    _call("comfy.sd", "CLIP.encode_from_tokens_scheduled", positional=("self", "tokens")),
    _call("comfy.model_patcher", "ModelPatcher", positional=("model", "load_device", "offload_device")),
    _call("comfy.model_patcher", "ModelPatcher.clone", positional=("self",)),
    _call("comfy.model_patcher", "ModelPatcher.add_object_patch", positional=("self", "name", "obj")),
    _call("comfy.model_patcher", "ModelPatcher.cleanup", positional=("self",)),
    _call("comfy.model_patcher", "ModelPatcher.get_model_object", positional=("self", "name")),
    _call("comfy.model_patcher", "ModelPatcher.patch_weight_to_device", "device_to", "return_weight", positional=("self", "key")),
    # The full-load call the resident ledger makes at build time, the same one
    # comfy's own manager makes under force_full_load (actor/resident_ledger.py).
    _call("comfy.model_patcher", "ModelPatcher.partially_load", positional=("self", "device_to", "extra_memory")),
    _call("comfy.model_patcher", "get_key_weight", positional=("model", "key")),
    _call("comfy.lora", "calculate_weight", positional=("patches", "weight", "key")),
    _call("comfy.float", "stochastic_rounding", "seed", positional=("value", "dtype")),
    _attr("comfy.quant_ops", "QUANT_ALGOS"),
    _call("comfy.quant_ops", "get_layout_class", positional=("name",)),
    _call("comfy.quant_ops", "QuantizedTensor", positional=("qdata", "layout_cls", "params")),
    _call("comfy_kitchen.tensor", "QuantizedTensor", positional=("qdata", "layout_cls", "params")),
    _call("comfy.quant_ops", "QuantizedTensor.dequantize", positional=("self",)),
    # The nvfp4 activation fallback the shared-scale hook replaces
    # (adapters/quant_activation_scale.py): the keyword it hands the layout, and
    # the two constants the scale is divided by. That the fallback is a
    # whole-tensor statistic is seam 26 in tests/canary/comfy_seam_contracts.py.
    _call("comfy.quant_ops", "TensorCoreNVFP4Layout.quantize", "scale", positional=("tensor",)),
    _attr("comfy_kitchen.float_utils", "F8_E4M3_MAX"),
    _attr("comfy_kitchen.float_utils", "F4_E2M1_MAX"),
    _call("comfy.quant_ops", "ck.apply_rope_split_half", positional=("xq", "xk", "freqs_cis")),
    _call("comfy.nested_tensor", "NestedTensor", positional=("tensors",)),
    _call("comfy.nested_tensor", "NestedTensor.unbind", positional=("self",)),
    # KSampler, custom sampler, and guider surfaces used by nodes and workers.
    _attr("comfy.samplers", "KSampler.SAMPLERS"),
    _attr("comfy.samplers", "KSampler.SCHEDULERS"),
    _call("comfy.samplers", "calculate_sigmas", positional=("model_sampling", "scheduler_name", "steps")),
    _call("comfy.samplers", "CFGGuider", positional=("model_patcher",)),
    _call("comfy.samplers", "CFGGuider.inner_set_conds", positional=("self", "conds")),
    _call("comfy.samplers", "CFGGuider.set_conds", positional=("self", "positive", "negative")),
    _call("comfy.samplers", "CFGGuider.set_cfg", positional=("self", "cfg")),
    _call("comfy.samplers", "CFGGuider.sample", "denoise_mask", "callback", "disable_pbar", "seed", positional=("self", "noise", "latent_image", "sampler", "sigmas")),
    _call("comfy.samplers", "calc_cond_batch", positional=("model", "conds", "x_in", "timestep", "model_options")),
    _call("comfy.sample", "fix_empty_latent_channels", positional=("model", "latent_image", "downscale_ratio_spacial", "downscale_ratio_temporal")),
    _call("comfy.sample", "prepare_noise", positional=("latent_image", "seed", "noise_inds")),
    _call("comfy.sample", "sample", "denoise", "disable_noise", "start_step", "last_step", "force_full_denoise", "noise_mask", "callback", "disable_pbar", "seed", positional=("model", "noise", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image")),
    _call("comfy_extras.nodes_custom_sampler", "Guider_DualModel", positional=("model_patcher", "uncond_model_patcher")),
    _call("comfy_extras.nodes_custom_sampler", "Guider_DualModel.set_conds", positional=("self", "positive", "negative")),
    _call("comfy_extras.nodes_custom_sampler", "Guider_DualModel.set_cfg", positional=("self", "cfg")),
    _call("comfy_extras.nodes_custom_sampler", "Guider_DualModel.sample", "denoise_mask", "callback", "disable_pbar", "seed", positional=("self", "noise", "latent_image", "sampler", "sigmas")),
    _call("comfy_extras.nodes_custom_sampler", "Noise_RandomNoise", positional=("seed",)),
    _call("comfy_extras.nodes_custom_sampler", "Noise_RandomNoise.generate_noise", positional=("self", "input_latent")),
    _call("comfy_extras.nodes_custom_sampler", "Noise_EmptyNoise"),
    _call("comfy_extras.nodes_custom_sampler", "Noise_EmptyNoise.generate_noise", positional=("self", "input_latent")),
    _call("comfy_extras.nodes_model_advanced", "ModelSamplingSD3"),
    _call("comfy_extras.nodes_model_advanced", "ModelSamplingSD3.patch", positional=("self", "model", "shift")),
    _attr("comfy.patcher_extension", "WrappersMP.DIFFUSION_MODEL"),
    _attr("comfy.patcher_extension", "WrappersMP.APPLY_MODEL"),
    _attr("comfy.patcher_extension", "WrappersMP.CALC_COND_BATCH"),
    _attr("comfy.patcher_extension", "WrappersMP.PREPARE_SAMPLING"),
    _call("comfy.sampler_helpers", "prepare_sampling", "model_options", "force_full_load", positional=("model", "noise_shape", "conds")),
    _call("comfy.patcher_extension", "get_wrappers_with_key", "is_model_options", positional=("wrapper_type", "key", "transformer_options")),
    _call("comfy.patcher_extension", "add_wrapper_with_key", "is_model_options", positional=("wrapper_type", "key", "wrapper", "transformer_options")),
    _call("comfy.text_encoders.llama", "precompute_freqs_cis", "rope_dims", "device", "interleaved_mrope", positional=("head_dim", "position_ids", "theta")),
    # Comfy filesystem and server registration/event contracts.
    _call("folder_paths", "get_filename_list", positional=("folder_name",)),
    _call("folder_paths", "get_folder_paths", positional=("folder_name",)),
    _call("folder_paths", "get_full_path", positional=("folder_name", "filename")),
    _call("folder_paths", "get_output_directory"),
    _attr("folder_paths", "base_path"),
    _attr("folder_paths", "__file__"),
    _attr("server", "PromptServer.instance"),
    _attr("server", "PromptServer.instance.routes"),
    _call("server", "PromptServer.instance.routes.get", positional=("path",)),
    _call("server", "PromptServer.instance.routes.post", positional=("path",)),
    _call("server", "PromptServer.instance.send_sync", positional=("event", "data")),
    _call("server", "PromptServer.instance.add_on_prompt_handler", positional=("handler",)),
    # Helper leaves imported by rewritten model-family forwards, the stock
    # forwards those adapters rebind, and their stock callers and callees
    # (comfy_forward_contracts.py).
    *FAMILY_FORWARD_TOUCHPOINTS,
)


MODEL_PATCHER_INSTANCE_ATTRIBUTES: tuple[str, ...] = (
    "model",
    "model_options",
    "patches",
    "backup",
    "backup_buffers",
    "offload_device",
)
GUIDER_INSTANCE_ATTRIBUTES: tuple[str, ...] = ("model_patcher",)

QUANT_ALGORITHM_FORMATS: tuple[str, ...] = (
    "float8_e4m3fn",
    "float8_e5m2",
    "mxfp8",
    "nvfp4",
    "int8_tensorwise",
)
QUANT_ALGORITHM_FIELDS: tuple[str, ...] = ("storage_t", "comfy_tensor_layout")
QUANT_LAYOUT_PARAMETERS: dict[str, tuple[str, ...]] = dict.fromkeys(
    QUANT_ALGORITHM_FORMATS, ("scale", "orig_dtype", "orig_shape")
)
QUANT_LAYOUT_PARAMETERS["nvfp4"] += ("block_scale",)
QUANT_LAYOUT_PARAMETERS["int8_tensorwise"] += ("convrot", "convrot_groupsize")
QUANTIZED_TENSOR_INSTANCE_ATTRIBUTES: tuple[str, ...] = ("_qdata", "_layout_cls", "_params")

# Checkpoint-free structural anchors for every duck-typed adapter detector.
# assert_comfy_surface fails the real canary on a new detector path until its
# upstream anchor is added here.
MODEL_DETECTION_ANCHORS: dict[str, tuple[str, str, str, str]] = {
    "diffusion_model": (
        "comfy.model_base.BaseModel.load_model_weights", "comfy.model_base", "BaseModel", "diffusion_model",
    ),
    "diffusion_model.pad_tokens_multiple": (
        "comfy.ldm.lumina.model.NextDiT", "comfy.ldm.lumina.model", "NextDiT", "pad_tokens_multiple",
    ),
}


# Real reads whose callers fall back safely when the name is absent. Listing
# them keeps them visible without making their absence an outage.
OPTIONAL_TOUCHPOINTS: tuple[str, ...] = (
    # The loader-site footprint preflight (nodes/loader_graph.py) reads this to
    # tell a driver stack comfy already holds from one it is about to build.
    # Absence means "nothing loaded", which over-charges rather than crashing.
    "comfy.model_management.current_loaded_models",
    "comfy.memory_management.aimdo_enabled",
    "comfy.utils.DISABLE_MMAP",
    "comfy.model_patcher.ModelPatcher.model.model_loaded_weight_memory",
    "comfy.model_patcher.ModelPatcher.model.model_lowvram",
    "comfy.model_patcher.ModelPatcher.load_device",
    # The per-module activation scale comfy's quantized Linear reads with a
    # default of None. No shipped nvfp4 artifact carries one; the shared-scale
    # hook writes it for one forward at a time (adapters/quant_activation_scale.py),
    # so its absence is normal.
    "comfy.ops.mixed_precision_ops.Linear.input_scale",
)


_REBOUND_CONTRACTS_BY_PATH = {
    contract.path: contract for contract in REBOUND_METHOD_CONTRACTS
}

def _missing(path: str) -> RuntimeError:
    return RuntimeError(f"Comfy touchpoint missing: {path}")


def _incompatible(path: str, detail: str) -> RuntimeError:
    return RuntimeError(f"Comfy touchpoint incompatible: {path} ({detail})")


def _resolve(root: object, attribute: str, full_path: str) -> object:
    value = root
    for part in attribute.split("."):
        try:
            value = getattr(value, part)
        except AttributeError as exc:
            raise _missing(full_path) from exc
    return value


def _assert_touchpoint(value: object, touchpoint: Touchpoint) -> None:
    if touchpoint.callable and not callable(value):
        raise _incompatible(touchpoint.path, "not callable")
    if not touchpoint.callable:
        return
    try:
        signature = inspect.signature(cast(Callable[..., object], value))
    except (TypeError, ValueError) as exc:
        raise _incompatible(touchpoint.path, "signature unavailable") from exc
    contract = _REBOUND_CONTRACTS_BY_PATH.get(touchpoint.path)
    if contract is not None:
        detail = rebound_signature_incompatibility(signature, contract)
        if detail is not None:
            raise _incompatible(contract.path, detail)
        # The exact check covers the prefix and bind checks below, and the
        # prefix check would re-fail an admitted optional_infix shape: the infix
        # moves a later name (transformer_options) out of its slot in
        # touchpoint.positional.
        return
    for parameter in touchpoint.parameters:
        declared = signature.parameters.get(parameter)
        if declared is None:
            raise _incompatible(touchpoint.path, f"signature missing parameter {parameter!r}")
        if declared.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise _incompatible(touchpoint.path, f"parameter {parameter!r} is positional-only")
    positional_names = tuple(
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )[: len(touchpoint.positional)]
    if positional_names != touchpoint.positional:
        raise _incompatible(touchpoint.path, "positional parameter order changed")
    args = [object()] * len(touchpoint.positional)
    kwargs = {parameter: object() for parameter in touchpoint.parameters}
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise _incompatible(touchpoint.path, "production call shape rejected") from exc


def _default_model_patcher(import_module: Callable[[str], object]) -> object:
    import torch

    module = import_module("comfy.model_patcher")
    model_patcher = _resolve(module, "ModelPatcher", "comfy.model_patcher.ModelPatcher")
    cpu = torch.device("cpu")
    factory = cast(Callable[..., object], model_patcher)
    try:
        return factory(torch.nn.Linear(1, 1), cpu, cpu)
    except Exception as exc:
        raise _incompatible("comfy.model_patcher.ModelPatcher", "CPU construction failed") from exc


def _assert_quant_algorithms(module: object) -> int:
    algorithms = _resolve(module, "QUANT_ALGOS", "comfy.quant_ops.QUANT_ALGOS")
    get_layout = cast(
        Callable[[str], object],
        _resolve(module, "get_layout_class", "comfy.quant_ops.get_layout_class"),
    )
    checked = 0
    for format_name in QUANT_ALGORITHM_FORMATS:
        format_path = f"comfy.quant_ops.QUANT_ALGOS[{format_name}]"
        try:
            config = algorithms[format_name]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise _missing(format_path) from exc
        for field in QUANT_ALGORITHM_FIELDS:
            path = f"{format_path}.{field}"
            try:
                config[field]
            except (KeyError, TypeError) as exc:
                raise _missing(path) from exc
            checked += 1
        layout_name = config["comfy_tensor_layout"]
        params_path = f"comfy.quant_ops.get_layout_class({layout_name!r}).Params"
        try:
            layout = get_layout(layout_name)
        except Exception as exc:
            raise _incompatible(params_path, "layout lookup failed") from exc
        params = _resolve(layout, "Params", params_path)
        if not callable(params):
            raise _incompatible(params_path, "not callable")
        try:
            inspect.signature(params).bind(
                **{name: object() for name in QUANT_LAYOUT_PARAMETERS[format_name]}
            )
        except (TypeError, ValueError) as exc:
            raise _incompatible(params_path, "production call shape rejected") from exc
        checked += 1
    return checked


def _default_quantized_tensor(import_module: Callable[[str], object]) -> object:
    import torch

    module = import_module("comfy.quant_ops")
    algorithms = _resolve(module, "QUANT_ALGOS", "comfy.quant_ops.QUANT_ALGOS")
    config = algorithms["float8_e4m3fn"]  # type: ignore[index]
    layout_name = config["comfy_tensor_layout"]
    get_layout = cast(
        Callable[[str], object],
        _resolve(module, "get_layout_class", "comfy.quant_ops.get_layout_class"),
    )
    layout = get_layout(layout_name)
    params_factory = cast(
        Callable[..., object],
        _resolve(layout, "Params", f"comfy.quant_ops.get_layout_class({layout_name!r}).Params"),
    )
    factory = cast(
        Callable[..., object],
        _resolve(module, "QuantizedTensor", "comfy.quant_ops.QuantizedTensor"),
    )
    try:
        params = params_factory(
            scale=torch.ones((), dtype=torch.float32),
            orig_dtype=torch.float32,
            orig_shape=(1,),
        )
        return factory(torch.zeros(1, dtype=config["storage_t"]), layout_name, params)
    except Exception as exc:
        raise _incompatible("comfy.quant_ops.QuantizedTensor", "CPU construction failed") from exc


def _default_guider(import_module: Callable[[str], object], model_patcher: object) -> object:
    module = import_module("comfy.samplers")
    factory = cast(
        Callable[..., object],
        _resolve(module, "CFGGuider", "comfy.samplers.CFGGuider"),
    )
    try:
        return factory(model_patcher)
    except Exception as exc:
        raise _incompatible("comfy.samplers.CFGGuider", "CPU construction failed") from exc


def assert_model_detection_surface(base_model: object, adapters: Iterable[object]) -> int:
    """Assert duck-typed Comfy fields used while selecting an adapter.

    Real model instances need a checkpoint, so the master canary anchors
    Z-Image's extra field to the ``NextDiT`` constructor source while family
    tests pass lightweight model instances through this helper.
    """
    from .adapters.detect import model_detection_touchpoints

    checked = 0
    for attribute in model_detection_touchpoints(adapters):
        path = f"comfy.model_base.BaseModel.{attribute}"
        _resolve(base_model, attribute, path)
        checked += 1
    return checked


def _assert_model_detection_sources(modules: dict[str, object]) -> int:
    for detection_path, (_, module_name, owner_name, attribute) in MODEL_DETECTION_ANCHORS.items():
        path = f"comfy.model_base.BaseModel.{detection_path}"
        owner = _resolve(modules[module_name], owner_name, f"{module_name}.{owner_name}")
        try:
            source = inspect.getsource(inspect.getattr_static(owner, "__init__"))
        except (OSError, TypeError) as exc:
            raise _incompatible(path, "constructor source unavailable") from exc
        if re.search(rf"\bself\s*\.\s*{re.escape(attribute)}\s*(?::[^=\n]+)?=", source) is None:
            raise _missing(path)
    return len(MODEL_DETECTION_ANCHORS)


def assert_comfy_surface(
    *,
    import_module: Callable[[str], object] = importlib.import_module,
    adapters: Iterable[object] | None = None,
    model_patcher: object | None = None,
    model_detection_object: object | None = None,
    quantized_tensor: object | None = None,
    guider: object | None = None,
    check_detection_sources: bool = True,
) -> int:
    """Assert every hard touchpoint and return the number checked.

    The keyword arguments are injectable so the unit suite can prove that
    deleting each declared fake attribute fails with its exact path. The real
    canary uses the defaults (tiny CPU patcher, quantized-tensor and guider
    probes) and a PromptServer instance that registers routes but never starts.
    """
    modules: dict[str, object] = {}
    checked = 0
    for touchpoint in TOUCHPOINTS:
        module = modules.get(touchpoint.module)
        if module is None:
            try:
                module = import_module(touchpoint.module)
            except (ImportError, ModuleNotFoundError) as exc:
                raise _missing(touchpoint.path) from exc
            except Exception as exc:
                raise _incompatible(touchpoint.path, "module import failed") from exc
            modules[touchpoint.module] = module
        value = _resolve(module, touchpoint.attribute, touchpoint.path)
        _assert_touchpoint(value, touchpoint)
        checked += 1

    checked += _assert_quant_algorithms(modules["comfy.quant_ops"])

    if adapters is None:
        from .adapters import ADAPTERS

        adapters = ADAPTERS
    adapters = tuple(adapters)
    from .adapters.detect import model_base_touchpoints, model_detection_touchpoints

    detection_paths = set(model_detection_touchpoints(adapters))
    if detection_paths != set(MODEL_DETECTION_ANCHORS):
        unanchored = sorted(detection_paths - set(MODEL_DETECTION_ANCHORS))
        stale = sorted(set(MODEL_DETECTION_ANCHORS) - detection_paths)
        detail = unanchored[0] if unanchored else stale[0]
        raise RuntimeError(f"Comfy touchpoint manifest missing model-detection anchor: {detail}")
    declared_paths = {touchpoint.path for touchpoint in TOUCHPOINTS}
    for detection_path, (anchor, *_source) in MODEL_DETECTION_ANCHORS.items():
        if anchor not in declared_paths:
            raise RuntimeError(
                f"Comfy touchpoint manifest missing model-detection anchor: {detection_path} -> {anchor}"
            )
    if check_detection_sources:
        checked += _assert_model_detection_sources(modules)

    model_base = modules.get("comfy.model_base")
    if model_base is None:
        model_base = import_module("comfy.model_base")
        modules["comfy.model_base"] = model_base
    model_names = tuple(dict.fromkeys(
        (*model_base_touchpoints(adapters), *ADDITIONAL_REBOUND_MODEL_BASE_TOUCHPOINTS)
    ))
    if not model_names:
        raise RuntimeError("Comfy touchpoint manifest is vacuous: no adapter model classes")
    for name in model_names:
        path = f"comfy.model_base.{name}"
        value = _resolve(model_base, name, path)
        if not inspect.isclass(value):
            raise _incompatible(path, "not a class")
        checked += 1
    adapter_families = {str(getattr(adapter, "family", "")) for adapter in adapters}
    subclass_issue = broad_adapter_subclass_incompatibility(model_base, adapter_families)
    if subclass_issue is not None:
        raise _incompatible(*subclass_issue)
    checked += sum(
        contract.family in adapter_families
        for contract in BROAD_ADAPTER_SUBCLASS_CONTRACTS
    )

    patcher = model_patcher if model_patcher is not None else _default_model_patcher(import_module)
    for attribute in MODEL_PATCHER_INSTANCE_ATTRIBUTES:
        path = f"comfy.model_patcher.ModelPatcher.{attribute}"
        _resolve(patcher, attribute, path)
        checked += 1

    guider_object = guider if guider is not None else _default_guider(import_module, patcher)
    for attribute in GUIDER_INSTANCE_ATTRIBUTES:
        path = f"comfy.samplers.CFGGuider.{attribute}"
        _resolve(guider_object, attribute, path)
        checked += 1

    quantized = quantized_tensor if quantized_tensor is not None else _default_quantized_tensor(import_module)
    for attribute in QUANTIZED_TENSOR_INSTANCE_ATTRIBUTES:
        path = f"comfy.quant_ops.QuantizedTensor.{attribute}"
        _resolve(quantized, attribute, path)
        checked += 1

    if model_detection_object is not None:
        checked += assert_model_detection_surface(model_detection_object, adapters)

    return checked


__all__ = [
    "GUIDER_INSTANCE_ATTRIBUTES",
    "MODEL_DETECTION_ANCHORS",
    "MODEL_PATCHER_INSTANCE_ATTRIBUTES",
    "OPTIONAL_TOUCHPOINTS",
    "QUANTIZED_TENSOR_INSTANCE_ATTRIBUTES",
    "QUANT_ALGORITHM_FIELDS",
    "QUANT_ALGORITHM_FORMATS",
    "QUANT_LAYOUT_PARAMETERS",
    "TOUCHPOINTS",
    "Touchpoint",
    "assert_comfy_surface",
    "assert_model_detection_surface",
]
