"""actor/store_detect: load options, precision readings, the FSDP checkpoint pin.

A silent drift in the option table (a new weight_dtype value falling through to
the default {}) would change what comfy loads without any node or gate noticing."""
import dis
import gc
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from dgx_monarch.actor.store_detect import (
    BF16_CORE_FP32_AUXILIARY_PROFILE,
    _combine_checkpoint_and_live_kinds,
    _detect_checkpoint_kind,
    _detect_live_precision,
    _detect_quant_kind,
    _is_quantized_kind,
    _quant_kind_from_options,
    _to_comfy_model_options,
    file_dtypes_preserved,
    validate_fsdp_request_checkpoint,
)
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.detect import FsdpLaunchQuantProof, detect_quant_from_header
from dgx_monarch.adapters.fsdp import WAN_FP32_INGRESS_PROFILE, fsdp_precision_profile_is_admitted
from dgx_monarch.adapters.fsdp_islands import ALL_FP16_PROFILE, FP32_ISLANDS_PROFILE
from dgx_monarch.safetensors_header import SafetensorsFileIdentity
from slab_lifetime_helpers import reset_slab_lifetime

_FSDP_IDENTITY = SafetensorsFileIdentity(1, 2, 3, 4, 5)


def _instruction_after_named_call(code, name):
    instructions = list(dis.get_instructions(code))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_ATTR"}
        and instruction.argval == name
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    return instructions[call_index + 1]


def _run_with_instruction_abort(code, target, primary, operation):
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-fsdp-pin-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    try:
        operation()
    finally:
        monitoring.set_local_events(tool_id, code, 0)
        monitoring.register_callback(
            tool_id, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool_id)


def _fds_for_path(path):
    expected = os.stat(path)
    matches = set()
    for entry in os.listdir("/proc/self/fd"):
        try:
            current = os.fstat(int(entry))
        except OSError:
            continue
        if (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino):
            matches.add(int(entry))
    return matches

_TABLE = [
    (None, {}),
    ({}, {}),
    ({"weight_dtype": "default"}, {}),
    ({"weight_dtype": "bf16"}, {"dtype": torch.bfloat16}),
    ({"weight_dtype": "fp8_e4m3fn"}, {"dtype": torch.float8_e4m3fn}),
    ({"weight_dtype": "fp8_e4m3fn_fast"},
     {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True}),
    ({"weight_dtype": "fp8_e5m2"}, {"dtype": torch.float8_e5m2}),
]


@pytest.fixture(autouse=True)
def _stock_wan_comfy_modules(monkeypatch):
    """Install exact fake Comfy class identities for detector contract tests."""
    comfy = ModuleType("comfy")
    model_management = ModuleType("comfy.model_management")
    model_base = ModuleType("comfy.model_base")
    ops = ModuleType("comfy.ops")
    ldm = ModuleType("comfy.ldm")
    wan = ModuleType("comfy.ldm.wan")
    wan_model = ModuleType("comfy.ldm.wan.model")

    # The audited Wan classes are defined below and exist by the time this runs.
    # The other tests need these fakes only to keep comfy imports hermetic.
    model_base.WAN21 = type("WAN21", (), {})
    model_base.WAN22 = _StockWanBase
    model_base.WAN21_FlowRVS = type("WAN21_FlowRVS", (), {})
    wan_model.WanModel = _StockWanModel
    ops.disable_weight_init = SimpleNamespace(Conv3d=_StockDisableConv3d)
    ops.manual_cast = SimpleNamespace(Conv3d=_StockManualConv3d)
    model_management.unload_all_models = lambda: None
    model_management.soft_empty_cache = lambda: None
    comfy.model_management = model_management
    comfy.model_base = model_base
    comfy.ops = ops
    comfy.ldm = ldm
    ldm.wan = wan
    wan.model = wan_model

    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.model_base", model_base)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    monkeypatch.setitem(sys.modules, "comfy.ldm", ldm)
    monkeypatch.setitem(sys.modules, "comfy.ldm.wan", wan)
    monkeypatch.setitem(sys.modules, "comfy.ldm.wan.model", wan_model)


@pytest.mark.parametrize("options,expected", _TABLE,
                         ids=[str(o) for o, _ in _TABLE])
def test_to_comfy_model_options_maps_every_weight_dtype(options, expected):
    assert _to_comfy_model_options(options) == expected


def test_qwen_cache_is_not_passed_as_a_comfy_load_option():
    assert _to_comfy_model_options({
        "qwen_image21_cache": {"device": "gpu", "dtype": "int8"},
    }) == {}


@pytest.mark.parametrize(("value", "error"), [
    ("unrecognized_value", ValueError),
    (None, TypeError),
    (1, TypeError),
    (True, TypeError),
    ([], TypeError),
    ({}, TypeError),
])
@pytest.mark.parametrize(
    "reader",
    [_to_comfy_model_options, _quant_kind_from_options],
)
def test_worker_rejects_unsupported_or_non_string_weight_dtype(
    value,
    error,
    reader,
):
    with pytest.raises(error, match="weight_dtype"):
        reader({"weight_dtype": value})


@pytest.mark.parametrize("options", ["bf16", [], 1, True])
@pytest.mark.parametrize(
    "reader",
    [_to_comfy_model_options, _quant_kind_from_options],
)
def test_worker_rejects_non_mapping_model_options(options, reader):
    with pytest.raises(TypeError, match="model options must be a mapping"):
        reader(options)


def _patcher_with_dtypes(*dtypes):
    diffusion_model = torch.nn.Module()
    diffusion_model.weights = torch.nn.ParameterList([
        torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        for dtype in dtypes
    ])
    return SimpleNamespace(model=SimpleNamespace(
        diffusion_model=diffusion_model,
        model_config=None,
    ))


@pytest.mark.parametrize(("torch_dtype", "header_dtype", "expected"), [
    (torch.bfloat16, "BF16", "bf16"),
    (torch.float16, "F16", "fp16"),
])
def test_header_and_live_full_precision_detection_agree(torch_dtype, header_dtype, expected):
    header = {"__tensors__": {"w": {"dtype": header_dtype}}}
    assert detect_quant_from_header(["w"], header) == expected
    assert _detect_quant_kind(_patcher_with_dtypes(torch_dtype), "bf16") == expected


def test_mixed_live_fp16_bf16_fails_closed_to_fp16():
    patcher = _patcher_with_dtypes(torch.bfloat16, torch.float16)
    live = _detect_live_precision(patcher, "bf16")
    assert live.quant_kind == "fp16"
    assert live.live_dtype_profile == "mixed_or_uniform_fp16"


def test_uniform_live_fp16_reads_as_the_admitted_fp16_profile():
    live = _detect_live_precision(_patcher_with_dtypes(torch.float16), "bf16")
    assert live.quant_kind == "fp16"
    assert live.live_dtype_profile == ALL_FP16_PROFILE
    assert fsdp_precision_profile_is_admitted(ALL_FP16_PROFILE, 0, 0, "fp16")
    assert not fsdp_precision_profile_is_admitted(ALL_FP16_PROFILE, 0, 0, "bf16")


def test_bf16_core_with_small_fp32_islands_reads_as_islands():
    diffusion_model = torch.nn.Module()
    diffusion_model.core = torch.nn.Parameter(
        torch.zeros(4096, dtype=torch.bfloat16), requires_grad=False)
    diffusion_model.scale_shift_table = torch.nn.Parameter(
        torch.zeros(2, dtype=torch.float32), requires_grad=False)
    patcher = SimpleNamespace(model=SimpleNamespace(
        diffusion_model=diffusion_model, model_config=None))
    live = _detect_live_precision(patcher, "bf16")
    assert (live.quant_kind, live.live_dtype_profile) == ("bf16", FP32_ISLANDS_PROFILE)
    assert (live.auxiliary_parameter_count, live.auxiliary_parameter_bytes) == (1, 8)
    assert fsdp_precision_profile_is_admitted(FP32_ISLANDS_PROFILE, 1, 8, "bf16")
    assert not fsdp_precision_profile_is_admitted(FP32_ISLANDS_PROFILE, 1, 8, "fp16")
    assert not fsdp_precision_profile_is_admitted(FP32_ISLANDS_PROFILE, 0, 0, "bf16")


def test_live_fp32_or_mixed_bf16_fp32_fails_closed_to_fp32():
    assert _detect_quant_kind(_patcher_with_dtypes(torch.float32), "bf16") == "fp32"
    patcher = _patcher_with_dtypes(torch.bfloat16, torch.float32)
    assert _detect_quant_kind(patcher, "bf16") == "fp32"


class _StockDisableConv3d(torch.nn.Conv3d):
    pass


class _StockManualConv3d(_StockDisableConv3d):
    pass


class _StockWanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 8 FP32 bytes of 8200 total stays under the 1e-3 Wan ingress fraction
        # bound without allocating a model-sized test fixture.
        self.core = torch.nn.Parameter(
            torch.zeros(4096, dtype=torch.bfloat16), requires_grad=False
        )
        self.patch_embedding = _StockDisableConv3d(
            1, 1, kernel_size=1, dtype=torch.float32
        )
        self.dim = 1
        self.in_dim = 1
        self.patch_size = (1, 1, 1)


_StockWanModel.__module__ = "comfy.ldm.wan.model"
_StockWanModel.__name__ = "WanModel"


class _StockWanBase:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.model_config = None


_StockWanBase.__module__ = "comfy.model_base"
_StockWanBase.__name__ = "WAN22"


def _wan_patcher(model=None):
    diffusion_model = model if model is not None else _StockWanModel()
    return SimpleNamespace(model=_StockWanBase(diffusion_model))


def test_exact_stock_wan_fp32_patch_ingress_has_audited_bf16_profile():
    live = _detect_live_precision(_wan_patcher(), "bf16")
    assert live.quant_kind == "bf16"
    assert live.live_dtype_profile == WAN_FP32_INGRESS_PROFILE
    assert live.auxiliary_parameter_count == 2
    assert live.auxiliary_parameter_bytes == 8
    assert live.checkpoint_kind is None

    bound = live.bind_checkpoint("bf16", "bf16")
    assert bound.public() == {
        "live_dtype_profile": WAN_FP32_INGRESS_PROFILE,
        "auxiliary_parameter_count": 2,
        "auxiliary_parameter_bytes": 8,
        "checkpoint_precision": "bf16",
    }


def test_exact_stock_wan_manual_cast_patch_ingress_is_audited():
    model = _StockWanModel()
    model.patch_embedding = _StockManualConv3d(
        1, 1, kernel_size=1, dtype=torch.float32
    )

    live = _detect_live_precision(_wan_patcher(model), "bf16")

    assert live.quant_kind == "bf16"
    assert live.live_dtype_profile == WAN_FP32_INGRESS_PROFILE


def test_wan_profile_rejects_unrecognized_custom_operations_metadata():
    patcher = _wan_patcher()
    patcher.model.model_config = SimpleNamespace(
        quant_config=None,
        custom_operations=SimpleNamespace(kind="future_quant_wrapper"),
    )

    live = _detect_live_precision(patcher, "bf16")

    assert live.quant_kind == "unknown"
    assert live.live_dtype_profile == "unrecognized_custom_operations"


def _assert_islands_not_wan(model) -> None:
    """A Wan-profile near-miss inside the islands ceiling reads as islands.

    The Wan profile stays exact. fp32 parameters that are not the audited pair
    fall back to the generic islands profile, which replicates them, instead of
    refusing the launch.
    """
    live = _detect_live_precision(_wan_patcher(model), "bf16")
    assert live.quant_kind == "bf16"
    assert live.live_dtype_profile == FP32_ISLANDS_PROFILE
    assert live.live_dtype_profile != WAN_FP32_INGRESS_PROFILE


def test_wan_fp32_ingress_exception_rejects_any_extra_fp32_parameter():
    model = _StockWanModel()
    model.extra_fp32 = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32), requires_grad=False
    )
    _assert_islands_not_wan(model)


def test_wan_fp32_ingress_exception_requires_exact_weight_and_bias_pair():
    model = _StockWanModel()
    model.patch_embedding = _StockDisableConv3d(
        1,
        1,
        kernel_size=1,
        bias=False,
        dtype=torch.float32,
    )

    _assert_islands_not_wan(model)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda model: setattr(model, "patch_size", (1, 1, 2)),
            id="patch-size-metadata-drift",
        ),
        pytest.param(
            lambda model: setattr(model, "dim", 2),
            id="output-dimension-drift",
        ),
        pytest.param(
            lambda model: setattr(model, "in_dim", 2),
            id="input-dimension-drift",
        ),
        pytest.param(
            lambda model: setattr(
                model,
                "patch_embedding",
                torch.nn.Linear(1, 1, dtype=torch.float32),
            ),
            id="not-conv3d",
        ),
    ],
)
def test_wan_fp32_ingress_exception_rejects_structural_near_misses(mutation):
    model = _StockWanModel()
    mutation(model)

    _assert_islands_not_wan(model)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("groups", 2),
        ("dilation", (1, 1, 2)),
        ("padding", (0, 0, 1)),
        ("padding_mode", "reflect"),
    ],
)
def test_wan_fp32_ingress_exception_rejects_altered_conv_semantics(attribute, value):
    model = _StockWanModel()
    setattr(model.patch_embedding, attribute, value)

    _assert_islands_not_wan(model)


def test_wan_fp32_ingress_exception_rejects_material_precision_share():
    model = _StockWanModel()
    model.core = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.bfloat16), requires_grad=False
    )

    assert _detect_quant_kind(_wan_patcher(model), "bf16") == "fp32"


def test_wan_fp32_ingress_exception_rejects_oversized_auxiliary_convolution():
    model = _StockWanModel()
    model.patch_embedding = _StockDisableConv3d(
        1024,
        1024,
        kernel_size=1,
        dtype=torch.float32,
    )
    model.dim = 1024
    model.in_dim = 1024

    assert _detect_quant_kind(_wan_patcher(model), "bf16") == "fp32"


def test_wan_fp32_ingress_exception_rejects_class_substitution():
    class LookalikeWanModel(_StockWanModel):
        pass

    model = LookalikeWanModel()

    _assert_islands_not_wan(model)


def test_wan_fp32_ingress_exception_rejects_conv3d_subclass():
    class Conv3dSubclass(_StockDisableConv3d):
        pass

    model = _StockWanModel()
    model.patch_embedding = Conv3dSubclass(
        1,
        1,
        kernel_size=1,
        dtype=torch.float32,
    )

    _assert_islands_not_wan(model)


def test_wan_fp32_ingress_exception_rejects_plain_torch_conv3d():
    model = _StockWanModel()
    model.patch_embedding = torch.nn.Conv3d(
        1,
        1,
        kernel_size=1,
        dtype=torch.float32,
    )

    _assert_islands_not_wan(model)


def test_wan_fp32_ingress_exception_rejects_base_class_substitution():
    model = _StockWanModel()
    patcher = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=model, model_config=None)
    )

    live = _detect_live_precision(patcher, "bf16")
    assert live.quant_kind == "bf16"
    assert live.live_dtype_profile == FP32_ISLANDS_PROFILE


def test_unrecognized_or_empty_live_precision_cannot_mint_bf16():
    assert _detect_quant_kind(_patcher_with_dtypes(torch.float64), "bf16") == "unknown"
    assert _detect_quant_kind(_patcher_with_dtypes(), "bf16") == "unknown"


@pytest.mark.parametrize(("checkpoint_kind", "live_kind", "expected"), [
    ("bf16", "bf16", "bf16"),
    ("fp16", "bf16", "fp16"),
    ("bf16", "fp16", "fp16"),
    ("fp32", "bf16", "fp32"),
    ("bf16", "fp32", "fp32"),
    ("fp16", "fp8", "fp8"),
    ("int8", "fp16", "int8"),
    ("unknown", "bf16", "unknown"),
])
def test_checkpoint_and_live_kind_merge_never_weakens_fp16_or_unknown(
        checkpoint_kind, live_kind, expected):
    assert _combine_checkpoint_and_live_kinds(checkpoint_kind, live_kind) == expected


@pytest.mark.parametrize(("checkpoint_kind", "live_kind", "expected"), [
    # The case this rule covers: a BF16 export carrying auxiliary FP32 tensors.
    # A copying load takes them in as BF16 and an assigning load keeps them
    # FP32; one checkpoint must not change its reported precision with the
    # residency that loaded it.
    ("bf16", "fp32", "bf16"),
    # A header that says full precision itself still decides.
    ("fp32", "fp32", "fp32"),
    ("fp16", "fp32", "fp16"),
    # Quantized readings keep their priority, from either side.
    ("bf16", "fp8", "fp8"),
    ("int8", "fp32", "int8"),
    # No header, or one nothing could classify, cannot be deferred to.
    (None, "fp32", "fp32"),
    ("unknown", "fp32", "fp32"),
])
def test_a_bf16_core_defers_its_fp32_reading_to_the_header(
        checkpoint_kind, live_kind, expected):
    assert _combine_checkpoint_and_live_kinds(
        checkpoint_kind, live_kind, live_dtypes_are_the_file=True,
        live_profile=BF16_CORE_FP32_AUXILIARY_PROFILE) == expected


@pytest.mark.parametrize("live_kind", ["fp32", "fp16", "unknown"])
@pytest.mark.parametrize("profile", ["mixed_or_uniform_fp32", "all_bf16", None])
def test_only_a_bf16_core_may_defer_to_the_header(live_kind, profile):
    """The header's BF16 answer is a compatibility default: detect.py returns it
    for any header with no FP8, INT8 or F16 marker, an all-FP32 file included.
    A model without a BF16 core must not inherit that default, or an FP32
    checkpoint could report different precision under different residencies."""
    expected = _combine_checkpoint_and_live_kinds("bf16", live_kind)
    assert _combine_checkpoint_and_live_kinds(
        "bf16", live_kind, live_dtypes_are_the_file=True,
        live_profile=profile) == expected


def test_an_all_fp32_model_reports_fp32_under_an_assigning_load():
    """End to end over the detector, not just the merge: no BF16 core, so the
    header cannot override it whatever it says."""
    live = _detect_live_precision(_patcher_with_dtypes(torch.float32), "bf16")
    assert live.live_dtype_profile == "mixed_or_uniform_fp32"
    assert _combine_checkpoint_and_live_kinds(
        "bf16", live.quant_kind, live_dtypes_are_the_file=True,
        live_profile=live.live_dtype_profile) == "fp32"


def test_a_bf16_core_with_fp32_auxiliaries_is_named_apart():
    live = _detect_live_precision(
        _patcher_with_dtypes(torch.bfloat16, torch.float32), "bf16")
    assert live.live_dtype_profile == BF16_CORE_FP32_AUXILIARY_PROFILE
    assert live.quant_kind == "fp32"
    assert _combine_checkpoint_and_live_kinds(
        "bf16", live.quant_kind, live_dtypes_are_the_file=True,
        live_profile=live.live_dtype_profile) == "bf16"


def test_the_file_dtype_flag_changes_nothing_for_a_copying_load():
    """Stock verdicts are the reference and must not move."""
    for checkpoint_kind in ("bf16", "fp16", "fp32", "unknown", None):
        for live_kind in ("bf16", "fp16", "fp32", "fp8", "int8", "unknown"):
            assert (_combine_checkpoint_and_live_kinds(checkpoint_kind, live_kind)
                    == _combine_checkpoint_and_live_kinds(
                        checkpoint_kind, live_kind, live_dtypes_are_the_file=False,
                        live_profile=BF16_CORE_FP32_AUXILIARY_PROFILE))


def test_mixed_fp32_evidence_counts_the_auxiliary_tensors():
    """The header decides the reported kind, so the evidence row is the only
    place the auxiliary tensors stay visible. The public LTX 2.5 BF16 export
    carries 290 FP32 scale-shift tables at 0.045 percent of its bytes (header
    census, 2026-08-12), and both rungs must be able to say so."""
    live = _detect_live_precision(_patcher_with_dtypes(torch.bfloat16, torch.float32, torch.float32), "bf16")
    assert live.quant_kind == "fp32"
    assert live.live_dtype_profile == BF16_CORE_FP32_AUXILIARY_PROFILE
    assert live.auxiliary_parameter_count == 2
    assert live.auxiliary_parameter_bytes == 2 * 4   # two one-element FP32 params
    # Counting them may not widen any FSDP grant.
    assert not fsdp_precision_profile_is_admitted(
        live.live_dtype_profile,
        live.auxiliary_parameter_count,
        live.auxiliary_parameter_bytes,
        "bf16",
    )


@pytest.mark.parametrize(("patcher", "expected"), [
    (SimpleNamespace(is_dynamic=lambda: True), True),
    (SimpleNamespace(is_dynamic=lambda: False), False),
    (SimpleNamespace(), False),                       # a classic patcher
    (SimpleNamespace(is_dynamic="not callable"), False),
])
def test_the_file_dtype_reading_follows_comfys_own_dynamic_flag(patcher, expected):
    """``load_model_weights(..., assign=model_patcher.is_dynamic())`` is the
    switch that decides whether live dtypes are the file's or the model's, so
    the detector reads that same flag instead of guessing from the rung."""
    assert file_dtypes_preserved(patcher) is expected


def test_a_patcher_that_raises_reads_as_a_copying_load():
    """A detector may not fail a load; the conservative reading is the one that
    keeps the live observation authoritative."""
    def _boom():
        raise RuntimeError("no")

    assert file_dtypes_preserved(SimpleNamespace(is_dynamic=_boom)) is False


def test_unclassifiable_safetensors_header_is_unknown_not_bf16(tmp_path):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(b"not a safetensors container")
    assert _detect_checkpoint_kind(str(path), "bf16") == "unknown"


def test_non_safetensors_and_explicit_fp8_keep_compatibility(tmp_path):
    assert _detect_checkpoint_kind(str(tmp_path / "legacy.ckpt"), "bf16") is None
    # An explicit loader cast is authoritative and does not require the file
    # to exist for header classification.
    assert _detect_checkpoint_kind(str(tmp_path / "missing.safetensors"), "fp8") == "fp8"


def test_ordinary_store_keeps_generic_header_classification(monkeypatch, tmp_path):
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(
        detect,
        "sniff_checkpoint",
        lambda _path: ("family", "bf16"),
    )
    assert _detect_checkpoint_kind(
        str(tmp_path / "bf16-with-integer-buffer.safetensors"),
        "bf16",
    ) == "bf16"


@pytest.mark.parametrize("strict_kind", ["fp32", "unknown"])
@pytest.mark.parametrize("options", [None, {"weight_dtype": "bf16"}])
def test_worker_local_fsdp_preflight_rejects_unproved_header(
    monkeypatch,
    strict_kind,
    options,
):
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(
        detect,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: FsdpLaunchQuantProof(strict_kind, _FSDP_IDENTITY),
    )
    with pytest.raises(
        UnsupportedModelError,
        match=rf"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant={strict_kind}\)",
    ):
        validate_fsdp_request_checkpoint(
            "/worker/models/model.safetensors",
            options,
            None,
        )


@pytest.mark.parametrize("proof_kind", ["bf16", "fp16"])
def test_worker_local_fsdp_preflight_retains_bf16_inode_proof(monkeypatch, proof_kind):
    from dgx_monarch.adapters import detect

    proof = FsdpLaunchQuantProof(proof_kind, _FSDP_IDENTITY)
    monkeypatch.setattr(
        detect,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: proof,
    )

    assert validate_fsdp_request_checkpoint(
        "/worker/models/model.safetensors",
        None,
        None,
    ) is proof


@pytest.mark.parametrize(
    ("platform", "proc_fd_available"),
    [("darwin", True), ("linux", False)],
)
def test_fsdp_pin_requires_linux_proc_fd_before_open_or_alias_creation(
    monkeypatch,
    platform,
    proc_fd_available,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, store_detect
    from dgx_monarch.mesh_safety import ArtifactBindingError

    side_effects = []

    def forbidden(name):
        def call(*_args, **_kwargs):
            side_effects.append(name)
            pytest.fail(f"unsupported worker reached {name}")

        return call

    monkeypatch.setattr(
        fsdp_checkpoint_pin,
        "sys",
        SimpleNamespace(platform=platform),
    )
    monkeypatch.setattr(
        fsdp_checkpoint_pin,
        "_PROC_FD_DIRECTORY",
        SimpleNamespace(is_dir=lambda: proc_fd_available),
    )
    monkeypatch.setattr(
        fsdp_checkpoint_pin, "open", forbidden("checkpoint open"), raising=False)
    monkeypatch.setattr(
        fsdp_checkpoint_pin.tempfile,
        "TemporaryDirectory",
        forbidden("pin alias creation"),
    )

    with pytest.raises(
        ArtifactBindingError,
        match=r"requires Linux with procfs mounted at /proc/self/fd",
    ):
        store_detect.pin_fsdp_checkpoint(
            "/worker/models/model.safetensors",
            FsdpLaunchQuantProof("bf16", _FSDP_IDENTITY),
        )

    assert side_effects == []


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize(
    ("boundary", "expected_target"),
    [("open", "STORE_FAST"), ("TemporaryDirectory", "STORE_FAST")],
)
def test_fsdp_pin_finalizable_acquisition_survives_call_store_interrupt(
    monkeypatch,
    tmp_path,
    boundary,
    expected_target,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, store_detect

    class PinAbort(BaseException):
        pass

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    identity = SafetensorsFileIdentity.from_stat(os.stat(checkpoint))
    proof = FsdpLaunchQuantProof("bf16", identity)
    pin_root = tmp_path / "pins"
    pin_root.mkdir()
    monkeypatch.setattr(fsdp_checkpoint_pin.tempfile, "tempdir", str(pin_root))
    monkeypatch.setattr(fsdp_checkpoint_pin, "_PIN_POISON_OWNERS", [])
    target = _instruction_after_named_call(
        fsdp_checkpoint_pin.pin_fsdp_checkpoint.__code__, boundary)
    assert target.opname == expected_target
    before = _fds_for_path(checkpoint)
    primary = PinAbort(f"{boundary} return interrupted")

    with pytest.raises(PinAbort) as caught:
        _run_with_instruction_abort(
            fsdp_checkpoint_pin.pin_fsdp_checkpoint.__code__,
            target.offset,
            primary,
            lambda: store_detect.pin_fsdp_checkpoint(str(checkpoint), proof),
        )

    assert caught.value is primary
    gc.collect()
    assert _fds_for_path(checkpoint) == before
    assert list(pin_root.iterdir()) == []
    assert fsdp_checkpoint_pin._PIN_POISON_OWNERS == []


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_fsdp_pin_return_is_owned_before_callee_handoff_interrupt(
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, store_detect

    class PinAbort(BaseException):
        pass

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    identity = SafetensorsFileIdentity.from_stat(os.stat(checkpoint))
    proof = FsdpLaunchQuantProof("bf16", identity)
    pin_root = tmp_path / "pins"
    pin_root.mkdir()
    monkeypatch.setattr(fsdp_checkpoint_pin.tempfile, "tempdir", str(pin_root))
    monkeypatch.setattr(fsdp_checkpoint_pin, "_PIN_POISON_OWNERS", [])
    code = fsdp_checkpoint_pin.pin_fsdp_checkpoint.__code__
    target = next(
        instruction for instruction in reversed(list(dis.get_instructions(code)))
        if instruction.opname == "RETURN_VALUE"
    )
    handoff = []
    before = _fds_for_path(checkpoint)
    primary = PinAbort("callee return interrupted")

    with pytest.raises(PinAbort) as caught:
        _run_with_instruction_abort(
            code,
            target.offset,
            primary,
            lambda: store_detect.pin_fsdp_checkpoint(
                str(checkpoint), proof, handoff=handoff),
        )

    assert caught.value is primary
    assert len(handoff) == 1
    pin = handoff[0]
    owned_fd = pin._fd
    assert os.path.exists(pin.loader_path)
    os.fstat(owned_fd)
    assert fsdp_checkpoint_pin._PIN_POISON_OWNERS == []

    pin.close()

    assert pin._closed
    assert not os.path.exists(pin.loader_path)
    assert list(pin_root.iterdir()) == []
    assert _fds_for_path(checkpoint) == before
    with pytest.raises(OSError):
        os.fstat(owned_fd)


def test_partial_pin_close_failure_preserves_primary_and_retains_cleanup(
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, slab_lifetime, store_detect
    from dgx_monarch.adapters.detect import FsdpLaunchQuantProof

    class PinAbort(BaseException):
        pass

    class CleanupAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    identity = SafetensorsFileIdentity.from_stat(os.stat(checkpoint))
    proof = FsdpLaunchQuantProof("bf16", identity)
    pin_dir = tmp_path / "pin"
    alias = pin_dir / "checkpoint.safetensors"
    primary = PinAbort("pin construction cancelled")
    opened = []
    healthy_open = open
    healthy_stat = fsdp_checkpoint_pin.os.stat
    healthy_unlink = fsdp_checkpoint_pin.os.unlink
    healthy_rmdir = fsdp_checkpoint_pin.os.rmdir

    def record_open(*args, **kwargs):
        owner = healthy_open(*args, **kwargs)
        opened.append(owner.fileno())

        class CloseThenAbort:
            def fileno(self):
                return owner.fileno()

            def close(self):
                owner.close()
                raise CleanupAbort("cleanup interrupted after release")

        return CloseThenAbort()

    class PinDirectory:
        def __init__(self, **_kwargs):
            pin_dir.mkdir()
            self.name = str(pin_dir)

        def cleanup(self):
            healthy_rmdir(self.name)

    def abort_alias_stat(target, *args, **kwargs):
        if os.fspath(target) == str(alias):
            raise primary
        return healthy_stat(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(fsdp_checkpoint_pin, "open", record_open, raising=False)
        patch.setattr(fsdp_checkpoint_pin.tempfile, "TemporaryDirectory", PinDirectory)
        patch.setattr(fsdp_checkpoint_pin.os, "stat", abort_alias_stat)
        patch.setattr(
            fsdp_checkpoint_pin.os,
            "unlink",
            lambda target: healthy_unlink(target),
        )
        with pytest.raises(PinAbort) as caught:
            store_detect.pin_fsdp_checkpoint(str(checkpoint), proof)

    assert caught.value is primary
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.retained_resource_count() == 1
    retained, cleanup_error = slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES[0]
    assert isinstance(retained, fsdp_checkpoint_pin.PinnedFsdpCheckpoint)
    assert retained._close_uncertain
    assert not retained._closed
    assert retained._fd == -1
    assert isinstance(cleanup_error, CleanupAbort)
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    with pytest.raises(FileNotFoundError):
        os.stat(alias)
    with pytest.raises(FileNotFoundError):
        os.stat(pin_dir)


def test_partial_pin_directory_cleanup_cancellation_outranks_construction_error(
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, slab_lifetime, store_detect

    reset_slab_lifetime(monkeypatch)
    monkeypatch.setattr(fsdp_checkpoint_pin, "_PIN_POISON_OWNERS", [])
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    identity = SafetensorsFileIdentity.from_stat(os.stat(checkpoint))
    proof = FsdpLaunchQuantProof("bf16", identity)
    pin_dir = tmp_path / "pin"
    primary = RuntimeError("ordinary alias creation failure")
    cancellation = KeyboardInterrupt("pin directory cleanup cancelled")

    class PinDirectory:
        def __init__(self, **_kwargs):
            pin_dir.mkdir()
            self.name = str(pin_dir)

        def cleanup(self):
            raise cancellation

    monkeypatch.setattr(
        fsdp_checkpoint_pin.tempfile,
        "TemporaryDirectory",
        PinDirectory,
    )
    monkeypatch.setattr(
        fsdp_checkpoint_pin.os,
        "symlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        store_detect.pin_fsdp_checkpoint(str(checkpoint), proof)

    assert caught.value is cancellation
    assert caught.value.__cause__ is primary
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime.retained_resource_count() == 1
    retained = slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES[0][0]
    assert retained._closed
    assert retained._fd == -1


def test_partial_pin_unlink_failure_retains_fd_until_explicit_cleanup(
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, slab_lifetime, store_detect

    class PinAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    identity = SafetensorsFileIdentity.from_stat(os.stat(checkpoint))
    proof = FsdpLaunchQuantProof("bf16", identity)
    pin_dir = tmp_path / "pin"
    alias = pin_dir / "checkpoint.safetensors"
    primary = PinAbort("pin construction cancelled")
    healthy_stat = fsdp_checkpoint_pin.os.stat
    healthy_unlink = fsdp_checkpoint_pin.os.unlink

    class PinDirectory:
        def __init__(self, **_kwargs):
            pin_dir.mkdir()
            self.name = str(pin_dir)

        def cleanup(self):
            pin_dir.rmdir()

    def abort_alias_stat(target, *args, **kwargs):
        if os.fspath(target) == str(alias):
            raise primary
        return healthy_stat(target, *args, **kwargs)

    def fail_without_unlink(target):
        if os.fspath(target) == str(alias):
            raise OSError("pin alias unlink made no progress")
        return healthy_unlink(target)

    with monkeypatch.context() as patch:
        patch.setattr(fsdp_checkpoint_pin.tempfile, "TemporaryDirectory", PinDirectory)
        patch.setattr(fsdp_checkpoint_pin.os, "stat", abort_alias_stat)
        patch.setattr(fsdp_checkpoint_pin.os, "unlink", fail_without_unlink)
        with pytest.raises(PinAbort) as caught:
            store_detect.pin_fsdp_checkpoint(str(checkpoint), proof)

    assert caught.value is primary
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.retained_resource_count() == 1
    assert os.path.lexists(alias)
    retained = slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES[0][0]
    owned_fd = retained._fd
    os.fstat(owned_fd)

    slab_lifetime.release_after_explicit_unload()

    assert not slab_lifetime.cleanup_pending()
    assert not os.path.lexists(alias)
    assert not pin_dir.exists()
    with pytest.raises(OSError):
        os.fstat(owned_fd)


@pytest.mark.parametrize("suffix", [".ckpt", ".bin"])
def test_worker_explicit_bf16_legacy_fsdp_refuses_before_header_sniff(
    monkeypatch,
    suffix,
):
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(
        detect,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: pytest.fail("explicit BF16 legacy request reached sniff"),
    )
    with pytest.raises(
        UnsupportedModelError,
        match=r"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant=unknown\)",
    ):
        validate_fsdp_request_checkpoint(
            f"/worker/models/model{suffix}",
            {"weight_dtype": "bf16"},
            None,
        )


@pytest.mark.parametrize("suffix", [".ckpt", ".bin"])
def test_worker_default_legacy_fsdp_refuses_without_source_proof(
    monkeypatch,
    suffix,
):
    from dgx_monarch.adapters import detect

    seen = []
    monkeypatch.setattr(
        detect,
        "sniff_fsdp_launch_quant_proof",
        lambda path: seen.append(path) or FsdpLaunchQuantProof(None, None),
    )

    with pytest.raises(
        UnsupportedModelError,
        match=r"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant=unknown\)",
    ):
        validate_fsdp_request_checkpoint(
            f"/worker/models/model{suffix}",
            None,
            None,
        )

    assert seen == [f"/worker/models/model{suffix}"]


@pytest.mark.parametrize(("kind", "quantized"), [
    ("bf16", False),
    ("fp16", False),
    ("fp32", False),
    ("fp8", True),
    ("int8", True),
    ("mxfp8", True),
    ("nvfp4", True),
    ("unknown-future-kind", True),
])
def test_only_quantized_kinds_need_conversion_spike_invalidation(kind, quantized):
    assert _is_quantized_kind(kind) is quantized
