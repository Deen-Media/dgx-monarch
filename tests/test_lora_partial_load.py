"""actor/lora_partial_load.py: a one-GPU partial load with a LoRA stack is made whole, or refused.

With lora_low_rss off ComfyUI may load a patched model only partly. After its
load and before the first forward the Worker takes the low-RSS route for that
load by itself where it can (a full-load bake, then the backup and the patches
freed). Where it cannot, a weight-only format renders from cast-time weights,
and a layer that also quantizes its input (FP8 here) is refused, class C, with
nothing half-baked. Real ComfyUI on CPU (skipped where it is not importable);
the rig is the reviewer's FP8 probe of 2026-11-11.
"""
from __future__ import annotations

import json
import os
import sys
import weakref
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import lora_partial_load
from dgx_monarch.refusal import RefusalClass, RefusalTag, parse_leading_refusal_tag
from module_location_helpers import from_checkout

OUT, IN, RANK = 64, 512, 4
LAYOUTS = {"float8_e4m3fn": "TensorCoreFP8E4M3Layout", "int8_tensorwise": "TensorWiseINT8Layout"}


@pytest.fixture(scope="module")
def comfy_cpu():
    """Real ComfyUI with --cpu, restored afterwards."""
    def is_comfy(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved = {name: module for name, module in sys.modules.items() if is_comfy(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), sys.argv
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR") or os.environ.get("COMFYUI_DIR") or "../ComfyUI")
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-lora-kept-bake", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI on sys.path")
            options.enable_args_parsing()
            pytest.importorskip("comfy.lora")
            pytest.importorskip("comfy.quant_ops")
        finally:
            sys.argv = argv_before
        yield
    finally:
        from dgx_monarch.actor import lora_cast_time

        if "comfy.model_patcher" in sys.modules:
            lora_cast_time.uninstall()
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


@pytest.fixture
def installed(comfy_cpu):
    from dgx_monarch.actor import lora_cast_time

    assert lora_cast_time.install() is True
    try:
        yield
    finally:
        assert lora_cast_time.uninstall() is True


def _rig(formats=("float8_e4m3fn",)):
    """A root with one patched quantized Linear per format, each with a bias, and its patcher."""
    import comfy.lora
    import comfy.model_patcher
    import comfy.ops
    import torch
    from comfy.quant_ops import QuantizedTensor

    cpu = torch.device("cpu")
    ops = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)
    names = [f"q{index}" for index in range(len(formats))]

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in names:
                setattr(self, name, ops.Linear(IN, OUT, bias=True))

    class _Root(torch.nn.Module):
        manual_cast_dtype = torch.bfloat16

        def __init__(self):
            super().__init__()
            self.diffusion_model = _Block()

        def get_dtype_inference(self):
            return torch.bfloat16

    gen = torch.Generator().manual_seed(7)
    sd, lora, key_map = {}, {}, {}
    for name, fmt in zip(names, formats, strict=True):
        w = (torch.randn(OUT, IN, generator=gen) * 0.05).to(torch.bfloat16)
        if fmt == "float8_e4m3fn":
            qt, conf = QuantizedTensor.from_float(w, LAYOUTS[fmt]), {"format": fmt}
        else:
            qt = QuantizedTensor.from_float(w, LAYOUTS[fmt], per_channel=True, convrot=True,
                                            convrot_groupsize=256, is_weight=True)
            conf = {"format": fmt, "convrot": True, "convrot_groupsize": 256}
        unit = f"diffusion_model.{name}"
        sd[f"{unit}.weight"] = qt._qdata
        sd[f"{unit}.weight_scale"] = qt._params.scale
        sd[f"{unit}.comfy_quant"] = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
        sd[f"{unit}.bias"] = torch.randn(OUT, generator=gen).to(torch.bfloat16)
        lora[f"{unit}.lora_up.weight"] = torch.randn(OUT, RANK, generator=gen) * 0.01
        lora[f"{unit}.lora_down.weight"] = torch.randn(RANK, IN, generator=gen) * 0.01
        lora[f"{unit}.alpha"] = torch.tensor(float(RANK))
        key_map[unit] = f"{unit}.weight"
    patches = comfy.lora.load_lora(lora, key_map)

    def build():
        root = _Root()
        root.load_state_dict({key: value.clone() for key, value in sd.items()}, strict=True)
        patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
        patcher.add_patches(patches, 0.8)
        return root, patcher

    x = torch.randn(3, IN, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)

    def outputs(root):
        with torch.no_grad():
            return [getattr(root.diffusion_model, name)(x) for name in names]

    return build, outputs, cpu


def _offload(patcher, cpu, route):
    if route == "partial load":
        patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    else:
        patcher.load(cpu, full_load=True)
        patcher.partially_unload(cpu, memory_to_free=1e12)


def _full_load(build, outputs, cpu):
    root, patcher = build()
    patcher.load(cpu, full_load=True)
    return outputs(root)


class _Slot:
    """What the route reads and writes on a store slot (actor/stored_model.py)."""

    def __init__(self, base_patcher, active_patcher):
        self.base_patcher, self.active_patcher = base_patcher, active_patcher
        self.base_baked, self.unbake = False, "a record from an earlier stack"


def _slot(patcher, active=None):
    """A store slot for the patcher's model, registered as StoredModel registers itself."""
    slot = _Slot(patcher, active if active is not None else patcher)
    setattr(patcher.model, lora_partial_load.STORED, weakref.ref(slot))
    return slot


@pytest.fixture(autouse=True)
def _comfyui_forgets_the_models_of_each_test():
    yield
    management = sys.modules.get("comfy.model_management")
    if management is not None:
        management.current_loaded_models.clear()


def test_which_layers_quantize_their_input(comfy_cpu):
    build, _outputs, _cpu = _rig(("float8_e4m3fn", "int8_tensorwise"))
    root, _patcher = build()
    fp8, int8 = root.diffusion_model.q0, root.diffusion_model.q1
    assert lora_partial_load.quantizes_input(fp8) is True
    assert lora_partial_load.quantizes_input(int8) is False  # a weight-only format
    fp8._full_precision_mm = True
    assert lora_partial_load.quantizes_input(fp8) is False
    assert lora_partial_load.quantizes_input(SimpleNamespace(weight=None)) is False


def test_the_route_is_for_a_load_device_that_shares_the_hosts_memory(monkeypatch):
    """There a partial load is ComfyUI's budget at work; a GPU of its own was loaded partly because it is full."""
    import torch

    def device(kind):
        return SimpleNamespace(load_device=SimpleNamespace(type=kind, index=0))

    assert lora_partial_load.shares_host_memory(device("cpu")) is True
    assert lora_partial_load.shares_host_memory(device("mps")) is False
    assert lora_partial_load.shares_host_memory(SimpleNamespace()) is False
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(is_integrated=1))
    assert lora_partial_load.shares_host_memory(device("cuda")) is True  # GB10 in a DGX Spark
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(is_integrated=0))
    assert lora_partial_load.shares_host_memory(device("cuda")) is False

    def no_answer(d):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", no_answer)
    assert lora_partial_load.shares_host_memory(device("cuda")) is False


def test_a_store_slot_is_reachable_from_its_model():
    from dgx_monarch.actor.stored_model import StoredModel

    model = SimpleNamespace()
    slot = StoredModel(base_key=("m",), request_key=("m", ()), base_patcher=SimpleNamespace(model=model),
                       active_patcher=None, quant_kind="", precision_evidence=None, family="",
                       artifact_identity={})
    assert getattr(model, lora_partial_load.STORED)() is slot
    StoredModel(base_key=("m",), request_key=("m", ()), base_patcher=None, active_patcher=None, quant_kind="",
                precision_evidence=None, family="", artifact_identity={})  # a slot with no model yet


@pytest.mark.parametrize("formats", [("float8_e4m3fn",), ("int8_tensorwise",), ("int8_tensorwise", "float8_e4m3fn")],
                         ids=["fp8", "int8", "int8 and fp8"])
def test_a_partial_load_with_a_lora_is_made_whole_by_the_low_rss_bake(installed, formats, caplog):
    """The route lora_low_rss takes, taken for this load: bake under a full load, free the backup and the patches."""
    import torch

    build, outputs, cpu = _rig(formats)
    expected = _full_load(build, outputs, cpu)
    root, patcher = build()
    active = patcher.clone()  # the store's own patcher of the same stack, as a render clones it
    slot = _slot(patcher, active)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    assert root.model_lowvram is True and patcher.backup == {}  # every patched layer is off the device
    with caplog.at_level("INFO"):
        lora_partial_load.check(patcher)
    assert root.model_lowvram is False
    for module in root.diffusion_model.children():
        assert module.weight_function == [] and module.comfy_patched_weights is True
    assert patcher.patches == {} and patcher.backup == {} and active.patches == {}
    assert slot.base_baked is True and slot.unbake is None  # a stack change now reloads from disk
    assert "took the low-RSS route for this load" in caplog.text
    for got, want in zip(outputs(root), expected, strict=True):
        assert torch.equal(got, want)
    assert lora_partial_load.inexact(patcher) == []
    # ComfyUI may still move a layer off the device under pressure: it moves a baked weight with no patch.
    patcher.partially_unload(cpu, memory_to_free=1e12)
    for module in root.diffusion_model.children():
        assert module.weight_function == []
    for got, want in zip(outputs(root), expected, strict=True):
        assert torch.equal(got, want)
    lora_partial_load.check(patcher)  # nothing left to do or to refuse


def _refused(patcher):
    with pytest.raises(lora_partial_load.OffloadedInputQuantError) as raised:
        lora_partial_load.check(patcher)
    message = str(raised.value)
    assert parse_leading_refusal_tag(message) == RefusalTag(
        refusal_class=RefusalClass.CAPACITY, guard="partial_load_divergence", waivable=False)
    for words in ("float8_e4m3fn", "diffusion_model.q0.weight", "Nothing was sampled", "lora_low_rss on",
                  "The Worker tried to bake the whole stack under a full load"):
        assert words in message, words
    return message


@pytest.mark.parametrize("wall", ["no room", "a GPU of its own", "not a store model", "the full load fails"])
def test_where_the_route_cannot_be_taken_an_fp8_layer_is_refused_and_nothing_is_half_baked(
        installed, monkeypatch, wall):
    import torch

    from dgx_monarch.actor import store_bake

    build, outputs, cpu = _rig()
    (expected,) = _full_load(build, outputs, cpu)
    root, patcher = build()
    slot = None if wall == "not a store model" else _slot(patcher)
    q = root.diffusion_model.q0
    pristine = q.weight._qdata.clone()
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    if wall == "no room":
        monkeypatch.setattr(lora_partial_load, "_host_room", lambda: 0)
    elif wall == "a GPU of its own":
        monkeypatch.setattr(lora_partial_load, "shares_host_memory", lambda patcher: False)
    elif wall == "the full load fails":
        def fails(active, base_path=None):
            raise RuntimeError("out of memory")
        monkeypatch.setattr(store_bake, "_merge_and_free", fails)
    message = _refused(patcher)
    said = {"no room": "is available above the 5 GiB floor", "a GPU of its own": "this GPU has memory of its own",
            "not a store model": "not one the Worker's store holds",
            "the full load fails": "the full load it needs failed (RuntimeError: out of memory)"}[wall]
    assert said in message
    # The model is as ComfyUI's own load left it: pristine weight, its patch, its stack, no backup, the slot unmarked.
    assert torch.equal(q.weight._qdata, pristine) and len(q.weight_function) == 1
    assert root.model_lowvram is True and "diffusion_model.q0.weight" in patcher.patches and patcher.backup == {}
    assert slot is None or (slot.base_baked is False and slot.unbake == "a record from an earlier stack")
    assert not torch.equal(outputs(root)[0], expected)  # what the refusal stops: the input is not quantized
    # With the wall gone, the next render's check makes the load whole.
    monkeypatch.undo()
    if wall != "not a store model":
        lora_partial_load.check(patcher)
        assert torch.equal(outputs(root)[0], expected) and slot.base_baked is True


def test_where_the_route_cannot_be_taken_a_weight_only_format_renders_from_cast_time_weights(
        installed, monkeypatch, caplog):
    """INT8 has no input to quantize: the cast-time weight is exact, so the render goes on, one line said."""
    import torch

    build, outputs, cpu = _rig(("int8_tensorwise",))
    (expected,) = _full_load(build, outputs, cpu)
    root, patcher = build()
    slot = _slot(patcher)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    monkeypatch.setattr(lora_partial_load, "_host_room", lambda: 0)
    with caplog.at_level("INFO"):
        lora_partial_load.check(patcher)
    assert "stays partly loaded" in caplog.text and "is available above the 5 GiB floor" in caplog.text
    assert root.model_lowvram is True and slot.base_baked is False
    assert getattr(root.diffusion_model.q0.weight_function[0], "_dgxm_full_load_weight", False)
    assert torch.equal(outputs(root)[0], expected)


def test_a_whole_load_a_dynamic_patcher_and_a_model_with_no_stack_are_left_alone(installed, monkeypatch):
    build, _outputs, cpu = _rig()
    root, patcher = build()
    slot = _slot(patcher)

    def never(patcher):
        raise AssertionError("the route must not run")

    monkeypatch.setattr(lora_partial_load, "make_whole", never)
    patcher.load(cpu, full_load=True)
    lora_partial_load.check(patcher)  # loaded whole
    root.model_lowvram = True
    monkeypatch.setattr(type(patcher), "is_dynamic", lambda self: True)
    lora_partial_load.check(patcher)  # comfy-managed residency
    monkeypatch.undo()
    monkeypatch.setattr(lora_partial_load, "make_whole", never)
    patcher.patches.clear()
    lora_partial_load.check(patcher)  # no stack, as after a low-RSS bake
    assert slot.base_baked is False


def test_the_single_rank_check_rides_prepare_sampling_once(installed, monkeypatch):
    """partial_load_guard.install puts the check on a one-rank patcher; it runs after ComfyUI's load."""
    import comfy.patcher_extension as pe
    import torch

    from dgx_monarch.actor import partial_load_guard

    build, outputs, cpu = _rig()
    (expected,) = _full_load(build, outputs, cpu)
    root, patcher = build()
    slot = _slot(patcher)
    partial_load_guard.install(patcher, 1)
    partial_load_guard.install(patcher, None)  # a second install adds nothing
    wrappers = pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, lora_partial_load.WRAPPER_KEY, patcher.model_options, is_model_options=True)
    assert len(wrappers) == 1
    assert not pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, partial_load_guard.WRAPPER_KEY, patcher.model_options,
        is_model_options=True)  # the cross-rank guard is for a world above one
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    result = wrappers[0](lambda *args, **kwargs: ("real model", "conds", []), patcher, (1, 4, 8, 8), {})
    assert result == ("real model", "conds", [])
    assert root.model_lowvram is False and torch.equal(outputs(root)[0], expected) and slot.base_baked is True
