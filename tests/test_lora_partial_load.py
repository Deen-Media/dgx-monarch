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


@pytest.mark.parametrize("wall", ["no room", "a GPU of its own", "not a store model"])
def test_where_the_route_cannot_be_taken_an_fp8_layer_is_refused_and_nothing_is_half_baked(
        installed, monkeypatch, wall):
    import torch


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
    message = _refused(patcher)
    said = {"no room": "is available above the 5 GiB floor", "a GPU of its own": "this GPU has memory of its own",
            "not a store model": "not one the Worker's store holds"}[wall]
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


# A WHOLE load: ComfyUI's backup of every patched weight stands through the render. Where the room
# left is under ComfyUI's own full estimate for sampling the latent, the Worker frees it (the route's
# last step alone). On hardware (2026-11-12, one GPU of a DGX Spark under its clock cap, max 2100 MHz)
# a 241-frame Wan-Animate 2 render took 52.4 GiB where 81 frames takes 39, with 17.7 GiB of backup.

GIB = 2 ** 30
SAMPLING = ((1, 16, 62, 104, 60), {"positive": [], "negative": []})


def _whole(build, cpu):
    """A patched model ComfyUI loaded whole, in a store slot, with the store's own clone of the stack."""
    root, patcher = build()
    active = patcher.clone()
    slot = _slot(patcher, active)
    patcher.load(cpu, full_load=True)
    assert root.model_lowvram is False and patcher.backup  # the classic bake kept the originals
    return root, patcher, active, slot


def _asks(monkeypatch, ask_gib, room_gib):
    seen = []

    def ask(patcher, noise_shape, conds):
        seen.append((noise_shape, conds))
        return None if ask_gib is None else int(ask_gib * GIB)

    monkeypatch.setattr(lora_partial_load, "_sampling_ask", ask)
    monkeypatch.setattr(lora_partial_load, "_host_room", lambda: int(room_gib * GIB))
    return seen


@pytest.mark.parametrize("formats", [("float8_e4m3fn",), ("int8_tensorwise",), ("int8_tensorwise", "float8_e4m3fn")],
                         ids=["fp8", "int8", "int8 and fp8"])
def test_a_whole_load_gives_up_its_backup_where_the_sampler_needs_the_room(installed, monkeypatch, formats, caplog):
    import torch

    build, outputs, cpu = _rig(formats)
    expected = _full_load(build, outputs, cpu)
    root, patcher, active, slot = _whole(build, cpu)
    seen = _asks(monkeypatch, ask_gib=34.8, room_gib=20.4)  # the 241-frame clip at ComfyUI's whole-load edge
    with caplog.at_level("INFO"):
        lora_partial_load.check(patcher, SAMPLING)
    assert seen == [SAMPLING]
    assert patcher.backup == {} and patcher.patches == {} and active.patches == {}
    assert slot.base_baked is True and slot.unbake is None  # a stack change now reloads from disk
    assert root.model_lowvram is False  # no weight moved
    for module in root.diffusion_model.children():
        assert module.weight_function == [] and module.comfy_patched_weights is True
    assert "freed ComfyUI's weight backup before the first step" in caplog.text
    assert "20.4 GiB is available above the 5 GiB floor where ComfyUI estimates 34.8 GiB" in caplog.text
    for got, want in zip(outputs(root), expected, strict=True):
        assert torch.equal(got, want)
    # ComfyUI may still move a layer off the device under pressure: it moves a baked weight with no patch.
    patcher.partially_unload(cpu, memory_to_free=1e12)
    for got, want in zip(outputs(root), expected, strict=True):
        assert torch.equal(got, want)
    seen.clear()
    lora_partial_load.check(patcher, SAMPLING)  # no stack is left to weigh
    assert seen == []


@pytest.mark.parametrize("why", ["room enough", "room exactly the ask", "no estimate", "no latent given",
                                 "a GPU of its own", "not a store model", "no backup held"])
def test_a_whole_load_keeps_its_backup_wherever_the_step_does_not_apply(installed, monkeypatch, why, caplog):
    """With room for the sampler the backup stays, and a stack change swaps in place as before."""
    build, _outputs, cpu = _rig()
    root, patcher, active, slot = _whole(build, cpu)
    held, stack = dict(patcher.backup), dict(patcher.patches)
    _asks(monkeypatch, ask_gib=None if why == "no estimate" else 12.4,
          room_gib={"room enough": 40.0, "room exactly the ask": 12.4}.get(why, 1.0))
    if why == "a GPU of its own":
        monkeypatch.setattr(lora_partial_load, "shares_host_memory", lambda patcher: False)
    elif why == "not a store model":
        delattr(root, lora_partial_load.STORED)
    elif why == "no backup held":
        held = {}
        patcher.backup.clear()
    with caplog.at_level("INFO"):
        lora_partial_load.check(patcher, None if why == "no latent given" else SAMPLING)
    assert dict(patcher.backup) == held and dict(patcher.patches) == stack and active.patches
    assert slot.base_baked is False and slot.unbake == "a record from an earlier stack"
    assert "freed ComfyUI's weight backup" not in caplog.text


def test_the_ask_is_comfyuis_own_full_estimate_for_the_latent(installed, monkeypatch):
    """The first of estimate_memory's two figures, the doubled batch; no figure when ComfyUI gives none."""
    import comfy.sampler_helpers

    build, _outputs, cpu = _rig()
    _root, patcher, _active, _slot_ = _whole(build, cpu)
    asked = []

    def estimate(model, noise_shape, conds):
        asked.append((model, noise_shape, conds))
        return 34.8 * GIB, 17.4 * GIB

    monkeypatch.setattr(comfy.sampler_helpers, "estimate_memory", estimate)
    assert lora_partial_load._sampling_ask(patcher, *SAMPLING) == int(34.8 * GIB)
    assert asked == [(patcher, *SAMPLING)]

    def fails(model, noise_shape, conds):
        raise RuntimeError("this model has no estimate")

    monkeypatch.setattr(comfy.sampler_helpers, "estimate_memory", fails)
    assert lora_partial_load._sampling_ask(patcher, *SAMPLING) is None


def test_a_backup_freed_on_a_whole_load_is_checked_against_the_model_itself(installed, monkeypatch):
    """The same second wall as after the route: anything short of a full load's state refuses."""
    build, _outputs, cpu = _rig(("int8_tensorwise",))
    root, patcher, _active, slot = _whole(build, cpu)
    _asks(monkeypatch, ask_gib=34.8, room_gib=20.4)
    free = lora_partial_load.free_backup_for_sampling

    def freed_and_then_unpatched(patcher, noise_shape, conds):
        assert free(patcher, noise_shape, conds) is True
        root.current_weight_patches_uuid = None
        return True

    monkeypatch.setattr(lora_partial_load, "free_backup_for_sampling", freed_and_then_unpatched)
    with pytest.raises(lora_partial_load.OffloadedInputQuantError) as raised:
        lora_partial_load.check(patcher, SAMPLING)
    message = str(raised.value)
    _class_c(message)
    assert "ComfyUI loaded whole, and after the Worker freed its weight backup for the sampler" in message
    assert "ComfyUI holds its weights as unpatched" in message and "Reset the attached mesh" in message
    assert slot.base_baked is True


def test_the_wrapper_hands_the_check_the_latent_and_the_conds_of_the_sampling(installed, monkeypatch):
    import comfy.patcher_extension as pe

    from dgx_monarch.actor import partial_load_guard

    build, _outputs, cpu = _rig()
    root, patcher, _active, slot = _whole(build, cpu)
    other_root, other, _other_active, other_slot = _whole(build, cpu)
    seen = _asks(monkeypatch, ask_gib=34.8, room_gib=20.4)
    partial_load_guard.install(patcher, 1)
    (wrapper,) = pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, lora_partial_load.WRAPPER_KEY, patcher.model_options, is_model_options=True)
    shape, conds = SAMPLING
    wrapper(lambda *args, **kwargs: ("real model", "conds", [other]), patcher, shape, conds)
    assert seen == [SAMPLING, SAMPLING]  # the model and the other model the same sampling loaded
    assert slot.base_baked is True and other_slot.base_baked is True
    assert root.model_lowvram is False and other_root.model_lowvram is False
    # Called by keyword, as ComfyUI may: the same two arguments.
    third_root, third, _third_active, third_slot = _whole(build, cpu)
    partial_load_guard.install(third, 1)
    (third_wrapper,) = pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, lora_partial_load.WRAPPER_KEY, third.model_options, is_model_options=True)
    third_wrapper(lambda *args, **kwargs: ("real model", "conds", []), third, noise_shape=shape, conds=conds)
    assert third_slot.base_baked is True and third_root.model_lowvram is False


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


def _class_c(message):
    assert parse_leading_refusal_tag(message) == RefusalTag(
        refusal_class=RefusalClass.CAPACITY, guard="partial_load_divergence", waivable=False)


def test_a_full_load_that_fails_refuses_whatever_the_format(installed, monkeypatch):
    """ComfyUI undoes a load that raises: the model is left unpatched and unloaded.

    A weight-only stack then has nothing "inexact" on it, and the check used to
    let the render go on with no LoRA at all. ComfyUI's own bake is made to run
    out of memory at its second key, in the middle of the route.
    """
    import comfy.model_patcher as comfy_mp
    import torch

    build, outputs, cpu = _rig(("int8_tensorwise", "int8_tensorwise"))
    pristine_root, _unused = build()
    no_lora = outputs(pristine_root)
    root, patcher = build()
    slot = _slot(patcher)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    routed = comfy_mp.ModelPatcher.patch_weight_to_device
    calls = []

    def runs_out_of_memory(self, key, *args, **kwargs):
        calls.append(key)
        if len(calls) == 2:
            raise torch.OutOfMemoryError("out of memory in the middle of the bake")
        return routed(self, key, *args, **kwargs)

    monkeypatch.setattr(comfy_mp.ModelPatcher, "patch_weight_to_device", runs_out_of_memory)
    with pytest.raises(lora_partial_load.OffloadedInputQuantError) as raised:
        lora_partial_load.check(patcher)
    monkeypatch.undo()
    message = str(raised.value)
    _class_c(message)
    for words in ("the full load the Worker ran to bake it whole failed (OutOfMemoryError",
                  "ComfyUI unloads a model whose load fails", "nothing was sampled", "lora_low_rss on"):
        assert words in message, words
    assert isinstance(raised.value.__cause__, torch.OutOfMemoryError)
    assert slot.base_baked is False and slot.unbake == "a record from an earlier stack"
    # What the refusal stops: ComfyUI's rollback left the model with no LoRA in its weights.
    assert all(torch.equal(got, bare) for got, bare in zip(outputs(root), no_lora, strict=True))


@pytest.mark.parametrize("left", ["still partial", "unpatched", "not counted as loaded", "dropped by ComfyUI"])
def test_a_route_that_reports_success_is_checked_against_the_model_itself(installed, monkeypatch, left):
    """The second wall reads the model, not the route: anything short of a full load's state refuses."""
    import comfy.model_management as mm

    build, _outputs, cpu = _rig(("int8_tensorwise",))
    root, patcher = build()
    slot = _slot(patcher)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    mm.current_loaded_models.insert(0, mm.LoadedModel(patcher))
    take = lora_partial_load.make_whole

    def says_it_worked(patcher):
        if left == "still partial":
            return None  # nothing was loaded
        assert take(patcher) is None
        if left == "unpatched":
            root.current_weight_patches_uuid = None
        elif left == "not counted as loaded":
            root.model_loaded_weight_memory = 0
        else:
            mm.current_loaded_models.clear()
        return None

    monkeypatch.setattr(lora_partial_load, "make_whole", says_it_worked)
    with pytest.raises(lora_partial_load.OffloadedInputQuantError) as raised:
        lora_partial_load.check(patcher)
    message = str(raised.value)
    _class_c(message)
    said = {"still partial": "part of it is still off the device with a cast-time patch",
            "unpatched": "ComfyUI holds its weights as unpatched",
            "not counted as loaded": "ComfyUI counts none of it as loaded",
            "dropped by ComfyUI": "ComfyUI no longer holds it as a loaded model"}[left]
    assert said in message and "is not as a full load leaves it" in message and "Reset the attached mesh" in message
    assert slot is not None


def test_the_route_leaves_the_other_models_of_the_sampling_loaded(installed, monkeypatch):
    """A dual-model guider's other checkpoint: ComfyUI's model manager would free it and unpatch its LoRA.

    The route loads the rest of one model through that model's own patcher and
    never calls the manager.
    """
    import comfy.model_management as mm
    import torch

    def never(*args, **kwargs):
        raise AssertionError("the route must not run ComfyUI's model manager: it frees every other model")

    build, outputs, cpu = _rig()
    (expected,) = _full_load(build, outputs, cpu)
    other_root, other = build()
    other.load(cpu, full_load=True)
    mm.current_loaded_models.insert(0, mm.LoadedModel(other))
    root, patcher = build()
    slot = _slot(patcher)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    mm.current_loaded_models.insert(0, mm.LoadedModel(patcher))
    monkeypatch.setattr(mm, "load_models_gpu", never)
    monkeypatch.setattr(mm, "free_memory", never)
    lora_partial_load.check(patcher)
    assert root.model_lowvram is False and slot.base_baked is True
    assert torch.equal(outputs(root)[0], expected)
    assert [loaded.model for loaded in mm.current_loaded_models] == [patcher, other]  # both still held
    assert other_root.current_weight_patches_uuid is not None and other.patches  # and the other still patched
    assert torch.equal(outputs(other_root)[0], expected)


@pytest.mark.parametrize("made_whole", [True, False], ids=["the route taken", "the route not taken"])
def test_the_stores_drop_and_the_swap_credit_read_what_the_route_left(installed, monkeypatch, made_whole):
    """The store's copy-free drop (actor/resident_ledger.py) and the swap credit decide by
    ``model_lowvram``: a partly loaded model is detached in place and, as the slot that stays,
    cannot be pinned, while a whole one is pinned and unloaded by ComfyUI's own unload. A model
    the route made whole must read whole to all of them, and one it left alone partly loaded."""
    import comfy.model_management as mm

    from dgx_monarch import mesh_safety
    from dgx_monarch.actor import resident_ledger, worker_capacity

    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.delenv(resident_ledger.COPY_FREE_DROP_DISABLE_ENV, raising=False)
    if not made_whole:
        monkeypatch.setattr(lora_partial_load, "_host_room", lambda: 0)   # no room to bake: the route is not taken
    build, _outputs, cpu = _rig(("int8_tensorwise",))
    root, patcher = build()
    slot = _slot(patcher, patcher.clone())
    slot.base_key, slot.request_key = ("m.safetensors", ()), ("m.safetensors", (), (), "int8")
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    record = mm.LoadedModel(patcher)
    record.real_model = weakref.ref(root)
    record.model_finalizer = weakref.finalize(root, mm.cleanup_models)
    record.model_finalizer.atexit = False
    mm.current_loaded_models.insert(0, record)
    assert root.model_lowvram is True and not resident_ledger.stays_on_device(slot)
    lora_partial_load.check(patcher)
    assert root.model_lowvram is not made_whole and slot.base_baked is made_whole

    # What the Worker's quote reports for the slot, and whether it could be pinned as the slot that stays.
    partials: dict = {}
    drops: dict = {}
    worker_capacity._resident_facts(None, SimpleNamespace(current=slot, uncond=None), {}, None, partials, drops)
    assert partials["cond"] is not made_whole
    assert drops["cond"] is True, "its own drop copies nothing either way"
    assert resident_ledger.stays_on_device(slot) is made_whole
    assert resident_ledger.drop_copies_nothing(_Slot(None, None), slot) is made_whole, (
        "a drop beside it is credited only where it can be pinned")

    # The drop itself: ComfyUI's own unload for a whole load, the bare detach for a partly loaded one.
    steps: list = []
    unload, detach = record.model_unload, patcher.detach
    record.model_unload = lambda *a, **k: steps.append("model_unload") or unload(*a, **k)
    patcher.detach = lambda unpatch_all=True: steps.append(("detach", unpatch_all)) or detach(unpatch_all)
    assert resident_ledger.unload_dropped(mm, slot, None) is True
    assert steps == (["model_unload", ("detach", True)] if made_whole else [("detach", False)])
    assert mm.current_loaded_models == []


def test_the_check_runs_on_every_model_prepare_sampling_loaded(installed):
    """A later load can leave an earlier model partly loaded after that model's own check has run."""
    import comfy.patcher_extension as pe
    import torch

    from dgx_monarch.actor import partial_load_guard

    build, outputs, cpu = _rig()
    (expected,) = _full_load(build, outputs, cpu)
    root, patcher = build()
    other_root, other = build()
    slots = [_slot(patcher), _slot(other)]
    partial_load_guard.install(patcher, 1)
    (wrapper,) = pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, lora_partial_load.WRAPPER_KEY, patcher.model_options, is_model_options=True)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    other.load(cpu, lowvram_model_memory=1, full_load=False)
    wrapper(lambda *args, **kwargs: ("real model", "conds", [other, patcher]), patcher, (1, 4, 8, 8), {})
    for model_root, slot in zip((root, other_root), slots, strict=True):
        assert model_root.model_lowvram is False and slot.base_baked is True
        assert torch.equal(outputs(model_root)[0], expected)


def test_the_room_rule_counts_every_moved_layer_when_the_weights_are_file_pages(installed, monkeypatch):
    """Under the mmap loader the host copies are page cache, so moving a layer onto the device takes new memory."""
    from dgx_monarch.actor import lora_cast_time

    build, _outputs, cpu = _rig(("int8_tensorwise", "float8_e4m3fn"))
    root, patcher = build()
    slot = _slot(patcher)
    patcher.patches.pop("diffusion_model.q0.weight")  # one layer off the device carries no patch
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    patched = lora_cast_time._module_bytes(root.diffusion_model.q1)
    off_device = int(patcher.model_size()) - int(patcher.loaded_size())
    assert off_device > patched
    pending = lora_partial_load._patched_off_device(patcher)
    monkeypatch.setattr(lora_partial_load, "_host_copies_are_file_pages", lambda: False)
    assert lora_partial_load._room_needed(patcher, pending) == patched  # pread: only ComfyUI's backups
    monkeypatch.setattr(lora_partial_load, "_host_copies_are_file_pages", lambda: True)
    assert lora_partial_load._room_needed(patcher, pending) == patched + off_device
    # Room for the backups alone is not enough then, and the route is not taken.
    monkeypatch.setattr(lora_partial_load, "_host_room", lambda: patched)
    why_not = lora_partial_load.make_whole(patcher)
    assert why_not is not None and "is available above the 5 GiB floor" in why_not
    assert root.model_lowvram is True and slot.base_baked is False


def test_a_pending_layer_is_classed_by_its_own_format_for_the_routes_room(installed):
    """The live modules, read as the classic bake's price reads the header: INT8 ConvRot, scaled FP8, the rest."""
    import torch

    from dgx_monarch import capacity_classic_bake as price
    from dgx_monarch.actor import lora_cast_time

    build, _outputs, cpu = _rig(("int8_tensorwise", "float8_e4m3fn"))
    root, patcher = build()
    _slot(patcher)
    int8, fp8 = root.diffusion_model.q0, root.diffusion_model.q1
    assert lora_partial_load._layer_class(int8) == (price.INT8_QUANTIZED, lora_cast_time._module_bytes(int8))
    assert lora_partial_load._layer_class(fp8) == (price.FP8_QUANTIZED, lora_cast_time._module_bytes(fp8))
    plain = torch.nn.Linear(8, 8, bias=False)
    assert lora_partial_load._layer_class(plain) == (price.FULL_PRECISION, 8 * 8 * 4)
    assert lora_partial_load._layer_class(plain.to(torch.bfloat16)) == (price.FULL_PRECISION, 8 * 8 * 2)
    assert lora_partial_load._layer_class(SimpleNamespace()) == (price.OTHER, 0)  # unreadable: the line
    # Both layers off the device: one INT8 share and one share of the line beside the fixed part.
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    pending = lora_partial_load._patched_off_device(patcher)
    assert len(pending) == 2
    assert lora_partial_load._bake_room(pending) == (
        price.CLASSIC_BAKE_FIXED_BYTES + price.LIGHT_SHARE_BYTES[price.INT8_QUANTIZED]
        + price.CLASSIC_BAKE_BYTES_PER_WEIGHT)


def test_the_pread_loader_is_told_from_the_mmap_loader_by_its_own_mark(monkeypatch):
    import safetensors

    monkeypatch.delattr(safetensors.safe_open, "_dgxm_pread", raising=False)
    assert lora_partial_load._host_copies_are_file_pages() is True
    marked = lambda *args, **kwargs: None  # noqa: E731
    marked._dgxm_pread = True
    monkeypatch.setattr(safetensors, "safe_open", marked)
    assert lora_partial_load._host_copies_are_file_pages() is False
