"""Per-block attention pins: detection, the render record and the class-K waiver.

ComfyUI lets a checkpoint, or a LoRA, pin an attention kernel per block through
a ``ComfyAttention`` child named ``comfy_attention`` (issue #30). Stand-in
modules here carry the two attributes the pack reads, ``config`` and
``function``; canary seam 39 holds the same reads against the real ComfyUI
tree, a real ``.config`` tensor and ComfyUI's own LoRA patcher.
"""
from __future__ import annotations

import json
import types

import pytest
import torch

from attention_pin_helpers import Attention, Block, int8, make_patcher, sol
from dgx_monarch import accuracy_waiver, attention_pins, consent_descriptor, consent_pending
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

SOL_KIND = "waive-known-wrong:sol-attn"


def _worker(**topology):
    full = {"ulysses": 1, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False, **topology}
    return types.SimpleNamespace(topology=full)


@pytest.fixture(autouse=True)
def _clean_waiver_scope():
    accuracy_waiver.clear()
    yield
    accuracy_waiver.clear()


def test_detection_counts_each_kernel_on_a_nested_model():
    patcher = make_patcher([
        Block(sol(), int8()),
        Block(sol(), Attention()),  # an unpinned cross-attention
        Block(Attention({"attention": "not_a_kernel"}), None),  # ComfyUI ignores it
        Block(sol(tau=1.0), None),
    ])
    pins = attention_pins.scan_loaded(patcher)
    assert sorted(pins) == [
        "diffusion_model.blocks.0.cross_attn.comfy_attention",
        "diffusion_model.blocks.0.self_attn.comfy_attention",
        "diffusion_model.blocks.1.self_attn.comfy_attention",
        "diffusion_model.blocks.3.self_attn.comfy_attention",
    ]
    summary = attention_pins.summarize([patcher], sequence_parallel=False)
    assert summary is not None
    assert summary.kernels == {"comfy_kitchen_sol tau=0.7": 2, "comfy_kitchen_int8": 1,
                               "comfy_kitchen_sol tau=1": 1}
    assert summary.ran and summary.running == summary.kernels


def test_an_unbound_list_config_names_the_kernel_comfyui_tried_first():
    """ComfyUI binds the first kernel it can run from a list; none bound here.

    The record says this host ran no pin. A fleet still refuses, because the
    verdict must not read the binding: under cfg2+fsdp and dp2+fsdp no
    readiness exchange runs, and a peer that bound the kernel would refuse
    alone and leave this rank waiting at the first collective. A rank that
    renders alone has no peer, so it asks nothing for a pin it cannot run.
    """
    module = Attention([{"attention": "made_up"}, {"attention": "comfy_kitchen_int8"},
                         {"attention": "comfy_kitchen_sol", "tau": 0.6}])
    pin = attention_pins.block_pin(module)
    assert pin == attention_pins.BlockPin("comfy_kitchen_int8", None, bound=False)
    summary = attention_pins.summarize([make_patcher([Block(module)])], sequence_parallel=False)
    assert summary is not None and not summary.ran
    assert summary.public()["attention_pins_dropped"] == {
        attention_pins.DROPPED_UNBOUND: 1}
    assert summary.running == {} and summary.live == {"comfy_kitchen_int8": 1}
    with pytest.raises(UnsupportedModelError, match="comfy_kitchen_int8 on 1 attention module"):
        attention_pins.assert_waived(summary, {"cfg": 2, "fsdp": True})
    attention_pins.assert_waived(summary, {}, alone=True)  # no raise


def test_a_tau_that_is_not_a_number_never_fails_the_read():
    """ComfyUI binds whatever the JSON holds; the read must not raise on it."""
    import functools

    from attention_pin_helpers import attention_comfy_kitchen_sol

    module = Attention({"attention": "comfy_kitchen_sol", "tau": "auto"},
                       functools.partial(attention_comfy_kitchen_sol, tau="auto"))
    assert attention_pins.block_pin(module) == attention_pins.BlockPin(
        "comfy_kitchen_sol", None, bound=True)
    unbound = Attention({"attention": "comfy_kitchen_sol", "tau": [1]})
    assert attention_pins.block_pin(unbound) == attention_pins.BlockPin(
        "comfy_kitchen_sol", None, bound=False)
    patcher = make_patcher([Block(module)])
    attention_pins.scan_loaded(patcher)
    for topology in ({"ulysses": 2}, {"cfg": 2}):
        summary = attention_pins.render_summary(_worker(**topology), [patcher])
        assert summary is not None and summary.kernels == {"comfy_kitchen_sol": 1}


def test_a_bound_kernel_the_pack_does_not_know_counts_as_approximate():
    def some_future_kernel(*args, **kwargs):
        raise AssertionError("not run")

    module = Attention({"attention": "comfy_kitchen_fp4"}, some_future_kernel)
    summary = attention_pins.summarize([make_patcher([Block(module)])], sequence_parallel=False)
    assert summary is not None and summary.approximate_live == {"comfy_kitchen_fp4": 1}


def test_a_lora_pin_is_detected_from_the_patchers_object_patches():
    # The live module stays unpinned: only the staged object patch pins it,
    # as ComfyUI's add_patches leaves it until patch_model runs.
    patcher = make_patcher(
        [Block(Attention()), Block(sol())],
        object_patches={
            "diffusion_model.blocks.0.self_attn.comfy_attention": int8(),
            # A LoRA can also unpin a block the checkpoint pinned.
            "diffusion_model.blocks.1.self_attn.comfy_attention":
                Attention({"attention": "not_a_kernel"}),
            "model_sampling": object(),
        })
    attention_pins.scan_loaded(types.SimpleNamespace(model=patcher.model))
    summary = attention_pins.summarize([patcher], sequence_parallel=False)
    assert summary is not None and summary.kernels == {"comfy_kitchen_int8": 1}


def test_a_lora_pin_on_a_clone_reads_the_load_scan_its_base_cached():
    base = make_patcher([Block(sol())])
    attention_pins.scan_loaded(base)
    clone = types.SimpleNamespace(model=base.model, object_patches={
        "diffusion_model.blocks.0.self_attn.comfy_attention": int8()})
    summary = attention_pins.summarize([clone], sequence_parallel=False)
    assert summary is not None and summary.kernels == {"comfy_kitchen_int8": 1}
    assert base.model.diffusion_model.walks == 1


def test_a_model_loaded_without_the_scan_reads_the_backup_not_a_staged_lora():
    # patch_model swapped the LoRA's module in and kept the checkpoint's in the backup.
    patcher = make_patcher([Block(int8())])
    path = "diffusion_model.blocks.0.self_attn.comfy_attention"
    patcher.object_patches_backup = {path: Attention()}
    assert attention_pins.effective_pins(patcher) == {}
    patcher.object_patches[path] = int8()
    assert set(attention_pins.effective_pins(patcher)) == {path}


@pytest.mark.parametrize("topology", [{"ulysses": 2}, {"ring": 2}, {"ulysses": 2, "cfg": 2},
                                      {"ulysses": 2, "fsdp": True}])
def test_sequence_parallelism_drops_every_pin_and_asks_for_nothing(topology):
    patcher = make_patcher([Block(sol(), int8()), Block(sol(), None)])
    summary = attention_pins.render_summary(_worker(**topology), [patcher])
    assert summary is not None
    assert summary.public() == {
        "attention_pins": {"comfy_kitchen_sol tau=0.7": 2, "comfy_kitchen_int8": 1},
        "attention_pins_ran": False,
        "attention_pins_dropped": {attention_pins.DROPPED_SEQUENCE_PARALLEL: 3},
    }
    attention_pins.assert_waived(summary, _worker(**topology).topology)  # no raise


def _refusal(make, topology):
    summary = attention_pins.render_summary(
        _worker(**topology), [make_patcher([Block(make()), Block(make())])])
    assert summary is not None and summary.ran
    with pytest.raises(UnsupportedModelError) as caught:
        attention_pins.assert_waived(summary, _worker(**topology).topology)
    return summary, str(caught.value)


@pytest.mark.parametrize("topology,label", [({"cfg": 2}, "cfg2"), ({"dp": 2}, "dp2"),
                                            ({}, "single"), ({"cfg": 2, "fsdp": True},
                                                             "cfg2+fsdp")])
@pytest.mark.parametrize("kernel", ["sol", "int8"])
def test_without_sequence_parallelism_an_approximate_pin_runs_and_refuses_class_k(
        topology, label, kernel):
    summary, message = _refusal(sol if kernel == "sol" else int8, topology)
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == attention_pins.SOL_ATTN_PIN_GUARD and tag.waivable
    name = "comfy_kitchen_sol tau=0.7" if kernel == "sol" else "comfy_kitchen_int8"
    assert summary.kernels == {name: 2}
    # The message names the kernel, the module count, the topology and both remedies.
    assert f"{name} on 2 attention modules, set by the checkpoint" in message
    assert f"Topology {label} runs no sequence parallelism" in message
    assert "known-wrong math by construction" in message
    assert "no fidelity measurement" not in message
    assert "grant the Sol-Attn accuracy waiver" in message
    assert "pick a sequence-parallel topology such as uly2" in message
    assert "the Init widget's kernel runs (exact under TORCH_FLASH)" in message
    assert "DGXM_WAIVE_KNOWN_WRONG=sol_attn" in message
    assert "docs/TROUBLESHOOTING.md #117" in message


def test_an_exact_pinned_kernel_is_recorded_and_asks_nothing(monkeypatch):
    monkeypatch.setattr(attention_pins, "EXACT_KERNELS", frozenset({"comfy_kitchen_int8"}))
    summary = attention_pins.render_summary(_worker(cfg=2), [make_patcher([Block(int8())])])
    assert summary is not None and summary.ran
    assert summary.public()["attention_pins"] == {"comfy_kitchen_int8": 1}
    attention_pins.assert_waived(summary, {"cfg": 2})  # no raise
    # A Sol pin beside it still asks.
    mixed = attention_pins.render_summary(
        _worker(cfg=2), [make_patcher([Block(int8()), Block(sol())])])
    with pytest.raises(UnsupportedModelError):
        attention_pins.assert_waived(mixed, {"cfg": 2})


def test_a_granted_sol_waiver_lets_the_pin_run_and_stamps_once():
    accuracy_waiver.activate(None)
    accuracy_waiver._scope().grants[SOL_KIND] = {
        "id": "a" * 32, "kind": SOL_KIND, "consent_source": "panel"}
    summary = attention_pins.render_summary(_worker(cfg=2), [make_patcher([Block(sol())])])
    attention_pins.assert_waived(summary, {"cfg": 2})
    attention_pins.assert_waived(summary, {"cfg": 2})
    stamps = accuracy_waiver.stamps()
    assert [stamp["guard"] for stamp in stamps] == [attention_pins.SOL_ATTN_PIN_GUARD]
    assert stamps[0]["stamp"].startswith("rendered-under-waiver: waive-known-wrong:sol-attn")


def test_the_card_names_the_pin_its_module_count_and_the_sequence_parallel_alternative():
    accuracy_waiver.bind_model("pinned.safetensors", {}, [], {"cfg": 2}, 2)
    summary = attention_pins.render_summary(_worker(cfg=2), [make_patcher([Block(sol()) for _ in range(3)])])
    with pytest.raises(UnsupportedModelError) as caught:
        attention_pins.assert_waived(summary, {"cfg": 2}, family="wan")
    descriptor = consent_descriptor.parse(str(caught.value))
    assert descriptor is not None and descriptor.kind == SOL_KIND
    assert descriptor.measured["probe"] == attention_pins.SOL_ATTN_PIN_GUARD
    detail = descriptor.measured["detail"]
    assert detail.startswith("pinned by the checkpoint or a LoRA in its stack: ")
    assert "comfy_kitchen_sol tau=0.7 on 3 attention modules" in detail
    assert "runs on cfg2" in detail
    assert "a sequence-parallel topology drops the pin instead" in detail
    # The card shows it where a capacity card shows its figures.
    assert consent_pending.numbers_from_measured(descriptor.measured) == detail
    spec = consent_pending.KIND_SPECS[SOL_KIND]
    assert spec.title == "Render {artifact} with an approximate attention kernel"


def test_the_pin_card_states_no_measurement_the_widget_route_took():
    """The kind's Sol measurement is the widget's; a pinned kernel was never measured."""
    spec = consent_pending.KIND_SPECS[SOL_KIND]
    pin_text = spec.measured_for(attention_pins.SOL_ATTN_PIN_GUARD)
    assert pin_text is not None and pin_text != spec.measured
    assert "approximate by construction" in pin_text and "measured 20" not in pin_text
    assert spec.measured_for("sol_attn:minimax_h3") == spec.measured
    accuracy_waiver.bind_model("pinned.safetensors", {}, [], {"cfg": 2}, 2)
    summary = attention_pins.render_summary(_worker(cfg=2), [make_patcher([Block(int8())])])
    with pytest.raises(UnsupportedModelError) as caught:
        attention_pins.assert_waived(summary, {"cfg": 2})
    descriptor = consent_descriptor.parse(str(caught.value))
    assert descriptor is not None
    assert descriptor.human_reason.endswith(f"Measured: {pin_text}.")
    assert descriptor.measured["measured"] == pin_text
    # The driver's card reads the same text through the descriptor's guard.
    card = consent_pending._card(consent_pending.PendingConsent(
        id="p-1", key="k", descriptor=consent_pending.descriptor_from_refusal(
            descriptor, path="/models/pinned.safetensors", combo_key="c", artifacts="a",
            memo_context=dict(descriptor.memo_context))))
    assert card["measured"] == pin_text


def test_a_dispatch_starts_with_no_bound_checkpoint():
    """A card must never name a checkpoint an earlier dispatch bound on this thread."""
    accuracy_waiver.bind_model("earlier.safetensors", {}, [], {"cfg": 2}, 2)
    accuracy_waiver.activate(None)
    summary = attention_pins.render_summary(_worker(cfg=2), [make_patcher([Block(sol())])])
    with pytest.raises(UnsupportedModelError) as caught:
        attention_pins.assert_waived(summary, {"cfg": 2})
    assert consent_descriptor.parse(str(caught.value)) is None
    assert "earlier.safetensors" not in str(caught.value)


def test_the_bf16_fsdp_shard_build_clears_a_checkpoint_pin_and_the_record_says_so():
    patcher = make_patcher([Block(sol()), Block(sol())])
    attention_pins.scan_loaded(patcher)
    # fsdp_shard_build reloads through load_state_dict without the config key,
    # and ComfyAttention clears config and function on that load.
    for block in patcher.model.diffusion_model.blocks:
        block.self_attn.comfy_attention.config = None
        block.self_attn.comfy_attention.function = None
    attention_pins.note_injected(patcher)
    summary = attention_pins.render_summary(_worker(cfg=2, fsdp=True), [patcher])
    assert summary is not None and summary.public()["attention_pins_dropped"] == {
        attention_pins.DROPPED_FSDP: 2}
    attention_pins.assert_waived(summary, {"cfg": 2, "fsdp": True})  # nothing runs


def test_no_pin_costs_one_walk_per_load_and_nothing_per_render():
    patcher = make_patcher([Block(Attention(), Attention()) for _ in range(4)])
    assert attention_pins.scan_loaded(patcher) == {}
    attention_pins.note_injected(patcher)
    for _ in range(5):
        assert attention_pins.render_summary(_worker(cfg=2), [patcher, None]) is None
    assert patcher.model.diffusion_model.walks == 1
    # A pinned load walks once too, however many renders read it.
    pinned = make_patcher([Block(sol())])
    attention_pins.scan_loaded(pinned)
    for _ in range(5):
        attention_pins.render_summary(_worker(cfg=2), [pinned])
    assert pinned.model.diffusion_model.walks == 1


def test_ranks_merge_into_the_record_and_describe_reads_it():
    ran = {"attention_pins": {"attention_pins": {"comfy_kitchen_sol tau=0.7": 40},
                              "attention_pins_ran": True, "attention_pins_dropped": None}}
    record = attention_pins.merge([ran, ran])
    assert record == {"attention_pins": {"comfy_kitchen_sol tau=0.7": 40},
                      "attention_pins_ran": True, "attention_pins_dropped": None}
    assert attention_pins.describe(record) == \
        "pinned: comfy_kitchen_sol tau=0.7 x 40 modules, ran"
    dropped = {"attention_pins": {"attention_pins": {"comfy_kitchen_int8": 30},
                                  "attention_pins_ran": False,
                                  "attention_pins_dropped": {"sequence parallelism": 30}}}
    assert attention_pins.describe(attention_pins.merge([dropped, dropped])) == \
        "pinned: comfy_kitchen_int8 x 30 modules, dropped (sequence parallelism)"
    partial = {"attention_pins": {"attention_pins": {"comfy_kitchen_int8": 30},
                                  "attention_pins_ran": False,
                                  "attention_pins_dropped": {"FSDP shard build": 20}}}
    assert attention_pins.describe(attention_pins.merge([partial])) == \
        "pinned: comfy_kitchen_int8 x 30 modules, ran on 10, dropped on 20 (FSDP shard build)"
    # Dual-model cfg2: one rank holds a checkpoint with no pins.
    assert attention_pins.merge([ran, {"rank": 1}]) == record
    assert attention_pins.merge([{"rank": 0}, {"attention_pins": "junk"}]) is None
    assert attention_pins.describe(None) == ""


def test_dual_model_ranks_keep_each_models_counts_together():
    """Under dual-model cfg2 each cfg rank renders its own checkpoint.

    The cond rank ran 20 pins; the uncond rank pinned 10 that its host could
    not bind. The record sums the two models, as one rank holding both does
    (summarize), keeps each model's own fields, and the line names each.
    """
    sol_label = "comfy_kitchen_sol tau=0.7"
    cond = {"rank": 0, "attention_pins": {
        "attention_pins": {sol_label: 20}, "attention_pins_ran": True,
        "attention_pins_dropped": None, "attention_pins_model": "cond"}}
    uncond = {"rank": 1, "attention_pins": {
        "attention_pins": {sol_label: 10}, "attention_pins_ran": False,
        "attention_pins_dropped": {"kernel unavailable on this host": 10},
        "attention_pins_model": "uncond"}}
    record = attention_pins.merge([cond, uncond])
    assert record == {
        "attention_pins": {sol_label: 30}, "attention_pins_ran": False,
        "attention_pins_dropped": {"kernel unavailable on this host": 10},
        "attention_pins_models": {
            "cond": {"attention_pins": {sol_label: 20}, "attention_pins_ran": True,
                     "attention_pins_dropped": None},
            "uncond": {"attention_pins": {sol_label: 10}, "attention_pins_ran": False,
                       "attention_pins_dropped": {"kernel unavailable on this host": 10}}}}
    assert attention_pins.describe(record) == (
        "pinned: cond model: comfy_kitchen_sol tau=0.7 x 20 modules, ran; "
        "uncond model: comfy_kitchen_sol tau=0.7 x 10 modules, dropped "
        "(kernel unavailable on this host)")
    assert attention_pins.merge([uncond, cond]) == record
    # Both ran: 30 modules ran, never the larger rank's 20.
    both = attention_pins.merge([cond, {"attention_pins": {
        "attention_pins": {"comfy_kitchen_int8": 10, sol_label: 4}, "attention_pins_ran": True,
        "attention_pins_dropped": None, "attention_pins_model": "uncond"}}])
    assert both["attention_pins"] == {sol_label: 24, "comfy_kitchen_int8": 10}
    assert both["attention_pins_ran"] is True and both["attention_pins_dropped"] is None
    assert attention_pins.describe(both) == (
        "pinned: cond model: comfy_kitchen_sol tau=0.7 x 20 modules, ran; uncond model: "
        "comfy_kitchen_int8 x 10 modules, comfy_kitchen_sol tau=0.7 x 4 modules, ran")
    # Only one model pins: the plain line, its own counts.
    assert attention_pins.describe(attention_pins.merge([{"rank": 0}, uncond])) == (
        "pinned: comfy_kitchen_sol tau=0.7 x 10 modules, dropped "
        "(kernel unavailable on this host)")


def test_ranks_of_one_model_still_write_one_line():
    """cfg2 or dp2 on one checkpoint: every rank holds the same modules."""
    ran = {"attention_pins": {"attention_pins": {"comfy_kitchen_sol tau=0.7": 20},
                              "attention_pins_ran": True, "attention_pins_dropped": None}}
    unbound = {"attention_pins": {"attention_pins": {"comfy_kitchen_sol tau=0.7": 20},
                                  "attention_pins_ran": False,
                                  "attention_pins_dropped": {"kernel unavailable on this host": 20}}}
    assert attention_pins.describe(attention_pins.merge([ran, ran])) == \
        "pinned: comfy_kitchen_sol tau=0.7 x 20 modules, ran"
    # One host could not bind the kernel: the model's 20 modules, never 40,
    # and ran stays false because a rank did not run them.
    record = attention_pins.merge([ran, unbound])
    assert record == {"attention_pins": {"comfy_kitchen_sol tau=0.7": 20},
                      "attention_pins_ran": False,
                      "attention_pins_dropped": {"kernel unavailable on this host": 20}}
    assert attention_pins.merge([unbound, ran]) == record
    assert attention_pins.describe(record) == ("pinned: comfy_kitchen_sol tau=0.7 x 20 modules, "
                                               "dropped (kernel unavailable on this host)")
    # cfg2+dp2 under dual-model cfg2: two ranks per slot fold within the slot.
    cond = {"attention_pins": {**ran["attention_pins"], "attention_pins_model": "cond"}}
    assert attention_pins.merge([cond, cond])["attention_pins"] == {"comfy_kitchen_sol tau=0.7": 20}
    assert "attention_pins_models" not in attention_pins.merge([cond, cond])


def test_the_finish_record_carries_the_pins_beside_the_attention_fact():
    from dgx_monarch.telemetry import RenderProgress

    progress = RenderProgress()
    token = progress.start(4, {"attention": "TORCH_FLASH", "topology": "cfg2"}, token=object())
    record = {"attention_pins": {"comfy_kitchen_sol tau=0.7": 40}, "attention_pins_ran": True,
              "attention_pins_dropped": None}
    progress.note_pins(record, token=token)
    progress.finish(token)
    last = progress.snapshot()["last"]
    assert last["attention"] == "TORCH_FLASH"
    assert {field: last[field] for field in attention_pins.RECORD_FIELDS} == record
    # The next render without pins reports none rather than inheriting these.
    token = progress.start(4, {"attention": "TORCH_FLASH"}, token=object())
    progress.finish(token)
    assert progress.snapshot()["last"]["attention_pins"] is None


def test_an_overlapping_render_cannot_lend_its_pins():
    from dgx_monarch.telemetry import RenderProgress

    progress = RenderProgress()
    first = progress.start(4, {"model": "a"}, token=object())
    second = progress.start(4, {"model": "b"}, token=object())
    progress.note_pins({"attention_pins": {"comfy_kitchen_int8": 1},
                        "attention_pins_ran": True}, token=first)
    progress.finish(first)
    progress.finish(second)
    # The record carries the last start's facts (render b), which noted no pins.
    assert progress.snapshot()["last"]["attention_pins"] is None


def test_a_staged_lora_keeps_its_attention_config_on_the_host(monkeypatch):
    """ComfyUI decodes the config with Tensor.numpy(), which a device tensor refuses."""
    import sys

    from dgx_monarch.actor import model_store

    seen = {}
    config = torch.tensor(list(json.dumps({"attention": "comfy_kitchen_int8"}).encode()),
                          dtype=torch.uint8)
    lora = {"diffusion_model.blocks.0.attn.comfy_attention.config": config,
            "diffusion_model.blocks.0.attn.q.lora_up.weight": torch.ones(2, 2)}
    sd = types.ModuleType("comfy.sd")

    def load_lora_for_models(active, clip, lora_sd, strength, clip_strength):
        seen.update(lora_sd)
        return active, None

    sd.load_lora_for_models = load_lora_for_models
    utils = types.ModuleType("comfy.utils")
    utils.load_torch_file = lambda path, safe_load=True: dict(lora)
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device("meta")
    comfy = types.ModuleType("comfy")
    comfy.sd, comfy.utils, comfy.model_management = sd, utils, mm
    for name, module in [("comfy", comfy), ("comfy.sd", sd), ("comfy.utils", utils),
                         ("comfy.model_management", mm)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(model_store, "resolve_model_path", lambda folder, name: name)
    store = model_store.ModelStore()
    store.lora_low_rss = True
    base = types.SimpleNamespace(clone=lambda: types.SimpleNamespace())
    store._build_active(base, [{"name": "pin.safetensors", "strength": 1.0}])
    assert seen["diffusion_model.blocks.0.attn.comfy_attention.config"].device.type == "cpu"
    assert seen["diffusion_model.blocks.0.attn.q.lora_up.weight"].device.type == "meta"


def test_a_slab_keeps_the_attention_config_readable_on_the_host():
    from dgx_monarch.actor.slab import _slab_tensor_uses_cuda

    assert not _slab_tensor_uses_cuda("blocks.0.self_attn.comfy_attention.config")
    assert not _slab_tensor_uses_cuda("blocks.0.self_attn.q.comfy_quant")
    assert _slab_tensor_uses_cuda("blocks.0.self_attn.q.weight")


def test_an_aborted_ceremony_on_the_pin_guard_leaves_residency_alone():
    from dgx_monarch.adapters.sol_attention import is_sol_waiver_refusal

    summary = attention_pins.render_summary(_worker(cfg=2), [make_patcher([Block(sol())])])
    with pytest.raises(UnsupportedModelError) as caught:
        attention_pins.assert_waived(summary, {"cfg": 2})
    assert is_sol_waiver_refusal(caught.value)


class _Flux3Block(torch.nn.Module):
    """A FLUX.3 block owns its attention: the pin sits at <block>.comfy_attention."""

    def __init__(self, attention):
        super().__init__()
        self.comfy_attention = attention
        self.qkv = torch.nn.Linear(2, 6)


def _flux3_patcher():
    """The int8 files' layout: every image and single block pinned, the early
    text and reference blocks not (comfy/ldm/flux3/model.py module names)."""
    dm = torch.nn.Module()
    dm.content_mode_blocks = torch.nn.ModuleDict({
        "image": torch.nn.ModuleList([_Flux3Block(int8())]),
        "image_cond": torch.nn.ModuleList([_Flux3Block(Attention())])})
    dm.txt_mode_blocks = torch.nn.ModuleList([_Flux3Block(Attention())])
    dm.single_blocks = torch.nn.ModuleList([_Flux3Block(int8()), _Flux3Block(int8())])
    return types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dm),
                                 object_patches={})


def test_a_flux3_int8_layout_reads_its_block_level_pins():
    found = attention_pins.scan_loaded(_flux3_patcher())
    assert sorted(found) == [
        "diffusion_model.content_mode_blocks.image.0.comfy_attention",
        "diffusion_model.single_blocks.0.comfy_attention",
        "diffusion_model.single_blocks.1.comfy_attention",
    ]


@pytest.mark.parametrize("topology,label", [({"cfg": 2}, "cfg2"), ({"dp": 2}, "dp2"),
                                            ({}, "single"),
                                            ({"cfg": 2, "fsdp": True}, "cfg2+fsdp"),
                                            ({"dp": 2, "fsdp": True}, "dp2+fsdp")])
def test_a_flux3_int8_layout_off_sequence_parallelism_refuses_class_k(topology, label):
    summary = attention_pins.render_summary(_worker(**topology), [_flux3_patcher()])
    assert summary is not None and summary.ran
    assert summary.kernels == {"comfy_kitchen_int8": 3}
    with pytest.raises(UnsupportedModelError) as caught:
        attention_pins.assert_waived(summary, _worker(**topology).topology, family="flux3")
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == attention_pins.SOL_ATTN_PIN_GUARD and tag.waivable
    assert f"Topology {label} runs no sequence parallelism" in str(caught.value)


def test_a_flux3_int8_layout_under_ulysses_drops_its_pins_and_asks_nothing():
    summary = attention_pins.render_summary(_worker(ulysses=2), [_flux3_patcher()])
    assert summary is not None and not summary.ran
    assert summary.public()["attention_pins_dropped"] == {
        attention_pins.DROPPED_SEQUENCE_PARALLEL: 3}
    attention_pins.assert_waived(summary, _worker(ulysses=2).topology, family="flux3")
