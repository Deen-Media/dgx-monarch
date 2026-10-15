"""Where the pack reads a model's per-block attention pins (issue #30).

The load path scans once per fresh load, before injection, then marks what
injection cleared; the sample path reads that scan with a LoRA's staged pins
added, refuses an approximate pin that would run before any sampling, and
reports the pins in the rank's result row. The refusal sits inside the load
guard, so on a topology with a readiness exchange a rank that refuses still
crosses it and its peer refuses with it instead of waiting at the first
collective. Under FSDP no exchange runs, so every rank reaches the same
verdict from the pins alone.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from attention_pin_helpers import Attention, Block, int8, sol
from dgx_monarch import accuracy_waiver, attention_pins, consent_descriptor
from dgx_monarch.actor import partial_load_guard, sample_protocol
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.

SOL_KIND = "waive-known-wrong:sol-attn"


@pytest.fixture(autouse=True)
def _clean_waiver_scope():
    accuracy_waiver.clear()
    yield
    accuracy_waiver.clear()


def _pinned_patcher(*blocks, object_patches=None):
    model = types.SimpleNamespace(diffusion_model=torch.nn.ModuleDict(
        {"blocks": torch.nn.ModuleList(blocks)}))
    base = types.SimpleNamespace(model=model, object_patches={})
    attention_pins.scan_loaded(base)  # what fresh_load does for every load
    return types.SimpleNamespace(model=model, object_patches=dict(object_patches or {}))


def _worker(*, world=2, **topology):
    full = {"ulysses": 1, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False, **topology}
    return types.SimpleNamespace(
        _setup_key=object(), topology=full, world=world, rank=0,
        store=types.SimpleNamespace(family_override=None,
                                    current=types.SimpleNamespace(family="wan")),
        _attn=None, _inject_for_topology=object(), _check_uma_reserve=lambda: None)


def _run(monkeypatch, worker, patcher, *, crossed, extra=None, activate=False):
    import dgx_monarch.adapters as adapters_pkg
    from dgx_monarch import adoption_evidence
    from dgx_monarch.actor import sampling, worker_env

    monkeypatch.setattr(worker_env, "verify_sample_artifact_authorization",
                        lambda w, r, fn: [{"id": "cond"}, {"id": "uncond"}])
    if not activate:
        monkeypatch.setattr(worker_env, "activate_accuracy_waivers", lambda r: None)
    monkeypatch.setattr(worker_env, "assert_resident_artifact_identity",
                        lambda w, s, e: None)
    monkeypatch.setattr(worker_env, "sample_rescue_consent", lambda r: {})
    monkeypatch.setattr(adapters_pkg, "adapter_for",
                        lambda model, override=None: types.SimpleNamespace(
                            cfg_cond_padding="none", dp_cond_exempt_keys=frozenset()))
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(adoption_evidence, "build_worker_evidence", lambda *a, **k: None)
    monkeypatch.setattr(sample_protocol.store_fsdp, "ensure", lambda *a, **k: (patcher, "load"))
    monkeypatch.setattr(partial_load_guard, "all_ranks_true",
                        lambda flag: crossed.append(flag) or flag)
    sampled = []

    def run_ksampler(*args, **kwargs):
        sampled.append(1)
        return torch.zeros(1, 2), None

    result = sample_protocol.run_sample(
        worker, {"kind": "ksampler", "model": {"unet_name": "pinned.safetensors"},
                 "positive": None, "negative": None,
                 "latent": {"samples": torch.zeros(2, 4, 8, 8)}, **(extra or {})},
        None, None,
        equalize_cond_lengths=lambda *a: (None, None),
        model_sampling_render_clone=lambda pat, s: pat,
        request_artifact_identity=lambda name, loras: {"id": name},
        run_ksampler=run_ksampler,
        run_custom=lambda *a, **k: (torch.zeros(1, 2), None),
        latent_signature=lambda s: {"sig": 1})
    return result, sampled


@pytest.mark.parametrize("topology", [{"cfg": 2}, {"dp": 2}])
def test_an_approximate_pin_refuses_before_sampling_and_the_peer_is_told(
        monkeypatch, topology):
    crossed: list[bool] = []
    with pytest.raises(UnsupportedModelError) as caught:
        _run(monkeypatch, _worker(**topology), _pinned_patcher(Block(sol())), crossed=crossed)
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == attention_pins.SOL_ATTN_PIN_GUARD
    # This rank crossed the readiness flag as not ready, so its peer refuses
    # too rather than waiting at the first collective.
    assert crossed == [False]


def test_a_lora_pin_refuses_where_the_checkpoint_pins_nothing(monkeypatch):
    crossed: list[bool] = []
    patcher = _pinned_patcher(Block(Attention()), object_patches={
        "diffusion_model.blocks.0.self_attn.comfy_attention": int8()})
    with pytest.raises(UnsupportedModelError, match="comfy_kitchen_int8 on 1 attention module"):
        _run(monkeypatch, _worker(cfg=2), patcher, crossed=crossed)


def test_under_dual_model_cfg2_the_uncond_rank_card_names_the_conditional_checkpoint(
        monkeypatch):
    """The driver keys every grant of a dual-model render on the conditional
    checkpoint (consent_waiver.stamp_request), so the card the uncond rank
    raises names that checkpoint, and its detail names the one that pins. A
    binding an earlier dispatch left on this thread never reaches the card.
    """
    accuracy_waiver.bind_model("earlier.safetensors", {}, [], {"cfg": 2}, 2)
    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda w, r: "uncond")
    crossed: list[bool] = []
    with pytest.raises(UnsupportedModelError) as caught:
        _run(monkeypatch, _worker(cfg=2), _pinned_patcher(Block(sol())), crossed=crossed,
             extra={"uncond_model": {"unet_name": "uncond.safetensors"}}, activate=True)
    message = str(caught.value)
    assert "earlier.safetensors" not in message
    descriptor = consent_descriptor.parse(message)
    assert descriptor is not None and descriptor.unet_name == "pinned.safetensors"
    assert descriptor.measured["detail"].startswith(
        "pinned by the unconditional checkpoint uncond.safetensors or a LoRA in its stack: ")
    assert "set by the unconditional checkpoint uncond.safetensors" in message
    assert crossed == [False]  # the dm-cfg2 exchange told the conditional rank


@pytest.mark.parametrize("slot", ["cond", "uncond"])
def test_under_dual_model_cfg2_each_rank_row_names_the_model_it_rendered(monkeypatch, slot):
    """The driver keeps each model's counts together by this slot (attention_pins.merge)."""
    accuracy_waiver._scope().grants[SOL_KIND] = {
        "id": "c" * 32, "kind": SOL_KIND, "consent_source": "panel"}
    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda w, r: slot)
    crossed: list[bool] = []
    result, sampled = _run(monkeypatch, _worker(cfg=2), _pinned_patcher(Block(sol())),
                           crossed=crossed,
                           extra={"uncond_model": {"unet_name": "uncond.safetensors"}})
    assert sampled == [1] and crossed == [True]
    assert result[attention_pins.RESULT_KEY] == {
        "attention_pins": {"comfy_kitchen_sol tau=0.7": 1}, "attention_pins_ran": True,
        "attention_pins_dropped": None, attention_pins.MODEL_KEY: slot}


def test_an_fp8_checkpoint_keeps_its_pins_through_the_fsdp_wrap(monkeypatch):
    """Only the bf16 and fp16 shard build reloads through load_state_dict.

    A quantized checkpoint takes the direct wrap, which reloads nothing, so
    its pins stay and run under cfg2+fsdp and dp2+fsdp and ask for the waiver
    as on a resident topology. The stub shows the fp8 path never builds shards.
    """
    import torch.distributed
    import torch.distributed.fsdp

    from dgx_monarch.adapters import fsdp as fsdp_module
    from dgx_monarch.adapters import fsdp_quant, fsdp_shard_build

    wrapped: list = []
    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard",
                        lambda module, **kwargs: wrapped.append(module))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(fsdp_quant, "wrap_quantized_parameters", lambda dm: 1)
    monkeypatch.setattr(fsdp_module, "validate_fsdp_live_precision", lambda *args: None)

    def no_shard_build(*args, **kwargs):
        raise AssertionError("an fp8 checkpoint must take the direct wrap")

    monkeypatch.setattr(fsdp_shard_build, "build_shards", no_shard_build)
    dm = torch.nn.Module()
    dm.blocks = torch.nn.ModuleList([Block(sol()), Block(int8())])
    patcher = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dm), size=0)
    attention_pins.scan_loaded(patcher)
    fsdp_module.apply_fsdp_capacity_mode(patcher, "fp8", None)
    attention_pins.note_injected(patcher)
    assert dm.blocks[0] in wrapped and dm in wrapped
    summary = attention_pins.render_summary(_worker(cfg=2, fsdp=True), [patcher])
    assert summary is not None and summary.ran
    assert summary.running == {"comfy_kitchen_sol tau=0.7": 1, "comfy_kitchen_int8": 1}
    with pytest.raises(UnsupportedModelError):
        attention_pins.assert_waived(summary, {"cfg": 2, "fsdp": True})


def test_under_sequence_parallelism_the_pin_is_reported_dropped_and_the_render_runs(
        monkeypatch):
    crossed: list[bool] = []
    result, sampled = _run(monkeypatch, _worker(ulysses=2), _pinned_patcher(Block(sol())),
                           crossed=crossed)
    assert sampled == [1] and crossed == [True]
    assert result[attention_pins.RESULT_KEY] == {
        "attention_pins": {"comfy_kitchen_sol tau=0.7": 1},
        "attention_pins_ran": False,
        "attention_pins_dropped": {attention_pins.DROPPED_SEQUENCE_PARALLEL: 1},
    }


def test_with_the_waiver_granted_the_pin_runs_stamped_and_reported_ran(monkeypatch):
    accuracy_waiver._scope().grants[SOL_KIND] = {
        "id": "b" * 32, "kind": SOL_KIND, "consent_source": "panel"}
    crossed: list[bool] = []
    result, sampled = _run(monkeypatch, _worker(world=1), _pinned_patcher(Block(sol())),
                           crossed=crossed)
    assert sampled == [1]
    assert result[attention_pins.RESULT_KEY]["attention_pins_ran"] is True
    assert [stamp["guard"] for stamp in result[accuracy_waiver.RESULT_KEY]] == [
        attention_pins.SOL_ATTN_PIN_GUARD]


def test_an_unpinned_render_carries_no_pin_field(monkeypatch):
    crossed: list[bool] = []
    result, sampled = _run(monkeypatch, _worker(cfg=2), _pinned_patcher(Block(Attention())),
                           crossed=crossed)
    assert sampled == [1] and attention_pins.RESULT_KEY not in result


def test_fresh_load_scans_before_injection_and_marks_what_injection_cleared(rig):
    store, calls = rig
    comfy_sd = sys.modules["comfy.sd"]
    blocks = [Block(sol()), Block(int8())]

    def load_diffusion_model(path, model_options=None):
        calls.load += 1
        return types.SimpleNamespace(
            model=types.SimpleNamespace(diffusion_model=torch.nn.ModuleDict(
                {"blocks": torch.nn.ModuleList(blocks)})),
            offload_device="cpu")

    comfy_sd.load_diffusion_model = load_diffusion_model
    seen = {}

    def inject(base, quant, stack, evidence):
        # Injection sees the scan already cached, then clears one pin the way
        # the FSDP shard build's load_state_dict does.
        seen["scanned"] = dict(getattr(base.model, attention_pins._SCAN_ATTR))
        blocks[1].self_attn.comfy_attention.config = None

    active, _ = store.ensure("pinned.safetensors", None, None, on_base_loaded=inject)
    assert set(seen["scanned"]) == {
        "diffusion_model.blocks.0.self_attn.comfy_attention",
        "diffusion_model.blocks.1.self_attn.comfy_attention"}
    summary = attention_pins.summarize([active], sequence_parallel=False)
    assert summary is not None and summary.public()["attention_pins_dropped"] == {
        attention_pins.DROPPED_FSDP: 1}
    assert summary.running == {"comfy_kitchen_sol tau=0.7": 1}


def test_the_driver_folds_rank_rows_into_the_finish_record(monkeypatch):
    from dgx_monarch.nodes import render_result
    from dgx_monarch.telemetry import RenderProgress

    progress = RenderProgress()
    monkeypatch.setattr("dgx_monarch.telemetry.render_progress", progress)
    token = progress.start(2, {"attention": "TORCH_FLASH"}, token=object())
    rows = [{"rank": rank, "dp_rank": 0, "latent": "descriptor" if rank == 0 else None,
             "sample_s": 1.0,
             attention_pins.RESULT_KEY: {"attention_pins": {"comfy_kitchen_int8": 2},
                                         "attention_pins_ran": True,
                                         "attention_pins_dropped": None}}
            for rank in (0, 1)]
    monkeypatch.setattr(render_result, "verify_cross_rank_signatures", lambda r, t: None)
    monkeypatch.setattr(render_result, "_validate_adoption_evidence", lambda *a, **k: None)
    monkeypatch.setattr(render_result, "materialize_leader_samples",
                        lambda leaders, ranks, dp, read: (torch.zeros(1), []))
    monkeypatch.setattr(render_result, "collect_denoised_output", lambda *a: None)
    monkeypatch.setattr(render_result.consent_waiver, "stamp_result",
                        lambda out, results, latent: out)
    render_result._finish_render(rows, types.SimpleNamespace(dp=1), {},
                                 telemetry_token=token)
    progress.finish(token)
    last = progress.snapshot()["last"]
    assert last["attention"] == "TORCH_FLASH"
    assert last["attention_pins"] == {"comfy_kitchen_int8": 2}
    assert last["attention_pins_ran"] is True


def test_the_driver_keeps_each_dual_model_ranks_counts_in_the_finish_record(monkeypatch):
    from dgx_monarch.nodes import render_result
    from dgx_monarch.telemetry import RenderProgress

    progress = RenderProgress()
    monkeypatch.setattr("dgx_monarch.telemetry.render_progress", progress)
    token = progress.start(2, {"attention": "TORCH_FLASH"}, token=object())
    payloads = [
        {"attention_pins": {"comfy_kitchen_sol tau=0.7": 20}, "attention_pins_ran": True,
         "attention_pins_dropped": None, attention_pins.MODEL_KEY: "cond"},
        {"attention_pins": {"comfy_kitchen_sol tau=0.7": 10}, "attention_pins_ran": False,
         "attention_pins_dropped": {attention_pins.DROPPED_UNBOUND: 10},
         attention_pins.MODEL_KEY: "uncond"},
    ]
    rows = [{"rank": rank, "dp_rank": 0, "latent": "descriptor" if rank == 0 else None,
             "sample_s": 1.0, attention_pins.RESULT_KEY: payloads[rank]} for rank in (0, 1)]
    monkeypatch.setattr(render_result, "verify_cross_rank_signatures", lambda r, t: None)
    monkeypatch.setattr(render_result, "_validate_adoption_evidence", lambda *a, **k: None)
    monkeypatch.setattr(render_result, "materialize_leader_samples",
                        lambda leaders, ranks, dp, read: (torch.zeros(1), []))
    monkeypatch.setattr(render_result, "collect_denoised_output", lambda *a: None)
    monkeypatch.setattr(render_result.consent_waiver, "stamp_result",
                        lambda out, results, latent: out)
    render_result._finish_render(rows, types.SimpleNamespace(dp=1), {},
                                 telemetry_token=token)
    progress.finish(token)
    last = progress.snapshot()["last"]
    assert last["attention_pins"] == {"comfy_kitchen_sol tau=0.7": 30}
    assert last["attention_pins_ran"] is False
    assert last["attention_pins_dropped"] == {attention_pins.DROPPED_UNBOUND: 10}
    assert set(last[attention_pins.MODELS_FIELD]) == {"cond", "uncond"}
    assert attention_pins.describe(last) == (
        "pinned: cond model: comfy_kitchen_sol tau=0.7 x 20 modules, ran; "
        "uncond model: comfy_kitchen_sol tau=0.7 x 10 modules, dropped "
        "(kernel unavailable on this host)")
