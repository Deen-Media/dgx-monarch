"""MOSS-Image adapter: CPU contracts with no ComfyUI import at module scope.

The declarations, ``matches()``, the inject order and the bound forward on a
stand-in model shaped like comfy's ``MOSSImage`` (the stand-ins of
tests/test_moss_image_seams.py: the same attribute names and call shapes, toy
layers, and blocks whose stand-in attention takes the sharded override as
comfy's wrap_attn hands it over). One rank of a world-2 sequence-parallel
group runs it, with the gather and the ring degree stubbed and comfy's
prefetch queue recording its module list. What the stand-in checks is what
the real-model file (tests/test_moss_image_comfy_exactness.py) cannot see
without ComfyUI: the guard order before any tensor work, the options the text
embedder and each block get, the pad rows each attention call names, and the
quantization gate. The registry rows (adapters/__init__.py) are the lead's.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

import test_moss_image_seams as fakes
from dgx_monarch import accuracy_waiver
from dgx_monarch.adapters import base, moss_image
from dgx_monarch.adapters.attention_patches import (
    ATTENTION_PATCH_CAPABILITIES_ATTR,
    SEQUENCE_SHARD_LOCAL_CAPABILITY,
)
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.cfg_parallel import assert_sp_quant_supported
from dgx_monarch.adapters.moss_image import MOSSImageAdapter
from dgx_monarch.adapters.stock_attention import default_attention_override
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from fake_model_base_helpers import install_fake_model_base
from test_moss_image_seams import (
    ATTENTION_CALLS,
    NORM_CALLS,
    FakeMOSSImage,
    install_fake_moss_module,
    skip,
)

D = fakes.D


def _tag(exc: BaseException):
    return parse_leading_refusal_tag(str(exc))


def test_declarations_match_the_spec():
    adapter = MOSSImageAdapter()
    assert adapter.family == "moss_image"
    assert adapter.model_base_classes == adapter.exact_model_base_classes == ("MOSSImage",)
    assert adapter.cfg_cond_padding == "none"
    assert adapter.attention_head_dim == 128
    assert adapter.sp_validated_quants == frozenset({"bf16", "int8"})
    assert adapter.sp_honours_block_replace is True
    reason = adapter.sp_quant_refusal_reason
    for phrase in ("divisibility pad rows", "NVFP4", "adapters/mage_nvfp4_scale.py",
                   "fp8 weight_dtype cast", "unrecognized quantization format as fp8",
                   "INT8 ConvRot, which quantizes each row on its own"):
        assert phrase in reason, phrase
    assert adapter.usp_pad_exclusion_probe is None
    assert adapter.cfg_pad_restores_stock_call is False
    assert adapter.cfg_parallel_supported is True
    assert adapter.cfg_split_seam == "diffusion_model"
    assert adapter.cfg_batch_constant is None
    assert adapter.dp_cond_exempt_keys == frozenset()
    assert adapter.model_detection_attrs == ()
    for absent in ("prepare_base_model", "inject_cfg_pad_forward", "dual_model_cfg_supported"):
        assert not hasattr(adapter, absent), absent
    assert type(adapter).attention_dispatch is base.Adapter.attention_dispatch


def test_a_moss_image_subclass_refuses_class_p_and_names_the_override(monkeypatch):
    mb = install_fake_model_base(monkeypatch, {"MOSSImage": None, "_Other": None})
    assert MOSSImageAdapter().matches(mb.MOSSImage()) is True
    assert MOSSImageAdapter().matches(mb._Other()) is False
    finetune = type("MOSSImageFinetune", (mb.MOSSImage,), {})
    with pytest.raises(UnsupportedModelError, match="moss_image variant MOSSImageFinetune") as caught:
        MOSSImageAdapter().matches(finetune())
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    assert "family_adapter='moss_image'" in str(caught.value)


# --- the one-rank harness ----------------------------------------------------------


@pytest.fixture
def one_rank(monkeypatch):
    """Rank 0 of a world-2 group; returns (prefetch calls, attention calls, ring box, attention)."""
    install_fake_moss_module(monkeypatch)
    pops: list = []
    ring = {"world": 1}
    comfy = sys.modules["comfy"]
    flux_pkg = types.ModuleType("comfy.ldm.flux")
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.timestep_embedding = lambda t, dim: t.float()[:, None].expand(-1, dim) + 1.0
    prefetch = types.ModuleType("comfy.model_prefetch")

    def make_prefetch_queue(queue, device, options):
        pops.append(("queue", len(queue), dict(options)))
        return None

    def malloc_graph_begin(*_args, **_kwargs):
        raise AssertionError("the sharded loop opened the malloc graph")

    prefetch.make_prefetch_queue = make_prefetch_queue
    prefetch.prefetch_queue_pop = lambda queue, device, module, *args, **kwargs: pops.append(
        ("pop", module, args, kwargs))
    prefetch.malloc_graph_begin = prefetch.malloc_graph_end = malloc_graph_begin
    sys.modules["comfy.ldm"].flux, flux_pkg.layers, comfy.model_prefetch = flux_pkg, layers, prefetch
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_ring_parallel_world_size = lambda: ring["world"]
    for name, module in (("comfy.ldm.flux", flux_pkg), ("comfy.ldm.flux.layers", layers),
                         ("comfy.model_prefetch", prefetch),
                         ("xfuser", types.ModuleType("xfuser")),
                         ("xfuser.core", types.ModuleType("xfuser.core")),
                         ("xfuser.core.distributed", distributed)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(base, "sp_world", lambda: 2)
    monkeypatch.setattr(base, "sp_rank", lambda: 0)
    monkeypatch.setattr(moss_image, "ulysses_degree", lambda: 2)
    # Rank 0's half twice: the stand-in blocks do not mix rows, and the head
    # needs only the gathered length.
    monkeypatch.setattr(moss_image, "sp_gather",
                        lambda t, orig, dim=1: torch.cat([t, t], dim=dim).narrow(dim, 0, orig))
    calls: list = []

    def attention(q, k, v, heads, drop_rows=None, **kwargs):
        calls.append({"rows": q.shape[1], "drop_rows": drop_rows, "kwargs": kwargs})
        return q

    return pops, calls, ring, attention


def _injected(attention, layers: int = 3) -> FakeMOSSImage:
    model = FakeMOSSImage(layers)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.02)
    adapter = MOSSImageAdapter()
    for _ in range(2):  # a repeat load binds again and must not stack
        adapter.inject_usp(model, InjectionContext(topology_sp=2, usp_attention=attention,
                                                   pure_ulysses=True))
    return model


def _run(model, grid=(2, 2), text=3, options=None, channels=None, batch=1):
    x = torch.randn(batch, channels or model.latent_channels, *grid)
    context = torch.randn(batch, text, D)
    NORM_CALLS.clear()
    ATTENTION_CALLS.clear()
    with torch.no_grad():
        return model._forward(x, torch.tensor([0.5] * batch), context,
                              transformer_options=dict(options or {}))


def test_inject_checks_then_binds_every_block_and_the_forward(monkeypatch, one_rank):
    _pops, _calls, _ring, attention = one_rank
    order: list = []
    for name in ("assert_stock_tree", "refuse_indivisible_heads", "rope_call", "bind_norm_parity"):
        real = getattr(moss_image, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            order.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(moss_image, name, spy)
    real_bind = base.Adapter.bind

    def bind(module, method_name, func):
        order.append(f"bind {method_name}")
        real_bind(module, method_name, func)

    monkeypatch.setattr(MOSSImageAdapter, "bind", staticmethod(bind))
    model = FakeMOSSImage(42)
    MOSSImageAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=attention))
    assert order == (["assert_stock_tree", "refuse_indivisible_heads", "rope_call"]
                     + ["bind_norm_parity"] * 42 + ["bind _forward"])
    assert all("forward" in vars(block) for block in model.layers)
    assert "_forward" in vars(model)

    order.clear()
    monkeypatch.setattr(moss_image, "ulysses_degree", lambda: 5)
    fresh = FakeMOSSImage(4)
    with pytest.raises(UnsupportedModelError, match="Ulysses degree 5") as caught:
        MOSSImageAdapter().inject_usp(fresh, InjectionContext(topology_sp=5, usp_attention=attention))
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    assert str(caught.value).split("] ", 1)[1].startswith("moss_image: this checkpoint has 2 ")
    assert order == ["assert_stock_tree", "refuse_indivisible_heads"]
    assert not any("forward" in vars(block) for block in fresh.layers)
    assert "_forward" not in vars(fresh)


def test_a_subclass_tree_refuses_at_inject_before_any_bind(one_rank):
    _pops, _calls, _ring, attention = one_rank

    class Derivative(FakeMOSSImage):
        def _forward(self, x, timestep, context, transformer_options=None, **kwargs):
            return x

    model = Derivative()
    with pytest.raises(UnsupportedModelError, match=r"Derivative\._forward") as caught:
        MOSSImageAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=attention))
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    assert not any("forward" in vars(block) for block in model.layers)


def _spied(model):
    """Record the GEMMs and the text embedder the prelude would run; rope records itself."""
    work: list = []
    for name in ("video_embedder", "time_embedding", "shared_block_modulation",
                 "video_head_modulation", "final_linear_video", "text_embedder"):
        getattr(model, name).register_forward_pre_hook(
            lambda module, args, _name=name: work.append(_name))
    return work


def _marked(function):
    setattr(function, ATTENTION_PATCH_CAPABILITIES_ATTR, (SEQUENCE_SHARD_LOCAL_CAPABILITY,))
    return function


def _foreign(*_args, **_kwargs):
    raise AssertionError("never called")


def _unmarked_patch(args, extra):
    return {"img": args["img"] * 2}


GUARD_CASES = {
    "foreign override": ({"optimized_attention_override": _foreign}, (2, 2), 3, None,
                         RefusalClass.PHYSICS, "optimized_attention_override"),
    "attn1 hook": ({"patches": {"attn1_patch": [_foreign]}}, (2, 2), 3, None,
                   RefusalClass.PHYSICS, "attn1_patch"),
    "control channels": ({}, (2, 2), 3, 12, RefusalClass.PHYSICS, "12 latent channels"),
    "unvetted patch": ({"patches_replace": {"dit": {("single_block", 1): _unmarked_patch}}},
                       (2, 2), 3, None, RefusalClass.PHYSICS, "block 1"),
    "padded ring": ({}, (2, 2), 3, None, RefusalClass.KNOWN_WRONG, "ring/hybrid"),
}


@pytest.mark.parametrize("case", sorted(GUARD_CASES))
def test_the_forward_runs_its_guards_before_any_tensor_work(one_rank, case):
    _pops, calls, ring, attention = one_rank
    options, grid, text, channels, refusal_class, named = GUARD_CASES[case]
    if case == "padded ring":
        ring["world"] = 2   # 4 image rows and 3 text rows: N = 7 pads at sp 2
    model = _injected(attention)
    work = _spied(model)
    with pytest.raises(UnsupportedModelError, match=named) as caught:
        _run(model, grid=grid, text=text, options=options, channels=channels)
    tag = _tag(caught.value)
    assert tag is not None and tag.refusal_class is refusal_class
    if case == "padded ring":
        assert tag.guard == base.RING_PAD_GUARD and tag.waivable
    assert work == [] and model.ropes == [] and model.text_seen == []
    assert calls == [] and ATTENTION_CALLS == [] and NORM_CALLS == []


def test_an_even_ring_render_and_a_waived_padded_ring_proceed(monkeypatch, one_rank):
    _pops, calls, ring, attention = one_rank
    ring["world"] = 2
    model = _injected(attention)
    _run(model, grid=(2, 2), text=4)
    assert [call["drop_rows"] for call in calls] == [[]] * 3
    monkeypatch.setattr(accuracy_waiver, "waived", lambda guard: guard == base.RING_PAD_GUARD)
    calls.clear()
    _run(model, grid=(2, 2), text=3)
    assert [call["drop_rows"] for call in calls] == [[7]] * 3


def test_the_text_embedder_gets_the_stock_backend_and_the_blocks_the_usp_override(one_rank):
    _pops, calls, _ring, attention = one_rank
    model = _injected(attention)
    marked_upstream = base.usp_options({}, attention)["optimized_attention_override"]
    out = _run(model, grid=(2, 3), text=5, options={
        "rope_options": {"shift_t": 2.0}, "optimized_attention_override": marked_upstream})
    assert out.shape == (1, model.latent_channels, 2, 3)
    (text,) = model.text_seen
    assert text["rows"] == 5 and text["pe"] == 5
    assert text["options"]["optimized_attention_override"] is default_attention_override
    assert not {"total_blocks", "block_type", "block_index"} & set(text["options"])
    assert model.ropes == [{"rope_options": {"shift_t": 2.0},
                            "optimized_attention_override": marked_upstream}]
    assert len(ATTENTION_CALLS) == 3
    for index, record in enumerate(ATTENTION_CALLS):
        override = record["override"]
        assert override is not default_attention_override and override is not marked_upstream
        assert getattr(override, base.USP_ATTENTION_OVERRIDE_ATTR)
        assert record["keys"] == {"total_blocks": 3, "block_type": "single", "block_index": index}
        assert record["rows"] == record["pe"] == 6   # 11 rows pad to 12 at sp 2
    assert [call["drop_rows"] for call in calls] == [[11]] * 3
    assert all(call["kwargs"] == {} and call["rows"] == 6 for call in calls)


@pytest.mark.parametrize("text,drop", [(4, []), (3, [7])], ids=["even-N", "odd-N"])
def test_drop_rows_follow_the_parity_of_n(one_rank, text, drop):
    _pops, calls, _ring, attention = one_rank
    model = _injected(attention)
    _run(model, grid=(2, 2), text=text)
    assert [call["drop_rows"] for call in calls] == [drop] * 3


def test_every_norm_call_holds_stocks_rows(one_rank):
    _pops, _calls, _ring, attention = one_rank
    model = _injected(attention)
    _run(model, grid=(2, 2), text=3, batch=2)
    # Two block norms at N = 7 a block, then the head norm on the 4 gathered image rows.
    assert [tuple(call.shape[:2]) for call in NORM_CALLS] == [(2, 7)] * 6 + [(2, 4)]


def test_prefetch_mirrors_stock_and_no_malloc_graph_opens(one_rank):
    pops, _calls, _ring, attention = one_rank
    model = _injected(attention)
    _run(model)
    queue, *rest = pops
    assert queue[0] == "queue" and queue[1] == 3
    assert queue[2]["total_blocks"] == 3 and queue[2]["block_type"] == "single"
    assert rest == [("pop", block, (torch.float32,), {"malloc_scope": "block"})
                    for block in model.layers] + [("pop", None, (), {"malloc_scope": "block"})]


def test_a_vetted_patch_receives_rank_rows_and_a_parity_bound_original_block(one_rank):
    _pops, calls, _ring, attention = one_rank
    model = _injected(attention)
    seen: list = []

    def local(args, extra):
        seen.append({**args, "block_index": args["transformer_options"]["block_index"]})
        return extra["original_block"](args)

    patch = _marked(local)
    _run(model, grid=(2, 2), text=3, options={"patches_replace": {"dit": {
        ("single_block", 1): patch, ("single_block", 2): skip}}})
    (args,) = seen
    assert set(args) == {"img", "vec", "mod", "pe", "transformer_options", "block_index"}
    assert args["img"].shape[1] == args["pe"].shape[1] == 4
    assert args["vec"].shape == (1, D) and len(args["mod"]) == 6
    assert args["block_index"] == 1
    # Blocks 0 and 1 attend and norm at stock's 7 rows; the skipped block 2 runs nothing.
    assert len(calls) == 2
    assert [tuple(call.shape[:2]) for call in NORM_CALLS] == [(1, 7)] * 4 + [(1, 4)]


def test_an_unrouted_attention_refuses_class_p_after_the_loop(monkeypatch, one_rank):
    _pops, _calls, _ring, attention = one_rank
    model = _injected(attention)

    def bypass(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options):
        return q   # attends without the options that carry the sharded override

    monkeypatch.setattr(fakes, "attention", bypass)
    with pytest.raises(UnsupportedModelError, match="3 blocks ran and 0") as caught:
        _run(model)
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS


@pytest.mark.parametrize("quant", ["nvfp4", "fp8", "mxfp8", "unknown"])
def test_the_quant_gate(quant):
    adapter = MOSSImageAdapter()
    for kwargs in ({"sp": 2}, {"sp": 1, "cfg": 2}):
        sp = kwargs.pop("sp")
        with pytest.raises(UnsupportedModelError, match="moss_image shards its joint stream") as caught:
            assert_sp_quant_supported(adapter, sp, quant, **kwargs)
        assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
        # The cfg2 cause is named too, and the remedy offers both stock-forward paths.
        assert "the split of each step's [cond, uncond] batch" in str(caught.value)
        assert "Use dp2 with an even batch, or mode=local with gpus_per_host=1" in str(caught.value)
    for kept in ("bf16", "int8"):
        assert_sp_quant_supported(adapter, 2, kept)
        assert_sp_quant_supported(adapter, 1, kept, cfg=2)
    assert_sp_quant_supported(adapter, 1, quant)
    assert_sp_quant_supported(adapter, 1, quant, cfg=1)
