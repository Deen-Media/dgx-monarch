"""FLUX.3 Klein adapter: CPU contracts with no ComfyUI import at module scope.

The declarations, the registry and compile entries, and the bound forward on
a stand-in model shaped like comfy's ``Flux3`` (comfy/ldm/flux3/model.py):
the same attribute names and call shapes, toy layers, and blocks that record
what they are handed. One rank of a world-2 sequence-parallel group runs it,
with the gather and the ring degree stubbed and comfy's prefetch queue
recording its module list and answering None, as it does off the
comfy-managed residency rung. What the
stand-in checks is what the real-model file (tests/test_flux3_comfy_exactness.py)
cannot see without ComfyUI: the options each phase's blocks get, the pad rows
each attention call names, the typed refusals, and both input shapes.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch.adapters import (
    ADAPTERS,
    DUAL_MODEL_CFG_FAMILIES,
    SELECTABLE_FAMILIES,
    base,
    flux3,
    get_adapter,
)
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.flux3 import Flux3Adapter
from dgx_monarch.adapters.stock_attention import default_attention_override
from dgx_monarch.compile_policy import KNOWN_COMPILE_NOOP_FAMILIES, compile_dit_is_known_noop
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from fake_model_base_helpers import install_fake_model_base

D, C, HEADS = 6, 4, 2


def _refusal_class(exc: BaseException) -> RefusalClass | None:
    tag = parse_leading_refusal_tag(str(exc))
    return tag.refusal_class if tag is not None else None


def test_declarations_match_the_design():
    adapter = Flux3Adapter()
    assert adapter.family == "flux3"
    assert adapter.model_base_classes == adapter.exact_model_base_classes == ("Flux3",)
    assert adapter.cfg_cond_padding == "none"
    assert adapter.attention_head_dim == 128
    assert adapter.usp_pad_exclusion_probe is None
    assert adapter.cfg_parallel_supported is True
    assert adapter.cfg_split_seam == "diffusion_model"
    assert adapter.cfg_batch_constant is None
    assert adapter.dp_cond_exempt_keys == frozenset()
    assert not hasattr(adapter, "dual_model_cfg_supported")
    assert not hasattr(adapter, "inject_cfg_pad_forward")
    assert "flux3" not in DUAL_MODEL_CFG_FAMILIES
    assert "flux3" in SELECTABLE_FAMILIES
    assert sum(isinstance(a, Flux3Adapter) for a in ADAPTERS) == 1


def test_compile_is_a_known_noop_for_flux3():
    """The block compiler compiles only a `blocks` list, which Flux3 lacks."""
    assert "flux3" in KNOWN_COMPILE_NOOP_FAMILIES
    assert compile_dit_is_known_noop("flux3")


def test_get_adapter_names_flux3_in_its_refusal(monkeypatch):
    mb = install_fake_model_base(monkeypatch, {"_Unrelated": None})
    with pytest.raises(UnsupportedModelError, match="has no dgx-monarch adapter") as caught:
        get_adapter(mb._Unrelated())
    named = str(caught.value).split("Supported families: ", 1)[1]
    assert "flux3" in [name.strip(" .(") for name in named.split(",")]
    for family in SELECTABLE_FAMILIES:
        assert family in named, family


def test_a_flux3_subclass_refuses_class_p_and_names_the_override(monkeypatch):
    mb = install_fake_model_base(monkeypatch, {"Flux3": None})
    finetune = type("Flux3Finetune", (mb.Flux3,), {})
    with pytest.raises(UnsupportedModelError, match="flux3 variant Flux3Finetune") as caught:
        Flux3Adapter().matches(finetune())
    assert _refusal_class(caught.value) is RefusalClass.PHYSICS
    assert "family_adapter='flux3'" in str(caught.value)


# --- The stand-in model --------------------------------------------------------


class _ModulationOut:
    def __init__(self, vec):
        self.shift = self.scale = self.gate = vec[:, None, :]


class _Modulation(torch.nn.Module):
    def forward(self, vec):
        return (_ModulationOut(vec),)


class _Block(torch.nn.Module):
    """Records its rows, spans, rotary rows, block keys and the attention's pad rows."""

    num_heads = HEADS

    def __init__(self, name: str, log: list) -> None:
        super().__init__()
        self.name = name
        self.log = log
        self.comfy_attention = None

    def forward(self, x, spans, pe, transformer_options=None):
        transformer_options = transformer_options or {}
        override = transformer_options.get("optimized_attention_override")
        self.log.append({
            "block": self.name, "rows": x.shape[1], "pe": pe.shape[1],
            "spans": [(start, end) for start, end, _mod in spans],
            "keys": {key: transformer_options[key] for key in
                     ("total_blocks", "block_type", "block_index") if key in transformer_options},
            "override": override,
        })
        if override is not None:
            # wrap_attn's call: the backend body, then q, k, v and the heads.
            override(lambda q, *_args, **_kwargs: q, x, x, x, self.num_heads)
        return x


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(D, C)

    def forward(self, x, vec):
        return self.linear(x)


class _Flux3(torch.nn.Module):
    """comfy's Flux3 attribute and method names over toy layers."""

    in_channels = C

    def __init__(self, flow_map: bool = False):
        super().__init__()
        self.flow_map = flow_map
        self.log: list = []
        self.preprocessed: list = []
        modes = ("image", "image_cond", "txt")
        self.early_stream_modulations = torch.nn.ModuleDict({m: _Modulation() for m in modes})
        self.single_stream_modulations = torch.nn.ModuleDict({m: _Modulation() for m in modes})
        self.emb_in = torch.nn.ModuleDict({m: torch.nn.Linear(C, D) for m in modes[:2]})
        self.content_mode_blocks = torch.nn.ModuleDict({
            "image": torch.nn.ModuleList([_Block("image0", self.log)]),
            "image_cond": torch.nn.ModuleList([_Block("ref0", self.log)])})
        self.txt_mode_blocks = torch.nn.ModuleList([_Block("txt0", self.log)])
        self.single_blocks = torch.nn.ModuleList(
            [_Block("single0", self.log), _Block("single1", self.log)])
        self.final_layer = torch.nn.ModuleDict({"image": _Head()})

    def time_in(self, emb):
        return emb[:, :D]

    def ref_time_in(self, emb):
        return emb[:, :D] * 0.5

    def image_ids(self, h, w, index, device):
        ids = torch.zeros(h * w, 4)
        ids[:, 3] = index
        return ids

    def ref_ids(self, shapes, device):
        return torch.cat([self.image_ids(h, w, -1 - i, device) for i, (h, w) in enumerate(shapes)])

    def text_ids(self, length, device):
        return torch.zeros(length, 4)

    def rope(self, ids):
        return ids.unsqueeze(0)

    def preprocess_conds(self, context, ref_latents=None, vec_target=None, transformer_options=None):
        self.preprocessed.append(vec_target)
        for block in self.txt_mode_blocks:
            block(context, [(0, context.shape[1], None)], self.rope(self.text_ids(1, None)))
        refs = None
        if ref_latents:
            refs = self.emb_in["image_cond"](
                torch.cat([ref.flatten(2).transpose(1, 2) for ref in ref_latents], dim=1))
        return context, refs


@pytest.fixture
def one_rank(monkeypatch):
    """Rank 0 of a world-2 group; returns (prefetch calls, attention calls,
    ring degree box, attention)."""
    pops: list = []
    ring = {"world": 1}
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    flux_pkg = types.ModuleType("comfy.ldm.flux")
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.timestep_embedding = lambda t, dim: t.float()[:, None].expand(-1, dim) + 1.0
    prefetch = types.ModuleType("comfy.model_prefetch")

    def make_prefetch_queue(queue, device, options):
        pops.append(("queue", [getattr(module, "name", module) for module in queue], {}))
        return None

    prefetch.make_prefetch_queue = make_prefetch_queue
    prefetch.prefetch_queue_pop = lambda queue, device, module, *args, **kwargs: pops.append(
        (getattr(module, "name", module), args, kwargs))
    comfy.ldm, ldm.flux, flux_pkg.layers, comfy.model_prefetch = ldm, flux_pkg, layers, prefetch
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_ring_parallel_world_size = lambda: ring["world"]
    for name, module in (("comfy", comfy), ("comfy.ldm", ldm), ("comfy.ldm.flux", flux_pkg),
                         ("comfy.ldm.flux.layers", layers), ("comfy.model_prefetch", prefetch),
                         ("xfuser", types.ModuleType("xfuser")),
                         ("xfuser.core", types.ModuleType("xfuser.core")),
                         ("xfuser.core.distributed", distributed)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(base, "sp_world", lambda: 2)
    monkeypatch.setattr(base, "sp_rank", lambda: 0)
    # Rank 0's half twice: the stand-in blocks do not mix rows, and the head
    # needs only the gathered length.
    monkeypatch.setattr(flux3, "sp_gather",
                        lambda t, orig, dim=1: torch.cat([t, t], dim=dim).narrow(dim, 0, orig))
    calls: list = []

    def attention(*args, drop_rows=None, **kwargs):
        calls.append(drop_rows)
        return args[0]

    return pops, calls, ring, attention


def _injected(attention, flow_map=False) -> _Flux3:
    model = _Flux3(flow_map=flow_map)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    adapter = Flux3Adapter()
    for _ in range(2):  # a repeat load binds again and must not stack
        adapter.inject_usp(model, InjectionContext(topology_sp=2, usp_attention=attention,
                                                   pure_ulysses=True))
    return model


def _run(model, grid=(3, 3), text=4, refs=(), cached=False, options=None):
    x = torch.randn(1, C, *grid)
    context = torch.randn(1, text, D)
    references = [torch.randn(1, C, *shape) for shape in refs] or None
    kwargs = {}
    if cached:
        txt, ref = model.preprocess_conds(context, references)
        model.preprocessed.clear()
        kwargs = {"txt_states": txt, "ref_states": ref,
                  "ref_shapes": [tuple(shape) for shape in refs] or None}
    model.log.clear()
    with torch.no_grad():
        return model._forward(x, torch.tensor([0.5]), context, references,
                              transformer_options=dict(options or {}), **kwargs)


def test_each_phase_gets_stocks_options_and_names_its_own_pad_rows(one_rank):
    """3 x 3 image rows and 4 text rows: 9 image rows pad to 10 and the 13-row
    joint stream to 14 at world 2, so each phase names its own tail row."""
    pops, calls, _ring, attention = one_rank
    model = _injected(attention)
    out = _run(model)
    assert out.shape == (1, C, 3, 3)
    sharded = [entry for entry in model.log if entry["block"] in ("image0", "single0", "single1")]
    image, single0, single1 = sharded
    assert image["keys"] == {} and image["rows"] == image["pe"] == 5
    assert image["spans"] == [(0, 5)]
    assert single0["keys"] == {"total_blocks": 2, "block_type": "single", "block_index": 0}
    assert single1["keys"] == {"total_blocks": 2, "block_type": "single", "block_index": 1}
    assert single0["rows"] == single0["pe"] == 7
    # Rank 0 holds joint rows [0, 7), all of them image rows.
    assert single0["spans"] == [(0, 7)]
    sharded_calls = calls[-3:]
    assert sharded_calls == [[9], [13], [13]]
    # Stock's prefetch: one queue over the image blocks then the single
    # blocks, one pop per block with the dtype and the block scope, then the
    # closing None pop.
    assert [(name, args, kwargs) for name, args, kwargs in pops] == [
        ("queue", ["image0", "single0", "single1"], {}),
        ("image0", (torch.float32,), {"malloc_scope": "block"}),
        ("single0", (torch.float32,), {"malloc_scope": "block"}),
        ("single1", (torch.float32,), {"malloc_scope": "block"}),
        (None, (), {"malloc_scope": "block"}),
    ]


def test_an_even_stream_names_no_pad_rows(one_rank):
    _pops, calls, _ring, attention = one_rank
    model = _injected(attention)
    _run(model, grid=(2, 2), text=4)
    assert calls[-3:] == [[], [], []]


def test_a_reference_joins_the_joint_stream_and_its_span_is_clipped(one_rank):
    """4 image rows, 6 reference rows, 2 text rows: rank 0 holds [0, 6) of 12."""
    _pops, calls, _ring, attention = one_rank
    model = _injected(attention)
    _run(model, grid=(2, 2), text=2, refs=[(2, 3)])
    single = next(entry for entry in model.log if entry["block"] == "single0")
    assert single["spans"] == [(0, 4), (4, 6)]
    assert calls[-1] == []


def test_cached_states_skip_preprocess_and_uncached_run_it_once(one_rank):
    _pops, _calls, _ring, attention = one_rank
    model = _injected(attention)
    cached = _run(model, refs=[(1, 2)], cached=True)
    assert model.preprocessed == []
    torch.manual_seed(0)
    uncached = _run(model, refs=[(1, 2)])
    assert len(model.preprocessed) == 1 and model.preprocessed[0] is None
    assert cached.shape == uncached.shape


def test_the_early_blocks_run_the_stock_backend_and_the_sharded_blocks_the_override(one_rank):
    _pops, _calls, _ring, attention = one_rank
    model = _injected(attention)
    _run(model, refs=[(1, 2)])
    by_block = {entry["block"]: entry["override"] for entry in model.log}
    assert by_block["txt0"] is default_attention_override
    model.log.clear()
    model.content_mode_blocks["image_cond"][0](torch.zeros(1, 2, D), [(0, 2, None)],
                                              torch.zeros(1, 2, 4))
    assert model.log[0]["override"] is default_attention_override
    for name in ("image0", "single0", "single1"):
        assert by_block[name] is not default_attention_override
        assert getattr(by_block[name], base.USP_ATTENTION_OVERRIDE_ATTR)


def test_the_flow_map_reads_the_sampler_schedule_or_refuses_class_p(one_rank):
    _pops, _calls, _ring, attention = one_rank
    model = _injected(attention, flow_map=True)
    with pytest.raises(UnsupportedModelError, match="sigma schedule") as caught:
        _run(model)
    assert _refusal_class(caught.value) is RefusalClass.PHYSICS
    assert model.log == []
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    _run(model, options={"sample_sigmas": sigmas})
    # The step-distilled prelude embeds the next table sigma and hands it to
    # stock's preprocess_conds.
    assert len(model.preprocessed) == 1 and model.preprocessed[0] is not None


def test_a_padded_ring_render_refuses_class_k_before_any_block(one_rank, monkeypatch):
    _pops, _calls, ring, attention = one_rank
    ring["world"] = 2
    model = _injected(attention)
    with pytest.raises(UnsupportedModelError, match="ring/hybrid") as caught:
        _run(model)
    assert _refusal_class(caught.value) is RefusalClass.KNOWN_WRONG
    assert [entry["block"] for entry in model.log] == ["txt0"]   # preprocess only
    model.log.clear()
    _run(model, grid=(2, 2), text=4)   # an even render needs no exclusion
    assert model.log


def test_a_ulysses_degree_that_does_not_divide_the_heads_refuses_class_p(monkeypatch, one_rank):
    _pops, _calls, _ring, attention = one_rank
    monkeypatch.setattr(flux3, "ulysses_degree", lambda: 4)
    with pytest.raises(UnsupportedModelError, match="2 attention heads") as caught:
        _injected(attention)
    assert _refusal_class(caught.value) is RefusalClass.PHYSICS
