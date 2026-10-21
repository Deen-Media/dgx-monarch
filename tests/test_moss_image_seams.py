"""adapters/moss_image_seams.py on the CPU, with no ComfyUI import.

The seams the MOSS-Image sharded forward refuses, routes and cuts through:
the two ``rope`` forms and the third that refuses, the identity predicate
behind Skip Layer Guidance and the vetting of block replacements, the joint
window with its rotary rows and pad rows, the fused-norm route that launches
both of a block's norm calls at stock's row count, the stock-tree check, the
control-channel refusal and the attention ledger. The stand-in block below
mirrors stock's ``Block.forward`` call shape: it reads ``modulated_rms_norm``
and ``attention`` as module globals of this file, which is what the route and
the tree check read. tests/test_moss_image_comfy_exactness.py runs the same
seams through ComfyUI's own MOSS-Image model.

The stand-ins are shared with tests/test_moss_image_adapter.py.
"""
from __future__ import annotations

import functools
import math
import sys
import types

import pytest
import torch

from dgx_monarch.adapters import base, moss_image_seams
from dgx_monarch.adapters.attention_patches import (
    ATTENTION_PATCH_CAPABILITIES_ATTR,
    SEQUENCE_SHARD_LOCAL_CAPABILITY,
)
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.flux3_packing import refuse_indivisible_heads
from dgx_monarch.adapters.moss_image_seams import (
    AttentionLedger,
    assert_stock_tree,
    bind_norm_parity,
    is_identity_block_patch,
    refuse_control_channels,
    refuse_reshaped_block_rows,
    refuse_unsharded_attention,
    rope_call,
    rope_with_options,
    rope_without_options,
    shard_joint,
    vet_block_replacements,
)
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

D = 256   # two heads of 128, so Ulysses 2 divides them
NORM_CALLS: list[torch.Tensor] = []
ATTENTION_CALLS: list[dict] = []


def refusal_class(exc: BaseException) -> RefusalClass | None:
    tag = parse_leading_refusal_tag(str(exc))
    return tag.refusal_class if tag is not None else None


def modulated_rms_norm(x, norm, scale, shift):
    """Stock's fused norm stand-in: row-local, and it records each call it gets."""
    NORM_CALLS.append(x.detach().clone())
    return x * (1 + scale) + shift


def _backend(q, k, v, heads, **kwargs):
    """The kernel wrap_attn would run: row-local here, so a window computes stock's rows."""
    return q * 0.5 + k * 0.25 + v


def attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options):
    """Stock's attention() call shape: an override in the options takes the call, as in wrap_attn."""
    options = transformer_options or {}
    override = options.get("optimized_attention_override")
    ATTENTION_CALLS.append({"rows": q.shape[1], "pe": pe.shape[1], "override": override,
                            "keys": {key: options[key] for key in
                                     ("total_blocks", "block_type", "block_index") if key in options}})
    if override is not None:
        return override(_backend, q, k, v, heads, transformer_options=options,
                        _inside_attn_wrapper=True)
    return _backend(q, k, v, heads)


class FakeBlock(torch.nn.Module):
    """Stock ``Block.forward``'s call shape: two fused norms around attention, gated residuals."""

    num_heads = 2

    def __init__(self) -> None:
        super().__init__()
        self.comfy_attention = None

    def forward(self, x, mod, pe, transformer_options=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        y = modulated_rms_norm(x, None, scale_a, shift_a)
        x = x + attention(y, y, y, None, None, pe, self.num_heads, self.comfy_attention,
                          transformer_options) * gate_a
        y = modulated_rms_norm(x, None, scale_f, shift_f)
        return x + y * gate_f


class FakeTextBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.comfy_attention = None


class FakeTextEmbedder(torch.nn.Module):
    """Records the options and rotary rows it gets; returns the text unchanged."""

    def __init__(self, seen: list) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([FakeTextBlock(), FakeTextBlock()])
        self.seen = seen

    def forward(self, x, pe, transformer_options=None):
        self.seen.append({"rows": x.shape[1], "pe": pe.shape[1], "options": transformer_options})
        return x


class FakeMOSSImage(torch.nn.Module):
    """comfy's MOSSImage attribute and method names over toy layers (two heads of 128)."""

    latent_channels = 4

    def __init__(self, layers: int = 3) -> None:
        super().__init__()
        self.text_seen: list = []
        self.ropes: list = []
        self.layers = torch.nn.ModuleList([FakeBlock() for _ in range(layers)])
        self.text_embedder = FakeTextEmbedder(self.text_seen)
        self.video_embedder = torch.nn.Linear(self.latent_channels, D)
        self.time_embedding = torch.nn.Linear(256, D)
        self.shared_block_modulation = torch.nn.Linear(D, 6 * D)
        self.video_head_modulation = torch.nn.Linear(D, 2 * D)
        self.final_norm_video = None
        self.final_linear_video = torch.nn.Linear(D, self.latent_channels, bias=False)

    def rope(self, h, w, text_len, device, transformer_options=None):
        self.ropes.append(transformer_options)
        rows = h * w + text_len
        return torch.arange(rows, dtype=torch.float32).reshape(1, rows, 1, 1, 1, 1).expand(
            1, rows, 1, 2, 2, 2).contiguous() + 1

    def forward(self, x, timestep, context, transformer_options=None, **kwargs):
        return self._forward(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options=None, **kwargs):
        raise AssertionError("the stand-in has no stock forward; the adapter binds one")


def install_fake_moss_module(monkeypatch, **overrides) -> types.ModuleType:
    """A fake ``comfy.ldm.moss_image.model`` exposing the stand-in classes and globals."""
    moss = types.ModuleType("comfy.ldm.moss_image.model")
    moss.MOSSImage, moss.Block, moss.TextEmbedder = FakeMOSSImage, FakeBlock, FakeTextEmbedder
    moss.modulated_rms_norm, moss.attention = modulated_rms_norm, attention
    for name, value in overrides.items():
        setattr(moss, name, value)
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    package = types.ModuleType("comfy.ldm.moss_image")
    comfy.ldm, ldm.moss_image, package.model = ldm, package, moss
    for name, module in (("comfy", comfy), ("comfy.ldm", ldm), ("comfy.ldm.moss_image", package),
                         ("comfy.ldm.moss_image.model", moss)):
        monkeypatch.setitem(sys.modules, name, module)
    return moss


def mod_tuple(batch: int, seed: int = 3) -> tuple:
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(batch, 1, D, generator=generator) * 0.1 for _ in range(6))


@pytest.fixture(autouse=True)
def _clear_records():
    NORM_CALLS.clear()
    ATTENTION_CALLS.clear()
    yield
    NORM_CALLS.clear()
    ATTENTION_CALLS.clear()


# --- rope ----------------------------------------------------------------------


def test_rope_call_passes_options_on_the_master_form():
    model = FakeMOSSImage()
    rope = rope_call(model)
    assert rope is rope_with_options
    options = {"rope_options": {"shift_t": 1.0}}
    pe = rope(model, 2, 2, 3, "cpu", options)
    assert model.ropes == [options] and pe.shape[1] == 7


def test_rope_call_drops_options_on_the_v042_form():
    seen: list = []

    class V042(FakeMOSSImage):
        def rope(self, h, w, text_len, device):
            seen.append((h, w, text_len, device))
            return torch.zeros(1, h * w + text_len, 1, 2, 2, 2)

    model = V042()
    rope = rope_call(model)
    assert rope is rope_without_options
    assert rope(model, 2, 3, 4, "cpu", {"rope_options": {"shift_t": 1.0}}).shape[1] == 10
    assert seen == [(2, 3, 4, "cpu")]


def test_an_unknown_rope_signature_refuses_class_p():
    class Moved(FakeMOSSImage):
        def rope(self, h, w, device):
            raise AssertionError("never called")

    with pytest.raises(UnsupportedModelError, match=r"MOSSImage\.rope") as caught:
        rope_call(Moved())
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert "(h, w, device)" in str(caught.value)


# --- the identity predicate and the vetting -------------------------------------


def skip(args, extra_args):
    return args


def _returns_the_other(args, extra_args):
    return extra_args


def _rebuilds(args, extra_args):
    return {"img": args["img"]}


def _one(args):
    return args


def _three(args, extra_args, more):
    return args


def _star(args, *extra):
    return args


def _keywords(args, extra_args, **more):
    return args


def _keyword_only(args, extra_args, *, flag=None):
    return args


def _closure():
    kept = {}

    def inner(args, extra_args):
        return kept or args
    return inner


class _CallableObject:
    def __call__(self, args, extra_args):
        return args

    def method(self, args):
        return args


def test_the_identity_predicate_accepts_comfys_skip_shape():
    assert is_identity_block_patch(skip)
    assert is_identity_block_patch(lambda a, e: a)


@pytest.mark.parametrize("patch", [
    _returns_the_other, _rebuilds, _closure(), _star, _keywords, _keyword_only, _one, _three,
    functools.partial(skip), _CallableObject().method, _CallableObject(), None,
], ids=["extra_args", "rebuilt", "closure", "varargs", "varkw", "kwonly", "one", "three",
        "partial", "bound", "object", "none"])
def test_the_identity_predicate_rejects_every_non_identity(patch):
    assert not is_identity_block_patch(patch)


def _marked(function):
    setattr(function, ATTENTION_PATCH_CAPABILITIES_ATTR, (SEQUENCE_SHARD_LOCAL_CAPABILITY,))
    return function


def test_vet_admits_identity_and_sequence_shard_local_patches():
    def local(args, extra):
        return extra["original_block"](args)

    marked = _marked(local)
    options = {"patches_replace": {"dit": {("single_block", 1): skip, ("single_block", 2): marked}}}
    assert vet_block_replacements(options, 42) == {1: skip, 2: marked}
    assert vet_block_replacements({}, 42) == {}
    assert vet_block_replacements({"patches_replace": {}}, 42) == {}


def test_vet_refuses_an_unmarked_single_block_patch_class_p():
    def scale(args, extra):
        return {"img": args["img"] * 2}

    options = {"patches_replace": {"dit": {("single_block", 0): skip, ("single_block", 3): scale}}}
    with pytest.raises(UnsupportedModelError, match="block 3") as caught:
        vet_block_replacements(options, 42)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert SEQUENCE_SHARD_LOCAL_CAPABILITY in str(caught.value)


def test_vet_ignores_double_block_and_out_of_range_keys():
    def scale(args, extra):
        return {"img": args["img"] * 2}

    options = {"patches_replace": {"dit": {("double_block", 7): scale, ("single_block", 99): scale,
                                           ("single_block", 5): skip}}}
    assert vet_block_replacements(options, 42) == {5: skip}


@pytest.mark.parametrize(("options", "named"), [
    ({"patches_replace": {"dit": [("single_block", 1)]}}, "the 'dit' replacement table is a list"),
    ({"patches_replace": [("dit", {})]}, "the patches_replace option is a list"),
], ids=["dit-list", "patches-list"])
def test_vet_refuses_a_malformed_dit_table_class_p(options, named):
    with pytest.raises(UnsupportedModelError, match="not the dict") as caught:
        vet_block_replacements(options, 42)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert named in str(caught.value)


@pytest.mark.parametrize(("img", "pe", "named"), [
    (torch.zeros(1, 12, 4), torch.zeros(1, 11, 1, 2, 2, 2), "a (1, 12, 4) stream with 11 rotary rows"),
    (torch.zeros(1, 10, 4), torch.zeros(1, 11, 1, 2, 2, 2), "a (1, 10, 4) stream with 11 rotary rows"),
    (torch.zeros(1, 11, 4), torch.zeros(1, 10, 1, 2, 2, 2), "a (1, 11, 4) stream with 10 rotary rows"),
    (torch.zeros(2, 11, 4), torch.zeros(1, 11, 1, 2, 2, 2), "a (2, 11, 4) stream with 11 rotary rows"),
    (torch.zeros(11, 4), None, "a (11, 4) stream with no rotary rows"),
], ids=["widened", "narrowed", "table-cut", "batch", "flat"])
def test_a_marked_patch_that_reshapes_the_window_refuses_class_p(img, pe, named):
    refuse_reshaped_block_rows(3, {"img": torch.zeros(1, 11, 4), "pe": torch.zeros(1, 11, 1, 2, 2, 2)}, 1, 11)
    with pytest.raises(UnsupportedModelError, match="replacement on block 3") as caught:
        refuse_reshaped_block_rows(3, {"img": img, "pe": pe}, 1, 11)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert named in str(caught.value)
    assert "[1, 11, D] window with 11 rotary rows" in str(caught.value)
    assert "dp2 or topology 'single'" in str(caught.value)


# --- the joint window ------------------------------------------------------------


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [12, 13])
@pytest.mark.parametrize("batch", [1, 2])
def test_shard_joint_rows_pads_and_contiguity(monkeypatch, world, rows, batch):
    stream = torch.randn(batch, rows, 6) + 3
    pe = torch.randn(1, rows, 1, 2, 2, 2) + 3
    n = math.ceil(rows / world)
    monkeypatch.setattr(base, "sp_world", lambda: world)
    for rank in range(world):
        monkeypatch.setattr(base, "sp_rank", lambda rank=rank: rank)
        shard = shard_joint(stream, pe)
        assert shard.tokens.is_contiguous() and shard.pe.is_contiguous()
        assert shard.pe.shape[1] == shard.tokens.shape[1] == n
        assert shard.drop_rows == list(range(rows, n * world))
        assert shard.orig == rows
        real = max(0, min(n, rows - rank * n))
        assert torch.equal(shard.tokens[:, :real], stream[:, rank * n:rank * n + real])
        assert torch.equal(shard.pe[:, :real], pe[:, rank * n:rank * n + real])
        assert torch.count_nonzero(shard.tokens[:, real:]) == 0
        assert torch.count_nonzero(shard.pe[:, real:]) == 0


# --- the fused-norm route --------------------------------------------------------


def _window(batch=2, rows=5, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, rows, D), mod_tuple(batch)


def test_the_bound_block_launches_both_norm_calls_at_stock_rows():
    x, mod = _window()
    expected = FakeBlock()(x.clone(), mod, torch.zeros(1, 5), {})
    assert [tuple(call.shape) for call in NORM_CALLS] == [(2, 5, D)] * 2
    NORM_CALLS.clear()
    block = FakeBlock()
    assert bind_norm_parity(block)
    out = block(x.clone(), mod, torch.zeros(1, 5), {}, stock_norm_rows=9)
    assert [tuple(call.shape) for call in NORM_CALLS] == [(2, 9, D)] * 2
    assert torch.equal(NORM_CALLS[0][:, :5], x)
    for call in NORM_CALLS:
        assert torch.count_nonzero(call[:, 5:]) == 0
    assert torch.equal(out, expected)


def test_without_stock_rows_the_bound_forward_is_the_class_forward():
    x, mod = _window()
    block = FakeBlock()
    bind_norm_parity(block)
    expected = FakeBlock()(x.clone(), mod, torch.zeros(1, 5), {})
    NORM_CALLS.clear()
    out = block(x.clone(), mod, torch.zeros(1, 5), {})
    assert [tuple(call.shape) for call in NORM_CALLS] == [(2, 5, D)] * 2
    assert torch.equal(out, expected)


def test_a_repeat_bind_replaces_and_leaves_class_and_module_untouched():
    x, mod = _window()
    class_forward, module_norm = vars(FakeBlock)["forward"], globals()["modulated_rms_norm"]
    block = FakeBlock()
    for _ in range(3):
        bind_norm_parity(block)
    block(x.clone(), mod, torch.zeros(1, 5), {}, stock_norm_rows=9)
    assert [tuple(call.shape) for call in NORM_CALLS] == [(2, 9, D)] * 2   # one route, not three
    assert vars(FakeBlock)["forward"] is class_forward
    assert globals()["modulated_rms_norm"] is module_norm
    assert "forward" not in vars(FakeBlock())


class _ThirdCall(FakeBlock):
    def forward(self, x, mod, pe, transformer_options=None):
        x = super().forward(x, mod, pe, transformer_options)
        return modulated_rms_norm(x, None, mod[1], mod[0])


class _FlatCall(FakeBlock):
    def forward(self, x, mod, pe, transformer_options=None):
        y = modulated_rms_norm(x.flatten(0, 1), None, mod[1][0], mod[0][0])
        return modulated_rms_norm(y.view_as(x), None, mod[4], mod[3])


class _WrongRows(FakeBlock):
    def forward(self, x, mod, pe, transformer_options=None):
        y = modulated_rms_norm(x[:, :-1], None, mod[1], mod[0])
        return modulated_rms_norm(y, None, mod[4], mod[3])


class _RowScale(FakeBlock):
    def forward(self, x, mod, pe, transformer_options=None):
        y = modulated_rms_norm(x, None, mod[1].expand_as(x), mod[0].expand_as(x))
        return modulated_rms_norm(y, None, mod[4], mod[3])


class _OneCall(FakeBlock):
    def forward(self, x, mod, pe, transformer_options=None):
        return modulated_rms_norm(x, None, mod[1], mod[0])


@pytest.mark.parametrize("block_class", [_ThirdCall, _FlatCall, _WrongRows, _RowScale, _OneCall],
                         ids=["third-call", "2-d-call", "wrong-rows", "row-scale", "one-call"])
def test_a_stray_norm_call_refuses_class_p(block_class):
    x, mod = _window()
    block = block_class()
    assert bind_norm_parity(block)
    with pytest.raises(UnsupportedModelError, match=r"Block\.forward") as caught:
        block(x.clone(), mod, torch.zeros(1, 5), {}, stock_norm_rows=9)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert "canary seam 43" in str(caught.value)


def test_a_missing_norm_global_refuses_class_p(monkeypatch):
    x, mod = _window()
    block = FakeBlock()
    assert bind_norm_parity(block)
    monkeypatch.setitem(globals(), "modulated_rms_norm", None)
    with pytest.raises(UnsupportedModelError, match="missing or not callable") as caught:
        block(x.clone(), mod, torch.zeros(1, 5), {}, stock_norm_rows=9)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS


class _NoNorm(torch.nn.Module):
    def forward(self, x, mod, pe, transformer_options=None):
        return x + 1


def test_bind_reports_an_unroutable_forward():
    block = _NoNorm()
    assert not bind_norm_parity(block)
    x, mod = _window()
    assert torch.equal(block(x, mod, None, {}, stock_norm_rows=9), x + 1)


# --- the stock tree ---------------------------------------------------------------


def test_the_stock_tree_check_passes_the_exact_tree(monkeypatch):
    install_fake_moss_module(monkeypatch)
    assert_stock_tree(FakeMOSSImage())


class _OwnForward(FakeMOSSImage):
    def _forward(self, x, timestep, context, transformer_options=None, **kwargs):
        return x


class _OwnOuter(FakeMOSSImage):
    def forward(self, x, timestep, context, transformer_options=None, **kwargs):
        return x


class _OwnRope(FakeMOSSImage):
    def rope(self, h, w, text_len, device, transformer_options=None):
        return torch.zeros(1, h * w + text_len)


class _OtherBlock(FakeBlock):
    def forward(self, x, mod, pe, transformer_options=None):
        return x


class _OtherText(torch.nn.Module):
    blocks = ()


def _with_layer(model):
    model.layers[1] = _OtherBlock()
    return model


def _with_text(model):
    model.text_embedder = _OtherText()
    return model


def _with_width(model):
    model.video_embedder = torch.nn.Linear(4, 96)
    return model


def _without_layers(model):
    model.layers = torch.nn.ModuleList()
    return model


@pytest.mark.parametrize("build,named", [
    (lambda: _OwnForward(), "_OwnForward._forward"),
    (lambda: _OwnOuter(), "_OwnOuter.forward"),
    (lambda: _OwnRope(), "_OwnRope.rope"),
    (lambda: _with_layer(FakeMOSSImage()), "layer 1 is _OtherBlock"),
    (lambda: _with_text(FakeMOSSImage()), "text embedder is _OtherText"),
    (lambda: _with_width(FakeMOSSImage()), "model width 96 over 2 heads"),
    (lambda: _without_layers(FakeMOSSImage()), "no layers"),
    (lambda: torch.nn.Linear(2, 2), "not MOSSImage"),
], ids=["_forward", "forward", "rope", "layer", "text", "head-dim", "no-layers", "other-class"])
def test_the_stock_tree_check_refuses_each_difference_class_p(monkeypatch, build, named):
    install_fake_moss_module(monkeypatch)
    with pytest.raises(UnsupportedModelError, match="not ComfyUI's stock MOSS-Image tree") as caught:
        assert_stock_tree(build())
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert named in str(caught.value)


def test_the_stock_tree_check_refuses_a_block_without_the_two_globals(monkeypatch):
    class Bare(FakeBlock):
        def forward(self, x, mod, pe, transformer_options=None):
            return x

    install_fake_moss_module(monkeypatch, Block=Bare)
    model = FakeMOSSImage()
    model.layers = torch.nn.ModuleList([Bare(), Bare()])
    with pytest.raises(UnsupportedModelError) as caught:
        assert_stock_tree(model)
    for name in ("modulated_rms_norm", "attention"):
        assert f"no longer reads the module global {name}" in str(caught.value)


# --- the control channels and the attention ledger -------------------------------


def test_control_channels_refuse_class_p():
    refuse_control_channels(128, 128)
    with pytest.raises(UnsupportedModelError, match="384 latent channels") as caught:
        refuse_control_channels(384, 128)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert "docs/TROUBLESHOOTING.md #127" in str(caught.value)


def test_the_attention_ledger_counts_passes_adopted_and_refuses_unknown_keywords():
    seen: list = []

    def attn(q, k, v, heads, drop_rows=None, **kwargs):
        seen.append((heads, drop_rows, kwargs))
        return q

    ledger = AttentionLedger()
    guarded = ledger.guard(attn)
    q = torch.zeros(1, 2, 4)
    assert guarded(q, q, q, 2, drop_rows=[3]) is q
    guarded(q, q, q, 2, attn_precision=None)
    assert ledger.attended == 2 and ledger.started == 0
    assert seen == [(2, [3], {}), (2, None, {"attn_precision": None})]
    with pytest.raises(UnsupportedModelError, match="'scale'") as caught:
        guarded(q, q, q, 2, scale=0.1)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert "docs/TROUBLESHOOTING.md #116" in str(caught.value)
    assert ledger.attended == 2


def test_unsharded_attention_refuses_class_p():
    refuse_unsharded_attention(42, 42)
    with pytest.raises(UnsupportedModelError, match="42 blocks ran and 41") as caught:
        refuse_unsharded_attention(42, 41)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS


def test_the_head_refusal_names_the_family_it_is_given():
    with pytest.raises(UnsupportedModelError) as caught:
        refuse_indivisible_heads(24, 5, family="moss_image",
                                 divisors="2, 4 or 8 for the 24-head MOSS-Image files")
    text = str(caught.value)
    assert refusal_class(caught.value) is RefusalClass.PHYSICS
    assert text.split("] ", 1)[1].startswith("moss_image: this checkpoint has 24 attention heads")
    assert "Ulysses degree 5" in text and "24-head MOSS-Image files" in text
    with pytest.raises(UnsupportedModelError) as default:
        refuse_indivisible_heads(24, 5)
    assert str(default.value).split("] ", 1)[1].startswith("flux3: ")
    assert "24-head FLUX.3 Klein files" in str(default.value)


def test_the_module_names_comfys_globals():
    assert moss_image_seams.MOSS_MODEL_MODULE == "comfy.ldm.moss_image.model"
    assert moss_image_seams.NORM_GLOBAL == "modulated_rms_norm"
    assert moss_image_seams.ATTENTION_GLOBAL == "attention"
    assert moss_image_seams.ADOPTED_ATTENTION_KEYWORDS == {
        "mask", "attn_precision", "skip_reshape", "skip_output_reshape", "enable_gqa"}
