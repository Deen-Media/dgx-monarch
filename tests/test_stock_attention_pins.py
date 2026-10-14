"""No stock attention call honors a block's kernel pin under sequence parallelism.

Issue #30. ComfyUI lets a checkpoint pin an attention kernel per block, and
comfy's ``wrap_attn`` runs the pin unless ``transformer_options`` carries an
override, which it checks first. Under sequence parallelism the pack leaves
some stock attention in place: cross-attention against the replicated text,
the refiners every rank runs whole before the sharded block loop, and the text
refiners comfy runs from ``extra_conds``. Each family's injected forward must
hand those calls the override from ``adapters/stock_attention.py``, so the pin
never runs there and the output equals stock with no pin, bit for bit.

Each test drives one family's injected forward at one rank through stand-in
models whose stock sites call a stub of comfy's dispatch
(tests/stock_attention_helpers.py). The helper also runs each forward with the
override removed and checks that every pin then runs, so a site the fixture
misses fails the test rather than passing it. The canary's "attention keyword
handoff" seam drives tiny real ComfyUI models through the same sites.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch.adapters import (
    anima,
    boogu,
    hunyuan,
    kandinsky5,
    krea2,
    ltx,
    minimax_h3,
    omnigen2,
    stock_attention,
    wan_animate2,
    wan_family,
    wan_variants,
    zimage,
)
from dgx_monarch.adapters.base import USP_ATTENTION_OVERRIDE_ATTR, InjectionContext
from stock_attention_helpers import (
    Pin,
    StockAttention,
    assert_pins_dropped,
    install,
    one_rank_attention,
)
from test_attention_container_seam import _AttentionTensorContainer, _wrap_attn

DIM = 4


@pytest.fixture
def stubs(monkeypatch):
    install(monkeypatch)
    return monkeypatch


def _ctx(**kwargs):
    kwargs.setdefault("usp_attention", one_rank_attention)
    return InjectionContext(topology_sp=1, **kwargs)


def _seeded(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


class _Identity(torch.nn.Module):
    """A sharded block: the pack's own attention runs there, not a stock call."""

    def forward(self, x, *_args, **_kwargs):
        return x


class _Mix(torch.nn.Module):
    """A sharded block that carries the text rows into the image rows, as the
    joint attention does, so a change in the text reaches the output."""

    def forward(self, x, *_args, **_kwargs):
        return x + x.mean(dim=1, keepdim=True)


def _mix_streams(img, txt):
    return img + txt.mean(dim=1, keepdim=True), txt


# --- the helper itself -------------------------------------------------------


def test_override_runs_the_unpinned_backend_with_the_call_unchanged():
    """The override hands the call to the backend function wrap_attn would run
    with no pin, with every argument as received, and keeps its mark."""
    seen = []

    def backend(*args, **kwargs):
        seen.append((args, kwargs))
        return "ran"

    options = stock_attention.stock_attention_options({"patches": {"k": 1}})
    override = options["optimized_attention_override"]
    assert override is stock_attention.default_attention_override
    assert getattr(override, USP_ATTENTION_OVERRIDE_ATTR) is True
    assert not hasattr(override, "container_function")
    assert override(backend, 1, 2, 3, 4, mask=None, transformer_options=options,
                    _inside_attn_wrapper=True) == "ran"
    assert seen == [((1, 2, 3, 4), {"mask": None, "transformer_options": options,
                                     "_inside_attn_wrapper": True})]


def test_options_are_a_copy_and_an_existing_override_is_kept():
    original = {"patches": {}}
    copied = stock_attention.stock_attention_options(original)
    assert copied is not original and "optimized_attention_override" not in original
    assert copied["patches"] is original["patches"]
    assert stock_attention.stock_attention_options(None).keys() == {
        "optimized_attention_override"}

    def graph_override(func, *args, **kwargs):
        return func(*args, **kwargs)

    kept = stock_attention.stock_attention_options(
        {"optimized_attention_override": graph_override})
    assert kept["optimized_attention_override"] is graph_override


def test_override_through_wrap_attn_equals_unpinned_stock_with_boxes():
    """Through the reproduction of comfy's wrap_attn, boxed q, k and v reach the
    backend as stock passes them with no pin, and the pin never runs."""
    calls = []

    def backend(q, k, v, heads, **kwargs):
        calls.append(sorted(kwargs))
        return q * 2 + k - v

    wrapped = _wrap_attn(backend)
    q, k, v = _seeded(1, 3, 4, seed=1), _seeded(1, 3, 4, seed=2), _seeded(1, 3, 4, seed=3)

    def boxes():
        return tuple(_AttentionTensorContainer(t.clone()) for t in (q, k, v))

    pin = Pin(True)
    options = stock_attention.stock_attention_options({})
    dropped = wrapped(*boxes(), 2, preferred_attention=pin, transformer_options=options)
    stock = wrapped(*boxes(), 2, preferred_attention=Pin(False), transformer_options={})
    assert pin.runs == 0
    assert torch.equal(dropped, stock)
    assert calls[0] == calls[1] == ["_inside_attn_wrapper", "transformer_options"]


def test_bound_forward_adds_the_override_and_notes_the_pin_once(stubs):
    from dgx_monarch.adapters import attention_inputs

    lines = []
    stubs.setattr(attention_inputs, "_noted_sites", set())
    stubs.setattr(attention_inputs.log, "debug", lambda *args: lines.append(args))
    pins = []
    module = StockAttention(pins, pinned=True)
    reference = StockAttention([], pinned=False)
    x = _seeded(1, 3, DIM, seed=4)
    stock_attention.bind_stock_attention(module, "a refiner")
    stock_attention.bind_stock_attention(module, "a refiner")  # idempotent
    out = module(x)
    out_with_options = module(x, transformer_options={"patches": {}})
    assert pins[0].runs == 0
    assert torch.equal(out, reference(x)) and torch.equal(out_with_options, out)
    assert len(lines) == 1 and lines[0][1] == "a refiner" and lines[0][3] is True
    # The line names what runs the call: comfy's default backend, not the widget.
    assert lines[0][4] == attention_inputs.STOCK_BACKEND_RUNS


# --- Wan 2.1 and 2.2, SCAIL, WanDancer ----------------------------------------


class _WanBlock(torch.nn.Module):
    """WanAttentionBlock: the pack binds self_attn; cross_attn stays stock."""

    def __init__(self, pins, pinned):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.cross_attn = StockAttention(pins, pinned)

    def forward(self, x, e=None, freqs=None, context=None, context_img_len=None,
                transformer_options=None):
        return self.cross_attn(x, context, transformer_options=transformer_options)


class _MusicSelfAttention(torch.nn.Module):
    """WanDancer's MusicSelfAttention with comfy's stock forward."""

    def __init__(self, pins, pinned, seed):
        super().__init__()
        self.num_heads, self.head_dim = 2, DIM // 2
        self.comfy_attention = Pin(pinned)
        pins.append(self.comfy_attention)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (
            torch.nn.Linear(DIM, DIM) for _ in range(4))
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                parameter.copy_(_seeded(*parameter.shape, seed=seed + index))

    def forward(self, x, freqs):
        from comfy.ldm.flux.math import apply_rope1
        from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = apply_rope1(self.q_proj(x).view(b, s, n, d), freqs)
        k = apply_rope1(self.k_proj(x).view(b, s, n, d), freqs)
        q = AttentionTensorContainer(q.view(b, s, n * d))
        k = AttentionTensorContainer(k.view(b, s, n * d))
        v = AttentionTensorContainer(self.v_proj(x).view(b, s, n * d))
        x = optimized_attention(q, k, v, heads=self.num_heads,
                                preferred_attention=self.comfy_attention)
        return self.out_proj(x)


class _MusicLayer(torch.nn.Module):
    def __init__(self, pins, pinned, seed):
        super().__init__()
        self.self_attn = _MusicSelfAttention(pins, pinned, seed)

    def forward(self, x, freqs):
        return x + self.self_attn(x, freqs=freqs)


class _FakeWan(torch.nn.Module):
    """WanModel, SCAILWanModel or WanDancerModel at one rank, identity embeds."""

    dim, freq_dim = DIM, 2

    def __init__(self, pins, pinned, music=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_WanBlock(pins, pinned) for _ in range(2))
        self.ref_conv = self.img_emb = self.img_emb_refimage = None
        self.music_encoder = torch.nn.ModuleList(
            _MusicLayer(pins, pinned, seed=10 * i) for i in range(2) if music)
        self.music_injector = types.SimpleNamespace(
            injected_block_id={0: 0},
            injector=[types.SimpleNamespace(o=lambda t: t, v=lambda t: t)])

    def patch_embedding(self, x):
        return x

    def time_embedding(self, e):
        return torch.zeros(e.shape[0], self.dim)

    def time_projection(self, e):
        return torch.zeros(*e.shape[:-1], 6 * self.dim)

    def text_embedding(self, context):
        return context

    def music_projection(self, audio):
        return audio

    def music_rope_embedder(self, ids):
        return torch.zeros(1, 1, ids.shape[1], 1)

    def head(self, x, _e):
        return x

    def unpatchify(self, x, _grid):
        return x


_WAN_X = _seeded(1, DIM, 1, 2, 3, seed=5)
_WAN_CONTEXT = _seeded(1, 5, DIM, seed=6)
_WAN_FREQS = torch.zeros(1, 6, 2)


@pytest.mark.parametrize("adapter_cls, module", [
    (wan_family.WanAdapter, wan_family),
    (wan_variants.SCAILAdapter, wan_variants),
    (wan_variants.WanDancerAdapter, wan_variants),
])
def test_wan_cross_attention_drops_the_pin(stubs, adapter_cls, module):
    def run(pinned):
        pins = []
        model = _FakeWan(pins, pinned)
        adapter_cls().inject_usp(model, _ctx())
        return model.forward_orig(_WAN_X, torch.tensor([0.5]), _WAN_CONTEXT,
                                  freqs=_WAN_FREQS, transformer_options={}), pins

    assert_pins_dropped(stubs, run, module)


def test_wandancer_music_encoder_drops_the_pin(stubs):
    """Every rank runs the music encoder whole, and its stock call hands comfy no
    options, so the pack binds a copy of its forward that passes no pin."""
    audio = _seeded(1, 3, DIM, seed=7)

    def run(pinned):
        pins = []
        model = _FakeWan(pins, pinned, music=True)
        wan_variants.WanDancerAdapter().inject_usp(model, _ctx())
        return model.forward_orig(_WAN_X, torch.tensor([0.5]), _WAN_CONTEXT,
                                  freqs=_WAN_FREQS, audio_embed=audio,
                                  transformer_options={}), pins

    def stock_music_forward(patch):
        patch.setattr(wan_variants, "music_self_attention", _MusicSelfAttention.forward)

    assert_pins_dropped(stubs, run, wan_variants, undo=stock_music_forward)


def test_wandancer_music_encoder_notes_that_the_default_backend_runs_it(stubs):
    """The debug line at a stock site names comfy's default backend, since
    neither the Init widget nor the dispatcher picks that call's kernel."""
    from dgx_monarch.adapters import attention_inputs

    lines = []
    stubs.setattr(attention_inputs, "_noted_sites", set())
    stubs.setattr(attention_inputs.log, "debug", lambda *args: lines.append(args))
    model = _FakeWan([], True, music=True)
    wan_variants.WanDancerAdapter().inject_usp(model, _ctx())
    for _ in range(2):
        model.forward_orig(_WAN_X, torch.tensor([0.5]), _WAN_CONTEXT, freqs=_WAN_FREQS,
                           audio_embed=_seeded(1, 3, DIM, seed=7), transformer_options={})
    music = [line for line in lines if line[1] == wan_variants._MUSIC_ATTENTION_SITE]
    assert len(music) == 1 and music[0][3] is True
    assert music[0][4] == attention_inputs.STOCK_BACKEND_RUNS
    assert "widget" not in music[0][4]


# --- Wan-Animate 2 -------------------------------------------------------------


class _Animate2Block(torch.nn.Module):
    def __init__(self, pins, pinned):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.cross_attn = StockAttention(pins, pinned)

    def forward_gen(self, x, e, freqs, context, k_pose, v_pose, f_gen, hw, buffers,
                    ref_strength=1.0, context_img_len=257, transformer_options=None):
        return self.cross_attn(x, context, transformer_options=transformer_options)


class _FakeAnimate2(_FakeWan):
    def __init__(self, pins, pinned):
        super().__init__([], pinned)
        self.blocks = torch.nn.ModuleList(_Animate2Block(pins, pinned) for _ in range(2))


def test_wan_animate2_cross_attention_drops_the_pin(stubs):
    flash = types.SimpleNamespace(effective_kernel="TORCH_FLASH")

    def run(pinned):
        pins = []
        model = _FakeAnimate2(pins, pinned)
        wan_animate2.WanAnimate2Adapter().inject_usp(model, _ctx(usp_attention=flash))
        return model.forward_orig(_WAN_X, torch.tensor([0.5]), _WAN_CONTEXT,
                                  freqs=_WAN_FREQS, transformer_options={}), pins

    assert_pins_dropped(stubs, run, wan_animate2)


# --- LTX and LTX audio-video ---------------------------------------------------


class _LtxBlock(torch.nn.Module):
    def __init__(self, pins, pinned, av):
        super().__init__()
        self.attn1 = torch.nn.Module()
        self.attn2 = StockAttention(pins, pinned)
        if av:
            self.audio_attn1 = torch.nn.Module()
            self.audio_to_video_attn = torch.nn.Module()
            self.video_to_audio_attn = torch.nn.Module()
            self.audio_attn2 = StockAttention(pins, pinned)


class _ConnectorBlock(torch.nn.Module):
    """BasicTransformerBlock1D: its stock attention gets no options."""

    def __init__(self, pins, pinned):
        super().__init__()
        self.attn1 = StockAttention(pins, pinned)

    def forward(self, x, attention_mask=None, pe=None):
        return self.attn1(x, mask=attention_mask, pe=pe)


class _FakeLtx(torch.nn.Module):
    def __init__(self, pins, pinned, av):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            _LtxBlock(pins, pinned, av) for _ in range(2))
        if av:
            self.audio_inner_dim = DIM
            for name in ("video_embeddings_connector", "audio_embeddings_connector"):
                setattr(self, name, types.SimpleNamespace(transformer_1d_blocks=[
                    _ConnectorBlock(pins, pinned) for _ in range(2)]))

    def preprocess_text_embeds(self, context):
        """What comfy's extra_conds runs: both connectors, with no options."""
        outs = []
        for connector in (self.video_embeddings_connector, self.audio_embeddings_connector):
            hidden = context
            for block in connector.transformer_1d_blocks:
                hidden = block(hidden)
            outs.append(hidden)
        return torch.cat(outs, dim=-1)

    def _process_transformer_blocks(self, x, context, attention_mask, timestep, pe,
                                    transformer_options=None, self_attention_mask=None,
                                    **kwargs):
        for block in self.transformer_blocks:
            if isinstance(x, list):
                x = [block.attn2(x[0], context, transformer_options=transformer_options),
                     block.audio_attn2(x[1], context, transformer_options=transformer_options)]
            else:
                x = block.attn2(x, context, transformer_options=transformer_options)
        return x


@pytest.mark.parametrize("av", [False, True])
def test_ltx_text_cross_attention_and_connectors_drop_the_pin(stubs, av):
    stubs.setitem(sys.modules, "comfy.ldm.lightricks.av_model", types.SimpleNamespace(
        CompressedTimestep=type("CompressedTimestep", (), {})))
    video, audio = _seeded(1, 6, DIM, seed=8), _seeded(1, 3, DIM, seed=9)
    context = _seeded(1, 4, DIM, seed=10)
    pe = (torch.zeros(1, 6, 1, 1, 2, 2), "split")
    audio_pe = (torch.zeros(1, 3, 1, 1, 2, 2), "split")

    def run(pinned):
        pins = []
        model = _FakeLtx(pins, pinned, av)
        ltx.LTXAdapter().inject_usp(model, _ctx())
        if not av:
            return model._process_transformer_blocks(
                video, context, None, None, pe, transformer_options={}), pins
        out = model._process_transformer_blocks(
            [video, audio], context, None, [None] * 5,
            [(pe, pe), (audio_pe, audio_pe)], transformer_options={})
        text = model.preprocess_text_embeds(context)
        return torch.cat([out[0], out[1], text[:, :, :DIM]], dim=1), pins

    assert_pins_dropped(stubs, run, ltx)


# --- Kandinsky 5 -----------------------------------------------------------------


class _K5TextBlock(torch.nn.Module):
    def __init__(self, pins, pinned):
        super().__init__()
        self.self_attention = StockAttention(pins, pinned)

    def forward(self, x, time_embed, freqs, transformer_options=None):
        return self.self_attention(x, transformer_options=transformer_options)


class _K5VisualBlock(torch.nn.Module):
    def __init__(self, pins, pinned):
        super().__init__()
        self.self_attention = torch.nn.Module()
        self.cross_attention = StockAttention(pins, pinned)

    def forward(self, visual_embed, text_embed, time_embed, freqs=None,
                transformer_options=None):
        return self.cross_attention(visual_embed, text_embed,
                                    transformer_options=transformer_options)


class _FakeKandinsky5(torch.nn.Module):
    def __init__(self, pins, pinned):
        super().__init__()
        self.text_transformer_blocks = torch.nn.ModuleList(
            _K5TextBlock(pins, pinned) for _ in range(2))
        self.visual_transformer_blocks = torch.nn.ModuleList(
            _K5VisualBlock(pins, pinned) for _ in range(2))

    def text_embeddings(self, context):
        return context

    def time_embeddings(self, timestep, _dtype):
        return torch.zeros(timestep.shape[0], DIM)

    def pooled_text_embeddings(self, y):
        return torch.zeros(y.shape[0], DIM)

    def visual_embeddings(self, x):
        return x

    def out_layer(self, visual_embed, _time_embed):
        return visual_embed


def test_kandinsky5_text_blocks_and_cross_attention_drop_the_pin(stubs):
    x, context = _seeded(1, 1, 2, 3, DIM, seed=11), _seeded(1, 5, DIM, seed=12)

    def run(pinned):
        pins = []
        model = _FakeKandinsky5(pins, pinned)
        kandinsky5.Kandinsky5Adapter().inject_usp(model, _ctx())
        return model.forward_orig(x, torch.tensor([0.5]), context, torch.zeros(1, 2),
                                  torch.zeros(1, 6, 2), torch.zeros(1, 5, 2),
                                  transformer_options={}), pins

    assert_pins_dropped(stubs, run, kandinsky5)


# --- Anima ---------------------------------------------------------------------------


class _AnimaBlock(torch.nn.Module):
    def __init__(self, pins, pinned):
        super().__init__()
        self.self_attn = types.SimpleNamespace()
        self.cross_attn = StockAttention(pins, pinned)

    def forward(self, x, t_emb, context, transformer_options=None, **_kwargs):
        rows = x.flatten(1, 3)
        out = self.cross_attn(rows, context, transformer_options=transformer_options)
        return out.view(x.shape)


class _FakeAnima(torch.nn.Module):
    patch_temporal, patch_spatial = 1, 1

    def __init__(self, pins, pinned):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_AnimaBlock(pins, pinned) for _ in range(2))
        self.t_embedder = [lambda t: torch.zeros(t.shape[0], 1, DIM),
                           lambda e: (e, None)]

    def prepare_embedded_sequence(self, x, fps=None, padding_mask=None):
        t, h, w = x.shape[1:4]
        return x, torch.zeros(t * h * w, 1, 2, 2), None

    def t_embedding_norm(self, t):
        return t

    def final_layer(self, x, _t, adaln_lora_B_T_3D=None):
        return x

    def unpatchify(self, x):
        return x


def test_anima_cross_attention_drops_the_pin(stubs):
    x, context = _seeded(1, 1, 2, 3, DIM, seed=13), _seeded(1, 5, DIM, seed=14)

    def run(pinned):
        pins = []
        model = _FakeAnima(pins, pinned)
        anima.AnimaAdapter().inject_usp(model, _ctx())
        return model._forward(x, torch.tensor([0.5]), context,
                              transformer_options={}), pins

    assert_pins_dropped(stubs, run, anima)


# --- Z-Image -------------------------------------------------------------------------


class _Refiner(torch.nn.Module):
    """A refiner layer whose stock attention gets the options it is handed."""

    def __init__(self, pins, pinned):
        super().__init__()
        self.attention = StockAttention(pins, pinned)

    def forward(self, x, *_args, transformer_options=None, **_kwargs):
        return self.attention(x, transformer_options=transformer_options)


class _FakeZImage(torch.nn.Module):
    n_heads, time_scale, patch_size = 2, 1000.0, 1
    clip_text_pooled_proj = None

    def __init__(self, pins, pinned):
        super().__init__()
        self.context_refiner = torch.nn.ModuleList(_Refiner(pins, pinned) for _ in range(2))
        self.noise_refiner = torch.nn.ModuleList(_Refiner(pins, pinned) for _ in range(2))
        self.layers = torch.nn.ModuleList(_Identity() for _ in range(2))

    def t_embedder(self, t, dtype=None):
        return torch.zeros(t.shape[0], DIM)

    def patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens, ref_latents=(),
                           ref_contexts=(), siglip_feats=(), transformer_options=None):
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, None, transformer_options=transformer_options)
        img = x.flatten(2).transpose(1, 2)
        for layer in self.noise_refiner:
            img = layer(img, None, None, t, transformer_options=transformer_options)
        joint = torch.cat((cap_feats, img), dim=1)
        return joint, None, None, None, torch.zeros(1, joint.shape[1], 2), None

    def final_layer(self, img, _adaln, timestep_zero_index=None):
        return img

    def unpatchify(self, img, _img_size, _cap_size, return_tensor=True):
        return img.unsqueeze(-1)


def test_zimage_refiners_drop_the_pin(stubs):
    x, context = _seeded(1, DIM, 4, 4, seed=15), _seeded(1, 16, DIM, seed=16)

    def run(pinned):
        pins = []
        model = _FakeZImage(pins, pinned)
        zimage.ZImageAdapter().inject_usp(model, _ctx())
        return model._forward(x, torch.tensor([0.5]), context, 16,
                              transformer_options={}), pins

    assert_pins_dropped(stubs, run, zimage)


# --- OmniGen2 --------------------------------------------------------------------------


class _FakeOmniGen2(torch.nn.Module):
    patch_size = 1

    def __init__(self, pins, pinned):
        super().__init__()
        self.context_refiner = torch.nn.ModuleList(_Refiner(pins, pinned) for _ in range(2))
        self.noise_refiner = torch.nn.ModuleList(_Refiner(pins, pinned) for _ in range(2))
        self.ref_image_refiner = torch.nn.ModuleList(_Refiner(pins, pinned) for _ in range(2))
        self.layers = torch.nn.ModuleList(_Identity() for _ in range(2))

    def time_caption_embed(self, timestep, text, _dtype):
        return torch.zeros(timestep.shape[0], DIM), text

    def flat_and_pad_to_seq(self, hidden_states, ref):
        flat = hidden_states.flatten(2).transpose(1, 2)
        return (flat, ref.flatten(2).transpose(1, 2), None, None, [[ref.shape[-1]]],
                [flat.shape[1]], [None], [hidden_states.shape[-2:]])

    def rope_embedder(self, batch, text_len, _num_tokens, l_ref, l_img, *_args):
        total = text_len + l_ref[0][0] + l_img[0]
        return None, None, None, torch.zeros(batch, total, 2), [text_len], [total]

    def img_patch_embed_and_refine(self, hidden, ref, _img_mask, _ref_mask, _noise_rot,
                                   _ref_rot, _l_ref, _l_img, temb, transformer_options=None):
        for layer in self.noise_refiner:
            hidden = layer(hidden, transformer_options=transformer_options)
        for layer in self.ref_image_refiner:
            ref = layer(ref, transformer_options=transformer_options)
        return torch.cat((ref, hidden), dim=1)

    def norm_out(self, hidden_states, _temb):
        return hidden_states


def test_omnigen2_refiners_drop_the_pin(stubs):
    x, context = _seeded(1, DIM, 2, 3, seed=17), _seeded(1, 5, DIM, seed=18)
    ref = _seeded(1, DIM, 1, 2, seed=19)

    def run(pinned):
        pins = []
        model = _FakeOmniGen2(pins, pinned)
        omnigen2.Omnigen2Adapter().inject_usp(model, _ctx())
        return model.forward(x, torch.tensor([0.5]), context, 5, ref_latents=ref,
                             transformer_options={}), pins

    assert_pins_dropped(stubs, run, omnigen2)


# --- Boogu -------------------------------------------------------------------------------


def test_boogu_context_refiner_drops_the_pin(stubs):
    from test_boogu_adapter import _FakeBoogu

    context = _seeded(1, 3, DIM, seed=20)

    def run(pinned):
        pins = []
        model = _FakeBoogu([], with_ref=False)
        model.context_refiner = [_Refiner(pins, pinned) for _ in range(2)]
        model.double_stream_layers = [lambda img, instruct, *_a, **_k: _mix_streams(img, instruct)]
        boogu.BooguAdapter().inject_usp(model, _ctx(usp_attention=object()))
        return model.forward(torch.zeros(1, 1, 4, 4), torch.zeros(1), context, 3,
                             transformer_options={}), pins

    assert_pins_dropped(stubs, run, boogu)


# --- HunyuanImage 2.1 and HunyuanVideo 1.5 -------------------------------------------------


class _TokenRefiner(torch.nn.Module):
    def __init__(self, pins, pinned):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Refiner(pins, pinned) for _ in range(2))

    def forward(self, x, timesteps, mask, transformer_options=None):
        for block in self.blocks:
            x = block(x, transformer_options=transformer_options)
        return x


class _HunyuanBlock(torch.nn.Module):
    def forward(self, img=None, txt=None, **_kwargs):
        return _mix_streams(img, txt) if txt is not None else img


class _FakeHunyuan(torch.nn.Module):
    patch_size, out_channels = [1, 1], DIM
    params = types.SimpleNamespace(guidance_embed=False)
    time_r_in = vector_in = cond_type_embedding = byt5_in = None

    def __init__(self, pins, pinned):
        super().__init__()
        self.txt_in = _TokenRefiner(pins, pinned)
        self.double_blocks = torch.nn.ModuleList(_HunyuanBlock() for _ in range(2))
        self.single_blocks = torch.nn.ModuleList(_HunyuanBlock() for _ in range(2))

    def img_in(self, img):
        return img.flatten(2).transpose(1, 2)

    def time_in(self, t):
        return torch.zeros(t.shape[0], DIM)

    def pe_embedder(self, ids):
        return torch.zeros(1, ids.shape[1], 2)

    def final_layer(self, img, _vec, modulation_dims=None):
        return img


def test_hunyuan_token_refiner_drops_the_pin(stubs):
    img, txt = _seeded(1, DIM, 2, 3, seed=21), _seeded(1, 5, DIM, seed=22)

    def run(pinned):
        pins = []
        model = _FakeHunyuan(pins, pinned)
        hunyuan.HunyuanAdapter().inject_usp(model, _ctx())
        return model.forward_orig(img, torch.zeros(1, 6, 3), txt, torch.zeros(1, 5, 3),
                                  None, torch.tensor([0.5]), transformer_options={}), pins

    assert_pins_dropped(stubs, run, hunyuan)


# --- MiniMax H3 -------------------------------------------------------------------------


class _H3RefinerBlock(torch.nn.Module):
    """comfy's RefinerBlock: the attention gets the block's options."""

    def __init__(self, pins, pinned):
        super().__init__()
        self.attn = StockAttention(pins, pinned)

    def forward(self, x, transformer_options=None):
        return self.attn(x, transformer_options=transformer_options)


class _FakeH3(torch.nn.Module):
    def __init__(self, pins, pinned):
        super().__init__()
        self.blocks = []
        self.token_refiner = types.SimpleNamespace(blocks=[
            _H3RefinerBlock(pins, pinned) for _ in range(2)])

    def refine(self, text, transformer_options=None):
        """comfy's token refiner loop: from extra_conds with no options, or
        from the forward with the stock options."""
        kwargs = {} if transformer_options is None else {
            "transformer_options": transformer_options}
        for block in self.token_refiner.blocks:
            text = block(text, **kwargs)
        return text


def test_minimax_h3_token_refiner_drops_the_pin(stubs):
    text = _seeded(1, 5, DIM, seed=23)

    def run(pinned):
        pins = []
        model = _FakeH3(pins, pinned)
        minimax_h3.MiniMaxH3Adapter().inject_usp(model, _ctx())
        return torch.cat([model.refine(text), model.refine(text, {})], dim=1), pins

    assert_pins_dropped(stubs, run, minimax_h3)


# --- Krea2 and Kroma -----------------------------------------------------------------------


class _TextFusion(torch.nn.Module):
    """Krea2's text fusion: layerwise blocks across layers, then refiners."""

    def __init__(self, pins, pinned, refiner_pins):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(_Refiner(pins, pinned) for _ in range(2))
        self.refiner_blocks = torch.nn.ModuleList(
            _Refiner(refiner_pins, pinned) for _ in range(2))

    def projector(self, tokens):
        return tokens.sum(dim=-1, keepdim=True)

    def forward(self, context, mask=None, transformer_options=None):
        b, rows, layers, dim = context.shape
        tokens = context.reshape(b * rows, layers, dim)
        for block in self.layerwise_blocks:
            tokens = block(tokens.contiguous(), mask=None, transformer_options=transformer_options)
        tokens = self.projector(tokens.reshape(b, rows, layers, dim).transpose(2, 3)).squeeze(-1)
        for block in self.refiner_blocks:
            tokens = block(tokens, mask=None, transformer_options=transformer_options)
        return tokens


class _FakeKrea2(torch.nn.Module):
    patch, tdim, channels = 1, 2, DIM

    def __init__(self, pins, pinned, pure):
        super().__init__()
        # Under ring the refiner blocks carry the sharded override, not a stock call.
        self.txtfusion = _TextFusion(pins, pinned, pins if pure else [])
        self.blocks = torch.nn.ModuleList(_Mix() for _ in range(2))

    def _unpack_context(self, context):
        return context

    def first(self, img):
        return img

    def tmlp(self, t):
        return t

    def tproj(self, t):
        return t

    def txtmlp(self, context):
        return context

    def pe_embedder(self, pos):
        return torch.zeros(pos.shape[0], pos.shape[1], 2)

    def last(self, combined, _t):
        return combined


@pytest.mark.parametrize("pure", [True, False])
def test_krea2_text_fusion_drops_the_pin(stubs, pure):
    x, context = _seeded(1, DIM, 2, 3, seed=24), _seeded(1, 5, 3, DIM, seed=25)

    def run(pinned):
        pins = []
        model = _FakeKrea2(pins, pinned, pure)
        krea2.Krea2Adapter().inject_usp(model, _ctx(pure_ulysses=pure))
        return model._forward(x, torch.tensor([0.5]), context, transformer_options={}), pins

    assert_pins_dropped(stubs, run, krea2)
