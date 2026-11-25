"""Wan, SCAIL, SCAIL-2 and WanDancer pad rows: excluded under Ulysses, refused on ring.

These forwards shard one token stream (video tokens, then SCAIL's pose tokens)
and its RoPE and per-token modulation rows identically. An odd token count
leaves a pad row at the stream's tail on the last rank. Its RoPE row is zero,
so its key is a zero vector, which still takes softmax weight (logit 0) and
carries its value into every real row: stock one-GPU attention has no such
key. The forward names the pad rows on every topology; pure Ulysses drops them
at its full-axis point, and ring or hybrid refuses class K unless the ring_pad
waiver is granted.

A single stream chunked in order gathers in stock's key order, so every
self-attention call on every rank must receive stock's q, k and v. The harness
is tests/pad_rows_comfy_helpers.py: comfy's own models at toy size in float32,
the widest dtype comfy-kitchen's RoPE kernel takes, with NaN pad rows.
WanDancer's valid geometries hold a multiple of eight tokens per frame group,
so they never pad at degree 2 or 4; it runs at degree 3.
"""
from __future__ import annotations

import re

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.wan_family import WanAdapter
from dgx_monarch.adapters.wan_variants import SCAIL2Adapter, SCAILAdapter, WanDancerAdapter
from pad_rows_comfy_helpers import (
    WORLD,
    ScaleRecorder,
    assert_ring_pad_refusal,
    assert_scales_match,
    fabric_max,
    fill_parameters,
    grant_ring_pad_waiver,
    install,
    nan_pads,
    plain,
    real_comfy,
    replicas,
    run_ranks,
    same,
)

CHANNELS, TEXT_DIM, TEXT_ROWS = 4, 12, 7
COMMON = {"patch_size": (1, 2, 2), "text_len": TEXT_ROWS, "in_dim": CHANNELS, "dim": 32,
          "ffn_dim": 64, "freq_dim": 16, "text_dim": TEXT_DIM, "out_dim": CHANNELS,
          "num_heads": 2, "num_layers": 2, "dtype": torch.float32, "device": "cpu"}


@pytest.fixture(scope="module")
def comfy_wan():
    with real_comfy("wan-pad-exclusion", "comfy.ldm.wan.model",
                    "comfy.ldm.wan.model_wandancer") as modules:
        yield modules


def _model(comfy_wan, kind):
    ops, wan, dancer = comfy_wan
    settings = dict(COMMON, operations=ops.disable_weight_init)
    if kind == "wan":
        model = wan.WanModel(model_type="t2v", **settings)
    elif kind == "scail":
        model = wan.SCAILWanModel(**settings)
    elif kind == "scail2":
        model = wan.SCAIL2WanModel(mask_in_dim=3, **settings)
    else:
        model = dancer.WanDancerModel(audio_inject_layers=[0, 1], music_dim=16, music_heads=2,
                                      music_feature_dim=5, music_latent_dim=16, **settings)
    fill_parameters(model, seed=11)
    return model


ADAPTERS = {"wan": WanAdapter, "scail": SCAILAdapter, "scail2": SCAIL2Adapter,
            "wan_dancer": WanDancerAdapter}


def _inputs(kind, latent, batch, *, per_frame=False):
    generator = torch.Generator().manual_seed(100 * batch + sum(latent) + len(kind))

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    x = randn(batch, CHANNELS, *latent)
    frames = latent[0]
    if per_frame:
        timestep = torch.linspace(300.0, 700.0, frames, dtype=torch.float32).repeat(batch, 1)
    else:
        timestep = torch.tensor([400.0, 700.0], dtype=torch.float32)[:batch]
    context = randn(batch, TEXT_ROWS, TEXT_DIM)
    kwargs = {}
    if kind in ("scail", "scail2"):
        # Pose tokens join after the video tokens: half-resolution pose frames.
        kwargs["pose_latents"] = randn(batch, CHANNELS, frames, latent[1] // 2, latent[2] // 2)
    if kind == "scail2":
        kwargs["sam_latents"] = randn(batch, 3, frames, latent[1] // 2, latent[2] // 2)
    if kind == "wan_dancer":
        kwargs["audio_embed"] = randn(batch, 4 * frames, 5)
    return (x, timestep, context), kwargs


def _tokens(kind, latent):
    frames, height, width = latent
    video = frames * (height // 2) * (width // 2)
    if kind in ("scail", "scail2"):
        video += frames * (height // 4) * (width // 4)
    return video


def _stock(monkeypatch, comfy_wan, model, args, kwargs, tokens):
    """Stock's class forward, recording each self-attention call's operands."""
    _ops, wan, _dancer = comfy_wan
    kernel = wan.optimized_attention
    calls: list[tuple] = []

    def record(q, k, v, heads, **options):
        q, k, v = (plain(t) for t in (q, k, v))
        if q.shape[1] == k.shape[1] == tokens:
            calls.append((q.clone(), k.clone(), v.clone()))
        return kernel(q, k, v, heads)

    monkeypatch.setattr(wan, "optimized_attention", record)
    try:
        with torch.no_grad():
            out = type(model)._forward(model, *args, transformer_options={}, **kwargs)
    finally:
        monkeypatch.setattr(wan, "optimized_attention", kernel)
    return out, calls, kernel


def _sharded(monkeypatch, kind, model, args, kwargs, kernel, *, world=WORLD,
             pure_ulysses=True, ring_world=1):
    harness = install(monkeypatch, kernel, ring_world=ring_world, world=world)
    models = replicas(model, world)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    monkeypatch.setattr("dgx_monarch.adapters.wan_variants._requested_ring_degree",
                        lambda: ring_world)
    for rank_model in models:
        ADAPTERS[kind]().inject_usp(rank_model, InjectionContext(
            topology_sp=world, usp_attention=dispatch, pure_ulysses=pure_ulysses))
    outputs = run_ranks(harness, models,
                        lambda m: m._forward(*args, transformer_options={}, **kwargs))
    return outputs, harness


# (kind, latent frames/height/width, per-frame timesteps, world)
CASES = [
    ("wan", (2, 6, 6), False, 2),       # 18 tokens: nothing pads
    ("wan", (3, 6, 10), False, 2),      # 45 tokens
    ("wan", (3, 6, 10), True, 2),       # per-frame modulation rows shard too
    ("wan", (1, 6, 6), False, 2),       # 9 tokens
    ("scail", (1, 8, 12), False, 2),    # 24 video + 6 pose: nothing pads
    ("scail", (1, 8, 8), False, 2),     # 16 video + 4 pose: nothing pads
    ("scail", (3, 8, 12), False, 4),    # 72 + 18 = 90 rows on four ranks
    ("scail2", (1, 12, 12), False, 2),  # 36 + 9 = 45
    ("wan_dancer", (1, 8, 8), False, 3),    # 16 rows on three ranks
    ("wan_dancer", (2, 8, 8), False, 3),    # 32 rows on three ranks
]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("kind,latent,per_frame,world", CASES)
def test_pure_ulysses_attends_stock_keys_and_returns_stock_output(
    monkeypatch, comfy_wan, kind, latent, per_frame, world, batch
):
    model = _model(comfy_wan, kind)
    args, kwargs = _inputs(kind, latent, batch, per_frame=per_frame)
    tokens = _tokens(kind, latent)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs, tokens)
    nan_pads(monkeypatch)
    outputs, harness = _sharded(monkeypatch, kind, model, args, kwargs, kernel, world=world)

    assert len(stock_calls) == 2
    for rank in range(world):
        seen = harness.seen[rank]
        assert len(seen) == len(stock_calls)
        for (_entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_calls, strict=True):
            assert k.shape == stock_k.shape
            assert same(k, stock_k) and same(v, stock_v) and same(q, stock_q)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert not torch.isnan(outputs[0]).any()
    assert same(outputs[0], expected)


def _scale_run(monkeypatch, comfy_wan, kind, latent, per_frame, world, *, loud_context=False):
    model = _model(comfy_wan, kind)
    args, kwargs = _inputs(kind, latent, 1, per_frame=per_frame)
    if loud_context:
        # The text context's last row holds every key and value input's largest
        # value, so a scale that trimmed the replicated context as if it were the
        # sharded stream would lose it.
        args[2][:, -1] *= 100.0
    recorder = ScaleRecorder(monkeypatch)
    stock = replicas(model, 1)[0]
    assert ScaleRecorder.install(stock) > 0
    with torch.no_grad():
        type(stock)._forward(stock, *args, transformer_options={}, **kwargs)
    expected = recorder.take()[0]
    nan_pads(monkeypatch)
    harness = install(monkeypatch, comfy_wan[1].optimized_attention, world=world)
    models = replicas(model, world)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    monkeypatch.setattr("dgx_monarch.adapters.wan_variants._requested_ring_degree", lambda: 1)
    for rank_model in models:
        ADAPTERS[kind]().inject_usp(rank_model, InjectionContext(
            topology_sp=world, usp_attention=dispatch, pure_ulysses=True))
        ScaleRecorder.install(rank_model, fabric_max(harness))
    run_ranks(harness, models, lambda m: m._forward(*args, transformer_options={}, **kwargs))
    # WanDancer's audio attention has one key per frame group, so the sharded
    # forward skips its query and key projections (adapters/wan_variants.py).
    skipped = frozenset(f"music_injector.injector.{index}.{name}"
                        for index in range(2) for name in ("q", "k"))
    assert_scales_match(expected, recorder.take(), world,
                        stock_only=skipped if kind == "wan_dancer" else frozenset())


@pytest.mark.parametrize("kind,latent,per_frame,world", CASES)
def test_the_shared_nvfp4_scale_counts_only_real_rows(
    monkeypatch, comfy_wan, kind, latent, per_frame, world
):
    _scale_run(monkeypatch, comfy_wan, kind, latent, per_frame, world)


def test_a_renamed_module_refuses_rather_than_let_pads_into_the_scale(monkeypatch, comfy_wan):
    """A ComfyUI that renamed Wan's ``blocks`` would leave the role table naming
    none of the block Linears, and each would take its whole padded input into
    the scale. Modeled here by registering the same blocks under another name
    (the forward still reaches them as ``self.blocks``): every rank refuses,
    class P, at its first quantized Linear, before any reduction."""
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    model = _model(comfy_wan, "wan")
    args, kwargs = _inputs("wan", (3, 6, 10), 1)
    recorder = ScaleRecorder(monkeypatch)
    harness = install(monkeypatch, comfy_wan[1].optimized_attention, world=2)
    models = replicas(model, 2)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for rank_model in models:
        blocks = rank_model._modules.pop("blocks")
        rank_model._modules["transformer_blocks"] = blocks
        object.__setattr__(rank_model, "blocks", blocks)
        ADAPTERS["wan"]().inject_usp(rank_model, InjectionContext(
            topology_sp=2, usp_attention=dispatch, pure_ulysses=True))
        ScaleRecorder.install(rank_model, fabric_max(harness))
    assert any(path.startswith("transformer_blocks.") for path, _ in models[0].named_modules())
    with pytest.raises(base.UnsupportedModelError) as caught:
        run_ranks(harness, models, lambda m: m._forward(*args, transformer_options={}, **kwargs))
    assert parse_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "sharded stream 'x'" in str(caught.value)
    # The embeddings before the loop reduce their replicated inputs; no block
    # Linear reduced a padded input.
    calls = recorder.take()
    assert calls and not any(path.startswith("transformer_blocks.")
                             for rank in calls.values() for path, _amax in rank)


def test_the_replicated_context_keeps_every_row_in_its_scale(monkeypatch, comfy_wan):
    """13 video tokens shard to 7 local rows, the text context's own row count:
    the cross-attention keys and values are named replicated by module, so
    their scale reads all seven context rows on every rank."""
    _scale_run(monkeypatch, comfy_wan, "wan", (13, 2, 2), False, 2, loud_context=True)


@pytest.mark.parametrize("kind,latent", [("wan", (3, 6, 10)), ("scail", (1, 12, 12)),
                                         ("scail2", (1, 12, 12))])
def test_a_padded_ring_render_refuses_class_k_by_default(monkeypatch, comfy_wan, kind, latent):
    model = _model(comfy_wan, kind)
    args, kwargs = _inputs(kind, latent, 1)
    _expected, _calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs,
                                       _tokens(kind, latent))
    with pytest.raises(base.UnsupportedModelError) as caught:
        _sharded(monkeypatch, kind, model, args, kwargs, kernel, pure_ulysses=False,
                 ring_world=2)
    assert_ring_pad_refusal(caught.value)
    # Raised before the block loop, naming what to change, every remedy one that works.
    message = str(caught.value)
    assert "Use uly2" in message and "patch taller" not in message
    named = [int(n) for n in re.findall(r"(\d+) frames", message)]
    assert bool(named) is (latent[0] > 1)
    for count in named:   # each named length's whole stream divides, and none is the current
        frames = (count - 1) // 4 + 1
        assert frames != latent[0] and _tokens(kind, (frames, *latent[1:])) % 2 == 0


def test_the_ring_refusal_names_the_nearest_frame_counts_that_divide():
    """Wan at 81 frames and 624 square: 21 x 39 x 39 patched tokens, odd, so two
    ranks pad; 77 frames (20 latent frames) and 85 (22) divide. SCAIL-2 at 864 x
    480 with one reference frame: (1 + 21) x 30 x 54 video rows and 21 x 15 x 27
    pose rows, 44145, odd; each frame adds 1620 + 405 rows, the reference none."""
    from dgx_monarch.adapters.wan_family import dividing_frames, stream_rows_at

    assert dividing_frames(stream_rows_at(21 * 39 * 39, 21, 39 * 39), 21, 2) == [77, 85]
    assert dividing_frames(stream_rows_at(22 * 1620 + 21 * 405, 21, 1620 + 405), 21, 2) \
        == [77, 85]
    # Four ranks: neither 50 nor 25 tokens divide, so only frames above do.
    assert dividing_frames(stream_rows_at(75, 3, 25), 3, 4) == [13]
    assert dividing_frames(stream_rows_at(25, 1, 25), 1, 2) == [5]


@pytest.mark.parametrize(("heads", "uly4"), [(40, True), (6, False)])
def test_at_four_ranks_the_remedy_names_uly4_only_where_the_heads_divide(monkeypatch, heads,
                                                                         uly4):
    """auto on four machines sends these families to world 4: the refusal offers
    pure Ulysses there only when the heads tile it, and no canvas hint."""
    from dgx_monarch.adapters import wan_family

    monkeypatch.setattr(wan_family, "_ring_degree", lambda: 2)
    rows = 21 * 41 * 41
    with pytest.raises(base.UnsupportedModelError) as caught:
        wan_family.refuse_padded_ring("wan", rows, 4, 3, heads,
                                      wan_family.stream_rows_at(rows, 21, 41 * 41), 21)
    message = str(caught.value)
    assert ("Use uly4" in message) is uly4 and "uly2" not in message
    assert "patch taller" not in message and "topology 'single'" in message
    for count in (int(n) for n in re.findall(r"(\d+) frames", message)):
        assert ((count - 1) // 4 + 1) * 41 * 41 % 4 == 0


# A reference latent (one frame, before the video on the time axis) and, for
# SCAIL-2, its mask latents, as the shipped SCAIL workflows carry them.
def _with_reference(kind, kwargs, frames):
    kwargs["reference_latent"] = torch.randn(1, CHANNELS, 1, 6, 6)
    if kind == "scail2":
        kwargs["ref_mask_latents"] = torch.randn(1, 3, 1 + frames, 6, 6)
    return (1 + frames) * 9 + frames * 4   # video rows with the reference, pose rows


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize(("kind", "frames"), [("scail", 2), ("scail2", 2), ("scail", 3),
                                              ("scail2", 1)])
def test_scail_with_a_reference_attends_stock_keys_on_pure_ulysses(monkeypatch, comfy_wan,
                                                                   kind, frames, world):
    model = _model(comfy_wan, kind)
    args, kwargs = _inputs(kind, (frames, 6, 6), 1)
    rows = _with_reference(kind, kwargs, frames)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs, rows)
    nan_pads(monkeypatch)
    outputs, harness = _sharded(monkeypatch, kind, model, args, kwargs, kernel, world=world)
    assert len(stock_calls) == COMMON["num_layers"]
    for rank in range(world):
        for (_e, q, k, v), (sq, sk, sv) in zip(harness.seen[rank], stock_calls, strict=True):
            assert k.shape == sk.shape and same(k, sk) and same(v, sv) and same(q, sq)
    assert not torch.isnan(outputs[0]).any() and same(outputs[0], expected)


@pytest.mark.parametrize("kind", ["scail", "scail2"])
def test_scail_ring_refusal_with_a_reference_names_lengths_that_divide(monkeypatch, comfy_wan,
                                                                       kind):
    """The reference frame sits in the grid and stays fixed; the pose rows grow
    with the video. 5 video frames (2 latent) and a reference make 35 rows."""
    model = _model(comfy_wan, kind)
    args, kwargs = _inputs(kind, (2, 6, 6), 1)
    rows = _with_reference(kind, kwargs, 2)
    assert rows % 2 == 1
    _expected, _calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs, rows)
    with pytest.raises(base.UnsupportedModelError) as caught:
        _sharded(monkeypatch, kind, model, args, kwargs, kernel, pure_ulysses=False,
                 ring_world=2)
    named = [int(n) for n in re.findall(r"(\d+) frames", str(caught.value))]
    assert named == [1, 9]
    for count in named:
        frames = (count - 1) // 4 + 1
        assert ((1 + frames) * 9 + frames * 4) % 2 == 0


@pytest.mark.parametrize("kind", ["scail", "scail2"])
def test_scail_ring_refusal_counts_an_odd_pose_row_count_per_frame(monkeypatch, comfy_wan,
                                                                   kind):
    """A 10 x 10 latent gives 25 video rows a frame, and its half-resolution pose
    pads 5 x 5 to 6 x 6, 9 rows a frame. With the reference frame the stream is
    25 + 34 rows a frame, odd at every frame count, so at world 2 no frame count
    divides and the refusal names none; one that left the pose rows out would
    name counts that still pad."""
    def rows(frames):
        return (1 + frames) * 25 + frames * 9

    model = _model(comfy_wan, kind)
    args, kwargs = _inputs(kind, (2, 10, 10), 1)
    kwargs["reference_latent"] = torch.randn(1, CHANNELS, 1, 10, 10)
    if kind == "scail2":
        kwargs["ref_mask_latents"] = torch.randn(1, 3, 3, 10, 10)
    assert rows(2) % 2 == 1
    _expected, _calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs, rows(2))
    with pytest.raises(base.UnsupportedModelError) as caught:
        _sharded(monkeypatch, kind, model, args, kwargs, kernel, pure_ulysses=False,
                 ring_world=2)
    assert re.findall(r"(\d+) frames", str(caught.value)) == []


@pytest.mark.parametrize("nest", [True, False])
@pytest.mark.parametrize("kind", ["wan", "scail"])
def test_a_forward_reentered_from_a_block_patch_keeps_the_outer_pads(monkeypatch, comfy_wan,
                                                                     kind, nest):
    """A block patch that runs the model once more, as a guidance or preview
    patch might, re-enters the forward, which names its own pad rows; on return
    the outer forward finds its own rows again, so no pad (NaN here) reaches
    its output and the video is stock's."""
    model = _model(comfy_wan, kind)
    latent = (3, 6, 10) if kind == "wan" else (1, 12, 12)
    args, kwargs = _inputs(kind, latent, 1)
    expected, _calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs,
                                      _tokens(kind, latent))
    nan_pads(monkeypatch)
    harness = install(monkeypatch, kernel, world=2)
    models = replicas(model, 2)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    monkeypatch.setattr("dgx_monarch.adapters.wan_variants._requested_ring_degree", lambda: 1)
    for rank_model in models:
        ADAPTERS[kind]().inject_usp(rank_model, InjectionContext(
            topology_sp=2, usp_attention=dispatch, pure_ulysses=True))

    def call(rank_model):
        def patch(block_args, extra):
            if nest:
                rank_model._forward(*args, transformer_options={}, **kwargs)
            return extra["original_block"](block_args)

        options = {"patches_replace": {"dit": {("double_block", 0): patch}}}
        return rank_model._forward(*args, transformer_options=options, **kwargs)

    outputs = run_ranks(harness, models, call)
    assert not torch.isnan(outputs[0]).any()
    assert same(outputs[0], expected)


# Bernini's in-context latents: a source clip of the video's own length, which
# grows with it, and a reference image, which does not.
def _with_context(kwargs, frames):
    kwargs["context_latents"] = [torch.randn(1, CHANNELS, frames, 6, 6),
                                 torch.randn(1, CHANNELS, 1, 4, 4)]
    return frames * 9 + frames * 9 + 4


@pytest.mark.parametrize("world", [3, 4])
def test_wan_with_context_latents_attends_stock_keys_on_pure_ulysses(monkeypatch, comfy_wan,
                                                                     world):
    model = _model(comfy_wan, "wan")
    args, kwargs = _inputs("wan", (3, 6, 6), 1)
    rows = _with_context(kwargs, 3)
    assert rows % world
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs, rows)
    nan_pads(monkeypatch)
    outputs, harness = _sharded(monkeypatch, "wan", model, args, kwargs, kernel, world=world)
    assert len(stock_calls) == COMMON["num_layers"]
    for rank in range(world):
        for (_e, q, k, v), (sq, sk, sv) in zip(harness.seen[rank], stock_calls, strict=True):
            assert same(k, sk) and same(v, sv) and same(q, sq)
    assert not torch.isnan(outputs[0]).any() and same(outputs[0], expected)


def test_wan_ring_refusal_with_context_latents_names_lengths_that_divide(monkeypatch,
                                                                         comfy_wan):
    """9 video frames (3 latent) make 58 rows; at four ranks 5 and 13 frames
    divide only because the source clip grows with the video (counting it fixed
    would name 1 frame)."""
    model = _model(comfy_wan, "wan")
    args, kwargs = _inputs("wan", (3, 6, 6), 1)
    rows = _with_context(kwargs, 3)
    _expected, _calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs, rows)
    with pytest.raises(base.UnsupportedModelError) as caught:
        _sharded(monkeypatch, "wan", model, args, kwargs, kernel, pure_ulysses=False,
                 ring_world=2, world=4)
    named = [int(n) for n in re.findall(r"(\d+) frames", str(caught.value))]
    assert named == [5, 13]
    for count in named:
        frames = (count - 1) // 4 + 1
        assert (frames * 18 + 4) % 4 == 0


def test_a_waived_ring_render_attends_the_pad_and_differs_from_stock(monkeypatch, comfy_wan):
    model = _model(comfy_wan, "wan")
    args, kwargs = _inputs("wan", (3, 6, 10), 1)
    tokens = _tokens("wan", (3, 6, 10))
    expected, _calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs, tokens)
    grant_ring_pad_waiver(monkeypatch)
    outputs, harness = _sharded(monkeypatch, "wan", model, args, kwargs, kernel,
                                pure_ulysses=False, ring_world=2)
    assert harness.seen[0][0][2].shape[1] == tokens + 1
    assert torch.isfinite(outputs[0]).all()
    assert not same(outputs[0], expected)


def test_wandancer_refuses_image_tokens_its_audio_groups_do_not_divide(monkeypatch, comfy_wan):
    """Stock's audio injector reshapes the image tokens into one equal group per
    audio frame and raises on a remainder; the sharded forward refuses the same
    pair, class P, on every rank before it shards, where it used to fold the
    remainder into the last group and render."""
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    model = _model(comfy_wan, "wan_dancer")
    # 1 x 6 x 6 latent: 9 image tokens against 8 audio frame groups.
    args, kwargs = _inputs("wan_dancer", (1, 6, 6), 1)
    with pytest.raises(Exception, match=r"(?i)einops|shape|divis"):
        with torch.no_grad():
            type(model)._forward(model, *args, transformer_options={}, **kwargs)
    harness = install(monkeypatch, comfy_wan[1].optimized_attention, world=WORLD)
    models = replicas(model)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for rank_model in models:
        WanDancerAdapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=True))
    with pytest.raises(base.UnsupportedModelError) as caught:
        run_ranks(harness, models,
                  lambda m: m._forward(*args, transformer_options={}, **kwargs))
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "9 image tokens" in str(caught.value) and "8 frame groups" in str(caught.value)
    assert harness.seen[0] == harness.seen[1] == []


def test_an_unpadded_ring_render_needs_no_waiver(monkeypatch, comfy_wan):
    model = _model(comfy_wan, "wan")
    args, kwargs = _inputs("wan", (2, 6, 6), 1)
    expected, _calls, kernel = _stock(monkeypatch, comfy_wan, model, args, kwargs,
                                      _tokens("wan", (2, 6, 6)))
    outputs, _harness = _sharded(monkeypatch, "wan", model, args, kwargs, kernel,
                                 pure_ulysses=False, ring_world=2)
    assert same(outputs[0], expected)
