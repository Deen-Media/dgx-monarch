"""MiniMax-H3-Image's sharded forward equals stock (real ComfyUI 0.45.0 or newer, CPU).

ComfyUI builds MiniMax-H3-Image as ``MiniMaxH3Image`` around
``MiniMaxH3ImageModel``: the H3 DiT with no audio stream, one latent frame, and
a stacked PDD head sampled on the checkpoint's own ``pdd_sigmas`` grid. These
tests build a tiny real model in the released form (a six-block head, a
seven-point grid, the adaLN curve table) and ask the adapter for stock's bits:

- at world 1, with identity shard and gather, on text to image, an odd latent
  that pads to the patch, and edits with one and three reference pictures, on
  and off the head grid;
- at world 2 through the threaded two-rank harness the MOSS-Image and FLUX.3
  canaries use, where every block hands the kernel stock's operands, pad rows
  excluded, and both ranks return stock's output: by bits on aarch64, within
  2e-5 on the canary's x86 runner, and on GB10 a hardware leg;
- no attention pin running on a rank, and the block loop's prefetch calls;
- stock's ``block_index`` per block, the image model's ignored denoise mask,
  the grid check at inject, the registry order, the FSDP outside-forward row
  and the Worker VAE decode decline.

The uly2 compare on hardware is the maintainer's (docs/VALIDATION.md).
"""
from __future__ import annotations

import copy
import os
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters import minimax_h3_image as adapter_module
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.minimax_h3_image import MiniMaxH3ImageAdapter
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from module_location_helpers import from_checkout
from test_flux3_comfy_exactness import _plain, _same, _StockKernelHarness
from test_flux_pad_exclusion import _LOCAL, _install_stubs

COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
# The released grid and head depth (every Comfy-Org DiT repack's metadata).
GRID = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724,
        0.23200887441635132, 0.0]
PDD_STEPS = 6
CURVE_GRID, CURVE_DIM = 1025, 8
TEXT_DIM, LAYERS = 40, 2


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy", "nodes", "node_helpers") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real ComfyUI, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-minimax-h3-image", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            h3 = pytest.importorskip("comfy.ldm.minimax.model")
            if not hasattr(h3, "MiniMaxH3ImageModel"):
                pytest.skip("this ComfyUI predates MiniMax-H3-Image (ComfyUI 0.45.0 added it)")
            import comfy.ldm.minimax.vae as vae
            import comfy.model_base
            import comfy.ops
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(h3=h3, model_base=comfy.model_base, ops=comfy.ops, vae=vae)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _tiny(comfy, *, grid=GRID, pdd_steps=PDD_STEPS, layers=LAYERS):
    """A MiniMax-H3-Image DiT in the released form, scaled down, with seeded weights."""
    torch.manual_seed(0)
    model = comfy.h3.MiniMaxH3ImageModel(
        pdd_sigmas=grid, pdd_steps=pdd_steps, hidden_size=48, num_layers=layers,
        token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=24,
        ffn_hidden_size=64, text_dim=TEXT_DIM, time_embed_dim=CURVE_DIM, rope_inv_freq_len=4,
        adaln_curve_grid=CURVE_GRID, dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if name.endswith("inv_freq"):
                tensor.copy_(1.0 / (10000 ** (torch.arange(tensor.numel()) / tensor.numel())))
            elif "norm" in name and name.endswith("weight"):
                tensor.copy_(1.0 + 0.1 * torch.randn_like(tensor))
            else:
                tensor.copy_(0.2 * torch.randn_like(tensor))
    return model.requires_grad_(False)


def _bound(model, *, world=1, attention=None, pure_ulysses=False):
    clone = copy.deepcopy(model)
    MiniMaxH3ImageAdapter().inject_usp(clone, InjectionContext(
        topology_sp=world, usp_attention=attention or object(), pure_ulysses=pure_ulysses))
    return clone


@pytest.fixture
def world_one(monkeypatch):
    monkeypatch.setattr(adapter_module, "sp_world", lambda: 1)
    monkeypatch.setattr(adapter_module, "shard_seq", lambda tensor, dim=1: (tensor, tensor.shape[dim]))
    monkeypatch.setattr(adapter_module, "sp_gather", lambda tensor, _orig, dim=1: tensor)
    monkeypatch.setattr(adapter_module, "padded_row_indices", lambda _segments: [])
    monkeypatch.setattr(adapter_module, "usp_options", lambda options, _attn, drop_rows=None: dict(options))


def _refs_sized(sizes, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return [{"kind": "image", "latent_h": h, "latent_w": w,
             "latent": torch.randn(1, 24, 1, h, w, generator=generator)}
            for h, w in sizes]


def _refs(count, lat, seed=11):
    return _refs_sized([lat] * count, seed)


def _payload(comfy, text_len, lat, refs, tags=None, layout=True, aug=None):
    """What stock MiniMaxH3Image.extra_conds publishes for this conditioning.

    ``layout=False`` leaves the layout out, so the forward builds its own from
    the references; ``aug`` sets the payload's visual noise augmentation.
    """
    payload = {"seed": 7, "audio_scale": 1.0}
    if tags is not None:
        payload["text_token_tags"] = tags
    if refs:
        payload["refs"] = refs
    if aug is not None:
        payload["visual_cond_noise_aug"] = aug
    payload.update(comfy.model_base.MiniMaxH3._cond_latents(None, refs))
    if layout:
        payload["layout"] = comfy.h3.PackedLayout(
            text_len, 1, (lat[0] + 1) // 2 * 2, (lat[1] + 1) // 2 * 2, 0,
            refs=payload.get("refs"))
    return payload


ON_GRID = torch.tensor(GRID)
# KSampler `simple` at six steps lands near the grid, not on it (D6).
OFF_GRID = torch.tensor([1.0, 0.9671, 0.8879, 0.7493, 0.5331, 0.2316, 0.0])


def _inputs(lat, text=9, seed=3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 24, 1, *lat, generator=generator)
    context = torch.randn(1, text, TEXT_DIM, generator=generator)
    return x, context


def _run(comfy, network, x, context, sigma, table, refs, lat, tags=None, *, layout=True,
         aug=None, **kwargs):
    timestep = torch.tensor([sigma * 1000.0])
    options = {"sample_sigmas": table, "sigmas": timestep / 1000.0}
    payload = _payload(comfy, context.shape[1], lat, refs, tags, layout=layout, aug=aug)
    with torch.inference_mode():
        out = network.forward(x.clone(), timestep, context.clone(), options,
                              minimax_payload=payload, **kwargs)
    return out, options


CASES = [
    pytest.param((6, 6), 0, None, id="text-to-image"),
    pytest.param((5, 7), 0, None, id="odd-latent-pads"),
    pytest.param((6, 4), 1, None, id="edit-one-reference"),
    pytest.param((4, 6), 3, None, id="edit-three-references"),
    pytest.param((4, 4), 1, "tags", id="presentation-tags"),
]


def _tags(text):
    # A presentation prompt marks vision-pad positions with the video tag (0).
    tags = torch.ones(1, text, dtype=torch.long)
    tags[0, 2:5] = 0
    return tags


@pytest.mark.parametrize("table", [ON_GRID, OFF_GRID], ids=["on-grid", "off-grid"])
@pytest.mark.parametrize("lat, ref_count, tags", CASES)
# Sigma 0 holds the forward to stock's unclamped sigma (H3's video forward clamps at 1e-6).
@pytest.mark.parametrize("sigma", [1.0, 0.75, 0.81, 0.1, 0.0])
def test_the_bound_forward_equals_stock_at_world_one(comfy, world_one, lat, ref_count, tags, sigma, table):
    model = _tiny(comfy)
    bound = _bound(model)
    assert "_forward" in vars(bound) and "_forward" not in vars(model)
    x, context = _inputs(lat)
    refs = _refs(ref_count, lat)
    text_tags = _tags(context.shape[1]) if tags else None
    stock, _ = _run(comfy, model, x, context, sigma, table, refs, lat, text_tags)
    ours, _ = _run(comfy, bound, x, context, sigma, table, refs, lat, text_tags)
    assert stock.shape == x.shape
    assert torch.equal(stock, ours), "the image velocity differs from stock"
    assert torch.isfinite(stock).all() and stock.abs().sum() > 0


def test_the_refined_context_path_equals_stock(comfy, world_one):
    """A context already at the hidden width (extra_conds refined it) skips the refiner, as stock."""
    model = _tiny(comfy)
    bound = _bound(model)
    x, raw = _inputs((4, 6))
    context = model.preprocess_text_embeds(raw)
    stock, _ = _run(comfy, model, x, context, 0.75, ON_GRID, _refs(1, (4, 6)), (4, 6))
    ours, _ = _run(comfy, bound, x, context, 0.75, ON_GRID, _refs(1, (4, 6)), (4, 6))
    assert torch.equal(stock, ours)


def _record_block_index(comfy, monkeypatch):
    seen: list = []
    real = comfy.h3.DiTBlock.forward

    def forward(block, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
        transformer_options = {} if transformer_options is None else transformer_options
        seen.append(transformer_options.get("block_index"))
        return real(block, x, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options, attention=attention)

    monkeypatch.setattr(comfy.h3.DiTBlock, "forward", forward)
    return seen


def test_every_block_sees_stocks_block_index(comfy, world_one, monkeypatch):
    seen = _record_block_index(comfy, monkeypatch)
    model = _tiny(comfy, layers=3)
    bound = _bound(model)
    x, context = _inputs((4, 4))
    _, stock_options = _run(comfy, model, x, context, 0.75, ON_GRID, [], (4, 4))
    stock = list(seen)
    seen.clear()
    _, our_options = _run(comfy, bound, x, context, 0.75, ON_GRID, [], (4, 4))
    assert stock == [0, 1, 2] and seen == stock
    assert our_options["block_index"] == stock_options["block_index"] == 2


def test_stock_sends_the_image_model_no_mask_and_ignores_one(comfy):
    """extra_conds builds no mask cond for one image stream, and the forward ignores a mask."""
    values = comfy.model_base.MiniMaxH3._denoise_mask_values(
        None, torch.zeros(1, 1, 1, 8, 8), None)
    assert values == {}
    model = _tiny(comfy)
    x, context = _inputs((4, 4))
    plain, _ = _run(comfy, model, x, context, 0.75, ON_GRID, [], (4, 4))
    masked, _ = _run(comfy, model, x, context, 0.75, ON_GRID, [], (4, 4),
                     denoise_mask=torch.zeros(1, 1, 1, 2, 2))
    assert torch.equal(plain, masked)


def test_a_grid_that_misses_the_head_refuses_class_p_at_inject(comfy):
    model = _tiny(comfy, grid=GRID[:5])
    with pytest.raises(UnsupportedModelError) as caught:
        _bound(model)
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "6 blocks" in str(caught.value) and "5 points" in str(caught.value)
    assert "_forward" not in vars(model)
    # Stock fails inside the head on the same file: a short grid indexes past
    # its end, a long one cannot reshape the head's weight.
    x, context = _inputs((4, 4))
    with pytest.raises((IndexError, RuntimeError)):
        _run(comfy, model, x, context, 0.75, ON_GRID, [], (4, 4))


def test_a_one_block_head_never_reads_the_grid_and_goes_on(comfy, world_one):
    model = _tiny(comfy, grid=GRID[:3], pdd_steps=1)
    bound = _bound(model)
    x, context = _inputs((4, 4))
    stock, _ = _run(comfy, model, x, context, 0.75, ON_GRID, [], (4, 4))
    ours, _ = _run(comfy, bound, x, context, 0.75, ON_GRID, [], (4, 4))
    assert torch.equal(stock, ours)


def test_the_registry_hands_the_image_model_to_this_adapter(comfy):
    from dgx_monarch.adapters import get_adapter
    from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter

    image = object.__new__(comfy.model_base.MiniMaxH3Image)
    assert isinstance(get_adapter(image), MiniMaxH3ImageAdapter)
    assert isinstance(get_adapter(object.__new__(comfy.model_base.MiniMaxH3)), MiniMaxH3Adapter)

    class Unvetted(comfy.model_base.MiniMaxH3):
        pass

    with pytest.raises(UnsupportedModelError, match="has not vetted"):
        get_adapter(object.__new__(Unvetted))

    class UnvettedImage(comfy.model_base.MiniMaxH3Image):
        pass

    with pytest.raises(UnsupportedModelError, match="minimax_h3_image variant UnvettedImage"):
        get_adapter(object.__new__(UnvettedImage))


def test_fsdp_replicates_h3s_outside_forward_modules_for_the_image_model(comfy):
    from dgx_monarch.adapters.fsdp_islands import outside_forward_module_attrs

    attrs = outside_forward_module_attrs(object.__new__(comfy.model_base.MiniMaxH3Image))
    assert {"condition_proj", "token_refiner"} <= set(attrs)


def test_the_worker_vae_decode_declines_the_image_vae(comfy):
    from dgx_monarch.vae_decode_families import classify

    vae = SimpleNamespace(first_stage_model=object.__new__(comfy.vae.MiniMaxH3ImageVAE),
                          not_video=True)
    family, why = classify(vae)
    assert family is None
    assert why == "MiniMaxH3ImageVAE decodes single frames, so there is nothing to split"


# --- world 2: the threaded two-rank harness ---------------------------------------

WORLD = 2


def _stock_calls(comfy, monkeypatch, model, x, context, **run):
    kernel = comfy.h3.optimized_attention
    calls: list = []

    # Record the operands, then run stock's own call untouched.
    def recording(q, k, v, heads, *args, **kwargs):
        plain = [_plain(t) for t in (q, k, v)]
        flat = plain
        if kwargs.get("skip_reshape"):
            flat = [t.transpose(1, 2).reshape(t.shape[0], t.shape[2], -1) for t in plain]
        calls.append(tuple(flat))
        # The container hands its tensor out once, so the kernel takes the tensors.
        return kernel(*plain, heads, *args, **kwargs)

    monkeypatch.setattr(comfy.h3, "optimized_attention", recording)
    try:
        out, _ = _run(comfy, model, x, context, **run)
    finally:
        monkeypatch.setattr(comfy.h3, "optimized_attention", kernel)
    return out, calls, kernel


def _sharded(monkeypatch, comfy, model, kernel, x, context, *, ring_world=1, **run):
    harness = _StockKernelHarness(WORLD, kernel)
    _install_stubs(monkeypatch, harness)
    # The stubs set base's rank accessors; the adapter imported sp_world by name.
    monkeypatch.setattr(adapter_module, "sp_world", lambda: WORLD)
    monkeypatch.setattr(sys.modules["xfuser.core.distributed"],
                        "get_ring_parallel_world_size", lambda: ring_world)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    models = [_bound(model, world=WORLD, attention=dispatch, pure_ulysses=ring_world == 1)
              for _ in range(WORLD)]
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank], _ = _run(comfy, models[rank], x, context, **run)
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    raised = [error for error in errors if error is not None]
    for error in raised:
        if not isinstance(error, threading.BrokenBarrierError):
            raise error
    for error in raised:
        raise error
    return outputs, harness


TABLES = {"on": ON_GRID, "off": OFF_GRID}

# name: (latent, reference sizes, text rows, video-tag span in the text or None,
# table, payload carries its layout). Packed totals: even 9 + 9 = 18; odd-one-pad
# 7 + 6 + 6 = 19 (one pad row at world 2); three references 8 + 12 + 4 = 24; a
# reference larger than the target 7 + 16 + 4 = 27; text across the shard line
# 33 + 4 + 4 = 41, with a tag run over the line; the off-grid table; and a payload
# without its layout, which the forward rebuilds with the references.
WORLD_TWO_CASES = {
    "even": ((6, 6), [], 9, None, "on", True),
    "odd-one-pad": ((4, 6), [(4, 6)], 7, None, "on", True),
    "three-references": ((4, 4), [(4, 4)] * 3, 8, None, "on", True),
    "reference-larger-than-target": ((4, 4), [(8, 8)], 7, None, "on", True),
    "text-across-the-shard-line-with-tags": ((4, 4), [(4, 4)], 33, (19, 24), "on", True),
    "off-grid-table": ((4, 6), [(4, 6)], 7, None, "off", True),
    "no-layout-in-the-payload": ((4, 6), [(4, 6)], 7, None, "on", False),
}


def _case(name, sigma):
    lat, sizes, text, span, table, layout = WORLD_TWO_CASES[name]
    tags = None
    if span is not None:
        tags = torch.ones(1, text, dtype=torch.long)
        tags[0, span[0]:span[1]] = 0
    run = {"sigma": sigma, "table": TABLES[table], "refs": _refs_sized(sizes), "lat": lat,
           "tags": tags, "layout": layout}
    return lat, text, run


def _assert_stock_operands_and_output(calls, harness, outputs, stock):
    stock_blocks = calls[-LAYERS:]
    for rank, seen in harness.seen.items():
        assert len(seen) == LAYERS, (rank, len(seen))
        for (_entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_blocks, strict=True):
            # The same keys in the same order, no pad row, and the same queries and values.
            assert _same(k, stock_k) and _same(q, stock_q) and _same(v, stock_v)
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0].shape == stock.shape and _same(outputs[0], stock)


@pytest.mark.parametrize("case", sorted(WORLD_TWO_CASES))
@pytest.mark.parametrize("sigma", [1.0, 0.5])
def test_two_ranks_hand_every_block_stocks_operands_and_return_stocks_output(
        comfy, monkeypatch, case, sigma):
    """Two ranks through the threaded harness against stock on one.

    `_same` compares by bits on aarch64 and within 2e-5 elsewhere
    (tests/test_flux3_comfy_exactness.py), so the comfy canary's x86 runner checks
    this within that tolerance, not by bits. Bit identity is shown on aarch64 on
    CPU (this file on the Sparks) and is a hardware leg on GB10 (docs/VALIDATION.md).
    """
    lat, text, run = _case(case, sigma)
    model = _tiny(comfy)
    x, context = _inputs(lat, text=text)
    stock, calls, kernel = _stock_calls(comfy, monkeypatch, model, x, context, **run)
    outputs, harness = _sharded(monkeypatch, comfy, model, kernel, x, context, **run)
    _assert_stock_operands_and_output(calls, harness, outputs, stock)


def test_two_ranks_on_a_refined_context(comfy, monkeypatch):
    """A context extra_conds already refined to the hidden width skips the refiner, as stock."""
    lat, text, run = _case("odd-one-pad", 0.5)
    model = _tiny(comfy)
    x, raw = _inputs(lat, text=text)
    with torch.inference_mode():
        context = model.preprocess_text_embeds(raw)
    stock, calls, kernel = _stock_calls(comfy, monkeypatch, model, x, context, **run)
    outputs, harness = _sharded(monkeypatch, comfy, model, kernel, x, context, **run)
    _assert_stock_operands_and_output(calls, harness, outputs, stock)


class _PinRan(Exception):
    pass


def _pin_every_attention(comfy, model):
    from comfy.ldm.modules.attention import ComfyAttention

    def pinned_kernel(*_args, **_kwargs):
        raise _PinRan("a pinned kernel ran")

    count = 0
    for module in model.modules():
        if isinstance(module, ComfyAttention):
            module.function = pinned_kernel
            count += 1
    return count


def test_no_pin_runs_on_a_rank(comfy, monkeypatch):
    """Every ComfyAttention pinned with a recorder: stock calls it, the two ranks and
    extra_conds' refiner path on the bound model never do, and the output is stock's."""
    lat, text, run = _case("odd-one-pad", 0.5)
    model = _tiny(comfy)
    x, context = _inputs(lat, text=text)
    stock, calls, kernel = _stock_calls(comfy, monkeypatch, model, x, context, **run)
    pinned = copy.deepcopy(model)
    assert _pin_every_attention(comfy, pinned) == LAYERS + 1  # the blocks and the refiner
    with pytest.raises(_PinRan):
        _run(comfy, pinned, x, context, **run)  # the control: stock runs the pin
    outputs, harness = _sharded(monkeypatch, comfy, pinned, kernel, x, context, **run)
    _assert_stock_operands_and_output(calls, harness, outputs, stock)
    bound = _bound(pinned)
    with torch.inference_mode():
        refined = bound.preprocess_text_embeds(context)
    assert torch.isfinite(refined).all()


def test_the_block_loop_pops_as_stock_does(comfy, world_one, monkeypatch):
    """One queue over the blocks, every pop with the block scope, and the closing pop."""
    import comfy.model_prefetch as prefetch

    events: list = []
    real_queue, real_pop = prefetch.make_prefetch_queue, prefetch.prefetch_queue_pop

    def make_prefetch_queue(queue, device, options):
        events.append(("queue", [type(module).__name__ for module in queue]))
        return real_queue(queue, device, options)

    def prefetch_queue_pop(queue, device, module, *args, **kwargs):
        events.append(("pop", type(module).__name__ if module is not None else None,
                       args, tuple(sorted(kwargs.items()))))
        return real_pop(queue, device, module, *args, **kwargs)

    monkeypatch.setattr(prefetch, "make_prefetch_queue", make_prefetch_queue)
    monkeypatch.setattr(prefetch, "prefetch_queue_pop", prefetch_queue_pop)
    model = _tiny(comfy)
    bound = _bound(model)
    x, context = _inputs((4, 4))
    _run(comfy, model, x, context, 0.75, ON_GRID, [], (4, 4))
    stock = list(events)
    events.clear()
    _run(comfy, bound, x, context, 0.75, ON_GRID, [], (4, 4))
    assert events == stock
    assert stock[0] == ("queue", ["DiTBlock"] * LAYERS)
    assert [event for event in stock if event[0] == "pop"] == (
        [("pop", "DiTBlock", (), (("malloc_scope", "block"),))] * LAYERS
        + [("pop", None, (), (("malloc_scope", "block"),))])


def test_a_padded_ring_refuses_class_p_before_any_block(comfy, monkeypatch):
    lat, text, run = _case("odd-one-pad", 0.5)
    model = _tiny(comfy)
    x, context = _inputs(lat, text=text)
    _, _, kernel = _stock_calls(comfy, monkeypatch, model, x, context, **run)
    with pytest.raises(UnsupportedModelError) as caught:
        _sharded(monkeypatch, comfy, model, kernel, x, context, ring_world=2, **run)
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "Ulysses-only" in str(caught.value)


def test_the_detected_family_reaches_this_adapter_and_a_forced_h3_refuses(comfy):
    """Detection names the family from the model ComfyUI built; forcing H3's adapter on it
    refuses class P at the bind, naming the adapter that owns the model."""
    from dgx_monarch.adapters import adapter_for, detect_family

    image = object.__new__(comfy.model_base.MiniMaxH3Image)
    assert detect_family(image) == adapter_module.FAMILY
    assert isinstance(adapter_for(image, adapter_module.FAMILY), MiniMaxH3ImageAdapter)
    with pytest.raises(UnsupportedModelError) as caught:
        adapter_for(image, "minimax_h3")
    assert "which the minimax_h3_image adapter owns" in str(caught.value)
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS


def test_the_adaln_projections_build_fp32_so_the_fsdp_wrap_replicates_them(comfy):
    """The header counts the BF16 file's F16 adaLN linears as islands; ComfyUI builds
    them in fp32 on the curve basis, so the FSDP wrap's fp32 rule replicates them."""
    from dgx_monarch.adapters.fsdp_islands import island_parameters

    model = comfy.h3.MiniMaxH3ImageModel(  # a bf16 core, as the BF16 file builds
        pdd_sigmas=GRID, pdd_steps=PDD_STEPS, hidden_size=48, num_layers=LAYERS,
        token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=24,
        ffn_hidden_size=64, text_dim=TEXT_DIM, time_embed_dim=CURVE_DIM, rope_inv_freq_len=4,
        adaln_curve_grid=CURVE_GRID, dtype=torch.bfloat16, operations=comfy.ops.disable_weight_init)
    adaln = [p for name, p in model.named_parameters() if ".adaln_proj." in name]
    assert len(adaln) >= LAYERS + 1 and all(p.dtype == torch.float32 for p in adaln)
    assert model.blocks[0].attn.qkv_proj.weight.dtype == torch.bfloat16  # the core
    islands = island_parameters(model)
    assert all(any(p is q for q in islands) for p in adaln)
