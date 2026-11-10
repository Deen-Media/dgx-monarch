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
  excluded, and both ranks return stock's output;
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


def _refs(count, lat, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return [{"kind": "image", "latent_h": lat[0], "latent_w": lat[1],
             "latent": torch.randn(1, 24, 1, lat[0], lat[1], generator=generator)}
            for _ in range(count)]


def _payload(comfy, text_len, lat, refs, tags=None):
    """What stock MiniMaxH3Image.extra_conds publishes for this conditioning."""
    payload = {"seed": 7, "audio_scale": 1.0}
    if tags is not None:
        payload["text_token_tags"] = tags
    if refs:
        payload["refs"] = refs
    payload.update(comfy.model_base.MiniMaxH3._cond_latents(None, refs))
    payload["layout"] = comfy.h3.PackedLayout(
        text_len, 1, (lat[0] + 1) // 2 * 2, (lat[1] + 1) // 2 * 2, 0, refs=payload.get("refs"))
    return payload


ON_GRID = torch.tensor(GRID)
# KSampler `simple` at six steps lands near the grid, not on it (D6).
OFF_GRID = torch.tensor([1.0, 0.9671, 0.8879, 0.7493, 0.5331, 0.2316, 0.0])


def _inputs(lat, text=9, seed=3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 24, 1, *lat, generator=generator)
    context = torch.randn(1, text, TEXT_DIM, generator=generator)
    return x, context


def _run(comfy, network, x, context, sigma, table, refs, lat, tags=None, **kwargs):
    timestep = torch.tensor([sigma * 1000.0])
    options = {"sample_sigmas": table, "sigmas": timestep / 1000.0}
    payload = _payload(comfy, context.shape[1], lat, refs, tags)
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
@pytest.mark.parametrize("sigma", [1.0, 0.75, 0.81, 0.1])
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


def _stock_calls(comfy, monkeypatch, model, x, context, sigma, refs, lat):
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
        out, _ = _run(comfy, model, x, context, sigma, ON_GRID, refs, lat)
    finally:
        monkeypatch.setattr(comfy.h3, "optimized_attention", kernel)
    return out, calls, kernel


def _sharded(monkeypatch, comfy, model, kernel, x, context, sigma, refs, lat, *, ring_world=1):
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
            outputs[rank], _ = _run(comfy, models[rank], x, context, sigma, ON_GRID, refs, lat)
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


# (latent, refs, text rows) and the packed total: 9 text + 9 target = 18 (even);
# 7 text + 6 reference + 6 target = 19 (one pad row at world 2); 8 text + 3 x 4
# reference + 4 target = 24 (three references, even).
WORLD_TWO_CASES = {
    "even": ((6, 6), 0, 9),
    "odd-one-pad": ((4, 6), 1, 7),
    "three-references": ((4, 4), 3, 8),
}


@pytest.mark.parametrize("case", sorted(WORLD_TWO_CASES))
@pytest.mark.parametrize("sigma", [1.0, 0.5])
def test_two_ranks_hand_every_block_stocks_operands_and_return_stocks_bits(comfy, monkeypatch, case, sigma):
    lat, ref_count, text = WORLD_TWO_CASES[case]
    model = _tiny(comfy)
    x, context = _inputs(lat, text=text)
    refs = _refs(ref_count, lat)
    stock, calls, kernel = _stock_calls(comfy, monkeypatch, model, x, context, sigma, refs, lat)
    outputs, harness = _sharded(monkeypatch, comfy, model, kernel, x, context, sigma, refs, lat)
    stock_blocks = calls[-LAYERS:]
    for rank, seen in harness.seen.items():
        assert len(seen) == LAYERS, (rank, len(seen))
        for (_entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_blocks, strict=True):
            # The same keys in the same order, no pad row, and the same queries and values.
            assert _same(k, stock_k) and _same(q, stock_q) and _same(v, stock_v)
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0].shape == stock.shape and _same(outputs[0], stock)


def test_a_padded_ring_refuses_class_p_before_any_block(comfy, monkeypatch):
    lat, ref_count, text = WORLD_TWO_CASES["odd-one-pad"]
    model = _tiny(comfy)
    x, context = _inputs(lat, text=text)
    refs = _refs(ref_count, lat)
    _, _, kernel = _stock_calls(comfy, monkeypatch, model, x, context, 0.5, refs, lat)
    with pytest.raises(UnsupportedModelError) as caught:
        _sharded(monkeypatch, comfy, model, kernel, x, context, 0.5, refs, lat, ring_world=2)
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "Ulysses-only" in str(caught.value)
