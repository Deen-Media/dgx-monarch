"""MOSS-Image under pure Ulysses produces exactly the stock ComfyUI output (real model, CPU).

This builds comfy's own ``comfy.ldm.moss_image.model.MOSSImage`` at toy width
in float32 (width 256 over two heads, so the head dimension stays 128 for the
four 32-wide rotary axes; three blocks, so a skipped block sits in the
middle), runs its stock forward, then runs the adapter's bound forward on rank
threads. Only the all-to-all, the sequence gather and the attention kernel are
stubbed (the harness of tests/test_flux_pad_exclusion.py), and the stub kernel
is comfy's own ``optimized_attention`` on the full gathered rows, so the
attention math is stock's. Each kernel call records its q, k and v, which must
equal stock's call for the same block, with no pad row and in stock's key
order, and the output must equal stock's. Recording hooks prove the mechanism
as well as the result: the fused-norm launches, the rotary rows, the override
count, the prefetch order, the block replacements and the guards.

"Equal" follows tests/test_flux3_comfy_exactness.py: bit equality on this
rig's aarch64 CPU, float32 rounding (allclose 2e-5) on x86. Rank against rank
is always exact. These tests prove the plumbing; the GPU kernel branches are
tests/test_moss_image_parity.py's and the hardware ladder's.

Three halves, named by prefix so a run against another tree can pick one with
``-k``: ``test_forward_*`` (the sharded forward), ``test_lora_*`` (ComfyUI's
key map, the store's bake and un-bake, the FSDP bake, on a tiny real
``model_base.MOSSImage`` loaded from a vendor-layout file) and
``test_guidance_*`` (the conds and ComfyUI's own Skip Layer Guidance). The
fixture forces ``--cpu`` because comfy.cli_args parses argv on import, and
skips where ComfyUI or its MOSS-Image model is not importable.
"""
from __future__ import annotations

import copy
import inspect
import os
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import accuracy_waiver
from dgx_monarch.adapters import base, cfg_dispatch, moss_image_seams
from dgx_monarch.adapters.attention_patches import (
    ATTENTION_PATCH_CAPABILITIES_ATTR,
    SEQUENCE_SHARD_LOCAL_CAPABILITY,
)
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.moss_image import MOSSImageAdapter
from dgx_monarch.adapters.moss_image_seams import is_identity_block_patch, vet_block_replacements
from dgx_monarch.adapters.stock_attention import default_attention_override
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from module_location_helpers import from_checkout
from test_flux3_comfy_exactness import _plain, _same, _StockKernelHarness
from test_flux_pad_exclusion import _LOCAL, _install_stubs, _rank

WORLD = 2
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
DIM, HEADS, FFN, LAYERS, TEXT_DIM, CHANNELS = 256, 2, 512, 3, 64, 128
VENDOR_TEXT_DIM = 4096   # detection keeps the default text width, so a file must carry it


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_moss():
    """Import real ComfyUI's MOSS-Image model, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-moss-image-comfy-exactness", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            model = pytest.importorskip(
                "comfy.ldm.moss_image.model", reason="this ComfyUI predates MOSS-Image")
            import comfy.model_prefetch
            import comfy.quant_ops
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(ops=ops, model=model, prefetch=comfy.model_prefetch,
                              kitchen=comfy.quant_ops.ck)
    finally:
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _tiny(comfy, *, fused=True, heads=HEADS, text_dim=TEXT_DIM, seed=0):
    generator = torch.Generator().manual_seed(seed)
    model = comfy.model.MOSSImage(
        dim=128 * heads, ffn_dim=FFN, num_heads=heads, num_layers=LAYERS, text_dim=text_dim,
        latent_channels=CHANNELS, fused_qkv=fused, operations=comfy.ops.disable_weight_init,
        dtype=torch.float32, device="cpu")
    # comfy builds state-dict parameters with torch.empty; a real load fills them.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model


def _inputs(grid, text, batch=1, seed=1, text_dim=TEXT_DIM, channels=CHANNELS):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, channels, *grid, generator=generator)
    timestep = torch.tensor([0.6, 0.35][:batch])
    context = torch.randn(batch, text, text_dim, generator=generator)
    return x, timestep, context


def _stock(monkeypatch, comfy, model, inputs, options):
    """Stock's forward, recording every attention call's operands."""
    kernel = comfy.model.optimized_attention
    calls: list[tuple] = []

    def record(q, k, v, heads, *args, **kwargs):
        q, k, v = (_plain(t) for t in (q, k, v))
        calls.append((q, k, v))
        return kernel(q, k, v, heads, *args, **kwargs)

    monkeypatch.setattr(comfy.model, "optimized_attention", record)
    try:
        with torch.no_grad():
            out = model(*inputs, transformer_options=copy.copy(options))
    finally:
        monkeypatch.setattr(comfy.model, "optimized_attention", kernel)
    return out, calls, kernel


def _sharded(monkeypatch, model, inputs, options, kernel, *, world=WORLD, ring_world=1,
             harness=None, wrap_dispatch=None):
    harness = harness if harness is not None else _StockKernelHarness(world, kernel)
    _install_stubs(monkeypatch, harness)
    monkeypatch.setattr(sys.modules["xfuser.core.distributed"],
                        "get_ring_parallel_world_size", lambda: ring_world)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    if wrap_dispatch is not None:
        dispatch = wrap_dispatch(dispatch)
    models = [copy.deepcopy(model) for _ in range(world)]
    for rank_model in models:
        MOSSImageAdapter().inject_usp(rank_model, InjectionContext(
            topology_sp=world, usp_attention=dispatch, pure_ulysses=ring_world == 1))
    outputs: list = [None] * world
    errors: list = [None] * world

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                outputs[rank] = models[rank](*inputs, transformer_options=copy.copy(options))
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for error in errors:
        if error is not None:
            raise error
    return outputs, harness


def _assert_stock_operands_and_output(stock_calls, harness, expected, outputs, blocks=LAYERS):
    stock_blocks = stock_calls[-blocks:]
    for rank, seen in harness.seen.items():
        assert len(seen) == blocks, (rank, len(seen))
        for (_entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_blocks, strict=True):
            # The same keys in the same order, no pad row, and the same queries and values.
            assert _same(k, stock_k)
            assert _same(q, stock_q)
            assert _same(v, stock_v)
    for rank in range(1, len(outputs)):
        assert torch.equal(outputs[0], outputs[rank])
    assert outputs[0].shape == expected.shape
    assert _same(outputs[0], expected)


def _tag(exc: BaseException):
    return parse_leading_refusal_tag(str(exc))


def _marked(function):
    setattr(function, ATTENTION_PATCH_CAPABILITIES_ATTR, (SEQUENCE_SHARD_LOCAL_CAPABILITY,))
    return function


# --- the forward half ---------------------------------------------------------------

# (grid, text rows, batch, fused): an even stream (N 22); the boundary inside the
# image block (N 19, rank 1 starts at row 10) and inside the text block (N 13,
# rank 1 starts at row 7, past the 4 image rows); batch 2 at equal T, where the
# shard is a strided chunk; split q, k, v.
EXACT_CASES = {
    "even": ((4, 4), 6, 1, True),
    "image-boundary": ((4, 4), 3, 1, True),
    "text-boundary": ((2, 2), 9, 1, True),
    "batch-2": ((3, 3), 5, 2, True),
    "split-qkv": ((3, 3), 6, 1, False),
}


@pytest.mark.parametrize("case", sorted(EXACT_CASES))
def test_forward_hands_every_kernel_stock_operands_and_returns_stock_bits(monkeypatch, comfy_moss, case):
    grid, text, batch, fused = EXACT_CASES[case]
    model = _tiny(comfy_moss, fused=fused)
    assert ("qkv" in model.layers[0].proj) is fused
    inputs = _inputs(grid, text, batch)
    contiguous: list[bool] = []
    for block in model.layers:
        # Each rank's deep copy keeps the hook: the block input must be a contiguous window.
        block.register_forward_pre_hook(
            lambda module, args, kwargs: contiguous.append(args[0].is_contiguous()), with_kwargs=True)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    contiguous.clear()
    outputs, harness = _sharded(monkeypatch, model, inputs, {}, kernel)
    _assert_stock_operands_and_output(stock_calls, harness, expected, outputs)
    assert contiguous and all(contiguous)
    rows = grid[0] * grid[1] + text
    entries = {entry for rank in harness.seen for entry, *_ in harness.seen[rank]}
    assert entries == {"exact" if rows % WORLD else "xfuser"}


def test_forward_an_odd_stream_drops_its_pad_and_a_kept_pad_moves_the_output(monkeypatch, comfy_moss):
    """3 x 3 image rows and 6 text rows, N 15: rank 1 holds one pad row."""
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    outputs, harness = _sharded(monkeypatch, model, inputs, {}, kernel)
    _assert_stock_operands_and_output(stock_calls, harness, expected, outputs)
    for rank in range(WORLD):
        assert all(entry == "exact" and k.shape[1] == 15 for entry, _q, k, _v in harness.seen[rank])
    # Negative control: no drop rows, and the pad row is attended.
    monkeypatch.setattr(moss_image_seams, "padded_row_indices", lambda segments: [])
    attended, harness = _sharded(monkeypatch, model, inputs, {}, kernel)
    assert all(k.shape[1] == 16 for _entry, _q, k, _v in harness.seen[0])
    assert not _same(attended[0], expected)


def test_forward_four_ranks_with_one_rank_of_pads_only(monkeypatch, comfy_moss):
    """A 1 x 1 grid and 4 text rows, N 5 at world 4: rank 3 holds two pad rows."""
    model = _tiny(comfy_moss, heads=4)
    inputs = _inputs((1, 1), 4)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    outputs, harness = _sharded(monkeypatch, model, inputs, {}, kernel, world=4)
    _assert_stock_operands_and_output(stock_calls, harness, expected, outputs)


def _rope_takes_options(comfy) -> bool:
    return "transformer_options" in inspect.signature(comfy.model.MOSSImage.rope).parameters


ROPE_OPTIONS = {"shift_t": 1.5, "shift_x": 0.5, "shift_y": -0.5, "scale_x": 2.0, "scale_y": 0.0}


def test_forward_rope_options_reach_stocks_table(monkeypatch, comfy_moss):
    """Master's rope reads rope_options; the sharded forward hands them over unchanged."""
    if not _rope_takes_options(comfy_moss):
        pytest.skip("this ComfyUI's MOSSImage.rope reads no options (the v0.42 form)")
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 4), 5)
    options = {"rope_options": dict(ROPE_OPTIONS)}
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, options)
    plain, _calls, _kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    assert not torch.equal(plain, expected)    # the options move the output
    outputs, harness = _sharded(monkeypatch, model, inputs, options, kernel)
    _assert_stock_operands_and_output(stock_calls, harness, expected, outputs)


def test_forward_the_v042_rope_form_is_called_without_options(monkeypatch, comfy_moss):
    """The class's rope swapped for the four-argument form v0.42.0 ships."""
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 5)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    stock_rope = comfy_moss.model.MOSSImage.rope
    # Read before the patch: inside the call the class already holds the four-argument form.
    stock_takes_options = _rope_takes_options(comfy_moss)
    seen: list[int] = []

    def rope(self, h, w, text_len, device):
        seen.append(4)
        if stock_takes_options:
            return stock_rope(self, h, w, text_len, device, {})
        return stock_rope(self, h, w, text_len, device)

    monkeypatch.setattr(comfy_moss.model.MOSSImage, "rope", rope)
    options = {"rope_options": dict(ROPE_OPTIONS)}   # the v0.42 form ignores them, as its stock does
    outputs, harness = _sharded(monkeypatch, model, inputs, options, kernel)
    assert seen == [4] * WORLD
    _assert_stock_operands_and_output(stock_calls, harness, expected, outputs)


def _skip(args, extra_args):
    return args


def test_forward_an_identity_skip_is_honoured_and_live(monkeypatch, comfy_moss):
    """Block 1 skipped by hand, as Skip Layer Guidance installs it."""
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6)
    options = {"patches_replace": {"dit": {("single_block", 1): _skip}}}
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, options)
    plain, _calls, _kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    assert not torch.equal(plain, expected)
    outputs, harness = _sharded(monkeypatch, model, inputs, options, kernel)
    _assert_stock_operands_and_output(stock_calls, harness, expected, outputs, blocks=LAYERS - 1)


def test_forward_a_marked_patch_is_honoured_and_the_same_patch_unmarked_refuses(monkeypatch, comfy_moss):
    """A row-local patch that doubles the stream, then runs the block."""

    def double(args, extra):
        return extra["original_block"]({**args, "img": args["img"] * 2})

    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6)
    options = {"patches_replace": {"dit": {("single_block", 2): _marked(double)}}}
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, options)
    outputs, harness = _sharded(monkeypatch, model, inputs, options, kernel)
    _assert_stock_operands_and_output(stock_calls, harness, expected, outputs)

    def unmarked(args, extra):
        return extra["original_block"]({**args, "img": args["img"] * 2})

    harness = _StockKernelHarness(WORLD, kernel)
    options = {"patches_replace": {"dit": {("single_block", 2): unmarked}}}
    with pytest.raises(UnsupportedModelError, match="block 2") as caught:
        _sharded(monkeypatch, model, inputs, options, kernel, harness=harness)
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    assert harness.seen == {rank: [] for rank in range(WORLD)}


def _adds_a_row(args, extra):
    img = args["img"]
    out = extra["original_block"]({**args, "img": torch.cat((img, img[:, -1:]), dim=1)})["img"]
    return {"img": out[:, :-1]}


def _drops_a_row(args, extra):
    img = args["img"]
    out = extra["original_block"]({**args, "img": img[:, :-1].contiguous()})["img"]
    return {"img": torch.cat((out, img[:, -1:]), dim=1)}


@pytest.mark.parametrize("patch", [_adds_a_row, _drops_a_row], ids=["adds-a-row", "drops-a-row"])
def test_forward_a_marked_patch_that_changes_the_window_rows_refuses_class_p(monkeypatch, comfy_moss, patch):
    """A marked patch that hands stock's block one row more or one row fewer than the
    rank's window refuses class P on every rank before that block attends."""
    model = _tiny(comfy_moss)
    inputs = _inputs((4, 4), 6)   # N 22, an 11-row window a rank, no pad
    harness = _StockKernelHarness(WORLD, comfy_moss.model.optimized_attention)
    options = {"patches_replace": {"dit": {("single_block", 1): _marked(patch)}}}
    with pytest.raises(UnsupportedModelError, match="replacement on block 1") as caught:
        _sharded(monkeypatch, model, inputs, options, harness.kernel, harness=harness)
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    assert "[1, 11, D] window with 11 rotary rows" in str(caught.value)
    # Block 1 never reached the kernel on any rank, and the refusing rank attended
    # block 0 first. The other rank may still be waiting in block 0's all-to-all
    # when the refusal aborts the fabric barrier, so its count is 0 or 1.
    counts = {rank: len(seen) for rank, seen in harness.seen.items()}
    assert max(counts.values()) == 1 and set(counts.values()) <= {0, 1}


@pytest.mark.parametrize("grid,text,batch", [((3, 3), 6, 1), ((4, 4), 3, 2)], ids=["odd-N", "batch-2"])
def test_forward_every_kitchen_launch_holds_stocks_rows(monkeypatch, comfy_moss, grid, text, batch):
    """Every fused-norm launch on every rank is stock's launch, call for call, and
    every fused q/k norm and RoPE call hands the table the rows q has."""
    kitchen = comfy_moss.kitchen
    adaln, rope = kitchen.rms_adaln, kitchen.rms_rope_split_half
    launches: dict = {"stock": [], **{rank: [] for rank in range(WORLD)}}
    ropes: list = []

    def who():
        return "stock" if threading.current_thread() is threading.main_thread() else _rank()

    def record_adaln(x, *args, **kwargs):
        launches[who()].append(tuple(x.shape[:2]))
        return adaln(x, *args, **kwargs)

    def record_rope(q, k, freqs_cis, *args, **kwargs):
        ropes.append((q.shape[1], k.shape[1], freqs_cis.shape[1]))
        return rope(q, k, freqs_cis, *args, **kwargs)

    monkeypatch.setattr(kitchen, "rms_adaln", record_adaln)
    monkeypatch.setattr(kitchen, "rms_rope_split_half", record_rope)
    model = _tiny(comfy_moss)
    inputs = _inputs(grid, text, batch)
    expected, _calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    rows, hw = grid[0] * grid[1] + text, grid[0] * grid[1]
    assert launches["stock"] == [(batch, rows)] * (2 * LAYERS) + [(batch, hw)]
    outputs, _harness = _sharded(monkeypatch, model, inputs, {}, kernel)
    for rank in range(WORLD):
        assert launches[rank] == launches["stock"], (rank, launches)
    assert ropes and all(q == k == table for q, k, table in ropes)
    assert _same(outputs[0], expected)


def test_forward_the_override_runs_once_a_block_and_never_for_text(monkeypatch, comfy_moss):
    """L sharded attentions a forward a rank, each on the rank's rows; the text
    embedder runs ComfyUI's default backend; an attention that drops the options
    refuses after the loop."""
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6)
    _expected, _calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    sharded_rows: dict = {rank: [] for rank in range(WORLD)}
    overrides: dict = {rank: [] for rank in range(WORLD)}

    def wrap(dispatch):
        def recording(q, k, v, heads, **kwargs):
            sharded_rows[_rank()].append(q.shape[1])
            return dispatch(q, k, v, heads, **kwargs)
        return recording

    def record(q, k, v, heads, *args, **kwargs):
        overrides[_rank()].append((kwargs.get("transformer_options") or {}).get(
            "optimized_attention_override"))
        return kernel(q, k, v, heads, *args, **kwargs)

    monkeypatch.setattr(comfy_moss.model, "optimized_attention", record)
    _sharded(monkeypatch, model, inputs, {}, kernel, wrap_dispatch=wrap)
    for rank in range(WORLD):
        assert sharded_rows[rank] == [8] * LAYERS      # n = ceil(15 / 2), never T = 6
        text_calls = len(model.text_embedder.blocks)
        assert overrides[rank][:text_calls] == [default_attention_override] * text_calls
        assert all(getattr(o, base.USP_ATTENTION_OVERRIDE_ATTR, False)
                   and o is not default_attention_override for o in overrides[rank][text_calls:])
    monkeypatch.setattr(comfy_moss.model, "optimized_attention", kernel)

    stock_attention = comfy_moss.model.attention

    def no_options(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options):
        return stock_attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, {})

    monkeypatch.setattr(comfy_moss.model, "attention", no_options)
    with torch.no_grad():
        model(*inputs, transformer_options={})        # stock still runs
    with pytest.raises(UnsupportedModelError, match="3 blocks ran and 0") as caught:
        _sharded(monkeypatch, model, inputs, {}, kernel)
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS


def test_forward_prefetch_follows_stock_without_the_malloc_graph(monkeypatch, comfy_moss):
    """The queue over list(self.layers), stock's pops with the block scope, no graph."""
    prefetch = comfy_moss.prefetch
    real = {name: getattr(prefetch, name) for name in
            ("make_prefetch_queue", "prefetch_queue_pop", "malloc_graph_begin", "malloc_graph_end")}
    events: dict = {}

    def note(event):
        key = "stock" if threading.current_thread() is threading.main_thread() else _rank()
        events.setdefault(key, []).append(event)

    def make_prefetch_queue(queue, device, options):
        note(("queue", [type(module).__name__ for module in queue]))
        return real["make_prefetch_queue"](queue, device, options)

    def prefetch_queue_pop(queue, device, module, *args, **kwargs):
        note(("pop", type(module).__name__ if module is not None else None, args,
              kwargs.get("malloc_scope")))
        return real["prefetch_queue_pop"](queue, device, module, *args, **kwargs)

    def malloc_graph_begin(*args, **kwargs):
        note(("graph begin",))
        return real["malloc_graph_begin"](*args, **kwargs)

    def malloc_graph_end(*args, **kwargs):
        note(("graph end",))
        return real["malloc_graph_end"](*args, **kwargs)

    for name, function in (("make_prefetch_queue", make_prefetch_queue),
                           ("prefetch_queue_pop", prefetch_queue_pop),
                           ("malloc_graph_begin", malloc_graph_begin),
                           ("malloc_graph_end", malloc_graph_end)):
        monkeypatch.setattr(prefetch, name, function)
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6)
    _expected, _calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    stock = events.pop("stock")
    assert ("queue", ["Block"] * LAYERS) in stock and ("graph begin",) in stock
    _sharded(monkeypatch, model, inputs, {}, kernel)
    expected = [event for event in stock if event[0] not in ("graph begin", "graph end")]
    for rank in range(WORLD):
        assert events[rank] == expected, (rank, events[rank])
    block_pops = [event for event in expected if event[1] == "Block"]
    assert block_pops == [("pop", "Block", (torch.float32,), "block")] * LAYERS


def test_forward_a_padded_ring_refuses_class_k_and_a_waived_ring_attends_the_pad(monkeypatch, comfy_moss):
    """Ring and hybrid have no full-sequence point to drop a pad at."""
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6)
    expected, _calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, {})
    texts: list = []
    model.text_embedder.register_forward_pre_hook(lambda module, args: texts.append(1))
    with pytest.raises(UnsupportedModelError, match="ring/hybrid") as caught:
        _sharded(monkeypatch, model, inputs, {}, kernel, ring_world=2)
    tag = _tag(caught.value)
    assert tag.refusal_class is RefusalClass.KNOWN_WRONG and tag.guard == base.RING_PAD_GUARD
    assert texts == []
    monkeypatch.setattr(accuracy_waiver, "waived", lambda guard: guard == base.RING_PAD_GUARD)
    outputs, harness = _sharded(monkeypatch, model, inputs, {}, kernel, ring_world=2)
    assert all(k.shape[1] == 16 for _entry, _q, k, _v in harness.seen[0])
    assert not _same(outputs[0], expected)


def _foreign_override(func, *args, **kwargs):
    return func(*args, **kwargs)


def _attn1_hook(q, k, v, extra_options):
    return q, k, v


GUARDS = {
    "foreign-override": ({"optimized_attention_override": _foreign_override}, CHANNELS,
                         "optimized_attention_override"),
    "attn1-hook": ({"patches": {"attn1_patch": [_attn1_hook]}}, CHANNELS, "attn1_patch"),
    "control-channels": ({}, 3 * CHANNELS, "384 latent channels"),
}


@pytest.mark.parametrize("case", sorted(GUARDS))
def test_forward_guards_refuse_before_the_first_collective(monkeypatch, comfy_moss, case):
    """Each guard refuses class P on every rank before any exchange."""
    options, channels, named = GUARDS[case]
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6, channels=channels)
    harness = _StockKernelHarness(WORLD, comfy_moss.model.optimized_attention)
    with pytest.raises(UnsupportedModelError, match=named) as caught:
        _sharded(monkeypatch, model, inputs, options, harness.kernel, harness=harness)
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    assert harness.fabric.counts == {}


def test_forward_an_attention_keyword_the_dispatcher_drops_refuses_class_p(monkeypatch, comfy_moss):
    """A ComfyUI that starts passing scale= to the block attention is refused (#116)."""
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6)
    kernel = comfy_moss.model.optimized_attention

    def with_scale(q, k, v, heads, *args, **kwargs):
        override = (kwargs.get("transformer_options") or {}).get("optimized_attention_override")
        if override is not None and override is not default_attention_override:
            kwargs["scale"] = 0.1
        return kernel(q, k, v, heads, *args, **kwargs)

    monkeypatch.setattr(comfy_moss.model, "optimized_attention", with_scale)
    harness = _StockKernelHarness(WORLD, kernel)
    with pytest.raises(UnsupportedModelError, match="'scale'") as caught:
        _sharded(monkeypatch, model, inputs, {}, kernel, harness=harness)
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    assert "docs/TROUBLESHOOTING.md #116" in str(caught.value)
    assert harness.fabric.counts == {}


def test_forward_a_subclass_tree_refuses_at_inject_on_the_forced_path(comfy_moss):
    """The forced family_adapter path binds without matches(); the tree check refuses."""
    stock = comfy_moss.model.MOSSImage

    class Derivative(stock):
        def _forward(self, x, timestep, context, transformer_options={}, **kwargs):  # noqa: B006
            return stock._forward(self, x, timestep, context, transformer_options, **kwargs)

    model = _tiny(comfy_moss)
    model.__class__ = Derivative
    with pytest.raises(UnsupportedModelError, match=r"Derivative\._forward") as caught:
        MOSSImageAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=None))
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    assert not any("forward" in vars(block) for block in model.layers)
    assert "_forward" not in vars(model)


# --- the LoRA half: ComfyUI's key map and merge on a tiny real model_base.MOSSImage ---


def _vendor_file(moss, tmp_path, *, fused=True, dtype=torch.bfloat16, seed=0) -> str:
    """The tiny model's weights under the vendor file's key names, rope.bands included."""
    from safetensors.torch import save_file

    tiny = _tiny(moss, fused=fused, text_dim=VENDOR_TEXT_DIM, seed=seed)
    state = {key: value.to(dtype).contiguous() for key, value in tiny.state_dict().items()}
    state["rope.bands"] = torch.arange(16, dtype=torch.float32)
    path = tmp_path / f"moss_tiny_{'fused' if fused else 'split'}_{str(dtype)[6:]}_{seed}.safetensors"
    save_file(state, str(path))
    return str(path)


def _base_model(moss, tmp_path, *, fused=True, dtype=torch.bfloat16, seed=0):
    """A tiny real model_base.MOSSImage through ComfyUI's own diffusion-model load."""
    import comfy.model_base
    import comfy.sd

    path = _vendor_file(moss, tmp_path, fused=fused, dtype=dtype, seed=seed)
    patcher = comfy.sd.load_diffusion_model(path, model_options={"dtype": dtype})
    assert type(patcher.model) is comfy.model_base.MOSSImage
    return patcher, path


_SHAPES = {"q": (DIM, DIM), "k": (DIM, DIM), "v": (DIM, DIM), "o": (DIM, DIM),
           "ffn_up_gate": (2 * FFN, DIM), "ffn_down": (DIM, FFN), "proj": (DIM, VENDOR_TEXT_DIM)}


def _rank4_lora(units, *, seed=7, rank=4) -> dict[str, torch.Tensor]:
    """Kohya keys as the Kromoss DMD files write them, at rank 4."""
    generator = torch.Generator().manual_seed(seed)
    lora = {}
    for unit in units:
        rows, cols = _SHAPES[unit.rsplit(".", 1)[1]]
        lora[f"diffusion_model.{unit}.lora_up.weight"] = torch.randn(rows, rank, generator=generator) * 0.1
        lora[f"diffusion_model.{unit}.lora_down.weight"] = torch.randn(rank, cols, generator=generator) * 0.1
        lora[f"diffusion_model.{unit}.alpha"] = torch.tensor(float(rank))
    return lora


_LORA_UNITS = (*(f"layers.0.proj.{name}" for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")),
               "layers.2.proj.v")


def test_lora_the_load_pops_rope_bands_and_holds_no_fp32_parameter(comfy_moss, tmp_path):
    """Loading a vendor file drops the rope bands and keeps every parameter in BF16 with fused qkv."""
    patcher, _path = _base_model(comfy_moss, tmp_path)
    assert not any("rope.bands" in key for key in patcher.model.state_dict())
    assert {parameter.dtype for parameter in patcher.model.parameters()} == {torch.bfloat16}
    assert patcher.model.diffusion_model.layers[0].fused_qkv is True


def test_lora_comfys_key_map_routes_split_projections_to_fused_qkv_thirds(comfy_moss, tmp_path):
    """ComfyUI's key map sends split q, k and v LoRA keys to their thirds of the fused qkv weight."""
    import comfy.lora

    patcher, _path = _base_model(comfy_moss, tmp_path)
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    for layer in range(LAYERS):
        qkv = f"diffusion_model.layers.{layer}.proj.qkv.weight"
        for index, name in enumerate(("q", "k", "v")):
            expected = (qkv, (0, index * DIM, DIM))
            assert key_map[f"diffusion_model.layers.{layer}.proj.{name}"] == expected
            assert key_map[f"lora_unet_layers_{layer}_proj_{name}"] == expected
    assert key_map["diffusion_model.layers.0.proj.o"] == "diffusion_model.layers.0.proj.o.weight"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_lora_the_stores_bake_equals_comfys_own_merge_on_every_patched_key(comfy_moss, tmp_path, dtype):
    """The ambient-verify contract (actor/store_bake.py) on MOSS-Image's own keys."""
    import comfy.lora

    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.unbake import live_tensor

    reference, _path = _base_model(comfy_moss, tmp_path, dtype=dtype)
    ours, _path = _base_model(comfy_moss, tmp_path, dtype=dtype)
    key_map = comfy.lora.model_lora_keys_unet(ours.model, {})
    patches = comfy.lora.load_lora(_rank4_lora(_LORA_UNITS), key_map)
    reference.add_patches(patches, 0.7)
    ours.add_patches(patches, 0.7)
    keys = sorted(ours.patches)
    assert len(ours.patches["diffusion_model.layers.0.proj.qkv.weight"]) == 3
    assert len(keys) == 5
    cpu = torch.device("cpu")
    for key in keys:
        expected = reference.patch_weight_to_device(key, device_to=cpu, return_weight=True)
        ModelStore._bake_key(ours, key)
        baked = live_tensor(ours.model, key)
        assert baked.dtype == expected.dtype == dtype
        assert torch.equal(baked, expected), key


def test_lora_a_q_only_delta_changes_only_qs_rows_and_the_unbake_restores_the_bytes(comfy_moss, tmp_path):
    """A q-only LoRA changes only q's rows of the fused weight, and the un-bake restores its bytes."""
    import comfy.lora

    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.unbake import capture_unbake_record, live_tensor, restore_pristine

    patcher, path = _base_model(comfy_moss, tmp_path)
    key = "diffusion_model.layers.1.proj.qkv.weight"
    before = live_tensor(patcher.model, key).clone()
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    patches = comfy.lora.load_lora(_rank4_lora(("layers.1.proj.q",)), key_map)
    assert list(patches) == [(key, (0, 0, DIM))]
    active = patcher.clone()
    active.add_patches(patches, 1.0)
    record = capture_unbake_record(active.model, list(active.patches), path)
    ModelStore._bake_key(active, key)
    after = live_tensor(active.model, key)
    assert not torch.equal(after[:DIM], before[:DIM])
    assert torch.equal(after[DIM:], before[DIM:])
    restore_pristine(active.model, record)
    assert torch.equal(live_tensor(active.model, key), before)


def test_lora_the_fsdp_bake_equals_comfys_merge_on_a_sliced_and_a_root_group_key(comfy_moss, tmp_path):
    """FSDP2 on a one-rank gloo group, wrapped as adapters/fsdp.py wraps MOSS-Image:
    the blocks one by one through ``layers``, the text embedder and every other
    module in the root group. A sliced qkv key and a root-group key both bake
    through actor/fsdp_lora.py to ComfyUI's merge on an unwrapped twin."""
    import comfy.lora
    import torch.distributed.fsdp
    from torch.distributed.device_mesh import init_device_mesh

    from dgx_monarch.actor import fsdp_lora
    from dgx_monarch.adapters.fsdp import _BLOCK_LIST_ATTRS
    from dgx_monarch.adapters.fsdp_islands import auxiliary_module_parameters, resolve_module_path

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    reference, _path = _base_model(comfy_moss, tmp_path)
    active, path = _base_model(comfy_moss, tmp_path)
    model = active.model
    key_map = comfy.lora.model_lora_keys_unet(model, {})
    patches = comfy.lora.load_lora(_rank4_lora(("layers.0.proj.q", "text_embedder.proj")), key_map)
    assert set(patches) == {("diffusion_model.layers.0.proj.qkv.weight", (0, 0, DIM)),
                            "diffusion_model.text_embedder.proj.weight"}
    reference.add_patches(patches, 0.7)
    active.add_patches(patches, 0.7)
    cpu = torch.device("cpu")
    expected = {key: reference.patch_weight_to_device(key, device_to=cpu, return_weight=True)
                for key in reference.patches}
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        dm = model.diffusion_model
        for parameter in dm.parameters():
            parameter.requires_grad = False
        ignored = auxiliary_module_parameters(dm, model)
        wrapped = 0
        for attr in _BLOCK_LIST_ATTRS:
            for block in resolve_module_path(dm, attr) or ():
                torch.distributed.fsdp.fully_shard(block, mesh=mesh, ignored_params=ignored)
                wrapped += 1
        assert wrapped == LAYERS
        torch.distributed.fsdp.fully_shard(dm, mesh=mesh, reshard_after_forward=True,
                                           ignored_params=ignored)
        assert type(dm.layers[0].proj["qkv"].weight).__name__ == "DTensor"
        assert type(dm.text_embedder.proj.weight).__name__ == "DTensor"
        fsdp_lora.apply_stack(SimpleNamespace(swap_verify=-1), active, path, None)
        for key, want in expected.items():
            live = fsdp_lora.live_tensor(model, key)
            local = live if type(live) is torch.nn.Parameter else fsdp_lora.local_shard(live)
            assert local.dtype == want.dtype == torch.bfloat16
            assert torch.equal(local, want), key
    finally:
        torch.distributed.destroy_process_group()


# --- the conds and guidance half ------------------------------------------------------


def _conds(base_model, length, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return base_model.extra_conds(cross_attn=torch.randn(1, length, VENDOR_TEXT_DIM, generator=generator),
                                  noise=torch.zeros(1, CHANNELS, 4, 4), device=torch.device("cpu"))


def test_guidance_conds_are_cond_regular_and_fold_only_at_equal_length(comfy_moss, tmp_path):
    """No pad is exact, so an unequal pair never folds into one call."""
    import comfy.conds

    patcher, _path = _base_model(comfy_moss, tmp_path)
    base_model = patcher.model
    short, long, other = _conds(base_model, 10), _conds(base_model, 23), _conds(base_model, 10, seed=1)
    assert set(short) == {"c_crossattn"}
    assert type(short["c_crossattn"]) is comfy.conds.CONDRegular
    assert not any(isinstance(cond, comfy.conds.CONDConstant) for cond in short.values())
    assert not short["c_crossattn"].can_concat(long["c_crossattn"])
    assert short["c_crossattn"].can_concat(other["c_crossattn"])
    assert base_model.concat_cond(noise=torch.zeros(1, CHANNELS, 4, 4), device=torch.device("cpu")) is None
    x_in = torch.zeros(2, CHANNELS, 4, 4)
    assert cfg_dispatch.conds_fold([[{"model_conds": short}], [{"model_conds": long}]], x_in) is False
    assert cfg_dispatch.conds_fold([[{"model_conds": short}], [{"model_conds": other}]], x_in) is True


def test_guidance_comfys_skip_is_admitted_and_the_sharded_forward_honours_it(monkeypatch, comfy_moss, tmp_path):
    """ComfyUI's own SkipLayerGuidanceDiTSimple, its two in-window calls, and the
    exact callable it installs, admitted by the vetting and honored bit for bit."""
    # Plain imports: every ComfyUI that ships MOSS-Image ships this node, so a
    # missing module is a broken tree, not a reason to skip.
    import comfy.samplers
    from comfy_extras import nodes_slg

    patcher, _path = _base_model(comfy_moss, tmp_path)
    out = nodes_slg.SkipLayerGuidanceDiTSimple.execute(
        patcher, 0.0, 1.0, double_layers="", single_layers="1")
    patched = out[0]
    assert patched is not patcher
    function = patched.model_options["sampler_calc_cond_batch_function"]
    seen: list = []

    def calc_cond_batch(model, conds, x_in, timestep, model_options):
        seen.append((conds, model_options))
        return [torch.zeros_like(x_in) for _ in conds]

    monkeypatch.setattr(comfy.samplers, "calc_cond_batch", calc_cond_batch)
    cond, uncond = [{"cond": 1}], [{"uncond": 1}]
    x = torch.zeros(1, CHANNELS, 4, 4)
    function({"conds": [cond, uncond], "input": x, "sigma": torch.tensor([0.5]),
              "model": patcher.model, "model_options": patched.model_options})
    assert [conds for conds, _options in seen] == [[cond, None], [None, uncond]]
    assert seen[0][1] is patched.model_options
    skip = seen[1][1]["transformer_options"]["patches_replace"]["dit"][("single_block", 1)]
    assert is_identity_block_patch(skip)
    options = {"patches_replace": {"dit": {("single_block", 1): skip}}}
    assert vet_block_replacements(options, LAYERS) == {1: skip}
    model = _tiny(comfy_moss)
    inputs = _inputs((3, 3), 6)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_moss, model, inputs, options)
    outputs, harness = _sharded(monkeypatch, model, inputs, options, kernel)
    _assert_stock_operands_and_output(stock_calls, harness, expected, outputs, blocks=LAYERS - 1)
