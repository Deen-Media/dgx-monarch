"""Real ComfyUI Boogu: the two-rank pure-Ulysses forward produces exactly the stock output.

This builds ComfyUI's own ``BooguTransformer2DModel`` at toy size (float32,
GQA kept, one refiner layer per stage, two double and two single blocks), runs
its stock forward once, then runs the adapter's sharded forward on two rank
threads through the real ``base.make_usp_attention`` dispatch and full-axis
call. Only the collectives are stubbed (the fabric of
tests/test_flux_pad_exclusion.py), and the stub hands each rank every query
row, as Ulysses does.

The kernel is pinned to comfy's SDPA attention on both sides, because the CPU
default (sub-quadratic) picks its chunk size from free memory. The harness
kernel undoes the adapter's K/V head expansion and calls SDPA exactly as stock
does, with ``enable_gqa`` and the K/V heads stock passes, so whether a GQA
flash kernel equals an expanded one is a hardware question this file does not
answer. It does show that every attention call receives stock's q, k and v, in
stock's order, with no pad row, and the output equals stock's with a largest
absolute difference of 0.0. Boogu takes no guidance input, and
ComfyUI publishes no text mask when every token is real; a partial mask fails
the ``num_tokens`` guard before any shard, so neither is a case here.

ComfyUI imports from ``COMFYUI_DIR`` (default ``~/ComfyUI``); the file skips
where it is not importable. comfy.cli_args parses argv on import, so the
fixture forces ``--cpu``.
"""
from __future__ import annotations

import os
import sys
import threading

import pytest
import torch

import test_flux_pad_exclusion as flux_harness
from dgx_monarch.adapters import boogu_ulysses, chroma_text
from dgx_monarch.adapters.base import InjectionContext, make_usp_attention
from dgx_monarch.adapters.boogu import BooguAdapter
from test_boogu_ulysses_order import SHAPES
from test_flux_pad_exclusion import _LOCAL, _Harness, _install_stubs, _rank

WORLD = 2
HEADS, KV_HEADS, HIDDEN, FEATURES, CHANNELS = 4, 2, 32, 16, 4


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real comfy ops, Boogu and attention, then undo it."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-boogu-exactness", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            boogu = pytest.importorskip("comfy.ldm.boogu.model")
            attention = pytest.importorskip("comfy.ldm.modules.attention")
        finally:
            sys.argv = original_argv
        yield ops, boogu, attention
    finally:
        for name in [n for n in sys.modules if _is_comfy_module(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


@pytest.fixture
def one_thread():
    """Keep the CPU light and every GEMM on one partition."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _model(comfy, state=None):
    ops, boogu, _attention = comfy
    model = boogu.BooguTransformer2DModel(
        patch_size=2, in_channels=CHANNELS, hidden_size=HIDDEN, num_layers=2,
        num_double_stream_layers=2, num_refiner_layers=1, num_attention_heads=HEADS,
        num_kv_heads=KV_HEADS, multiple_of=16, axes_dim_rope=(2, 2, 4), axes_lens=(64, 64, 64),
        instruction_feat_dim=FEATURES, timestep_scale=1000.0, dtype=torch.float32,
        device="cpu", operations=ops.disable_weight_init)
    if state is None:
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
    else:
        model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)   # grad mode is per thread, so no_grad cannot reach the ranks
    return model


def _sdpa(comfy):
    attention = comfy[2].attention_pytorch
    return getattr(attention, "__wrapped__", attention)


class _ComfyHarness(_Harness):
    """The flux fabric with comfy's SDPA as the kernel, recording its operands."""

    def __init__(self, world, sdpa):
        super().__init__(world, strict=False)
        self.sdpa = sdpa
        self.operands: dict[int, list] = {rank: [] for rank in range(world)}

    def attend(self, entry, q, k, v):
        """(B, L, H, D) over the whole axis, K and V expanded to the query heads."""
        repeats = q.shape[2] // KV_HEADS
        k_kv, v_kv = (t[:, :, ::repeats].contiguous() for t in (k, v))
        assert torch.equal(k, k_kv.repeat_interleave(repeats, dim=2))
        assert torch.equal(v, v_kv.repeat_interleave(repeats, dim=2))
        q_h, k_h, v_h = (t.transpose(1, 2) for t in (q, k_kv, v_kv))
        self.operands[_rank()].append((entry, q_h, k_h, v_h))
        out = self.sdpa(q_h, k_h, v_h, q_h.shape[1], None, skip_reshape=True,
                        skip_output_reshape=True, enable_gqa=True)
        return out.transpose(1, 2)


class _WholeAxisUSP:
    """xfuser's own Ulysses call as each rank sees it: every query row
    against every key, then this rank's rows back."""

    ulysses_pg = ring_pg = attn_type = attn_processor = None
    q_descale = k_descale = v_descale = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, attn, q_l, k_l, v_l, **kwargs):
        harness = flux_harness._ACTIVE
        q, k, v = (torch.cat(harness.fabric.exchange(t), dim=1) for t in (q_l, k_l, v_l))
        return harness.attend("xfuser", q, k, v).chunk(harness.world, dim=1)[_rank()]

    @staticmethod
    def ring_attn_fn(q, k, v, **kwargs):
        return flux_harness._ACTIVE.attend("exact", q, k, v)


def _recorder(sdpa, calls):
    """The override stock and the replicated caption refiner run through."""
    def override(func, q, k, v, heads, mask=None, **kwargs):
        calls[_rank()].append(("comfy", q, k, v))
        return sdpa(q, k, v, heads, mask, skip_reshape=kwargs.get("skip_reshape", False),
                    enable_gqa=kwargs.get("enable_gqa", False))
    return override


def _inputs(text_rows, grid, refs, batch):
    generator = torch.Generator().manual_seed(1000 * batch + 10 * text_rows + len(refs))
    x = torch.randn(batch, CHANNELS, 2 * grid[0], 2 * grid[1], generator=generator)
    context = torch.randn(batch, text_rows, FEATURES, generator=generator)
    timesteps = torch.rand(batch, generator=generator)
    # Each reference is one latent row of `count` patches.
    ref_latents = [torch.randn(batch, CHANNELS, 2, 2 * count, generator=generator)
                   for count in refs] or None
    return x, timesteps, context, ref_latents


def _run(monkeypatch, comfy, text_rows, grid, refs, batch, *, pure_ulysses=True, prepare=None):
    sdpa = _sdpa(comfy)
    stock = _model(comfy)
    x, timesteps, context, ref_latents = _inputs(text_rows, grid, refs, batch)
    stock_calls: dict[int, list] = {0: []}
    expected = stock.forward(x, timesteps, context, text_rows, ref_latents=ref_latents,
                             transformer_options={"optimized_attention_override":
                                                  _recorder(sdpa, stock_calls)})

    harness = _ComfyHarness(WORLD, sdpa)
    _install_stubs(monkeypatch, harness)
    monkeypatch.setattr(sys.modules["xfuser.core.long_ctx_attention"],
                        "xFuserLongContextAttention", _WholeAxisUSP)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: WORLD)
    kernel = make_usp_attention("TORCH_FLASH")
    replicated: dict[int, list] = {rank: [] for rank in range(WORLD)}
    models = [_model(comfy, stock.state_dict()) for _ in range(WORLD)]
    for rank, model in enumerate(models):
        if prepare is not None:
            prepare(model, rank)
        BooguAdapter().inject_usp(model, InjectionContext(
            topology_sp=WORLD, usp_attention=kernel, pure_ulysses=pure_ulysses))
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank].forward(
                x, timesteps, context, text_rows, ref_latents=ref_latents,
                transformer_options={"optimized_attention_override": _recorder(sdpa, replicated)})
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for error in errors:
        if error is not None:
            raise error
    calls = {rank: replicated[rank] + harness.operands[rank] for rank in range(WORLD)}
    return expected, outputs, stock_calls[0], calls, stock


@pytest.mark.usefixtures("one_thread")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,refs", SHAPES)
def test_two_rank_pure_ulysses_is_stock_bit_for_bit(
    monkeypatch, comfy, text_rows, grid, refs, batch
):
    expected, outputs, stock_calls, calls, stock = _run(
        monkeypatch, comfy, text_rows, grid, refs, batch)
    # One caption-refiner call per layer stays on comfy's own path; every
    # sharded stage reaches the full-axis kernel or xfuser's own call.
    refiner = len(stock.context_refiner)
    per_stage = len(stock.noise_refiner) + (len(stock.ref_image_refiner) if refs else 0)
    assert len(stock_calls) == refiner + per_stage + 2 * len(stock.double_stream_layers) + len(
        stock.single_stream_layers)
    for rank in range(WORLD):
        assert len(calls[rank]) == len(stock_calls), rank
        assert [entry for entry, *_ in calls[rank][:refiner]] == ["comfy"] * refiner
        for index, ((_entry, *got), (_stock, *want)) in enumerate(
                zip(calls[rank], stock_calls, strict=True)):
            for name, value, reference in zip("qkv", got, want, strict=True):
                assert value.shape == reference.shape, (rank, index, name)
                assert torch.equal(value, reference), (rank, index, name)
        assert outputs[rank].shape == expected.shape
        assert (outputs[rank] - expected).abs().max().item() == 0.0, rank
        assert torch.equal(outputs[rank], expected), rank


def _spy_instruct_linears(model, sink):
    """Record the operand every real instruct Linear receives. The full-row
    helper wraps the instance forward it finds, so the spy sees what the
    wrapped call hands the original module."""
    for index, block in enumerate(model.double_stream_layers):
        for module, joint in boogu_ulysses.instruct_projections(block):
            original = module.forward

            def spy(value, original=original, joint=joint, index=index):
                sink.append((index, joint, value.shape[1], value.stride()))
                return original(value)
            module.forward = spy


@pytest.mark.usefixtures("one_thread")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,refs", [(5, (3, 3), ()), (4, (2, 3), (3,))])
def test_real_instruct_linears_see_every_instruct_row(
    monkeypatch, comfy, text_rows, grid, refs, batch
):
    """This CPU's GEMM gives the same bits at either row count, so the output
    test above cannot see the projection treatment; the operands can."""
    seen: dict[int, list] = {rank: [] for rank in range(WORLD)}
    expected, outputs, *_rest = _run(
        monkeypatch, comfy, text_rows, grid, refs, batch,
        prepare=lambda model, rank: _spy_instruct_linears(model, seen[rank]))
    image = grid[0] * grid[1] + sum(refs)
    for rank in range(WORLD):
        assert len(seen[rank]) == 7 * 2, rank
        for _index, joint, rows, stride in seen[rank]:
            assert rows == text_rows
            if joint:
                assert stride == ((text_rows + image) * HIDDEN, HIDDEN, 1)
        assert torch.equal(outputs[rank], expected)


@pytest.mark.usefixtures("one_thread")
@pytest.mark.parametrize("text_rows,grid,refs", [(5, (3, 3), ()), (4, (3, 3), (2,))])
def test_ring_and_hybrid_contexts_keep_the_rank_major_path(monkeypatch, comfy, text_rows, grid, refs):
    """Without pure Ulysses the forward names no order and no pad rows, so the
    joint keys arrive rank-major and the pad rows are attended."""
    _expected, outputs, stock_calls, calls, _stock = _run(
        monkeypatch, comfy, text_rows, grid, refs, 1, pure_ulysses=False)
    assert torch.equal(outputs[0], outputs[1])
    joint = 1 + 1 + (1 if refs else 0)   # caption, noise, reference, then the first joint call
    assert calls[0][joint][2].shape[2] > stock_calls[joint][2].shape[2]   # a pad row attended


def test_real_comfy_block_selects_seven_instruct_linears(comfy):
    ops, boogu, _attention = comfy

    def block(operations):
        return boogu.BooguDoubleStreamBlock(HIDDEN, HEADS, KV_HEADS, 16, None, 1e-5,
                                            dtype=torch.float32, device="cpu",
                                            operations=operations)

    for family in ("disable_weight_init", "manual_cast", "fp8_ops"):
        real = block(getattr(ops, family))
        processor, feed_forward = real.img_instruct_attn.processor, real.instruct_feed_forward
        selected = boogu_ulysses.instruct_projections(real)
        assert [module for module, _ in selected] == [
            processor.instruct_to_q, processor.instruct_to_k, processor.instruct_to_v,
            processor.instruct_out, feed_forward.linear_1, feed_forward.linear_3,
            feed_forward.linear_2], family
        assert [joint for _, joint in selected] == [False, False, False, True, False, False, False]
    mixed = block(ops.mixed_precision_ops({}, torch.float32))
    assert not isinstance(mixed.img_instruct_attn.processor.instruct_to_q, torch.nn.Linear)
    assert boogu_ulysses.instruct_projections(mixed) == ()
