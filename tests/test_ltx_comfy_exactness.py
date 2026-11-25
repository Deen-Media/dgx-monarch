"""LTX under pure Ulysses hands its row-count-keyed kernels stock's row counts (real Comfy, CPU).

This builds comfy's own ``LTXAVModel`` at toy sizes in float32, with the per-head gates and the
cross-attention AdaLN LTX 2.3 ships, and comfy's own ``LTXVModel`` with and without the
cross-attention AdaLN (LTXV has no gates; its route wraps ``apply_cross_attention_adaln`` in the
block forward itself). Each runs stock ``_process_transformer_blocks`` once, then the adapter's
bound one on two rank threads through the real ``base.make_usp_attention`` dispatch.
Only the all-to-all, the sequence gather and the kernel are stubbed (the harness of
tests/test_flux_pad_exclusion.py). The claims:

* every gate projection call on a rank receives its stream's stock row count, the rank's rows
  first and equal to stock's gate input for those rows, then zero rows;
* every fused AdaLN call on a rank launches over its stream's stock row count;
* each rank's self-attentions reach comfy's paired rope helper as often as stock's do;
* each rank's gathered output equals stock's, and the two ranks' outputs equal each other.

"Equal" follows tests/test_flux3_comfy_exactness.py: bit equality on this rig's aarch64 CPU,
float32 rounding (allclose 2e-5) on x86, whose BLAS may pick another kernel for a halved row
count. Rank against rank is always exact, since both ranks hold the same gathered rows, and so
are the row counts, the call counts and the zero fill rows. On the rig's CPU the GEMM and
comfy-kitchen's eager kernels give a row the same bits at any row count, so the output claim
would hold here without the route too; it pins that the route changes nothing a CPU can see.
Whether cuBLAS and comfy-kitchen's CUDA kernels keep a row's bits at a rank's row count is the
hardware question the route exists for (docs/VALIDATION.md, LTX 2.3 T2V). Skips where ComfyUI
is not importable.
"""
from __future__ import annotations

import copy
import os
import platform
import sys
import threading

import pytest
import torch

from dgx_monarch.adapters import base, ltx_parity
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.ltx import LTXAdapter
from module_location_helpers import from_checkout
from test_flux_pad_exclusion import _LOCAL, _Harness, _install_stubs

WORLD = 2
V_HEADS, A_HEADS, HEAD_DIM = 4, 2, 8
V_DIM, A_DIM = V_HEADS * HEAD_DIM, A_HEADS * HEAD_DIM
CONTEXT_ROWS, BLOCKS = 5, 2
_EXACT_GEMM = platform.machine() == "aarch64"
# (video rows, audio rows, batch, per-token video timestep): even and odd video streams, a
# batch of two, a per-token timestep (its modulation shards with the rows), and audio padded
# on one side at a time. With both streams padded at once (13 video, 7 audio rows) the
# gathered video reads 3.7e-9 from stock on this CPU harness with or without the route, in
# the audio-video cross attention that drops pad rows on both sides; that predates the
# route and is outside what this file claims.
CASES = [(12, 0, 1, False), (13, 0, 1, False), (12, 0, 2, True), (13, 6, 1, False),
         (12, 7, 2, True)]
# LTXV: (video rows, batch, per-token timestep, cross-attention AdaLN). At batch 2 a rank's shard
# and a padded attention's output are strided views; the odd batch-2 cases pin that the plain text
# cross attention and the ungated to_out get them contiguous, as stock's linears do.
LTXV_CASES = [(12, 1, False, True), (13, 1, False, True), (12, 2, True, True),
              (13, 2, False, True), (12, 1, False, False), (13, 2, True, False)]


def _same(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    """Equal bits where the GEMMs are known to agree per row, else float32 rounding."""
    if candidate.shape != reference.shape:
        return False
    if _EXACT_GEMM:
        return torch.equal(candidate, reference)
    return torch.equal(candidate, reference) or torch.allclose(
        candidate, reference, rtol=2e-5, atol=2e-5)


def _gap(candidate: torch.Tensor, reference: torch.Tensor) -> str:
    """The shapes and, where they match, the max abs difference, for an assertion message."""
    if candidate.shape != reference.shape:
        return f"shape {tuple(candidate.shape)} against stock's {tuple(reference.shape)}"
    return (f"shape {tuple(candidate.shape)}, max abs difference "
            f"{float((candidate - reference).abs().max()):.3g}")


def _comfy_dir() -> str:
    return os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_ltx():
    """Import real comfy on its CPU path, then put sys.modules back as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path, original_argv = list(sys.path), sys.argv
    comfy_dir = _comfy_dir()
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-ltx-comfy-exactness", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            model_module = pytest.importorskip("comfy.ldm.lightricks.model")
            av_module = pytest.importorskip("comfy.ldm.lightricks.av_model")
            quant_ops = pytest.importorskip("comfy.quant_ops")
        finally:
            sys.argv = original_argv
        yield ops, model_module, av_module, quant_ops
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def _kernel(ops, q, k, v):
    """The one attention every side runs: comfy's CPU SDPA on (B, H, L, D)."""
    return ops.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())


class _Attention:
    """Stands in for native attention on stock's calls and the stock text cross attention."""

    def __init__(self, ops) -> None:
        self.ops = ops

    def __call__(self, _func, q, k, v, heads, mask=None, **kwargs):
        assert mask is None
        batch, rows, inner = q.shape
        q, k, v = (t.reshape(batch, -1, heads, inner // heads).transpose(1, 2) for t in (q, k, v))
        return _kernel(self.ops, q, k, v).transpose(1, 2).reshape(batch, rows, inner)


setattr(_Attention, base.USP_ATTENTION_OVERRIDE_ATTR, True)  # the rank calls ride our hook


class _KernelHarness(_Harness):
    """The flux harness with the real kernel in place of the tag-sorting one."""

    def __init__(self, world: int, ops) -> None:
        super().__init__(world, strict=False)
        self.ops = ops

    def attend(self, entry, q, k, v):  # (B, L, H, D) over the whole axis
        return _kernel(self.ops, *(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)


def _toy(comfy_ltx):
    ops, _model_module, av_module, _quant_ops = comfy_ltx
    torch.manual_seed(2311)
    model = av_module.LTXAVModel(
        in_channels=8, audio_in_channels=8, cross_attention_dim=V_DIM,
        audio_cross_attention_dim=A_DIM, attention_head_dim=HEAD_DIM,
        audio_attention_head_dim=HEAD_DIM, num_attention_heads=V_HEADS,
        audio_num_attention_heads=A_HEADS, caption_channels=16, num_layers=BLOCKS,
        apply_gated_attention=True, cross_attention_adaln=True,
        # The text connectors run before the block loop and nothing here calls them;
        # at their production defaults (30 heads of 128, two layers) they hold 709M
        # float32 parameters, 2.8 GB a copy and three copies a case.
        connector_attention_head_dim=HEAD_DIM, connector_num_attention_heads=1,
        connector_num_layers=1,
        dtype=torch.float32, device="cpu", operations=ops.disable_weight_init)
    # comfy builds parameters with torch.empty; a checkpoint load fills them.
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.05)
    return model


def _rotation(generator, batch, rows, heads):
    angles = torch.rand(batch, rows, heads, HEAD_DIM // 2, generator=generator) * 6.2831853
    cos, sin = angles.cos(), angles.sin()
    matrix = torch.stack((cos, -sin, sin, cos), dim=-1)
    return (matrix.reshape(batch, rows, heads, HEAD_DIM // 2, 2, 2), True)


def _inputs(av_module, v_rows, a_rows, batch, per_token):
    """The block loop's arguments, built fresh each call: stock updates the streams in place."""
    g = torch.Generator().manual_seed(v_rows * 1000 + a_rows * 10 + batch)
    vx = torch.randn(batch, v_rows, V_DIM, generator=g)
    ax = torch.randn(batch, a_rows, A_DIM, generator=g)
    context = [torch.randn(batch, CONTEXT_ROWS, V_DIM, generator=g),
               torch.randn(batch, CONTEXT_ROWS, A_DIM, generator=g)]
    compressed = av_module.CompressedTimestep

    def emb(rows, width):
        return torch.randn(batch, rows, width, generator=g) * 0.1

    video_ts = compressed(emb(v_rows if per_token else 1, 9 * V_DIM), None if per_token else 4)
    # comfy builds the audio and cross-modal timesteps even with no audio latent
    # (LTXAV.process_timestep always passes a_timestep), one broadcast row each.
    audio_ts = emb(1, 9 * A_DIM)
    cross = [emb(1, 4 * A_DIM), compressed(emb(1, 4 * V_DIM), 4),
             compressed(emb(1, V_DIM), 4), emb(1, A_DIM)]
    timestep = [video_ts, audio_ts, cross, emb(1, 2 * V_DIM), emb(1, 2 * A_DIM)]
    pe = [(_rotation(g, batch, v_rows, V_HEADS), _rotation(g, batch, v_rows, A_HEADS)),
          (_rotation(g, batch, a_rows, A_HEADS), _rotation(g, batch, a_rows, A_HEADS))]
    return [vx, ax], context, timestep, pe


class _Spies:
    """Gate inputs, fused-AdaLN row counts and paired-rope calls, by rank (None for stock)."""

    def __init__(self, monkeypatch, comfy_ltx, models) -> None:
        ops, model_module, _av, quant_ops = comfy_ltx
        self.lock = threading.Lock()
        self.gates: dict = {}
        self.adaln: dict = {}
        self.paired: dict = {}
        gate_stream = {}
        for model in models:
            for block in model.transformer_blocks:
                for name, stream in ltx_parity.GATE_STREAMS:
                    gate = getattr(getattr(block, name, None), "to_gate_logits", None)
                    if gate is not None:
                        gate_stream[id(gate)] = (stream, name)
        linear_forward = ops.disable_weight_init.Linear.forward
        spies = self

        def linear(self, x, *args, **kwargs):
            key = gate_stream.get(id(self))
            if key is not None:
                spies._add(spies.gates, (key, x.clone()))
            return linear_forward(self, x, *args, **kwargs)

        rms_adaln = quant_ops.ck.rms_adaln

        def adaln(x, scale, shift, *args, **kwargs):
            spies._add(spies.adaln, tuple(x.shape[:-1]))
            return rms_adaln(x, scale, shift, *args, **kwargs)

        paired = model_module.apply_rotary_emb_qk

        def rope(q, k, pe):
            spies._add(spies.paired, tuple(q.shape))
            return paired(q, k, pe)

        # Patched before inject, so the bound gate captures the spy as stock's forward.
        monkeypatch.setattr(ops.disable_weight_init.Linear, "forward", linear)
        monkeypatch.setattr(quant_ops.ck, "rms_adaln", adaln)
        monkeypatch.setattr(model_module, "apply_rotary_emb_qk", rope)

    def _add(self, store, item):
        with self.lock:
            store.setdefault(getattr(_LOCAL, "rank", None), []).append(item)


def _compare(monkeypatch, comfy_ltx, model, make_call):
    """Stock's block loop on ``model``, then the bound one on a copy per rank thread.

    ``make_call()`` returns the loop's arguments and keywords, built fresh each call,
    since stock updates the streams in place.
    """
    ops = comfy_ltx[0]
    ranks = [copy.deepcopy(model) for _ in range(WORLD)]
    spies = _Spies(monkeypatch, comfy_ltx, [model, *ranks])
    # Stock runs on this thread as rank None; undone at teardown, so a later file in the
    # same process still reads rank 0 on its main thread.
    monkeypatch.setattr(_LOCAL, "rank", None, raising=False)
    with torch.no_grad():
        args, kwargs = make_call()
        stock = type(model)._process_transformer_blocks(
            model, *args, transformer_options={"optimized_attention_override": _Attention(ops)},
            self_attention_mask=None, **kwargs)

    _install_stubs(monkeypatch, _KernelHarness(WORLD, ops))
    kernel = base.make_usp_attention("TORCH_FLASH")
    for replica in ranks:
        LTXAdapter().inject_usp(replica, InjectionContext(
            topology_sp=WORLD, usp_attention=kernel, pure_ulysses=True))
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                args, kwargs = make_call()
                outputs[rank] = ranks[rank]._process_transformer_blocks(
                    *args, transformer_options={"optimized_attention_override": _Attention(ops)},
                    **kwargs)
        except BaseException as exc:  # surfaced below with its rank
            errors[rank] = exc

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for rank, exc in enumerate(errors):
        if exc is not None:
            raise AssertionError(f"rank {rank} failed") from exc
    return stock, outputs, spies


def _run(monkeypatch, comfy_ltx, case):
    av_module = comfy_ltx[2]

    def make_call():
        x, context, timestep, pe = _inputs(av_module, *case)
        return (x, context, None, timestep, pe), {}

    return _compare(monkeypatch, comfy_ltx, _toy(comfy_ltx), make_call)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"v{c[0]}-a{c[1]}-b{c[2]}-{'tok' if c[3] else 'bc'}")
def test_each_rank_launches_stock_row_counts_and_returns_stock_bits(monkeypatch, comfy_ltx, case):
    v_rows, a_rows, batch, _per_token = case
    stock, outputs, spies = _run(monkeypatch, comfy_ltx, case)
    stock_rows = {ltx_parity.VIDEO: v_rows, ltx_parity.AUDIO: a_rows}

    for rank, out in enumerate(outputs):
        assert _same(out[0], stock[0]), f"rank {rank} video differs from stock: {_gap(out[0], stock[0])}"
        assert torch.equal(out[0], outputs[0][0]), f"rank {rank} video differs from rank 0"
        if a_rows:
            assert _same(out[1], stock[1]), (
                f"rank {rank} audio differs from stock: {_gap(out[1], stock[1])}")
            assert torch.equal(out[1], outputs[0][1]), f"rank {rank} audio differs from rank 0"

    stock_gates = spies.gates.get(None, [])
    assert stock_gates, "stock ran no gate"
    for rank in range(WORLD):
        gates = spies.gates.get(rank, [])
        assert [key for key, _ in gates] == [key for key, _ in stock_gates]
        for ((stream, name), call), (_key, stock_call) in zip(gates, stock_gates, strict=True):
            rows = stock_rows[stream]
            local = -(-rows // WORLD)
            start = rank * local
            real = max(0, min(local, rows - start))
            assert call.shape == stock_call.shape == (batch, rows, call.shape[-1]), name
            # FLUX.3's at_stock_rows: the rank's rows first, pad rows included, then zero rows.
            mine, theirs = call[:, :real], stock_call[:, start:start + real]
            assert _same(mine, theirs), f"{name}: {_gap(mine, theirs)}"
            assert not call[:, local:].any(), f"{name}: the fill rows are not zero"

    for rank in range(WORLD):
        assert sorted(spies.adaln.get(rank, [])) == sorted(spies.adaln[None])
        assert all(shape[-1] in (v_rows, a_rows) for shape in spies.adaln[rank])

    expected_pairs = BLOCKS * (2 if a_rows else 1)
    assert len(spies.paired[None]) == expected_pairs
    for rank in range(WORLD):
        assert len(spies.paired.get(rank, [])) == expected_pairs, f"rank {rank} rope"


def test_outside_the_block_loop_every_bound_call_is_stocks(monkeypatch, comfy_ltx):
    """A bound block run with no shard geometry is stock's code on stock's rows: the same
    GEMM shapes, so its bits equal stock's on every machine."""
    ops, _model_module, av_module, _quant_ops = comfy_ltx
    model = _toy(comfy_ltx)
    replica = copy.deepcopy(model)
    ltx_parity.bind_stock_row_parity(replica, ltx_parity.StockRows())
    options = {"optimized_attention_override": _Attention(ops)}
    with torch.no_grad():
        outs = []
        for subject in (model, replica):
            x, context, timestep, pe = _inputs(av_module, 12, 6, 1, False)
            outs.append(type(subject)._process_transformer_blocks(
                subject, x, context, None, timestep, pe, transformer_options=options))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_every_ltxav_gate_is_bound_and_named_with_its_stream(comfy_ltx):
    model = _toy(comfy_ltx)
    gates = ltx_parity.bind_stock_row_parity(model, ltx_parity.StockRows())
    assert gates == BLOCKS * len(ltx_parity.GATE_STREAMS)
    for block in model.transformer_blocks:
        for name, _stream in ltx_parity.GATE_STREAMS:
            assert "forward" in vars(getattr(block, name).to_gate_logits)
        assert "forward" in vars(block) and ltx_parity.TEXT_CROSS_METHOD in vars(block)


def _toy_ltxv(comfy_ltx, cross_adaln):
    ops, model_module = comfy_ltx[0], comfy_ltx[1]
    torch.manual_seed(2312)
    model = model_module.LTXVModel(
        in_channels=8, cross_attention_dim=V_DIM, attention_head_dim=HEAD_DIM,
        num_attention_heads=V_HEADS, caption_channels=16, num_layers=BLOCKS,
        cross_attention_adaln=cross_adaln, dtype=torch.float32, device="cpu",
        operations=ops.disable_weight_init)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.05)
    return model


def _ltxv_call(v_rows, batch, per_token, cross_adaln):
    """LTXV's block loop arguments: one stream, one timestep tensor, the prompt timestep apart."""
    g = torch.Generator().manual_seed(v_rows * 100 + batch * 10 + per_token + 2 * cross_adaln)
    x = torch.randn(batch, v_rows, V_DIM, generator=g)
    context = torch.randn(batch, CONTEXT_ROWS, V_DIM, generator=g)
    params = 9 if cross_adaln else 6
    timestep = torch.randn(batch, v_rows if per_token else 1, params * V_DIM, generator=g) * 0.1
    pe = _rotation(g, batch, v_rows, V_HEADS)
    kwargs = {}
    if cross_adaln:
        kwargs["prompt_timestep"] = torch.randn(batch, 1, 2 * V_DIM, generator=g) * 0.1
    return (x, context, None, timestep, pe), kwargs


@pytest.mark.parametrize("case", LTXV_CASES,
                         ids=lambda c: f"v{c[0]}-b{c[1]}-{'tok' if c[2] else 'bc'}-{'xadaln' if c[3] else 'plain'}")
def test_ltxv_launches_every_fused_adaln_at_stock_rows_and_returns_stock_bits(
        monkeypatch, comfy_ltx, case):
    """LTXV's route wraps ``apply_cross_attention_adaln`` in the block forward itself."""
    v_rows, _batch, _per_token, cross_adaln = case
    stock, outputs, spies = _compare(monkeypatch, comfy_ltx, _toy_ltxv(comfy_ltx, cross_adaln),
                                     lambda: _ltxv_call(*case))
    for rank, out in enumerate(outputs):
        assert _same(out, stock), f"rank {rank} differs from stock: {_gap(out, stock)}"
        assert torch.equal(out, outputs[0]), f"rank {rank} differs from rank 0"
    assert not spies.gates, "an LTXV block ran a gate projection"
    # The self-attention norm in every block, and the text cross attention's with its AdaLN.
    expected_norms = BLOCKS * (2 if cross_adaln else 1)
    assert len(spies.adaln[None]) == expected_norms
    for rank in range(WORLD):
        assert sorted(spies.adaln.get(rank, [])) == sorted(spies.adaln[None]), f"rank {rank}"
        assert all(shape[-1] == v_rows for shape in spies.adaln[rank])
        assert len(spies.paired.get(rank, [])) == len(spies.paired[None]) == BLOCKS, f"rank {rank} rope"


@pytest.mark.parametrize("exact", [True, False], ids=["aarch64", "x86"])
def test_the_comparison_admits_float32_rounding_off_aarch64_and_refuses_a_real_error(
        monkeypatch, exact):
    """The helper on both branches, whatever this machine is: on x86 a rounding-sized gap
    passes and a 1e-3 error fails, and on aarch64 any gap fails."""
    monkeypatch.setattr(sys.modules[__name__], "_EXACT_GEMM", exact)
    reference = torch.linspace(-1.0, 1.0, 64).reshape(2, 32)
    rounded = reference + 1e-7
    assert _same(reference.clone(), reference)
    assert _same(rounded, reference) is (not exact)
    assert not _same(reference + 1e-3, reference)
    assert not _same(reference[:1], reference)
    assert "max abs difference 0.001" in _gap(reference + 1e-3, reference)
