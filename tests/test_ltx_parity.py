"""LTX stock-row parity on CPU without ComfyUI: geometry, the fills, the gate and norm routes.

adapters/ltx_parity.py launches the two row-count-keyed kernels in an LTX block, comfy-kitchen's
fused AdaLN and the narrow per-head gate projection, at stock's row count while the sharded block
loop runs. These tests pin what each call is handed; tests/test_ltx_comfy_exactness.py runs the
route on real ComfyUI blocks, and the hardware leg is what shows cuBLAS and the CUDA kernels
keep stock's bits.
"""
from __future__ import annotations

import threading
import types

import pytest
import torch

from dgx_monarch.adapters import ltx_parity
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.ltx_parity import (
    AUDIO,
    VIDEO,
    ShardRows,
    StockRows,
    at_stream_rows,
    bind_gate_parity,
    comfy_overlay,
    parity_rms_adaln,
    with_globals,
)
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag


def _row_local(x):
    """A row-local op whose output for a row depends only on that row."""
    return torch.cat((x * 2.0 + 1.0, x.sum(dim=-1, keepdim=True)), dim=-1)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(("stock", "local"), [(12, 6), (13, 7), (5, 2)])
def test_a_call_fills_the_ranks_rows_to_stock_rows_at_the_front(batch, stock, local):
    torch.manual_seed(0)
    piece = torch.randn(batch, local, 4)
    seen = []

    def op(call):
        seen.append(call.clone())
        return _row_local(call)

    out = at_stream_rows(op, piece, ShardRows(stock, local), "gate projection")
    (call,) = seen
    assert call.shape == (batch, stock, 4)
    assert torch.equal(call[:, :local], piece) and not call[:, local:].any()
    assert torch.equal(out, _row_local(piece))
    assert out.stride() == torch.empty(batch, local, 5).stride()


def test_a_call_at_stock_rows_is_the_op_itself():
    x = torch.randn(1, 6, 4)
    seen = []
    out = at_stream_rows(lambda t: seen.append(t) or _row_local(t), x, ShardRows(6, 6), "gate")
    assert seen[0] is x and torch.equal(out, _row_local(x))


def test_a_call_of_another_row_count_refuses_class_p():
    with pytest.raises(UnsupportedModelError) as raised:
        at_stream_rows(_row_local, torch.randn(1, 5, 3), ShardRows(8, 4), "gate projection")
    assert parse_refusal_tag(str(raised.value)).refusal_class is RefusalClass.PHYSICS
    assert "carried 5 rows where this rank holds 4" in str(raised.value)
    assert "canary seam 51" in str(raised.value)


def test_stock_rows_live_only_inside_the_block_loop_and_on_one_thread():
    state = StockRows()
    assert state.rows(VIDEO) is None
    seen = []
    with state.sharded({VIDEO: ShardRows(8, 4)}):
        assert state.rows(VIDEO) == ShardRows(8, 4) and state.rows(AUDIO) is None
        other = threading.Thread(target=lambda: seen.append(state.rows(VIDEO)))
        other.start()
        other.join()
    assert seen == [None]
    assert state.rows(VIDEO) is None
    with pytest.raises(RuntimeError), state.sharded({VIDEO: ShardRows(8, 4)}):
        raise RuntimeError("the block loop failed")
    assert state.streams is None


class _Gated:
    def __init__(self, gated=True):
        torch.manual_seed(3)
        self.to_gate_logits = torch.nn.Linear(4, 2) if gated else None


def test_the_gate_runs_stock_rows_inside_the_loop_and_stock_code_outside():
    attn, state = _Gated(), StockRows()
    stock_gate = torch.nn.Linear(4, 2)
    stock_gate.load_state_dict(attn.to_gate_logits.state_dict())
    calls = []
    original = torch.nn.Linear.forward

    def spy(self, x):
        if self is attn.to_gate_logits:
            calls.append(tuple(x.shape))
        return original(self, x)

    torch.nn.Linear.forward = spy
    try:
        assert bind_gate_parity(attn, VIDEO, state)
        x = torch.randn(1, 3, 4)
        assert torch.equal(attn.to_gate_logits(x), stock_gate(x))
        full = torch.randn(1, 6, 4)
        with state.sharded({VIDEO: ShardRows(6, 3)}):
            out = attn.to_gate_logits(full[:, 3:].clone())
    finally:
        torch.nn.Linear.forward = original
    assert calls == [(1, 3, 4), (1, 6, 4)]
    assert torch.equal(out, stock_gate(full)[:, 3:])


def test_an_attention_without_a_gate_binds_nothing():
    assert not bind_gate_parity(_Gated(gated=False), VIDEO, StockRows())


def test_the_fused_norm_fills_rows_and_a_per_token_modulation_alike():
    state, seen = StockRows(), []

    def stock(x, scale, shift, eps=1e-6):
        seen.append((tuple(x.shape), tuple(scale.shape), tuple(shift.shape), eps))
        return x * (1 + scale) + shift

    norm = parity_rms_adaln(state, VIDEO, stock)
    x, broadcast, per_token = torch.randn(1, 3, 4), torch.randn(1, 1, 4), torch.randn(1, 3, 4)
    norm(x, broadcast, broadcast)
    with state.sharded({VIDEO: ShardRows(7, 3)}):
        filled = norm(x, broadcast, broadcast, 1e-5)
        tokens = norm(x, per_token, per_token)
        with pytest.raises(UnsupportedModelError):
            norm(torch.randn(1, 4, 4), broadcast, broadcast)
    with state.sharded({VIDEO: ShardRows(3, 3)}):
        norm(x, broadcast, broadcast)
    assert seen == [((1, 3, 4), (1, 1, 4), (1, 1, 4), 1e-6),
                    ((1, 7, 4), (1, 1, 4), (1, 1, 4), 1e-5),
                    ((1, 7, 4), (1, 7, 4), (1, 7, 4), 1e-6),
                    ((1, 3, 4), (1, 1, 4), (1, 1, 4), 1e-6)]
    assert torch.equal(filled, x * (1 + broadcast) + broadcast)
    assert torch.equal(tokens, x * (1 + per_token) + per_token)


def test_the_comfy_overlay_replaces_only_the_fused_norm():
    ck = types.SimpleNamespace(rms_adaln=lambda *a: "stock", apply_rope=object())
    comfy = types.SimpleNamespace(quant_ops=types.SimpleNamespace(ck=ck, other=1),
                                  model_management=types.SimpleNamespace(in_training=False))
    overlay = comfy_overlay(comfy, StockRows(), VIDEO)
    assert overlay.model_management is comfy.model_management
    assert overlay.quant_ops.other == 1 and overlay.quant_ops.ck.apply_rope is ck.apply_rope
    assert overlay.quant_ops.ck.rms_adaln is not ck.rms_adaln
    assert overlay.quant_ops.ck.rms_adaln(torch.zeros(1, 2, 3), 0, 0) == "stock"


def _code(source: str, **module_globals):
    namespace = dict(module_globals)
    exec(source, namespace)  # noqa: S102 - builds functions with chosen module globals
    return namespace


def test_with_globals_reads_a_per_call_copy_and_leaves_the_module_alone():
    namespace = _code("def f():\n    return comfy.value\n", comfy=types.SimpleNamespace(value=1))
    routed = with_globals(namespace["f"], {"comfy": lambda c: types.SimpleNamespace(value=c.value + 1)})
    assert routed() == 2 and namespace["f"]() == 1
    namespace["comfy"] = types.SimpleNamespace(value=10)
    assert with_globals(namespace["f"], {"comfy": lambda c: c})() == 10


_AV_SOURCE = '''
class Block:
    cross_attention_adaln = True
    def forward(self, x):
        return comfy.quant_ops.ck.rms_adaln(x, x, x)
    def _apply_text_cross_attention(self, x, context, attn):
        return apply_cross_attention_adaln(x, context, attn)
def apply_cross_attention_adaln(x, context, attn):
    return comfy.quant_ops.ck.rms_adaln(x, x, x)
class Changed:
    cross_attention_adaln = True
    def forward(self, x):
        return ck.rms_adaln(x, 0, 0)
    def _apply_text_cross_attention(self, x, context, attn):
        return other(x)
'''


def test_unroutable_names_each_stock_function_that_stopped_reading_a_routed_global():
    ns = _code(_AV_SOURCE)
    assert ltx_parity.unroutable([ns["Block"]()], ns["apply_cross_attention_adaln"]) == []
    assert ltx_parity.unroutable([ns["Changed"]()], ns["Changed"]._apply_text_cross_attention) == [
        "Changed._apply_text_cross_attention", "Changed.forward", "apply_cross_attention_adaln"]


def test_bind_refuses_before_binding_anything_on_a_changed_block(monkeypatch):
    ns = _code(_AV_SOURCE)
    lightricks = types.ModuleType("comfy.ldm.lightricks.model")
    lightricks.apply_cross_attention_adaln = ns["apply_cross_attention_adaln"]
    for name in ("comfy", "comfy.ldm", "comfy.ldm.lightricks"):
        monkeypatch.setitem(__import__("sys").modules, name, types.ModuleType(name))
    monkeypatch.setitem(__import__("sys").modules, "comfy.ldm.lightricks.model", lightricks)
    good, changed = ns["Block"](), ns["Changed"]()
    model = types.SimpleNamespace(transformer_blocks=[good, changed])
    with pytest.raises(UnsupportedModelError) as raised:
        ltx_parity.bind_stock_row_parity(model, StockRows())
    assert "Changed.forward" in str(raised.value) and "canary seam 51" in str(raised.value)
    assert "forward" not in vars(good), "a block was bound before the refusal"


def test_the_text_cross_attention_routes_each_streams_norm_by_its_attention():
    seen = []
    ck = types.SimpleNamespace(rms_adaln=lambda x, scale, shift, eps=1e-6: seen.append(x.shape[1]) or x)
    comfy = types.SimpleNamespace(quant_ops=types.SimpleNamespace(ck=ck))
    ns = _code(_AV_SOURCE, comfy=comfy)
    block = ns["Block"]()
    block.attn2, block.audio_attn2 = object(), object()
    state = StockRows()
    ltx_parity.bind_norm_parity(block, state)
    with state.sharded({VIDEO: ShardRows(6, 3), AUDIO: ShardRows(4, 2)}):
        block.forward(torch.zeros(1, 3, 2))
        block._apply_text_cross_attention(torch.zeros(1, 3, 2), None, block.attn2)
        block._apply_text_cross_attention(torch.zeros(1, 2, 2), None, block.audio_attn2)
        with pytest.raises(UnsupportedModelError):
            block._apply_text_cross_attention(torch.zeros(1, 2, 2), None, object())
    block._apply_text_cross_attention(torch.zeros(1, 2, 2), None, object())  # stock outside
    assert seen == [6, 6, 4, 2]
