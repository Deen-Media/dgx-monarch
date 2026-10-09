"""Quantized checkpoints under FSDP.

comfy_kitchen's QuantizedTensor cannot be chunked by FSDP2's init; the
ShardedQuantWeight subclass adds the structural ops and the two all-gather
hooks. Pinned here on CPU: chunking with row-aligned scales, the hook round
trip for even and uneven shards, the launch admission, and a one-rank gloo
FSDP2 forward whose module weight is the wrapper during compute.
"""
from __future__ import annotations

import pytest
import torch

ck = pytest.importorskip("comfy_kitchen.tensor")

from dgx_monarch.adapters import fsdp_quant  # noqa: E402
from dgx_monarch.adapters.base import UnsupportedModelError  # noqa: E402
from dgx_monarch.adapters.fsdp import (  # noqa: E402
    FSDP_ADMITTED_CHECKPOINT_KINDS,
    fsdp_precision_profile_is_admitted,
    refuse_fsdp_live_cast,
    validate_fsdp_launch_quant,
)

QuantizedTensor = ck.QuantizedTensor


class _Mesh:
    def __init__(self, world):
        self._world = world

    def size(self):
        return self._world


def _fp8(rows=6, cols=4):
    w = torch.randn(rows, cols, dtype=torch.bfloat16)
    return QuantizedTensor.from_float(w, "TensorCoreFP8Layout")


def _int8_rowwise(rows=6, cols=4):
    w = torch.randn(rows, cols, dtype=torch.bfloat16)
    return QuantizedTensor.from_float(w, "TensorWiseINT8Layout", per_channel=True)


def test_admitted_kinds_and_profiles():
    assert {"fp8", "int8"} <= FSDP_ADMITTED_CHECKPOINT_KINDS
    validate_fsdp_launch_quant("fp8")
    validate_fsdp_launch_quant("int8")
    with pytest.raises(UnsupportedModelError, match="no registered shard layout"):
        validate_fsdp_launch_quant("nvfp4")
    with pytest.raises(UnsupportedModelError, match="no registered shard layout"):
        validate_fsdp_launch_quant("mxfp8")
    assert fsdp_precision_profile_is_admitted("uniform_or_quantized_fp8", 0, 0, "fp8")
    assert fsdp_precision_profile_is_admitted("uniform_or_quantized_int8", 0, 0, "int8")
    assert not fsdp_precision_profile_is_admitted("uniform_or_quantized_fp8", 0, 0, "bf16")
    assert not fsdp_precision_profile_is_admitted("uniform_or_quantized_fp8", 1, 8, "fp8")


def test_live_cast_refuses_typed():
    with pytest.raises(UnsupportedModelError, match="no file-backed bytes") as raised:
        refuse_fsdp_live_cast("fp8_e4m3fn")
    assert "[dgxm:P" in str(raised.value)


def test_layout_refusals_name_the_layout():
    assert fsdp_quant.shard_layout_refusal("TensorCoreFP8Layout", _fp8()._params) is None
    assert "no registered shard layout" in fsdp_quant.shard_layout_refusal(
        "TensorCoreNVFP4Layout", _fp8()._params)
    import dataclasses

    transposed = dataclasses.replace(_int8_rowwise()._params, transposed=True)
    assert "transposed" in fsdp_quant.shard_layout_refusal("TensorWiseINT8Layout", transposed)


def test_chunk_pad_copy_view_keep_bytes_and_row_scales():
    cls = fsdp_quant.sharded_quant_class()
    qi = _int8_rowwise(7, 4)
    s = cls.from_quantized(qi)
    assert isinstance(s, QuantizedTensor) and tuple(s.shape) == (7, 4)
    chunks = torch.chunk(s, 2, dim=0)
    assert [tuple(c.shape) for c in chunks] == [(4, 4), (3, 4)]
    assert [tuple(c._params.scale.shape) for c in chunks] == [(4, 1), (3, 1)]
    padded = chunks[1].new_zeros((4, 4))
    assert tuple(padded._params.scale.shape) == (4, 1)
    padded.narrow(0, 0, 3).copy_(chunks[1])
    assert torch.equal(padded._qdata[:3], qi._qdata[4:7])
    assert torch.equal(padded._params.scale[:3], qi._params.scale[4:7])
    flat = padded.view(-1)
    assert tuple(flat.shape) == (16,)
    again = torch.as_strided(s, s.shape, s.stride())
    assert again is not s and again._qdata.data_ptr() == s._qdata.data_ptr()
    assert isinstance(torch.nn.Parameter(s, requires_grad=False), QuantizedTensor)
    with pytest.raises(RuntimeError, match="dim-0"):
        torch.chunk(s, 2, dim=1)


@pytest.mark.parametrize(("make", "rows"), [(_fp8, 6), (_fp8, 7), (_int8_rowwise, 6), (_int8_rowwise, 7)])
def test_hook_round_trip_rebuilds_the_full_wrapper(make, rows):
    cls = fsdp_quant.sharded_quant_class()
    qt = make(rows, 4)
    s = cls.from_quantized(qt)
    world = 2
    chunks = torch.chunk(s, world, dim=0)
    gathered_inputs = [c.fsdp_pre_all_gather(_Mesh(world), s.shape, s.stride(), None, None)
                       for c in chunks]
    sizes = {tuple(inp.shape) for inputs, _ in gathered_inputs for inp in inputs[:1]}
    assert len(sizes) == 1  # every rank hands FSDP the padded chunk size
    outputs = tuple(torch.cat([inputs[i] for inputs, _ in gathered_inputs], dim=0)
                    for i in range(len(gathered_inputs[0][0])))
    metadata = gathered_inputs[0][1]
    full, keep = chunks[0].fsdp_post_all_gather(outputs, metadata, torch.bfloat16)
    assert type(full) is cls and tuple(full.shape) == (rows, 4)
    assert torch.equal(full._qdata, qt._qdata)
    assert torch.equal(full._params.scale, qt._params.scale)
    assert keep == outputs
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    assert torch.allclose(torch.nn.functional.linear(x, full).float(),
                          torch.nn.functional.linear(x, qt).float())
    # The out= path rebinds the views onto the (re-allocated) gather buffers.
    assert chunks[0].fsdp_post_all_gather(outputs, metadata, torch.bfloat16, out=full) is None
    assert torch.equal(full._qdata, qt._qdata)


def test_wrap_quantized_parameters_replaces_only_quantized_params():
    lin = torch.nn.Linear(4, 6, bias=True, dtype=torch.bfloat16)
    lin.weight = torch.nn.Parameter(_fp8(6, 4), requires_grad=False)
    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList([lin])
    assert fsdp_quant.wrap_quantized_parameters(root) == 1
    assert type(lin.weight).__name__ == "ShardedQuantWeight"
    assert type(lin.bias) is torch.nn.Parameter
    assert fsdp_quant.local_nbytes(lin.weight) == 24  # 6x4 fp8 bytes, not 2-byte bf16


def test_fsdp2_forward_computes_on_the_wrapper(tmp_path):
    import torch.distributed.fsdp
    from torch.distributed.device_mesh import init_device_mesh

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    cls = fsdp_quant.sharded_quant_class()
    seen: list[str] = []

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            seen.append(type(self.lin.weight).__name__)
            return torch.nn.functional.linear(x, self.lin.weight)

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([_Block()])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = _Root()
    qt = _fp8(4, 4)
    model.blocks[0].lin.weight = torch.nn.Parameter(qt, requires_grad=False)
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(x, qt)
    # Gloo otherwise resolves the machine hostname, which a fresh CPU or
    # sandbox host need not have in DNS; this one-rank canary is local.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
        torch.distributed.init_process_group(
            backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
        try:
            assert fsdp_quant.wrap_quantized_parameters(model) == 1
            mesh = init_device_mesh("cpu", (1,))
            for p in model.parameters():
                p.requires_grad = False
            for block in model.blocks:
                torch.distributed.fsdp.fully_shard(block, mesh=mesh)
            torch.distributed.fsdp.fully_shard(model, mesh=mesh, reshard_after_forward=True)
            sharded = model.blocks[0].lin.weight
            assert type(sharded).__name__ == "DTensor"
            assert type(sharded._local_tensor) is cls
            with torch.no_grad():
                out = model(x)
                out2 = model(x)
            assert seen == ["ShardedQuantWeight", "ShardedQuantWeight"]
            assert torch.equal(out.float(), reference.float())
            assert torch.equal(out2.float(), reference.float())
            assert model.blocks[0].lin.weight is sharded  # resharded after forward
        finally:
            torch.distributed.destroy_process_group()


def test_the_empty_filler_chunk_keeps_the_column_shape():
    """FSDP2's _chunk_with_empty hands a flat new_empty(0) to the rank whose
    dim-0 chunk is empty (a [1, C] tensor at world 2, found on the 2026-08-26
    census int8-convrot row). The wrapper must return a zero-row
    buffer with the stored column shape instead of refusing the load."""
    cls = fsdp_quant.sharded_quant_class()
    for make in (_fp8, _int8_rowwise):
        w = cls.from_quantized(make(rows=1, cols=12))
        for fn in (torch.Tensor.new_empty, torch.Tensor.new_zeros):
            out = fn(w, 0)
            assert isinstance(out, QuantizedTensor)
            assert tuple(out.shape) == (0, 12)
        # A wrong column shape still refuses.
        with pytest.raises(UnsupportedModelError, match="column shape"):
            w.new_empty((3, 5))
