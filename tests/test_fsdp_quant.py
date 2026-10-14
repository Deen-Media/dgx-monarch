"""Quantized checkpoints under FSDP.

comfy_kitchen's QuantizedTensor cannot be chunked by FSDP2's init; the
ShardedQuantWeight subclass adds the structural ops and the two all-gather
hooks. Pinned here on CPU: chunking that keeps every scale whole, the hook
round trip through FSDP2's own init ops for every row count from 1 to 8 at
world 2 and 4, the launch admission, and a one-rank gloo FSDP2 forward whose
module weight is the wrapper during compute.
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


def _int8_convrot(rows=6, cols=16):
    w = torch.randn(rows, cols, dtype=torch.bfloat16)
    return QuantizedTensor.from_float(w, "TensorWiseINT8Layout", per_channel=True,
                                      convrot=True, convrot_groupsize=16)


_LAYOUTS = {"fp8": _fp8, "int8_rowwise": _int8_rowwise, "int8_convrot": _int8_convrot}


def _fsdp2_local_shard(full, world, rank):
    """This rank's local shard, built with FSDP2's own init ops.

    Follows torch 2.12 ``FSDPParam._init_sharded_param`` step for step (chunk
    with empty chunks, ``new_zeros``, ``narrow``, ``copy_``, the flat view),
    then the lazy-init re-pad in ``reset_sharded_param``, which a tensor
    subclass always takes because it never matches the same-storage check.
    """
    # The chunking FSDP2's own init runs. It is private to torch, so the
    # import stays here, where a moved symbol fails only the tests that use it.
    from torch.distributed.fsdp._fully_shard._fsdp_common import _chunk_with_empty

    chunks = _chunk_with_empty(full, world, dim=0)
    sharded = chunks[rank]
    padded_size = chunks[0].size()
    padded = full.new_zeros(padded_size)
    if sharded.numel() > 0:
        padded.narrow(dim=0, start=0, length=sharded.size(0)).copy_(sharded)
    assert tuple(padded.view(-1).shape) == (padded.numel(),)
    length = sharded.size(0) if sharded.numel() > 0 else 0
    local = padded.narrow(dim=0, start=0, length=length)
    assert local.is_contiguous()
    if local.size() != padded_size:
        repad = local.new_zeros(padded_size)
        repad.narrow(dim=0, start=0, length=length).copy_(local)
        local = repad.narrow(dim=0, start=0, length=length)
    return local, padded_size


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
    assert s._row_fields == ("scale",)
    chunks = torch.chunk(s, 2, dim=0)
    assert [tuple(c.shape) for c in chunks] == [(4, 4), (3, 4)]
    # The per-row scale stays whole and shared on every chunk; a shard never
    # infers its params from its own row count.
    assert all(c._params.scale is qi._params.scale for c in chunks)
    assert all(c._row_fields == ("scale",) for c in chunks)
    padded = chunks[1].new_zeros((4, 4))
    assert padded._params.scale is qi._params.scale
    padded.narrow(0, 0, 3).copy_(chunks[1])
    assert torch.equal(padded._qdata[:3], qi._qdata[4:7])
    assert not padded._qdata[3:].any()
    assert torch.equal(padded._params.scale, qi._params.scale)
    flat = padded.view(-1)
    assert tuple(flat.shape) == (16,)
    again = torch.as_strided(s, s.shape, s.stride())
    assert again is not s and again._qdata.data_ptr() == s._qdata.data_ptr()
    assert isinstance(torch.nn.Parameter(s, requires_grad=False), QuantizedTensor)
    with pytest.raises(RuntimeError, match="dim-0"):
        torch.chunk(s, 2, dim=1)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize("kind", sorted(_LAYOUTS))
def test_hook_round_trip_rebuilds_the_full_wrapper(kind, rows, world):
    """Every rank hands FSDP2 the same input count at the padded size, and
    every rank rebuilds the original weight from its own metadata.

    Before the row-aligned params stayed whole, a per-row int8 scale at world
    2 rebuilt wrong at 2 rows, sent one input on one rank and two on the
    other at 3 rows, and failed torch's padded-size check on rank 1 alone at
    every odd row count from 5 up.
    """
    import dataclasses

    cls = fsdp_quant.sharded_quant_class()
    torch.manual_seed(10 * rows + world)
    cols = 16
    qt = _LAYOUTS[kind](rows, cols)
    full = cls.from_quantized(qt)
    ranks = []
    for rank in range(world):
        local, padded_size = _fsdp2_local_shard(full, world, rank)
        assert local._row_fields == full._row_fields
        for field in dataclasses.fields(qt._params):
            value = getattr(qt._params, field.name)
            if isinstance(value, torch.Tensor):
                assert torch.equal(getattr(local._params, field.name), value)  # whole on every rank
        inputs, metadata = local.fsdp_pre_all_gather(
            _Mesh(world), full.shape, full.stride(), None, None)
        # torch's rule (_fsdp_param.py all_gather_inputs) holds every input
        # of a padded rank to the padded sharded size; here every input on
        # every rank has it.
        assert [t.size() for t in inputs] == [padded_size]
        ranks.append((local, inputs, metadata))
    assert {len(inputs) for _, inputs, _ in ranks} == {1}
    outputs = tuple(torch.cat([inputs[i] for _, inputs, _ in ranks], dim=0)
                    for i in range(len(ranks[0][1])))
    expected = qt._qdata.float() * qt._params.scale.float()
    for local, _, metadata in ranks:
        rebuilt, keep = local.fsdp_post_all_gather(outputs, metadata, torch.bfloat16)
        assert type(rebuilt) is cls and tuple(rebuilt.shape) == (rows, cols)
        assert rebuilt._row_fields == full._row_fields
        assert torch.equal(rebuilt._qdata, qt._qdata)
        assert torch.equal(rebuilt._params.scale, qt._params.scale)
        assert torch.equal(rebuilt._qdata.float() * rebuilt._params.scale.float(), expected)
        assert keep == outputs
    x = torch.ones(2, cols, dtype=torch.bfloat16)
    assert torch.allclose(torch.nn.functional.linear(x, rebuilt).float(),
                          torch.nn.functional.linear(x, qt).float())
    # The out= path rebinds the views onto the (re-allocated) gather buffers.
    assert local.fsdp_post_all_gather(outputs, metadata, torch.bfloat16, out=rebuilt) is None
    assert torch.equal(rebuilt._qdata, qt._qdata)
    assert torch.equal(rebuilt._params.scale, qt._params.scale)


def test_a_shard_whose_row_param_lost_its_rows_refuses_before_the_gather():
    """The whole-param invariant is checked where FSDP2 asks for the gather
    inputs. Every rank holds the same param, so every rank refuses together."""
    import dataclasses

    cls = fsdp_quant.sharded_quant_class()
    full = cls.from_quantized(_int8_rowwise(6, 4))
    local, _ = _fsdp2_local_shard(full, 2, 1)
    sliced = local._with(local._qdata, dataclasses.replace(
        local._params, scale=local._params.scale[:3]))
    with pytest.raises(RuntimeError, match="does not hold the weight's 6 rows"):
        sliced.fsdp_pre_all_gather(_Mesh(2), full.shape, full.stride(), None, None)


def test_wrap_quantized_parameters_replaces_only_quantized_params():
    lin = torch.nn.Linear(4, 6, bias=True, dtype=torch.bfloat16)
    lin.weight = torch.nn.Parameter(_fp8(6, 4), requires_grad=False)
    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList([lin])
    assert fsdp_quant.wrap_quantized_parameters(root) == 1
    assert type(lin.weight).__name__ == "ShardedQuantWeight"
    assert type(lin.bias) is torch.nn.Parameter
    assert fsdp_quant.local_nbytes(lin.weight) == 24  # 6x4 fp8 bytes, not 2-byte bf16


@pytest.mark.parametrize("make", [_fp8, _int8_rowwise])
def test_fsdp2_forward_computes_on_the_wrapper(tmp_path, make):
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
    qt = make(4, 4)
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
            # The recorded row-aligned names survive Parameter and DTensor wrapping.
            row_fields = ("scale",) if make is _int8_rowwise else ()
            assert sharded._local_tensor._row_fields == row_fields
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
