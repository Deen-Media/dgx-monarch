"""LoRA on FSDP-sharded weights: the full-then-slice bake in actor/fsdp_lora.py.

Geometry against torch.chunk, file reads against a real safetensors file, and
an end-to-end bake on a one-rank gloo FSDP model with comfy's LoRA math faked
as "times two": the shard is written through the DTensor's local view and the
next forward under real FSDP2 sees the baked value, the aliasing the bake
relies on. The verify parks the pristine full weight on the module and puts
the identical DTensor Parameter object back.
"""
from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor import fsdp_lora
from dgx_monarch.actor.unbake import UnbakeError
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.fsdp import validate_fsdp_launch_loras
from dgx_monarch.refusal import RefusalClass, RefusalTag, parse_leading_refusal_tag


@pytest.mark.parametrize("rows", [1, 2, 3, 7, 8, 16, 17])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_shard_rows_match_torch_chunk(rows, world):
    chunks = torch.chunk(torch.arange(rows), world, dim=0)
    for rank in range(world):
        start, stop = fsdp_lora.shard_rows(rows, world, rank)
        expected = chunks[rank].tolist() if rank < len(chunks) else []
        assert list(range(start, stop)) == expected


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header = {}
    blob = b""
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes() if tensor.dtype != torch.uint8 \
            else tensor.numpy().tobytes()
        dtype = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16"}[tensor.dtype]
        header[name] = {"dtype": dtype, "shape": list(tensor.shape),
                        "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


def test_read_full_and_rows_reproduce_the_file(tmp_path):
    weight = torch.arange(12, dtype=torch.float32).reshape(6, 2).to(torch.bfloat16)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {"blocks.0.lin.weight": weight})
    header = fsdp_lora.read_safetensors_header(path)
    ft = header["blocks.0.lin.weight"]
    with open(path, "rb") as f:
        assert torch.equal(fsdp_lora.read_full(f.fileno(), ft), weight)
        assert torch.equal(fsdp_lora.read_rows(f.fileno(), ft, 2, 5), weight[2:5])
        assert fsdp_lora.read_rows(f.fileno(), ft, 6, 6).shape == (0, 2)


def test_lora_on_fsdp_needs_low_rss():
    validate_fsdp_launch_loras(None, lora_low_rss=False)  # no stack: fine
    validate_fsdp_launch_loras(("a",), lora_low_rss=True)
    validate_fsdp_launch_loras(("a",), lora_low_rss=None)  # unknown here: the worker decides
    with pytest.raises(UnsupportedModelError, match="lora_low_rss"):
        validate_fsdp_launch_loras(("a",), lora_low_rss=False)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.lin(x)


class _Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x




def _install_comfy_stub(monkeypatch):
    def get_attr(obj, attr):
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj

    mm = types.ModuleType("comfy.model_management")
    mm.lora_compute_dtype = lambda device: torch.float32
    mm.cast_to_device = lambda w, device, dtype, copy=False: w.detach().clone().to(dtype)
    mp = types.ModuleType("comfy.model_patcher")
    mp.get_key_weight = lambda model, key: (get_attr(model, key), None, None)
    lora = types.ModuleType("comfy.lora")
    lora.calculate_weight = lambda patches, temp, key: temp * 2
    cfloat = types.ModuleType("comfy.float")
    cfloat.stochastic_rounding = lambda out, dtype, seed=0: out.to(dtype)
    cutils = types.ModuleType("comfy.utils")
    cutils.string_to_seed = lambda key: 0
    cutils.get_attr = get_attr
    comfy = types.ModuleType("comfy")
    for name, mod in [("comfy.model_management", mm), ("comfy.model_patcher", mp),
                      ("comfy.lora", lora), ("comfy.float", cfloat), ("comfy.utils", cutils)]:
        setattr(comfy, name.split(".")[1], mod)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "comfy", comfy)


class _Active:
    """The slice of ModelPatcher the bake touches, over a real module tree."""

    def __init__(self, model, patches):
        self.model = model
        self.patches = dict(patches)
        self.backup: dict = {}
        self.backup_buffers: dict = {}
        self.load_device = torch.device("cpu")

    def patch_weight_to_device(self, key, device_to=None, return_weight=False):
        import comfy.float
        import comfy.lora
        import comfy.model_patcher as mp

        weight, _set, _convert = mp.get_key_weight(self.model, key)
        out = comfy.lora.calculate_weight(self.patches[key], weight.detach().clone().float(), key)
        return comfy.float.stochastic_rounding(out, weight.dtype, seed=0)


def _one_rank_fsdp(tmp_path, model):
    from torch.distributed.device_mesh import init_device_mesh

    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        torch.distributed.fsdp.fully_shard(block, mesh=mesh)
    torch.distributed.fsdp.fully_shard(model, mesh=mesh, reshard_after_forward=True)
    return model


def test_bake_writes_the_shard_and_the_next_forward_sees_it(tmp_path, monkeypatch):
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model = _Root()
    pristine = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16)
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(pristine)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {"blocks.0.lin.weight": pristine})
    key = "blocks.0.lin.weight"
    try:
        model = _one_rank_fsdp(tmp_path, model)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        with torch.no_grad():
            before = model(x)
        active = _Active(model, {key: ["patch"]})
        store = SimpleNamespace(swap_verify=-1)
        record = fsdp_lora.apply_stack(store, active, str(path), None)
        assert set(record.mapped) == {key}
        assert active.patches == {}
        sharded = model.blocks[0].lin.weight
        assert type(sharded).__name__ == "DTensor"
        assert torch.equal(fsdp_lora.local_shard(sharded), pristine * 2)
        with torch.no_grad():
            after = model(x)
        assert torch.equal(after, before * 2)
        # A second stack that drops the key restores the pristine rows from disk.
        active2 = _Active(model, {})
        record2 = fsdp_lora.apply_stack(store, active2, str(path), record)
        assert record2.mapped == {}
        assert torch.equal(fsdp_lora.local_shard(sharded), pristine)
        with torch.no_grad():
            assert torch.equal(model(x), before)
    finally:
        torch.distributed.destroy_process_group()


def test_verify_restores_the_identical_parameter_object(tmp_path, monkeypatch):
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model = _Root()
    pristine = torch.ones(4, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(pristine)
    key = "blocks.0.lin.weight"
    try:
        model = _one_rank_fsdp(tmp_path, model)
        sharded = model.blocks[0].lin.weight
        active = _Active(model, {key: ["patch"]})
        fsdp_lora.bake_key(active, key, pristine, 0, 1)
        fsdp_lora.verify_key(active, key, pristine, 0, 1)
        assert model.blocks[0].lin.weight is sharded
        with torch.no_grad():
            out = model(torch.ones(1, 4, dtype=torch.bfloat16))
        assert out.shape == (1, 4)
        # A drifted shard fails the verify and names the key.
        with torch.no_grad():
            fsdp_lora.local_shard(sharded).zero_()
        with pytest.raises(UnbakeError, match="MISMATCH"):
            fsdp_lora.verify_key(active, key, pristine, 0, 1)
        assert model.blocks[0].lin.weight is sharded
    finally:
        torch.distributed.destroy_process_group()


def test_capture_refuses_a_shard_that_is_not_pristine(tmp_path, monkeypatch):
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model = _Root()
    pristine = torch.ones(4, 4, dtype=torch.bfloat16)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {"blocks.0.lin.weight": pristine})
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(pristine * 3)  # live differs from disk
    try:
        model = _one_rank_fsdp(tmp_path, model)
        with pytest.raises(UnbakeError, match="not pristine"):
            fsdp_lora.capture_record(model, ["blocks.0.lin.weight"], str(path), 0, 1)
    finally:
        torch.distributed.destroy_process_group()


# The header preflight (adapters.detect.fsdp_lora_admission_property) and the
# live backstop (adapters.fsdp_lora_admission.refuse_unless_fsdp_lora_admits)
# catch a comfy-kitchen quantized checkpoint before the bake. Neither can
# predict a fp32-island checkpoint's live cast (docs/TROUBLESHOOTING.md #17),
# so _file_tensor's dtype-mismatch check is the only enforcement for that
# shape, and it is typed the same way: a class P refusal, not a bare
# UnbakeError, so a worker raise reads as a deliberate CONSUMED lease
# (nodes/pending.py typed_worker_refusal), not an abandoned one.

def test_capture_refuses_a_krea2_style_dtype_mismatch_typed(tmp_path, monkeypatch):
    """The checkpoint stores this key as fp32 (an unaudited fp32 island) but
    the live FSDP-wrapped parameter is bf16 (cast at load), the shape that
    broke Krea2 RAW bf16. It raises through the real _file_tensor (via
    capture_record) as a tagged class P refusal."""
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model = _Root()  # blocks[0].lin.weight is real bf16, per _Block.__init__
    path = tmp_path / "m.safetensors"
    _write_safetensors(
        path, {"blocks.0.lin.weight": torch.ones(4, 4, dtype=torch.float32)})
    try:
        model = _one_rank_fsdp(tmp_path, model)
        with pytest.raises(
            UnbakeError, match="needs the file's own dtype",
        ) as raised:
            fsdp_lora.capture_record(model, ["blocks.0.lin.weight"], str(path), 0, 1)
        message = str(raised.value)
        assert parse_leading_refusal_tag(message) == RefusalTag(
            refusal_class=RefusalClass.PHYSICS, guard=None, waivable=False)
        assert "resident topology" in message
    finally:
        torch.distributed.destroy_process_group()


class _FakeQuantParam:
    """Stand-in for comfy_kitchen's QuantizedTensor: _is_quantized checks only
    for layout_cls and params, as in tests/test_unbake.py."""

    def __init__(self):
        self.layout_cls = "TensorCoreFP8E4M3Layout"
        self.params = SimpleNamespace()


def test_capture_refuses_a_quantized_key_typed(tmp_path):
    """No FSDP wrap needed: capture_record's quantized check runs before any
    shard read, and raises a tagged class P refusal, not a bare UnbakeError."""
    path = tmp_path / "m.safetensors"
    _write_safetensors(
        path, {"blocks.0.weight": torch.ones(4, 4, dtype=torch.bfloat16)})
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].weight = _FakeQuantParam()  # not a Parameter: stored plain

    with pytest.raises(
        UnbakeError, match="LoRA on quantized shards",
    ) as raised:
        fsdp_lora.capture_record(model, ["blocks.0.weight"], str(path), 0, 1)
    message = str(raised.value)
    assert parse_leading_refusal_tag(message) == RefusalTag(
        refusal_class=RefusalClass.PHYSICS, guard=None, waivable=False)
    assert "resident topology" in message


def test_bake_key_refuses_a_quantized_setter_typed(monkeypatch):
    """bake_key's own quantized detection (a comfy set_func or convert_func on
    the key) carries the same tag as capture_record's."""
    _install_comfy_stub(monkeypatch)
    import comfy.model_patcher as comfy_mp

    quantized_weight = torch.ones(2, 2, dtype=torch.bfloat16)
    monkeypatch.setattr(
        comfy_mp, "get_key_weight",
        lambda _model, _key: (quantized_weight, lambda *_a, **_k: None, None))
    active = SimpleNamespace(
        model=SimpleNamespace(), patches={"w": ["patch"]},
        load_device=torch.device("cpu"))

    with pytest.raises(
        UnbakeError, match="LoRA on quantized shards",
    ) as raised:
        fsdp_lora.bake_key(active, "w", quantized_weight, 0, 1)
    message = str(raised.value)
    assert parse_leading_refusal_tag(message) == RefusalTag(
        refusal_class=RefusalClass.PHYSICS, guard=None, waivable=False)
