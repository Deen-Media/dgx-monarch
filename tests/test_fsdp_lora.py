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
import time
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


# A key ComfyUI renames as it loads the file. FLUX checkpoints store a q/k norm
# weight as "<...>_norm.scale"; ComfyUI's model config renames it to
# "<...>_norm.weight" (supported_models.Flux.process_unet_state_dict), and a
# LoRA names the model's key. On 2026-11-11 the FLUX.2 Klein distill LoRA,
# which carries diffs for 27 such weights, crashed uly2+fsdp with a bare
# "no checkpoint tensor for key ...norm.query_norm.weight".

class _NormBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.query_norm = torch.nn.RMSNorm(4, dtype=torch.bfloat16)

    def forward(self, x):
        return self.lin(self.query_norm(x))


def _flux_style_renames(state_dict):
    """ComfyUI's Flux rule, as its model config applies it to a state dict."""
    return {(k[:-len(".scale")] + ".weight" if k.endswith("_norm.scale") else k): v
            for k, v in state_dict.items()}


class _NormRoot(torch.nn.Module):
    def __init__(self, process=_flux_style_renames):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_NormBlock()])
        self.model_config = SimpleNamespace(process_unet_state_dict=process)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


_NORM_KEY = "blocks.0.query_norm.weight"
_NORM_FILE_KEY = "blocks.0.query_norm.scale"


def _norm_model(tmp_path, *, file_norm=None, file_key=_NORM_FILE_KEY, process=_flux_style_renames):
    model = _NormRoot(process)
    lin = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16)
    norm = torch.tensor([1.0, 1.5, 2.0, 2.5]).to(torch.bfloat16)
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(lin)
        model.blocks[0].query_norm.weight.copy_(norm)
    path = tmp_path / "m.safetensors"
    tensors = {"blocks.0.lin.weight": lin}
    if file_key is not None:
        tensors[file_key] = norm if file_norm is None else file_norm
    _write_safetensors(path, tensors)
    return model, path, lin, norm


def test_bake_finds_a_key_comfy_renames_at_load_under_its_file_name(tmp_path, monkeypatch):
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model, path, lin, norm = _norm_model(tmp_path)
    keys = ["blocks.0.lin.weight", _NORM_KEY]
    try:
        model = _one_rank_fsdp(tmp_path, model)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        with torch.no_grad():
            before = model(x)
        store = SimpleNamespace(swap_verify=-1)
        record = fsdp_lora.apply_stack(store, _Active(model, {key: ["patch"] for key in keys}),
                                       str(path), None)
        assert set(record.mapped) == set(keys)
        sharded = model.blocks[0].query_norm.weight
        assert type(sharded).__name__ == "DTensor"
        assert torch.equal(fsdp_lora.local_shard(sharded), norm * 2)
        assert torch.equal(fsdp_lora.local_shard(model.blocks[0].lin.weight), lin * 2)
        with torch.no_grad():
            assert not torch.equal(model(x), before)
        # The record binds the model's key to the file's tensor, so a stack that
        # drops the key restores this rank's rows from disk.
        record2 = fsdp_lora.apply_stack(store, _Active(model, {}), str(path), record)
        assert record2.mapped == {}
        assert torch.equal(fsdp_lora.local_shard(sharded), norm)
        with torch.no_grad():
            assert torch.equal(model(x), before)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("own_name_holds", ["other bytes", "another dtype", "another shape"])
def test_a_key_the_file_holds_under_both_names_binds_the_one_whose_bytes_match(tmp_path, monkeypatch, own_name_holds):
    """ComfyUI loaded the renamed tensor; the file also holds one under the key's own name.

    The lookup used to stop at the first name the header has, bind the wrong
    tensor and call the model not pristine. It now byte-checks each candidate,
    as the resident capture does, and binds the tensor the model holds.
    """
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model, path, lin, norm = _norm_model(tmp_path)
    decoy = {"other bytes": torch.tensor([9.0, 9.0, 9.0, 9.0]).to(torch.bfloat16),
             "another dtype": torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float32),
             "another shape": torch.ones(8).to(torch.bfloat16)}[own_name_holds]
    _write_safetensors(path, {"blocks.0.lin.weight": lin, _NORM_KEY: decoy, _NORM_FILE_KEY: norm})
    try:
        model = _one_rank_fsdp(tmp_path, model)
        record = fsdp_lora.capture_record(model, [_NORM_KEY], str(path), 0, 1)
        header = fsdp_lora.read_safetensors_header(str(path))
        assert record.mapped[_NORM_KEY] == header[_NORM_FILE_KEY]
        assert record.mapped[_NORM_KEY] != header[_NORM_KEY]
        # With neither tensor equal to the live rows the model is not pristine, as before.
        with torch.no_grad():
            fsdp_lora.local_shard(model.blocks[0].query_norm.weight).add_(1)
        with pytest.raises(UnbakeError, match=r"not pristine|needs the file's own dtype"):
            fsdp_lora.capture_record(model, [_NORM_KEY], str(path), 0, 1)
    finally:
        torch.distributed.destroy_process_group()


def test_a_renamed_key_still_has_to_match_the_files_bytes(tmp_path, monkeypatch):
    """The rename only names a candidate; the byte check decides."""
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    other = torch.tensor([9.0, 9.0, 9.0, 9.0]).to(torch.bfloat16)
    model, path, _lin, _norm = _norm_model(tmp_path, file_norm=other)
    try:
        model = _one_rank_fsdp(tmp_path, model)
        with pytest.raises(UnbakeError, match="not pristine"):
            fsdp_lora.capture_record(model, [_NORM_KEY], str(path), 0, 1)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("case", ["no tensor in the file", "no rename known", "config reads tensors"])
def test_capture_refuses_a_key_with_no_checkpoint_tensor_typed(tmp_path, monkeypatch, case):
    """A patched key the file holds under no name is a class P refusal raised
    before any shard is written, not a bare UnbakeError (the 2026-11-11 crash)."""
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)

    def reads_tensors(state_dict):
        return {key: value * 2 for key, value in state_dict.items()}  # fails on a name

    kwargs = {"no tensor in the file": {"file_key": None},
              "no rename known": {"process": None},
              "config reads tensors": {"process": reads_tensors}}[case]
    model, path, lin, norm = _norm_model(tmp_path, **kwargs)
    try:
        model = _one_rank_fsdp(tmp_path, model)
        active = _Active(model, {"blocks.0.lin.weight": ["patch"], _NORM_KEY: ["patch"]})
        with pytest.raises(UnbakeError, match="holds no tensor under that name") as raised:
            fsdp_lora.apply_stack(SimpleNamespace(swap_verify=-1), active, str(path), None)
        message = str(raised.value)
        assert parse_leading_refusal_tag(message) == RefusalTag(
            refusal_class=RefusalClass.PHYSICS, guard=None, waivable=False)
        assert _NORM_KEY in message and "resident topology" in message
        # Raised by the capture, before the bake: no shard changed, the stack is intact.
        assert torch.equal(fsdp_lora.local_shard(model.blocks[0].lin.weight), lin)
        assert torch.equal(fsdp_lora.local_shard(model.blocks[0].query_norm.weight), norm)
        assert set(active.patches) == {"blocks.0.lin.weight", _NORM_KEY}
    finally:
        torch.distributed.destroy_process_group()


# A weight large enough that the sampled check reads only part of it: 64 x 128
# bf16 is 16 KiB, sampled in three 4 KiB windows, so bytes 4096 to 6143 are
# never read by the sampled check.
_BIG_KEY = "blocks.0.lin.weight"
_GAP_ELEMENT = 2100                     # byte 4200, between the first two windows


class _BigBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.lin(x)


class _BigRoot(_Root):
    def __init__(self, process=None):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_BigBlock()])
        if process is not None:
            self.model_config = SimpleNamespace(process_unet_state_dict=process)


def _big_weight() -> torch.Tensor:
    return (torch.arange(64 * 128, dtype=torch.float32).reshape(64, 128) % 97 / 97).to(torch.bfloat16)


def _gap_twin(weight: torch.Tensor) -> torch.Tensor:
    """The same tensor with one element changed where no sampled window reads."""
    twin = weight.clone()
    twin.view(-1)[_GAP_ELEMENT] += 1
    return twin


def test_a_checkpoint_replaced_after_its_header_is_read_bakes_nothing(tmp_path, monkeypatch):
    """The bake reads through one descriptor bound to the header it read.

    A same-size replacement that differs only outside the sampled windows
    passed the sampled check, and the bake read the new file's bytes at the old
    offsets into the shard. Now the open descriptor must be the inode the
    header named, so the replacement refuses before any write.
    """
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model = _BigRoot()
    weight = _big_weight()
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(weight)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {_BIG_KEY: weight})
    replacement = tmp_path / "replacement.safetensors"
    _write_safetensors(replacement, {_BIG_KEY: _gap_twin(weight)})
    real_header = fsdp_lora.read_safetensors_header

    def header_then_replace(*args, **kwargs):
        out = real_header(*args, **kwargs)
        if replacement.exists():
            replacement.replace(path)               # atomic: a new inode at the same name
        return out

    monkeypatch.setattr(fsdp_lora, "read_safetensors_header", header_then_replace)
    try:
        model = _one_rank_fsdp(tmp_path, model)
        with pytest.raises(UnbakeError, match="changed after its header was read"):
            fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0),
                                  _Active(model, {_BIG_KEY: ["patch"]}), str(path), None)
        assert torch.equal(fsdp_lora.local_shard(model.blocks[0].lin.weight), weight)
    finally:
        torch.distributed.destroy_process_group()


def test_a_checkpoint_rewritten_while_the_bake_reads_it_refuses_after_the_reads(tmp_path, monkeypatch):
    """The same inode, its times moved after the first read: the after-read check catches it.

    The bake reads each patched key whole, key after key, through the bound
    descriptor; a file rewritten in place between two reads mixes old and new
    bytes, so the identity is checked again after the last read and the store
    then drops the model and reloads.
    """
    import os

    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model = _Root()
    pristine = torch.ones(4, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(pristine)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {"blocks.0.lin.weight": pristine})
    real_read_full = fsdp_lora.read_full
    inode = os.stat(path).st_ino

    def read_then_touch(fd, ft):
        out = real_read_full(fd, ft)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))   # in place: the same inode
        return out

    monkeypatch.setattr(fsdp_lora, "read_full", read_then_touch)
    try:
        model = _one_rank_fsdp(tmp_path, model)
        with pytest.raises(UnbakeError, match="changed while LoRA bake reads were in progress"):
            fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0),
                                  _Active(model, {"blocks.0.lin.weight": ["patch"]}), str(path), None)
        assert os.stat(path).st_ino == inode
    finally:
        torch.distributed.destroy_process_group()


def _pinned_identity(path):
    import os

    from dgx_monarch.safetensors_header import SafetensorsFileIdentity

    return SafetensorsFileIdentity.from_stat(os.stat(path))


@pytest.mark.parametrize("change", ["another file", "a metadata change"])
def test_the_bake_refuses_a_checkpoint_that_changed_since_the_load_pinned_it(tmp_path, monkeypatch, change):
    """The load pinned the file it built the model from; the bake reads that file or none.

    Another inode at the path, or the same inode after a chmod, no longer
    matches the pin. The refusal says what was seen, and nothing is written.
    """
    import os

    from dgx_monarch.safetensors_header import SafetensorsFileIdentity

    _install_comfy_stub(monkeypatch)
    model = _BigRoot()
    weight = _big_weight()
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(weight)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {_BIG_KEY: weight})
    pinned = _pinned_identity(path)
    if change == "another file":
        expected = SafetensorsFileIdentity(pinned.file_dev, pinned.file_ino + 1, pinned.file_size,
                                           pinned.file_mtime_ns, pinned.file_ctime_ns)
    else:
        expected = pinned
        time.sleep(0.05)    # past a coarse timestamp tick, so the change time moves
        os.chmod(path, (os.stat(path).st_mode & 0o777) ^ 0o100)
    monkeypatch.setattr(fsdp_lora, "replicated_parameter_ids",
                        lambda _model: frozenset({id(model.blocks[0].lin.weight)}))
    with pytest.raises(UnbakeError, match="changed since this model was loaded from it"):
        fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0),
                              _Active(model, {_BIG_KEY: ["patch"]}), str(path), None, expected)
    assert torch.equal(model.blocks[0].lin.weight, weight)
    # The pin of the file as it is now admits the same bake.
    fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0), _Active(model, {_BIG_KEY: ["patch"]}),
                          str(path), None, _pinned_identity(path))
    assert torch.equal(model.blocks[0].lin.weight, weight * 2)


def test_the_fresh_bake_and_the_lazy_swap_hand_the_load_pin_to_the_bake(monkeypatch):
    """Both entries pass the pinned file's identity through to apply_stack."""
    from dgx_monarch.actor import model_store as ms

    seen = []
    monkeypatch.setattr(fsdp_lora, "apply_stack",
                        lambda store, active, base_path, previous, expected=None: (
                            seen.append((base_path, previous, expected))
                            or fsdp_lora.UnbakeRecord(path=base_path, file_size=1, file_mtime_ns=1)))
    monkeypatch.setattr(ms, "resolve_model_path", lambda kind, name: f"/models/{name}")
    pin = SimpleNamespace(file_identity="IDENTITY_OF_THE_PINNED_INODE")
    fsdp_lora.bake_stack(SimpleNamespace(), SimpleNamespace(), "/models/m.safetensors", pin)
    stored = SimpleNamespace(unbake="PREVIOUS", fsdp_checkpoint_pin=pin)
    fsdp_lora.lazy_swap(SimpleNamespace(), stored, SimpleNamespace(), "m.safetensors")
    assert seen == [("/models/m.safetensors", None, "IDENTITY_OF_THE_PINNED_INODE"),
                    ("/models/m.safetensors", "PREVIOUS", "IDENTITY_OF_THE_PINNED_INODE")]


def test_the_bake_drops_the_checkpoint_pages_it_read(tmp_path, monkeypatch):
    """Each patched key is read whole through the page cache; the bake gives those pages back."""
    import os

    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model = _Root()
    pristine = torch.ones(4, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(pristine)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {"blocks.0.lin.weight": pristine})
    advised: list[tuple] = []
    real_advise = os.posix_fadvise

    def advise(fd, offset, length, advice):
        advised.append((os.fstat(fd).st_ino, offset, length, advice))
        return real_advise(fd, offset, length, advice)

    monkeypatch.setattr(os, "posix_fadvise", advise)
    try:
        model = _one_rank_fsdp(tmp_path, model)
        fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0),
                              _Active(model, {"blocks.0.lin.weight": ["patch"]}), str(path), None)
        assert (os.stat(path).st_ino, 0, 0, os.POSIX_FADV_DONTNEED) in advised
    finally:
        torch.distributed.destroy_process_group()


def _rename_big(state_dict):
    """A config rename like ComfyUI's: the file's ``lin_file`` is the model's ``lin``."""
    return {key.replace(".lin_file.", ".lin."): value for key, value in state_dict.items()}


def test_two_candidates_that_agree_in_the_sampled_windows_bind_by_every_byte(tmp_path):
    """A file holds the key under its own name and under the name ComfyUI renames to it.

    The two tensors agree in every sampled window of this rank's rows and
    differ between them. The bake reads the bound tensor whole, so the first
    sampled match bound the wrong one; with more than one candidate the
    binding compares every byte of this rank's rows.
    """
    model = _BigRoot(_rename_big)
    weight = _big_weight()
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(weight)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {_BIG_KEY: _gap_twin(weight), "blocks.0.lin_file.weight": weight})
    replicated = frozenset({id(model.blocks[0].lin.weight)})
    record = fsdp_lora.capture_record(model, [_BIG_KEY], str(path), 0, 1, replicated)
    header = fsdp_lora.read_safetensors_header(str(path))
    assert record.mapped[_BIG_KEY] == header["blocks.0.lin_file.weight"]


def _rank_of_two(process, weight: torch.Tensor, rank: int) -> _BigRoot:
    """A model whose patched weight is rank ``rank``'s shard of two, holding ``weight``'s rows."""
    model = _BigRoot(process)
    param = torch.nn.Parameter(weight.clone(), requires_grad=False)
    param._local_tensor = weight[32 * rank:32 * (rank + 1)].clone()
    model.blocks[0].lin.weight = param
    return model


def test_two_ranks_whose_rows_match_different_names_bind_the_same_tensor(tmp_path):
    """The two tensors differ only in rank 1's rows of 2.

    Rank 0's rows equal both and rank 1's rows equal one. The binding used to
    follow each rank's rows, so rank 0 refused and dropped its model while
    rank 1 baked and went on to wait for it at the first collective. It now
    reads the file alone: ComfyUI's own load function puts the renamed
    tensor under the key, and both ranks bind it.
    """
    weight = _big_weight()
    decoy = weight.clone()
    decoy[40, 7] += 1                                   # a row rank 1 holds
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {_BIG_KEY: decoy, "blocks.0.lin_file.weight": weight})
    header = fsdp_lora.read_safetensors_header(str(path))
    bound = [fsdp_lora.capture_record(_rank_of_two(_rename_big, weight, rank), [_BIG_KEY],
                                      str(path), rank, 2).mapped[_BIG_KEY] for rank in (0, 1)]
    assert bound == [header["blocks.0.lin_file.weight"]] * 2


@pytest.mark.parametrize("spellings", ["differ", "agree"])
def test_two_spellings_of_a_key_end_alike_on_every_rank_before_any_write(tmp_path, monkeypatch, spellings):
    """With no rename, the file names the key twice; nothing in it says which one ComfyUI loaded.

    Where the two tensors differ (here only in rank 1's rows), every rank
    refuses with the same typed class P refusal before a weight is written.
    Where they agree in every byte, either serves and every rank binds.
    """
    from dgx_monarch.actor import unbake

    _install_comfy_stub(monkeypatch)
    weight = _big_weight()
    other = weight.clone()
    if spellings == "differ":
        other[40, 7] += 1                               # a row rank 1 holds
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {f"model.{_BIG_KEY}": other, _BIG_KEY: weight})
    writes: list[str] = []
    monkeypatch.setattr(fsdp_lora, "bake_key", lambda _active, key, *_a, **_k: writes.append(key))
    monkeypatch.setattr(fsdp_lora, "replicated_parameter_ids", lambda _model: frozenset())
    outcomes = []
    for rank in (0, 1):
        monkeypatch.setattr(fsdp_lora, "mesh_rank_world", lambda rank=rank: (rank, 2))
        active = _Active(_rank_of_two(None, weight, rank), {_BIG_KEY: ["patch"]})
        try:
            fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0), active, str(path), None)
            outcomes.append("bound")
        except unbake.UnbakeRefused as refused:
            outcomes.append(str(refused))
    if spellings == "differ":
        assert outcomes[0] == outcomes[1] and "[dgxm:P" in outcomes[0]
        assert "does not say which one ComfyUI loaded" in outcomes[0]
        assert writes == []
    else:
        assert outcomes == ["bound", "bound"] and writes == [_BIG_KEY, _BIG_KEY]


def test_a_rank_whose_rows_fail_on_one_key_still_meets_its_peers_refusal_on_the_next(tmp_path):
    """Every key binds from the file before any rank's rows are checked.

    Rank 0's rows of the first key do not match the file; the second key sits
    under two spellings that differ. Checking each key's rows as it bound
    would end rank 0 on "not pristine" at the first key while rank 1 went on
    to the second key's refusal. Both ranks must raise the same refusal.
    """
    from dgx_monarch.actor import unbake

    norm_key = "blocks.0.norm.weight"
    weight = _big_weight()
    norm = torch.linspace(0.5, 1.5, 64).to(torch.bfloat16)
    other = norm.clone()
    other[40] += 1
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {_BIG_KEY: weight, norm_key: norm, f"model.{norm_key}": other})
    outcomes = []
    for rank in (0, 1):
        model = _rank_of_two(None, weight, rank)
        if rank == 0:
            model.blocks[0].lin.weight._local_tensor.add_(1)   # rank 0's rows left the file
        param = torch.nn.Parameter(norm.clone(), requires_grad=False)
        param._local_tensor = norm[32 * rank:32 * (rank + 1)].clone()
        model.blocks[0].norm = torch.nn.Module()
        model.blocks[0].norm.weight = param
        with pytest.raises(UnbakeError) as raised:
            fsdp_lora.capture_record(model, [_BIG_KEY, norm_key], str(path), rank, 2)
        outcomes.append((type(raised.value), str(raised.value)))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] is unbake.UnbakeRefused and "does not say which one" in outcomes[0][1]


def test_a_quantized_setter_refuses_before_a_leaving_key_is_restored(tmp_path, monkeypatch):
    """Every refusal the stack can meet runs before the first write.

    The setter check ran inside the bake loop, after the keys leaving the stack
    were restored, so a refusal left a half-changed model behind and the store
    had to drop it. Now it refuses first, typed, and the resident keeps the
    stack it had.
    """
    from dgx_monarch.actor import unbake

    _install_comfy_stub(monkeypatch)
    model = _BigRoot()
    weight = _big_weight()
    norm = torch.nn.Parameter(torch.ones(64, dtype=torch.bfloat16), requires_grad=False)
    model.blocks[0].norm = torch.nn.Module()
    model.blocks[0].norm.weight = norm
    with torch.no_grad():
        model.blocks[0].lin.weight.copy_(weight)
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {_BIG_KEY: weight, "blocks.0.norm.weight": torch.ones(64, dtype=torch.bfloat16)})
    replicated = frozenset({id(model.blocks[0].lin.weight), id(norm)})
    monkeypatch.setattr(fsdp_lora, "replicated_parameter_ids", lambda _model: replicated)
    previous = fsdp_lora.capture_record(model, [_BIG_KEY], str(path), 0, 1, replicated)
    with torch.no_grad():
        model.blocks[0].lin.weight.mul_(2)          # the previous stack's bake
    baked = model.blocks[0].lin.weight.detach().clone()
    stub = sys.modules["comfy.model_patcher"]
    real_get = stub.get_key_weight
    monkeypatch.setattr(stub, "get_key_weight", lambda m, key: (
        (real_get(m, key)[0], lambda *a, **k: None, None) if key == "blocks.0.norm.weight"
        else real_get(m, key)))
    with pytest.raises(UnbakeError, match="LoRA on quantized shards") as raised:
        fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0),
                              _Active(model, {"blocks.0.norm.weight": ["patch"]}), str(path), previous)
    assert isinstance(raised.value, getattr(unbake, "UnbakeRefused", ()))
    assert torch.equal(model.blocks[0].lin.weight, baked)   # the leaving key was not restored
