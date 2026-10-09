"""Tests for the actor/unbake.py file-backed lazy LoRA backup, run without comfy.

Capture maps patched keys to checkpoint slices only when sampled bytes verify
(anything else stays resident or aborts the record), restore is bit-exact from
a perturbed ("baked") state for plain and quantized weights (comfy stubbed for
the wrapper rebuild), and every guard that must force the full-reload fallback
raises."""
import json
import os
import struct
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dgx_monarch.actor import unbake as ub
from dgx_monarch.actor.unbake import (
    CaptureAborted,
    UnbakeError,
    capture_unbake_record,
    read_safetensors_header,
    restore_pristine,
)

_TAG = {torch.bfloat16: "BF16", torch.float32: "F32",
        torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8",
        torch.int8: "I8"}


def _to_bytes(t: torch.Tensor) -> bytes:
    return t.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()


def _quant_marker(conf: object) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(conf).encode("utf-8")),
        dtype=torch.uint8,
    )


def _raw_quant_marker(payload: bytes) -> torch.Tensor:
    return torch.tensor(list(payload), dtype=torch.uint8)


def write_safetensors(path, tensors: dict[str, torch.Tensor], metadata: dict | None = None):
    """Minimal writer for the public safetensors layout (u64 len + JSON header
    + raw tensor bytes), so the tests carry no safetensors dependency."""
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        b = _to_bytes(t)
        header[name] = {"dtype": _TAG[t.dtype], "shape": list(t.shape),
                        "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        header["__metadata__"] = metadata
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


class _Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)


class _TinyModel(nn.Module):
    """Mimics the patcher.model shape: weights under diffusion_model.*, with a
    ModuleList so digit path segments ("blocks.0") are exercised."""

    def __init__(self, dim=8):
        super().__init__()
        dit = nn.Module()
        dit.blocks = nn.ModuleList([_Block(dim), _Block(dim)])
        dit.scale = nn.Parameter(torch.randn(dim, dtype=torch.float32))
        dit.register_buffer("pos", torch.randn(dim, dtype=torch.float32))
        self.diffusion_model = dit


KEYS = ["diffusion_model.blocks.0.wq.weight", "diffusion_model.blocks.1.wq.weight",
        "diffusion_model.scale", "diffusion_model.pos"]


@pytest.fixture
def rig(tmp_path):
    torch.manual_seed(7)
    model = _TinyModel()
    for blk in model.diffusion_model.blocks:
        blk.wq.weight.data = torch.randn(8, 8).to(torch.bfloat16)
    file_tensors = {
        "blocks.0.wq.weight": model.diffusion_model.blocks[0].wq.weight.detach().clone(),
        "blocks.1.wq.weight": model.diffusion_model.blocks[1].wq.weight.detach().clone(),
        "scale": model.diffusion_model.scale.detach().clone(),
        "pos": model.diffusion_model.pos.detach().clone(),
    }
    path = str(tmp_path / "base.safetensors")
    write_safetensors(path, file_tensors)
    return model, path, file_tensors


def _bake(model):
    """Perturb every weight in place, like a lora bake would."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))
        model.diffusion_model.pos.add_(1.0)


def test_header_reader_absolute_offsets(rig):
    _, path, file_tensors = rig
    header = read_safetensors_header(path)
    assert set(header) == set(file_tensors)
    ft = header["blocks.0.wq.weight"]
    assert ft.dtype == torch.bfloat16
    with open(path, "rb") as f:
        raw = os.pread(f.fileno(), ft.nbytes, ft.start)
    assert raw == _to_bytes(file_tensors["blocks.0.wq.weight"])


def test_header_reader_supports_packed_f4_and_e8m0_scale(tmp_path):
    f4 = getattr(torch, "float4_e2m1fn_x2", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if f4 is None or e8m0 is None:
        pytest.skip("torch build predates packed F4/E8M0 dtypes")
    path = tmp_path / "packed.safetensors"
    header = {
        "weight": {"dtype": "F4", "shape": [2, 8], "data_offsets": [0, 8]},
        "weight_scale": {
            "dtype": "F8_E8M0", "shape": [2], "data_offsets": [8, 10],
        },
    }
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 10)

    parsed = read_safetensors_header(str(path))

    assert parsed["weight"].dtype == f4
    assert parsed["weight"].shape == (2, 4)
    assert parsed["weight"].nbytes == 8
    assert parsed["weight_scale"].dtype == e8m0


def test_capture_maps_everything_and_restore_is_bit_exact(rig):
    model, path, file_tensors = rig
    record = capture_unbake_record(model, KEYS, path)
    assert set(record.mapped) == set(KEYS)
    assert not record.resident and not record.quant
    assert record.mapped_bytes > 0

    _bake(model)
    assert not torch.equal(model.diffusion_model.blocks[0].wq.weight,
                           file_tensors["blocks.0.wq.weight"])
    info = restore_pristine(model, record)
    assert info["restored_keys"] == len(KEYS)
    assert torch.equal(model.diffusion_model.blocks[0].wq.weight,
                       file_tensors["blocks.0.wq.weight"])
    assert torch.equal(model.diffusion_model.blocks[1].wq.weight,
                       file_tensors["blocks.1.wq.weight"])
    assert torch.equal(model.diffusion_model.scale, file_tensors["scale"])
    assert torch.equal(model.diffusion_model.pos, file_tensors["pos"])


def test_capture_rejects_replacement_after_header_validation(
    rig, tmp_path, monkeypatch
):
    model, path, file_tensors = rig
    replacement = str(tmp_path / "replacement.safetensors")
    write_safetensors(replacement, file_tensors)
    real_open = open
    swapped = False

    def replace_before_component_read(name, *args, **kwargs):
        nonlocal swapped
        if not swapped and os.fspath(name) == path:
            swapped = True
            os.replace(replacement, path)
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(ub, "open", replace_before_component_read, raising=False)
    with pytest.raises(CaptureAborted, match="header was validated"):
        capture_unbake_record(model, KEYS, path)


def test_capture_rejects_metadata_change_during_component_reads(rig, monkeypatch):
    model, path, _file_tensors = rig
    real_match = ub._bytes_match
    changed = False

    def match_then_touch(fd, ft, tensor):
        nonlocal changed
        result = real_match(fd, ft, tensor)
        if not changed:
            changed = True
            st = os.stat(path)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        return result

    monkeypatch.setattr(ub, "_bytes_match", match_then_touch)
    with pytest.raises(CaptureAborted, match="while capturing"):
        capture_unbake_record(model, KEYS, path)


def test_capture_accepts_ctime_only_change_during_component_reads(rig, monkeypatch):
    model, path, _file_tensors = rig
    real_match = ub._bytes_match
    changed = False

    def match_then_chmod(fd, ft, tensor):
        nonlocal changed
        result = real_match(fd, ft, tensor)
        if not changed:
            changed = True
            before = os.stat(path)
            os.chmod(path, (before.st_mode & 0o777) ^ 0o100)
            after = os.stat(path)
            assert (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            assert before.st_ctime_ns != after.st_ctime_ns
        return result

    monkeypatch.setattr(ub, "_bytes_match", match_then_chmod)
    record = capture_unbake_record(model, KEYS, path)

    assert changed
    assert set(record.mapped) == set(KEYS)


def test_capture_rejects_atomic_replacement_during_component_reads(
    rig, tmp_path, monkeypatch
):
    model, path, file_tensors = rig
    captured = os.stat(path)
    replacement = str(tmp_path / "replacement-during-capture.safetensors")
    write_safetensors(replacement, file_tensors)
    os.utime(replacement, ns=(captured.st_atime_ns, captured.st_mtime_ns))
    replacement_stat = os.stat(replacement)
    assert replacement_stat.st_dev == captured.st_dev
    assert replacement_stat.st_ino != captured.st_ino
    assert replacement_stat.st_size == captured.st_size
    assert replacement_stat.st_mtime_ns == captured.st_mtime_ns
    real_match = ub._bytes_match
    swapped = False

    def match_then_replace(fd, ft, tensor):
        nonlocal swapped
        result = real_match(fd, ft, tensor)
        if not swapped:
            swapped = True
            os.replace(replacement, path)
        return result

    monkeypatch.setattr(ub, "_bytes_match", match_then_replace)
    with pytest.raises(CaptureAborted, match="while capturing"):
        capture_unbake_record(model, KEYS, path)


def test_transformed_key_stays_resident_and_still_restores(rig):
    model, path, file_tensors = rig
    # Simulate a loader transform: the live blocks.1 differs from the file.
    with torch.no_grad():
        model.diffusion_model.blocks[1].wq.weight.data[0, 0] += 1
    transformed = model.diffusion_model.blocks[1].wq.weight.detach().clone()

    record = capture_unbake_record(model, KEYS, path)
    assert "diffusion_model.blocks.1.wq.weight" in record.resident
    assert "diffusion_model.blocks.0.wq.weight" in record.mapped
    # Resident snapshots are clones, so baking over the live tensor cannot
    # corrupt them.
    _bake(model)
    restore_pristine(model, record)
    assert torch.equal(model.diffusion_model.blocks[1].wq.weight, transformed)
    assert torch.equal(model.diffusion_model.blocks[0].wq.weight,
                       file_tensors["blocks.0.wq.weight"])


def test_key_missing_from_file_stays_resident(rig):
    model, path, _ = rig
    extra = torch.randn(4, dtype=torch.float32)
    model.diffusion_model.extra = nn.Parameter(extra.clone())
    record = capture_unbake_record(model, [*KEYS, "diffusion_model.extra"], path)
    assert "diffusion_model.extra" in record.resident
    _bake(model)
    with torch.no_grad():
        model.diffusion_model.extra.add_(5.0)
    restore_pristine(model, record)
    assert torch.equal(model.diffusion_model.extra, extra)


def test_resident_over_cap_aborts(rig, monkeypatch):
    model, path, _ = rig
    monkeypatch.setattr(ub, "_RESIDENT_CAP_BYTES", 4)
    with torch.no_grad():
        model.diffusion_model.blocks[1].wq.weight.data[0, 0] += 1
    with pytest.raises(CaptureAborted):
        capture_unbake_record(model, KEYS, path)


def test_cast_mapped_key_restores_exactly(tmp_path):
    """Checkpoint stores F32, model runs bf16 (Krea2 keeps 174 tensors F32):
    the mapping records the loader's cast and restore replays it bit-exactly."""
    f32 = torch.randn(16, 16, dtype=torch.float32)
    path = str(tmp_path / "mixed.safetensors")
    write_safetensors(path, {"proj.weight": f32})
    live = nn.Parameter(f32.to(torch.bfloat16))          # what the loader made
    model = SimpleNamespace(diffusion_model=SimpleNamespace(
        proj=SimpleNamespace(weight=live)))
    record = capture_unbake_record(model, ["diffusion_model.proj.weight"], path)
    assert "diffusion_model.proj.weight" in record.mapped
    ft = record.mapped["diffusion_model.proj.weight"]
    assert ft.cast_to == torch.bfloat16 and ft.dtype == torch.float32

    pristine = live.detach().clone()
    with torch.no_grad():
        live.add_(1.0)                                    # the bake
    restore_pristine(model, record)
    assert torch.equal(model.diffusion_model.proj.weight, pristine)


def test_cast_mismatch_stays_resident(tmp_path):
    f32 = torch.randn(16, 16, dtype=torch.float32)
    path = str(tmp_path / "mixed2.safetensors")
    write_safetensors(path, {"proj.weight": f32})
    perturbed = f32.to(torch.bfloat16)
    perturbed[0, 0] += 1                                  # loader did more than cast
    model = SimpleNamespace(diffusion_model=SimpleNamespace(
        proj=SimpleNamespace(weight=nn.Parameter(perturbed))))
    record = capture_unbake_record(model, ["diffusion_model.proj.weight"], path)
    assert "diffusion_model.proj.weight" in record.resident


def test_model_prefix_candidate(tmp_path):
    t = torch.randn(4, 4, dtype=torch.float32)
    path = str(tmp_path / "prefixed.safetensors")
    write_safetensors(path, {"model.diffusion_model.deep.weight": t})
    model = SimpleNamespace(
        diffusion_model=SimpleNamespace(deep=SimpleNamespace(weight=t.clone())))
    record = capture_unbake_record(model, ["diffusion_model.deep.weight"], path)
    assert "diffusion_model.deep.weight" in record.mapped


def test_stale_file_raises(rig):
    model, path, _ = rig
    record = capture_unbake_record(model, KEYS, path)
    os.utime(path, ns=(record.file_mtime_ns + 10, record.file_mtime_ns + 10))
    with pytest.raises(UnbakeError):
        restore_pristine(model, record)


def test_open_race_surfaces_reader_error_instead_of_hanging(rig, monkeypatch):
    model, path, _ = rig
    record = capture_unbake_record(model, KEYS, path)

    def vanished(_path):
        raise FileNotFoundError("checkpoint disappeared after stat")

    monkeypatch.setattr(ub, "_open_for_restore", vanished)
    with pytest.raises(UnbakeError, match="checkpoint read failed"):
        restore_pristine(model, record)


def test_mutation_during_restore_reads_is_rejected(rig, monkeypatch):
    model, path, _ = rig
    record = capture_unbake_record(model, KEYS, path)
    _bake(model)
    real_pread = ub._pread_into_at
    mutated = False

    # Force the buffered path so the test can interpose exactly after the
    # first successful read, while the reader still holds its verified fd.
    monkeypatch.setattr(
        ub, "_open_for_restore", lambda value: (os.open(value, os.O_RDONLY), False))

    def pread_then_mutate(fd, offset, nbytes, scratch, dst_off):
        nonlocal mutated
        real_pread(fd, offset, nbytes, scratch, dst_off)
        if not mutated:
            mutated = True
            with open(path, "r+b", buffering=0) as stream:
                stream.seek(-1, os.SEEK_END)
                value = stream.read(1)
                stream.seek(-1, os.SEEK_END)
                stream.write(bytes([value[0] ^ 0xFF]))

    monkeypatch.setattr(ub, "_pread_into_at", pread_then_mutate)
    with pytest.raises(UnbakeError, match="restore reads were in progress"):
        restore_pristine(model, record)
    assert mutated


def test_live_model_drift_raises(rig):
    model, path, _ = rig
    record = capture_unbake_record(model, KEYS, path)
    model.diffusion_model.scale = nn.Parameter(torch.randn(3, dtype=torch.float32))
    with pytest.raises(UnbakeError):
        restore_pristine(model, record)


class _FakeQuantTensor:
    """Duck-typed stand-in for comfy_kitchen's QuantizedTensor: _is_quantized
    only checks for layout_cls + params."""

    def __init__(self):
        self.layout_cls = "TensorCoreFP8E4M3Layout"
        self.params = SimpleNamespace()
        self.device = torch.device("cpu")


class _FakeQuantLinear:
    """A plain object, because nn.Module.__setattr__ sends the Parameter the
    restore assigns to register_parameter, which raises KeyError on the weight
    property instead of calling its setter; unbake needs only attribute access
    and state_dict(), not nn.Module."""

    def __init__(self, qdata, scale):
        self.quant_weight = _FakeQuantTensor()
        self.qdata = qdata
        self.scale = scale
        self.quant_format = "float8_e4m3fn"
        self._full_precision_mm_config = False
        self.factory_kwargs = {"device": None, "dtype": torch.bfloat16}
        self._orig_shape = tuple(qdata.shape)
        self.rebuilt = None

    @property
    def weight(self):
        return self.rebuilt if self.rebuilt is not None else self.quant_weight

    @weight.setter
    def weight(self, value):
        self.rebuilt = value  # restore assigns here

    def state_dict(self, *a, **k):
        conf = {"format": self.quant_format}
        if self._full_precision_mm_config:
            conf["full_precision_matrix_mult"] = True
        return {"weight": self.qdata, "weight_scale": self.scale,
                "comfy_quant": torch.tensor(
                    list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)}


@pytest.fixture
def quant_rig(tmp_path, monkeypatch):
    qdata = (torch.randn(8, 8) * 0.1).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.0123, dtype=torch.float32)
    meta = {"_quantization_metadata": json.dumps(
        {"layers": {"blk": {"format": "float8_e4m3fn"}}})}
    path = str(tmp_path / "quant.safetensors")
    write_safetensors(path, {"blk.weight": qdata.clone(),
                             "blk.weight_scale": scale.clone()}, metadata=meta)

    module = _FakeQuantLinear(qdata.clone(), scale.clone())
    model = SimpleNamespace(diffusion_model=SimpleNamespace(blk=module))

    # comfy.quant_ops stub for the restore-side wrapper rebuild
    recorded = {}

    def fake_qt(qdata, layout_type, params):
        recorded["layout_type"] = layout_type
        recorded["params"] = params
        return qdata  # a real tensor, so nn.Parameter accepts it

    layout = SimpleNamespace(Params=lambda **kw: SimpleNamespace(**kw))
    quant_ops = types.ModuleType("comfy.quant_ops")
    quant_ops.QUANT_ALGOS = {
        "float8_e4m3fn": {
            "storage_t": torch.float8_e4m3fn,
            "comfy_tensor_layout": "TensorCoreFP8E4M3Layout"},
        "int8_tensorwise": {
            "storage_t": torch.int8,
            "comfy_tensor_layout": "TensorWiseINT8Layout"}}
    quant_ops.get_layout_class = lambda name: layout
    quant_ops.QuantizedTensor = fake_qt
    comfy = types.ModuleType("comfy")
    comfy.quant_ops = quant_ops
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.quant_ops", quant_ops)
    return model, module, path, qdata, scale, recorded


def test_quant_capture_and_rebuild(quant_rig):
    model, module, path, qdata, scale, recorded = quant_rig
    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    spec = record.quant["diffusion_model.blk.weight"]
    assert set(spec.components) == {"weight", "weight_scale"}
    assert not spec.resident and record.resident_count == 0

    # bake: requant replaced data and scale
    module.qdata = (torch.randn(8, 8) * 0.2).to(torch.float8_e4m3fn)
    module.scale = torch.tensor(0.5, dtype=torch.float32)

    restore_pristine(model, record)
    rebuilt = module.rebuilt
    assert isinstance(rebuilt, torch.nn.Parameter)
    assert torch.equal(rebuilt.data.view(torch.uint8), qdata.view(torch.uint8))
    assert recorded["layout_type"] == "TensorCoreFP8E4M3Layout"
    assert torch.equal(recorded["params"].scale, scale)
    assert recorded["params"].orig_shape == (8, 8)


def test_per_layer_quant_marker_capture_and_rebuild(quant_rig, tmp_path):
    model, module, _path, qdata, scale, recorded = quant_rig
    conf = {"format": "float8_e4m3fn"}
    path = str(tmp_path / "per-layer-quant.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(conf),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    spec = record.quant["diffusion_model.blk.weight"]
    assert spec.layer_conf == conf
    assert set(spec.components) == {"weight", "weight_scale"}
    assert not spec.resident and record.resident_count == 0

    module.qdata = (torch.randn(8, 8) * 0.2).to(torch.float8_e4m3fn)
    module.scale = torch.tensor(0.5, dtype=torch.float32)
    restore_pristine(model, record)

    assert torch.equal(module.rebuilt.data.view(torch.uint8), qdata.view(torch.uint8))
    assert torch.equal(recorded["params"].scale, scale)


def test_per_layer_quant_marker_binds_supported_namespace(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "per-layer-prefixed.safetensors")
    write_safetensors(
        path,
        {
            "model.diffusion_model.blk.weight": qdata.clone(),
            "model.diffusion_model.blk.weight_scale": scale.clone(),
            "model.diffusion_model.blk.comfy_quant": _quant_marker(
                {"format": "float8_e4m3fn"}
            ),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)

    assert set(record.quant["diffusion_model.blk.weight"].components) == {
        "weight",
        "weight_scale",
    }


@pytest.mark.parametrize(
    "marker",
    [
        torch.tensor([], dtype=torch.uint8),
        _raw_quant_marker(b"not-json"),
        torch.ones(8, dtype=torch.int8),
        torch.ones((2, 2), dtype=torch.uint8),
        torch.ones(4097, dtype=torch.uint8),
        _quant_marker(["float8_e4m3fn"]),
        _raw_quant_marker(
            b'{"format":"float8_e4m3fn","format":"float8_e5m2"}'
        ),
        _raw_quant_marker(b'{"format":NaN}'),
        _raw_quant_marker(b'{"format":Infinity}'),
        _raw_quant_marker(b'{"format":"float8_e4m3fn"} trailing'),
        _raw_quant_marker(b'{"format":"\xff"}'),
    ],
)
def test_per_layer_quant_marker_rejects_malformed_payload(
    quant_rig, tmp_path, marker
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "per-layer-malformed.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": marker,
        },
    )

    with pytest.raises(
        CaptureAborted,
        match="malformed checkpoint quantization authority",
    ):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_per_layer_quant_marker_rejects_live_contract_mismatch(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "per-layer-live-mismatch.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker({"format": "float8_e5m2"}),
        },
    )

    with pytest.raises(CaptureAborted, match="live quantization contract"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_per_layer_quant_marker_accepts_direct_int8_contract(
    quant_rig, tmp_path
):
    """int8 has an exact restore recipe, so capture admits its direct marker
    too. A gate that admits only FP8 makes every LoRA stack change on a
    direct-marker int8 checkpoint a full reload (2026-08-06,
    docs/TROUBLESHOOTING.md #41)."""
    model, module, _path, qdata, scale, _recorded = quant_rig
    conf = {"format": "int8_tensorwise"}
    module.quant_format = "int8_tensorwise"
    module.state_dict = lambda *args, **kwargs: {
        "weight": module.qdata,
        "weight_scale": module.scale,
        "comfy_quant": _quant_marker(conf),
    }
    path = str(tmp_path / "per-layer-int8.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(conf),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)

    spec = record.quant["diffusion_model.blk.weight"]
    assert spec.layer_conf == conf
    assert set(spec.components) == {"weight", "weight_scale"}
    assert not spec.resident and record.resident_count == 0


def test_embedded_quant_metadata_precedes_per_layer_marker(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    conf = {"format": "float8_e4m3fn"}
    path = str(tmp_path / "metadata-precedes-per-layer.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _raw_quant_marker(b"malformed lower authority"),
        },
        metadata={
            "_quantization_metadata": json.dumps({"layers": {"blk": conf}})
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)

    assert record.quant["diffusion_model.blk.weight"].layer_conf == conf


def test_embedded_quant_metadata_accepts_comfy_canonical_live_marker(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    conf = {"format": "float8_e4m3fn", "params": {}}
    path = str(tmp_path / "metadata-canonical-live.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
        },
        metadata={
            "_quantization_metadata": json.dumps({"layers": {"blk": conf}})
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)

    assert record.quant["diffusion_model.blk.weight"].layer_conf == conf


def test_invalid_metadata_and_live_configs_cannot_match(quant_rig, tmp_path):
    model, module, _path, qdata, scale, _recorded = quant_rig
    conf = {"format": "int8_tensorwise", "params": "invalid"}
    module.quant_format = "int8_tensorwise"
    module.state_dict = lambda *args, **kwargs: {
        "weight": module.qdata,
        "weight_scale": module.scale,
        "comfy_quant": _quant_marker(conf),
    }
    path = str(tmp_path / "metadata-invalid-live.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
        },
        metadata={
            "_quantization_metadata": json.dumps({"layers": {"blk": conf}})
        },
    )

    with pytest.raises(CaptureAborted, match="live quantization contract"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_precedes_per_layer_marker(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-precedes-per-layer.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
            "blk.comfy_quant": _raw_quant_marker(b"malformed lower authority"),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)

    assert record.quant["diffusion_model.blk.weight"].layer_conf == {
        "format": "float8_e4m3fn"
    }


def test_embedded_metadata_omission_uses_direct_marker_not_legacy(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    conf = {"format": "float8_e4m3fn"}
    path = str(tmp_path / "metadata-omits-direct-layer.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(conf),
        },
        metadata={
            "_quantization_metadata": json.dumps(
                {"layers": {"other": {"format": "float8_e5m2"}}}
            )
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)

    spec = record.quant["diffusion_model.blk.weight"]
    assert spec.layer_conf == conf
    assert set(spec.components) == {"weight", "weight_scale"}


def test_direct_marker_accepts_semantic_key_order_and_pins_full_precision(
    quant_rig, tmp_path
):
    model, module, _path, qdata, scale, _recorded = quant_rig
    module._full_precision_mm_config = True
    path = str(tmp_path / "direct-full-precision.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _raw_quant_marker(
                b'{"full_precision_matrix_mult":true,'
                b'"format":"float8_e4m3fn"}'
            ),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    spec = record.quant["diffusion_model.blk.weight"]
    assert spec.full_precision_mm_config is True

    module._full_precision_mm_config = False
    with pytest.raises(UnbakeError, match="full-precision matmul config"):
        restore_pristine(model, record)


def test_direct_marker_pins_absent_full_precision_to_false(
    quant_rig, tmp_path
):
    model, module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "direct-default-full-precision.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(
                {"format": "float8_e4m3fn"}
            ),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    spec = record.quant["diffusion_model.blk.weight"]
    assert spec.full_precision_mm_config is False

    module._full_precision_mm_config = True
    with pytest.raises(UnbakeError, match="full-precision matmul config"):
        restore_pristine(model, record)


def test_direct_marker_accepts_explicit_false_full_precision(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    conf = {
        "format": "float8_e4m3fn",
        "full_precision_matrix_mult": False,
    }
    path = str(tmp_path / "direct-explicit-false.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(conf),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)

    assert record.quant["diffusion_model.blk.weight"].layer_conf == conf
    assert record.quant[
        "diffusion_model.blk.weight"
    ].full_precision_mm_config is False


def test_direct_marker_rejects_unreproducible_contract(quant_rig, tmp_path):
    model, module, _path, qdata, scale, _recorded = quant_rig
    module.state_dict = lambda *args, **kwargs: {
        "weight": module.qdata,
        "weight_scale": module.scale,
        "comfy_quant": _quant_marker(
            {"format": "float8_e4m3fn", "num_experts": 2}
        ),
    }
    path = str(tmp_path / "direct-unsupported-contract.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(
                {"format": "float8_e4m3fn", "num_experts": 2}
            ),
        },
    )

    with pytest.raises(CaptureAborted, match="cannot be reproduced"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_direct_marker_rejects_unreproducible_live_contract(
    quant_rig, tmp_path
):
    model, module, _path, qdata, scale, _recorded = quant_rig
    module.state_dict = lambda *args, **kwargs: {
        "weight": module.qdata,
        "weight_scale": module.scale,
        "comfy_quant": _quant_marker(
            {"format": "float8_e4m3fn", "num_experts": 2}
        ),
    }
    path = str(tmp_path / "direct-unsupported-live-contract.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(
                {"format": "float8_e4m3fn"}
            ),
        },
    )

    with pytest.raises(CaptureAborted, match="cannot be reproduced"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_direct_marker_rejects_unsupported_owning_module(quant_rig, tmp_path):
    model, module, _path, qdata, scale, _recorded = quant_rig
    del module.factory_kwargs  # mirrors Comfy's distinct quantized Embedding
    path = str(tmp_path / "direct-unsupported-module.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(
                {"format": "float8_e4m3fn"}
            ),
        },
    )

    with pytest.raises(CaptureAborted, match="cannot be reproduced"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_direct_marker_rejects_unexpected_weight_scale(quant_rig, tmp_path):
    model, module, _path, qdata, scale, _recorded = quant_rig
    module.state_dict = lambda *args, **kwargs: {
        "weight": module.qdata,
        "weight_scale": module.scale,
        "weight_scale_2": module.scale,
        "comfy_quant": _quant_marker(
            {"format": "float8_e4m3fn"}
        ),
    }
    path = str(tmp_path / "direct-unexpected-scale.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker(
                {"format": "float8_e4m3fn"}
            ),
        },
    )

    with pytest.raises(CaptureAborted, match="cannot be reproduced"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_direct_marker_rejects_ambiguous_namespace_bindings(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    marker = _quant_marker({"format": "float8_e4m3fn"})
    path = str(tmp_path / "direct-ambiguous-bindings.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": marker.clone(),
            "model.diffusion_model.blk.weight": qdata.clone(),
            "model.diffusion_model.blk.weight_scale": scale.clone(),
            "model.diffusion_model.blk.comfy_quant": marker.clone(),
        },
    )

    with pytest.raises(CaptureAborted, match="2 coherent checkpoint bindings"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_malformed_second_binding_cannot_promote_first_binding(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "direct-invalid-second-binding.safetensors")
    write_safetensors(
        path,
        {
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
            "blk.comfy_quant": _quant_marker({"format": "float8_e4m3fn"}),
            "model.diffusion_model.blk.weight": qdata.clone(),
            "model.diffusion_model.blk.weight_scale": scale.clone(),
            "model.diffusion_model.blk.comfy_quant": _raw_quant_marker(
                b"malformed second authority"
            ),
        },
    )

    with pytest.raises(
        CaptureAborted,
        match="malformed checkpoint quantization authority",
    ):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


# The contract the pruned MiniMax H3 repacks carry (recorded 2026-08-06),
# spelled as their headers spell it: no `__metadata__` at all, so the per-layer
# marker is the only quantization authority, int8 data under a per-row f32
# scale (an fp8 layer's scale is a scalar), on a fused qkv projection. Krea2
# carries the same recipe in embedded metadata.
_H3_DIRECT_CONF = {
    "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
_H3_KEY = "diffusion_model.blocks.0.attn.qkv_proj.weight"


class _FakeInt8ConvRotLinear(_FakeQuantLinear):
    """A fused qkv projection quantized int8_tensorwise with convrot."""

    def __init__(self, qdata, scale, conf):
        super().__init__(qdata, scale)
        self.conf = conf
        self.quant_format = conf["format"]

    def state_dict(self, *a, **k):
        return {"weight": self.qdata, "weight_scale": self.scale,
                "bias": torch.zeros(self.qdata.shape[0], dtype=torch.bfloat16),
                "comfy_quant": _quant_marker(self.conf)}


def _write_h3_layer(path, qdata, scale, marker=None, metadata=None):
    tensors = {"blocks.0.attn.qkv_proj.weight": qdata.clone(),
               "blocks.0.attn.qkv_proj.weight_scale": scale.clone()}
    if marker is not None:
        tensors["blocks.0.attn.qkv_proj.comfy_quant"] = _quant_marker(marker)
    write_safetensors(path, tensors, metadata=metadata)


@pytest.fixture
def h3_direct_rig(quant_rig):
    """The pruned-H3 layer on top of quant_rig's comfy stub."""
    *_fp8, recorded = quant_rig
    qdata = torch.randint(-127, 128, (16, 4), dtype=torch.int8)
    scale = torch.rand(16, 1, dtype=torch.float32) + 0.5
    module = _FakeInt8ConvRotLinear(qdata.clone(), scale.clone(), _H3_DIRECT_CONF)
    blocks = SimpleNamespace()
    setattr(blocks, "0", SimpleNamespace(attn=SimpleNamespace(qkv_proj=module)))
    model = SimpleNamespace(diffusion_model=SimpleNamespace(blocks=blocks))
    return model, module, qdata, scale, recorded


def test_direct_int8_convrot_marker_captures_and_restores_exactly(
    h3_direct_rig, tmp_path
):
    model, module, qdata, scale, recorded = h3_direct_rig
    path = str(tmp_path / "pruned-int8-convrot.safetensors")
    _write_h3_layer(path, qdata, scale, marker=_H3_DIRECT_CONF)

    record = capture_unbake_record(model, [_H3_KEY], path)
    spec = record.quant[_H3_KEY]
    assert spec.layer_conf == _H3_DIRECT_CONF
    assert set(spec.components) == {"weight", "weight_scale"}
    assert not spec.resident and record.resident_count == 0

    # bake: comfy requantizes the data and every per-row scale
    module.qdata = torch.randint(-127, 128, (16, 4), dtype=torch.int8)
    module.scale = torch.rand(16, 1, dtype=torch.float32) + 0.5
    restore_pristine(model, record)

    assert torch.equal(
        module.rebuilt.data.view(torch.uint8), qdata.view(torch.uint8))
    assert torch.equal(recorded["params"].scale, scale)
    assert recorded["layout_type"] == "TensorWiseINT8Layout"
    assert recorded["params"].convrot is True
    assert recorded["params"].convrot_groupsize == 256
    assert recorded["params"].orig_shape == (16, 4)


def test_direct_int8_marker_accepts_comfy_nested_params_spelling(
    h3_direct_rig, tmp_path
):
    """Comfy reads convrot flat or nested, and re-emits it flat, so the two
    spellings have to agree instead of reading as a live-contract mismatch."""
    model, _module, qdata, scale, _recorded = h3_direct_rig
    conf = {"format": "int8_tensorwise",
            "params": {"convrot": True, "convrot_groupsize": 256}}
    path = str(tmp_path / "direct-int8-nested-params.safetensors")
    _write_h3_layer(path, qdata, scale, marker=conf)

    record = capture_unbake_record(model, [_H3_KEY], path)

    assert record.quant[_H3_KEY].layer_conf == conf


def test_embedded_metadata_int8_convrot_contract_still_captures(
    h3_direct_rig, tmp_path
):
    """The krea2 spelling of the same recipe: embedded metadata, no marker."""
    model, _module, qdata, scale, _recorded = h3_direct_rig
    path = str(tmp_path / "metadata-int8-convrot.safetensors")
    _write_h3_layer(path, qdata, scale, metadata={
        "_quantization_metadata": json.dumps(
            {"layers": {"blocks.0.attn.qkv_proj": _H3_DIRECT_CONF}})})

    record = capture_unbake_record(model, [_H3_KEY], path)

    spec = record.quant[_H3_KEY]
    assert spec.layer_conf == _H3_DIRECT_CONF
    assert set(spec.components) == {"weight", "weight_scale"}


def test_direct_int8_marker_names_the_key_it_cannot_replay(
    h3_direct_rig, tmp_path
):
    """A marker naming `per_row`, which no restore reads, still aborts capture,
    and the message names the key that caused it."""
    model, module, qdata, scale, _recorded = h3_direct_rig
    conf = {**_H3_DIRECT_CONF, "per_row": True}
    module.conf = conf
    path = str(tmp_path / "direct-int8-per-row.safetensors")
    _write_h3_layer(path, qdata, scale, marker=conf)

    with pytest.raises(CaptureAborted, match=r"cannot be reproduced.*'per_row'"):
        capture_unbake_record(model, [_H3_KEY], path)


def test_direct_marker_names_a_format_with_no_direct_recipe(
    h3_direct_rig, tmp_path
):
    model, module, qdata, scale, _recorded = h3_direct_rig
    module.conf = {"format": "mxfp8"}
    module.quant_format = "mxfp8"
    path = str(tmp_path / "direct-mxfp8.safetensors")
    _write_h3_layer(path, qdata, scale, marker=module.conf)

    with pytest.raises(
        CaptureAborted, match="'mxfp8' has no direct-marker restore recipe"
    ):
        capture_unbake_record(model, [_H3_KEY], path)


def test_direct_quant_conf_reason_names_each_blocker():
    reason = ub._direct_quant_conf_reason
    assert reason(_H3_DIRECT_CONF) is None
    assert reason({"format": "float8_e4m3fn"}) is None
    assert reason({"format": "int8_tensorwise", "params": {"convrot": True}}) is None
    assert "'nvfp4'" in reason({"format": "nvfp4"})
    assert "'per_row'" in reason({**_H3_DIRECT_CONF, "per_row": True})
    assert "'quant_group_size'" in reason(
        {"format": "int8_tensorwise", "params": {"quant_group_size": 64}})


def test_legacy_scaled_fp8_capture_maps_comfy_component_alias(quant_rig, tmp_path):
    model, module, _path, qdata, scale, recorded = quant_rig
    path = str(tmp_path / "legacy-fp8.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    spec = record.quant["diffusion_model.blk.weight"]
    assert spec.layer_conf == {"format": "float8_e4m3fn"}
    assert set(spec.components) == {"weight", "weight_scale"}
    assert not spec.resident and record.resident_count == 0

    module.qdata = (torch.randn(8, 8) * 0.2).to(torch.float8_e4m3fn)
    module.scale = torch.tensor(0.5, dtype=torch.float32)
    restore_pristine(model, record)
    assert torch.equal(module.rebuilt.data.view(torch.uint8), qdata.view(torch.uint8))
    assert torch.equal(recorded["params"].scale, scale)


def test_legacy_scaled_fp8_accepts_comfy_empty_sentinel(quant_rig, tmp_path):
    model, module, _path, qdata, scale, recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-empty-sentinel.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.empty(0, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    spec = record.quant["diffusion_model.blk.weight"]
    assert spec.layer_conf == {"format": "float8_e4m3fn"}
    assert set(spec.components) == {"weight", "weight_scale"}
    assert not spec.resident and record.resident_count == 0

    module.qdata = (torch.randn(8, 8) * 0.2).to(torch.float8_e4m3fn)
    module.scale = torch.tensor(0.5, dtype=torch.float32)
    restore_pristine(model, record)
    assert torch.equal(module.rebuilt.data.view(torch.uint8), qdata.view(torch.uint8))
    assert torch.equal(recorded["params"].scale, scale)


def test_legacy_scaled_fp8_empty_sentinel_rejects_live_full_precision(
    quant_rig, tmp_path
):
    model, module, _path, qdata, scale, _recorded = quant_rig
    module._full_precision_mm_config = True
    path = str(tmp_path / "legacy-fp8-empty-sentinel-live-drift.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.empty(0, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )

    with pytest.raises(CaptureAborted, match="live converted"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_preserves_full_precision_marker(quant_rig, tmp_path):
    model, module, _path, qdata, scale, _recorded = quant_rig
    module._full_precision_mm_config = True
    path = str(tmp_path / "legacy-fp8-full-mm.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(2, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    assert record.quant["diffusion_model.blk.weight"].layer_conf == {
        "format": "float8_e4m3fn",
        "full_precision_matrix_mult": True,
    }


def test_legacy_scaled_fp8_accepts_comfy_f32_marker(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-f32-marker.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float32),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )
    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    assert record.quant["diffusion_model.blk.weight"].layer_conf == {
        "format": "float8_e4m3fn"
    }


@pytest.mark.parametrize(
    ("marker", "label"),
    [
        (torch.ones(1, dtype=torch.int8), "dtype"),
        (torch.ones(3, dtype=torch.float8_e4m3fn), "cardinality"),
    ],
)
def test_legacy_scaled_fp8_rejects_unvouched_marker(
    quant_rig, tmp_path, marker, label
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / f"legacy-fp8-invalid-{label}.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": marker,
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )
    with pytest.raises(CaptureAborted):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_marker_is_bound_to_model_prefix(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-prefixed.safetensors")
    write_safetensors(
        path,
        {
            "model.diffusion_model.scaled_fp8": torch.ones(
                1, dtype=torch.float8_e4m3fn
            ),
            "model.diffusion_model.blk.weight": qdata.clone(),
            "model.diffusion_model.blk.scale_weight": scale.clone(),
        },
    )

    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    assert set(record.quant["diffusion_model.blk.weight"].components) == {
        "weight",
        "weight_scale",
    }

    root_prefix = str(tmp_path / "legacy-fp8-root-prefix.safetensors")
    write_safetensors(
        root_prefix,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "model.diffusion_model.blk.weight": qdata.clone(),
            "model.diffusion_model.blk.scale_weight": scale.clone(),
        },
    )
    root_record = capture_unbake_record(
        model, ["diffusion_model.blk.weight"], root_prefix
    )
    assert set(root_record.quant["diffusion_model.blk.weight"].components) == {
        "weight",
        "weight_scale",
    }

    wrong_prefix = str(tmp_path / "legacy-fp8-wrong-prefix.safetensors")
    write_safetensors(
        wrong_prefix,
        {
            "other.scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "model.diffusion_model.blk.weight": qdata.clone(),
            "model.diffusion_model.blk.scale_weight": scale.clone(),
        },
    )
    with pytest.raises(CaptureAborted):
        capture_unbake_record(
            model, ["diffusion_model.blk.weight"], wrong_prefix
        )


def test_legacy_scaled_fp8_rejects_unstripped_diffusion_namespace(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-unstripped-diffusion.safetensors")
    write_safetensors(
        path,
        {
            "diffusion_model.scaled_fp8": torch.ones(
                1, dtype=torch.float8_e4m3fn
            ),
            "diffusion_model.blk.weight": qdata.clone(),
            "diffusion_model.blk.scale_weight": scale.clone(),
        },
    )
    with pytest.raises(CaptureAborted):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_root_marker_precedes_namespace_marker(
    quant_rig, tmp_path
):
    model, module, _path, qdata, scale, _recorded = quant_rig
    module._full_precision_mm_config = True
    path = str(tmp_path / "legacy-fp8-root-precedence.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(2, dtype=torch.float8_e4m3fn),
            "model.diffusion_model.scaled_fp8": torch.ones(
                1, dtype=torch.float8_e4m3fn
            ),
            "model.diffusion_model.blk.weight": qdata.clone(),
            "model.diffusion_model.blk.scale_weight": scale.clone(),
        },
    )
    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    assert record.quant["diffusion_model.blk.weight"].layer_conf[
        "full_precision_matrix_mult"
    ] is True


def test_legacy_scaled_fp8_rejects_multiple_coherent_bindings(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-ambiguous-bindings.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
            "model.diffusion_model.blk.weight": qdata.clone(),
            "model.diffusion_model.blk.scale_weight": scale.clone(),
        },
    )
    with pytest.raises(CaptureAborted, match="2 coherent checkpoint bindings"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_embedded_quant_metadata_never_falls_back_to_legacy_marker(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "metadata-precedes-legacy.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
        metadata={"_quantization_metadata": json.dumps({"layers": {}})},
    )
    with pytest.raises(CaptureAborted):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_malformed_embedded_metadata_never_falls_back_to_legacy_marker(
    quant_rig, tmp_path
):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "malformed-metadata-precedes-legacy.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
        metadata={"_quantization_metadata": "not-json"},
    )
    with pytest.raises(CaptureAborted):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda module: setattr(module, "quant_format", "float8_e5m2"), "live converted"),
        (
            lambda module: setattr(module, "_full_precision_mm_config", True),
            "live converted",
        ),
    ],
)
def test_legacy_scaled_fp8_requires_exact_live_conversion(
    quant_rig, tmp_path, mutate, match
):
    model, module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-live-drift.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )
    mutate(module)
    with pytest.raises(CaptureAborted, match=match):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_requires_valid_live_marker(quant_rig, tmp_path):
    model, module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-invalid-live-marker.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )
    original = module.state_dict

    def invalid_marker(*args, **kwargs):
        state = original(*args, **kwargs)
        state["comfy_quant"] = torch.tensor([0xFF], dtype=torch.uint8)
        return state

    module.state_dict = invalid_marker
    with pytest.raises(CaptureAborted, match="live converted"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_requires_live_weight_and_scale(quant_rig, tmp_path):
    model, module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-missing-live-scale.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )
    original = module.state_dict

    def missing_scale(*args, **kwargs):
        state = original(*args, **kwargs)
        del state["weight_scale"]
        return state

    module.state_dict = missing_scale
    with pytest.raises(CaptureAborted, match="live converted"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_requires_scale_weight_spelling(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-new-scale-name.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.weight_scale": scale.clone(),
        },
    )
    with pytest.raises(CaptureAborted):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_rejects_component_alias_collision(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-alias-collision.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
            "blk.weight_scale": scale.clone(),
        },
    )
    with pytest.raises(CaptureAborted, match="live converted"):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)


def test_legacy_scaled_fp8_scale_mismatch_stays_resident(quant_rig, tmp_path):
    model, module, _path, qdata, scale, recorded = quant_rig
    path = str(tmp_path / "legacy-fp8-resident-scale.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )
    module.scale = torch.tensor(0.9999, dtype=torch.float32)
    pristine_scale = module.scale.clone()
    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    spec = record.quant["diffusion_model.blk.weight"]
    assert "weight" in spec.components
    assert "weight_scale" in spec.resident

    module.scale = torch.tensor(0.5, dtype=torch.float32)
    restore_pristine(model, record)
    assert torch.equal(recorded["params"].scale, pristine_scale)


def test_quant_scale_mismatch_stays_resident(quant_rig):
    model, module, path, _qdata, _scale, recorded = quant_rig
    module.scale = torch.tensor(0.9999, dtype=torch.float32)
    pristine_scale = module.scale.clone()
    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    spec = record.quant["diffusion_model.blk.weight"]
    assert "weight" in spec.components and "weight_scale" in spec.resident

    module.scale = torch.tensor(0.5, dtype=torch.float32)
    restore_pristine(model, record)
    assert torch.equal(recorded["params"].scale, pristine_scale)


def test_quant_without_metadata_aborts(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _ = quant_rig
    bare = str(tmp_path / "bare.safetensors")
    write_safetensors(bare, {"blk.weight": qdata.clone(),
                             "blk.weight_scale": scale.clone()})
    with pytest.raises(CaptureAborted):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], bare)


def test_quant_format_drift_raises(quant_rig):
    model, module, path, _, _, _ = quant_rig
    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    module.quant_format = "nvfp4"
    with pytest.raises(UnbakeError):
        restore_pristine(model, record)


def test_legacy_quant_full_precision_config_drift_raises(quant_rig, tmp_path):
    model, module, _path, qdata, scale, _ = quant_rig
    path = str(tmp_path / "legacy-fp8-restore-config-drift.safetensors")
    write_safetensors(
        path,
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": qdata.clone(),
            "blk.scale_weight": scale.clone(),
        },
    )
    record = capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
    module._full_precision_mm_config = True
    with pytest.raises(UnbakeError, match="full-precision matmul config"):
        restore_pristine(model, record)


def test_pipelined_restore_after_key_ordering(rig):
    """The after_key hook must see each key already restored (lazy_swap bakes
    on it), and fire exactly once per restored key."""
    model, path, file_tensors = rig
    record = capture_unbake_record(model, KEYS, path)
    _bake(model)
    seen = []

    def check(key):
        live = model
        for part in key.split("."):
            live = getattr(live, part)
        name = key[len("diffusion_model."):]
        assert torch.equal(live, file_tensors[name]), f"{key} not yet restored"
        seen.append(key)

    restore_pristine(model, record, after_key=check)
    assert sorted(seen) == sorted(KEYS)


def test_direct_read_alignment_math():
    """_pread_direct must land the payload at dst_off+pad regardless of the
    tensor's byte offset in the file. Runs O_DIRECT where the filesystem allows
    it and checks the buffered read where it does not (tmpfs on older kernels)."""
    import tempfile

    from dgx_monarch.actor.unbake import _aligned_buffer, _open_for_restore, _pread_direct

    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as d:
        path = os.path.join(d, "blob.bin")
        payload = bytes(range(256)) * 64                     # 16 KiB, recognizable
        with open(path, "wb") as f:
            f.write(b"\x00" * 5)                             # force an unaligned start
            f.write(payload)
        fd, direct = _open_for_restore(path)
        try:
            buf = _aligned_buffer(len(payload) + 3 * 4096)
            if direct:
                pad = _pread_direct(fd, 5, len(payload), buf, 0)
                got = bytes(buf[pad:pad + len(payload)].numpy().tobytes())
            else:  # filesystem without O_DIRECT: buffered path still serves
                got = os.pread(fd, len(payload), 5)
            assert got == payload
        finally:
            os.close(fd)


def test_unknown_quant_format_aborts_capture(quant_rig, tmp_path):
    model, _module, _path, qdata, scale, _ = quant_rig
    meta = {"_quantization_metadata": json.dumps(
        {"layers": {"blk": {"format": "int4_hypothetical"}}})}
    path = str(tmp_path / "future.safetensors")
    write_safetensors(path, {"blk.weight": qdata.clone(),
                             "blk.weight_scale": scale.clone()}, metadata=meta)
    with pytest.raises(CaptureAborted):
        capture_unbake_record(model, ["diffusion_model.blk.weight"], path)
