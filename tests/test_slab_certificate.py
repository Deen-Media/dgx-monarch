"""Check actor/slab_certificate.py's slab-load byte certificate.

A detected mismatch with checkpoint bytes must abort the load. Tests cover
successful certificates, tamper diagnostics, gaps that sampled mode misses but
full mode catches, and exception boundaries that keep corruption distinct from
dtype fallback, capacity refusal and header parse errors.
"""
import json
import os
import struct
import sys
import types

import pytest
import torch

from dgx_monarch import mesh_safety
from dgx_monarch.actor import slab as slab_module
from dgx_monarch.actor import slab_certificate
from dgx_monarch.actor.slab import WeightSlab
from dgx_monarch.actor.slab_certificate import (
    FULL_VERIFY_MAX_BYTES,
    MODE_FULL,
    MODE_SAMPLED,
    CertificateBuilder,
    SlabCertificateError,
    fingerprint_windows,
)
from dgx_monarch.safetensors_header import (
    SafetensorsHeaderError,
    UnsupportedSafetensorsDtypeError,
    read_safetensors_header,
)
from slab_lifetime_helpers import reset_slab_lifetime

_TAG = {torch.bfloat16: "BF16", torch.float32: "F32"}


def write_safetensors(path, tensors, metadata=None):
    header, blobs, off = {}, [], 0
    for name, tensor in tensors.items():
        raw = tensor.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": _TAG[tensor.dtype], "shape": list(tensor.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for raw in blobs:
            handle.write(raw)


@pytest.fixture(autouse=True)
def _isolated_slab_lifetime(monkeypatch):
    """Start each test from a clean actor/slab_lifetime registry.

    A load whose cleanup fails poisons that registry (its fail-closed contract),
    so this file's failures must not leak into another test's store.
    """
    reset_slab_lifetime(monkeypatch)


@pytest.fixture
def ckpt(tmp_path):
    """A checkpoint with a tensor big enough to leave unsampled gaps."""
    torch.manual_seed(11)
    tensors = {
        "blocks.0.w": torch.randn(64, 64, dtype=torch.float32).to(torch.bfloat16),
        "blocks.0.b": torch.randn(64, dtype=torch.float32).to(torch.bfloat16),
        "blocks.1.big": torch.randn(128, 128, dtype=torch.float32).to(torch.bfloat16),
        "small": torch.arange(10, dtype=torch.float32).to(torch.bfloat16),
    }
    path = tmp_path / "model.safetensors"
    write_safetensors(str(path), tensors, metadata={"fmt": "test"})
    return str(path), tensors


def _descriptor(path, name):
    return read_safetensors_header(path).tensors[name]


def _tamper_preadv(monkeypatch, *, at_file_offset: int, byte_index: int):
    """Corrupt exactly one byte of the tensor whose read starts at that offset."""
    real = os.preadv
    hits = []

    def tampering(fd, buffers, offset):
        got = real(fd, buffers, offset)
        if offset == at_file_offset:
            view = buffers[0]
            view[byte_index] = view[byte_index] ^ 0xFF
            hits.append(offset)
        return got

    monkeypatch.setattr(slab_module.os, "preadv", tampering)
    return hits


def test_certificate_issued_for_a_real_file(ckpt):
    path, tensors = ckpt
    slab = WeightSlab(path)
    try:
        certificate = slab.certificate
        assert certificate is not None
        assert certificate.mode == MODE_SAMPLED
        assert certificate.file_name == "model.safetensors"
        assert certificate.file_identity == slab.file_identity
        assert certificate.tensor_count == len(tensors)
        assert certificate.verified_tensors == certificate.tensor_count
        assert certificate.total_bytes == sum(
            ft.nbytes for ft in slab.headers.values())
        assert certificate.verified_bytes > 0
        assert len(certificate.layout_digest) == 64
        assert len(certificate.header_digest) == 64
        payload = json.dumps(certificate.public(), allow_nan=False)
        assert json.loads(payload)["summary"] == certificate.digest_summary()
        assert slab.telemetry()["slab_cert"]["mode"] == MODE_SAMPLED
    finally:
        slab.close()


def test_tamper_a_byte_refuses(ckpt, monkeypatch):
    """One wrong byte aborts the load, and the refusal names where."""
    path, _tensors = ckpt
    descriptor = _descriptor(path, "blocks.0.w")
    hits = _tamper_preadv(monkeypatch, at_file_offset=descriptor.start, byte_index=17)

    with pytest.raises(SlabCertificateError) as raised:
        WeightSlab(path)

    message = str(raised.value)
    assert hits == [descriptor.start]
    assert "blocks.0.w" in message
    assert f"file offset {descriptor.start}+0" in message
    assert "slab offset" in message
    assert "first differing byte 17" in message
    assert "nothing was adopted" in message
    assert "Nothing was quarantined" in message
    # The failed load leaves no live slab registered for this checkpoint.
    assert not any(key[0] == path for key in slab_module._OPEN_SLABS)


def test_sampled_windows_tile_completely_at_or_below_the_full_threshold():
    """At or below the threshold the sampled windows already cover every byte.

    So verifying the whole tensor there is one read instead of three with the
    same coverage. The first gap opens above it.
    """
    for nbytes in (1, 4095, 4096, 4097, 8192, FULL_VERIFY_MAX_BYTES):
        covered = set()
        for offset, length in fingerprint_windows(nbytes):
            covered.update(range(offset, offset + length))
        assert covered == set(range(nbytes)), nbytes

    gapped = FULL_VERIFY_MAX_BYTES + 2
    covered = set()
    for offset, length in fingerprint_windows(gapped):
        covered.update(range(offset, offset + length))
    assert covered != set(range(gapped))


def test_tamper_outside_the_windows_is_missed_in_sampled_and_caught_in_full(
        ckpt, monkeypatch):
    """Sampled mode misses a byte between its windows; full mode catches it."""
    path, _tensors = ckpt
    descriptor = _descriptor(path, "blocks.1.big")
    assert descriptor.nbytes > FULL_VERIFY_MAX_BYTES
    gap = next(
        index for index in range(descriptor.nbytes)
        if not any(offset <= index < offset + length
                   for offset, length in fingerprint_windows(descriptor.nbytes))
    )
    _tamper_preadv(monkeypatch, at_file_offset=descriptor.start, byte_index=gap)

    slab = WeightSlab(path)          # sampled mode does not see it
    try:
        assert slab.certificate is not None
        assert slab.certificate.mode == MODE_SAMPLED
    finally:
        slab.close()

    monkeypatch.setenv(slab_certificate.SLAB_CERTIFY_ENV, MODE_FULL)
    with pytest.raises(SlabCertificateError, match=r"blocks\.1\.big"):
        WeightSlab(path)


def test_short_source_read_refuses(ckpt, monkeypatch):
    path, _tensors = ckpt
    real = os.pread

    def short(fd, length, offset):
        data = real(fd, length, offset)
        return data[:-1] if len(data) > 1 else data

    monkeypatch.setattr(slab_certificate.os, "pread", short)
    with pytest.raises(SlabCertificateError, match="returned"):
        WeightSlab(path)


def test_layout_digest_is_stable_and_offset_sensitive(ckpt):
    path, _tensors = ckpt
    first = WeightSlab(path)
    try:
        second = WeightSlab(path)
        try:
            assert first.certificate is not None and second.certificate is not None
            assert first.certificate.layout_digest == second.certificate.layout_digest
            assert first.certificate.header_digest == second.certificate.header_digest
        finally:
            second.close()
    finally:
        first.close()

    descriptor = _descriptor(path, "small")
    fd = os.open(path, os.O_RDONLY)
    try:
        digests = []
        for offset in (0, 256):
            buffer = bytearray(offset + descriptor.nbytes)
            buffer[offset:offset + descriptor.nbytes] = os.pread(
                fd, descriptor.nbytes, descriptor.start)
            builder = CertificateBuilder(
                path=path, file_identity="dev:ino", header_len=8,
                tensor_count=1, total_bytes=descriptor.nbytes)
            builder.verify_region(
                fd, "small", first.headers["small"], memoryview(buffer), offset)
            digests.append(builder._layout.hexdigest())
        assert digests[0] != digests[1]
    finally:
        os.close(fd)


def test_issue_refuses_when_a_tensor_was_skipped(ckpt):
    path, _tensors = ckpt
    fd = os.open(path, os.O_RDONLY)
    try:
        builder = CertificateBuilder(
            path=path, file_identity="dev:ino", header_len=8,
            tensor_count=4, total_bytes=0)
        with pytest.raises(SlabCertificateError, match="0 of 4 tensors"):
            builder.issue(fd)
    finally:
        os.close(fd)


def test_certificate_error_stays_out_of_every_handler_family():
    error = SlabCertificateError("x")
    # A dtype capability miss falls back to the stock loader; a byte mismatch
    # must never take that branch.
    assert not isinstance(error, UnsupportedSafetensorsDtypeError)
    # A header parse error is swallowed by `except (OSError, ValueError)` on
    # the file-read paths.
    assert not isinstance(error, SafetensorsHeaderError)
    assert not isinstance(error, ValueError)
    assert not isinstance(error, OSError)
    # Corruption is not capacity: the cross-residency gate leg must not file it
    # as CAPACITY, in either the local or the Monarch-wrapped form.
    assert not isinstance(error, mesh_safety.StockLoadCapacityError)
    assert not mesh_safety.is_stock_load_capacity_error(error)
    assert not mesh_safety.is_stock_load_capacity_error(
        RuntimeError(f"ActorError: SlabCertificateError: {error}"))


def test_certificate_error_does_not_take_the_stock_fallback(monkeypatch, tmp_path):
    """comfy_bridge falls back to stock on a dtype miss, never on corruption."""
    from dgx_monarch.actor import comfy_bridge

    loads = []

    class Failing:
        def __enter__(self):
            raise SlabCertificateError("slab byte-verify FAILED for m.safetensors")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        comfy_bridge, "slab_load", lambda _path, **_kwargs: Failing())
    mm_stub = types.ModuleType("comfy.model_management")
    mm_stub.get_torch_device = lambda: "cuda:0"
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm_stub)

    with pytest.raises(SlabCertificateError):
        comfy_bridge.load_diffusion_model_slab(
            str(tmp_path / "m.safetensors"), {}, lambda *a, **k: loads.append(a))
    assert loads == []


def test_mode_from_env(monkeypatch, caplog):
    monkeypatch.delenv(slab_certificate.SLAB_CERTIFY_ENV, raising=False)
    assert slab_certificate.certificate_mode() == MODE_SAMPLED
    monkeypatch.setenv(slab_certificate.SLAB_CERTIFY_ENV, "FULL")
    assert slab_certificate.certificate_mode() == MODE_FULL
    monkeypatch.setenv(slab_certificate.SLAB_CERTIFY_ENV, "paranoid")
    with caplog.at_level("WARNING"):
        assert slab_certificate.certificate_mode() == MODE_SAMPLED
    assert "DGXM_SLAB_CERTIFY" in caplog.text


def test_certificate_mode_is_allowed_through_to_workers():
    from dgx_monarch.cli import worker_process_env

    assert worker_process_env.worker_environment_key_allowed(
        slab_certificate.SLAB_CERTIFY_ENV)
