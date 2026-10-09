"""Tensor-tree packing and latent return descriptors: message path, RDMA fallback and poison, split reads."""
import itertools
import pickle
import threading

import torch

import dgx_monarch.transfer as transfer
from dgx_monarch.transfer import (
    LatentReturn,
    pack_conditioning,
    read_latent_result,
    tree_to_cpu,
)
from transfer_helpers import (  # noqa: F401  # autouse fixture import.
    _FakeRDMABuffer,
    _isolate_process_lifetime_rdma_poison,
    _record_unlocked_poison_errors,
)


def test_tree_to_cpu_conditioning_shape():
    cond = [[torch.randn(1, 77, 768), {"pooled_output": torch.randn(1, 768), "note": "x"}]]
    packed = pack_conditioning(cond)
    assert packed[0][0].device.type == "cpu"
    assert packed[0][1]["pooled_output"].shape == (1, 768)
    assert packed[0][1]["note"] == "x"


def test_tree_handles_nesting_and_none():
    tree = {"a": [None, (torch.ones(2), {"b": torch.zeros(3)})], "c": 5}
    out = tree_to_cpu(tree)
    assert out["a"][0] is None
    assert out["a"][1][0].sum() == 2
    assert out["c"] == 5


def test_message_latent_roundtrip():
    lr = LatentReturn("message")
    t = torch.randn(1, 4, 8, 8)
    desc = lr.pack("samples", t)
    assert desc["kind"] == "message"
    back = read_latent_result(desc)
    assert torch.equal(back, t)


def test_rdma_falls_back_without_ibverbs(monkeypatch):
    # is_ibverbs_available() is True wherever libibverbs is installed, even with
    # no usable fabric (CI runners), where a real RDMABuffer aborts the process
    # (SIGABRT, which try/except cannot catch; transfer_utils.native_rdma_preflight
    # probes it in a child). Forcing the guard off reaches the fallback the same
    # way on every box and runs no native probe.
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: False)
    # min_bytes=1 disables the size gate so this exercises the ibverbs branch,
    # not the small-latent shortcut.
    lr = LatentReturn("rdma", min_bytes=1)
    t = torch.randn(1, 4, 8, 8)
    desc = lr.pack("samples", t)
    assert desc["kind"] == "message"
    assert lr._keepalive == {}
    assert torch.equal(read_latent_result(desc), t)


def test_direct_rdma_falls_back_when_buffer_construction_fails(monkeypatch):
    # The direct (tokenless) fallback: ibverbs reports available but RDMABuffer
    # construction raises, so pack() returns a message descriptor and adds nothing
    # to `_keepalive`; the poison record owns the tensor until the process exits.
    def _raise(*_a, **_k):
        raise RuntimeError("no RDMA device")

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", _raise)
    lr = LatentReturn("rdma", min_bytes=1)
    t = torch.randn(1, 4, 8, 8)
    desc = lr.pack("samples", t)
    assert desc["kind"] == "message"
    assert lr._keepalive == {}
    assert torch.equal(read_latent_result(desc), t)


def test_constructor_poison_logs_once_then_falls_back_silently(
    monkeypatch, caplog,
):
    calls = 0

    def fail_registration(_data):
        nonlocal calls
        calls += 1
        raise RuntimeError("private constructor detail")

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", fail_registration)
    latent_return = LatentReturn("rdma", min_bytes=1)
    tensor = torch.arange(8, dtype=torch.uint8)

    first = latent_return.pack("samples", tensor)

    assert first["kind"] == "message"
    poison_records = [
        record for record in caplog.records
        if "RDMA registration ownership POISONED" in record.getMessage()
    ]
    assert len(poison_records) == 1
    assert poison_records[0].levelname == "ERROR"
    poison_message = poison_records[0].getMessage()
    assert "native registration outcome is unknown" in poison_message
    assert "Attached mesh reset required" in poison_message
    assert "private constructor detail" not in poison_message

    caplog.clear()
    second = latent_return.pack("samples", tensor + 1)

    assert second["kind"] == "message"
    assert calls == 1
    assert caplog.records == []


def test_concurrent_constructor_poison_fallbacks_share_one_error(
    monkeypatch, caplog,
):
    from concurrent.futures import ThreadPoolExecutor

    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    contenders_seen = threading.Condition()
    contender_threads: set[int] = set()
    registration_calls = 0

    def fail_registration(_data):
        nonlocal registration_calls
        registration_calls += 1
        constructor_entered.set()
        if not release_constructor.wait(timeout=5):
            raise RuntimeError("constructor test timed out")
        raise RuntimeError("registration failed")

    real_rdma_poisoned = transfer._rdma_poisoned

    def observe_contention():
        if constructor_entered.is_set():
            with contenders_seen:
                contender_threads.add(threading.get_ident())
                contenders_seen.notify_all()
        return real_rdma_poisoned()

    monkeypatch.setattr(transfer, "_rdma_poisoned", observe_contention)
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", fail_registration)
    latent_return = LatentReturn("rdma", min_bytes=1)

    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(
            latent_return.pack, "samples", torch.arange(8, dtype=torch.uint8))
        assert constructor_entered.wait(timeout=2)
        contenders = [
            executor.submit(
                latent_return.pack, "samples",
                torch.arange(8, dtype=torch.uint8) + index,
            )
            for index in range(1, 4)
        ]
        with contenders_seen:
            all_waiting = contenders_seen.wait_for(
                lambda: len(contender_threads) == 3, timeout=2)
        release_constructor.set()
        assert all_waiting
        results = [first.result(), *(future.result() for future in contenders)]

    assert all(result["kind"] == "message" for result in results)
    assert registration_calls == 1
    poison_records = [
        record for record in caplog.records
        if "RDMA registration ownership POISONED" in record.getMessage()
    ]
    assert len(poison_records) == 1
    assert poison_records[0].levelname == "ERROR"
    fallback_records = [
        record for record in caplog.records
        if "RDMA latent return unavailable" in record.getMessage()
    ]
    assert len(fallback_records) == 1


def test_tentative_constructor_intent_never_logs(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    poison_errors = _record_unlocked_poison_errors(monkeypatch)
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    contender_entered = threading.Event()
    registration_calls = 0

    def register(_data):
        nonlocal registration_calls
        registration_calls += 1
        if registration_calls == 1:
            constructor_entered.set()
            if not release_constructor.wait(timeout=5):
                raise RuntimeError("constructor test timed out")
        return object()

    real_rdma_poisoned = transfer._rdma_poisoned

    def observe_contention():
        if constructor_entered.is_set():
            contender_entered.set()
        return real_rdma_poisoned()

    monkeypatch.setattr(transfer, "_rdma_poisoned", observe_contention)
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            LatentReturn("rdma", min_bytes=1).pack,
            "samples", torch.arange(8, dtype=torch.uint8))
        assert constructor_entered.wait(timeout=2)
        second = executor.submit(
            LatentReturn("rdma", min_bytes=1).pack,
            "samples", torch.arange(8, dtype=torch.uint8) + 1)
        assert contender_entered.wait(timeout=2)
        assert len(transfer._RDMA_POISONED_OWNERS) == 1
        assert poison_errors == []
        release_constructor.set()
        results = [first.result(), second.result()]

    assert [result["kind"] for result in results] == ["rdma", "rdma"]
    assert registration_calls == 2
    assert transfer._RDMA_POISONED_OWNERS == []
    assert poison_errors == []


def test_byte_split_tiles_without_gaps():
    from dgx_monarch.transfer import _byte_split

    assert _byte_split(10, 1) == [(0, 10)]
    assert _byte_split(10, 2) == [(0, 5), (5, 5)]
    assert _byte_split(10, 3) == [(0, 3), (3, 3), (6, 4)]  # the last range takes the remainder
    assert _byte_split(3, 8) == [(0, 1), (1, 1), (2, 1)]   # n clamped to nbytes
    for n in (1, 2, 3, 4, 7):
        ranges = _byte_split(100, n)
        assert ranges[0][0] == 0
        assert sum(size for _, size in ranges) == 100
        for (off0, size0), (off1, _size1) in itertools.pairwise(ranges):
            assert off0 + size0 == off1  # contiguous, no gaps/overlap


def test_rdma_size_gate_messages_small_latent(monkeypatch):
    # In rdma mode a latent below the size gate never builds an RDMABuffer, so
    # image latents stay on the message path they were benchmarked on.
    def _boom(*_a, **_k):
        raise AssertionError("RDMABuffer must not be built below the size gate")

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", _boom)
    lr = LatentReturn("rdma")  # default 8 MiB gate
    t = torch.randn(1, 4, 8, 8)  # 1 KiB
    desc = lr.pack("samples", t)
    assert desc["kind"] == "message"
    assert lr._keepalive == {}
    assert torch.equal(read_latent_result(desc), t)


def test_rdma_single_buffer_roundtrip(monkeypatch):
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.delenv("DGXM_RDMA_QP_SPLIT", raising=False)  # default split=1
    lr = LatentReturn("rdma", min_bytes=1)
    t = torch.randn(1, 16, 32, 32)
    desc = lr.pack("samples", t)
    assert desc["kind"] == "rdma" and len(desc["parts"]) == 1
    assert desc["parts"][0]["offset"] == 0
    assert ("samples", 0) in lr._keepalive
    assert torch.equal(read_latent_result(desc), t)
    assert desc["parts"][0]["buffer"].dropped == 1
    assert transfer._RDMA_POISONED_OWNERS == []


def test_rdma_split_roundtrip_is_byte_identical(monkeypatch):
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")
    lr = LatentReturn("rdma", min_bytes=1)
    t = torch.randn(2, 4, 16, 16)
    desc = lr.pack("samples", t)
    assert desc["kind"] == "rdma"
    parts = desc["parts"]
    assert len(parts) == 2
    nbytes = t.numel() * t.element_size()
    assert parts[0]["offset"] == 0
    assert parts[1]["offset"] == parts[0]["nbytes"]
    assert parts[0]["nbytes"] + parts[1]["nbytes"] == nbytes  # disjoint, whole
    assert ("samples", 0) in lr._keepalive
    back = read_latent_result(desc)  # drives _read_parts_concurrent
    assert back.dtype == t.dtype and list(back.shape) == list(t.shape)
    assert torch.equal(back, t)
    assert all(part["buffer"].dropped == 1 for part in parts)
    lr.drop()
    assert lr._keepalive == {}


def test_rdma_split_odd_bytes_reconstructs(monkeypatch):
    # A byte count not divisible by the split: the last QP absorbs the remainder
    # and reconstruction is still exact.
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "3")
    lr = LatentReturn("rdma", min_bytes=1)
    t = torch.arange(1, 26, dtype=torch.uint8)  # 25 bytes, not a multiple of 3
    desc = lr.pack("samples", t)
    assert [p["nbytes"] for p in desc["parts"]] == [8, 8, 9]
    assert torch.equal(read_latent_result(desc), t)


def test_broken_fallback_logger_does_not_break_message_fallback(monkeypatch):
    created: list[_FakeRDMABuffer] = []

    def register(data):
        if created:
            raise RuntimeError("registration failed")
        buffer = _FakeRDMABuffer(data)
        created.append(buffer)
        return buffer

    class LogFailure(BaseException):
        pass

    def fail_log(*_args, **_kwargs):
        raise LogFailure("diagnostic handler failed")

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setattr(transfer.log, "warning", fail_log)
    monkeypatch.setattr(transfer.log, "error", fail_log)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")

    desc = LatentReturn("rdma", min_bytes=1).pack(
        "samples", torch.arange(24, dtype=torch.uint8)
    )

    assert desc["kind"] == "message"
    assert created[0].dropped == 1


class NestedTensor:
    """Stand-in for ComfyUI's NestedTensor: is_nested_tensor matches the class name and unbind()."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def test_nested_latent_pickle_message_path_has_no_rdma_or_read_authority(monkeypatch):
    source = NestedTensor((
        torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        torch.arange(10, dtype=torch.float64).reshape(1, 2, 5),
    ))

    def fail_rdma(*_args, **_kwargs):
        raise AssertionError("NestedTensor must never enter RDMABuffer")

    class Guard:
        def read_authority(self):
            raise AssertionError("message transport must not claim an RDMA read token")

    monkeypatch.setattr(transfer, "RDMABuffer", fail_rdma)
    latent_return = LatentReturn("rdma", min_bytes=1)
    descriptor = latent_return.pack("samples", source)
    # Trusted in-process fixture: only simulate the actor serialization boundary.
    wire_descriptor = pickle.loads(pickle.dumps(descriptor))  # noqa: S301
    restored = read_latent_result(wire_descriptor, Guard())

    assert descriptor["kind"] == "message"
    assert type(restored) is NestedTensor
    assert [tuple(part.shape) for part in restored.unbind()] == [(1, 3, 4), (1, 2, 5)]
    assert [part.dtype for part in restored.unbind()] == [torch.float32, torch.float64]
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(restored.unbind(), source.unbind(), strict=True)
    )
    assert latent_return._keepalive == {}
