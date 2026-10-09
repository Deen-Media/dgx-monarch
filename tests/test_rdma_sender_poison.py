"""Drop and poison ownership on the read_latent_result cleanup path (transfer._poison_failed_drops)."""
import pytest

import dgx_monarch.transfer as transfer
from dgx_monarch.transfer import read_latent_result
from transfer_helpers import (  # noqa: F401  # autouse fixture import.
    _FakeFuture,
    _isolate_process_lifetime_rdma_poison,
)


def test_rdma_invalid_dtype_drop_failure_is_owned_without_destination():
    class DropFailureFuture:
        def get(self, timeout=None):
            raise RuntimeError("metadata cleanup drop failed")

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return DropFailureFuture()

    buffer = Buffer()
    desc = {
        "kind": "rdma",
        "dtype": "not_a_torch_dtype",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    with pytest.raises(AttributeError, match="not_a_torch_dtype"):
        read_latent_result(desc)

    assert buffer.dropped == 1
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.phase == "latent read preparation cleanup"
    assert owner.parts[0]["buffer"] is buffer
    assert owner.keepalive is None


@pytest.mark.parametrize("stage", ["allocation", "byte view"])
def test_rdma_destination_preparation_baseexception_is_primary(
    monkeypatch, stage
):
    class PreparationStop(BaseException):
        pass

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    primary = PreparationStop(f"{stage} interrupted")
    buffer = Buffer()

    if stage == "allocation":
        def fail_empty(*_args, **_kwargs):
            raise primary

        monkeypatch.setattr(transfer.torch, "empty", fail_empty)
    else:
        class BrokenDestination:
            def flatten(self):
                raise primary

        monkeypatch.setattr(
            transfer.torch, "empty", lambda *_args, **_kwargs: BrokenDestination()
        )

    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    with pytest.raises(PreparationStop) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is primary
    assert buffer.dropped == 1
    assert transfer._RDMA_POISONED_OWNERS == []


def test_poison_publication_interruption_never_retries_ambiguous_drop(
    monkeypatch,
):
    class ReadFailure(RuntimeError):
        pass

    class PoisonStop(BaseException):
        pass

    class DropFailureFuture:
        def get(self, timeout=None):
            raise RuntimeError("drop outcome unknown")

    primary = ReadFailure("read failed first")
    poison_stop = PoisonStop("interrupted after poison publication")
    publications = 0

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def read_into(self, _dst, timeout=None):
            raise primary

        def drop(self):
            self.dropped += 1
            return DropFailureFuture()

    real_poison = transfer._poison_failed_drops

    def publish_then_stop(*args, **kwargs):
        nonlocal publications
        publications += 1
        real_poison(*args, **kwargs)
        raise poison_stop

    monkeypatch.setattr(transfer, "_poison_failed_drops", publish_then_stop)
    buffer = Buffer()
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    with pytest.raises(PoisonStop) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is poison_stop
    assert poison_stop.__cause__ is primary
    assert poison_stop.__cause__ is not poison_stop
    assert poison_stop.__context__ is not poison_stop
    assert publications == 1
    assert buffer.dropped == 0
    assert len(transfer._RDMA_POISONED_OWNERS) == 1


def test_poison_publication_retries_one_shot_before_owner_append(monkeypatch):
    class PreparationStop(BaseException):
        pass

    class PublicationStop(BaseException):
        pass

    class DropFailureFuture:
        def get(self, timeout=None):
            raise RuntimeError("drop outcome unknown")

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return DropFailureFuture()

    primary = PreparationStop("allocation interrupted")
    real_owner_factory = transfer._new_poison_owner
    publications = 0

    def fail_first_owner_construction(*args, **kwargs):
        nonlocal publications
        publications += 1
        if publications == 1:
            raise PublicationStop("interrupted before owner append")
        return real_owner_factory(*args, **kwargs)

    def fail_empty(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(transfer, "_new_poison_owner", fail_first_owner_construction)
    monkeypatch.setattr(transfer.torch, "empty", fail_empty)
    buffer = Buffer()
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    with pytest.raises(PreparationStop) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is primary
    assert publications == 2
    assert buffer.dropped == 1
    assert len(transfer._RDMA_POISONED_OWNERS) == 1
    assert transfer._RDMA_POISONED_OWNERS[0].parts[0]["buffer"] is buffer


def test_release_boundary_interruption_owns_ambiguous_and_drains_later_parts(
    monkeypatch,
):
    class CleanupStop(BaseException):
        pass

    cleanup_stop = CleanupStop("release loop interrupted")

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    class InterruptingParts(list):
        def __iter__(self):
            iterator = super().__iter__()
            yield next(iterator)
            raise cleanup_stop

    monkeypatch.setattr(transfer, "_read_parts_concurrent", lambda *_args: None)
    buffers = [Buffer(), Buffer()]
    parts = InterruptingParts([
        {"buffer": buffers[0], "offset": 0, "nbytes": 1},
        {"buffer": buffers[1], "offset": 1, "nbytes": 1},
    ])
    desc = {"kind": "rdma", "dtype": "uint8", "shape": [2], "parts": parts}

    with pytest.raises(CleanupStop) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is cleanup_stop
    assert isinstance(cleanup_stop.__cause__, RuntimeError)
    assert "read completed but 1 buffer" in str(cleanup_stop.__cause__)
    assert cleanup_stop.__cause__ is not cleanup_stop
    assert cleanup_stop.__context__ is not cleanup_stop
    assert [buffer.dropped for buffer in buffers] == [1, 1]
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert [part["buffer"] for part in owner.parts] == [buffers[0]]


def test_release_recovery_interruption_poison_owns_all_without_masking(monkeypatch):
    class ReadFailure(BaseException):
        pass

    class LoopStop(BaseException):
        pass

    class RecordingStop(BaseException):
        pass

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    class InterruptingParts(list):
        def __iter__(self):
            iterator = super().__iter__()
            yield next(iterator)
            raise LoopStop("release loop interrupted")

    primary = ReadFailure("read failed first")
    real_drop_failure = transfer._drop_failure
    recordings = 0

    def fail_first_recording(*args, **kwargs):
        nonlocal recordings
        recordings += 1
        if recordings == 1:
            raise RecordingStop("failure recording interrupted")
        return real_drop_failure(*args, **kwargs)

    def fail_read(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(transfer, "_read_parts_concurrent", fail_read)
    monkeypatch.setattr(transfer, "_drop_failure", fail_first_recording)
    buffers = [Buffer(), Buffer()]
    parts = InterruptingParts([
        {"buffer": buffers[0], "offset": 0, "nbytes": 1},
        {"buffer": buffers[1], "offset": 1, "nbytes": 1},
    ])
    desc = {"kind": "rdma", "dtype": "uint8", "shape": [2], "parts": parts}

    with pytest.raises(ReadFailure) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is primary
    assert [buffer.dropped for buffer in buffers] == [0, 0]
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert [part["buffer"] for part in owner.parts] == buffers
    assert any("failure recording interrupted" in note
               for note in primary.__notes__)
