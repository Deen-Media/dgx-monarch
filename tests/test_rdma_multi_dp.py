"""Multi-leader descriptor draining in latent_outputs.materialize_leader_samples."""
import gc
import inspect
import sys
import weakref

import pytest
import torch

import dgx_monarch.transfer as transfer
from dgx_monarch.nodes.latent_outputs import materialize_leader_samples
from dgx_monarch.transfer import read_latent_result
from transfer_helpers import (  # noqa: F401  # autouse fixture import.
    _FakeRDMABuffer,
    _isolate_process_lifetime_rdma_poison,
)


def test_multi_dp_materialization_settles_every_descriptor_before_reraising():
    class FirstReadFailure(BaseException):
        pass

    class LaterReadFailure(RuntimeError):
        def __repr__(self):
            raise RuntimeError("broken failure repr")

    first_failure = FirstReadFailure("dp0 read failed")
    later_failure = LaterReadFailure("dp2 read failed")

    class Buffer(_FakeRDMABuffer):
        def __init__(self, data, failure=None):
            super().__init__(data)
            self.failure = failure
            self.reads = 0
            self.destination_ref = None

        def read_into(self, dst, timeout=None):
            self.reads += 1
            root = dst
            while isinstance(getattr(root, "_base", None), torch.Tensor):
                root = root._base
            self.destination_ref = weakref.ref(root)
            if self.failure is not None:
                raise self.failure
            return super().read_into(dst, timeout=timeout)

    buffers = [
        Buffer(torch.tensor([0], dtype=torch.uint8), first_failure),
        Buffer(torch.tensor([1], dtype=torch.uint8)),
        Buffer(torch.tensor([2], dtype=torch.uint8), later_failure),
    ]
    leaders = [
        {"latent": {
            "kind": "rdma",
            "dtype": "uint8",
            "shape": [1],
            "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
        }}
        for buffer in buffers
    ]

    with pytest.raises(FirstReadFailure) as exc_info:
        materialize_leader_samples(leaders, [0, 1, 2], 3, read_latent_result)

    assert exc_info.value is first_failure
    assert [buffer.reads for buffer in buffers] == [1, 1, 1]
    assert [buffer.dropped for buffer in buffers] == [0, 1, 0]
    poisoned = [owner.parts[0]["buffer"]
                for owner in transfer._RDMA_POISONED_OWNERS]
    assert poisoned == [buffers[0], buffers[2]]
    assert all(owner.keepalive is not None
               for owner in transfer._RDMA_POISONED_OWNERS)
    assert any("dp rank 2" in note and "LaterReadFailure: repr failed" in note
               for note in first_failure.__notes__)
    middle_destination = buffers[1].destination_ref
    assert middle_destination is not None
    gc.collect()
    assert middle_destination() is None


class _LeaderKeyboardInterrupt(KeyboardInterrupt):
    def __bool__(self):
        raise AssertionError("leader exception truthiness must not be evaluated")


class _LeaderSystemExit(SystemExit):
    def __bool__(self):
        raise AssertionError("leader exception truthiness must not be evaluated")


@pytest.mark.parametrize(
    "termination_type", (_LeaderKeyboardInterrupt, _LeaderSystemExit))
@pytest.mark.parametrize(
    "ordering", ("ordinary-termination", "termination-ordinary", "same-termination"))
def test_multi_dp_later_cancellation_wins_after_every_unique_reader(
    termination_type, ordering,
):
    ordinary = RuntimeError("ordinary leader read failed")
    termination = termination_type("leader read cancelled")
    failures = {
        "ordinary-termination": (ordinary, termination),
        "termination-ordinary": (termination, ordinary),
        "same-termination": (termination, termination),
    }[ordering]
    calls: list[int] = []

    def reader(descriptor):
        calls.append(descriptor)
        if descriptor < 2:
            raise failures[descriptor]
        return torch.tensor([descriptor], dtype=torch.uint8)

    with pytest.raises(termination_type) as exc_info:
        materialize_leader_samples(
            [{"latent": 0}, {"latent": 1}, {"latent": 2}],
            [0, 1, 2],
            3,
            reader,
        )

    assert exc_info.value is termination
    assert calls == [0, 1, 2]
    if ordering == "ordinary-termination":
        assert termination.__cause__ is ordinary
    assert termination.__cause__ is not termination
    assert termination.__context__ is not termination


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_LeaderKeyboardInterrupt, _LeaderSystemExit),
     (_LeaderSystemExit, _LeaderKeyboardInterrupt)),
)
def test_multi_dp_first_cancellation_wins_after_every_unique_reader(
    first_type, later_type,
):
    first = first_type("first leader read cancelled")
    later = later_type("later leader read cancelled")
    calls: list[int] = []

    def reader(descriptor):
        calls.append(descriptor)
        if descriptor == 0:
            raise first
        if descriptor == 1:
            raise later
        return torch.tensor([descriptor], dtype=torch.uint8)

    with pytest.raises(first_type) as exc_info:
        materialize_leader_samples(
            [{"latent": 0}, {"latent": 1}, {"latent": 2}],
            [0, 1, 2],
            3,
            reader,
        )

    assert exc_info.value is first
    assert calls == [0, 1, 2]
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    "target_text",
    ["if first_error is None:", "first_error = exc", "tensors.clear()"],
)
@pytest.mark.parametrize("truthiness", ["falsey", "explosive"])
def test_multi_dp_failure_bookkeeping_interruption_still_drains(
    target_text, truthiness,
):
    class TruthinessFailure(BaseException):
        pass

    class FirstFailure(BaseException):
        def __bool__(self):
            if truthiness == "explosive":
                raise TruthinessFailure("exception truthiness was consulted")
            return False

    class Boundary(BaseException):
        pass

    primary = FirstFailure("leader zero failed")
    boundary = Boundary(f"interrupted at {target_text}")
    calls = []

    def reader(desc):
        calls.append(desc)
        if desc == 0:
            raise primary
        return torch.tensor([desc], dtype=torch.uint8)

    source, start = inspect.getsourcelines(materialize_leader_samples)
    handler = next(
        offset for offset, line in enumerate(source)
        if "except BaseException as exc:" in line
    )
    target = next(
        start + offset for offset, line in enumerate(source[handler:], handler)
        if line.strip() == target_text
    )

    def interrupt(frame, event, _arg):
        if (event == "line"
                and frame.f_code is materialize_leader_samples.__code__
                and frame.f_lineno == target):
            sys.settrace(None)
            raise boundary
        return interrupt

    sys.settrace(interrupt)
    with pytest.raises(FirstFailure) as exc_info:
        try:
            materialize_leader_samples(
                [{"latent": 0}, {"latent": 1}, {"latent": 2}],
                [0, 1, 2], 3, reader)
        finally:
            sys.settrace(None)

    assert exc_info.value is primary
    assert calls == [0, 1, 2]


@pytest.mark.parametrize("termination_type", (_LeaderKeyboardInterrupt, _LeaderSystemExit))
@pytest.mark.parametrize(
    "target_text", ["if first_error is None:", "first_error = exc", "tensors.clear()"])
def test_multi_dp_bookkeeping_cancellation_outranks_ordinary_reader(
    termination_type, target_text,
):
    ordinary = RuntimeError("leader zero failed")
    termination = termination_type(f"interrupted at {target_text}")
    calls = []

    def reader(desc):
        calls.append(desc)
        if desc == 0:
            raise ordinary
        return torch.tensor([desc], dtype=torch.uint8)

    source, start = inspect.getsourcelines(materialize_leader_samples)
    handler = next(
        offset for offset, line in enumerate(source)
        if "except BaseException as exc:" in line
    )
    target = next(
        start + offset for offset, line in enumerate(source[handler:], handler)
        if line.strip() == target_text
    )

    def interrupt(frame, event, _arg):
        if (event == "line"
                and frame.f_code is materialize_leader_samples.__code__
                and frame.f_lineno == target):
            sys.settrace(None)
            raise termination
        return interrupt

    sys.settrace(interrupt)
    with pytest.raises(termination_type) as exc_info:
        try:
            materialize_leader_samples(
                [{"latent": 0}, {"latent": 1}, {"latent": 2}],
                [0, 1, 2], 3, reader)
        finally:
            sys.settrace(None)

    assert exc_info.value is termination
    assert termination.__cause__ is ordinary
    assert termination.__cause__ is not termination
    assert termination.__context__ is not termination
    assert calls == [0, 1, 2]


def test_multi_dp_success_publication_interruption_still_drains():
    class Boundary(BaseException):
        pass

    boundary = Boundary("interrupted after leader zero read")
    calls = []

    def reader(desc):
        calls.append(desc)
        return torch.tensor([desc], dtype=torch.uint8)

    source, start = inspect.getsourcelines(materialize_leader_samples)
    target = next(
        start + offset for offset, line in enumerate(source)
        if "tensors.append(value)" in line
    )

    def interrupt(frame, event, _arg):
        if (event == "line"
                and frame.f_code is materialize_leader_samples.__code__
                and frame.f_lineno == target):
            sys.settrace(None)
            raise boundary
        return interrupt

    sys.settrace(interrupt)
    with pytest.raises(Boundary) as exc_info:
        try:
            materialize_leader_samples(
                [{"latent": 0}, {"latent": 1}, {"latent": 2}],
                [0, 1, 2], 3, reader)
        finally:
            sys.settrace(None)

    assert exc_info.value is boundary
    assert calls == [0, 1, 2]
