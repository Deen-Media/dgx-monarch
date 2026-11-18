"""A rank's reply the driver cannot read is named like any other failed rank.

torchmonarch decodes each rank's reply in a ValueMesh only when ``items()``
reaches it (0.7.0 and 0.8.0 alike: ``call().get()`` returns, and the read
raises). ``mesh_rpc.call_all_bound`` read the replies after its error handling,
so such a failure reached the user as a bare unpickling error, with no box and
none of the handling a failed call gets. It now reads them inside: the error
says which endpoint's reply, the box it came from leads, and an ambiguous
mutation's latch clears as for any confirmed failure (the call did complete).
"""
from __future__ import annotations

import threading
import types

import pytest

from dgx_monarch import driver_footprint, mesh_rpc, mesh_setup
from dgx_monarch.refusal import read_rank


class _Unpicklable(RuntimeError):
    """What unpickling a reply raises on the driver, for example a missing class."""


class _LazyValueMesh:
    """torchmonarch's ValueMesh shape: items() decodes each rank as it reaches it."""

    def __init__(self, replies):
        self._replies = replies

    def items(self):
        for rank, reply in enumerate(self._replies):
            if isinstance(reply, BaseException):
                raise reply
            yield types.SimpleNamespace(rank=rank), reply


def _pair(monkeypatch, value_mesh):
    monkeypatch.setattr(driver_footprint, "host_is_local", lambda name: name == "box-a")
    handle = types.SimpleNamespace(
        world=2, gpus_per_host=1, owns_hosts=True, active_worker_args={},
        lock=threading.RLock(), setup_generation=3, setup_cleanup_state=None,
        config=types.SimpleNamespace(hosts=tuple(
            types.SimpleNamespace(name=name, address=f"tcp://{name}:26600")
            for name in ("box-a", "box-b"))))
    handle.workers = types.SimpleNamespace(
        load_model=types.SimpleNamespace(call=lambda *_a, **_k: object()),
        unload=types.SimpleNamespace(call=lambda *_a, **_k: object()))
    handle._await_or_evict = lambda _future, _timeout: value_mesh
    handle._latch_ambiguous_mutation = lambda *_a: pytest.fail("a completed call is not ambiguous")
    monkeypatch.setattr(mesh_setup, "dispatch_endpoint",
                        lambda _handle, _name, send, setup_token=None: send())
    return handle


def test_an_unreadable_reply_names_its_box_and_its_endpoint(monkeypatch):
    cause = _Unpicklable("No module named 'missing_on_the_driver'")
    handle = _pair(monkeypatch, _LazyValueMesh([{"rank": 0}, cause]))
    monkeypatch.setattr(mesh_setup, "AMBIGUOUS_MUTATION_ENDPOINTS", frozenset())
    with pytest.raises(RuntimeError, match="reply could not be read") as caught:
        mesh_rpc.call_all_bound(handle, "unload")
    text = str(caught.value)
    assert text.startswith("On the sibling, the unload reply could not be read on the driver"), text
    assert "_Unpicklable: No module named 'missing_on_the_driver'" in text
    assert read_rank(text) == (text, None), "the rank mark is taken off once the box is named"
    assert caught.value.__cause__ is cause


def test_rank_zeros_unreadable_reply_names_the_head(monkeypatch):
    handle = _pair(monkeypatch, _LazyValueMesh([_Unpicklable("truncated"), {"rank": 1}]))
    monkeypatch.setattr(mesh_setup, "AMBIGUOUS_MUTATION_ENDPOINTS", frozenset())
    with pytest.raises(RuntimeError, match="reply could not be read") as caught:
        mesh_rpc.call_all_bound(handle, "unload")
    assert str(caught.value).startswith("On the head, the unload reply could not be read"), caught.value


def test_a_completed_mutation_whose_reply_is_unreadable_clears_its_latch(monkeypatch):
    handle = _pair(monkeypatch, _LazyValueMesh([{"rank": 0}, _Unpicklable("bad bytes")]))
    assert "unload" in mesh_setup.AMBIGUOUS_MUTATION_ENDPOINTS
    with pytest.raises(RuntimeError, match="reply could not be read"):
        mesh_rpc.call_all_bound(handle, "unload")
    assert handle.setup_cleanup_state is None


def test_readable_replies_are_returned_in_rank_order(monkeypatch):
    handle = _pair(monkeypatch, _LazyValueMesh([{"rank": 0}, {"rank": 1}]))
    monkeypatch.setattr(mesh_setup, "AMBIGUOUS_MUTATION_ENDPOINTS", frozenset())
    assert mesh_rpc.call_all_bound(handle, "unload") == [{"rank": 0}, {"rank": 1}]


class _Future:
    def __init__(self, value):
        self._value = value

    def get(self, timeout=None):
        return self._value


def test_an_unreadable_render_reply_names_its_box_and_keeps_the_fleet(monkeypatch):
    """The render's own collect (mesh_lease.collect_sample) reads its replies the same way.

    A reply that cannot be read came from a call that completed, so it is no
    lost fleet: the eviction never sees it.
    """
    from dgx_monarch import mesh_helpers, mesh_lease

    cause = _Unpicklable("No module named 'missing_on_the_driver'")
    handle = _pair(monkeypatch, None)
    evictions: list = []
    real = mesh_helpers.mark_defunct_preserving_primary
    monkeypatch.setattr(mesh_helpers, "mark_defunct_preserving_primary",
                        lambda h, exc, **kw: (evictions.append(exc), real(h, exc, **kw)))
    with pytest.raises(RuntimeError, match="reply could not be read") as caught:
        mesh_lease.collect_sample(handle, _Future(_LazyValueMesh([{"rank": 0}, cause])), 30.0)
    text = str(caught.value)
    assert text.startswith("On the sibling, the sample reply could not be read on the driver"), text
    assert read_rank(text) == (text, None) and caught.value.__cause__ is cause
    assert evictions == [] and not getattr(handle, "defunct", False)
    replies = mesh_lease.collect_sample(handle, _Future(_LazyValueMesh([{"rank": 0}, {"rank": 1}])), 30.0)
    assert replies == [{"rank": 0}, {"rank": 1}]


def test_an_interrupt_while_reading_is_not_reworded():
    class _Interrupting:
        def items(self):
            raise KeyboardInterrupt
            yield  # pragma: no cover - makes this a generator

    with pytest.raises(KeyboardInterrupt):
        mesh_rpc.read_values(_Interrupting(), "unload")
