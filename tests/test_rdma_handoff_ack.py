"""Handoff settlement: rdma_ownership.HandoffRegistry and mesh_lease.SetupBoundFuture."""
import threading

import pytest
import torch

import dgx_monarch.transfer as transfer
from dgx_monarch.mesh_lease import SetupBoundFuture
from dgx_monarch.nodes.latent_outputs import materialize_leader_samples
from dgx_monarch.rdma_ownership import HandoffRegistry
from dgx_monarch.transfer import read_latent_result
from transfer_helpers import (  # noqa: F401  # autouse fixture import.
    _FakeRDMABuffer,
    _isolate_process_lifetime_rdma_poison,
    _LoopRefusingBuffer,
)


@pytest.mark.parametrize("status", ["released", "already_released"])
def test_setup_bound_ack_accepts_exactly_one_owner(status):
    token = "2" * 32

    class Handle:
        world = 2

        def __init__(self):
            self.calls = []

        def call_all(self, endpoint, *args, timeout_s):
            self.calls.append((endpoint, args, timeout_s))
            return [
                {"setup_generation": 7, "token": token,
                 "status": status, "rank": 0},
                {"setup_generation": 7, "token": token,
                 "status": "unknown", "rank": 0},
            ]

    handle = Handle()
    SetupBoundFuture(None, handle, 7).ack_latent(7, token)
    assert handle.calls == [
        ("ack_latent_handoff", (7, token), 120.0)]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"setup_generation": 7, "token": "3" * 32, "status": "unknown"}],
        [
            {"setup_generation": 7, "token": "3" * 32, "status": "released"},
            {"setup_generation": 7, "token": "3" * 32, "status": "released"},
        ],
        [{"setup_generation": 7, "token": "3" * 32, "status": "invalid"}],
        [{"setup_generation": 8, "token": "3" * 32, "status": "released"}],
        [{"setup_generation": 7, "token": "4" * 32, "status": "released"}],
        [None],
    ],
)
def test_setup_bound_ack_rejects_zero_duplicate_or_malformed_owners(rows):
    class Handle:
        world = max(1, len(rows))

        def call_all(self, _endpoint, *_args, timeout_s):
            assert timeout_s == 120.0
            return rows

    with pytest.raises(RuntimeError, match="RDMA handoff ACK"):
        SetupBoundFuture(None, Handle(), 7).ack_latent(7, "3" * 32)


def test_setup_bound_ack_rejects_partial_response_with_one_visible_owner():
    class Handle:
        world = 2

        def call_all(self, _endpoint, *_args, timeout_s):
            assert timeout_s == 120.0
            return [{
                "setup_generation": 7, "token": "3" * 32,
                "status": "released", "rank": 0,
            }]

    with pytest.raises(RuntimeError, match="partial worker response"):
        SetupBoundFuture(None, Handle(), 7).ack_latent(7, "3" * 32)


@pytest.mark.parametrize("world", [True, 0, 2.0])
def test_setup_bound_ack_rejects_invalid_expected_worker_count(world):
    class Handle:
        def call_all(self, _endpoint, *_args, timeout_s):
            assert timeout_s == 120.0
            return [
                {"setup_generation": 7, "token": "3" * 32,
                 "status": "released", "rank": 0},
            ]

    handle = Handle()
    handle.world = world
    with pytest.raises(RuntimeError, match="invalid expected worker count"):
        SetupBoundFuture(None, handle, 7).ack_latent(7, "3" * 32)


def test_setup_bound_ack_rejects_descriptor_generation_before_rpc():
    class Handle:
        def call_all(self, *_args, **_kwargs):
            raise AssertionError("mismatched generation reached workers")

    with pytest.raises(RuntimeError, match="other than its sample lease"):
        SetupBoundFuture(None, Handle(), 7).ack_latent(8, "5" * 32)


def test_resource_bearing_settled_handoff_remains_registry_owned():
    registry = HandoffRegistry()
    handoff = {}
    registry.publish(handoff, 7, 1)
    token = handoff["token"]
    backing = torch.arange(4, dtype=torch.uint8)
    part = {"buffer": object(), "offset": 0, "nbytes": 4}
    handoff.update(parts=[part], keepalive=backing, state="settled")

    assert registry.reconcile(handoff) is False
    assert registry.get(7, token) is handoff
    assert handoff["parts"] == [part]
    assert handoff["keepalive"] is backing


def test_preconstruction_keepalive_does_not_consume_registry_capacity():
    """No part means native construction never received operation authority."""
    registry = HandoffRegistry()
    abandoned = {}
    registry.publish(abandoned, 7, 1)
    token = abandoned["token"]
    backing = object()
    abandoned.update(parts=[], keepalive=backing, state="registering")

    assert registry.live_count == 1
    assert registry.reconcile(abandoned) is True
    assert registry.get(7, token) is None
    assert abandoned["keepalive"] is None
    assert registry.live_count == 0

    successor = {}
    registry.publish(successor, 8, 1)
    assert registry.owns(successor)
    assert registry.live_count == 1


def test_postconstruction_empty_parts_keepalive_remains_fail_closed():
    registry = HandoffRegistry()
    handoff = {}
    registry.publish(handoff, 7, 1)
    token = handoff["token"]
    backing = object()
    handoff.update(parts=[], keepalive=backing, state="registered")

    assert registry.reconcile(handoff) is False
    assert registry.get(7, token) is handoff
    assert handoff["keepalive"] is backing
    assert registry.live_count == 1


def test_two_leader_broadcast_ack_retires_only_descriptor_owner():
    generation = 7
    registries = [HandoffRegistry(), HandoffRegistry()]
    handoffs, descriptors, buffers = [], [], []
    for index, registry in enumerate(registries):
        backing = torch.tensor([index], dtype=torch.uint8)
        buffer = _FakeRDMABuffer(backing)
        handoff = {}
        registry.publish(handoff, generation, 2)
        handoff.update(
            parts=[{"buffer": buffer, "offset": 0, "nbytes": 1}],
            keepalive=backing, state="registered")
        registry.mark_ready(handoff)
        handoffs.append(handoff)
        buffers.append(buffer)
        descriptors.append({
            "kind": "rdma", "dtype": "uint8", "shape": [1],
            "parts": handoff["parts"], "owner_token": handoff["token"],
            "setup_generation": generation,
        })
    assert handoffs[0]["token"] != handoffs[1]["token"]

    class Handle:
        world = 2
        lock = threading.RLock()
        sample_leases = {generation: 1}
        abandoned_sample_leases = {}
        deferred_supervision_error = None

        def __init__(self):
            self.calls = []

        def call_all(self, endpoint, setup_generation, token, timeout_s):
            assert endpoint == "ack_latent_handoff" and timeout_s == 120.0
            self.calls.append(token)
            return [
                {"setup_generation": setup_generation, "token": token,
                 "status": registry.acknowledge(setup_generation, token), "rank": 0}
                for registry in registries
            ]

    handle = Handle()
    guard = SetupBoundFuture(None, handle, generation)
    assert torch.equal(read_latent_result(descriptors[0], guard), torch.tensor([0]))
    assert handle.calls == [handoffs[0]["token"]]
    assert handoffs[0]["parts"] == [] and handoffs[0]["keepalive"] is None
    assert handoffs[1]["parts"] and handoffs[1]["keepalive"] is not None
    assert buffers[0].dropped == 1 and buffers[1].dropped == 0

    assert torch.equal(read_latent_result(descriptors[1], guard), torch.tensor([1]))
    assert handle.calls == [handoffs[0]["token"], handoffs[1]["token"]]
    assert all(not handoff["parts"] and handoff["keepalive"] is None
               for handoff in handoffs)
    assert [buffer.dropped for buffer in buffers] == [1, 1]


def test_applied_but_lost_ack_response_retries_real_registry_seam():
    import asyncio

    generation = 8
    token_registry = HandoffRegistry()
    nonowner_registry = HandoffRegistry()
    backing = torch.arange(4, dtype=torch.uint8)
    buffer = _LoopRefusingBuffer(backing)
    handoff = {}
    token_registry.publish(handoff, generation, 1)
    handoff.update(
        parts=[{"buffer": buffer, "offset": 0, "nbytes": 4}],
        keepalive=backing, state="registered")
    token_registry.mark_ready(handoff)
    token = handoff["token"]

    class Handle:
        world = 2
        lock = threading.RLock()
        sample_leases = {generation: 1}
        abandoned_sample_leases = {}
        deferred_supervision_error = None

        def __init__(self):
            self.broadcasts = 0
            self.threads = []

        def call_all(self, _endpoint, setup_generation, owner_token, timeout_s):
            self.broadcasts += 1
            self.threads.append(threading.current_thread().name)
            rows = [
                {"setup_generation": setup_generation, "token": owner_token,
                 "status": registry.acknowledge(setup_generation, owner_token),
                 "rank": 0}
                for registry in (token_registry, nonowner_registry)
            ]
            if self.broadcasts == 1:
                assert [row["status"] for row in rows] == ["released", "unknown"]
                raise RuntimeError("ACK response lost after remote apply")
            assert [row["status"] for row in rows] == [
                "already_released", "unknown"]
            return rows

    handle = Handle()
    guard = SetupBoundFuture(None, handle, generation)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [4],
        "parts": handoff["parts"], "owner_token": token,
        "setup_generation": generation,
    }

    async def on_loop():
        return read_latent_result(desc, guard)

    assert torch.equal(asyncio.run(on_loop()), backing)
    assert handle.broadcasts == 2
    assert handle.threads == ["dgxm-latent-read", "dgxm-latent-read"]
    assert buffer.dropped == 1
    assert not handoff["parts"] and handoff["keepalive"] is None
    assert token_registry.get(generation, token)["_registry_state"] == "RELEASED"


def test_applied_ack_cancellation_retries_settlement_but_propagates():
    generation = 9
    primary = KeyboardInterrupt("cancelled after ACK response")
    token_registry = HandoffRegistry()
    nonowner_registry = HandoffRegistry()
    backing = torch.arange(4, dtype=torch.uint8)
    buffer = _FakeRDMABuffer(backing)
    handoff = {}
    token_registry.publish(handoff, generation, 1)
    handoff.update(
        parts=[{"buffer": buffer, "offset": 0, "nbytes": 4}],
        keepalive=backing, state="registered")
    token_registry.mark_ready(handoff)
    token = handoff["token"]

    class Handle:
        world = 2
        lock = threading.RLock()
        sample_leases = {generation: 1}
        abandoned_sample_leases = {}
        deferred_supervision_error = None

        def __init__(self):
            self.calls = []

        def call_all(self, _endpoint, setup_generation, owner_token, timeout_s):
            rows = [
                {"setup_generation": setup_generation, "token": owner_token,
                 "status": registry.acknowledge(setup_generation, owner_token),
                 "rank": 0}
                for registry in (token_registry, nonowner_registry)
            ]
            self.calls.append([row["status"] for row in rows])
            if len(self.calls) == 1:
                raise primary
            return rows

    handle = Handle()
    guard = SetupBoundFuture(None, handle, generation)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [4],
        "parts": handoff["parts"], "owner_token": token,
        "setup_generation": generation,
    }
    with pytest.raises(KeyboardInterrupt) as exc_info:
        read_latent_result(desc, guard)
    assert exc_info.value is primary
    assert handle.calls == [
        ["released", "unknown"], ["already_released", "unknown"]]
    assert buffer.dropped == 1
    assert not handoff["parts"] and handoff["keepalive"] is None
    assert token_registry.get(generation, token)["_registry_state"] == "RELEASED"


def test_ack_retry_ordinary_then_applied_cancellation_keeps_exact_identity():
    generation = 12
    ordinary = RuntimeError("first ACK failed before apply")
    cancellation = KeyboardInterrupt("second ACK cancelled after apply")
    registry = HandoffRegistry()
    backing = torch.arange(4, dtype=torch.uint8)
    buffer = _FakeRDMABuffer(backing)
    handoff = {}
    registry.publish(handoff, generation, 1)
    handoff.update(
        parts=[{"buffer": buffer, "offset": 0, "nbytes": 4}],
        keepalive=backing, state="registered")
    registry.mark_ready(handoff)
    token = handoff["token"]

    class Handle:
        world = 1

        def __init__(self):
            self.lock = threading.RLock()
            self.sample_leases = {generation: 1}
            self.abandoned_sample_leases = {}
            self.deferred_supervision_error = None

        def call_all(self, _endpoint, setup_generation, owner_token, timeout_s):
            return [{
                "setup_generation": setup_generation,
                "token": owner_token,
                "status": registry.acknowledge(setup_generation, owner_token),
                "rank": 0,
            }]

    class Guard(SetupBoundFuture):
        def __init__(self, *args):
            super().__init__(*args)
            self.ack_readers = []

        def ack_latent(self, setup_generation, owner_token):
            self.ack_readers.append(self.readers)
            if len(self.ack_readers) == 1:
                raise ordinary
            super().ack_latent(setup_generation, owner_token)
            raise cancellation

    handle = Handle()
    guard = Guard(None, handle, generation)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [4],
        "parts": [dict(handoff["parts"][0])], "owner_token": token,
        "setup_generation": generation,
    }
    with pytest.raises(KeyboardInterrupt) as exc_info:
        read_latent_result(desc, guard)

    assert exc_info.value is cancellation
    assert exc_info.value.__cause__ is ordinary
    assert guard.ack_readers == [1, 1] and guard.readers == 0
    assert buffer.dropped == 1
    assert registry.get(generation, token)["_registry_state"] == "RELEASED"
    assert registry.live_count == 0 and registry.tombstone_count == 1
    assert registry.acknowledge(generation, token) == "already_released"
    assert buffer.dropped == 1


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_ack_retry_reused_exception_never_self_chains(error_type):
    error = error_type("same ACK failure object")
    calls = 0

    def fail_ack(*_identity):
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(error_type) as exc_info:
        transfer.rdma_ownership.acknowledge_with_retry(
            fail_ack, 12, "a" * 32)

    assert exc_info.value is error and calls == 2
    assert error.__cause__ is not error
    assert error.__context__ is not error


def test_failed_leader_is_not_acked_while_later_leader_is_settled():
    primary = RuntimeError("leader zero read failed")
    generation = 9

    class FailingBuffer(_FakeRDMABuffer):
        def read_into(self, _dst, timeout=None):
            raise primary

    registries = [HandoffRegistry(), HandoffRegistry()]
    handoffs, buffers, leaders = [], [], []
    for index, registry in enumerate(registries):
        backing = torch.tensor([index], dtype=torch.uint8)
        buffer = (FailingBuffer(backing) if index == 0
                  else _FakeRDMABuffer(backing))
        handoff = {}
        registry.publish(handoff, generation, 2)
        handoff.update(
            parts=[{"buffer": buffer, "offset": 0, "nbytes": 1}],
            keepalive=backing, state="registered")
        registry.mark_ready(handoff)
        handoffs.append(handoff)
        buffers.append(buffer)
        leaders.append({"latent": {
            "kind": "rdma", "dtype": "uint8", "shape": [1],
            "parts": handoff["parts"], "owner_token": handoff["token"],
            "setup_generation": generation,
        }})

    class Handle:
        world = 2
        lock = threading.RLock()
        sample_leases = {generation: 1}
        abandoned_sample_leases = {}
        deferred_supervision_error = None

        def __init__(self):
            self.calls = []

        def call_all(self, _endpoint, setup_generation, token, timeout_s):
            self.calls.append(token)
            return [
                {"setup_generation": setup_generation, "token": token,
                 "status": registry.acknowledge(setup_generation, token), "rank": 0}
                for registry in registries
            ]

    handle = Handle()
    guard = SetupBoundFuture(None, handle, generation)
    with pytest.raises(RuntimeError) as exc_info:
        materialize_leader_samples(leaders, [0, 1], 2,
                                   lambda desc: read_latent_result(desc, guard))
    assert exc_info.value is primary
    assert handle.calls == [handoffs[1]["token"]]
    assert registries[0].get(generation, handoffs[0]["token"]) is handoffs[0]
    assert handoffs[0]["parts"] and handoffs[0]["keepalive"] is not None
    assert not handoffs[1]["parts"] and handoffs[1]["keepalive"] is None
    assert [buffer.dropped for buffer in buffers] == [0, 1]


def test_first_leader_ack_failure_still_settles_every_later_leader():
    primary = RuntimeError("first leader ACK failed")
    retry = RuntimeError("first leader ACK retry failed")
    generation = 10
    registries = [HandoffRegistry(), HandoffRegistry()]
    handoffs, buffers, leaders = [], [], []
    for index, registry in enumerate(registries):
        backing = torch.tensor([index], dtype=torch.uint8)
        buffer = _FakeRDMABuffer(backing)
        handoff = {}
        registry.publish(handoff, generation, 2)
        handoff.update(
            parts=[{"buffer": buffer, "offset": 0, "nbytes": 1}],
            keepalive=backing, state="registered")
        registry.mark_ready(handoff)
        handoffs.append(handoff)
        buffers.append(buffer)
        leaders.append({"latent": {
            "kind": "rdma", "dtype": "uint8", "shape": [1],
            "parts": handoff["parts"], "owner_token": handoff["token"],
            "setup_generation": generation,
        }})

    class Handle:
        world = 2
        lock = threading.RLock()
        sample_leases = {generation: 1}
        abandoned_sample_leases = {}
        deferred_supervision_error = None

        def __init__(self):
            self.calls = []

        def call_all(self, _endpoint, setup_generation, token, timeout_s):
            self.calls.append(token)
            if token == handoffs[0]["token"]:
                raise primary if self.calls.count(token) == 1 else retry
            return [
                {"setup_generation": setup_generation, "token": token,
                 "status": registry.acknowledge(setup_generation, token), "rank": 0}
                for registry in registries
            ]

    handle = Handle()
    guard = SetupBoundFuture(None, handle, generation)
    with pytest.raises(RuntimeError) as exc_info:
        materialize_leader_samples(leaders, [0, 1], 2,
                                   lambda desc: read_latent_result(desc, guard))
    assert exc_info.value is primary
    assert handle.calls == [
        handoffs[0]["token"], handoffs[0]["token"], handoffs[1]["token"]]
    assert handoffs[0]["parts"] and handoffs[0]["keepalive"] is not None
    assert not handoffs[1]["parts"] and handoffs[1]["keepalive"] is None
    assert [buffer.dropped for buffer in buffers] == [1, 1]
