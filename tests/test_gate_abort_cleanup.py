"""Every identity-gate abort path must reach a clean, healable state.

Two halves of one failure, seen on 2026-09-01. A cfg-parallel proof render
refuses on the worker with an untagged error, so the driver abandons its sample
lease; the gate then pushes its stock quarantine, the driver refuses that push
before any RPC leaves the process, and the gate latches the mesh DIRTY for a
call no worker ever saw. The first six tests pin the ceremony half, the next
two the raise site, and the last three how the FSDP proof reads a settled
refusal.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch import mesh, mesh_safety, mesh_setup
from dgx_monarch.mesh import MeshAttachError
from dgx_monarch.nodes import gate_identity as gate_identity_mod
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from topology_transition_helpers import (
    _Endpoint,
    _Future,
    _handle,
    _ValueMesh,
    _Workers,
)


def _model(handle):
    return ModelSpec(
        mesh=MeshSpec(
            handle=handle,
            topology_preset="cfg2",
            attention="SAGE_AUTO",
            sync_ulysses=False,
            worker_args={"lora_low_rss": True, "slab_weights": True},
        ),
        unet_name="model.safetensors",
    )


def _wedged_handle():
    """A handle as an aborted cfg-parallel proof render leaves it."""
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    handle.abandoned_sample_leases = {1: 1}
    return handle, workers


def test_a_pre_dispatch_quarantine_refusal_leaves_the_mesh_clean():
    """The gate must not publish a DIRTY verdict for a call it never sent.

    ``MeshHandle.apply_worker_args`` runs four driver-side guards before its
    first state store and its first send, so their refusals carry no
    ambiguity. Latching one turns a refusal the abandoned-lease heal
    (docs/TROUBLESHOOTING.md #87) can clear into a MeshAttachError only a
    manual recycle clears.
    """
    handle, workers = _wedged_handle()

    with pytest.raises(mesh_setup.SampleResultBusyError):
        gate_identity_mod.force_stock_quarantine(_model(handle), handle)

    assert workers.apply_worker_args.calls == []
    assert handle.setup_cleanup_state is None
    assert handle.worker_args_key is None and handle.active_worker_args == {}


def test_the_render_after_a_clean_abort_is_healable_rather_than_wedged():
    """ensure_live must pass, so the next submit meets the healable refusal."""
    handle, _workers = _wedged_handle()

    with pytest.raises(mesh_setup.SampleResultBusyError):
        gate_identity_mod.force_stock_quarantine(_model(handle), handle)

    assert mesh_safety.coherent_lifecycle_verdict(handle, MeshAttachError) == "live"
    # The refusal the next submit meets is the one the abandoned-lease heal catches.
    with pytest.raises(mesh_setup.LifecycleBusyError):
        handle.ensure_setup(handle.topology, "TORCH_FLASH", True, {})


def test_a_dispatched_quarantine_of_unknown_outcome_still_latches_dirty():
    """The guard stays intact where the outcome is unknown."""
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(
        _Future(error=TimeoutError("policy completion unknown")))
    handle, _old, _new = _handle(workers)

    with pytest.raises(TimeoutError):
        gate_identity_mod.force_stock_quarantine(_model(handle), handle)

    assert handle.setup_cleanup_state is not None
    assert mesh_safety.coherent_lifecycle_verdict(
        handle, MeshAttachError) == "unresolved"


def test_a_dispatched_quarantine_keeps_the_phase_the_rpc_published():
    """The verdict a dispatch published for itself survives the gate.

    ``MeshHandle.apply_worker_args`` names the phase an operator has to act on
    for everything it sent. The gate rewriting that with its own, vaguer phase
    loses the first cause and buys no safety, because the mesh reads dirty
    either way.
    """
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(
        _Future(error=TimeoutError("policy completion unknown")))
    handle, _old, _new = _handle(workers)

    with pytest.raises(TimeoutError):
        gate_identity_mod.force_stock_quarantine(_model(handle), handle)

    state = handle.setup_cleanup_state
    assert state is not None
    assert state.outcome is mesh_setup.SetupCleanupOutcome.TIMEOUT_UNKNOWN
    assert state.phase == "apply_worker_args RPC completion"


def test_a_dirty_mesh_keeps_the_first_cause_that_marked_it():
    """An exemption may skip a write. It may never clear or reword a latch.

    A handle already dirty from a lost proc stop refuses the push at the same
    preflight, so the exempt arm runs on a mesh whose state names a different
    cause. Rewriting it would cost the operator the fault they have to act on.
    """
    handle, _workers = _wedged_handle()
    first = mesh_setup.cleanup_failure(
        int(handle.setup_generation), "topology teardown", 600.0,
        RuntimeError("proc stop outcome lost"))
    handle.setup_cleanup_state = first

    with pytest.raises(mesh_setup.TopologyTransitionError):
        gate_identity_mod.force_stock_quarantine(_model(handle), handle)

    assert handle.setup_cleanup_state is first
    with pytest.raises(MeshAttachError):
        mesh.ensure_live(handle)


def test_the_abandoned_lease_guard_still_refuses_a_setup_change():
    """A clean mesh does not mean a permissive one."""
    handle, _workers = _wedged_handle()
    from dgx_monarch.topology import Topology

    with pytest.raises(mesh_setup.SampleResultBusyError, match="abandoned"):
        handle.ensure_setup(Topology(ring=2, world=2), "TORCH_FLASH", True, {})


def _batch_one_wrapper(monkeypatch):
    from types import SimpleNamespace

    from dgx_monarch.adapters import cfg_parallel

    monkeypatch.setattr(cfg_parallel, "cfg_world", lambda: 2)
    monkeypatch.setattr(cfg_parallel, "cfg_rank", lambda: 0)
    adapter = SimpleNamespace(family="flux1")
    wrapper = cfg_parallel.make_cfg_parallel_wrapper(adapter)

    def original(x, timestep=None):
        raise AssertionError("the guard must refuse before the forward runs")

    return wrapper, SimpleNamespace(original=original)


def test_the_cfg_split_refusal_carries_its_class_p_tag(monkeypatch):
    """A guard decided from replicated request facts must say so in its text."""
    from dgx_monarch.refusal import parse_leading_refusal_tag

    wrapper, executor = _batch_one_wrapper(monkeypatch)

    with pytest.raises(Exception) as caught:
        wrapper(executor, torch.zeros(1, 4, 8, 8))

    assert "does not split into 2 equal slices" in str(caught.value)
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class.value == "P"


def test_the_cfg_split_refusal_consumes_its_sample_lease(monkeypatch):
    """A request-decided, rank-symmetric refusal ends the render completely."""
    from monarch.actor import ActorError

    from dgx_monarch.nodes.pending import typed_worker_refusal

    wrapper, executor = _batch_one_wrapper(monkeypatch)
    with pytest.raises(Exception) as caught:
        wrapper(executor, torch.zeros(1, 4, 8, 8))

    assert typed_worker_refusal(ActorError(caught.value)) is True


def test_the_cfg_split_refusal_reads_as_a_settled_refusal(monkeypatch):
    """The FSDP proof must recognise this card as an answer.

    The proof leg meets it wrapped in an ActorError. Reading its class there is
    what stops the ceremony rewriting a class P answer into an untyped abort.
    """
    from monarch.actor import ActorError

    from dgx_monarch.nodes.gate_fsdp import settled_refusal_tag

    wrapper, executor = _batch_one_wrapper(monkeypatch)
    with pytest.raises(Exception) as caught:
        wrapper(executor, torch.zeros(1, 4, 8, 8))

    tag = settled_refusal_tag(ActorError(caught.value))
    assert tag is not None and tag.refusal_class.value == "P"
    assert tag.waivable is False


def test_the_divisibility_pad_card_reads_as_a_waivable_class_k_refusal():
    """The card whose waiver a ceremony always strips."""
    from monarch.actor import ActorError

    from dgx_monarch.adapters.base import assert_ulysses_only_padding
    from dgx_monarch.nodes.gate_fsdp import settled_refusal_tag

    with pytest.raises(Exception) as caught:
        assert_ulysses_only_padding(2, 1)

    tag = settled_refusal_tag(ActorError(caught.value))
    assert tag is not None and tag.refusal_class.value == "K"
    assert tag.waivable is True and tag.guard == "ring_pad"


def test_the_abort_refusal_itself_is_not_a_settled_refusal():
    """The proof's own untyped abort must never read as somebody's answer."""
    from dgx_monarch.nodes.gate_fsdp import (
        FsdpGateProofError,
        settled_refusal_tag,
    )

    assert settled_refusal_tag(FsdpGateProofError(
        "FSDP clean-reload proof aborted before a terminal verdict: boom"
    )) is None
