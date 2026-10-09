"""One crashed render must not fail every queued job behind it (docs/TROUBLESHOOTING.md #87).

The heal authorizes a single observed recycle-and-retry when the only
dispatch obstruction is abandoned leases; every other busy state re-raises.
The retry also has to leave nothing behind: it is the one production path
that stamps a single render twice, each time into a fresh waiver-audit slot.
"""
from __future__ import annotations

import pytest

from dgx_monarch import mesh_setup
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.nodes import consent_waiver, recycle_drain, render_submit
from dgx_monarch.nodes.pending import PendingRenderHandoff
from dgx_monarch.nodes.submit_guard import SubmitRenderGuard
from render_sessions_helpers import _real_handle, _stub_direct_submit


class _Outcome:
    def __init__(self, ok):
        self.ok = ok
        self.status = type("S", (), {"value": "recycled" if ok else "blocked"})()
        self.detail = "probe"


def _handle(abandoned=0, active=0, recycle_ok=True, recycles=None):
    handle = MeshHandle.__new__(MeshHandle)
    handle.abandoned_sample_leases = {1: abandoned} if abandoned else {}
    handle.sample_leases = {1: active} if active else {}
    if recycles is None:
        recycles = []
    handle.recycle_detailed = lambda: recycles.append(1) or _Outcome(recycle_ok)
    handle._recycles = recycles
    return handle


class _Spec:
    def __init__(self, handle):
        self.handle = handle


class _Model:
    def __init__(self, handle):
        self.mesh = _Spec(handle)


@pytest.fixture(autouse=True)
def _no_pending_reads(monkeypatch):
    import dgx_monarch.rdma_read_job as rdma_read_job

    monkeypatch.setattr(
        rdma_read_job, "pending_job_count_for_handle", lambda handle: 0)


@pytest.fixture(autouse=True)
def _empty_audit_window():
    """The dispatch join window is process-wide, so every test starts it empty."""
    consent_waiver.reset_pending_audit()
    yield
    consent_waiver.reset_pending_audit()


def test_abandoned_only_wedge_heals_with_one_recycle():
    handle = _handle(abandoned=2)
    exc = mesh_setup.LifecycleBusyError("cannot dispatch")
    assert render_submit._heal_abandoned_wedge_once(_Model(handle), exc) is True
    assert handle._recycles == [1]


def test_live_lease_is_never_destroyed():
    handle = _handle(abandoned=1, active=1)
    exc = mesh_setup.LifecycleBusyError("cannot dispatch")
    assert render_submit._heal_abandoned_wedge_once(_Model(handle), exc) is False
    assert handle._recycles == []


def test_pending_rdma_read_blocks_the_heal(monkeypatch):
    import dgx_monarch.rdma_read_job as rdma_read_job

    monkeypatch.setattr(
        rdma_read_job, "pending_job_count_for_handle", lambda handle: 1)
    handle = _handle(abandoned=1)
    exc = mesh_setup.LifecycleBusyError("cannot dispatch")
    assert render_submit._heal_abandoned_wedge_once(_Model(handle), exc) is False
    assert handle._recycles == []


def test_no_abandonment_means_no_heal():
    handle = _handle(abandoned=0)
    exc = mesh_setup.LifecycleBusyError("cannot dispatch")
    assert render_submit._heal_abandoned_wedge_once(_Model(handle), exc) is False


def test_failed_recycle_keeps_the_original_refusal():
    handle = _handle(abandoned=1, recycle_ok=False)
    exc = mesh_setup.LifecycleBusyError("cannot dispatch")
    assert render_submit._heal_abandoned_wedge_once(_Model(handle), exc) is False
    assert handle._recycles == [1]


def test_healed_retry_reuses_the_adoption_evidence(monkeypatch):
    """The claim is one-shot: a healed retry must reuse the consumed context,
    never re-claim or re-consume."""
    consumed, attempts = [], []
    monkeypatch.setattr(
        render_submit, "_claim_adoption_context", lambda: "claim-token")
    monkeypatch.setattr(
        render_submit, "_consume_adoption_context_claim",
        lambda claim: consumed.append(claim) or {"evidence": "wire"})

    def fake_guarded(*args, **kwargs):
        attempts.append(kwargs["adoption_context"])
        if len(attempts) == 1:
            raise mesh_setup.LifecycleBusyError("cannot dispatch")
        return "pending"

    monkeypatch.setattr(render_submit, "_submit_render_guarded", fake_guarded)
    monkeypatch.setattr(
        render_submit, "run_guarded_submit",
        lambda guard, body, on_failure: body())
    monkeypatch.setattr(
        render_submit, "_heal_abandoned_wedge_once", lambda model, exc: True)
    out = render_submit.submit_render(
        object(), {}, {}, 1.0, 8, handoff=None)
    assert out == "pending"
    assert consumed == ["claim-token"]  # exactly one consume across the heal
    assert attempts == [{"evidence": "wire"}, {"evidence": "wire"}]


def test_non_mesh_handle_is_ignored():
    class _Foreign:
        mesh = _Spec(object())

    exc = mesh_setup.LifecycleBusyError("cannot dispatch")
    assert render_submit._heal_abandoned_wedge_once(_Foreign(), exc) is False


def test_a_healed_retry_frees_the_first_attempt_s_waiver_audit_slot(monkeypatch):
    """The abandoned attempt must free the audit slot its stamp took.

    A dispatch takes its slot when the request is stamped and PendingRender,
    which owns every later retirement, is built at the bottom of the same
    call. The heal crosses that window twice for one queued job: the first
    attempt stamps, fails inside submit_sample, and the retry mints a fresh
    render id that can never join the first slot. Held slots are never
    evicted, so a leaked one is held for the life of the ComfyUI process and
    512 of them refuse every later waived dispatch.
    """
    handle = _real_handle()
    handle.abandoned_sample_leases = {1: 1}
    handle.sample_leases = {}
    handle.recycle_detailed = lambda: _Outcome(True)
    stamped: list[str] = []

    def stamp_request(request, model, topo, world, render_id):
        stamped.append(render_id)
        consent_waiver._PENDING_AUDIT[render_id] = {"combo_key": "c"}
        return 1

    def submit_sample(*_args, **_kwargs):
        if len(stamped) == 1:
            raise mesh_setup.LifecycleBusyError("cannot dispatch")
        return object()

    spec = _stub_direct_submit(monkeypatch, handle, submit_sample)
    monkeypatch.setattr(consent_waiver, "stamp_request", stamp_request)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)

    pending = render_submit.submit_render(
        spec, {}, {"samples": object()}, 1.0, 2, handoff=PendingRenderHandoff())

    assert len(stamped) == 2 and stamped[0] != stamped[1]
    assert list(consent_waiver._PENDING_AUDIT) == [stamped[1]]
    pending.abandon()
    assert consent_waiver._PENDING_AUDIT == {}


def test_the_guard_retires_a_stamped_slot_only_while_it_still_owns_it():
    """Ownership passes to PendingRender at publication, not before or after.

    An untransferred submit with no PendingRender is the only owner left, so
    it retires. A published one leaves the slot to PendingRender._close, which
    is also the path that writes the mandatory use row first.
    """
    abandoned = SubmitRenderGuard()
    abandoned.render_id = "r-abandoned"
    consent_waiver._PENDING_AUDIT["r-abandoned"] = {"combo_key": "c"}
    abandoned.cleanup(None)
    assert "r-abandoned" not in consent_waiver._PENDING_AUDIT

    transferred = SubmitRenderGuard()
    transferred.render_id = "r-published"
    transferred.transferred = True
    consent_waiver._PENDING_AUDIT["r-published"] = {"combo_key": "c"}
    transferred.cleanup(None)
    assert "r-published" in consent_waiver._PENDING_AUDIT


def test_a_guard_holding_a_pending_render_leaves_the_slot_to_its_abandon():
    """One retirement per slot: the pending branch already owns this one."""
    calls: list[str] = []
    guard = SubmitRenderGuard()
    guard.render_id = "r-pending"
    guard.pending = type(
        "Pending", (), {"abandon": lambda _self: calls.append("abandon")})()
    consent_waiver._PENDING_AUDIT["r-pending"] = {"combo_key": "c"}

    guard.cleanup(None)

    assert calls == ["abandon"]
    assert "r-pending" in consent_waiver._PENDING_AUDIT
