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
from dgx_monarch.nodes import (
    abandoned_heal,
    common,
    consent_waiver,
    recycle_drain,
    render_submit,
)
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


class _Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def info(self, message, *args):
        self.lines.append(("info", message % args))

    def warning(self, message, *args):
        self.lines.append(("warning", message % args))


def _fleet_handle(name, events, *, abandoned=0):
    """A direct-submit handle whose recycle retires it, as the real one does.

    The retired handle keeps its abandoned count, and a second recycle answers
    ok from the already-stopped branch.
    """
    handle = _real_handle()
    handle.name = name
    handle.world = 1
    handle.cancel_sample = lambda *_args, **_kwargs: None
    handle.sample_leases = {}
    handle.abandoned_sample_leases = {1: abandoned} if abandoned else {}

    def recycle_detailed():
        events.append(("recycle", name))
        handle.defunct = True
        handle.teardown_complete = True
        return _Outcome(True)

    handle.recycle_detailed = recycle_detailed
    return handle


def _wire_session(monkeypatch, events, init, spares):
    """Direct-submit seams plus a mesh cache: ensure_live answers the cached
    fleet and builds the next spare once the cached one is recycled."""

    def submit_sample(handle, *_args, **_kwargs):
        # The check _dispatch_sample_bound makes before it leases a sample.
        mesh_setup.require_no_abandoned_samples(handle, "dispatch another sample")
        events.append(("submit", handle.name))
        return object()

    spec = _stub_direct_submit(monkeypatch, init, submit_sample)
    cache = {"live": init}

    def ensure_live(handle):
        if cache["live"].defunct:
            cache["live"] = spares.pop(0)
        events.append(("ensure_live", handle.name, cache["live"].name))
        return cache["live"]

    log = _Log()
    monkeypatch.setattr(common, "ensure_live", ensure_live)
    monkeypatch.setattr(abandoned_heal, "log", log)
    monkeypatch.setattr(
        recycle_drain, "drop_residency_memos",
        lambda: events.append(("drop_memos",)))
    return spec, log


def _heal_line(count):
    return (
        "info",
        f"healing the abandoned-lease wedge: a crashed render left {count} "
        "abandoned sample lease(s), so dispatch raises LifecycleBusyError; "
        "recycling the attached mesh once, then retrying this submit on the new "
        "fleet (docs/TROUBLESHOOTING.md #87)",
    )


def test_a_second_wedge_in_the_same_session_heals_the_live_handle(monkeypatch):
    """The second heal recycles the fleet that refused, not the Init handle.

    After the first heal the Init node's cached handle is retired and keeps
    its abandoned count, while ensure_live dispatches on the fresh fleet.
    Healing the cached handle answers ok from its already-stopped branch and
    retries on the still-wedged fleet, so the retry refuses again.
    """
    events: list[tuple] = []
    init = _fleet_handle("init", events, abandoned=1)
    second = _fleet_handle("second", events)
    third = _fleet_handle("third", events)
    spec, log = _wire_session(monkeypatch, events, init, [second, third])

    first = render_submit.submit_render(
        spec, {}, {"samples": object()}, 1.0, 2, handoff=PendingRenderHandoff())
    first.abandon()
    assert spec.mesh.handle is init and init.defunct
    assert init.abandoned_sample_leases == {1: 1}  # the stale count stays

    second.abandoned_sample_leases = {1: 3}  # a crashed render on the new fleet
    events.clear()
    retried = render_submit.submit_render(
        spec, {}, {"samples": object()}, 1.0, 2, handoff=PendingRenderHandoff())
    retried.abandon()

    assert events == [
        ("ensure_live", "init", "second"),
        ("recycle", "second"),
        ("drop_memos",),
        ("ensure_live", "init", "third"),
        ("submit", "third"),
    ]
    assert log.lines == [_heal_line(1), _heal_line(3)]


def test_a_retired_init_handle_s_stale_count_never_triggers_a_recycle(monkeypatch):
    """A busy refusal on a clean live fleet is not an abandoned-lease wedge.

    The live fleet refuses setup because another render holds a sample lease.
    The Init handle's stale abandoned count must not authorize a recycle, a
    memo drop or a retry: the original refusal is raised at once.
    """
    events: list[tuple] = []
    init = _fleet_handle("init", events, abandoned=1)
    init.defunct = True
    init.teardown_complete = True
    live = _fleet_handle("live", events)
    live.sample_leases = {1: 1}
    spec, log = _wire_session(monkeypatch, events, init, [live])

    def setup(handle, *_args, **_kwargs):
        events.append(("setup", handle.name))
        mesh_setup.require_no_sample_leases(handle, "change distributed setup")
        return object()

    monkeypatch.setattr(mesh_setup, "ensure_request_setup", setup)

    with pytest.raises(mesh_setup.SampleResultBusyError):
        render_submit.submit_render(
            spec, {}, {"samples": object()}, 1.0, 2,
            handoff=PendingRenderHandoff())

    assert events == [("ensure_live", "init", "live"), ("setup", "live")]
    assert log.lines == []


# Eager-setup sites. Init with an explicit preset, the UNET Loader and the
# BasicScheduler set the fleet up before any render. On hardware (2026-10-13)
# an untagged worker crash left one abandoned lease, and every later Init that
# changed topology or levers refused with SampleResultBusyError until a manual
# Reset, although docs/TROUBLESHOOTING.md #87 promised a later wedge heals.


def _eager_line(count, site):
    return (
        "info",
        f"healing the abandoned-lease wedge: a crashed render left {count} "
        f"abandoned sample lease(s), so {site} would refuse its setup change; "
        f"recycling the attached mesh once, then setting up {site} on the new "
        "fleet (docs/TROUBLESHOOTING.md #87)",
    )


def _eager_handle(name, events, *, abandoned=0, active=0):
    handle = _fleet_handle(name, events, abandoned=abandoned)
    handle.world = 2
    handle.sample_leases = {1: active} if active else {}
    handle.call_all = lambda endpoint, *_a, **_k: (
        events.append((endpoint, name)) or [{"host": name, "transition": "load"}])
    return handle


@pytest.fixture
def eager_seams(monkeypatch):
    """Setup refuses as the real one does; the session and preflights are inert."""
    import contextlib

    from dgx_monarch.nodes import render_preflight, render_session

    events: list[tuple] = []

    def ensure_request_setup(handle, *_args, **_kwargs):
        mesh_setup.require_no_sample_leases(handle, "change distributed setup")
        events.append(("setup", handle.name))

    def setup_on_handle(handle):
        handle.ensure_setup = lambda *_a, **_k: ensure_request_setup(handle)
        return handle

    log = _Log()
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", ensure_request_setup)
    monkeypatch.setattr(render_preflight, "preflight_sol_sequence_parallel",
                        lambda *_args: None)
    monkeypatch.setattr(render_session, "mutation_render_session",
                        lambda _handle: contextlib.nullcontext())
    monkeypatch.setattr(abandoned_heal, "log", log)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos",
                        lambda: events.append(("drop_memos",)))
    return events, log, setup_on_handle


def _cache(wedged, fresh):
    """ensure_live: the cached fleet until it is recycled, then the fresh one."""
    return lambda handle, **_kwargs: fresh if wedged.defunct else wedged


def test_an_explicit_init_heals_the_wedge_before_its_setup(eager_seams, monkeypatch):
    from dgx_monarch.nodes import init as init_mod

    events, log, setup_on_handle = eager_seams
    wedged = setup_on_handle(_eager_handle("wedged", events, abandoned=1))
    fresh = setup_on_handle(_eager_handle("fresh", events))
    monkeypatch.setattr(init_mod, "get_mesh", lambda **_kw: _cache(wedged, fresh)(None))

    (spec,) = init_mod.DGXMonarchInit().init(topology="cfg2", mode="local")

    assert spec.handle is fresh
    assert events == [("recycle", "wedged"), ("drop_memos",), ("setup", "fresh")]
    assert log.lines == [_eager_line(1, "the Init node")]


def test_an_explicit_init_leaves_a_live_lease_alone(eager_seams, monkeypatch):
    from dgx_monarch.nodes import init as init_mod

    events, log, setup_on_handle = eager_seams
    busy = setup_on_handle(_eager_handle("busy", events, abandoned=1, active=1))
    monkeypatch.setattr(init_mod, "get_mesh", lambda **_kw: busy)

    with pytest.raises(mesh_setup.SampleResultBusyError, match="sample result lease"):
        init_mod.DGXMonarchInit().init(topology="cfg2", mode="local")
    assert events == [] and log.lines == []


def test_an_auto_init_sets_nothing_up_and_leaves_the_heal_to_the_render(
        eager_seams, monkeypatch):
    from dgx_monarch.nodes import init as init_mod

    events, log, setup_on_handle = eager_seams
    wedged = setup_on_handle(_eager_handle("wedged", events, abandoned=1))
    monkeypatch.setattr(init_mod, "get_mesh", lambda **_kw: wedged)

    (spec,) = init_mod.DGXMonarchInit().init(topology="auto", mode="local")
    assert spec.handle is wedged
    assert events == [] and log.lines == []


def test_an_explicit_loader_heals_the_wedge_before_its_eager_load(
        eager_seams, monkeypatch):
    from dgx_monarch import capacity_agreement
    from dgx_monarch.nodes import loader_preflight, loaders
    from dgx_monarch.nodes.common import MeshSpec

    events, log, _setup_on_handle = eager_seams
    wedged = _eager_handle("wedged", events, abandoned=1)
    fresh = _eager_handle("fresh", events)
    monkeypatch.setattr(loaders, "ensure_live", _cache(wedged, fresh))
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        lambda *_args, **_kwargs: object())
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *_a, **_k: None)
    monkeypatch.setattr(capacity_agreement, "agree_load", lambda *_a, **_k: None)
    mesh = MeshSpec(handle=wedged, topology_preset="uly2", attention="TORCH_FLASH",
                    sync_ulysses=True)

    loaders.DGXMonarchUNETLoader().load(mesh, "x.safetensors")

    assert events == [("recycle", "wedged"), ("drop_memos",), ("setup", "fresh"),
                      ("load_model", "fresh")]
    assert log.lines == [_eager_line(1, "the UNET Loader")]


def test_the_basic_scheduler_heals_the_wedge_before_its_setup(eager_seams, monkeypatch):
    import types

    import torch

    from dgx_monarch.nodes import samplers

    events, log, _setup_on_handle = eager_seams
    wedged = _eager_handle("wedged", events, abandoned=2)
    fresh = _eager_handle("fresh", events)
    fresh.call_all = lambda endpoint, *_a, **_k: (
        events.append((endpoint, "fresh")) or [torch.tensor([1.0, 0.0])])
    monkeypatch.setattr(samplers, "ensure_live", _cache(wedged, fresh))
    mesh = types.SimpleNamespace(handle=wedged, topology_preset="uly2",
                                 attention="TORCH_FLASH", sync_ulysses=True,
                                 worker_args={})
    model = types.SimpleNamespace(mesh=mesh, request_dict=lambda: {})

    (sigmas,) = samplers.DGXMonarchBasicScheduler().get_sigmas(model, "normal", 2, 1.0)

    assert torch.equal(sigmas, torch.tensor([1.0, 0.0]))
    assert events == [("recycle", "wedged"), ("drop_memos",), ("setup", "fresh"),
                      ("compute_sigmas", "fresh")]
    assert log.lines == [_eager_line(2, "the BasicScheduler")]


def test_a_failed_recycle_before_eager_setup_raises_the_abandoned_refusal(
        eager_seams):
    events, _log, _setup_on_handle = eager_seams
    handle = _eager_handle("wedged", events, abandoned=1)
    handle.recycle_detailed = lambda: events.append(("recycle", "wedged")) or _Outcome(False)
    respawned = []

    with pytest.raises(mesh_setup.LifecycleBusyError, match="abandoned sample"):
        abandoned_heal.heal_before_eager_setup(
            handle, site="the Init node", respawn=respawned.append)
    assert events == [("recycle", "wedged")] and respawned == []


def test_a_healed_loader_keeps_the_credit_a_second_loader_of_its_checkpoint_reads(
        eager_seams, monkeypatch, tmp_path):
    """The heal drops the residency memos after the first footprint price wrote
    them, so the loader prices again before the spawn. The eager load on the
    fresh fleet then leaves its credit behind, and a second loader of the same
    checkpoint on the same Init is not charged those weights again (the real
    preflight, against MemAvailable that the first load lowered)."""
    import sys
    import types

    from dgx_monarch import (
        capacity_agreement,
        consent_store,
        driver_footprint,
        mesh_safety,
    )
    from dgx_monarch.adapters import detect
    from dgx_monarch.nodes import consent_projection, loader_preflight, loaders
    from dgx_monarch.nodes.common import MeshSpec

    events, log, _setup_on_handle = eager_seams
    gib = 2 ** 30
    name = "dit.safetensors"
    checkpoint = tmp_path / name
    checkpoint.write_bytes(b"dit")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda folder, unet: (  # type: ignore[attr-defined]
        str(checkpoint) if (folder, unet) == ("diffusion_models", name) else None)
    folder_paths.get_output_directory = lambda: str(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(detect, "sniff_checkpoint", lambda _path: ("minimax_h3", "bf16"))
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda _path: 40 * gib)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    memory = {"available": 114 * gib}
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: memory["available"])
    monkeypatch.setattr(loader_preflight, "_CLEARED_UNETS", {})
    monkeypatch.setattr(loader_preflight, "_UNPRICED_STACK_WARNED", {"minimax_h3"})
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "consent_memo.json"))
    monkeypatch.setattr(consent_projection, "project_for_loader", lambda *_a, **_k: None)
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *_a, **_k: None)
    monkeypatch.setattr(capacity_agreement, "agree_load", lambda *_a, **_k: None)
    # The real drop: the memos the first price wrote go with the old fleet.
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: (
        events.append(("drop_memos",)), loader_preflight.reset_memos()))
    price = loader_preflight.preflight_loader_footprint
    key = loader_preflight._memo_key(name, 0)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", lambda *a, **k: (
        events.append(("price", "credited" if key in loader_preflight._CLEARED_UNETS
                       else "charged")), price(*a, **k))[1])

    wedged = _eager_handle("wedged", events, abandoned=1)
    fresh = _eager_handle("fresh", events)
    for handle in (wedged, fresh):
        handle.owns_hosts = False  # rank 0 shares the driver host: weights are charged
    loaded = set()

    def load_on_fresh(endpoint, unet_name, *_args, **_kwargs):
        events.append((endpoint, "fresh"))
        if unet_name not in loaded:  # the second load reuses the resident weights
            loaded.add(unet_name)
            memory["available"] -= 40 * gib
        return [{"host": "fresh", "transition": "load"}]

    fresh.call_all = load_on_fresh
    monkeypatch.setattr(loaders, "ensure_live", _cache(wedged, fresh))
    mesh = MeshSpec(handle=wedged, topology_preset="uly2", attention="TORCH_FLASH",
                    sync_ulysses=True)

    loaders.DGXMonarchUNETLoader().load(mesh, name)
    assert events == [("price", "charged"), ("recycle", "wedged"), ("drop_memos",),
                      ("price", "charged"), ("setup", "fresh"), ("load_model", "fresh")]
    assert log.lines == [_eager_line(1, "the UNET Loader")]
    assert memory["available"] == 74 * gib  # where charging 40 GiB again refuses

    events.clear()
    loaders.DGXMonarchUNETLoader().load(mesh, name)  # no capacity refusal
    assert events == [("price", "credited"), ("setup", "fresh"), ("load_model", "fresh")]
