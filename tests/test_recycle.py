"""Attached-mesh reset: shutdown + eviction semantics, no monarch needed."""


def _idle_handle(mesh_mod, *, stop):
    import threading
    import types

    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.defunct = False
    handle.teardown_complete = False
    handle.replacement_blocked = None
    handle.setup_key = None
    handle.setup_generation = 0
    handle.setup_cleanup_state = None
    handle.sample_leases = {}
    handle.abandoned_sample_leases = {}
    handle.lock = threading.RLock()
    handle.procs = types.SimpleNamespace(stop=stop)
    return handle


def test_recycle_marks_defunct_and_evicts(monkeypatch):
    import threading
    import types

    from dgx_monarch import mesh as mesh_mod

    calls = []

    class _F:
        def get(self, timeout=None):
            return None

    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.defunct = False
    handle.setup_key = ("topo",)
    handle.lock = threading.RLock()
    handle.workers = types.SimpleNamespace(teardown_group=types.SimpleNamespace(
        call=lambda **flags: calls.append(("teardown", flags)) or _F()))
    handle.procs = types.SimpleNamespace(stop=lambda reason: calls.append("stop") or _F())
    monkeypatch.setattr(mesh_mod, "_MESHES", {("k",): handle})
    outcome = handle.recycle_detailed()
    assert outcome.status is mesh_mod.RecycleStatus.RECYCLED
    assert outcome.already_recycled is False
    assert outcome.as_dict()["ok"] is True
    # The stop frees the weights, so the workers are told to skip their unload.
    done = [("teardown", {"release_models": False}), "stop"]
    assert calls == done
    assert handle.setup_key is None
    # A second recycle succeeds as a no-op and does not stop the procs again.
    second = handle.recycle_detailed()
    assert second.ok is True
    assert second.already_recycled is True
    assert "Attached mesh" in second.detail
    assert calls == done
    assert handle.defunct is True
    assert mesh_mod._MESHES == {}  # evicted, so the next get_mesh spawns a fresh fleet


def test_recycle_reports_failure_when_proc_stop_raises(monkeypatch):
    import threading
    import types

    from dgx_monarch import mesh as mesh_mod

    class _F:
        def get(self, timeout=None):
            return None

    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.defunct = False
    handle.setup_key = None
    handle.lock = threading.RLock()

    def boom(reason):
        raise RuntimeError("stop hung")

    handle.procs = types.SimpleNamespace(stop=boom)
    monkeypatch.setattr(mesh_mod, "_MESHES", {("k",): handle})
    outcome = handle.recycle_detailed()
    assert outcome.status is mesh_mod.RecycleStatus.PROC_STOP_FAILED
    assert outcome.ok is False
    assert handle.defunct is True
    assert handle.replacement_blocked
    assert mesh_mod._MESHES == {("k",): handle}  # replacement over live procs is blocked
    assert handle.recycle() is False


def test_detailed_recycle_reports_active_lease_without_stopping_procs(monkeypatch):
    from dgx_monarch import mesh as mesh_mod

    stops = []
    handle = _idle_handle(
        mesh_mod, stop=lambda reason: stops.append(reason))
    handle.sample_leases = {7: 1}
    monkeypatch.setattr(mesh_mod, "_MESHES", {("k",): handle})

    outcome = handle._recycle_detailed_impl()

    assert outcome.status is mesh_mod.RecycleStatus.ACTIVE_WORK
    assert outcome.retryable is True
    assert outcome.ok is False
    assert stops == []
    assert handle.defunct is False


def test_detailed_recycle_reports_rooted_rdma_job_as_active_work(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import rdma_read_job

    stops = []
    handle = _idle_handle(
        mesh_mod, stop=lambda reason: stops.append(reason))
    assert handle.sample_leases == {}
    monkeypatch.setattr(
        rdma_read_job,
        "pending_job_count_for_handle",
        lambda candidate: int(candidate is handle),
    )

    outcome = handle._recycle_detailed_impl()

    assert outcome.status is mesh_mod.RecycleStatus.ACTIVE_WORK
    assert outcome.retryable is True
    assert "sample/result ownership" in outcome.detail
    assert stops == []
    assert handle.defunct is False


def test_detailed_recycle_keeps_lock_contention_distinct_from_active_work():
    import threading

    from dgx_monarch import mesh as mesh_mod

    stops = []
    handle = _idle_handle(
        mesh_mod, stop=lambda reason: stops.append(reason))
    handle.lock = threading.Lock()
    handle.lock.acquire()
    try:
        outcome = handle._recycle_detailed_impl(lock_timeout_s=0.001)
    finally:
        handle.lock.release()

    assert outcome.status is mesh_mod.RecycleStatus.LIFECYCLE_BUSY
    assert outcome.retryable is True
    assert stops == []
    assert handle.defunct is False


def test_detailed_recycle_reports_active_render_session_without_stopping_procs(
        monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import mesh_session

    stops = []
    handle = _idle_handle(
        mesh_mod, stop=lambda reason: stops.append(reason))
    monkeypatch.setattr(
        mesh_session,
        "require_mutation_authority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            mesh_session.ConcurrentRenderSessionError(
                "another render session owns this mesh handle")),
    )

    outcome = handle._recycle_detailed_impl()

    assert outcome.status is mesh_mod.RecycleStatus.ACTIVE_WORK
    assert outcome.retryable is True
    assert "render session" in outcome.detail
    assert stops == []


def test_detailed_recycle_distinguishes_prior_unknown_from_new_stop_failure(
        monkeypatch):
    from dgx_monarch import mesh as mesh_mod

    stops = []
    prior = _idle_handle(
        mesh_mod, stop=lambda reason: stops.append(reason))
    monkeypatch.setattr(
        mesh_mod.mesh_safety, "token_present", lambda token: token == id(prior))
    outcome = prior._recycle_detailed_impl()
    assert outcome.status is mesh_mod.RecycleStatus.PRIOR_TEARDOWN_UNKNOWN
    assert stops == []

    class _TimedOut:
        def get(self, timeout=None):
            raise TimeoutError("no acknowledgement")

    failing = _idle_handle(
        mesh_mod,
        stop=lambda reason: stops.append(reason) or _TimedOut(),
    )
    monkeypatch.setattr(mesh_mod.mesh_safety, "token_present", lambda _token: False)
    monkeypatch.setattr(mesh_mod, "_MESHES", {("k",): failing})
    outcome = failing._recycle_detailed_impl()
    assert outcome.status is mesh_mod.RecycleStatus.PROC_STOP_TIMED_OUT
    assert outcome.retryable is False
    assert stops == ["dgx-monarch recycle"]
    assert failing.replacement_blocked


def test_clear_vram_node_busy_recycle_does_not_mutate_workers_or_driver(monkeypatch):
    import json
    import types

    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.nodes import ops

    class _Handle:
        outcome = None

        def recycle_detailed(self):
            return self.outcome

        def call_all(self, *_args, **_kwargs):
            raise AssertionError("busy recycle must not issue worker RPCs")

    handle = _Handle()
    monkeypatch.setattr(ops, "ensure_live", lambda _handle: handle)
    monkeypatch.setattr(
        ops, "_free_driver_models",
        lambda _level: (_ for _ in ()).throw(
            AssertionError("busy recycle must not unload driver models")),
    )
    for status in (
        mesh_mod.RecycleStatus.ACTIVE_WORK,
        mesh_mod.RecycleStatus.LIFECYCLE_BUSY,
    ):
        handle.outcome = mesh_mod.RecycleOutcome(
            status, "recycle is busy", retryable=True)
        response = ops.DGXMonarchClearVRAM().clear(
            types.SimpleNamespace(handle=handle), "recycle", include_driver=True)
        payload = json.loads(response["result"][0])

        assert payload["recycle"]["status"] == status.value
        assert payload["recycle"]["retryable"] is True
        assert payload["driver"].startswith("not freed")


def test_clear_vram_recycle_forwards_exact_latent_after_resolved_handle_cleanup(monkeypatch):
    import types

    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.nodes import ops, recycle_drain

    stale = object()
    replacement = types.SimpleNamespace(
        recycle_detailed=lambda: mesh_mod.RecycleOutcome(
            mesh_mod.RecycleStatus.RECYCLED, "done"))
    drained = []
    samples = {"samples": object(), "batch_index": [0]}
    monkeypatch.setattr(ops, "ensure_live", lambda handle: replacement if handle is stale else None)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: drained.append(True))

    response = ops.DGXMonarchClearVRAM().clear(
        types.SimpleNamespace(handle=stale), "recycle", include_driver=False,
        samples=samples)

    assert response["result"][1] is samples
    assert drained == [True]
    assert ops.DGXMonarchClearVRAM.RETURN_TYPES == ("STRING", "LATENT")
    assert ops.DGXMonarchClearVRAM.RETURN_NAMES == ("status", "samples")


def test_clear_vram_recycle_refuses_latent_forward_before_driver_cleanup(monkeypatch):
    import types

    import pytest

    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.nodes import ops, recycle_drain

    handle = types.SimpleNamespace(outcome=None)
    handle.recycle_detailed = lambda: handle.outcome
    monkeypatch.setattr(ops, "ensure_live", lambda _handle: handle)
    monkeypatch.setattr(
        ops, "_free_driver_models",
        lambda _level: (_ for _ in ()).throw(
            AssertionError("failed recycle must not unload driver models")),
    )
    monkeypatch.setattr(
        recycle_drain, "drop_residency_memos",
        lambda: (_ for _ in ()).throw(
            AssertionError("failed recycle must not drain residency memos")),
    )

    for status in mesh_mod.RecycleStatus:
        if status is mesh_mod.RecycleStatus.RECYCLED:
            continue
        handle.outcome = mesh_mod.RecycleOutcome(status, "recycle failed")
        with pytest.raises(RuntimeError, match="cannot forward samples"):
            ops.DGXMonarchClearVRAM().clear(
                types.SimpleNamespace(handle=object()), "recycle", include_driver=True,
                samples={"samples": object(), "metadata": {"kept": True}})


def test_clear_vram_cache_callback_accepts_appended_samples():
    from dgx_monarch.nodes import ops

    assert str(ops.DGXMonarchClearVRAM.IS_CHANGED(
        object(), "recycle", include_driver=False, samples={"samples": object()})) == "nan"


def test_panel_presents_retryable_recycle_conflicts():
    from pathlib import Path

    panel = (Path(__file__).parents[1] / "web/js/dgx_monarch_panel.js").read_text()
    assert 'body.status === "active_work"' in panel
    assert 'actionState.recycle.label = "render session active; wait"' in panel
    assert 'body.status === "lifecycle_busy"' in panel
    assert 'actionState.recycle.label = "attached mesh changing; wait"' in panel
    assert 'actionState.recycle.label = "mesh outcome unknown; inspect"' in panel
    assert 'actionState.recycle.label = "request outcome unknown; inspect"' in panel
    assert "Persistent DGX Monarch worker services stay " in panel
    assert "Active work is refused " in panel
    assert "without stopping the mesh" in panel
