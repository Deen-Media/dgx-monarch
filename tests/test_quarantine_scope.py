"""Scope quarantine changes to the combination whose first-use gate aborted or failed.

A measured class K refusal answers accuracy for one combination and leaves its
policy switches unchanged. Other aborts disable the switches and keep that
combination on stock. A different combination recovers the graph's requested
policy because the aborted gate did not validate its weights. A FAIL never
touches the graph's policy dict: each dispatch of the failed combination gets
a clone with the switches off.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import accuracy_waiver, gate_ledger, refusal
from dgx_monarch.nodes import (
    auto_gate,
    common,
    gate_identity,
    gate_inconclusive,
    gate_process_state,
    gate_quarantine_scope,
)

CHROMA_GUARD = "shard_quant_scale:chroma"
SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"


class _Mesh:
    def __init__(self, worker_args, handle=None, topology_preset="auto"):
        self.worker_args = dict(worker_args)
        self.handle = handle
        self.topology_preset = topology_preset
        self.auto_gate = "first_use"
        self.attention = "TORCH_FLASH"
        self.sync_ulysses = True


class _Model:
    """One graph. Two combinations share a mesh the way a driver session does."""

    def __init__(self, mesh, unet_name="chroma_nvfp4.safetensors", loras=()):
        self.mesh = mesh
        self.unet_name = unet_name
        self.options = {}
        self.loras = tuple(loras)


class _InnerRefusal(RuntimeError):
    """Stands in for the worker exception a Monarch ActorError carries."""


class _ActorError(RuntimeError):
    """The wrapper the driver catches: the text is on ``exception``."""

    def __init__(self, inner):
        super().__init__(f"ActorError({inner!r})")
        self.exception = inner


def _tagged(guard=CHROMA_GUARD, waivable=True):
    return refusal.refusal(
        refusal.RefusalClass.KNOWN_WRONG,
        f"the {guard} guard measured this combination past the fidelity floor",
        guard=guard,
        waivable=waivable,
        panel_action=(accuracy_waiver.panel_action(guard) if waivable else None),
    )


def _chroma_abort():
    """A tagged class K guard refusal inside the wrapper, as the driver sees it."""
    return _ActorError(_InnerRefusal(_tagged()))


def _graph(**worker_args):
    return _Model(_Mesh(worker_args))


def _sibling(model, unet_name="hunyuan_image_bf16.safetensors"):
    """Another combination on the same graph, sharing its policy dict."""
    return _Model(model.mesh, unet_name=unet_name)


def test_a_measured_class_k_abort_is_read_through_the_actor_error_wrapper():
    """The refusal the driver catches is a wrapper; the tag is on the inner."""
    assert gate_quarantine_scope.measured_abort_guard(
        _chroma_abort()) == CHROMA_GUARD


def test_an_untyped_abort_is_unproven_and_names_no_guard():
    assert gate_quarantine_scope.measured_abort_guard(
        RuntimeError("the worker actor died")) is None
    assert gate_quarantine_scope.measured_abort_guard(None) is None


def test_an_unwaivable_known_wrong_refusal_is_not_the_exemption():
    """``auto`` never substitutes known-wrong math, and it names no guard.

    That refusal is class K with no card behind it, so it cannot claim an
    exemption that exists because a card answers the render instead.
    """
    unwaivable = _ActorError(_InnerRefusal(refusal.refusal(
        refusal.RefusalClass.KNOWN_WRONG,
        "auto resolved a topology this shape is measured wrong on")))
    assert gate_quarantine_scope.measured_abort_guard(unwaivable) is None


def test_a_class_k_ceremony_abort_takes_no_lever_from_the_graph():
    """The guard is measured. It says nothing about slab residency."""
    model = _graph(slab_weights=True, lora_low_rss=True)

    gate_identity.quarantine_unproven_paths(model, _chroma_abort())

    assert model.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_a_class_k_abort_leaves_the_next_combination_nothing_to_undo():
    """Nothing was written, so nothing is remembered and nothing is restored."""
    model = _graph(slab_weights=True, lora_low_rss=True)
    gate_identity.quarantine_unproven_paths(model, _chroma_abort())

    assert gate_quarantine_scope.restore_requested_levers(_sibling(model)) == []
    assert model.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_an_unproven_abort_quarantines_the_combination_that_aborted():
    """The aborting combination fails closed."""
    model = _graph(slab_weights=True, lora_low_rss=True)

    gate_identity.quarantine_unproven_paths(model, RuntimeError("actor died"))

    assert model.mesh.worker_args == {
        "slab_weights": False, "lora_low_rss": False}
    # Its own next render still reads the quarantine: this is the one
    # combination a ceremony left unproven.
    assert gate_quarantine_scope.restore_requested_levers(model) == []
    assert model.mesh.worker_args == {
        "slab_weights": False, "lora_low_rss": False}


def test_an_unproven_abort_under_fsdp_leaves_lora_low_rss_alone():
    """FSDP has no stock LoRA path to fall back to, so quarantining
    lora_low_rss off does not reach a safe state, only a guaranteed refusal
    on the graph's next render. slab_weights still goes off: it is moot
    under FSDP and costs nothing to keep in the fail-closed contract."""
    model = _Model(
        _Mesh({"slab_weights": True, "lora_low_rss": True},
              topology_preset="uly2+fsdp"),
        loras=[{"name": "adapter.safetensors", "strength": 0.5}],
    )

    gate_identity.quarantine_unproven_paths(model, RuntimeError("actor died"))

    assert model.mesh.worker_args == {
        "slab_weights": False, "lora_low_rss": True}


def test_an_unproven_abort_without_fsdp_still_forces_both_off_with_loras():
    """Without FSDP, stock is a real fallback, so both levers go off."""
    model = _Model(
        _Mesh({"slab_weights": True, "lora_low_rss": True}),
        loras=[{"name": "adapter.safetensors", "strength": 0.5}],
    )

    gate_identity.quarantine_unproven_paths(model, RuntimeError("actor died"))

    assert model.mesh.worker_args == {
        "slab_weights": False, "lora_low_rss": False}


def test_another_combination_starts_from_the_graphs_requested_levers():
    """A hunyuan slab cell after an aborted chroma cell gets the graph's levers."""
    model = _graph(slab_weights=True, lora_low_rss=True)
    gate_identity.quarantine_unproven_paths(model, RuntimeError("actor died"))

    hunyuan = _sibling(model)
    restored = gate_quarantine_scope.restore_requested_levers(hunyuan)

    assert sorted(restored) == ["lora_low_rss", "slab_weights"]
    assert hunyuan.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_a_lever_the_graph_never_named_comes_back_absent():
    """Absent means worker-side auto, and an explicit None is a third answer.

    Worker args bind every gate capability context, so restoring a key the
    graph never wrote would invalidate rows this graph would otherwise match.
    """
    model = _graph(slab_weights=True)
    gate_identity.quarantine_unproven_paths(model, RuntimeError("actor died"))
    assert model.mesh.worker_args == {
        "slab_weights": False, "lora_low_rss": False}

    gate_quarantine_scope.restore_requested_levers(_sibling(model))

    assert model.mesh.worker_args == {"slab_weights": True}


def test_a_lever_set_after_the_quarantine_is_not_clobbered():
    """Only a lever still reading False was this record's to give back."""
    model = _graph(slab_weights=True, lora_low_rss=True)
    gate_identity.quarantine_unproven_paths(model, RuntimeError("actor died"))
    model.mesh.worker_args["slab_weights"] = True

    restored = gate_quarantine_scope.restore_requested_levers(_sibling(model))

    assert restored == ["lora_low_rss"]
    assert model.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_a_second_graph_is_never_touched_by_another_graphs_quarantine():
    """The record is bound to one policy dict, matched by identity."""
    aborted = _graph(slab_weights=True, lora_low_rss=True)
    gate_identity.quarantine_unproven_paths(aborted, RuntimeError("actor died"))
    fresh = _graph(slab_weights=True, lora_low_rss=True)

    assert gate_quarantine_scope.restore_requested_levers(fresh) == []
    assert fresh.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_a_second_quarantine_still_gives_back_what_the_graph_asked_for():
    """After a restore, a second quarantine records the graph's ask again."""
    model = _graph(slab_weights=True, lora_low_rss=True)
    gate_identity.quarantine_unproven_paths(model, RuntimeError("first"))
    gate_quarantine_scope.restore_requested_levers(_sibling(model))
    second = _sibling(model, "wan_bf16.safetensors")
    gate_identity.quarantine_unproven_paths(second, RuntimeError("second"))

    gate_quarantine_scope.restore_requested_levers(_sibling(model, "third.st"))

    assert model.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_a_second_quarantine_keeps_the_first_value_it_recorded():
    """The record answers with the graph's ask, not with what a quarantine wrote.

    No restore between the two writes, so the second one finds the record the
    first one left. Somebody cleared the lever in between, so the second write
    sees it risky again and records it; keeping that value would hand the next
    combination an absent lever where the graph asked for slab.
    """
    model = _graph(slab_weights=True, lora_low_rss=True)
    gate_identity.quarantine_unproven_paths(model, RuntimeError("first"))
    model.mesh.worker_args.pop("slab_weights")
    gate_identity.quarantine_unproven_paths(model, RuntimeError("second"))

    assert sorted(gate_quarantine_scope.restore_requested_levers(
        _sibling(model))) == ["lora_low_rss", "slab_weights"]
    assert model.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_the_remembered_records_are_bounded():
    """Each record holds a policy dict alive, so the list cannot grow forever."""
    for index in range(gate_quarantine_scope._RECORD_LIMIT + 8):
        model = _graph(slab_weights=True, lora_low_rss=True)
        model.unet_name = f"model_{index}.safetensors"
        gate_identity.quarantine_unproven_paths(model, RuntimeError("actor died"))

    assert (len(gate_quarantine_scope._RECORDS)
            == gate_quarantine_scope._RECORD_LIMIT)


def test_the_quarantine_log_line_names_the_combination_and_the_class(caplog):
    model = _graph(slab_weights=True, lora_low_rss=True)
    tagged = _ActorError(_InnerRefusal(refusal.refusal(
        refusal.RefusalClass.UNPROVEN,
        "this load has no first-load-stock memo",
        guard="first_load_stock_memo")))

    with caplog.at_level(logging.ERROR):
        gate_identity.quarantine_unproven_paths(model, tagged)

    line = next(record.getMessage() for record in caplog.records
                if "could not complete" in record.getMessage())
    assert "chroma_nvfp4.safetensors" in line
    assert gate_quarantine_scope.combination(model) in line
    assert "class U first_load_stock_memo" in line
    assert "lora_low_rss, slab_weights" in line


def test_the_class_k_log_line_says_the_levers_stayed(caplog):
    model = _graph(slab_weights=True, lora_low_rss=True)

    with caplog.at_level(logging.ERROR):
        gate_identity.quarantine_unproven_paths(model, _chroma_abort())

    line = next(record.getMessage() for record in caplog.records
                if "could not complete" in record.getMessage())
    assert CHROMA_GUARD in line
    assert gate_quarantine_scope.combination(model) in line
    assert "takes no residency lever" in line


def test_a_measured_abort_pushes_no_worker_policy(monkeypatch):
    """No lever changed, so there is no policy to broadcast and none is."""
    pushes = []
    monkeypatch.setattr(
        gate_identity, "apply_worker_policy",
        lambda *args, **kwargs: pushes.append(args))
    model = _graph(slab_weights=True, lora_low_rss=True)
    handle = SimpleNamespace(setup_key=("ready",), config=None)

    assert gate_identity.force_stock_quarantine(
        model, handle, cause=_chroma_abort()) == []

    assert pushes == []
    assert model.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_an_unproven_abort_still_pushes_the_stock_policy(monkeypatch):
    pushes = []

    def apply_policy(_handle, worker_args, timeout_s=600.0):
        pushes.append(dict(worker_args))
        return []

    monkeypatch.setattr(gate_identity, "apply_worker_policy", apply_policy)
    model = _graph(slab_weights=True, lora_low_rss=True)
    handle = SimpleNamespace(setup_key=("ready",), config=None)

    gate_identity.force_stock_quarantine(
        model, handle, cause=RuntimeError("actor died"))

    assert pushes == [{"slab_weights": False, "lora_low_rss": False}]


def test_a_delivery_failure_abort_takes_the_levers_and_pushes_nothing(monkeypatch):
    """Issue #47: a gate leg on a mesh whose Worker loops restarted ends on the
    price wait's fast refusal. The actors that would take the stock policy are
    gone, and a 600 s push to them would hold the abort, so none is sent; the
    combination's levers still go off, as for any unproven abort."""
    from dgx_monarch.delivery_faults import WorkerDeliveryError

    pushes = []
    monkeypatch.setattr(
        gate_identity, "apply_worker_policy",
        lambda *args, **kwargs: pushes.append(args))
    model = _graph(slab_weights=True, lora_low_rss=True)
    handle = SimpleNamespace(setup_key=("ready",), config=None)
    refused = WorkerDeliveryError(refusal.refusal(
        refusal.RefusalClass.CAPACITY, "undeliverable", guard="cross_rank_capacity",
        waivable=False, troubleshooting=2))

    assert gate_identity.force_stock_quarantine(model, handle, cause=refused) == []

    assert pushes == []
    assert model.mesh.worker_args == {"slab_weights": False, "lora_low_rss": False}


def _patch_auto_state(monkeypatch, token):
    lock = threading.Lock()
    monkeypatch.setattr(
        gate_process_state, "_AUTO_GATE_ACTIVE", SimpleNamespace(on=False))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_LOCK", lock)
    monkeypatch.setattr(
        gate_process_state, "_AUTO_GATE_CONDITION", threading.Condition(lock))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_RUNNING", set())
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_SESSION", {})
    monkeypatch.setattr(gate_process_state, "_PROCESS_GATE_DENIALS", {})
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_WAIT_S", 1.0)
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", token))


def _run_maybe_auto_gate(monkeypatch, error):
    token = gate_ledger.gate_verdict_token("combo", "artifact", "known", {})
    _patch_auto_state(monkeypatch, token)
    from dgx_monarch.nodes import gate

    def explode(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(gate, "run_identity_ceremony", explode)
    return auto_gate.maybe_auto_gate(
        _graph(slab_weights=True), {"kind": "ksampler", "steps": 2}, {}, 1.0, 2)


@pytest.mark.parametrize("waivable", [True, False])
def test_the_abort_arm_reports_a_measured_class_k_abort(monkeypatch, waivable):
    """The render path reads the class off the verdict, not off the exception."""
    error = _ActorError(_InnerRefusal(_tagged(waivable=waivable)))
    verdict = _run_maybe_auto_gate(monkeypatch, error)

    if waivable:
        assert verdict == gate_inconclusive.KNOWN_WRONG_ABORT_VERDICT
        assert verdict in gate_inconclusive.NO_QUARANTINE_VERDICTS
    else:
        assert verdict == "ERROR"
        assert verdict not in gate_inconclusive.NO_QUARANTINE_VERDICTS


def test_the_abort_arm_still_reports_an_unproven_abort_as_error(monkeypatch):
    verdict = _run_maybe_auto_gate(monkeypatch, RuntimeError("actor died"))

    assert verdict == "ERROR"
    assert verdict not in gate_inconclusive.NO_QUARANTINE_VERDICTS


def _ceremony_that_publishes_then_aborts(monkeypatch, token, error):
    """The real abort shape, including the row the ceremony opened first.

    A ceremony publishes INCONCLUSIVE for every context it is about to prove
    before it renders anything, and an ordinary residency abort withdraws
    none of it. So the denial outlives the abort and the next render of this
    combination reads it instead of running a second ceremony.
    """
    from dgx_monarch.nodes import gate

    calls = []

    def explode(*_args, **_kwargs):
        calls.append(1)
        common._record_process_gate_verdicts([token], "INCONCLUSIVE")
        raise error

    monkeypatch.setattr(gate, "run_identity_ceremony", explode)
    return calls


def _gate_twice(monkeypatch, error):
    token = gate_ledger.gate_verdict_token("combo", "artifact", "known", {})
    _patch_auto_state(monkeypatch, token)
    calls = _ceremony_that_publishes_then_aborts(monkeypatch, token, error)
    model = _graph(slab_weights=True)
    request = {"kind": "ksampler", "steps": 2}
    first = auto_gate.maybe_auto_gate(model, request, {}, 1.0, 2)
    second = auto_gate.maybe_auto_gate(model, request, {}, 1.0, 2)
    return token, calls, first, second


def test_the_class_k_rule_outlives_the_denial_its_own_ceremony_left(monkeypatch):
    """The repeat render of the aborting combination takes no lever either.

    Without the memo the second gate reads the ceremony's own INCONCLUSIVE
    back from the cache, which is not a no-quarantine verdict, so the class K
    rule would hold for one render only.
    """
    _token, calls, first, second = _gate_twice(monkeypatch, _chroma_abort())

    assert len(calls) == 1  # the second gate answered from the cache
    assert first == gate_inconclusive.KNOWN_WRONG_ABORT_VERDICT
    assert second == gate_inconclusive.KNOWN_WRONG_ABORT_VERDICT
    assert second in gate_inconclusive.NO_QUARANTINE_VERDICTS


def test_the_cached_denial_a_measured_guard_earned_still_denies(monkeypatch):
    """Fail closed: the memo answers the render path, never authorization.

    ``authorize_normal_render`` reads the raw verdict, so the dispatch behind
    this combination is still stamped stock. Only the lever question reads
    the class.
    """
    token, _calls, _first, _second = _gate_twice(monkeypatch, _chroma_abort())

    assert gate_process_state._PROCESS_GATE_DENIALS[token] == "INCONCLUSIVE"
    assert common._process_gate_verdict(token) == "INCONCLUSIVE"


def test_an_unproven_aborts_cached_denial_still_quarantines(monkeypatch):
    """The other side of the same read, and the mutation guard for it."""
    _token, calls, first, second = _gate_twice(
        monkeypatch, RuntimeError("actor died"))

    assert len(calls) == 1
    assert first == "ERROR"
    assert second == "INCONCLUSIVE"
    assert second not in gate_inconclusive.NO_QUARANTINE_VERDICTS


def test_the_class_k_memo_outlives_the_capped_session_cache(monkeypatch):
    """A flood of later combinations may evict a row; it may not evict a class.

    The denial map is uncapped and this memo answers beside it, so a cap on
    one of the pair alone would quarantine a graph on an answer the guard
    already gave.
    """
    token, _calls, _first, _second = _gate_twice(monkeypatch, _chroma_abort())

    for index in range(gate_process_state._AUTO_GATE_SESSION_LIMIT + 8):
        common._record_process_gate_verdicts(
            [("combo", f"artifact-{index}", "commit", "context")],
            "INCONCLUSIVE")

    assert gate_inconclusive.cached_verdict(token, "INCONCLUSIVE") == (
        gate_inconclusive.KNOWN_WRONG_ABORT_VERDICT)


def test_a_later_publication_supersedes_the_class_k_memo(monkeypatch):
    """A fresh answer for the token retires the abort's class with the row."""
    token, _calls, _first, _second = _gate_twice(monkeypatch, _chroma_abort())

    common._record_process_gate_verdicts([token], "INCONCLUSIVE")

    assert gate_inconclusive.cached_verdict(token, "INCONCLUSIVE") == (
        "INCONCLUSIVE")


def test_every_render_path_restores_before_it_reads_the_risk():
    """A quarantine still on the dict reads as no risk at all.

    If a render path reads the risk first, a combination that asks for slab
    after another combination's abort turned slab off finds nothing to prove,
    runs no ceremony and loads stock. So each render path restores before it
    reads the risk.
    """
    for module, marker in (
        ("common.py", "gate_result = _maybe_auto_gate("),
        ("pipeline.py", "common.auto_gate_required("),
        ("samplers.py", "if auto_gate_required("),
    ):
        source = (SRC / "nodes" / module).read_text()
        assert (source.index("_restore_requested_levers(model)")
                < source.index(marker)), module


def test_every_no_quarantine_verdict_is_a_verdict_the_render_path_can_see():
    """The set is the render path's whole vocabulary for taking no lever."""
    assert gate_inconclusive.NO_QUARANTINE_VERDICTS == frozenset({
        None, "PASS", "DUAL_MODEL_STOCK",
        gate_inconclusive.NO_MATERIAL_VERDICT,
        gate_inconclusive.KNOWN_WRONG_ABORT_VERDICT,
        # A ceremony the request's capacity stopped before its proof render.
        gate_inconclusive.CAPACITY_STOP_DISPATCH_VERDICT,
        # FAIL quarantines each dispatch of its own combination, never the graph.
        "FAIL",
    })


class _Pending:
    def result(self):
        return {"samples": "done"}

    def abandon(self):
        pass


def _render_rig(monkeypatch, verdicts):
    """Drive ``common.run_render`` and record what the gate read each time.

    Everything the gate itself would do is stubbed. The quarantine and the
    restore stay real, because those are what this pins.
    """
    seen: list[dict] = []

    def record_gate(model, *_args):
        seen.append(dict(model.mesh.worker_args))
        return verdicts[len(seen) - 1]

    monkeypatch.setattr(
        common, "_bind_packed_render_model", lambda model, *_args: (model, None))
    monkeypatch.setattr(common, "model_for_request", lambda model, _request: model)
    monkeypatch.setattr(common, "_maybe_auto_gate", record_gate)
    monkeypatch.setattr(
        common, "submit_render", lambda *_args, **_kwargs: _Pending())
    return seen


def test_the_next_combination_reaches_its_ceremony_with_the_graphs_levers(
        monkeypatch):
    """The chroma-then-hunyuan case, driven down the render path.

    The seam tests above call the restore directly. This one does not, so it
    fails if the restore is dropped, moved behind the gate, or handed a policy
    dict the graph does not share.
    """
    seen = _render_rig(monkeypatch, ["ERROR", "PASS"])
    chroma = _graph(slab_weights=True, lora_low_rss=True)

    common.run_render(chroma, {}, {}, None, 1)

    assert seen[0] == {"slab_weights": True, "lora_low_rss": True}
    assert chroma.mesh.worker_args == {
        "slab_weights": False, "lora_low_rss": False}

    hunyuan = _sibling(chroma)
    common.run_render(hunyuan, {}, {}, None, 1)

    assert seen[1] == {"slab_weights": True, "lora_low_rss": True}
    assert hunyuan.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


def test_the_aborting_combination_reaches_its_own_next_ceremony_quarantined(
        monkeypatch):
    """Fail-closed for the one combination a ceremony left unproven."""
    seen = _render_rig(monkeypatch, ["ERROR", "ERROR"])
    chroma = _graph(slab_weights=True, lora_low_rss=True)

    common.run_render(chroma, {}, {}, None, 1)
    common.run_render(chroma, {}, {}, None, 1)

    assert seen[1] == {"slab_weights": False, "lora_low_rss": False}
    assert chroma.mesh.worker_args == {
        "slab_weights": False, "lora_low_rss": False}


def test_a_measured_class_k_verdict_takes_no_lever_down_the_render_path(
        monkeypatch):
    """The verdict the abort arm returns has to survive the render path too."""
    _render_rig(monkeypatch, [gate_inconclusive.KNOWN_WRONG_ABORT_VERDICT])
    model = _graph(slab_weights=True, lora_low_rss=True)

    common.run_render(model, {}, {}, None, 1)

    assert model.mesh.worker_args == {
        "slab_weights": True, "lora_low_rss": True}


# A measured or persisted FAIL quarantines its own combination in its own
# capability context (docs/DESIGN.md, invariant 2). Every checkpoint wired to
# one Init node shares that node's policy dict, so the FAIL lands on each
# dispatch clone and never on the dict.

_ASKED = {"lora_low_rss": True, "slab_weights": True}
_OFF = {"lora_low_rss": False, "slab_weights": False}


def _two_checkpoints(monkeypatch, tmp_path, worker_args=None):
    """Checkpoints A and B on one Init node, over a real gate ledger on disk."""
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec

    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local", world=1, n_hosts=1, gpus_per_host=1)
    mesh = MeshSpec(
        handle=handle, topology_preset="single", attention="TORCH_FLASH",
        sync_ulysses=False, worker_args=dict(worker_args or _ASKED))
    style = ({"name": "style.safetensors", "strength": 1.0},)
    a = ModelSpec(mesh=mesh, unet_name="a.safetensors", loras=style)
    b = ModelSpec(mesh=mesh, unet_name="b.safetensors", loras=style)
    artifacts = {
        name: gate_ledger.artifact_set_signature([f"{name}-sig"])
        for name in ("a.safetensors", "b.safetensors")}
    monkeypatch.setattr(gate_mod, "_combo_of", lambda model: (
        gate_quarantine_scope.combination(model), artifacts[model.unet_name]))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gate_ledger, "comfy_commit", lambda: "commit")
    monkeypatch.setattr(gate_process_state._AUTO_GATE_ACTIVE, "on", False, raising=False)
    monkeypatch.setattr(gate_process_state, "_PROCESS_GATE_DENIALS", {})
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_SESSION", {})
    return handle, a, b, artifacts


def _dispatch_context(model, handle, topology=None):
    from dgx_monarch.nodes import gate as gate_mod

    return gate_mod.gate_capability_context(
        model, handle, resolved_topology=topology,
        resolved_attention=None if topology is None else "TORCH_FLASH")


def _persist_fail(tmp_path, model, handle, artifacts, topology=None):
    """The row a FAIL ceremony leaves for exactly this dispatch's context."""
    gate_ledger.GateLedger(str(tmp_path)).record(
        gate_quarantine_scope.combination(model), artifacts[model.unet_name],
        "commit", "FAIL", {"quarantine_levers": ["slab_weights", "lora_low_rss"]},
        _dispatch_context(model, handle, topology))


def _enforce(model, handle, topology):
    from dgx_monarch.nodes import render_quarantine

    return render_quarantine._enforce_persisted_quarantine(
        model, handle, resolved_topology=topology, resolved_attention="TORCH_FLASH")


def test_a_persisted_fail_on_a_leaves_bs_levers_as_the_graph_asked(
        monkeypatch, tmp_path, caplog):
    from dgx_monarch.topology import Topology

    handle, a, b, artifacts = _two_checkpoints(monkeypatch, tmp_path)
    topology = Topology(world=1)
    _persist_fail(tmp_path, a, handle, artifacts, topology)

    with caplog.at_level(logging.ERROR):
        quarantined = _enforce(a, handle, topology)
    line = next(record.getMessage() for record in caplog.records
                if "previously FAILED" in record.getMessage())
    assert gate_quarantine_scope.combination_label(a) in line
    assert quarantined.mesh.worker_args == _OFF
    assert quarantined.mesh.handle is handle

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        assert _enforce(b, handle, topology) is None
    # B is not affected by A's FAIL, so nothing names a quarantine for B.
    assert not [record for record in caplog.records
                if "FAILED" in record.getMessage()]
    assert b.mesh.worker_args == _ASKED
    assert a.mesh.worker_args is b.mesh.worker_args


@pytest.mark.parametrize("origin", ["explicit", "auto_first_use"])
def test_a_measured_fail_on_a_leaves_bs_levers(monkeypatch, tmp_path, caplog, origin):
    """The in-ceremony FAIL pushes A's levers off and leaves the shared dict.

    Both origins: the automatic first-use ceremony that run_render and the
    pipeline take must not write the graph's levers either, and must not log
    the abort line that says it did.
    """
    import torch

    from dgx_monarch.nodes import gate as gate_mod
    from gate_orchestration_helpers import _cross_mode_rig

    # The explicit ceremony publishes process verdicts; keep them in this test.
    monkeypatch.setattr(gate_process_state, "_PROCESS_GATE_DENIALS", {})
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_SESSION", {})
    worker_args = dict(_ASKED)
    handle, _ledger, a = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.full((1,), 5.0), torch.ones(1)], worker_args)
    b = _Model(a.mesh, unet_name="b.safetensors")

    with caplog.at_level(logging.WARNING):
        result = gate_mod.run_identity_ceremony(
            a, {"noise_seed": 1, "steps": 2, "cfg": 1.0},
            {"samples": torch.zeros(1)}, 1.0, 2, origin, run_id="x")

    assert result["verdict"] == "FAIL"
    pushes = [args[0] for method, args in handle.calls
              if method == "apply_worker_args"]
    off = [pushed for pushed in pushes
           if pushed["slab_weights"] is False and pushed["lora_low_rss"] is False]
    assert pushes[-1] in off
    assert len(off) == 1  # the verdict's own push, and no second graph-wide one
    assert a.mesh.worker_args is worker_args
    assert worker_args == _ASKED
    assert b.mesh.worker_args == _ASKED
    assert gate_identity.model_for_request(b, {}).mesh.worker_args == _ASKED
    assert gate_quarantine_scope._RECORDS == []
    assert not [record for record in caplog.records
                if "could not complete" in record.getMessage()]


def test_a_session_fail_quarantines_a_even_when_its_ledger_row_was_lost(
        monkeypatch, tmp_path):
    """The process verdict cache holds a FAIL the ledger could not record."""
    from dgx_monarch.topology import Topology

    handle, a, b, artifacts = _two_checkpoints(monkeypatch, tmp_path)
    topology = Topology(world=1)
    token = gate_ledger.gate_verdict_token(
        gate_quarantine_scope.combination(a), artifacts["a.safetensors"].current,
        "commit", _dispatch_context(a, handle, topology))
    common._record_process_gate_verdicts([token], "FAIL")

    assert _enforce(a, handle, topology).mesh.worker_args == _OFF
    assert _enforce(b, handle, topology) is None
    assert a.mesh.worker_args == _ASKED


def test_a_stays_quarantined_on_its_next_render_after_b_ran(monkeypatch, tmp_path):
    """A, then B, then A again through the render path, across a restart.

    In a fresh session a ledger FAIL runs no ceremony (the gate answers None),
    so only the submit-time re-read can quarantine A. B renders with the levers
    the graph asked for and reaches its own ceremony with them.
    """
    from dgx_monarch.topology import Topology

    handle, a, b, artifacts = _two_checkpoints(monkeypatch, tmp_path)
    topology = Topology(world=1)
    _persist_fail(tmp_path, a, handle, artifacts, topology)
    seen = _render_rig(monkeypatch, [None, "PASS", None])
    sent: list[tuple[str, dict]] = []

    def submit(model, *_args, **_kwargs):
        dispatched = _enforce(model, handle, topology) or model
        sent.append((model.unet_name, dict(dispatched.mesh.worker_args)))
        return _Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    for model in (a, b, a):
        common.run_render(model, {}, {}, None, 1)

    assert sent == [("a.safetensors", _OFF), ("b.safetensors", _ASKED),
                    ("a.safetensors", _OFF)]
    assert seen == [_ASKED, _ASKED, _ASKED]
    assert a.mesh.worker_args == _ASKED


def test_a_quarantined_combination_still_dispatches_fleet_with_the_levers_off(
        monkeypatch, tmp_path):
    """Fleet's authorization reads the dispatch clone, not the shared dict."""
    import torch

    from dgx_monarch.nodes import fleet as fleet_mod

    handle, a, b, artifacts = _two_checkpoints(monkeypatch, tmp_path)
    _persist_fail(tmp_path, a, handle, artifacts)  # Fleet reads with no topology

    class Session:
        def bind(self, _handle):
            return None

        def activate(self):
            from contextlib import nullcontext

            return nullcontext()

        def close(self):
            return None

    seen: list[tuple[str, dict, dict]] = []

    def fleet_bound(_self, spec, *_args, **_kwargs):
        authorization = fleet_mod._fleet_worker_policy(spec, handle)
        seen.append((spec.unet_name, dict(spec.mesh.worker_args),
                     dict(authorization.worker_args)))
        return ({"samples": torch.zeros(1)},)

    monkeypatch.setattr(fleet_mod, "RenderSession", Session)
    monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(fleet_mod.DGXMonarchFleetKSampler, "_fleet_bound", fleet_bound)

    fleet_mod.DGXMonarchFleetKSampler().fleet(
        [a], [object()], ["NEG"], [{"samples": torch.zeros(1)}],
        [""], [42], [1], [1.0], ["euler"], ["simple"], positive=["POS"])

    name, dispatched, authorized = seen[0]
    assert name == "a.safetensors"
    assert dispatched == _OFF
    assert authorized["lora_low_rss"] is False and authorized["slab_weights"] is False
    assert b.mesh.worker_args == _ASKED


def test_a_quarantine_turns_comfy_managed_off_for_the_whole_init_node(
        monkeypatch, tmp_path, caplog):
    """comfy_managed is a bootstrap policy, so it alone stays mesh-wide.

    A live worker refuses to change it for one dispatch, and a per-checkpoint
    flip would force a mesh reset at every switch between checkpoints. The
    log line says so, instead of saying every other checkpoint kept it.
    """
    from dgx_monarch.topology import Topology

    rung = {"comfy_managed": True, "slab_weights": False, "lora_low_rss": False}
    handle, a, b, artifacts = _two_checkpoints(monkeypatch, tmp_path, rung)
    topology = Topology(world=1)
    _persist_fail(tmp_path, a, handle, artifacts, topology)

    with caplog.at_level(logging.ERROR):
        quarantined = _enforce(a, handle, topology)

    assert quarantined.mesh.worker_args["comfy_managed"] is False
    assert a.mesh.worker_args["comfy_managed"] is False
    assert b.mesh.worker_args == {**rung, "comfy_managed": False}
    line = next(record.getMessage() for record in caplog.records
                if "previously FAILED" in record.getMessage())
    assert "lora_low_rss+slab_weights off for that combination" in line
    assert "comfy_managed is now off for the whole Init node" in line


def test_a_quarantine_line_without_comfy_managed_names_only_the_pair(
        monkeypatch, tmp_path, caplog):
    """A classic graph never carried comfy_managed, so the line never names it."""
    from dgx_monarch.topology import Topology

    handle, a, _b, artifacts = _two_checkpoints(monkeypatch, tmp_path)
    topology = Topology(world=1)
    _persist_fail(tmp_path, a, handle, artifacts, topology)

    with caplog.at_level(logging.ERROR):
        _enforce(a, handle, topology)

    line = next(record.getMessage() for record in caplog.records
                if "previously FAILED" in record.getMessage())
    assert "lora_low_rss+slab_weights off for that combination" in line
    assert "other checkpoints on this Init node keep theirs" in line
    assert "comfy_managed" not in line


def test_a_cached_fail_keeps_the_pipeline_full(monkeypatch):
    """No ceremony runs for a cached FAIL, so the pipeline need not drain."""
    token = ("combo", "artifacts", "commit", "context")
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", token))
    monkeypatch.setattr(gate_process_state, "_PROCESS_GATE_DENIALS", {})
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_SESSION", {})
    monkeypatch.setattr(gate_process_state._AUTO_GATE_ACTIVE, "on", False, raising=False)

    common._record_process_gate_verdicts([token], "FAIL")
    assert auto_gate.auto_gate_required(SimpleNamespace()) is False

    common._record_process_gate_verdicts([token], "INCONCLUSIVE")
    assert auto_gate.auto_gate_required(SimpleNamespace()) is True


# Two checkpoints in one RenderPipeline. The KSampler Pipeline node pushes one
# model per pipeline, so only the RenderPipeline surface carries two. A FAIL's
# stock clone and another checkpoint's requested levers need different worker
# settings, and an earlier render's result lease refuses a settings change
# (MeshHandle.ensure_setup), so the submit collects the earlier renders first.


class _StopNow(BaseException):
    """An interruption, so the failed collection takes no supervision path."""


def _mixed_pipeline(monkeypatch, tmp_path, gating, *, interrupt=None):
    """A and B on one Init node, through the real pipeline and submit paths.

    The handle is a MeshHandle double whose real ``ensure_setup`` refuses a
    settings change while a sample lease is live, and whose dispatch takes
    real leases. Its live settings start as the graph's risky pair. A carries
    a FAIL in the process verdict cache and no ledger row; B is proved.
    """
    import dataclasses

    import dgx_monarch.nodes.render_submit as submit_mod
    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes import render_preflight, render_quarantine
    from dgx_monarch.nodes.gate_identity import NormalRenderAuthorization
    from dgx_monarch.topology import Topology
    from render_sessions_helpers import _ready_real_handle, _stub_direct_submit

    real_setup = mesh_setup.ensure_request_setup
    names, a, b, artifacts = _two_checkpoints(monkeypatch, tmp_path)
    handle, _token = _ready_real_handle()
    for name in ("config", "config_fingerprint", "n_hosts", "gpus_per_host"):
        setattr(handle, name, getattr(names, name))
    mesh = dataclasses.replace(a.mesh, handle=handle, auto_gate=gating)
    a, b = (dataclasses.replace(model, mesh=mesh) for model in (a, b))
    events: list[tuple[str, object]] = []

    def apply(worker_args, timeout_s=120.0, merge_config=True):
        leases = sum(handle.sample_leases.values())
        events.append(("apply", (dict(worker_args), leases)))
        handle.worker_args_key = handle._worker_args_key(worker_args)
        handle.active_worker_args = dict(worker_args)
        return []

    def dispatch(mesh_handle, request, _port, token, authority=None):
        events.append(("dispatch", request["model"]["unet_name"]))
        return mesh_setup.dispatch_sample(
            mesh_handle, object, setup_token=token, authority=authority)

    def collect(_future, **_kwargs):
        events.append(("collect", None))
        if interrupt is not None:
            raise interrupt
        return [{"latent": "wire"}]

    _stub_direct_submit(monkeypatch, handle, dispatch)
    monkeypatch.setattr(
        submit_mod, "_enforce_persisted_quarantine",
        render_quarantine._enforce_persisted_quarantine)
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", real_setup)
    monkeypatch.setattr(
        submit_mod, "_finish_render",
        lambda _results, _topo, latent, _future, **_kwargs: {"samples": latent["samples"]})
    monkeypatch.setattr(
        render_preflight, "activation_footprint_preflight_for_request",
        lambda *_args: None)

    def no_ceremony(*_args):
        raise AssertionError("a proved or failed combination ran a ceremony")

    monkeypatch.setattr(common, "_maybe_auto_gate", no_ceremony)
    handle.apply_worker_args = apply
    handle.collect_sample = collect
    topology = Topology(world=1)
    attention = common.resolve_sample_attention("TORCH_FLASH", False)
    handle.setup_key = mesh_setup.nccl_setup_key(topology, mesh.attention, mesh.sync_ulysses)
    handle.worker_args_key = handle._worker_args_key(handle.effective_worker_args(_ASKED))

    # The FAIL A's earlier ceremony left in this session; its ledger write was lost.
    context = gate_mod.gate_capability_context(
        a, handle, resolved_topology=topology, resolved_attention=attention)
    token_a = gate_ledger.gate_verdict_token(
        gate_quarantine_scope.combination(a), artifacts["a.safetensors"].current,
        "commit", context)
    common._record_process_gate_verdicts([token_a], "FAIL")
    if gating == "first_use":
        token_b = ("b-combo", "b-artifacts", "commit", "b-context")
        states = {"a.safetensors": ("unknown", token_a),
                  "b.safetensors": ("pass", token_b)}
        monkeypatch.setattr(
            common, "_auto_gate_context",
            lambda model, *_args: states[model.unet_name])

        def proved(model, _handle, **_kwargs):
            # B's ledger PASS grants its requested levers; A's clone asks for none.
            policy = dict(model.mesh.worker_args)
            return NormalRenderAuthorization(
                policy, model.request_dict(), None,
                "stock" if policy == _OFF else "required")

        monkeypatch.setattr(submit_mod, "authorize_normal_render", proved)
    return handle, a, b, events


@pytest.mark.parametrize("gating", ["first_use", "off"])
@pytest.mark.parametrize("order", ["proved_then_failed", "failed_then_proved"])
def test_a_pipelined_settings_change_collects_the_earlier_render_first(
        monkeypatch, tmp_path, gating, order):
    """Depth 2: a proved checkpoint B holding a result lease, then checkpoint A
    with a cached FAIL, or the reverse. The second dispatch needs other worker
    settings than the first, so the submit collects the first render before
    the change instead of aborting the pipeline with SampleResultBusyError."""
    from dgx_monarch.nodes.render_session import RenderSession

    handle, a, b, events = _mixed_pipeline(monkeypatch, tmp_path, gating)
    first, second = (b, a) if order == "proved_then_failed" else (a, b)
    pipe = common.RenderPipeline(depth=2)

    pipe.push(first, {"kind": "ksampler"}, {"samples": first.unet_name}, 1.0, 2)
    assert sum(handle.sample_leases.values()) == 1  # the first render's lease is live
    pipe.push(second, {"kind": "ksampler"}, {"samples": second.unet_name}, 1.0, 2)
    results = pipe.drain()

    assert results == [{"samples": first.unet_name}, {"samples": second.unet_name}]
    stock_a = ("apply", (_OFF, 0))
    if order == "proved_then_failed":
        # B dispatches under the live risky settings; A's stock clone changes
        # them only after B's result came back and its lease retired.
        assert events == [
            ("dispatch", "b.safetensors"), ("collect", None), stock_a,
            ("dispatch", "a.safetensors"), ("collect", None)]
    else:
        # A changes the settings with nothing in flight, so nothing is
        # collected early; B's change back waits for A's result.
        assert events == [
            stock_a, ("dispatch", "a.safetensors"), ("collect", None),
            ("apply", (_ASKED, 0)), ("dispatch", "b.safetensors"), ("collect", None)]
    assert a.mesh.worker_args == _ASKED  # the FAIL stayed off the shared dict
    assert sum(handle.sample_leases.values()) == 0
    assert not handle.abandoned_sample_leases
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_an_interrupted_early_collection_aborts_before_the_settings_change(
        monkeypatch, tmp_path):
    """The early collection fails the way a collection does: the pipeline
    aborts, the interruption surfaces, A is never dispatched, the settings
    are not changed, and the render session is released."""
    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes.render_session import RenderSession

    stop = _StopNow("collection interrupted")
    handle, a, b, events = _mixed_pipeline(
        monkeypatch, tmp_path, "first_use", interrupt=stop)
    pipe = common.RenderPipeline(depth=2)
    pipe.push(b, {"kind": "ksampler"}, {"samples": "b"}, 1.0, 2)

    with pytest.raises(_StopNow) as raised:
        pipe.push(a, {"kind": "ksampler"}, {"samples": "a"}, 1.0, 2)

    assert raised.value is stop
    # It came from the early collection, not from a refused settings change
    # whose abort then collected B.
    chain, seen = raised.value, []
    while chain is not None and chain not in seen:
        seen.append(chain)
        chain = chain.__cause__ or chain.__context__
    assert not [error for error in seen
                if isinstance(error, mesh_setup.SampleResultBusyError)]
    assert [kind for kind, _detail in events] == ["dispatch", "collect"]
    assert sum(handle.sample_leases.values()) == 0
    assert list(pipe._inflight) == []
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()
