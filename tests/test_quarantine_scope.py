"""Scope quarantine changes to the combination whose first-use gate aborted.

A measured class K refusal answers accuracy for one combination and leaves its
policy switches unchanged. Other aborts disable the switches and keep that
combination on stock. A different combination recovers the graph's requested
policy because the aborted gate did not validate its weights.
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
