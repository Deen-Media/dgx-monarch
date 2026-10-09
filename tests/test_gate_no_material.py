"""An INCONCLUSIVE with nothing to prove costs one render, not the session.

A LoRA-free graph with slab residency not engaged reaches INCONCLUSIVE.
Without the classification (added 2026-08-05), that verdict logs at ERROR
through the quarantine and turns both residency levers off for every later
render in the ComfyUI process. The classification separates that outcome from
a prevented ceremony; every other INCONCLUSIVE keeps the fail-closed path
unchanged.
"""
from __future__ import annotations

import json
import types

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
import dgx_monarch.runtime_provenance as runtime_provenance
from dgx_monarch.actor import gate_cycle
from dgx_monarch.actor import model_store as model_store_mod
from dgx_monarch.mesh_setup import SetupToken
from dgx_monarch.nodes import gate_inconclusive, gate_process_state
from dgx_monarch.nodes.slab_proof import SlabProof


@pytest.fixture(autouse=True)
def _forget_classifications():
    """Leave no memo, denial or session row behind: these are process state."""
    denials = dict(gate_process_state._PROCESS_GATE_DENIALS)
    session = dict(common._AUTO_GATE_SESSION)
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
    yield
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
    gate_process_state._PROCESS_GATE_DENIALS = denials
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_SESSION.update(session)


def _identity(unet_name, loras):
    artifacts = [{"kind": "diffusion_models", "name": unet_name,
                  "signature": "model-sig"}]
    artifacts.extend(
        {"kind": "loras", "name": entry["name"], "signature": f"lora-sig-{index}"}
        for index, entry in enumerate(loras or [])
    )
    return {
        "digest": ledger_mod.artifact_set_signature(
            item["signature"] for item in artifacts).current,
        "comfy": "known",
        "artifacts": artifacts,
    }


def _slab_proof(**overrides):
    """The proof an unvouched, LoRA-free ceremony produces on real hardware."""
    fields = {
        "b": {},
        "cycle": [{"host": "gn100", "rank": 0, "world": 1,
                   "setup_generation": 4, "conclusive": False, "no_material": True,
                   "family": "minimax_h3", "slab_active": False}],
        "complete": True,
        "conclusive": False,
        "reasons": ["no LoRA/slab lineage and no complete FSDP clean-reload proof"],
        "expected": False,
        "active": False,
        "family": "minimax_h3",
    }
    fields.update(overrides)
    return SlabProof(**fields)


def _result(**overrides):
    row = {
        "verdict": "INCONCLUSIVE",
        "origin": "auto_first_use",
        "model": "minimax_h3.safetensors",
        "loras": 0,
        "cross_mode": None,
        "inconclusive_reasons": [
            "no LoRA/slab lineage and no complete FSDP clean-reload proof",
            "no lora stack: lazy swap is not applicable",
        ],
    }
    row.update(overrides)
    return row


def _classify(result=None, slab_proof=None, worker_args=None, fsdp_scope=False):
    return gate_inconclusive.classify(
        result or _result(),
        slab_proof if slab_proof is not None else _slab_proof(),
        {"lora_low_rss": True} if worker_args is None else worker_args,
        fsdp_scope=fsdp_scope,
    )


def test_a_leverless_ceremony_that_ran_clean_has_no_material():
    assert _classify() == gate_inconclusive.KIND_NO_MATERIAL


@pytest.mark.parametrize(
    "kwargs",
    [
        # A LoRA stack is a swap lineage the ceremony failed to prove.
        pytest.param({"result": _result(loras=1)}, id="lora-stack"),
        # Any cross-residency finding is evidence about slab, not silence.
        pytest.param(
            {"result": _result(cross_mode={"verdict": "ERROR", "detail": "x"})},
            id="cross-mode-error"),
        pytest.param(
            {"result": _result(cross_mode={"verdict": "CAPACITY", "detail": "x"})},
            id="cross-mode-capacity"),
        pytest.param({"slab_proof": _slab_proof(expected=True)}, id="slab-expected"),
        pytest.param({"slab_proof": _slab_proof(active=True)}, id="slab-active"),
        pytest.param(
            {"slab_proof": _slab_proof(
                error="slab residency was requested but not exercised")},
            id="slab-error"),
        # A rank that never reported, or ranks that disagree on the resident
        # family, is a finding: slab_proof.family is None for both.
        pytest.param({"slab_proof": _slab_proof(cycle=[], family=None)}, id="no-ranks"),
        pytest.param({"slab_proof": _slab_proof(family=None)}, id="family-drift"),
        # A no-material report skips later ceremonies only once the slab-proof
        # validator binds every row to the current rank cohort (complete=True).
        pytest.param(
            {"slab_proof": _slab_proof(
                complete=False,
                cycle=[{"rank": 0, "world": 2, "setup_generation": 4,
                        "conclusive": False, "no_material": True,
                        "family": "minimax_h3", "slab_active": False},
                       {"rank": 0, "world": 2, "setup_generation": 4,
                        "conclusive": False, "no_material": True,
                        "family": "minimax_h3", "slab_active": False}])},
            id="duplicate-rank-cohort"),
        pytest.param(
            {"slab_proof": _slab_proof(
                complete=False,
                cycle=[{"rank": 0, "world": 1, "setup_generation": 3,
                        "conclusive": False, "no_material": True,
                        "family": "minimax_h3", "slab_active": False}])},
            id="stale-generation-cohort"),
        # A row without the marker (a worker too old to stamp it) is unproven.
        pytest.param(
            {"slab_proof": _slab_proof(
                cycle=[{"conclusive": False, "family": "minimax_h3"}])},
            id="unstamped-rank"),
        pytest.param(
            {"slab_proof": _slab_proof(
                cycle=[{"conclusive": False, "no_material": True, "family": "m"},
                       {"conclusive": False, "family": "m"}])},
            id="one-unstamped-rank"),
        pytest.param({"fsdp_scope": True}, id="fsdp-proof-scope"),
        # Under comfy-managed residency the rung itself is the lever under
        # proof, so a leverless run there reaching INCONCLUSIVE means something
        # else went wrong (residency_mode.leverless_ceremony_verdict).
        pytest.param(
            {"worker_args": {"comfy_managed": True, "lora_low_rss": False,
                             "slab_weights": False}},
            id="comfy-managed-rung"),
    ],
)
def test_every_other_inconclusive_stays_unproven(kwargs):
    assert _classify(**kwargs) == gate_inconclusive.KIND_UNPROVEN


def test_the_no_material_line_is_calm_and_names_what_stayed_on(caplog):
    result = _result()
    with caplog.at_level("INFO"):
        kind = gate_inconclusive.record_inconclusive(
            result, _slab_proof(), {"lora_low_rss": True}, fsdp_scope=False)

    assert kind == gate_inconclusive.KIND_NO_MATERIAL
    assert result[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    [record] = [r for r in caplog.records if "identity gate" in r.getMessage()]
    assert record.levelname == "INFO"
    message = record.getMessage()
    assert "nothing to gate on this graph" in message
    assert "residency levers unchanged" in message


def test_a_prevented_ceremony_keeps_todays_warning_word_for_word(caplog):
    result = _result(cross_mode={"verdict": "ERROR", "detail": "leg failed"},
                     inconclusive_reasons=["cross-residency reference leg failed"])
    with caplog.at_level("INFO"):
        kind = gate_inconclusive.record_inconclusive(
            result, _slab_proof(expected=True), {"lora_low_rss": True},
            fsdp_scope=False)

    assert kind == gate_inconclusive.KIND_UNPROVEN
    assert result[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_UNPROVEN
    [record] = [r for r in caplog.records if "identity gate" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert record.getMessage() == (
        "identity gate INCONCLUSIVE (auto_first_use, minimax_h3.safetensors): "
        "cross-residency reference leg failed; the swaps did not take the "
        "lazy path, so there is nothing to compare"
    )


def test_the_worker_stamps_exactly_the_key_the_driver_classifies_by():
    def worker(lora_low_rss=True):
        store = types.SimpleNamespace(
            lora_low_rss=lora_low_rss,
            current=types.SimpleNamespace(family="minimax_h3", slab=None),
            ensure=lambda *_args, **_kwargs: (object(), "load"),
        )
        return types.SimpleNamespace(
            _setup_key=("live",), _setup_generation=4, rank=0, world=1, store=store,
            _inject_for_topology=lambda *_args, **_kwargs: None,
        )

    empty = gate_cycle.run(worker(), "m", {}, lora_stack=[])
    assert empty[gate_inconclusive.CYCLE_KEY] is True
    assert empty["reason"] == "no lora stack: lazy swap is not applicable"

    # A stack with the lever turned off is a different case: the swap path
    # exists and this run did not exercise it.
    lever_off = gate_cycle.run(
        worker(lora_low_rss=False), "m", {}, lora_stack=[{"name": "a", "strength": 1.0}])
    assert gate_inconclusive.CYCLE_KEY not in lever_off


def _ceremony_rig(monkeypatch, tmp_path, cycle, worker_args):
    class Handle:
        setup_generation = 4
        setup_key = ("setup",)
        worker_args_key = ("worker",)
        world = 1

        def __init__(self):
            self.calls = []

        def call_all(self, method, *args, **kwargs):
            self.calls.append(method)
            if method == "provenance_baseline":
                return [{
                    "rank": 0,
                    "world": 1,
                    "setup_generation": args[0],
                    "source_manifest_sha256": (
                        runtime_provenance.cached_dgx_source_manifest_sha256()
                    ),
                }]
            if method == "unload":
                return [{"unloaded": True}]
            if method == "gate_swap_cycle":
                return [
                    {
                        "rank": rank,
                        "world": self.world,
                        "setup_generation": self.setup_generation,
                        **dict(entry),
                    }
                    for rank, entry in enumerate(cycle)
                ]
            return [{"ok": True}]

    class Ledger:
        def __init__(self):
            self.records = []

        def begin_retest_required(self, *args):
            pass

        def record(self, *args):
            self.records.append(args)

    handle, ledger = Handle(), Ledger()
    model = types.SimpleNamespace(
        unet_name="minimax_h3.safetensors", options={}, loras=(),
        mesh=types.SimpleNamespace(handle=handle, worker_args=worker_args),
    )
    monkeypatch.setattr(
        gate_mod, "run_render",
        lambda *_args, **_kwargs: {"samples": torch.zeros(1)})
    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        gate_mod,
        "render_setup_token",
        lambda *_args: SetupToken(4, ("setup",), ("worker",)),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gate_mod, "GateLedger", lambda directory: ledger)
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)
    monkeypatch.setattr("dgx_monarch.telemetry.emit", lambda *args, **kwargs: None)
    return handle, ledger, model


def test_the_ceremony_records_the_kind_and_leaves_the_graph_policy_alone(
        monkeypatch, tmp_path, caplog):
    worker_args = {"lora_low_rss": True}
    handle, ledger, model = _ceremony_rig(
        monkeypatch, tmp_path,
        [{"host": "gn100", "conclusive": False, "no_material": True,
          "reason": "no lora stack: lazy swap is not applicable",
          "family": "minimax_h3", "slab_active": False}],
        worker_args,
    )

    with caplog.at_level("INFO"):
        result = gate_mod.run_identity_ceremony(
            model, {"noise_seed": 1, "steps": 2, "cfg": 1.0},
            {"samples": torch.zeros(1)}, 1.0, 2, "auto_first_use", run_id="x")

    assert result["verdict"] == "INCONCLUSIVE"
    assert result[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    # One INCONCLUSIVE row per published context, and no worker policy push.
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE"] * 4
    # Only the row whose context ran carries the grant: one on a Fleet or
    # explicit-on sibling row would skip the ceremony for residency nobody
    # exercised.
    assert [record[4].get(gate_inconclusive.RESULT_KEY) for record in ledger.records] == [
        gate_inconclusive.KIND_NO_MATERIAL, None, None, None]
    assert worker_args == {"lora_low_rss": True}
    assert "apply_worker_args" not in handle.calls
    assert any(r.levelname == "INFO" and "nothing to gate on this graph" in r.getMessage()
               for r in caplog.records)
    with open(tmp_path / "dgxm_gate_reports.jsonl") as stream:
        row = json.loads(stream.read().strip())
    assert row[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL


def test_a_ceremony_that_was_prevented_still_quarantines_inside_its_session(
        monkeypatch, tmp_path, caplog):
    """The paired arm: a rank report without the no-material marker is a
    finding, and the ceremony still turns both levers off inside its session."""
    worker_args = {"lora_low_rss": True}
    handle, _ledger, model = _ceremony_rig(
        monkeypatch, tmp_path,
        [{"host": "gn100", "conclusive": False,
          "reason": "swaps fell back to reload", "family": "minimax_h3",
          "slab_active": False}],
        worker_args,
    )

    with caplog.at_level("INFO"):
        result = gate_mod.run_identity_ceremony(
            model, {"noise_seed": 1, "steps": 2, "cfg": 1.0},
            {"samples": torch.zeros(1)}, 1.0, 2, "auto_first_use", run_id="x")

    assert result["verdict"] == "INCONCLUSIVE"
    assert result[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_UNPROVEN
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
    assert "apply_worker_args" in handle.calls
    assert any(r.levelname == "ERROR" and "could not complete" in r.getMessage()
               for r in caplog.records)


def _dispatch_rig(monkeypatch, result):
    token = ("combo", "artifacts", "commit", "no-material-context")
    result = {**result, "_gate_token": token, "_gate_tokens": [token]}
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", token))
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony", lambda *_args, **_kwargs: dict(result))
    common._AUTO_GATE_SESSION.pop(token, None)
    gate_process_state._PROCESS_GATE_DENIALS.pop(token, None)
    common._AUTO_GATE_ACTIVE.on = False
    return token


def test_the_dispatcher_verdict_survives_the_session_cache(monkeypatch):
    token = _dispatch_rig(
        monkeypatch,
        _result(**{gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL}))
    request = {"kind": "ksampler", "steps": 2}

    first = common._maybe_auto_gate(types.SimpleNamespace(), request, {}, 1.0, 2)
    second = common._maybe_auto_gate(types.SimpleNamespace(), request, {}, 1.0, 2)

    assert first == second == gate_inconclusive.NO_MATERIAL_VERDICT
    # The ledger, the denial map and every authorization predicate still read
    # INCONCLUSIVE; only the dispatcher's copy carries the kind.
    assert common._process_gate_verdict(token) == "INCONCLUSIVE"


def test_a_genuine_inconclusive_reaches_the_dispatcher_unchanged(monkeypatch):
    token = _dispatch_rig(
        monkeypatch,
        _result(**{gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_UNPROVEN}))
    request = {"kind": "ksampler", "steps": 2}

    assert common._maybe_auto_gate(
        types.SimpleNamespace(), request, {}, 1.0, 2) == "INCONCLUSIVE"
    assert common._maybe_auto_gate(
        types.SimpleNamespace(), request, {}, 1.0, 2) == "INCONCLUSIVE"
    assert common._process_gate_verdict(token) == "INCONCLUSIVE"


_TOKEN = ("combo", "artifacts", "commit", "context")


def _no_material(token=_TOKEN):
    return _result(
        _gate_token=token,
        **{gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL})


@pytest.mark.parametrize(
    ("verdict", "ceremony"),
    [
        # The retest guard opens every ceremony and carries no result at all.
        # An exemption that survived it would outlive a ceremony that aborted.
        pytest.param("INCONCLUSIVE", None, id="retest-guard"),
        pytest.param(
            "INCONCLUSIVE",
            _result(_gate_token=_TOKEN,
                    **{gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_UNPROVEN}),
            id="unproven"),
        pytest.param("FAIL", _result(verdict="FAIL", _gate_token=_TOKEN), id="fail"),
        pytest.param("PASS", _result(verdict="PASS", _gate_token=_TOKEN), id="pass"),
    ],
)
def test_the_next_publication_replaces_an_earlier_classification(verdict, ceremony):
    """The last publication for a token decides its classification, in both
    directions."""
    common._record_process_gate_verdicts([_TOKEN], "INCONCLUSIVE", _no_material())
    assert gate_inconclusive.cached_verdict(_TOKEN, "INCONCLUSIVE") == (
        gate_inconclusive.NO_MATERIAL_VERDICT)

    common._record_process_gate_verdicts([_TOKEN], verdict, ceremony)
    assert gate_inconclusive.cached_verdict(_TOKEN, "INCONCLUSIVE") == "INCONCLUSIVE"


def test_a_classification_outlives_any_number_of_later_combinations():
    """The memo shares the denial map's session lifetime: no eviction.

    Both stores gain one entry per distinct combination in the same
    publication step. Never cap the memo alone: under a 128-entry cap
    (removed 2026-08-05) the first entry fell back to a session quarantine
    once 128 later combinations rendered.
    """
    common._record_process_gate_verdicts([_TOKEN], "INCONCLUSIVE", _no_material())
    for index in range(200):
        other = ("combo", f"artifact-{index}", "commit", "context")
        common._record_process_gate_verdicts(
            [other], "INCONCLUSIVE", _no_material(other))
    assert gate_inconclusive.cached_verdict(_TOKEN, "INCONCLUSIVE") == (
        gate_inconclusive.NO_MATERIAL_VERDICT)


def test_a_render_after_an_explicit_no_material_gate_keeps_its_levers(monkeypatch):
    """The explicit half of the same failure: the operator ran the Gate node
    and no auto ceremony ran. The dispatcher reaches the same decision from
    the memo."""
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", _TOKEN))
    common._AUTO_GATE_ACTIVE.on = False
    common._record_process_gate_verdicts([_TOKEN], "INCONCLUSIVE", _no_material())

    verdict = common._maybe_auto_gate(
        types.SimpleNamespace(), {"kind": "ksampler", "steps": 2}, {}, 1.0, 2)

    assert verdict == gate_inconclusive.NO_MATERIAL_VERDICT
    assert common._process_gate_verdict(_TOKEN) == "INCONCLUSIVE"


def test_the_explicit_ceremony_stamps_only_the_context_it_ran(
        monkeypatch, tmp_path):
    """A ceremony publishes four tokens and exercises one. The Fleet scope and
    the two explicit-on slab siblings are published to be revoked, and a graph
    that resolves to one of them wants residency this ceremony never ran."""
    _handle, _ledger, model = _ceremony_rig(
        monkeypatch, tmp_path,
        [{"host": "gn100", "conclusive": False, "no_material": True,
          "reason": "no lora stack: lazy swap is not applicable",
          "family": "minimax_h3", "slab_active": False}],
        {"lora_low_rss": True},
    )

    result = gate_mod.run_identity_ceremony(
        model, {"noise_seed": 1, "steps": 2, "cfg": 1.0},
        {"samples": torch.zeros(1)}, 1.0, 2, "explicit", run_id="x")

    assert result[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    own = tuple(result["_gate_token"])
    published = [tuple(token) for token in result["_gate_tokens"]]
    assert len(published) == 4
    assert gate_inconclusive.cached_verdict(own, "INCONCLUSIVE") == (
        gate_inconclusive.NO_MATERIAL_VERDICT)
    assert [gate_inconclusive.cached_verdict(token, "INCONCLUSIVE")
            for token in published if token != own] == ["INCONCLUSIVE"] * 3
    # The real retest guard published INCONCLUSIVE on all four before the
    # ceremony ran; the ceremony's publication left the exercised token denied
    # (and exempt) and lifted the guard from the three it did not run.
    assert common._process_gate_verdict(own) == "INCONCLUSIVE"
    assert [common._process_gate_verdict(token)
            for token in published if token != own] == [None] * 3


_FLEET = ("combo", "artifacts", "commit", "fleet-context")
_EXPLICIT_ON = ("combo", "artifacts", "commit", "explicit-on-context")
_EXPLICIT_ON_FLEET = ("combo", "artifacts", "commit", "explicit-on-fleet-context")


def test_a_no_material_ceremony_denies_only_the_context_it_exercised():
    """The four published tokens reach the session cache as one exempt denial
    and three contexts with no verdict. A sibling that reads INCONCLUSIVE here
    is quarantined on its first render and never runs its own ceremony
    (hardware, 2026-09-09)."""
    common._record_process_gate_verdicts(
        [_TOKEN, _FLEET, _EXPLICIT_ON, _EXPLICIT_ON_FLEET], "INCONCLUSIVE",
        _no_material())

    assert common._process_gate_verdict(_TOKEN) == "INCONCLUSIVE"
    assert gate_inconclusive.cached_verdict(
        _TOKEN, common._process_gate_verdict(_TOKEN)) == (
            gate_inconclusive.NO_MATERIAL_VERDICT)
    assert [common._process_gate_verdict(token)
            for token in (_FLEET, _EXPLICIT_ON, _EXPLICIT_ON_FLEET)] == [None] * 3
    assert _EXPLICIT_ON not in common._AUTO_GATE_SESSION


def test_a_no_material_ceremony_lifts_the_retest_guard_and_a_sibling_pass():
    """Every ceremony opens by publishing INCONCLUSIVE on its retest tokens, the
    siblings included (gate_ceremony's retest guard). The no-material result
    must lift that denial from the siblings: on hardware on 2026-09-09 it stayed
    on the explicit-on sibling and quarantined that sibling's first render. A
    PASS the session held for a sibling goes too
    (gate_inconclusive.session_denial_tokens)."""
    common._record_process_gate_verdicts([_EXPLICIT_ON], "PASS")
    common._record_process_gate_verdicts(
        [_TOKEN, _FLEET, _EXPLICIT_ON, _EXPLICIT_ON_FLEET], "INCONCLUSIVE")  # the guard
    assert common._process_gate_verdict(_EXPLICIT_ON_FLEET) == "INCONCLUSIVE"

    common._record_process_gate_verdicts(
        [_TOKEN, _FLEET, _EXPLICIT_ON, _EXPLICIT_ON_FLEET], "INCONCLUSIVE",
        _no_material())

    assert common._process_gate_verdict(_TOKEN) == "INCONCLUSIVE"
    assert [common._process_gate_verdict(token)
            for token in (_FLEET, _EXPLICIT_ON, _EXPLICIT_ON_FLEET)] == [None] * 3


def test_every_other_result_still_publishes_every_token_it_names():
    for verdict, ceremony in (
        ("INCONCLUSIVE", _result(
            _gate_token=_TOKEN,
            **{gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_UNPROVEN})),
        ("ERROR", None),
        ("FAIL", _result(_gate_token=_TOKEN, verdict="FAIL")),
    ):
        gate_process_state._PROCESS_GATE_DENIALS = {}
        common._AUTO_GATE_SESSION.clear()
        common._record_process_gate_verdicts(
            [_TOKEN, _FLEET, _EXPLICIT_ON, _EXPLICIT_ON_FLEET], verdict, ceremony)
        assert [common._process_gate_verdict(token) for token in
                (_TOKEN, _FLEET, _EXPLICIT_ON, _EXPLICIT_ON_FLEET)] == [verdict] * 4


def test_the_explicit_on_sibling_of_a_no_material_ceremony_runs_its_own_ceremony(
        monkeypatch):
    """The render order seen on hardware on 2026-09-09: an auto-slab render of
    a checkpoint that fits stock, then the explicit-on render of it in the same
    driver session. The second must claim its own ceremony, not take the
    sibling's cached INCONCLUSIVE and quarantine."""
    current = {"token": _TOKEN}
    ceremonies = []

    def ceremony(*_args, **_kwargs):
        ceremonies.append(current["token"])
        if current["token"] == _TOKEN:
            return {
                **_no_material(),
                "_gate_tokens": [_TOKEN, _FLEET, _EXPLICIT_ON, _EXPLICIT_ON_FLEET],
            }
        return {"verdict": "PASS", "_gate_token": _EXPLICIT_ON,
                "_gate_tokens": [_EXPLICIT_ON]}

    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", current["token"]))
    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    common._AUTO_GATE_ACTIVE.on = False
    request = {"kind": "ksampler", "steps": 2}

    first = common._maybe_auto_gate(types.SimpleNamespace(), request, {}, 1.0, 2)
    assert first == gate_inconclusive.NO_MATERIAL_VERDICT

    current["token"] = _EXPLICIT_ON
    second = common._maybe_auto_gate(types.SimpleNamespace(), request, {}, 1.0, 2)

    assert ceremonies == [_TOKEN, _EXPLICIT_ON]
    assert second == "PASS"
    assert common._process_gate_verdict(_EXPLICIT_ON) == "PASS"


def _render_rig(monkeypatch, gate_result):
    quarantines = []

    class Pending:
        def result(self):
            return {"samples": "done"}

        def abandon(self):
            pass

    monkeypatch.setattr(
        common, "_bind_packed_render_model", lambda model, *_args: (model, None))
    monkeypatch.setattr(common, "model_for_request", lambda model, _request: model)
    monkeypatch.setattr(common, "_maybe_auto_gate", lambda *_args: gate_result)
    monkeypatch.setattr(common, "submit_render", lambda *_args, **_kwargs: Pending())
    monkeypatch.setattr(
        common, "_quarantine_unproven_paths",
        lambda model: quarantines.append(model))
    return quarantines


def test_no_material_dispatch_never_quarantines_the_session(monkeypatch):
    quarantines = _render_rig(monkeypatch, gate_inconclusive.NO_MATERIAL_VERDICT)
    assert common.run_render(object(), {}, {}, None, 1) == {"samples": "done"}
    assert quarantines == []


@pytest.mark.parametrize("gate_result", ["INCONCLUSIVE", "FAIL", "ERROR"])
def test_every_other_unproven_dispatch_still_quarantines(monkeypatch, gate_result):
    quarantines = _render_rig(monkeypatch, gate_result)
    assert common.run_render(object(), {}, {}, None, 1) == {"samples": "done"}
    assert len(quarantines) == 1


def _grant_rig(monkeypatch, tmp_path):
    """A LoRA-free graph whose only risky lever is worker-side slab auto.

    Only the combination key, ledger directory, commit and ensure_live are
    stubbed: the GateLedger writes a real file in ``tmp_path``, and the
    capability context and token are real, because the field bug was a row
    with the right kind that nothing read.
    """
    handle = types.SimpleNamespace(
        config=types.SimpleNamespace(worker_args={}),
        world=2, n_hosts=2, gpus_per_host=1)
    model = types.SimpleNamespace(
        unet_name="minimax_h3.safetensors", options={}, loras=(),
        mesh=types.SimpleNamespace(
            handle=handle, auto_gate="first_use", worker_args={"lora_low_rss": True},
            topology_preset="uly2", attention="TORCH_FLASH", sync_ulysses=True),
    )
    artifacts = ledger_mod.artifact_set_signature(["model-sig"])
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("combo", artifacts))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")
    return model, artifacts


def _persist(tmp_path, artifacts, token, verdict, detail):
    """One durable row for exactly the context this token names."""
    ledger_mod.GateLedger(str(tmp_path)).record(
        "combo", artifacts, "commit", verdict, detail, token[3])


def test_a_no_material_row_grants_the_skip_in_a_fresh_process(monkeypatch, tmp_path):
    """In the field, H3 re-ran the ceremony after a driver restart while the
    grant lived only in process memory; the durable row must grant the skip."""
    model, artifacts = _grant_rig(monkeypatch, tmp_path)

    state, token = common._auto_gate_context(model, "ksampler")
    assert state == "unknown"
    assert common.auto_gate_required(model, "ksampler") is True

    _persist(tmp_path, artifacts, token, "INCONCLUSIVE",
             {gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL})

    granted, same_token = common._auto_gate_context(model, "ksampler")
    assert same_token == token
    assert granted == gate_inconclusive.GRANTED_STATE
    assert common.auto_gate_required(model, "ksampler") is False


def test_the_granted_dispatch_never_starts_a_ceremony(monkeypatch, tmp_path):
    model, artifacts = _grant_rig(monkeypatch, tmp_path)
    resolve = common._auto_gate_context
    _state, token = resolve(model, "ksampler")
    _persist(tmp_path, artifacts, token, "INCONCLUSIVE",
             {gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL})
    ceremonies = []
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony",
        lambda *_args, **_kwargs: ceremonies.append(1))
    monkeypatch.setattr(
        common, "_auto_gate_context",
        lambda subject, _kind, *_args: resolve(subject, "ksampler"))
    common._AUTO_GATE_ACTIVE.on = False

    verdict = common._maybe_auto_gate(
        model, {"kind": "ksampler", "steps": 2}, {}, 1.0, 2)

    assert verdict == gate_inconclusive.NO_MATERIAL_VERDICT
    assert ceremonies == []
    # A grant proves nothing new, so it publishes nothing: the denial map and
    # the session cache stay empty for a combination no ceremony touched.
    assert common._process_gate_verdict(token) is None


def test_a_pass_row_grants_the_skip_it_always_did(monkeypatch, tmp_path):
    model, artifacts = _grant_rig(monkeypatch, tmp_path)
    _state, token = common._auto_gate_context(model, "ksampler")

    _persist(tmp_path, artifacts, token, "PASS", {"quarantine_levers": []})

    assert common._auto_gate_context(model, "ksampler")[0] == "pass"
    assert common.auto_gate_required(model, "ksampler") is False


@pytest.mark.parametrize(
    "detail",
    [
        # A row written before the classification existed carries no kind, so
        # it re-proves: absence of evidence is never a grant.
        pytest.param({}, id="unclassified-row"),
        pytest.param(
            {gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_UNPROVEN},
            id="prevented"),
        pytest.param(
            {gate_inconclusive.RESULT_KEY: True}, id="not-even-a-kind"),
    ],
)
def test_every_other_inconclusive_row_still_re_proves(monkeypatch, tmp_path, detail):
    model, artifacts = _grant_rig(monkeypatch, tmp_path)
    _state, token = common._auto_gate_context(model, "ksampler")

    _persist(tmp_path, artifacts, token, "INCONCLUSIVE", detail)

    assert common._auto_gate_context(model, "ksampler")[0] == "inconclusive"
    assert common.auto_gate_required(model, "ksampler") is True


def test_a_fail_row_stays_sticky_over_an_earlier_grant(monkeypatch, tmp_path):
    model, artifacts = _grant_rig(monkeypatch, tmp_path)
    _state, token = common._auto_gate_context(model, "ksampler")
    _persist(tmp_path, artifacts, token, "INCONCLUSIVE",
             {gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL})

    _persist(tmp_path, artifacts, token, "FAIL",
             {"quarantine_levers": ["slab_weights"]})

    assert common._auto_gate_context(model, "ksampler")[0] == "fail"


def test_a_new_context_re_proves_a_granted_combination(monkeypatch, tmp_path):
    """The release re-gate rule, no-material included: a capability-context or
    release change is a new context and costs one ceremony."""
    model, artifacts = _grant_rig(monkeypatch, tmp_path)
    _state, token = common._auto_gate_context(model, "ksampler")
    _persist(tmp_path, artifacts, token, "INCONCLUSIVE",
             {gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL})
    assert common._auto_gate_context(model, "ksampler")[0] == (
        gate_inconclusive.GRANTED_STATE)

    model.mesh.attention = "SAGE_AUTO"
    assert common._auto_gate_context(model, "ksampler")[0] == "stale"

    model.mesh.attention = "TORCH_FLASH"
    monkeypatch.setattr(ledger_mod, "__version__", "0.0.0-next")
    assert common._auto_gate_context(model, "ksampler")[0] == "stale"


def test_an_fsdp_proof_scope_never_reads_a_grant():
    """FSDP needs an exact clean-reload PASS, and a no-material classification
    is never made under a proof scope; a row claiming both gets the ceremony."""
    granted = _lookup({"verdict": "INCONCLUSIVE",
                       gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL})

    assert gate_inconclusive.granted_state(granted, False) == (
        gate_inconclusive.GRANTED_STATE)
    assert gate_inconclusive.granted_state(granted, True) == "inconclusive"


def _consent_call(lookup, process_verdict=None):
    from dgx_monarch.nodes.consent_projection import capacity_consent_authorization

    return capacity_consent_authorization(
        types.SimpleNamespace(), {"slab_weights": None}, {"loras": []},
        {"comfy": "commit", "digest": "d"}, "combo", {}, lookup, _TOKEN,
        process_verdict)


def _lookup(detail, *, damage_free=True):
    return ledger_mod.GateLedgerLookup("inconclusive", detail, damage_free)


def test_the_forcing_stock_arm_stays_silent_for_a_granted_combination(caplog):
    granted = _lookup({"verdict": "INCONCLUSIVE",
                       gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL})
    with caplog.at_level("INFO"):
        assert _consent_call(granted) is None

    [record] = caplog.records
    assert record.levelname == "INFO"
    assert "no exact Gate PASS" not in record.getMessage()
    assert "found nothing to gate" in record.getMessage()
    assert "stock residency" in record.getMessage()


@pytest.mark.parametrize(
    ("lookup", "process_verdict", "why"),
    [
        pytest.param(
            _lookup({"verdict": "INCONCLUSIVE",
                     gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_UNPROVEN}),
            None, "prevented", id="prevented"),
        # A RETESTING guard reads inconclusive too, and carries no kind.
        pytest.param(
            _lookup({"verdict": "RETESTING", "blocked_contexts": ["{}"]}),
            None, "retest in flight", id="retesting"),
        # Damage newer than the row could conceal a FAIL.
        pytest.param(
            _lookup({"verdict": "INCONCLUSIVE",
                     gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL},
                    damage_free=False),
            None, "damaged ledger", id="damage"),
        # The ceremony wrote its row and the driver then rejected the run.
        pytest.param(
            _lookup({"verdict": "INCONCLUSIVE",
                     gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL}),
            "ERROR", "process error", id="process-error"),
    ],
)
def test_every_other_dispatch_keeps_the_no_pass_warning(
        caplog, lookup, process_verdict, why):
    del why  # the id carries it
    with caplog.at_level("INFO"):
        assert _consent_call(lookup, process_verdict) is None

    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert "no exact Gate PASS" in record.getMessage()
    assert "forcing stock residency for this dispatch" in record.getMessage()


def test_this_session_own_no_material_verdict_still_grants(caplog):
    """The render right after the ceremony that wrote the row: the denial map
    holds INCONCLUSIVE for the token, and the memo says which INCONCLUSIVE."""
    common._record_process_gate_verdicts([_TOKEN], "INCONCLUSIVE", _no_material())
    granted = _lookup({"verdict": "INCONCLUSIVE",
                       gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL})

    with caplog.at_level("INFO"):
        assert _consent_call(granted, "INCONCLUSIVE") is None

    assert caplog.records[0].levelname == "INFO"


def test_the_pipeline_push_makes_the_same_decision(monkeypatch):
    """Twin of test_pipeline_gates_before_first_unproven_submit_and_quarantines
    (tests/test_pipeline.py): the same push and gate, with the levers still
    on."""
    from dgx_monarch.nodes.pipeline import RenderPipeline

    events = []
    model = type("Model", (), {})()
    model.mesh = type("Mesh", (), {"worker_args": {
        "lora_low_rss": True, "slab_weights": True}})()

    class Pending:
        def result(self):
            return {"samples": 1}

    monkeypatch.setattr(common, "auto_gate_required", lambda *_args: True)
    monkeypatch.setattr(
        common, "_maybe_auto_gate",
        lambda *_args: events.append("gate") or gate_inconclusive.NO_MATERIAL_VERDICT)

    def submit(*_args, **_kwargs):
        events.append("submit")
        assert model.mesh.worker_args == {
            "lora_low_rss": True, "slab_weights": True}
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    pipe = RenderPipeline(depth=2)
    pipe.push(model, {"kind": "ksampler"}, {}, 0.0, 2)

    assert pipe.drain() == [{"samples": 1}]
    assert events == ["gate", "submit"]
