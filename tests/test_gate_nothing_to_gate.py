"""The first-use gate settles "nothing to gate" from Worker evidence, with no proof render.

A LoRA-free graph on a family whose ``auto`` slab residency resolves stock can
only reach INCONCLUSIVE with nothing to compare, after an unload, a load and
two proof renders at the user's own shape (585.8 s on a pair for an H3 still
guide at 896x1152 and 362 frames, 2026-11-13). Where the driver reads that
graph, nodes/gate_nothing_to_gate.py runs the ceremony without its renders:
capacity stop, attested unload, load and empty-stack swap cycle, the
ceremony's own establish and classify, then the exercised row alone. This file
holds the predicate term by term, with the real readers too; the Worker
sequence; the row and tokens equal to a ceremony's where they must be and the
stated difference; every Worker doubt falling through to the full ceremony;
siblings untouched; later queues reading both rows alike; and the explicit
gate node still rendering.
"""
from __future__ import annotations

import json
import sys
import types

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
import dgx_monarch.runtime_provenance as runtime_provenance
from dgx_monarch import consent_store, first_render
from dgx_monarch.actor import model_store as model_store_mod
from dgx_monarch.gate_runtime import runtime_build_identity
from dgx_monarch.mesh_setup import SetupToken
from dgx_monarch.nodes import (
    gate_capacity_stop,
    gate_inconclusive,
    gate_nothing_to_gate,
    gate_process_state,
)
from dgx_monarch.topology import Topology

H3 = "minimax_h3"
# The real readers, bound before the ``readable`` fixture stands in for them.
_FAMILY = gate_nothing_to_gate._family
_CONSENT = gate_nothing_to_gate._rescue_consent_live


@pytest.fixture(autouse=True)
def _forget_process_state():
    denials = dict(gate_process_state._PROCESS_GATE_DENIALS)
    session = dict(common._AUTO_GATE_SESSION)
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
    yield
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
    gate_process_state._PROCESS_GATE_DENIALS = denials
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_SESSION.update(session)


@pytest.fixture
def readable(monkeypatch):
    """The driver reads an unvouched H3 header and finds no rescue consent."""
    reads = {"family": (H3, False), "consent": False}
    monkeypatch.setattr(gate_nothing_to_gate, "_family", lambda _model: reads["family"])
    monkeypatch.setattr(gate_nothing_to_gate, "_rescue_consent_live",
                        lambda _model: reads["consent"])
    return reads


def _model(loras=(), worker_args=None, request_loras=None, unet_name="minimax_h3.safetensors"):
    model = types.SimpleNamespace(
        unet_name=unet_name, options={}, loras=tuple(loras),
        mesh=types.SimpleNamespace(worker_args=dict(worker_args or {})))
    if request_loras is not None:
        model.request_dict = lambda: {"loras": list(request_loras)}
    return model


ULY2 = Topology(ulysses=2, world=2)
DECLINED = "the full ceremony runs, because the driver cannot rule out material to compare"

# --- the predicate, term by term ------------------------------------------------


def test_a_lora_free_unvouched_graph_under_slab_auto_has_nothing_to_gate(readable):
    reason = gate_nothing_to_gate.nothing_to_gate(_model(), {}, {"lora_low_rss": True}, ULY2)
    assert reason == ("no LoRA stack, one model, family minimax_h3 takes no slab "
                      "residency under slab_weights=auto")
    assert gate_nothing_to_gate.nothing_to_gate(
        _model(), {}, {"slab_weights": "auto"}, ULY2) == reason


def test_slab_off_and_a_forced_unvouched_family_have_nothing_to_gate(readable):
    assert gate_nothing_to_gate.nothing_to_gate(
        _model(), {}, {"slab_weights": False}, ULY2) == (
        "no LoRA stack, one model, slab residency is off")
    readable["family"] = ("wan", True)
    assert "the forced family wan takes no slab residency" in (
        gate_nothing_to_gate.nothing_to_gate(_model(), {}, {}, ULY2) or "")


@pytest.mark.parametrize(("case", "model", "request_", "effective", "topology", "reads"), [
    ("a second model", _model(), {"uncond_model": {"unet_name": "x"}}, {}, ULY2, {}),
    ("a LoRA on the model", _model(loras=[{"name": "a", "strength": 1.0}]), {}, {}, ULY2, {}),
    ("a LoRA in the model request", _model(request_loras=[{"name": "a"}]), {}, {}, ULY2, {}),
    ("comfy-managed", _model(), {}, {"comfy_managed": True}, ULY2, {}),
    ("an FSDP layout", _model(), {}, {}, Topology(ulysses=2, fsdp=True, world=2), {}),
    ("no resolved layout", _model(), {}, {}, None, {}),
    ("slab on", _model(), {}, {"slab_weights": True}, ULY2, {}),
    ("an unknown slab spelling", _model(), {}, {"slab_weights": "maybe"}, ULY2, {}),
    ("an unread family", _model(), {}, {}, ULY2, {"family": (None, False)}),
    ("a vouched family", _model(), {}, {}, ULY2, {"family": ("krea2", False)}),
    ("flux2", _model(), {}, {}, ULY2, {"family": ("flux2", False)}),
    ("a forced vouched family", _model(), {}, {}, ULY2, {"family": ("krea2", True)}),
    ("a live rescue consent", _model(), {}, {}, ULY2, {"consent": True}),
])
def test_anything_the_driver_cannot_read_for_certain_runs_the_ceremony(
        readable, caplog, case, model, request_, effective, topology, reads):
    readable.update(reads)
    with caplog.at_level("INFO"):
        assert gate_nothing_to_gate.nothing_to_gate(
            model, request_, effective, topology) is None, case
    # The declining term is named, so a hardware leg can say why a ceremony ran.
    assert [r for r in caplog.records if DECLINED in r.getMessage()], case


def _safetensors(path, keys):
    """A header-only checkpoint: one bf16 element per key."""
    header = {key: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
              for i, key in enumerate(keys)}
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * (2 * len(keys)))


def test_the_real_readers_find_nothing_to_gate_on_an_unvouched_header(
        monkeypatch, tmp_path, caplog):
    """No stand-in for either reader: a real header and an empty consent store."""
    monkeypatch.setattr(gate_nothing_to_gate, "_family", _FAMILY)
    monkeypatch.setattr(gate_nothing_to_gate, "_rescue_consent_live", _CONSENT)
    files = {"h3.safetensors": tmp_path / "h3.safetensors",
             "krea2.safetensors": tmp_path / "krea2.safetensors"}
    _safetensors(files["h3.safetensors"], ["video_patch_proj.weight", "audio_patch_proj.weight"])
    _safetensors(files["krea2.safetensors"], ["txtfusion.weight"])
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = (lambda kind, name: str(files[name])
                                  if kind == "diffusion_models" and name in files else None)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "consent_memo.json"))
    for name in ("DGXM_AUTO_RESCUE", "DGXM_ALLOW_SLAB_RESCUE"):
        monkeypatch.delenv(name, raising=False)

    h3 = _model(unet_name="h3.safetensors")
    with caplog.at_level("INFO"):
        assert gate_nothing_to_gate._family(h3) == (H3, False)
        assert gate_nothing_to_gate._rescue_consent_live(h3) is False
        assert gate_nothing_to_gate.nothing_to_gate(h3, {}, {}, ULY2) == (
            "no LoRA stack, one model, family minimax_h3 takes no slab residency "
            "under slab_weights=auto")
        krea2 = _model(unet_name="krea2.safetensors")
        assert gate_nothing_to_gate._family(krea2) == ("krea2", False)
        assert gate_nothing_to_gate.nothing_to_gate(krea2, {}, {}, ULY2) is None
    # The consent read ran to its answer: render_grant and live_grant each
    # swallow a fault, and both would say so at WARNING.
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]


def test_a_reader_that_fails_declines(monkeypatch):
    """No file: the header reads as no family. A raising consent read reads as live."""
    assert _FAMILY(_model()) == (None, False)

    def broken(_model):
        raise RuntimeError("consent store unreadable")

    monkeypatch.setattr("dgx_monarch.nodes.consent_rescue.render_grant", broken)
    assert _CONSENT(_model()) is True


# --- the ceremony rig (tests/test_gate_no_material.py builds the same one) -----


def _identity(unet_name, loras):
    artifacts = [{"kind": "diffusion_models", "name": unet_name, "signature": "model-sig"}]
    return {"digest": ledger_mod.artifact_set_signature(
        item["signature"] for item in artifacts).current,
        "comfy": "known", "artifacts": artifacts}


def _rig(monkeypatch, tmp_path, cycle=None, unet_name="minimax_h3.safetensors"):
    """``cycle`` is what each rank's swap cycle stamps; the default is no material on H3."""
    stamps = {"conclusive": False, "no_material": True,
              "reason": "no lora stack: lazy swap is not applicable",
              "family": H3, "slab_active": False} if cycle is None else cycle

    class Handle:
        setup_generation = 4
        setup_key = ("setup",)
        worker_args_key = ("worker",)
        world = 1
        source = None

        def __init__(self):
            self.calls = []

        def call_all(self, method, *args, **kwargs):
            self.calls.append(method)
            if method == "provenance_baseline":
                return [{"rank": 0, "world": 1, "setup_generation": args[0],
                         "source_manifest_sha256": self.source or (
                             runtime_provenance.cached_dgx_source_manifest_sha256()),
                         "runtime": runtime_build_identity()}]
            if method == "unload":
                return [{"unloaded": True}]
            if method == "gate_swap_cycle":
                if stamps == []:
                    return []
                return [{"rank": 0, "world": 1, "setup_generation": 4,
                         "host": "gn100", **stamps}]
            return [{"ok": True}]

    class Ledger:
        def __init__(self):
            self.records = []

        def begin_retest_required(self, *args):
            pass

        def record(self, *args):
            self.records.append(args)

    handle, ledger, renders, emits = Handle(), Ledger(), [], []
    model = types.SimpleNamespace(
        unet_name=unet_name, options={}, loras=(),
        mesh=types.SimpleNamespace(handle=handle, worker_args={"lora_low_rss": True}))

    def render(*_args, **_kwargs):
        renders.append(1)
        return {"samples": torch.zeros(1)}

    monkeypatch.setattr(gate_mod, "run_render", render)
    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(gate_mod, "render_setup_token",
                        lambda *_args: SetupToken(4, ("setup",), ("worker",)))
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gate_mod, "GateLedger", lambda directory: ledger)
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)
    monkeypatch.setattr("dgx_monarch.telemetry.emit",
                        lambda kind, *_args, **fields: emits.append((kind, fields)))
    return types.SimpleNamespace(handle=handle, ledger=ledger, model=model, renders=renders,
                                 emits=emits, reports=tmp_path / "dgxm_gate_reports.jsonl")


def _run(rig, origin="auto_first_use"):
    return gate_mod.run_identity_ceremony(
        rig.model, {"noise_seed": 1, "steps": 2, "cfg": 1.0},
        {"samples": torch.zeros(1)}, 1.0, 2, origin, run_id="x")


def _report(rig):
    return json.loads(rig.reports.read_text().strip().splitlines()[-1])


WORKER_CALLS = ("provenance_baseline", "unload", "load_model", "gate_swap_cycle")


def test_the_workers_settle_it_with_no_proof_render(monkeypatch, tmp_path, readable, caplog):
    rig = _rig(monkeypatch, tmp_path)
    with caplog.at_level("INFO"):
        result = _run(rig)

    assert rig.renders == []
    # Attested before and after; unload first so the cycle reads this checkpoint.
    assert [call for call in rig.handle.calls if call in WORKER_CALLS] == [
        "provenance_baseline", "unload", "load_model", "gate_swap_cycle", "provenance_baseline"]
    assert "apply_worker_args" not in rig.handle.calls
    assert result["verdict"] == "INCONCLUSIVE"
    assert gate_inconclusive.finished_no_material(result)
    assert gate_inconclusive.dispatch_verdict(result) == gate_inconclusive.NO_MATERIAL_VERDICT
    assert result["proof_renders"] == 0
    assert result["_gate_tokens"] == [result["_gate_token"]]
    [row] = rig.ledger.records
    assert row[3] == "INCONCLUSIVE"
    assert row[4][gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    assert row[4][gate_inconclusive.BASIS_KEY] == gate_inconclusive.BASIS_NO_RENDER
    [line] = [r.getMessage() for r in caplog.records if "nothing to gate" in r.getMessage()]
    assert ("the Workers' swap cycle stamped this with no proof render, as the driver "
            "read the graph (no LoRA stack") in line
    assert line.endswith("INCONCLUSIVE recorded, residency levers unchanged")
    [stamps] = [r.getMessage() for r in caplog.records if "swap-cycle stamps" in r.getMessage()]
    assert ("rank 0 on gn100: family minimax_h3, no_material True, slab_active False, "
            "setup generation 4") in stamps
    report = _report(rig)
    assert report[gate_inconclusive.BASIS_KEY] == gate_inconclusive.BASIS_NO_RENDER
    assert report[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    assert "latent" not in report and not any(key.startswith("_") for key in report)


def test_the_exercised_row_equals_a_ceremonys_and_no_sibling_is_touched(
        monkeypatch, tmp_path, readable, caplog):
    """Run today's ceremony (the family unread) and the no-render path on the same rig."""
    readable["family"] = (None, False)
    (tmp_path / "ran").mkdir()
    ran = _rig(monkeypatch, tmp_path / "ran")
    with caplog.at_level("INFO"):
        ceremony = _run(ran)
    assert len(ran.renders) == 2 and "unload" in ran.handle.calls
    assert gate_inconclusive.finished_no_material(ceremony)
    ceremony_line = [r.getMessage() for r in caplog.records if "nothing to gate" in r.getMessage()]
    assert ceremony_line and "no proof render" not in ceremony_line[0]

    readable["family"] = (H3, False)
    (tmp_path / "skipped").mkdir()
    skipped_rig = _rig(monkeypatch, tmp_path / "skipped")
    skipped = _run(skipped_rig)

    # The exercised row: same key, artifacts, commit, verdict, context and kind.
    [ours] = skipped_rig.ledger.records
    theirs = ran.ledger.records[0]
    assert ours[:4] + ours[5:] == theirs[:4] + theirs[5:]
    rendered_only = {"max_abs_latent_diff", gate_inconclusive.BASIS_KEY}
    assert ({k: v for k, v in ours[4].items() if k not in rendered_only}
            == {k: v for k, v in theirs[4].items() if k not in rendered_only})
    assert gate_inconclusive.BASIS_KEY not in theirs[4] and ours[4]["max_abs_latent_diff"] is None
    # The ceremony also wrote the Fleet and explicit-on sibling rows; this path
    # wrote none of them, so a PASS held there stays.
    assert len(ran.ledger.records) > 1
    assert all(row[5] != ours[5] for row in ran.ledger.records[1:])
    assert skipped["_gate_token"] == ceremony["_gate_token"]
    assert skipped["_gate_tokens"] == [ceremony["_gate_token"]]
    # Publication denies and remembers that one token alone.
    token = tuple(skipped["_gate_token"])
    assert gate_inconclusive.session_denial_tokens(skipped["_gate_tokens"], skipped) == (token,)
    gate_inconclusive.remember_kind(skipped["_gate_tokens"], skipped)
    assert gate_inconclusive.cached_verdict(token, "INCONCLUSIVE") == (
        gate_inconclusive.NO_MATERIAL_VERDICT)

    same = ("verdict", "origin", "run_id", "model", "loras", "proof_kind", "cross_mode",
            gate_inconclusive.RESULT_KEY, "inconclusive_reasons", "swap_transitions",
            "seed", "sampler", "scheduler", "steps", "cfg")
    mine, reference = _report(skipped_rig), _report(ran)
    assert {k: mine.get(k) for k in same} == {k: reference.get(k) for k in same}
    # The stated difference: nothing rendered.
    assert mine[gate_inconclusive.BASIS_KEY] == gate_inconclusive.BASIS_NO_RENDER
    assert gate_inconclusive.BASIS_KEY not in reference
    assert mine["proof_renders"] == 0 and "proof_renders" not in reference
    assert mine["latents_identical"] is None and reference["latents_identical"] is True

    # The gate telemetry event: the same fields but the latent difference,
    # which only a render can measure.
    [(_kind, ours_event)] = [e for e in skipped_rig.emits if e[0] == "gate"]
    [(_kind, their_event)] = [e for e in ran.emits if e[0] == "gate"]
    assert {k: v for k, v in ours_event.items() if k != "max_diff"} == {
        k: v for k, v in their_event.items() if k != "max_diff"}
    assert ours_event["max_diff"] is None and their_event["max_diff"] == 0.0


def _published(result, siblings):
    """The process maps after the auto path publishes ``result``, from an empty
    session in which every sibling context holds a session PASS."""
    from dgx_monarch.nodes.gate_process_verdicts import (
        process_gate_verdict,
        record_process_gate_verdicts,
    )

    gate_process_state._PROCESS_GATE_DENIALS = {}
    gate_process_state._AUTO_GATE_SESSION.clear()
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
    gate_process_state._AUTO_GATE_SESSION.update(dict.fromkeys(siblings, "PASS"))
    tokens = [tuple(token) for token in result["_gate_tokens"]]
    record_process_gate_verdicts(tokens, "INCONCLUSIVE", result)
    exercised = tuple(result["_gate_token"])
    return {"exercised": process_gate_verdict(exercised),
            "dispatch": gate_inconclusive.cached_verdict(
                exercised, process_gate_verdict(exercised)),
            "siblings": [process_gate_verdict(token) for token in siblings]}


def test_publication_and_the_closing_notice_against_a_ceremonys(
        monkeypatch, tmp_path, readable):
    readable["family"] = (None, False)
    (tmp_path / "ran").mkdir()
    ceremony = _run(_rig(monkeypatch, tmp_path / "ran"))
    readable["family"] = (H3, False)
    (tmp_path / "skipped").mkdir()
    skipped = _run(_rig(monkeypatch, tmp_path / "skipped"))
    siblings = [tuple(token) for token in ceremony["_gate_tokens"][1:]]
    assert siblings

    theirs, ours = _published(ceremony, siblings), _published(skipped, siblings)
    # The exercised context: denied in-process as INCONCLUSIVE, dispatched as no material.
    assert ours["exercised"] == theirs["exercised"] == "INCONCLUSIVE"
    assert ours["dispatch"] == theirs["dispatch"] == gate_inconclusive.NO_MATERIAL_VERDICT
    # The stated difference: a ceremony revokes the siblings' session PASS;
    # this path proved nothing about them and leaves it.
    assert theirs["siblings"] == [None] * len(siblings)
    assert ours["siblings"] == ["PASS"] * len(siblings)

    seen = []
    monkeypatch.setattr(first_render, "notice",
                        lambda phase, message, note, **kw: seen.append(
                            (phase, message, kw.get("severity", "info"))))
    first_render.gate_finished(ceremony)
    first_render.gate_finished(skipped)
    (phase_c, text_c, level_c), (phase_s, text_s, level_s) = seen
    assert phase_c == phase_s == "gate_done"
    assert "identity gate INCONCLUSIVE" in text_c and level_c == "warn"
    # The stated difference: the closing line says no proof render ran.
    assert "ran no proof render" in text_s and level_s == "info"


def test_the_whole_path_runs_on_the_default_readers(monkeypatch, tmp_path, caplog):
    """No stand-in for the family or consent read: a real header file and an empty store."""
    checkpoint = tmp_path / "h3.safetensors"
    _safetensors(checkpoint, ["video_patch_proj.weight", "audio_patch_proj.weight"])
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = (lambda kind, name: str(checkpoint)
                                  if kind == "diffusion_models" and name == checkpoint.name
                                  else None)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "consent_memo.json"))
    for name in ("DGXM_AUTO_RESCUE", "DGXM_ALLOW_SLAB_RESCUE"):
        monkeypatch.delenv(name, raising=False)
    rig = _rig(monkeypatch, tmp_path, unet_name=checkpoint.name)
    with caplog.at_level("INFO"):
        result = _run(rig)
    assert rig.renders == []
    assert result[gate_inconclusive.BASIS_KEY] == gate_inconclusive.BASIS_NO_RENDER
    [row] = rig.ledger.records
    assert row[4][gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    assert not [r for r in caplog.records if DECLINED in r.getMessage()]


@pytest.mark.parametrize(("case", "cycle"), [
    ("the Workers hold a vouched family", {
        "conclusive": False, "no_material": True,
        "reason": "no lora stack: lazy swap is not applicable",
        "family": "krea2", "slab_active": False}),
    ("slab residency is active", {
        "conclusive": False, "no_material": True,
        "reason": "no lora stack: lazy swap is not applicable",
        "family": H3, "slab_active": True}),
    ("no rank answered", []),
    ("a rank did not stamp no material", {
        "conclusive": False, "reason": "lora_low_rss is off on this worker",
        "family": H3, "slab_active": False}),
])
def test_any_other_worker_answer_runs_the_full_ceremony(
        monkeypatch, tmp_path, readable, caplog, case, cycle):
    rig = _rig(monkeypatch, tmp_path, cycle=cycle)
    with caplog.at_level("INFO"):
        try:
            result = _run(rig)
        except Exception:  # the full ceremony's own answer; not this path's to judge
            result = {}
    # The full ceremony ran: its stock-lineage render precedes its swap cycle.
    assert rig.renders, case
    assert gate_inconclusive.BASIS_KEY not in result, case
    assert not [row for row in rig.ledger.records
                if isinstance(row[4], dict) and gate_inconclusive.BASIS_KEY in row[4]], case
    assert [r for r in caplog.records if DECLINED in r.getMessage()], case


def _path_alone(rig):
    """This path by itself, as nodes/gate.py calls it, with no ceremony after it."""
    return gate_nothing_to_gate.record_if_provable(
        runtime=vars(gate_mod), model=rig.model,
        request={"noise_seed": 1, "steps": 2, "cfg": 1.0}, latent={"samples": torch.zeros(1)},
        cfg_value=1.0, origin="auto_first_use", run_id="x", handle=rig.handle)


@pytest.mark.parametrize("cycle", [
    {"conclusive": False, "no_material": True, "family": "krea2", "slab_active": False},
    {"conclusive": False, "no_material": True, "family": H3, "slab_active": True},
    [],
])
def test_a_return_after_the_load_leaves_the_workers_empty(monkeypatch, tmp_path, readable, cycle):
    """The ceremony's capacity stop charges the DiT as a load to come, so a model
    this path loaded must not stay resident under it."""
    rig = _rig(monkeypatch, tmp_path, cycle=cycle)
    assert _path_alone(rig) is None
    assert "load_model" in rig.handle.calls and rig.handle.calls[-1] == "unload"
    assert rig.ledger.records == []


@pytest.mark.parametrize("cycle", [
    {"conclusive": False, "no_material": True, "family": "krea2", "slab_active": False},
    {"conclusive": False, "no_material": True, "family": H3, "slab_active": True},
    [],
    {"conclusive": False, "reason": "lora_low_rss is off on this worker", "family": H3,
     "slab_active": False},
])
def test_the_full_ceremony_prices_a_cold_state_after_a_fall_through(
        monkeypatch, tmp_path, readable, cycle):
    """The last Worker call before the ceremony's own capacity stop is an unload."""
    rig = _rig(monkeypatch, tmp_path, cycle=cycle)
    priced = gate_capacity_stop.stop_unless_request_fits

    def marked(*args, **kwargs):
        rig.handle.calls.append("STOP")
        return priced(*args, **kwargs)

    monkeypatch.setattr(gate_capacity_stop, "stop_unless_request_fits", marked)
    try:
        _run(rig)
    except Exception:  # the full ceremony's own answer
        pass
    stops = [index for index, call in enumerate(rig.handle.calls) if call == "STOP"]
    assert len(stops) == 2  # this path's, then the full ceremony's
    assert "load_model" in rig.handle.calls[stops[0]:stops[1]]
    assert rig.handle.calls[stops[1] - 1] == "unload"


def test_a_declined_graph_is_not_frozen_by_this_path(monkeypatch, tmp_path, readable):
    """The predicate reads the caller's request and latent; only the ceremony clones them."""
    rig = _rig(monkeypatch, tmp_path)
    frozen = []
    freeze = gate_mod.freeze_transaction

    def counted(transaction):
        frozen.append(1)
        return freeze(transaction)

    monkeypatch.setattr(gate_mod, "freeze_transaction", counted)
    readable["family"] = ("krea2", False)
    _run(rig)
    assert len(rig.renders) >= 2 and len(frozen) == 1  # the ceremony's own copy
    frozen.clear()
    readable["family"] = (H3, False)
    (tmp_path / "again").mkdir()
    _run(_rig(monkeypatch, tmp_path / "again"))
    assert len(frozen) == 1  # the path's copy, taken once it had a basis


@pytest.mark.parametrize("case", ["a LoRA stack", "a second model", "a vouched family"])
def test_this_path_never_freezes_a_graph_it_declines(monkeypatch, tmp_path, readable, case):
    rig = _rig(monkeypatch, tmp_path)
    frozen = []
    freeze = gate_mod.freeze_transaction
    monkeypatch.setattr(gate_mod, "freeze_transaction",
                        lambda transaction: frozen.append(1) or freeze(transaction))
    request = {"noise_seed": 1, "steps": 2, "cfg": 1.0}
    if case == "a LoRA stack":
        rig.model.loras = ({"name": "a.safetensors", "strength": 1.0},)
    elif case == "a second model":
        request["uncond_model"] = {"unet_name": "u.safetensors"}
    else:
        readable["family"] = ("krea2", False)
    assert gate_nothing_to_gate.record_if_provable(
        runtime=vars(gate_mod), model=rig.model, request=request,
        latent={"samples": torch.zeros(1)}, cfg_value=1.0, origin="auto_first_use",
        run_id="x", handle=rig.handle) is None
    assert frozen == [] and rig.handle.calls == []


def test_a_return_before_the_load_unloads_nothing_more(monkeypatch, tmp_path, readable):
    rig = _rig(monkeypatch, tmp_path)
    rig.handle.source = "0" * 64
    assert _path_alone(rig) is None
    assert rig.handle.calls == ["provenance_baseline"]


def test_a_provenance_mismatch_meets_the_ceremony_s_own_refusal(monkeypatch, tmp_path, readable):
    rig = _rig(monkeypatch, tmp_path)
    rig.handle.source = "0" * 64
    with pytest.raises(Exception, match="provenance"):
        _run(rig)
    assert rig.renders == []
    assert "load_model" not in rig.handle.calls and "unload" not in rig.handle.calls
    assert not [row for row in rig.ledger.records
                if isinstance(row[4], dict) and gate_inconclusive.BASIS_KEY in row[4]]


def test_a_capacity_stop_ends_there_as_in_the_ceremony(monkeypatch, tmp_path, readable):
    rig = _rig(monkeypatch, tmp_path)
    priced = []

    def stop(*_args, **_kwargs):
        priced.append(1)
        raise gate_capacity_stop.CeremonyCapacityStop("[class C] this request cannot fit")

    monkeypatch.setattr(gate_capacity_stop, "stop_unless_request_fits", stop)
    with pytest.raises(gate_capacity_stop.CeremonyCapacityStop):
        _run(rig)
    assert priced == [1] and rig.renders == []
    assert not {"provenance_baseline", "unload", "load_model"} & set(rig.handle.calls)


def test_an_fsdp_proof_scope_declines(monkeypatch, tmp_path, readable):
    rig = _rig(monkeypatch, tmp_path)
    assert gate_nothing_to_gate.record_if_provable(
        runtime=vars(gate_mod), model=rig.model, request={"noise_seed": 1, "steps": 2},
        latent={"samples": torch.zeros(1)}, cfg_value=1.0, origin="auto_first_use",
        run_id="x", handle=rig.handle, fsdp_scope_state={"required": True}) is None
    assert rig.handle.calls == []


def test_the_explicit_gate_node_still_renders(monkeypatch, tmp_path, readable):
    rig = _rig(monkeypatch, tmp_path)
    result = _run(rig, origin="explicit")
    assert len(rig.renders) == 2
    assert gate_inconclusive.BASIS_KEY not in result


def test_a_failed_read_runs_the_ceremony(monkeypatch, tmp_path, readable):
    rig = _rig(monkeypatch, tmp_path)

    def broken(*_args, **_kwargs):
        raise OSError("the topology could not be resolved")

    monkeypatch.setattr(gate_nothing_to_gate, "_resolved", broken)
    result = _run(rig)
    assert len(rig.renders) == 2
    assert gate_inconclusive.BASIS_KEY not in result
    assert gate_inconclusive.finished_no_material(result)


# --- later queues and the render that follows ---------------------------------

_TOKEN = ("combo", "artifacts", "commit", "context")


def _lookup(detail):
    return ledger_mod.GateLedgerLookup("inconclusive", detail, True)


@pytest.mark.parametrize("basis", [None, gate_inconclusive.BASIS_NO_RENDER])
def test_later_queues_read_both_rows_alike(basis, caplog):
    detail = {"verdict": "INCONCLUSIVE",
              gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL,
              **({} if basis is None else {gate_inconclusive.BASIS_KEY: basis})}
    assert gate_inconclusive.granted_state(_lookup(detail), False) == (
        gate_inconclusive.GRANTED_STATE)
    assert gate_inconclusive.dispatch_grants_skip(_lookup(detail), _TOKEN, None)
    from dgx_monarch.nodes.consent_projection import capacity_consent_authorization

    with caplog.at_level("INFO"):
        # None: no consent envelope, so authorize_normal_render returns its
        # stock policy (slab_weights and lora_low_rss off) for the dispatch.
        assert capacity_consent_authorization(
            types.SimpleNamespace(), {"slab_weights": None}, {"loras": []},
            {"comfy": "commit", "digest": "d"}, "combo", {}, _lookup(detail), _TOKEN,
            None) is None
    assert "found nothing to gate" in caplog.records[0].getMessage()


def test_the_closing_notice_says_no_proof_render_ran(monkeypatch):
    seen = []
    monkeypatch.setattr(first_render, "notice",
                        lambda phase, message, note, **_kw: seen.append((phase, message)))
    first_render.gate_finished({"verdict": "INCONCLUSIVE",
                                gate_inconclusive.BASIS_KEY: gate_inconclusive.BASIS_NO_RENDER})
    [(phase, message)] = seen
    assert phase == "gate_done"
    assert "ran no proof render" in message and "nothing to compare" in message
    assert first_render.NO_MATERIAL_BASIS_KEY == gate_inconclusive.BASIS_KEY
