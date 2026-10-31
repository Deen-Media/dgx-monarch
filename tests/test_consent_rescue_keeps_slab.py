"""A capacity rescue keeps slab residency where stock residency cannot load.

docs/DESIGN.md, "Where the ceremony cannot help", and TROUBLESHOOTING #5: the
identity gate has no stock reference, so it stays INCONCLUSIVE with
cross_mode CAPACITY, and an active rescue consent keeps slab_weights and
lora_low_rss for the render. The consent is one thing whichever way it was
given: the panel click, the auto-rescue toggle, DGXM_AUTO_RESCUE or
DGXM_ALLOW_SLAB_RESCUE.

On 2026-10-30 the MiniMax H3 bf16 suite found every hop of that missing. The
ceremony's own INCONCLUSIVE row read back as a denial, so the rule never fired
and the render was refused under stock; the panel hid the card the loader
offered after it; and an actor's offer on a layout where the driver holds no
rank became a card nobody could list or accept. Each test below drives the
real writer and the real reader of the gate ledger, the consent store and the
pending registry.
"""
from __future__ import annotations

import json
import logging
import struct
import sys
import types

import pytest
import torch

import dgx_monarch.nodes.gate as gate_mod
from dgx_monarch import (
    consent_descriptor,
    consent_pending,
    consent_store,
    driver_footprint,
    gate_ledger,
    mesh_residency,
    mesh_rpc,
    mesh_safety,
    mesh_setup,
    runtime_provenance,
)
from dgx_monarch.actor import model_store as model_store_mod
from dgx_monarch.actor import store_residency
from dgx_monarch.capacity_fit import StockFit
from dgx_monarch.gate_audit import measured_tag
from dgx_monarch.gate_audit_vocab import waiver_key
from dgx_monarch.log import get_logger
from dgx_monarch.mesh_safety import StockLoadCapacityError
from dgx_monarch.nodes import (
    consent_projection,
    consent_quarantine,
    consent_rescue,
    consent_routes,
    gate_identity,
    gate_inconclusive,
    gate_quarantine_scope,
    loader_preflight,
)
from dgx_monarch.refusal import RefusalClass, refusal
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
    _cross_mode_rig,
)

NAME = "minimax_h3_fl2va_bf16.safetensors"
LORA = "style.safetensors"
KIND = "rescue-slab"
FILE_BYTES = 66_280_487_368            # the 61.7 GiB bf16 file
MEM_AVAILABLE = int(114.5 * 2 ** 30)    # stock needs 134.6 GiB on this box; slab fits
SLAB_ON = {"lora_low_rss": True, "slab_weights": True}
TYPED = "StockLoadCapacityError/class-C guard=stock_load_preflight"

# A worker's stock load refusal carries these, so the ceremony can record the
# cross leg's certificate row beside its verdict (test_gate_capacity_certificate).
_MEASURED = {
    "probe": "stock_load_preflight",
    "checkpoint_bytes": 66266529792,
    "mem_available_bytes": 41234567168,
    "driver_projection_bytes": None,
    "worker_projection_bytes": 66266529792,
    "headroom_bytes": -25031962624,
}
_CERTIFICATE = {
    "algorithm": "memcmp-tensor-sampled-v1",
    "digest": "e4" + "a" * 62,
    "complete": True,
    "tensors_verified": 1042,
    "tensors_total": 1042,
    "bytes_verified": 66266529792,
    "checkpoint_bytes": 66266529792,
    "artifact_signature": "9d1c0b47ee2a6f3184c5b70a",
    "file_identity": "66306:12583041:66266529792:1754300000000000000:1754300000000000000",
}
_DOES_NOT_FIT = StockFit(False, True, FILE_BYTES, 47 * 2 ** 30)


@pytest.fixture(autouse=True)
def rig(monkeypatch, tmp_path):
    """A UMA driver host the bf16 file cannot stock-load on, a stub
    folder_paths over real (tiny) files, a private consent store, ledger and
    pending registry, and no headless consent in the environment."""
    _forget()
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "consent_memo.json"))
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(consent_routes, "_RATE",
                        {"tokens": float(consent_routes.RATE_LIMIT_N), "t": 0.0})
    for folder, name, weights in (
            ("diffusion_models", NAME, torch.arange(4096, dtype=torch.bfloat16)),
            ("loras", LORA, torch.ones(64, dtype=torch.bfloat16))):
        target = tmp_path / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_safetensors(target, {"blocks.0.w": weights})
    output = tmp_path / "output"
    output.mkdir()
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda folder, name: (  # type: ignore[attr-defined]
        str(tmp_path / folder / name) if (tmp_path / folder / name).exists() else None)
    fp.get_output_directory = lambda: str(output)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    checkpoint = str(tmp_path / "diffusion_models" / NAME)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: MEM_AVAILABLE)
    monkeypatch.setattr(driver_footprint, "file_size_bytes",
                        lambda path: FILE_BYTES if str(path) == checkpoint else 0)
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: ("minimax_h3", "bf16"))
    yield types.SimpleNamespace(tmp=tmp_path, output=output, checkpoint=checkpoint)
    _forget()


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        raw = tensor.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": "BF16", "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for raw in blobs:
            handle.write(raw)


def _forget():
    consent_pending.clear_all()
    consent_store.forget_headless()
    gate_quarantine_scope.reset_quarantine_scope()
    loader_preflight.reset_memos()


def _combo():
    return consent_rescue.combo_identity(
        NAME, {}, str(sys.modules["folder_paths"].get_full_path("diffusion_models", NAME)))


def _rows(rig, verdict=None):
    entries, _damage = gate_ledger.GateLedger(str(rig.output)).entries_with_integrity()
    return [entry for _line, entry in entries
            if verdict is None or entry.get("verdict") == verdict]


class _Handle:
    """The config-only reads the loader node and the consent reads make."""

    def __init__(self, co_resident: bool, worker_args: dict):
        # A this_host() mesh runs rank 0 in a child of the driver; a cluster
        # whose ranks are all on another host leaves this host none.
        self.owns_hosts = not co_resident
        self.gpus_per_host = 1
        self.config = types.SimpleNamespace(hosts=(), worker_args={})
        self.active_worker_args = dict(worker_args)

    def effective_worker_args(self, worker_args=None):
        return {**self.config.worker_args, **dict(worker_args or {})}


class _LoaderMesh:
    def __init__(self, worker_args=None, *, co_resident=True):
        self.worker_args = {"lora_low_rss": True} if worker_args is None else worker_args
        self.handle = _Handle(co_resident, self.worker_args)
        self.topology_preset = "uly2"
        self.world = 2


def _load_refusal(mesh) -> driver_footprint.DriverFootprintCapacityError:
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NAME, {}, None)
    return raised.value


def _click():
    """The panel click on the card the loader node raises for this file."""
    assert "Load with slab residency" in str(_load_refusal(_LoaderMesh()))
    (card,) = consent_routes.consent_state()["pending"]
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200, payload
    return card


def _consent(source: str, monkeypatch) -> str:
    """Give one active rescue consent the way an operator gives it; return its source."""
    if source == "panel":
        _click()
        return "panel"
    if source == "env":
        monkeypatch.setenv("DGXM_ALLOW_SLAB_RESCUE", "1")
        return "env"
    if source == "auto_env":
        monkeypatch.setenv(consent_pending.AUTO_RESCUE_ENV, "1")
        return "auto_rescue"
    assert source == "toggle"
    assert consent_routes.handle_action({"action": "auto_rescue", "value": True})[0] == 200
    return "auto_rescue"


def _withdraw(source: str, monkeypatch) -> None:
    """End that consent: revoke the memo, unset the variable or turn the toggle off."""
    if source == "panel":
        for row in consent_store.list_active():
            assert consent_routes.handle_action({"action": "revoke", "key": row["key"]})[0] == 200
    elif source == "env":
        monkeypatch.delenv("DGXM_ALLOW_SLAB_RESCUE")
    elif source == "auto_env":
        monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV)
    else:
        assert consent_routes.handle_action({"action": "auto_rescue", "value": False})[0] == 200


def _typed_capacity_refusal() -> StockLoadCapacityError:
    """A worker's stock load refusal, class tag first and numbers last."""
    return StockLoadCapacityError(refusal(
        RefusalClass.CAPACITY,
        f"stock residency cannot load {NAME}: it needs 134.6 GiB of unified memory "
        "and 113.5 GiB is available, and slab residency is off for this leg",
        guard="stock_load_preflight", waivable=False) + " " + measured_tag(_MEASURED))


def _cycle(loras):
    row = {"family": "minimax_h3", "slab_active": True, "certificate": dict(_CERTIFICATE)}
    if loras:
        return [{**row, "conclusive": True, "transitions": ["lazy"]}]
    return [{**row, "conclusive": False, "transitions": [],
             "reason": "no lora stack: lazy swap is not applicable"}]


def _identity(rig):
    def identity(unet_name, loras):
        items = [{"kind": "diffusion_models", "name": unet_name, "signature":
                  gate_ledger.artifact_signature(str(rig.tmp / "diffusion_models" / unet_name))}]
        items.extend(
            {"kind": "loras", "name": entry["name"], "signature":
             gate_ledger.artifact_signature(str(rig.tmp / "loras" / entry["name"]))}
            for entry in loras or [])
        digest = gate_ledger.artifact_set_signature(item["signature"] for item in items)
        return {"digest": digest.current, "comfy": "known", "artifacts": items}
    return identity


def _ceremony(monkeypatch, rig, *, loras=(), worker_args=None, b=None, a=None, cross=None):
    """One first-use ceremony on the bf16 file: two slab legs, then a stock
    reference that ``cross`` refuses. The real ledger records it."""
    wa = dict(SLAB_ON) if worker_args is None else worker_args
    renders = [torch.zeros(1) if b is None else b, torch.zeros(1) if a is None else a,
               _typed_capacity_refusal() if cross is None else cross]
    _handle, _fake, model = _cross_mode_rig(
        monkeypatch, rig.tmp, renders, wa, cycle_response=_cycle(loras))
    model.unet_name = NAME
    model.loras = tuple({"name": name, "strength": 1.0} for name in loras)
    monkeypatch.setattr(gate_mod, "GateLedger", gate_ledger.GateLedger)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(rig.output))
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity(rig))
    result = gate_mod.run_identity_ceremony(
        model, {"noise_seed": 1, "steps": 2, "cfg": 1.0}, {"samples": torch.zeros(1)},
        1.0, 2, "auto_first_use", run_id="x")
    return result, model


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def settlement_log():
    """The ceremony's verdict lines and every quarantine line, in order. These
    loggers do not propagate, so the handler goes on each one."""
    capture = _Capture()
    loggers = [get_logger(f"dgx_monarch.nodes.{name}") for name in (
        "gate_inconclusive", "gate_quarantine_scope", "consent_rescue")]
    for logger in loggers:
        logger.addHandler(capture)
    yield capture.messages
    for logger in loggers:
        logger.removeHandler(capture)


def _authorize(model):
    """The final residency boundary for the user's render after the ceremony."""
    combo, signature = _combo()
    return consent_projection.capacity_consent_authorization(
        model, dict(model.mesh.worker_args), {"unet_name": NAME, "options": {}, "loras": []},
        {"comfy": "known", "digest": signature.current}, combo, {},
        gate_ledger.GateLedgerLookup("inconclusive", None, False), ("token",), "INCONCLUSIVE")


# One consent, whichever way it was given ------------------------------------


def test_the_click_the_toggle_and_both_variables_produce_one_memo_shape(monkeypatch, rig):
    """Every later rule reads one record: the same columns, the same narrow
    context and the same artifact identity, whatever granted it. Only the
    channel, the id and the wording differ, and the headless ones are never
    written to the memo file."""
    context = consent_rescue.memo_context(NAME, {}, ())
    combo, signature = _combo()
    _click()
    panel = consent_store.lookup(KIND, rig.checkpoint, context)
    assert panel is not None
    _withdraw("panel", monkeypatch)
    records = {"panel": panel}
    for source in ("toggle", "auto_env", "env"):
        expected = _consent(source, monkeypatch)
        grant = consent_pending.resolve(kind=KIND, path=rig.checkpoint, context=context)
        assert grant is not None and grant.consent_source == expected and grant.key == ""
        record = consent_store.headless_memo(
            KIND, rig.checkpoint, context, source=expected, target_guard="stock_load_preflight",
            waiver_class="C", reason="unused: the memo exists already")
        assert record.id == grant.id
        records[source] = record
        _withdraw(source, monkeypatch)
    assert consent_store.read()["consents"] == {}
    columns = set(panel.as_json())
    for source, record in records.items():
        assert set(record.as_json()) == columns, source
        # The class C columns a persisted memo must carry, checked by the reader the file uses.
        assert consent_store.ConsentRecord.from_json(record.as_json()) == record, source
        assert (record.kind, record.waiver_class, record.memo_context) == (KIND, "C", context)
        assert (record.combo_key, record.artifacts, record.artifacts_legacy,
                record.artifacts_legacy_complete) == (
                    combo, signature.current, signature.legacy, signature.legacy_complete)
        assert record.file_identity == consent_store.file_identity(rig.checkpoint)
        assert (record.path, record.unet_name, record.artifact) == (rig.checkpoint, NAME, NAME)
        assert record.ledger_key == waiver_key(KIND, combo)
        assert record.capability_context is None and record.loras == 0
    assert [records[source].consent_source for source in records] == [
        "panel", "auto_rescue", "auto_rescue", "env"]


def test_a_headless_memo_lasts_for_the_process_and_joins_every_row(monkeypatch, rig):
    """One id across the WAIVER row, the certificate evidence and every
    resolve, as a click's memo has; gone with its source, and never listed
    as a panel consent the operator could revoke while the variable stands."""
    context = consent_rescue.memo_context(NAME, {}, ())
    _consent("env", monkeypatch)
    first = consent_pending.resolve(kind=KIND, path=rig.checkpoint, context=context)
    again = consent_pending.resolve(kind=KIND, path=rig.checkpoint, context=context)
    assert first is not None and again is not None and first.id == again.id
    assert consent_routes.consent_state()["active"] == []

    mesh = _LoaderMesh(co_resident=False)
    load = consent_projection.project_for_loader(mesh, NAME, {})
    assert load is not None and load.consent_id == first.id
    assert mesh.worker_args["slab_weights"] is True
    (waiver,) = _rows(rig, "WAIVER")
    assert waiver["consent_id"] == first.id and waiver["consent_source"] == "env"

    _withdraw("env", monkeypatch)
    assert consent_pending.resolve(kind=KIND, path=rig.checkpoint, context=context) is None


# The keep-slab rule ---------------------------------------------------------


@pytest.mark.parametrize("source", ["panel", "toggle", "auto_env", "env"])
def test_a_consented_capacity_inconclusive_keeps_slab_and_renders_under_it(
        monkeypatch, rig, settlement_log, source):
    """2026-10-30 cases 1 and 2: the slab legs agree, stock cannot load, and
    the consent is active. The gate records INCONCLUSIVE with cross_mode
    CAPACITY, takes no lever, and the render proceeds under slab."""
    expected = _consent(source, monkeypatch)
    result, model = _ceremony(monkeypatch, rig)

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "CAPACITY"
    assert result["cross_mode"]["capacity_detail"] == TYPED
    settled = result[gate_inconclusive.RESCUE_RESULT_KEY]
    assert settled["consent_source"] == expected
    verdict = gate_inconclusive.dispatch_verdict(result)
    assert verdict == gate_inconclusive.CAPACITY_RESCUE_DISPATCH_VERDICT
    assert verdict in gate_inconclusive.NO_QUARANTINE_VERDICTS
    assert model.mesh.worker_args == SLAB_ON
    # One line says why, and no quarantine line follows it.
    kept = [line for line in settlement_log if line.startswith("identity gate kept slab")]
    assert len(kept) == 1
    assert kept[0].startswith(
        f"identity gate kept slab residency for {NAME}: stock residency cannot load it ({TYPED})")
    assert f"capacity-rescue consent {settled['consent_id']} from {expected} is active" in kept[0]
    assert not [line for line in settlement_log if line.startswith("auto-gate")]

    # A later render of the combination in this session reads the same rows.
    gate_identity.quarantine_unproven_paths(model)
    assert model.mesh.worker_args == SLAB_ON
    authorization = _authorize(model)
    assert authorization.residency_mode == mesh_residency.RESIDENCY_MODE_CAPACITY_CONSENT
    assert authorization.worker_args["slab_weights"] is True
    assert authorization.worker_args["lora_low_rss"] is True
    assert authorization.residency_grant["consent"]["id"] == settled["consent_id"]
    assert "gate_token" not in authorization.residency_grant


def test_the_ledger_records_the_settlement_and_its_certificate_as_today(monkeypatch, rig):
    """The rows carry the FSDP capacity stop's vocabulary and no lever; the
    certificate row sits beside them; nothing reads as PASS."""
    _consent("env", monkeypatch)
    result, _model = _ceremony(monkeypatch, rig)
    combo, signature = _combo()

    trust = [entry for entry in _rows(rig) if entry.get("key") == combo]
    assert [entry["verdict"] for entry in trust] == ["RETESTING", "INCONCLUSIVE", "INCONCLUSIVE"]
    for row in trust[1:]:
        assert row["artifacts"] == signature.current
        assert (row["cross_mode"], row["capacity_detail"]) == ("CAPACITY", TYPED)
        assert (row["quarantine_levers"], row["loras"], row["max_abs_latent_diff"]) == ([], 0, 0.0)
        assert consent_quarantine.stock_reference_unavailable(row)
    (certificate,) = _rows(rig, "CAPACITY_CERTIFIED")
    assert certificate["key"] == "audit:capacity:" + combo
    assert certificate["certificate_origin"] == "gate_cross_leg"
    assert not _rows(rig, "PASS")
    report = (rig.output / "dgxm_gate_reports.jsonl").read_text()
    assert result[gate_inconclusive.RESCUE_RESULT_KEY]["consent_id"] in report


def test_without_a_consent_the_capacity_inconclusive_quarantines_as_before(monkeypatch, rig):
    """Explicit slab_weights=on and no consent: the ceremony is the same, but
    nothing answers for the slab dispatch, so the combination returns to
    stock for the session, as the design says."""
    result, model = _ceremony(monkeypatch, rig)
    assert gate_inconclusive.RESCUE_RESULT_KEY not in result
    assert gate_inconclusive.dispatch_verdict(result) == "INCONCLUSIVE"
    gate_identity.quarantine_unproven_paths(model)
    assert model.mesh.worker_args == {"lora_low_rss": False, "slab_weights": False}


@pytest.mark.parametrize("case", ["lora_stack", "identity_divergence", "untyped_refusal"])
def test_a_lora_stack_a_divergence_or_an_untyped_refusal_still_quarantines(
        monkeypatch, rig, case):
    """Each keeps today's behaviour with the consent active: a LoRA stack is a
    second unproven path, slab legs that disagree measured something against
    slab, and a refusal without a class tag is not a typed capacity refusal."""
    _consent("env", monkeypatch)
    if case == "lora_stack":
        result, model = _ceremony(monkeypatch, rig, loras=(LORA,))
    elif case == "identity_divergence":
        result, model = _ceremony(monkeypatch, rig, a=torch.ones(1))
    else:
        result, model = _ceremony(monkeypatch, rig, cross=StockLoadCapacityError(
            f"stock residency cannot load {NAME}"))
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "CAPACITY"
    assert gate_inconclusive.RESCUE_RESULT_KEY not in result
    assert gate_inconclusive.dispatch_verdict(result) == "INCONCLUSIVE"
    gate_identity.quarantine_unproven_paths(model)
    assert model.mesh.worker_args == {"lora_low_rss": False, "slab_weights": False}


_SETTLED = {"model": NAME, "loras": 0, "origin": "auto_first_use", "run_id": "x",
            "max_abs_latent_diff": 0.0, "quarantine_levers": [], "cross_mode": "CAPACITY",
            "capacity_detail": TYPED}


@pytest.mark.parametrize(("change", "state"), [
    ({}, "clear"),
    ({"capacity_detail": "StockLoadCapacityError/untagged"}, "denied"),
    ({"capacity_detail": "FleetCapacityError/class-K guard=ring_pad"}, "denied"),
    ({"capacity_detail": None}, "denied"),
    ({"loras": 1}, "denied"),
    ({"max_abs_latent_diff": 0.25}, "denied"),
    ({"max_abs_latent_diff": None}, "denied"),
    ({"quarantine_levers": ["slab_weights"]}, "denied"),
    ({"cross_mode": "ERROR"}, "denied"),
])
def test_the_consent_reader_reads_through_only_a_typed_capacity_settlement(
        rig, change, state):
    """The one INCONCLUSIVE a rescue consent reads past, field by field."""
    combo, signature = _combo()
    context = {"capability": "test", "worker_args": dict(SLAB_ON)}
    gate_ledger.GateLedger(str(rig.output)).record(
        combo, signature, "known", "INCONCLUSIVE", {**_SETTLED, **change}, context)
    assert consent_rescue.quarantine_decision(combo, signature).state == state


def test_a_capacity_settlement_never_hides_an_earlier_fail(rig):
    combo, signature = _combo()
    context = {"capability": "test", "worker_args": dict(SLAB_ON)}
    ledger = gate_ledger.GateLedger(str(rig.output))
    ledger.record(combo, signature, "known", "FAIL",
                  {"quarantine_levers": ["slab_weights"], "cross_mode": "FAIL"}, context)
    ledger.record(combo, signature, "known", "INCONCLUSIVE", dict(_SETTLED), context)
    assert consent_rescue.quarantine_decision(combo, signature).state == "fail"


# The card after the verdict ---------------------------------------------------


@pytest.mark.parametrize("build", ["this_build", "an_earlier_build"])
def test_the_card_is_offered_listed_and_accepted_after_a_capacity_verdict(
        monkeypatch, rig, build):
    """2026-10-30 case 3: the next queue, with no consent active, prices the
    load at stock. The verdict took no lever, so the refusal offers the card,
    and the panel lists it on the evidence the loader offered it on, rows
    written by an earlier build included, and accepting it loads slab."""
    _consent("env", monkeypatch)
    _result, model = _ceremony(monkeypatch, rig)
    assert model.mesh.worker_args == SLAB_ON
    _withdraw("env", monkeypatch)
    if build == "an_earlier_build":
        monkeypatch.setattr(runtime_provenance, "cached_dgx_source_manifest_sha256",
                            lambda: "a-later-source-build")

    mesh = _LoaderMesh()
    refused = str(_load_refusal(mesh))
    assert "waivable=1" in refused and "Load with slab residency" in refused
    (card,) = consent_routes.consent_state()["pending"]
    assert card["kind"] == KIND
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200, payload
    load = loader_preflight.preflight_loader_footprint(mesh, NAME, {}, None)
    assert load is not None and mesh.worker_args["slab_weights"] is True


def test_a_session_quarantine_is_named_where_slab_cannot_be_offered(monkeypatch, rig):
    """An unproven verdict with no consent still turns both levers off for the
    session. The loader then cannot offer slab, and its refusal says which
    ceremony took the levers and how to clear them, not "slab_weights=off"."""
    _result, model = _ceremony(monkeypatch, rig)
    gate_identity.quarantine_unproven_paths(model)
    mesh = _LoaderMesh(model.mesh.worker_args)

    refused = str(_load_refusal(mesh))
    assert "waivable=0" in refused
    assert f"the identity gate for {NAME} (combination" in refused
    assert "could not complete on an unproved verdict earlier in this session" in refused
    assert "restart ComfyUI to clear it" in refused
    assert consent_routes.consent_state()["pending"] == []


# Actor offers on a layout where the driver holds no rank -------------------------


def _actor_offer(rig) -> consent_descriptor.SlabResidencyRescueOffer:
    """What the Worker's store_residency.resolve raises for this file."""
    with pytest.raises(consent_descriptor.SlabResidencyRescueOffer) as raised:
        store_residency.resolve(
            path=rig.checkpoint, unet_name=NAME, model_options={}, slab_weights="auto",
            slab_capable_path=True, lora_low_rss=True, fsdp_launch=False,
            blocked_reason="", authoritative_slab_retry=False,
            memoized_family=lambda _path: None, vouched_families=frozenset({"krea2"}),
            fit_probe=lambda _path, _options: _DOES_NOT_FIT, node_options={}, lora_stack=[])
    return raised.value


def _remote_load_refused(monkeypatch, mesh, offer) -> None:
    """The driver frame that holds every load RPC, with the Worker's refusal."""
    from monarch._src.actor.actor_mesh import ActorError

    def refused(*_args, **_kwargs):
        raise ActorError(offer, "Actor call dgxm_worker.load_model failed.")

    monkeypatch.setattr(mesh_setup, "dispatch_endpoint", refused)
    with pytest.raises(ActorError):
        mesh_rpc.call_all_bound(mesh.handle, "load_model", NAME, {}, [])


@pytest.mark.parametrize("build", ["this_build", "an_earlier_build"])
def test_an_actor_offer_where_the_driver_holds_no_rank_becomes_a_card_to_accept(
        monkeypatch, rig, build):
    """2026-10-30 cases 4 and 5: the loader node charges this host nothing,
    the Worker refuses the stock load and offers slab, and the driver turns
    the offer into a card the panel lists and accepts, as the loader's own
    card would be, after the pair's ceremony left its rows."""
    _consent("env", monkeypatch)
    _ceremony(monkeypatch, rig)
    _withdraw("env", monkeypatch)
    if build == "an_earlier_build":
        monkeypatch.setattr(runtime_provenance, "cached_dgx_source_manifest_sha256",
                            lambda: "a-later-source-build")
    mesh = _LoaderMesh(co_resident=False)
    assert loader_preflight.preflight_loader_footprint(mesh, NAME, {}, None) is None
    assert consent_projection.project_for_loader(mesh, NAME, {}) is None

    _remote_load_refused(monkeypatch, mesh, _actor_offer(rig))
    (card,) = consent_routes.consent_state()["pending"]
    assert card["kind"] == KIND
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200, payload
    load = consent_projection.project_for_loader(mesh, NAME, {})
    assert load is not None and mesh.worker_args["slab_weights"] is True


@pytest.mark.parametrize("source", ["toggle", "auto_env", "env"])
@pytest.mark.parametrize("build", ["this_build", "an_earlier_build"])
def test_the_toggle_and_the_variables_reach_a_load_on_another_host(
        monkeypatch, rig, source, build):
    """The headless forms answer the same question before the Worker has to
    ask it: the projection reaches the load and the Worker loads slab."""
    _consent("env", monkeypatch)
    _ceremony(monkeypatch, rig)
    _withdraw("env", monkeypatch)
    if build == "an_earlier_build":
        monkeypatch.setattr(runtime_provenance, "cached_dgx_source_manifest_sha256",
                            lambda: "a-later-source-build")
    expected = _consent(source, monkeypatch)
    mesh = _LoaderMesh(co_resident=False)
    load = consent_projection.project_for_loader(mesh, NAME, {})
    assert load is not None and mesh.worker_args["slab_weights"] is True
    # The WAIVER row its first use wrote (once per source and fleet
    # generation) cites the consent the load carries.
    rescue_rows = [(row["consent_source"], row["consent_id"]) for row in _rows(rig, "WAIVER")
                   if row.get("waiver_kind") == KIND]
    assert (expected, load.consent_id) in rescue_rows
