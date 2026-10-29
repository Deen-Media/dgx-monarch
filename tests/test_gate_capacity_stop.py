"""An identity ceremony whose request cannot fit stops before its proof render.

CPU only. The ceremony runs on the shared cross-mode rig
(tests/gate_orchestration_helpers.py). The render site's checks for the
ceremony's path are a stub that refuses the way the real ones do: class C,
budget first. Source: the 2026-10-29 15 second MiniMax H3 cell, whose
first-use ceremony wrote a RETESTING row, paid a 4.7 minute proof render and
only then met the activation preflight, which refused the slab path 3.8 GiB
short; the ledger kept the RETESTING row as the combination's last row.
"""
from __future__ import annotations

import json
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import fsdp_reload_price, gate_audit_vocab, gate_ledger, mesh_safety
from dgx_monarch.log import get_logger
from dgx_monarch.nodes import (
    common,
    gate_capacity_stop,
    gate_inconclusive,
    gate_process_state,
    render_preflight,
)
from dgx_monarch.nodes import gate as gate_mod
from dgx_monarch.nodes.gate_capacity_stop import (
    CeremonyCapacityStop,
    proved_path,
    record_capacity_stop,
)
from dgx_monarch.refusal import GUARDS, RefusalClass, parse_leading_refusal_tag, refusal
from dgx_monarch.render_memory_price import RenderMemoryPriceError
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
    _cross_mode_rig,
    _run,
)

GUARD = "activation_footprint_preflight"
SLAB = {"lora_low_rss": True, "slab_weights": True}
BUDGET = ("minimax_h3 activation preflight: 60.1 GiB needed per rank, 56.3 GiB "
          "usable (61.3 GiB MemAvailable minus a 5.0 GiB reserve, the driver floor).")
STOP_SENTENCE = ("The identity gate ran no proof render: this request cannot fit "
                 "the slab path, so it recorded one INCONCLUSIVE capacity row and "
                 "took no residency lever.")


def _refused(error: type[Exception] = mesh_safety.StockLoadCapacityError) -> Exception:
    return error(refusal(
        RefusalClass.CAPACITY,
        f"{BUDGET} Refuses model.safetensors. Options: free host memory.",
        guard=GUARD, waivable=False))


@pytest.fixture
def check(monkeypatch):
    """The render site's checks for the ceremony's path, stubbed: each call is
    recorded, in ``events`` too once a test sets it, and ``refusal`` raises."""
    state = SimpleNamespace(calls=[], refusal=None, events=None)

    def ceremony_activation_preflight(model, request, latent):
        state.calls.append((model, request, latent))
        if state.events is not None:
            state.events.append(("check", None))
        if state.refusal is not None:
            raise state.refusal

    monkeypatch.setattr(
        gate_capacity_stop, "ceremony_activation_preflight", ceremony_activation_preflight)
    return state


@pytest.fixture
def journal():
    """Records from the loggers the stop path could write to. They do not
    propagate, so each gets the handler directly."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    loggers = [get_logger(name) for name in (
        "dgx_monarch.nodes.common", "dgx_monarch.nodes.gate",
        "dgx_monarch.nodes.gate_quarantine_scope",
        "dgx_monarch.nodes.gate_capacity_stop")]
    for logger in loggers:
        logger.addHandler(handler)
    yield records
    for logger in loggers:
        logger.removeHandler(handler)


# 1. The ceremony.

def test_a_request_that_cannot_fit_runs_no_proof_render(monkeypatch, tmp_path, check):
    """No unload, render, cycle, provenance read or policy push; no RETESTING
    row and no process verdict; the graph keeps the levers it asked for."""
    check.refusal = _refused()
    handle, ledger, model = _cross_mode_rig(monkeypatch, tmp_path, [], dict(SLAB))
    published: list = []
    monkeypatch.setattr(
        gate_mod, "_publish_process_gate_verdicts",
        lambda tokens, verdict, ceremony=None: published.append(verdict))

    with pytest.raises(CeremonyCapacityStop) as raised:
        _run(model)

    assert len(check.calls) == 1
    assert handle.calls == []
    assert ledger.retests == []
    assert published == []
    assert gate_process_state._PROCESS_GATE_DENIALS == {}
    assert model.mesh.worker_args == SLAB
    assert raised.value.__cause__ is check.refusal
    # The guard's class and remedy stay on the wire, budget first.
    message = str(raised.value)
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == GUARD and tag.waivable is False
    assert BUDGET in message[:200]
    assert "Options: free host memory." in message
    assert message.endswith(STOP_SENTENCE), message


def test_the_stop_writes_one_terminal_inconclusive_capacity_row(
        monkeypatch, tmp_path, check):
    """The FSDP stop's vocabulary, for the context the render's authorization
    reads, with what refused classified and its text kept out of the row."""
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1)] * 3, dict(SLAB))
    _run(model)
    retest_key, normal_context = ledger.retests[0][0], ledger.retests[0][3][0]

    check.refusal = _refused()
    _handle, ledger, model = _cross_mode_rig(monkeypatch, tmp_path, [], dict(SLAB))
    with pytest.raises(CeremonyCapacityStop):
        _run(model)

    assert len(ledger.records) == 1
    key, _artifacts, _commit, verdict, detail, context = ledger.records[0]
    assert key == retest_key
    assert verdict == fsdp_reload_price.CAPACITY_STOP_VERDICT == "INCONCLUSIVE"
    assert context == normal_context
    assert detail["cross_mode"] == fsdp_reload_price.CAPACITY_CROSS_MODE
    assert detail["quarantine_levers"] == []
    assert detail["max_abs_latent_diff"] is None
    assert detail["measured"] is None
    assert detail["capacity_detail"] == (
        "StockLoadCapacityError/class-C guard=activation_footprint_preflight")
    assert detail["inconclusive_reasons"] == [
        "the request's capacity check refused the slab path before the proof "
        "render, so the ceremony ran none"]
    assert (detail["model"], detail["loras"], detail["origin"], detail["run_id"]) == (
        "model.safetensors", 1, "explicit", "x")
    assert "60.1" not in json.dumps(detail)


def test_a_request_that_fits_runs_the_ceremony_as_before(monkeypatch, tmp_path, check):
    """The check comes before the first durable write and prices the
    ceremony's own policy on its pinned handle; then every leg runs."""
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1)] * 3, dict(SLAB))
    check.events = ledger.events

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert ledger.events[0] == ("check", None)
    assert ledger.events[1][0] == "retest"
    assert sum(method == "render" for method, _args in handle.calls) == 3
    priced, request, latent = check.calls[0]
    assert priced.mesh.handle is handle
    assert priced.mesh.worker_args["slab_weights"] is True
    assert request["noise_seed"] == 1 and torch.equal(latent["samples"], torch.zeros(1))
    assert [record[3] for record in ledger.records] == ["PASS", "PASS"]


def test_a_check_that_breaks_fails_open(monkeypatch, tmp_path, check):
    """Only a capacity refusal stops the ceremony. Anything else leaves it to
    the proof render's own preflight, as at every other preflight."""
    check.refusal = RuntimeError("the estimator broke")
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1)] * 3, dict(SLAB))
    assert _run(model)["verdict"] == "PASS"
    assert len(ledger.retests) == 1


def test_a_render_memory_refusal_stops_the_ceremony_too(monkeypatch, tmp_path, check):
    """It keeps its own family: the stop is no stock-load error, and the row
    names the class that refused."""
    check.refusal = _refused(RenderMemoryPriceError)
    _handle, ledger, model = _cross_mode_rig(monkeypatch, tmp_path, [], dict(SLAB))
    with pytest.raises(CeremonyCapacityStop) as raised:
        _run(model)
    assert not isinstance(raised.value, mesh_safety.StockLoadCapacityError)
    assert ledger.records[0][4]["capacity_detail"] == (
        "RenderMemoryPriceError/class-C guard=activation_footprint_preflight")


def test_the_fsdp_proof_keeps_its_own_price(monkeypatch, tmp_path, check):
    """A no-LoRA FSDP proof has its clean-reload price; this check is the
    residency ceremony's."""
    from test_gate_fsdp_proof import _ceremony

    check.refusal = _refused()
    _handle, _ledger, model, _published = _ceremony(monkeypatch, tmp_path)
    assert _run(model)["verdict"] == "PASS"
    assert check.calls == []


@pytest.mark.parametrize(("worker_args", "path"), [
    ({"slab_weights": True}, "the slab path"),
    ({}, "the slab path"),
    ({"slab_weights": False, "lora_low_rss": True}, "the lazy-swap path"),
    ({"comfy_managed": True}, "comfy-managed residency"),
])
def test_the_stop_names_the_path_it_could_not_fit(worker_args, path):
    assert proved_path(worker_args) == path


# 2. The vocabulary.

def test_the_vocabulary_accepts_the_stop_row(tmp_path):
    """The row reads back through a real ledger as an INCONCLUSIVE trust row:
    it denies, grants no skip, is no audit row, and its guard is a frozen
    measured probe. It needs no protocol change."""
    led = gate_ledger.GateLedger(str(tmp_path))
    key = gate_ledger.combo_key("model.safetensors", {}, [])
    context = {"worker_args": dict(SLAB), "world": 2}
    stop = record_capacity_stop(
        led, key, "artifact", "commit", context, "model.safetensors",
        origin="auto_first_use", run_id="", loras=0, worker_args=SLAB,
        refused=_refused())
    assert isinstance(stop, CeremonyCapacityStop)

    lookup = led.lookup_with_integrity(key, "artifact", "commit", context)
    assert lookup.state == "inconclusive"
    entry = lookup.entry
    assert entry["verdict"] == "INCONCLUSIVE" and entry["cross_mode"] == "CAPACITY"
    assert not gate_audit_vocab.is_audit_row(entry)
    assert gate_audit_vocab.trust_rows([entry]) == [entry]
    assert not gate_inconclusive.row_grants_skip(lookup)
    assert gate_inconclusive.granted_state(lookup, False) == "inconclusive"
    guard = entry["capacity_detail"].rsplit("guard=", 1)[1]
    assert guard in gate_audit_vocab.MEASURED_PROBES and guard in GUARDS
    assert gate_inconclusive.CAPACITY_STOP_DISPATCH_VERDICT in (
        gate_inconclusive.NO_QUARANTINE_VERDICTS)


def test_a_failed_append_still_stops(monkeypatch):
    """A ledger that cannot take the row logs it; the stop stands, and the
    next request prices again."""
    class Ledger:
        def record(self, *_args):
            raise OSError("read-only output directory")

    stop = record_capacity_stop(
        Ledger(), "combo", "artifact", "commit", {}, "model.safetensors",
        origin="explicit", run_id="x", loras=0, worker_args=SLAB, refused=_refused())
    assert str(stop).endswith(STOP_SENTENCE)


# 3. The automatic path.

def test_the_automatic_path_renders_stock_and_the_next_request_checks_again(
        monkeypatch, tmp_path, check, journal):
    check.refusal = _refused()
    handle, ledger, model = _cross_mode_rig(monkeypatch, tmp_path, [], dict(SLAB))
    token = ("capacity-stop-auto",)
    monkeypatch.setattr(common, "_auto_gate_context", lambda *_args: ("unknown", token))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_ACTIVE", SimpleNamespace(on=False))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_RUNNING", set())
    request = {"kind": "ksampler", "noise_seed": 1, "steps": 20, "cfg": 1.0}
    latent = {"samples": torch.zeros(1)}

    first = common._maybe_auto_gate(model, request, latent, 1.0, 20)
    second = common._maybe_auto_gate(model, request, latent, 1.0, 20)

    assert first == second == gate_inconclusive.CAPACITY_STOP_DISPATCH_VERDICT
    # No sticky verdict: the second request met the same check.
    assert len(check.calls) == 2
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE", "INCONCLUSIVE"]
    assert ledger.retests == [] and handle.calls == []
    assert gate_process_state._AUTO_GATE_SESSION == {}
    assert gate_process_state._PROCESS_GATE_DENIALS == {}
    assert gate_process_state._AUTO_GATE_RUNNING == set()
    assert model.mesh.worker_args == SLAB
    # The light request priced is the request's own conditioning and latent.
    _priced, light, _latent = check.calls[0]
    assert light["steps"] == 2 and light["noise_seed"] == 1
    # One line per stop, none of the abort's.
    lines = [record.getMessage() for record in journal]
    assert len(lines) == 2, lines
    assert all(record.levelno == logging.WARNING for record in journal)
    assert lines[0].startswith(
        "identity gate skipped for model.safetensors (combination ")
    assert ("and this render runs on stock residency; the next request of this "
            "combination meets the same check: [dgxm:C guard=activation_footprint_preflight "
            "waivable=0] " + BUDGET) in lines[0]
    assert lines[0].endswith(STOP_SENTENCE)


def test_an_fsdp_lora_stop_refuses_typed(monkeypatch, check):
    """A resolved FSDP topology has no stock residency to fall back to, so the
    stop's class C refusal answers the render."""
    token = ("combo", "artifact", "commit",
             json.dumps({"resolved_topology": {"fsdp": True}}))
    monkeypatch.setattr(common, "_auto_gate_context", lambda *_args: ("unknown", token))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_ACTIVE", SimpleNamespace(on=False))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_RUNNING", set())
    stop = CeremonyCapacityStop(f"{_refused()} {STOP_SENTENCE}")

    def ceremony(*_args, **_kwargs):
        raise stop

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    with pytest.raises(CeremonyCapacityStop) as raised:
        common._maybe_auto_gate(SimpleNamespace(), {"kind": "ksampler", "steps": 4},
                                {}, 1.0, 4)
    assert raised.value is stop
    assert gate_process_state._AUTO_GATE_RUNNING == set()


def test_the_render_after_a_stop_keeps_the_graph_levers(monkeypatch):
    """The stop disproved nothing, so run_render takes no lever off the graph;
    authorization finds no PASS and dispatches the render stock."""
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args=dict(SLAB)))
    submitted: list = []

    class Pending:
        def result(self):
            return {"samples": torch.zeros(1)}

    monkeypatch.setattr(
        common, "_maybe_auto_gate",
        lambda *_args: gate_inconclusive.CAPACITY_STOP_DISPATCH_VERDICT)

    def submit(*_args, **_kwargs):
        submitted.append(dict(model.mesh.worker_args))
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    common.run_render(model, {"kind": "ksampler"}, {}, 1.0, 2)
    assert submitted == [SLAB]


# 4. The posture: the render site's checks, priced for the path the ceremony
# is about to prove (gate_capacity_stop.ceremony_activation_preflight).

H3_UNET = "minimax_h3_ref2va_int8_pruned.safetensors"
H3_DIT_BYTES = 20_970_379_616   # the pruned int8 artifact, 19.53 GiB


class _Shape:
    """A tensor stand-in: ``shape`` is all h3_rows reads from it."""

    def __init__(self, shape):
        self.shape = shape


class _AudioVideo:
    """comfy's NestedTensor duck: video first, audio second."""

    def __init__(self, video, audio):
        self.tensors = [_Shape(video), _Shape(audio)]
        self.shape = video


def _h3_inputs():
    from dgx_monarch import h3_rows

    frames = h3_rows.align_frame_count(124)
    video = (1, 24, h3_rows.latent_t_for_frames(frames), 768 // 16, 1344 // 16)
    audio = (1, 32, 2, h3_rows.audio_t_for_frames(frames))
    return ({"positive": [[_Shape((1, 20, 4096)), {}]]},
            {"samples": _AudioVideo(video, audio)})


def _h3_rig(monkeypatch, tmp_path, *, preset, avail):
    """The H3 checks on a stubbed host: a uly2 pair whose rank 0 shares the
    driver host, a 19.53 GiB DiT file, and no ComfyUI or set-up mesh, so a
    refusal stands on its first reading."""
    from dgx_monarch import (
        driver_footprint,
        driver_unload,
        h3_activation,
        render_memory_price,
    )

    checkpoint = tmp_path / H3_UNET
    checkpoint.write_bytes(b"x")
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda _kind, _name: str(checkpoint)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("minimax_h3", "int8"))
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda _path: H3_DIT_BYTES)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(render_memory_price, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: avail)
    for name in (mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV,
                 driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV,
                 driver_unload.DRIVER_UNLOAD_DISABLE_ENV):
        monkeypatch.delenv(name, raising=False)
    for memo in ("_LAST_RENDERED_UNET", "_LAST_SUBMITTED_UNET", "_LAST_DRIVER_CHARGED_UNET"):
        monkeypatch.setattr(render_preflight, memo, None)
    handle = SimpleNamespace(
        owns_hosts=False, world=2, sample_leases={}, abandoned_sample_leases={},
        effective_worker_args=lambda requested=None: dict(requested or {}))
    mesh = SimpleNamespace(topology_preset=preset, worker_args=dict(SLAB),
                           world=2, handle=handle)
    estimates: list = []
    real = h3_activation.estimate_h3_activation

    def capture(**kwargs):
        estimates.append(real(**kwargs))
        return estimates[-1]

    monkeypatch.setattr(h3_activation, "estimate_h3_activation", capture)
    return SimpleNamespace(unet_name=H3_UNET, mesh=mesh, options={}), estimates


def test_the_ceremony_charges_a_load_the_render_site_only_cleared(monkeypatch, tmp_path):
    """Under ``auto`` the render site's own check stamps the cleared pending
    load a moment before the ceremony runs. The render site may credit that
    memo; the ceremony may not, because its load has not happened."""
    model, estimates = _h3_rig(monkeypatch, tmp_path, preset="auto", avail=2 ** 40)
    monkeypatch.setattr(render_preflight, "_LAST_DRIVER_CHARGED_UNET", H3_UNET)
    request, latent = _h3_inputs()
    render_preflight.activation_footprint_preflight_for_request(model, request, latent)
    gate_capacity_stop.ceremony_activation_preflight(model, request, latent)
    assert [estimate.weight_bytes for estimate in estimates] == [0, H3_DIT_BYTES]


@pytest.mark.parametrize(("preset", "memo", "charged"), [
    ("auto", None, H3_DIT_BYTES),
    ("auto", "_LAST_RENDERED_UNET", 0),
    ("auto", "_LAST_SUBMITTED_UNET", 0),
    ("uly2", None, 0),
])
def test_the_ceremony_credits_the_dit_only_when_it_is_loaded(
        monkeypatch, tmp_path, preset, memo, charged):
    """A finished render or a pipelined submission left the DiT loaded, and an
    explicit preset loaded it at the loader node; anything else is a pending
    load the ceremony charges."""
    model, estimates = _h3_rig(monkeypatch, tmp_path, preset=preset, avail=2 ** 40)
    if memo is not None:
        monkeypatch.setattr(render_preflight, memo, H3_UNET)
    gate_capacity_stop.ceremony_activation_preflight(model, *_h3_inputs())
    assert [estimate.weight_bytes for estimate in estimates] == [charged]


def test_a_request_the_slab_path_cannot_hold_stops_the_ceremony(monkeypatch, tmp_path):
    """MemAvailable between the credited and the charged need: the render
    site's check passes on its cleared memo, and the ceremony's check refuses,
    class C, before any worker call, so the ceremony stops. Sized from the live
    estimator, so the case holds whatever the H3 fit says; under a low enough
    fit the driver footprint check, which charges the same pending DiT,
    answers first."""
    model, estimates = _h3_rig(monkeypatch, tmp_path, preset="auto", avail=2 ** 40)
    request, latent = _h3_inputs()
    gate_capacity_stop.ceremony_activation_preflight(model, request, latent)
    charged = estimates[0]
    need = charged.projected + charged.reserve - H3_DIT_BYTES + 2 ** 30
    model, estimates = _h3_rig(monkeypatch, tmp_path, preset="auto", avail=need)
    monkeypatch.setattr(render_preflight, "_LAST_DRIVER_CHARGED_UNET", H3_UNET)
    render_preflight.activation_footprint_preflight_for_request(model, request, latent)

    class Ledger:
        def __init__(self):
            self.records: list = []

        def record(self, *args):
            self.records.append(args)

    ledger = Ledger()
    with pytest.raises(CeremonyCapacityStop) as raised:
        gate_capacity_stop.stop_unless_request_fits(
            model, request, latent,
            (ledger, "combo", "artifact", "commit", {}, H3_UNET),
            origin="auto_first_use", run_id="", loras=0, worker_args=SLAB)
    assert isinstance(raised.value.__cause__, mesh_safety.StockLoadCapacityError)
    tag = parse_leading_refusal_tag(str(raised.value))
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard in {"activation_footprint_preflight", "driver_footprint_preflight"}
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE"]
