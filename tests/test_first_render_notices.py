"""First-render cost notices: log, event ring and browser."""
from __future__ import annotations

import logging
import sys
import threading
import types

import pytest

import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
from dgx_monarch import first_render, telemetry
from dgx_monarch.nodes import gate_process_state


@pytest.fixture(autouse=True)
def _reset_notices():
    first_render.reset()
    yield
    first_render.reset()


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def logged():
    """The first_render logger's own records (it does not propagate)."""
    from dgx_monarch.log import get_logger

    capture = _CaptureHandler()
    logger = get_logger("dgx_monarch.first_render")
    logger.addHandler(capture)
    yield capture.records
    logger.removeHandler(capture)


class _Ring:
    """A race-free window over the process-wide event ring."""

    def __init__(self):
        self._mark = telemetry.event_sequence()

    def notices(self):
        return [event for event in telemetry.events_tail(256)
                if event["seq"] > self._mark and event["kind"] == "notice"]

    def phases(self):
        return [event["phase"] for event in self.notices()]


@pytest.fixture
def ring():
    return _Ring()


@pytest.fixture
def sent(monkeypatch):
    """A fake PromptServer that records every send_sync call."""
    captured: list[tuple[str, dict]] = []
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(
            send_sync=lambda event, data: captured.append((event, data))))
    monkeypatch.setitem(sys.modules, "server", fake_server)
    return captured


def test_one_call_reaches_log_ring_and_browser(logged, ring, sent):
    message = "the whole honest sentence, which is far too long for a ticker row"
    assert first_render.notice("gate", message, "short form") is True

    assert [record.getMessage() for record in logged] == [message]

    rows = ring.notices()
    assert len(rows) == 1
    assert rows[0]["phase"] == "gate"
    assert rows[0]["note"] == "short form"
    # The sidebar and the TUI ticker print every field but t and kind inline,
    # so a paragraph in the ring would push the row off screen.
    assert "message" not in rows[0]
    assert message not in " ".join(str(v) for v in rows[0].values())

    assert len(sent) == 1
    event, payload = sent[0]
    assert event == "dgx-monarch.notice"
    assert payload == {"phase": "gate", "message": message,
                       "note": "short form", "toast": True,
                       "severity": "info", "sticky": False,
                       "summary": "DGX Monarch: first render"}


def test_a_caller_with_another_story_titles_its_own_toast(sent):
    """Every caller shares notice(); only first-render costs take its default title."""
    assert first_render.notice("graph_advisor", "swap these nodes", "advice",
                               summary="DGX Monarch: workflow advice") is True
    assert sent[0][1]["summary"] == "DGX Monarch: workflow advice"


def test_headless_driver_keeps_the_log_and_the_ring(monkeypatch, logged, ring):
    monkeypatch.delitem(sys.modules, "server", raising=False)
    assert first_render.notice("deferred", "no browser here", "headless") is True
    assert [record.getMessage() for record in logged] == ["no browser here"]
    assert ring.phases() == ["deferred"]


def test_a_broken_prompt_server_never_reaches_the_caller(monkeypatch, ring):
    def boom(event, data):
        raise RuntimeError("websocket is gone")

    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(send_sync=boom))
    monkeypatch.setitem(sys.modules, "server", fake_server)

    assert first_render.notice("gate", "still fine", "still fine") is True
    assert ring.phases() == ["gate"]


def test_deferrals_announce_once_per_process(ring, sent):
    first_render.nccl_deferred()
    first_render.nccl_deferred()
    first_render.load_deferred("load_model")
    first_render.load_deferred("load_model")

    rows = ring.notices()
    assert [row["note"] for row in rows] == [
        "NCCL bring-up on first render", "cold load_model on first render"]
    assert len(sent) == 2

    first_render.reset()
    first_render.nccl_deferred()
    assert len(sent) == 3


def test_gate_start_sentence_carries_the_shape_and_the_escape(sent):
    first_render.gate_started("unknown")
    message = sent[0][1]["message"]
    # A ceremony runs for minutes; a 15 s toast would leave the rest silent.
    assert sent[0][1]["sticky"] is True
    assert "2-4 short proof renders" in message
    assert "then your render" in message
    assert "skip the gate" in message
    assert "auto_gate=off" in message
    assert "not yet gated" in message

    first_render.reset()
    sent.clear()
    first_render.gate_started("inconclusive")
    assert "still inconclusive" in sent[0][1]["message"]

    for state in ("stale", "error", "pass", "", "anything else"):
        first_render.reset()
        sent.clear()
        first_render.gate_started(state)
        assert ("stale or not vouched (a model or LoRA file, the dgx-monarch release or source, or the "
                "residency, worker, attention, cluster or topology settings changed; the ComfyUI commit "
                "changed or cannot be read; or the gate ledger holds a damaged line or a non-PASS row "
                "for this combination)") in sent[0][1]["message"]


def test_proof_renders_are_counted_and_never_toast(ring, sent):
    assert first_render.note_proof_render() is None

    first_render.gate_started("unknown")
    assert first_render.note_proof_render() == 1
    assert first_render.note_proof_render() == 2

    proofs = [payload for _event, payload in sent if payload["phase"] == "gate_proof"]
    assert [payload["note"] for payload in proofs] == [
        "proof render 1", "proof render 2"]
    assert all(payload["toast"] is False for payload in proofs)
    assert all("your own render has not started yet" in payload["message"]
               for payload in proofs)

    first_render.gate_finished({"verdict": "PASS"})
    before = len(ring.notices())
    assert first_render.note_proof_render() is None
    assert len(ring.notices()) == before


def test_the_cross_residency_leg_names_itself(ring, sent):
    """A proof counter cannot say which leg is running; this notice names the
    leg that swaps residency, and the `dgxm top` run block labels it from this
    phase.
    """
    first_render.cross_residency_check()

    assert ring.phases() == ["gate_cross"]
    assert ring.notices()[0]["note"] == "cross-residency check"
    assert sent[0][1]["toast"] is False
    assert "bit for bit" in sent[0][1]["message"]
    assert "your own render has not started yet" in sent[0][1]["message"]
    # No once_key: every ceremony that runs this leg announces it.
    first_render.cross_residency_check()
    assert ring.phases() == ["gate_cross", "gate_cross"]


def test_the_cross_leg_announces_only_when_it_actually_runs(ring):
    """The notice sits after the guards: a leg that stands down stays silent."""
    from dgx_monarch.nodes import gate_cross_mode

    stood_down = gate_cross_mode.run_cross_residency_reference(
        runtime={}, ceremony_model=None, handle=None, original_worker_args={},
        slab_proof=types.SimpleNamespace(error=None, expected=False),
        frozen_request={}, frozen_latent={}, artifact_binding={}, cfg_value=1.0,
        steps_hint=1, stock_latent={}, transaction_render=None, bind_request=None)
    assert stood_down == (None, None)
    assert ring.phases() == []

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no mesh in a unit test")

    verdict, latent = gate_cross_mode.run_cross_residency_reference(
        runtime={"_temporary_worker_policy": _boom,
                 "is_artifact_binding_error": lambda _exc: False,
                 "is_stock_load_capacity_error": lambda _exc: False,
                 "is_memory_exhaustion": lambda _exc: False,
                 "log": types.SimpleNamespace(warning=lambda *_a: None)},
        ceremony_model=None, handle=None, original_worker_args={},
        slab_proof=types.SimpleNamespace(error=None, expected=True),
        frozen_request={}, frozen_latent={}, artifact_binding={}, cfg_value=1.0,
        steps_hint=1, stock_latent={}, transaction_render=None, bind_request=None)
    assert verdict["verdict"] == "ERROR"
    assert latent is None
    # Announced before the render it names, not after the leg's outcome.
    assert ring.phases() == ["gate_cross"]


def test_summary_reports_verdict_wall_and_proof_count(sent):
    first_render.gate_started("unknown")
    first_render.note_proof_render()
    first_render.note_proof_render()
    sent.clear()
    first_render.gate_finished({"verdict": "PASS", "wall_s": 61.4})

    message = sent[0][1]["message"]
    assert "identity gate PASS in 61.4s over 2 proof renders" in message
    assert "skips the gate and the cold load" in message
    assert sent[0][1]["note"] == "gate PASS in 61.4s, render starting"
    assert sent[0][1]["severity"] == "info"
    assert sent[0][1]["sticky"] is False

    for wall in (None, True, "abc"):
        first_render.gate_started("unknown")
        sent.clear()
        first_render.gate_finished({"verdict": "PASS", "wall_s": wall})
        assert "identity gate PASS. Your render" in sent[0][1]["message"]


def test_a_non_pass_summary_never_claims_the_render_is_starting(sent):
    """Only a PASS authorizes the optimized residency paths.

    Any other verdict leaves them off for this combination or, under FSDP,
    refuses the render. The summary is the last thing on the canvas before
    that refusal, so it must not promise a render.
    """
    for verdict in ("FAIL", "ERROR", "INCONCLUSIVE", "?"):
        first_render.gate_started("unknown")
        first_render.note_proof_render()
        sent.clear()
        first_render.gate_finished({"verdict": verdict, "wall_s": 61.4})

        payload = sent[0][1]
        assert (f"identity gate {verdict} in 61.4s over 1 proof renders"
                in payload["message"])
        assert "starts now" not in payload["message"]
        assert "skips the gate" not in payload["message"]
        assert "optimized residency paths stay off" in payload["message"]
        assert "TROUBLESHOOTING.md #61" in payload["message"]
        assert payload["note"] == f"gate {verdict} in 61.4s, optimized residency off"
        assert payload["severity"] == "warn"


def test_failed_ceremony_replaces_the_sticky_toast_and_closes_the_window(
        ring, sent):
    """A ceremony that raised has no verdict, but its sticky gate-start toast
    is still on the canvas while the caller renders on stock residency or
    refuses. Without this notice the UI would still show an active ceremony
    after the gate has ended."""
    first_render.gate_started("unknown")
    first_render.note_proof_render()
    sent.clear()

    first_render.gate_finished(None)
    assert len(sent) == 1
    payload = sent[0][1]
    assert payload["phase"] == "gate_done"
    assert "did not finish" in payload["message"]
    assert "continues on stock residency or is refused" in payload["message"]
    assert payload["note"] == "gate did not finish, optimized residency off"
    assert payload["severity"] == "warn"
    assert first_render.note_proof_render() is None


def test_progress_bracket_counts_a_proof_leg(monkeypatch, sent):
    from monarch.actor import Channel

    from dgx_monarch.progress import ProgressReceiver

    class Future:
        def get(self, timeout=None):
            return {"done": True}

    class Receiver:
        def recv(self):
            return Future()

    class Port:
        def send(self, value):
            return None

    monkeypatch.setattr(Channel, "open", lambda: (Port(), Receiver()))

    with ProgressReceiver(2):
        pass
    assert [payload["phase"] for _event, payload in sent] == []

    first_render.gate_started("unknown")
    sent.clear()
    with ProgressReceiver(2):
        pass
    assert [payload["note"] for _event, payload in sent] == ["proof render 1"]


def test_a_ceremony_on_another_thread_is_not_this_threads_ceremony():
    first_render.gate_started("unknown")
    seen = []

    def other_thread():
        seen.append(first_render.note_proof_render())

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join(timeout=5)
    assert seen == [None]
    assert first_render.note_proof_render() == 1


@pytest.fixture
def gate_session():
    common._AUTO_GATE_SESSION.clear()
    gate_process_state._PROCESS_GATE_DENIALS = {}
    common._AUTO_GATE_ACTIVE.on = False
    yield
    common._AUTO_GATE_SESSION.clear()
    gate_process_state._PROCESS_GATE_DENIALS = {}


def test_real_auto_gate_ceremony_announces_start_and_summary(
    monkeypatch, gate_session, ring
):
    token = ("combo", "artifacts", "commit", "context")
    model = types.SimpleNamespace()
    calls = []
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("stale", token))
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony",
        lambda *_args, **_kwargs: calls.append(1) or {
            "verdict": "PASS", "_gate_token": token, "wall_s": 12.5},
    )

    request = {"kind": "ksampler", "steps": 4}
    assert common._maybe_auto_gate(model, request, {}, 1.0, 4) == "PASS"
    assert ring.phases() == ["gate", "gate_done"]

    # A cached PASS runs no ceremony and says nothing.
    assert common._maybe_auto_gate(model, request, {}, 1.0, 4) == "PASS"
    assert calls == [1]
    assert ring.phases() == ["gate", "gate_done"]


def test_a_real_non_pass_ceremony_announces_the_denial_not_a_render(
    monkeypatch, gate_session, ring, sent
):
    token = ("combo", "artifacts", "commit", "context")
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", token))
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony",
        lambda *_args, **_kwargs: {"verdict": "FAIL", "wall_s": 61.4},
    )

    assert common._maybe_auto_gate(
        types.SimpleNamespace(), {"kind": "ksampler", "steps": 4}, {}, 1.0, 4
    ) == "FAIL"
    assert ring.phases() == ["gate", "gate_done"]
    summary = next(payload for _event, payload in sent
                   if payload["phase"] == "gate_done")
    assert "identity gate FAIL in 61.4s" in summary["message"]
    assert "starts now" not in summary["message"]
    assert summary["severity"] == "warn"


def test_a_raising_ceremony_still_closes_the_window(
    monkeypatch, gate_session, ring
):
    token = ("combo", "artifacts", "commit", "context")
    model = types.SimpleNamespace()

    def explode(*_args, **_kwargs):
        raise RuntimeError("ceremony fell over")

    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", token))
    monkeypatch.setattr(gate_mod, "run_identity_ceremony", explode)

    request = {"kind": "ksampler", "steps": 4}
    assert common._maybe_auto_gate(model, request, {}, 1.0, 4) == "ERROR"
    # gate_finished(None) replaces the sticky gate-start toast before the
    # caller runs the render on stock residency.
    assert ring.phases() == ["gate", "gate_done"]
    assert ring.notices()[-1]["note"] == (
        "gate did not finish, optimized residency off")
    assert first_render.note_proof_render() is None


def test_a_ceremony_that_never_starts_leaves_no_window_behind(
    monkeypatch, gate_session, ring
):
    """The outer finally releases the ceremony claim and the inner one closes
    the ceremony window, so the window opens inside the inner try. A raise
    before that try (a malformed step count, or a cancellation in the gap)
    must release the claim and open no window. A leaked window lasts the life
    of the driver thread: every later render on it carries a ceremony flag and
    counts itself a proof render in the notices docs/TROUBLESHOOTING.md #61
    describes.
    """
    token = ("combo", "artifacts", "commit", "context")
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", token))

    request = {"kind": "ksampler", "steps": "not-a-number"}
    assert common._maybe_auto_gate(
        types.SimpleNamespace(), request, {}, 1.0, 2) == "ERROR"

    assert token not in common._AUTO_GATE_RUNNING  # the outer finally released it
    assert first_render.note_proof_render() is None
    assert not telemetry.render_progress._state.get("_ceremony_windows")

    marker = object()
    telemetry.render_progress.start(4, {"model": "later"}, token=marker)
    assert telemetry.render_progress.snapshot()["ceremony"] is False
    telemetry.render_progress.finish(marker)
    assert telemetry.render_progress.snapshot()["last"]["ceremony"] is False
    # Nothing opened, so nothing announced a gate either.
    assert ring.phases() == []


# Call sites: every notice must be reachable from a node.

def test_the_init_node_announces_the_deferred_nccl_bring_up(monkeypatch):
    """`auto` topology defers NCCL onto the first render; say so in the UI."""
    from dgx_monarch.nodes import init as init_module

    said = []
    monkeypatch.setattr(
        init_module, "get_mesh", lambda **_kwargs: types.SimpleNamespace(world=2))
    monkeypatch.setattr(first_render, "nccl_deferred", lambda: said.append(1))

    (spec,) = init_module.DGXMonarchInit()._init_with_bootstrap_policy(
        topology="auto", mode="cluster")
    assert spec.topology_preset == "auto"
    assert said == [1], "the auto branch no longer announces the deferred bring-up"


def test_the_unet_loaders_announce_the_deferred_cold_load(monkeypatch):
    """Both model slots defer their load under `auto`, and both say which."""
    from dgx_monarch.nodes import loaders

    said = []
    monkeypatch.setattr(
        loaders.loader_preflight, "preflight_loader_footprint",
        lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        loaders, "ensure_live", lambda handle, mesh_preflight=None: handle)
    monkeypatch.setattr(
        first_render, "load_deferred", lambda endpoint: said.append(endpoint))

    mesh = types.SimpleNamespace(
        handle=types.SimpleNamespace(world=1), topology_preset="auto",
        attention="TORCH_FLASH", sync_ulysses=True, worker_args={})
    for node in (loaders.DGXMonarchUNETLoader(), loaders.DGXMonarchUncondUNETLoader()):
        node.load(mesh, "model.safetensors")
    assert said == ["load_model", "load_uncond_model"]


def test_the_ticker_styles_a_notice_and_takes_no_marker_glyph():
    """`dgxm top` prints the row; the one-char marker lane must stay free."""
    from dgx_monarch.tui.braille import EVENT_GLYPHS, EVENT_STYLES

    assert EVENT_STYLES["notice"] == "cyan"
    # A glyph here would overwrite the S/G/L/Q marker at that tick column.
    assert "notice" not in EVENT_GLYPHS
