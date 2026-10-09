"""The teardown-absorption instrumentation (docs/TROUBLESHOOTING.md #60):
every site fires, fires once, fires by name, and changes nothing."""
from __future__ import annotations

import ast
import logging
import re
import time
from pathlib import Path

import pytest

from dgx_monarch import absorb_fire, mesh_helpers, mesh_safety
from dgx_monarch.log import get_logger

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"

_LINE = re.compile(r"^dgxm-absorb-fire site=(\S+) detail=(.*)$")
# The torchmonarch 0.5.0 fault shape absorb_fire was built against: two markers in one text.
_TRANSPORT_FAULT = ("undeliverable message for dgxm_worker.status(): "
                    "broken link: channel closed with reason server closed")
_CLEAN_STATE = ({}, float("-inf"), 0)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def pairs(self) -> list[tuple[str, str]]:
        """Every dgxm-absorb-fire record so far, as (site, detail)."""
        found = []
        for record in self.records:
            match = _LINE.match(record.getMessage())
            if match:
                found.append((match.group(1), ast.literal_eval(match.group(2))))
        return found

    def sites(self) -> list[str]:
        return [site for site, _detail in self.pairs()]


@pytest.fixture
def fired():
    """This test's records on the mesh logger; ``pairs()`` reads the tagged ones.

    The mesh logger does not propagate, so caplog cannot see it; the teardown
    tests install their own handler for the same reason.
    """
    capture = _Capture()
    logger = get_logger("dgx_monarch.mesh")
    logger.addHandler(capture)
    yield capture
    logger.removeHandler(capture)


def _troubleshooting_60() -> str:
    text = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    start = text.index("## 60. ")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def test_the_tag_and_the_site_names_are_the_documented_ones():
    assert absorb_fire.TAG == "dgxm-absorb-fire"
    section = _troubleshooting_60()
    documented = set(re.findall(r"^\|\s*`([a-z-]+)`\s*\|", section, re.MULTILINE))
    assert documented == set(absorb_fire.SITES), (
        "docs/TROUBLESHOOTING.md #60 must name exactly the emitted sites: "
        f"documented={sorted(documented)}, SITES={sorted(absorb_fire.SITES)}"
    )


def _call_site_names(relative: str) -> set[str]:
    """Every first string-literal argument of an ``_absorb.<attr>(...)`` call."""
    tree = ast.parse((SRC / relative).read_text())
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_absorb"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value)
    return names


def test_every_call_site_uses_a_declared_site_name():
    called = _call_site_names("mesh_safety.py") | _call_site_names("mesh_helpers.py")
    assert called == set(absorb_fire.SITES), (
        "a site name must be declared in absorb_fire.SITES and have a caller: "
        f"call sites={sorted(called)}, SITES={sorted(absorb_fire.SITES)}"
    )


def test_one_hit_is_one_info_record_with_tag_site_and_detail(fired):
    absorb_fire.fire("x", "y")
    assert len(fired.records) == 1
    assert fired.records[0].levelno == logging.INFO
    assert fired.records[0].getMessage() == "dgxm-absorb-fire site=x detail='y'"


def test_the_helpers_return_what_they_were_handed(fired):
    assert absorb_fire.fire_if("in-flight", False) is False
    assert fired.pairs() == []
    assert absorb_fire.fire_if("in-flight", True) is True
    assert fired.pairs() == [("in-flight", "")]
    assert absorb_fire.verdict("token-authority", True) is True
    assert absorb_fire.verdict("token-authority", False) is False
    assert fired.pairs()[-2:] == [("token-authority", "granted"),
                                  ("token-authority", "expired")]
    assert absorb_fire.hits("fault-marker", "no markers here", ("zzz",)) is False
    assert absorb_fire.hits("fault-marker", "a zzz b", ("zzz",)) is True
    assert fired.pairs()[-1] == ("fault-marker", "zzz")


@pytest.mark.parametrize("marker", mesh_safety._TEARDOWN_REASON_MARKERS)
def test_reason_marked_absorption_names_the_marker_that_proved_it(
        marker, fired, monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    assert mesh_safety.is_reason_marked_teardown(f"stopped: {marker}") is True
    assert fired.pairs() == [("reason-marked", marker)]


class _ReasonMarkedHandle:
    """Has no completed outcome or token; only the stop reason can absorb."""

    def __init__(self) -> None:
        self.defunct = False
        self.teardown_complete = False
        self.replacement_blocked = None


@pytest.mark.parametrize("marker", mesh_safety._TEARDOWN_REASON_MARKERS)
def test_handled_supervision_reuses_the_canonical_reason_markers(
        marker, fired, monkeypatch):
    from monarch._rust_bindings.monarch_hyperactor.supervision import (
        SupervisionError,
    )

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    handle = _ReasonMarkedHandle()
    mesh_helpers.mark_defunct_on_supervision_failure(
        handle, SupervisionError(f"proc has status: stopped: {marker}"))

    assert handle.defunct is True
    assert fired.pairs() == [("reason-marked", marker)]


def test_an_unmatched_fault_fires_nothing(fired, monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    assert mesh_safety.is_reason_marked_teardown(
        "actor panicked: assertion failed") is False
    assert fired.records == []


@pytest.mark.parametrize("marker", mesh_safety._TEARDOWN_FAULT_MARKERS)
def test_each_transport_marker_fires_under_its_own_name(
        marker, fired, monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    mesh_safety.is_deliberate_teardown_fault(f"fault: {marker}")
    assert fired.pairs() == [("fault-marker", marker)]


def test_a_fault_carrying_two_markers_fires_twice(fired, monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT)
    assert fired.pairs() == [("fault-marker", "undeliverable"),
                             ("fault-marker", "channel closed")]


def test_a_marker_line_is_not_an_absorption(fired, monkeypatch):
    """Another live mesh keeps the fault visible; marker lines still fire (docs/TROUBLESHOOTING.md #60)."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    mesh_safety.note_deliberate_teardown()
    assert mesh_safety.is_deliberate_teardown_fault(
        _TRANSPORT_FAULT, live_mesh_exists=True) is False
    assert fired.sites() == ["fault-marker", "fault-marker"]


def test_the_in_flight_and_grace_branches_fire_separately(fired, monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    assert mesh_safety.is_deliberate_teardown_fault(
        _TRANSPORT_FAULT, teardown_in_progress=True) is True
    assert "in-flight" in fired.sites()
    assert "grace-window" not in fired.sites()

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    fired.records.clear()
    mesh_safety.note_deliberate_teardown()
    assert mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT) is True
    assert "grace-window" in fired.sites()
    assert "in-flight" not in fired.sites()


def test_an_expired_grace_window_fires_only_the_marker(fired, monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    assert mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT) is False
    assert set(fired.sites()) == {"fault-marker"}


def test_token_authority_reports_granted_and_expired(fired, monkeypatch):
    token = 4242
    monkeypatch.setattr(
        mesh_safety, "_TEARDOWN_STATE", ({token: time.monotonic()}, float("-inf"), 0))
    assert mesh_safety.token_within_authority(token) is True
    assert fired.pairs() == [("token-authority", "granted")]

    fired.records.clear()
    stale = time.monotonic() - (mesh_safety.TOKEN_AUTHORITY_S + 10)
    monkeypatch.setattr(
        mesh_safety, "_TEARDOWN_STATE", ({token: stale}, float("-inf"), 0))
    assert mesh_safety.token_within_authority(token) is False
    assert fired.pairs() == [("token-authority", "expired")]

    fired.records.clear()
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    assert mesh_safety.token_within_authority(token) is False
    assert fired.records == []


class _TornDownHandle:
    """Takes the cheap deliberate branch, so no reconcile thread starts."""

    def __init__(self) -> None:
        self.defunct = False
        self.teardown_complete = True


@pytest.mark.parametrize("marker", ("Supervision", "ProcessExited",
                                    "connection lost", "peer closed"))
def test_string_fallback_fires_only_where_the_typed_check_missed(
        marker, fired, monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    mesh_helpers.mark_defunct_on_supervision_failure(
        _TornDownHandle(), RuntimeError(marker))
    assert fired.pairs() == [("string-fallback", marker)]


def test_a_typed_supervision_error_never_reaches_the_string_fallback(
        fired, monkeypatch):
    from monarch._rust_bindings.monarch_hyperactor.supervision import (
        SupervisionError,
    )

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    mesh_helpers.mark_defunct_on_supervision_failure(
        _TornDownHandle(), SupervisionError("proc has status: stopped"))
    assert [pair for pair in fired.pairs() if pair[0] == "string-fallback"] == []


def _truth_table() -> list[bool]:
    """The canonical absorption verdicts, recomputed from a clean state."""
    verdicts = []
    mesh_safety._TEARDOWN_STATE = _CLEAN_STATE
    verdicts.append(mesh_safety.is_deliberate_teardown_fault(
        "stopped: dgx-monarch recycle"))
    verdicts.append(mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT))
    mesh_safety.note_deliberate_teardown()
    verdicts.append(mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT))
    verdicts.append(mesh_safety.is_deliberate_teardown_fault(
        "actor panicked: assertion failed"))
    return verdicts


def test_no_verdict_changes_when_the_tracer_is_silenced(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    loud = _truth_table()
    logging.disable(logging.INFO)
    try:
        quiet = _truth_table()
    finally:
        logging.disable(logging.NOTSET)
    assert loud == [True, False, True, False]
    assert quiet == loud


def test_a_broken_tracer_cannot_change_reason_classification(monkeypatch):
    def fail_log(*_args, **_kwargs):
        raise KeyboardInterrupt("logging interrupted")

    monkeypatch.setattr(absorb_fire._log, "info", fail_log)
    assert mesh_safety.is_reason_marked_teardown(
        "stopped: dgx-monarch failed setup rollback"
    ) is True


def test_absorption_lines_are_never_errors(fired, monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", _CLEAN_STATE)
    mesh_safety.is_reason_marked_teardown("stopped: dgx-monarch recycle")
    mesh_safety.is_deliberate_teardown_fault(
        _TRANSPORT_FAULT, teardown_in_progress=True)
    mesh_safety.note_deliberate_teardown()
    mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT)
    monkeypatch.setattr(
        mesh_safety, "_TEARDOWN_STATE", ({7: time.monotonic()}, float("-inf"), 0))
    mesh_safety.token_within_authority(7)
    mesh_helpers.mark_defunct_on_supervision_failure(
        _TornDownHandle(), RuntimeError("connection lost"))
    assert set(fired.sites()) == set(absorb_fire.SITES)
    tagged = [r for r in fired.records if absorb_fire.TAG in r.getMessage()]
    assert tagged and all(r.levelno == logging.INFO for r in tagged)
