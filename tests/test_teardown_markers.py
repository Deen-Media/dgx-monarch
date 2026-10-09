"""Require a recognized marker for every deliberate procs.stop() reason.

An unlisted reason sends straggler faults through the weaker correlation check
instead of unconditional absorption. The source scan prevents new stop reasons
from omitting their marker.
"""
from __future__ import annotations

from pathlib import Path

from dgx_monarch.mesh_safety import _TEARDOWN_REASON_MARKERS, is_reason_marked_teardown

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"


def _stop_reasons_in_source() -> set[str]:
    """Every literal reason passed to a .stop("dgx-monarch ...") call in src."""
    import re

    reasons: set[str] = set()
    for path in SRC.rglob("*.py"):
        for match in re.finditer(r'\.stop\(\s*"(dgx-monarch [^"]+)"', path.read_text()):
            reasons.add(match.group(1))
    return reasons


def test_every_deliberate_stop_reason_is_reason_marked():
    reasons = _stop_reasons_in_source()
    assert reasons, "expected at least one deliberate stop reason in src/"
    for reason in sorted(reasons):
        assert is_reason_marked_teardown(
            f"actor mesh stopped: {reason}"
        ), f"stop reason not covered by _TEARDOWN_REASON_MARKERS: {reason!r}"


def test_partial_bring_up_rollback_is_marked():
    assert "dgx-monarch partial bring-up rollback" in " ".join(_TEARDOWN_REASON_MARKERS)
    assert is_reason_marked_teardown(
        "SupervisionError: proc has status: stopped: dgx-monarch partial bring-up rollback"
    )
