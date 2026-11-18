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


# Every stop call that takes the reason literally: a mesh's own stop, and the
# pack's two stop paths, which pass it on to ActorMesh.stop and ProcMesh.stop.
_STOP_CALLS = ("stop", "stop_with_reason", "stop_procs")


def _literals(scope) -> dict[str, str]:
    """Names bound to a "dgx-monarch ..." string literal directly in ``scope``'s body."""
    import ast

    bound: dict[str, str] = {}
    for node in ast.walk(scope):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                for target in targets:
                    if isinstance(target, ast.Name) and value.value.startswith("dgx-monarch "):
                        bound[target.id] = value.value
    return bound


def _stop_reasons_in_source() -> set[str]:
    """Every "dgx-monarch ..." reason a stop call in src receives.

    A reason given positionally or by keyword, as a literal or through a name
    bound to a literal in the same function or at module level, counts.
    """
    import ast

    reasons: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        module = _literals(tree)
        scopes = [node for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))] + [tree]
        for scope in scopes:
            names = {**module, **_literals(scope)} if scope is not tree else module
            for node in ast.walk(scope):
                if not isinstance(node, ast.Call):
                    continue
                if (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) not in _STOP_CALLS:
                    continue
                for arg in [*node.args, *(keyword.value for keyword in node.keywords)]:
                    if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                            and arg.value.startswith("dgx-monarch ")):
                        reasons.add(arg.value)
                    elif isinstance(arg, ast.Name) and arg.id in names:
                        reasons.add(names[arg.id])
    return reasons


def test_the_scan_reads_keywords_and_names_bound_to_a_literal(tmp_path, monkeypatch):
    """A reason the scan cannot see is a reason nobody checks for its marker."""
    import sys

    module = sys.modules[__name__]
    (tmp_path / "sample.py").write_text(
        'REASON = "dgx-monarch module reason"\n'
        "def a(procs, handle):\n"
        '    procs.stop(reason="dgx-monarch keyword reason")\n'
        '    reason = "dgx-monarch bound reason"\n'
        "    stop_with_reason(handle, reason, 30)\n"
        "    stop_procs(procs, REASON, 30)\n")
    monkeypatch.setattr(module, "SRC", tmp_path)
    assert _stop_reasons_in_source() == {
        "dgx-monarch keyword reason", "dgx-monarch bound reason", "dgx-monarch module reason"}


def test_every_deliberate_stop_reason_is_reason_marked():
    reasons = _stop_reasons_in_source()
    # The scan must still see each stop path's reason wherever the call moves.
    assert reasons >= {"dgx-monarch client detach", "dgx-monarch recycle",
                       "dgx-monarch failed setup rollback",
                       "dgx-monarch partial bring-up rollback"}, reasons
    for reason in sorted(reasons):
        assert is_reason_marked_teardown(
            f"actor mesh stopped: {reason}"
        ), f"stop reason not covered by _TEARDOWN_REASON_MARKERS: {reason!r}"


def test_partial_bring_up_rollback_is_marked():
    assert "dgx-monarch partial bring-up rollback" in " ".join(_TEARDOWN_REASON_MARKERS)
    assert is_reason_marked_teardown(
        "SupervisionError: proc has status: stopped: dgx-monarch partial bring-up rollback"
    )
