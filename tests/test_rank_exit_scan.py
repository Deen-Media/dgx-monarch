"""Every test rank that starts a Gloo process group in a subprocess ends through the shared ending.

``tests/rank_exit_helpers.py`` says why: on the success path a rank leaves
through ``os._exit(0)`` and never reaches interpreter finalization, where
PyTorch issue 179238 aborts a process whose Gloo threads outlived its group.
This scan reads the source of each ``tests/*.py`` and finds each such rank:

- any string constant that holds a script starting a Gloo group, assigned or
  passed inline, run by ``python -c`` or from a written file. The expression
  around it (string joins and ``textwrap.dedent``) must end with ``+ ENDING``,
  and the script must not call ``sys.exit``, ``os._exit``, ``exit`` or
  ``quit``, or ``raise SystemExit``, outside a function it defines, since that
  would leave before the ending;
- a module function that starts one and is handed to ``spawn`` or to a
  ``Process`` target. Its last statement must be a call to ``leave``, and it
  must hold no ``return`` of its own and no such exit before it;
- a helper module (not a test file) that starts one at module level, whose
  last statement must call ``leave``.

The scan is strict on purpose, so some shapes that would behave fail it and
need restructuring: an exit only on a failure path (raise an exception
instead, which still exits non-zero); a script built with ``.format`` or an
f-string before ``+ ENDING`` (pass the values through ``argv`` or the
environment and keep the script a plain string); and a docstring that quotes
the init call (describe it in words).

Not seen: a backend held in a name rather than the literal "gloo"; a rank
function handed to ``spawn`` through ``functools.partial`` or another wrapper;
a Gloo group that only xFuser's ``init_distributed_environment`` starts; an
exit inside a helper function the script defines and calls before the
ending; and files below ``tests/``. ``tests/fixtures/rank_exit_planted.py``
holds one of each shape, the seen and the unseen.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PLANTED = TESTS / "fixtures" / "rank_exit_planted.py"
_GLOO = re.compile(r"""init_process_group\(\s*(?:backend\s*=\s*)?["']gloo["']""")
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _named(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name) or (
        isinstance(node, ast.Attribute) and node.attr == name)


def _gloo_call(node: ast.AST) -> bool:
    """A call of ``init_process_group`` whose backend, by position or keyword, is "gloo"."""
    if not isinstance(node, ast.Call) or not _named(node.func, "init_process_group"):
        return False
    values = [*node.args[:1], *(keyword.value for keyword in node.keywords if keyword.arg == "backend")]
    return any(isinstance(value, ast.Constant) and value.value == "gloo" for value in values)


def _own(nodes: list[ast.stmt]):
    """Every node under ``nodes`` that belongs to their scope, not to a function or class defined inside."""
    pending: list[ast.AST] = [node for node in nodes if not isinstance(node, _SCOPES)]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, _SCOPES))


def _exits(nodes) -> bool:
    """A statement that leaves the process: sys.exit, os._exit, exit, quit, or raise SystemExit."""
    for node in nodes:
        if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Attribute) and node.func.attr in ("exit", "_exit"))
                or (isinstance(node.func, ast.Name) and node.func.id in ("exit", "quit", "_exit"))):
            return True
        if isinstance(node, ast.Raise) and node.exc is not None and _named(
                node.exc.func if isinstance(node.exc, ast.Call) else node.exc, "SystemExit"):
            return True
    return False


def _label(node: ast.AST, parents: dict) -> str:
    """The name a reader finds the rank by: the target it is assigned to, or the function holding it."""
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return "<module>"


def _wraps(parent: ast.AST, child: ast.AST) -> bool:
    """Whether ``parent`` builds the same script from ``child``: a join, or ``textwrap.dedent``."""
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add):
        return True
    return isinstance(parent, ast.Call) and _named(parent.func, "dedent") and child in parent.args


def _script_strings(tree: ast.Module, parents: dict):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and _GLOO.search(node.value)):
            continue
        whole: ast.AST = node
        while whole in parents and _wraps(parents[whole], whole):
            whole = parents[whole]
        ends = (isinstance(whole, ast.BinOp) and isinstance(whole.op, ast.Add)
                and _named(whole.right, "ENDING"))
        try:
            script = ast.parse(textwrap.dedent(node.value))
            leaves_early = _exits(_own(script.body))
        except SyntaxError:
            leaves_early = False
        yield _label(node, parents), node.lineno, ends and not leaves_early


def _spawned_functions(tree: ast.Module, _parents: dict):
    launched: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _named(node.func, "spawn") and node.args and isinstance(node.args[0], ast.Name):
                launched.add(node.args[0].id)
            for keyword in node.keywords:
                if keyword.arg == "target" and isinstance(keyword.value, ast.Name):
                    launched.add(keyword.value.id)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in launched \
                and any(_gloo_call(part) for part in ast.walk(node)):
            last = node.body[-1]
            ends = isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and _named(last.value.func, "leave")
            before = list(_own(node.body[:-1])) + [child for child in ast.walk(last) if child is not last.value]
            leaves_early = any(isinstance(part, ast.Return) for part in before) or _exits(before)
            yield node.name, node.lineno, ends and not leaves_early


def _module_rank(path: Path, tree: ast.Module):
    if path.name.startswith("test_") or path.name == "conftest.py":
        return
    module_level = list(_own(tree.body))
    if any(_gloo_call(part) for part in module_level):
        last = tree.body[-1]
        ends = isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and _named(last.value.func, "leave")
        yield path.stem, 1, ends and not _exits(_own(tree.body[:-1]))


def scan(path: Path) -> list[tuple[str, int, bool]]:
    """Each Gloo rank in ``path``: its name, its line, and whether it ends through the shared ending."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found = [*_script_strings(tree, parents), *_spawned_functions(tree, parents)]
    return found + list(_module_rank(path, tree))


def _ranks() -> list[tuple[str, str, int, bool]]:
    return [(path.name, *rank) for path in sorted(TESTS.glob("*.py")) for rank in scan(path)]


def test_every_gloo_rank_in_a_subprocess_ends_through_the_shared_ending():
    missing = [f"{name}:{line} ({label})" for name, label, line, ends in _ranks() if not ends]
    assert not missing, (
        "these test ranks start a Gloo process group in a subprocess and end some other way; end "
        "them with rank_exit_helpers.ENDING (a script string) or leave() (a spawned function), "
        f"which tests/rank_exit_helpers.py explains: {missing}")


def test_the_scan_sees_the_ranks_it_guards():
    """A scan that found nothing would pass for the wrong reason."""
    seen = {name for name, _label, _line, _ends in _ranks()}
    assert {"test_backup_agreement_two_ranks.py", "test_cond_batch_agreement.py",
            "test_cond_batch_agreement_four_ranks.py", "test_rank_readiness.py",
            "test_moss_image_attention_bias_worker.py", "test_adaln_parity_blocks.py",
            "test_qwen_image21_adapter.py", "test_wan_variants.py"} <= seen, seen


def test_the_scan_fails_each_planted_rank_that_could_bring_the_abort_back():
    """No ending, an exit, a raise SystemExit or a return before it, the ending
    first, an inline script, a Process target: each fails. A dedented script
    joined with the ending passes. The shapes the docstring lists as not seen
    stay unseen, or pass, as an exit inside a helper the script calls does."""
    verdicts = {label: ends for label, _line, ends in scan(PLANTED)}
    assert verdicts == {
        "P1_NO_ENDING_SHOULD_FAIL": False,
        "P2_EXITS_BEFORE_THE_ENDING": False,
        "p3_inline_script": False,
        "_p4_returns_early": False,
        "_p5_target_without_leave_should_fail": False,
        "P9_DEDENTED_WITH_ENDING_PASSES": True,
        "P10_ENDING_FIRST_SHOULD_FAIL": False,
        "_p12_exits_before_leave": False,
        "P13_RAISES_SYSTEMEXIT_BEFORE_THE_ENDING": False,
        "P14_EXIT_IN_A_HELPER_IT_CALLS": True,
    }, verdicts
