"""Keep observed yes, observed no and unobserved distinct before mutation.

Stop, restart, pull, pin and bind decisions must not treat a failed probe as an
observed absence. For example, a driver timeout cannot authorize an update, and
an unanswered SSH request cannot authorize restarting persistent Worker services.

These source scans reject helpers and call sites that collapse probe results
to booleans, while allowing reviewed predicates about static properties.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"

# The two probes that answer "is the driver up" and "is this loop up".
_ACTIVITY_MODULES = ("cli/comfy_ports.py", "cli/worker_health.py")

# Modules scanned for a collapsing call before a pull, pin or restart.
# `cli/lifecycle.py` is here because it owns `restart`, the third sink above;
# the two modules that call it are already on the list.
_MUTATION_SINKS = (
    "cli/legacy_update.py", "cli/lifecycle.py", "cli/main.py", "mesh_attach.py")

# A helper spelled this way answers a yes/no question about a live thing.
_COLLAPSING = re.compile(r"^_?(running_\w+|is_\w+|healthy)$")

# Legitimate non-mutation readers. Every entry is a predicate about something
# static, never about whether a driver or a loop is alive, so none of them can
# collapse a probe. Adding a name here is the reviewed way past the guard.
_NON_MUTATION_READERS = frozenset({
    "is_dir",     # pathlib: this checkout carries a .git directory
    "is_file",    # pathlib: this checkout carries a pyproject.toml
    "is_local",   # lifecycle_host: which host this process is, not who is up
    "_is_local",  # the same function under the alias lifecycle.py imports
})

_TRI_STATE_READERS = ("driver_probe", "passive_unhealthy_workers")


@functools.cache
def _modules() -> tuple[tuple[str, ast.Module], ...]:
    # Parsed once per run: five checks read the same source tree and none of
    # them mutates a node, so one parse of every module serves them all.
    trees = []
    for path in sorted(SRC.rglob("*.py")):
        trees.append((
            path.relative_to(SRC).as_posix(),
            ast.parse(path.read_text(encoding="utf-8")),
        ))
    return tuple(trees)


@pytest.fixture(scope="module", autouse=True)
def _parsed_tree_ends_with_this_module():
    # The cached trees hold every node of the source; let them go once the
    # five checks are done rather than carry them through the rest of the run.
    yield
    _modules.cache_clear()


def _with_parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _called_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _bound_name(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> str | None:
    """The name this call's whole result is bound to, if it is bound at all."""
    parent = parents.get(node)
    if isinstance(parent, ast.Assign) and parent.value is node:
        target = parent.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _keeps_the_whole_answer(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether this call's result travels on intact.

    Two shapes keep all three answers: an assignment binding the pair or the
    record, and a return handing it to the caller. Every other parent narrows
    the result before anyone branches on it, so this is an allow list rather
    than a list of bad spellings. A subscript (`driver_probe(host)[0]`), a
    comparison, a wrapper call and a plain `if` all fail the same way: the
    collapse has more spellings than a deny list can name.
    """
    parent = parents.get(node)
    if isinstance(parent, ast.Assign) and parent.value is node:
        target = parent.targets[0]
        if isinstance(target, ast.Name):
            return True
        return isinstance(target, ast.Tuple) and len(target.elts) == 2
    return isinstance(parent, ast.Return) and parent.value is node


def test_running_driver_is_gone_and_stays_gone():
    """The named collapse cannot come back under its own name.

    `running_driver` dropped `driver_probe`'s observed flag and returned None
    for both "no driver" and "no answer". `legacy_update` read that None as
    "ComfyUI is not running" and pulled, pinned, installed and restarted.
    """
    from dgx_monarch.cli import comfy_ports

    assert not hasattr(comfy_ports, "running_driver")

    for relative, tree in _modules():
        defined = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "running_driver" not in defined, relative


def test_no_collapsing_helper_is_called_on_a_mutation_path():
    """A yes/no helper cannot reach pull, pin, or restart.

    The names say so: `running_*`, `is_*`, and `healthy` answer with one
    bit, and one bit cannot carry "the probe never answered". A new
    `is_up()` in front of these mutations fails here until someone either
    gives it a tri-state or records it as a static predicate.
    """
    offenders = []
    exported = []
    for relative, tree in _modules():
        if relative in _ACTIVITY_MODULES:
            exported.extend(
                f"{relative}:{node.lineno} {node.name}"
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _COLLAPSING.fullmatch(node.name)
                and node.name not in _NON_MUTATION_READERS
            )
        if relative not in _MUTATION_SINKS:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if _COLLAPSING.fullmatch(name) and name not in _NON_MUTATION_READERS:
                offenders.append(f"{relative}:{node.lineno} {name}")
    # Nothing calls one on a mutation path, and the probe modules do not offer
    # one to call. The second half is why the rule says delete, not annotate.
    assert offenders == []
    assert exported == []


def test_driver_activity_and_worker_health_reach_callers_as_tri_states():
    """Every production call site keeps all three answers.

    `driver_probe` is unpacked as a pair or returned whole; the fleet health
    record is bound or returned whole. Anything that narrows the result at
    the call itself fails, because narrowing is where the third answer goes
    missing: `driver_probe(host)[0]` reads exactly like the deleted
    `running_driver` did, and no name in it matches `_COLLAPSING`.
    """
    offenders = []
    for relative, tree in _modules():
        parents = _with_parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _called_name(node) not in _TRI_STATE_READERS:
                continue
            if not _keeps_the_whole_answer(node, parents):
                offenders.append(f"{relative}:{node.lineno} answer narrowed")
    assert offenders == []


def test_the_fleet_health_result_is_read_by_field():
    """A module that asks for fleet health reads both collections.

    Reading only `dead` is the old collapse with a new spelling: the unknown
    hosts would vanish and the restart would fire on the remainder. The
    fields are checked against the name this call was bound to, not against
    every attribute in the module, so a module that happens to spell
    `probe_certainty.unobserved` somewhere else cannot satisfy it.
    """
    offenders = []
    for relative, tree in _modules():
        parents = _with_parents(tree)
        reads: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                reads.setdefault(node.value.id, set()).add(node.attr)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _called_name(node) != "passive_unhealthy_workers":
                continue
            if isinstance(parents.get(node), ast.Return):
                continue  # the record travels on whole
            bound = _bound_name(node, parents)
            if bound is None:
                offenders.append(f"{relative}:{node.lineno} not bound to a name")
                continue
            if not {"dead", "unobserved"} <= reads.get(bound, set()):
                offenders.append(f"{relative}:{node.lineno} {bound}")
    assert offenders == []


def test_probe_certainty_owns_the_only_unknown_vocabulary():
    """One home for "the probe never answered", spelled once.

    An alias that points back at `probe_certainty` is fine; a second literal
    is a fourth vocabulary, and the operator surfaces would drift apart. Both
    ways of minting one are checked: a module constant that spells the value
    itself, and a command that returns the bare exit code without naming it.
    """
    from dgx_monarch.cli import probe_certainty

    offenders = []
    for relative, tree in _modules():
        if relative == "cli/probe_certainty.py":
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if "UNKNOWN_EXIT" not in target.id and "UNOBSERVED" not in target.id:
                    continue
                if not isinstance(node.value, (ast.Name, ast.Attribute)):
                    offenders.append(f"{relative}:{node.lineno} {target.id}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Return):
                continue
            value = node.value
            if not isinstance(value, ast.Constant) or value.value is True:
                continue
            if value.value == probe_certainty.UNKNOWN_EXIT:
                offenders.append(f"{relative}:{node.lineno} bare unknown exit")
    assert offenders == []
