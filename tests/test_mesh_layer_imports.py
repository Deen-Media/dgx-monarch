"""The mesh layer stays below nodes/: no root ``mesh*.py`` imports from it.

The scan reads every import from the AST, so a function-local import in a
rarely taken branch counts like a module-level one, and a new mesh module joins
the scan by its name alone.
"""
from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"


def _names_a_nodes_module(module: str | None, level: int, names: list[str]) -> bool:
    """Whether one import statement reaches the nodes package."""
    if level == 0:
        if module is None:
            return False
        if module == "dgx_monarch":
            return any(name == "nodes" for name in names)
        return module == "dgx_monarch.nodes" or module.startswith("dgx_monarch.nodes.")
    if level == 1:
        if module is None:
            return any(name == "nodes" for name in names)
        return module == "nodes" or module.startswith("nodes.")
    return False


def _offenders(path: Path) -> list[str]:
    """Every nodes-reaching import in one file, as `path:line spelling`."""
    relative = path.relative_to(PACKAGE_ROOT).as_posix()
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.name == "dgx_monarch.nodes"
                        or alias.name.startswith("dgx_monarch.nodes.")):
                    found.append(f"{relative}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            if _names_a_nodes_module(node.module, node.level, names):
                spelling = "." * node.level + (node.module or "")
                found.append(
                    f"{relative}:{node.lineno} from {spelling} import "
                    + ", ".join(names))
    return found


def test_no_mesh_module_imports_from_the_nodes_package():
    scanned = sorted(PACKAGE_ROOT.glob("mesh*.py"))
    assert scanned, f"no mesh modules found under {PACKAGE_ROOT}; the scan proved nothing"
    offenders = []
    for path in scanned:
        offenders.extend(_offenders(path))
    assert not offenders, (
        "the mesh layer must not import from nodes/: " + "; ".join(offenders)
        + ". The owner lock lives in mesh_session.py; a mesh module that needs "
        "a nodes name means the name is in the wrong layer, so move the name "
        "down rather than adding an exception here."
    )


def test_the_nodes_import_detector_sees_every_spelling():
    """Five spellings that reach nodes/ must count as hits and four near misses must not.

    A detector blind to a spelling would let the scan above pass while a nodes import exists.
    """
    source = (
        "import dgx_monarch.nodes.render_session\n"
        "from dgx_monarch.nodes.gate_fsdp import FSDP_PROOF_SCOPE\n"
        "from .nodes.consent_observe import observe_refusal\n"
        "from . import nodes\n"
        "from dgx_monarch import nodes as bound\n"
    )
    tree = ast.parse(source)
    hits = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += sum(
                1 for alias in node.names
                if alias.name.startswith("dgx_monarch.nodes"))
        elif isinstance(node, ast.ImportFrom):
            hits += bool(_names_a_nodes_module(
                node.module, node.level, [alias.name for alias in node.names]))
    assert hits == 5
    assert not _names_a_nodes_module("dgx_monarch.nodesmith", 0, [])
    assert not _names_a_nodes_module("dgx_monarch", 0, ["mesh_setup"])
    assert not _names_a_nodes_module("mesh_setup", 1, [])
    assert not _names_a_nodes_module(None, 1, ["mesh_setup"])
