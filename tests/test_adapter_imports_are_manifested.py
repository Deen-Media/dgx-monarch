"""Every Comfy name an adapter imports is a declared touchpoint.

A production ``from comfy... import X`` or ``import comfy...`` in an adapter is
a live upstream dependency, and an undeclared one can be renamed upstream
without the canary naming it. The manifest in ``comfy_surface`` is the one
inventory; this scan checks adapter imports against it, at module level and
inside functions.

Whole-module and dotted imports are checked member by member: for example,
``comfy.ldm.common_dit.pad_to_patch_size`` must be declared under that exact
module; the same name declared under another module does not count.
"""
from __future__ import annotations

import ast
from pathlib import Path

from dgx_monarch.adapters import ADAPTERS as ADAPTER_CLASSES
from dgx_monarch.adapters.detect import model_base_touchpoints
from dgx_monarch.comfy_surface import TOUCHPOINTS

ADAPTERS = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch" / "adapters"

# Adapter.matches reaches model_base classes through ``getattr(model_base, name)``,
# where AST cannot see the member. assert_comfy_surface resolves those names at
# run time from each adapter's class tuples (model_base_touchpoints).
_MODULES_WITH_DYNAMIC_MEMBER_COVERAGE = frozenset({"comfy.model_base"})


def _declared() -> frozenset[tuple[str, str]]:
    return frozenset(
        [(touchpoint.module, touchpoint.attribute) for touchpoint in TOUCHPOINTS]
        + [
            ("comfy.model_base", name)
            for name in model_base_touchpoints(ADAPTER_CLASSES)
        ]
    )


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ()
    return (node.id, *reversed(parts))


def _maximal_attribute_chains(
    tree: ast.AST,
) -> tuple[tuple[tuple[str, ...], int], ...]:
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    chains: list[tuple[tuple[str, ...], int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue
        chain = _attribute_chain(node)
        if chain:
            chains.append((chain, node.lineno))
    return tuple(chains)


def _member_uses(
    chains: tuple[tuple[tuple[str, ...], int], ...],
    *,
    module: str,
    local_name: str,
    dotted_binding: bool,
) -> tuple[tuple[str, int], ...]:
    prefix = tuple(module.split(".")) if dotted_binding else (local_name,)
    uses: list[tuple[str, int]] = []
    for chain, lineno in chains:
        if chain[:len(prefix)] != prefix or len(chain) == len(prefix):
            continue
        uses.append((".".join(chain[len(prefix):]), lineno))
    return tuple(uses)


def _is_module_object(module: str, declared_modules: frozenset[str]) -> bool:
    return (
        module in declared_modules
        or module in _MODULES_WITH_DYNAMIC_MEMBER_COVERAGE
    )


def test_every_adapter_comfy_import_is_a_declared_touchpoint():
    declared = _declared()
    declared_modules = frozenset(module for module, _attribute in declared)
    missing: list[str] = []
    for path in sorted(ADAPTERS.glob("*.py")):
        tree = ast.parse(path.read_text())
        chains = _maximal_attribute_chains(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("comfy"):
                        continue
                    uses = _member_uses(
                        chains,
                        module=alias.name,
                        local_name=alias.asname or alias.name.split(".")[0],
                        dotted_binding=alias.asname is None,
                    )
                    if not uses and alias.name in _MODULES_WITH_DYNAMIC_MEMBER_COVERAGE:
                        continue
                    if not uses:
                        missing.append(
                            f"{path.name}:{node.lineno}: import {alias.name} "
                            "has no statically declared member use"
                        )
                    for member, lineno in uses:
                        if (alias.name, member) not in declared:
                            missing.append(
                                f"{path.name}:{lineno}: {alias.name}.{member}"
                            )
                continue

            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("comfy"):
                continue
            for alias in node.names:
                imported = f"{node.module}.{alias.name}"
                if _is_module_object(imported, declared_modules):
                    uses = _member_uses(
                        chains,
                        module=imported,
                        local_name=alias.asname or alias.name,
                        dotted_binding=False,
                    )
                    if not uses and imported in _MODULES_WITH_DYNAMIC_MEMBER_COVERAGE:
                        continue
                    if not uses:
                        missing.append(
                            f"{path.name}:{node.lineno}: from {node.module} import "
                            f"{alias.name} has no statically declared member use"
                        )
                    for member, lineno in uses:
                        if (imported, member) not in declared:
                            missing.append(f"{path.name}:{lineno}: {imported}.{member}")
                    continue
                if (node.module, alias.name) not in declared:
                    missing.append(
                        f"{path.name}:{node.lineno}: from {node.module} "
                        f"import {alias.name}"
                    )
    assert not missing, (
        "adapter comfy imports missing from the touchpoint manifest "
        "(declare the exact module/member in comfy_forward_contracts.py):\n"
        + "\n".join(missing)
    )


def test_the_dynamic_module_coverage_allowlist_is_not_stale():
    """A dynamically covered module that no adapter imports anymore must leave."""
    used: set[str] = set()
    for path in ADAPTERS.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    used.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    used.add(f"{node.module}.{alias.name}")
    stale = _MODULES_WITH_DYNAMIC_MEMBER_COVERAGE - used
    assert not stale, f"allowlist rows no adapter imports anymore: {sorted(stale)}"
