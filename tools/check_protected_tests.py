#!/usr/bin/env python3
"""Guard the review-required test inventory against unreviewed removal or edits.

Each protected test is recorded with a digest of its normalized source: its
decorators (parametrize names, rows and ids included), signature and body
without the docstring, plus every module-level definition it reaches by name
inside its own module (case tables, helpers, same-module fixtures, autouse
fixtures and ``pytestmark``). Removing a row from a table that a protected test
is parametrized over therefore changes that test's digest.

CI runs ``--check`` (the default): a missing, renamed or edited protected test
fails until ``--write`` re-baselines the inventory in the same diff. ``--write``
prints which protected tests it added, removed or changed, so the reviewer sees
which tests to read. A digest detects change; it does not prove an
edited test equivalent to the one it replaced, and it does not follow helpers
imported from other modules, conftest fixtures or the source under test.

The normalized form renders the syntax tree as JSON without positions,
load/store context, string-prefix kinds or empty optional fields, and folds the
literal pieces of f-strings. Python 3.11, 3.12 and 3.13 therefore produce the
same digest, and comments, formatting and docstrings never change one.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

MANIFEST = Path("tests/protected_test_inventory.json")
MANIFEST_VERSION = 2
# 64 bits: the guard detects accidental or unreviewed edits, and a reviewer
# reads every re-baselined test, so a longer digest adds nothing.
DIGEST_HEX_CHARS = 16
REQUIRED_FILES = (
    "tests/refusal_class_ledger.json",
    "tests/canary/bake_equivalence_canary.py",
    "tests/canary/comfy_entrypoint_canary.py",
    "tests/canary/comfy_seam_contracts.py",
    "tests/canary/import_canary.py",
    "tests/canary/monarch_surface_canary.py",
    "tests/canary/slab_hook_canary.py",
    "tests/canary/template_widget_canary.py",
)
SUITE_MODULES = (
    "tests/test_capacity_quote.py",
    "tests/test_gate_ledger.py",
    "tests/test_gate_ledger_properties.py",
    "tests/test_refusal_classes.py",
    "tests/test_sweep_matrix.py",
    "tests/test_sweep_run.py",
)
CAPACITY_MODULES = (
    "tests/test_capacity_agreement.py",
    "tests/test_capacity_lora_bake.py",
    "tests/test_capacity_memory.py",
    "tests/test_capacity_quote.py",
    "tests/test_comfy_managed_pricing.py",
    "tests/test_design_capacity_table.py",
    "tests/test_driver_footprint_preflight.py",
    "tests/test_fsdp_capacity.py",
    "tests/test_fsdp_reload_price.py",
    "tests/test_render_memory_price.py",
    "tests/test_residency_ladder.py",
    "tests/test_rescue_card_need.py",
    "tests/test_slab.py",
    "tests/test_slab_certificate.py",
    "tests/test_store_slab_routing.py",
)

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DOCUMENTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# Fields with no test semantics, or whose spelling differs between interpreter
# versions. Positions never appear because only ``_fields`` are walked; ``ctx``
# follows from the syntax, ``kind`` records a redundant ``u`` prefix and
# ``type_comment`` is a comment.
_OMITTED_FIELDS = frozenset({"ctx", "kind", "type_comment"})
# Names pytest applies to a test without the test naming them.
_IMPLICIT_MODULE_NAMES = frozenset({"pytestmark"})
_IMPLICIT_CLASS_NAMES = frozenset({
    "pytestmark", "setup_method", "teardown_method", "setup_class", "teardown_class",
})

# One definition a test can reach by name: a whole statement, or one imported
# alias rendered alone, so adding an unrelated name to an import line does not
# change every digest that uses that line.
_Unit = ast.AST | tuple[str, ...]


def _valid_relative_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
    )


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_explicit_refusal_module(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "RefusalClass":
            return True
        if isinstance(node, ast.alias) and node.name.endswith("RefusalClass"):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "[dgxm:" in node.value:
            return True
    return False


def _test_nodes(tree: ast.Module) -> Iterator[tuple[str, ast.AST, ast.ClassDef | None]]:
    for node in tree.body:
        if isinstance(node, _FUNCTIONS) and node.name.startswith("test_"):
            yield node.name, node, None
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for member in node.body:
                if isinstance(member, _FUNCTIONS) and member.name.startswith("test_"):
                    yield f"{node.name}.{member.name}", member, node


def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _literal(text: str) -> list[object]:
    return ["Constant", ["value", ["str", repr(text)]]]


def _folded_fstring(values: list[ast.expr]) -> list[object]:
    # Python 3.12 appends an empty literal to a nested format spec, which 3.11
    # and 3.13 do not; the rendered text is the same, so adjacent literal
    # pieces are joined and empty ones dropped.
    folded: list[object] = []
    pending = ""
    for value in values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            pending += value.value
            continue
        if pending:
            folded.append(_literal(pending))
            pending = ""
        folded.append(_canonical(value))
    if pending:
        folded.append(_literal(pending))
    return folded


def _canonical(node: object, *, omit_name: bool = False) -> object:
    if isinstance(node, ast.AST):
        rendered: list[object] = [type(node).__name__]
        for field in node._fields:
            if field in _OMITTED_FIELDS or (omit_name and field == "name"):
                continue
            value = getattr(node, field, None)
            if field == "body" and isinstance(node, _DOCUMENTED):
                value = _without_docstring(value)
            if field == "values" and isinstance(node, ast.JoinedStr):
                rendered.append([field, _folded_fstring(value)])
                continue
            # 3.12 added empty ``type_params`` and 3.13 fills absent optional
            # fields with None; an absent and an empty field render alike.
            if (value is None or value == []) and not isinstance(node, ast.Constant):
                continue
            rendered.append([field, _canonical(value)])
        return rendered
    if isinstance(node, list):
        return [_canonical(item) for item in node]
    if node is None:
        return None
    if node is Ellipsis or isinstance(node, (bool, int, float, complex, str, bytes)):
        return [type(node).__name__, repr(node)]
    raise TypeError(f"unexpected syntax tree value: {type(node).__name__}")


def _render(node: object, *, omit_name: bool = False) -> str:
    return json.dumps(_canonical(node, omit_name=omit_name), separators=(",", ":"))


def _render_unit(unit: _Unit) -> str:
    return json.dumps(list(unit)) if isinstance(unit, tuple) else _render(unit)


def _target_names(targets: Iterable[ast.AST]) -> set[str]:
    # Every name in a target counts, so ``TABLE[0] = row`` and ``a, b = rows``
    # both mark the statement as a definition of the names they touch.
    return {child.id for target in targets for child in ast.walk(target) if isinstance(child, ast.Name)}


def _mutated_name(statement: ast.Expr) -> set[str]:
    # ``ROWS.append(...)`` and ``ROWS.extend(...)`` extend a case table.
    call = statement.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return set()
    base = call.func.value
    while isinstance(base, ast.Attribute):
        base = base.value
    return {base.id} if isinstance(base, ast.Name) else set()


def _bound_names(statement: ast.stmt) -> set[str]:
    """Names one module- or class-level statement defines or mutates."""
    if isinstance(statement, _DOCUMENTED):
        return {statement.name}
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return {alias.asname or alias.name.split(".")[0] for alias in statement.names}
    if isinstance(statement, ast.Assign):
        return _target_names(statement.targets)
    if isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        return _target_names([statement.target])
    if isinstance(statement, ast.Expr):
        return _mutated_name(statement)
    names: set[str] = set()
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        names |= _target_names([statement.target])
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        names |= _target_names(item.optional_vars for item in statement.items if item.optional_vars)
    for field in ("body", "orelse", "finalbody", "handlers", "cases"):
        for child in getattr(statement, field, None) or ():
            if isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            nested = child.body if isinstance(child, (ast.ExceptHandler, ast.match_case)) else [child]
            for inner in nested:
                names |= _bound_names(inner)
    return names


def _definitions(statements: Iterable[ast.stmt]) -> dict[str, list[_Unit]]:
    definitions: dict[str, list[_Unit]] = {}
    for statement in statements:
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                unit = ("ImportFrom", statement.module or "", str(statement.level or 0),
                        alias.name, alias.asname or "")
                definitions.setdefault(alias.asname or alias.name, []).append(unit)
            continue
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                unit = ("Import", alias.name, alias.asname or "")
                definitions.setdefault(alias.asname or alias.name.split(".")[0], []).append(unit)
            continue
        for name in _bound_names(statement):
            definitions.setdefault(name, []).append(statement)
    return definitions


def _is_autouse_fixture(unit: _Unit) -> bool:
    if not isinstance(unit, _FUNCTIONS):
        return False
    return any(
        isinstance(decorator, ast.Call)
        and any(
            keyword.arg == "autouse"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
        for decorator in unit.decorator_list
    )


def _references(unit: _Unit) -> set[str]:
    """Names a definition may resolve in its enclosing scope, over-approximated."""
    names: set[str] = set()
    if isinstance(unit, tuple):
        return names
    for child in ast.walk(unit):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.arg):
            # A parameter names the fixture pytest injects for it.
            names.add(child.arg)
        elif (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id in {"self", "cls"}
        ):
            names.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            # usefixtures("x") and request.getfixturevalue("x") name fixtures.
            names.add(child.value)
    return names


class _Scope:
    """The definitions of one module or class body, with per-unit memos.

    Every unit stays referenced by ``definitions`` or by the parsed tree for
    the scope's lifetime, so ``id(unit)`` is a stable memo key.
    """

    def __init__(self, statements: Iterable[ast.stmt], implicit_names: frozenset[str]):
        self.definitions = _definitions(statements)
        self.implicit = [
            unit for name, units in self.definitions.items() for unit in units
            if name in implicit_names or _is_autouse_fixture(unit)
        ]
        self._names: dict[int, set[str]] = {}

    def references(self, unit: _Unit) -> set[str]:
        key = id(unit)
        if key not in self._names:
            self._names[key] = _references(unit)
        return self._names[key]

    def reach(self, roots: Iterable[_Unit], skip: ast.AST) -> list[_Unit]:
        """Every definition the roots reach by name, transitively, plus those
        pytest applies without a name (marks, hooks and autouse fixtures)."""
        reached: dict[int, _Unit] = {id(unit): unit for unit in self.implicit if unit is not skip}
        pending: list[_Unit] = [*roots, *reached.values()]
        while pending:
            for name in self.references(pending.pop()):
                for unit in self.definitions.get(name, ()):
                    if unit is skip or id(unit) in reached:
                        continue
                    reached[id(unit)] = unit
                    pending.append(unit)
        return list(reached.values())


def digest_tests(tree: ast.Module) -> dict[str, str]:
    """Digest every collected test in one parsed test module."""
    module = _Scope(tree.body, _IMPLICIT_MODULE_NAMES)
    classes: dict[int, _Scope] = {}
    texts: dict[int, str] = {}
    digests: dict[str, str] = {}
    for name, node, owner in _test_nodes(tree):
        reached: list[_Unit] = []
        if owner is not None:
            # A method also reaches its class's decorators, marks, hooks,
            # autouse fixtures and the members it names; all of those then
            # resolve their own names against the module.
            scope = classes.setdefault(id(owner), _Scope(owner.body, _IMPLICIT_CLASS_NAMES))
            reached += scope.reach([node], node)
            reached += owner.decorator_list
            reached += module.reach([node, *reached], owner)
        else:
            reached += module.reach([node], node)
        for unit in reached:
            if id(unit) not in texts:
                texts[id(unit)] = _render_unit(unit)
        rendered = sorted({texts[id(unit)] for unit in reached})
        payload = json.dumps([_render(node, omit_name=True), rendered], separators=(",", ":"))
        digests[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:DIGEST_HEX_CHARS]
    return dict(sorted(digests.items()))


def _parse(root: Path, rel: str) -> ast.Module:
    return ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)


def discover(root: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    protected = {*SUITE_MODULES, *CAPACITY_MODULES}
    errors: list[str] = []
    parsed: dict[str, ast.Module] = {}
    for path in sorted((root / "tests").glob("test_*.py")):
        rel = _relative(root, path)
        try:
            tree = _parse(root, rel)
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append(f"cannot inspect {rel}: {exc}")
            continue
        parsed[rel] = tree
        if rel.startswith("tests/test_capacity_") or _is_explicit_refusal_module(tree):
            protected.add(rel)
    for rel in (*REQUIRED_FILES, *SUITE_MODULES, *CAPACITY_MODULES):
        if not (root / rel).is_file():
            errors.append(f"required protected path is missing: {rel}")
    return ({rel: digest_tests(parsed[rel]) for rel in sorted(protected) if rel in parsed}, errors)


def build_inventory(root: Path) -> tuple[dict[str, object], dict[str, dict[str, str]], list[str]]:
    protected_tests, errors = discover(root)
    return ({
        "version": MANIFEST_VERSION,
        "required_files": list(REQUIRED_FILES),
        "protected_tests": protected_tests,
    }, protected_tests, errors)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    if value.get("version") != MANIFEST_VERSION:
        raise ValueError(f"manifest version must be {MANIFEST_VERSION}; re-baseline it with --write")
    if not isinstance(value.get("required_files"), list):
        raise ValueError("manifest required_files must be a list")
    if not isinstance(value.get("protected_tests"), dict):
        raise ValueError("manifest protected_tests must be an object")
    return value


def check(root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = _load(manifest_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot read protected-test inventory {manifest_path}: {exc}"]

    required = manifest["required_files"]
    if required != list(REQUIRED_FILES):
        errors.append("manifest required_files does not match the tool's REQUIRED_FILES list")
    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            errors.append(f"protected file is missing: {rel}")
    for rel in required:
        if not _valid_relative_path(rel):
            errors.append(f"invalid required file entry: {rel!r}")

    discovered, discovery_errors = discover(root)
    recorded = manifest["protected_tests"]
    if not isinstance(recorded, dict):
        return [*errors, "manifest protected_tests must be an object"]
    for rel, digests in recorded.items():
        if not _valid_relative_path(rel) or not isinstance(digests, dict):
            errors.append(f"invalid protected test entry: {rel!r}")
            continue
        path = root / rel
        if not path.is_file():
            errors.append(f"protected test module is missing: {rel}")
            continue
        actual = discovered.get(rel)
        if actual is None:
            try:
                actual = digest_tests(_parse(root, rel))
            except (OSError, UnicodeError, SyntaxError) as exc:
                errors.append(f"cannot inspect {rel}: {exc}")
                continue
        for name, digest in digests.items():
            if name not in actual:
                errors.append(f"protected test is missing or renamed: {rel}::{name}")
            elif actual[name] != digest:
                errors.append(f"protected test changed without an inventory re-baseline: {rel}::{name}")

    errors.extend(discovery_errors)
    for rel, digests in discovered.items():
        known = recorded.get(rel)
        if not isinstance(known, dict):
            errors.append(f"unrecorded protected test module: {rel}")
            continue
        for name in digests:
            if name not in known:
                errors.append(f"unrecorded protected test: {rel}::{name}")
    return errors


def _flatten(protected: object) -> dict[str, str | None]:
    """Map rel::name to its digest; a version 1 list recorded names only."""
    flat: dict[str, str | None] = {}
    if not isinstance(protected, dict):
        return flat
    for rel, tests in protected.items():
        if isinstance(tests, dict):
            flat.update({f"{rel}::{name}": digest for name, digest in tests.items()})
        elif isinstance(tests, list):
            flat.update({f"{rel}::{name}": None for name in tests if isinstance(name, str)})
    return flat


def rebaseline_report(previous: object, current: dict[str, dict[str, str]]) -> list[str]:
    """Name each protected test a re-baseline removes, adds or changes."""
    old = _flatten(previous)
    new = _flatten(current)
    lines = [f"removed: {key}" for key in sorted(old.keys() - new.keys())]
    lines += [f"added: {key}" for key in sorted(new.keys() - old.keys())]
    lines += [
        f"{'changed' if old[key] is not None else 'first digest'}: {key}"
        for key in sorted(old.keys() & new.keys())
        if old[key] != new[key]
    ]
    return lines


def _previous_tests(manifest: Path) -> object:
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return value.get("protected_tests") if isinstance(value, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true",
                      help="re-baseline the inventory and print each protected test it adds, removes or changes")
    mode.add_argument("--check", action="store_true", help="check the inventory (default)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    if args.write:
        inventory, protected_tests, errors = build_inventory(root)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        report = rebaseline_report(_previous_tests(manifest), protected_tests)
        # discover() builds every mapping in sorted order, so the file is
        # byte-stable while version stays the first key a reader sees.
        manifest.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        print("\n".join(report) if report else "protected-test inventory unchanged")
        return 0
    errors = check(root, manifest)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
