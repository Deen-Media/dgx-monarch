"""Focused fixtures for the protected-test inventory guard."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

TOOL = Path(__file__).parents[1] / "tools" / "check_protected_tests.py"
SPEC = importlib.util.spec_from_file_location("check_protected_tests", TOOL)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)

GATE = "tests/test_gate_ledger.py"


def _root(tmp_path: Path) -> Path:
    (tmp_path / "tests" / "canary").mkdir(parents=True)
    for rel in guard.REQUIRED_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# required\n", encoding="utf-8")
    for rel in (*guard.SUITE_MODULES, *guard.CAPACITY_MODULES):
        path = tmp_path / rel
        path.write_text("def test_baseline(): pass\n", encoding="utf-8")
    return tmp_path


def _write_manifest(root: Path) -> Path:
    manifest = root / "tests" / "protected_test_inventory.json"
    assert guard.main(["--root", str(root), "--manifest", str(manifest), "--write"]) == 0
    return manifest


def _digest(source: str, name: str = "test_case") -> str:
    return guard.digest_tests(ast.parse(source))[name]


def test_check_detects_a_deleted_required_file(tmp_path):
    root = _root(tmp_path)
    (root / GATE).write_text("def test_present(): pass\n", encoding="utf-8")
    manifest = _write_manifest(root)

    (root / "tests" / "canary" / "import_canary.py").unlink()

    assert "protected file is missing: tests/canary/import_canary.py" in guard.check(root, manifest)


def test_check_detects_a_renamed_protected_test(tmp_path):
    root = _root(tmp_path)
    module = root / GATE
    module.write_text("def test_kept(): pass\n", encoding="utf-8")
    manifest = _write_manifest(root)

    module.write_text("def test_renamed(): pass\n", encoding="utf-8")

    assert f"protected test is missing or renamed: {GATE}::test_kept" in guard.check(root, manifest)


def test_check_discovers_new_refusal_modules_and_new_tests(tmp_path):
    root = _root(tmp_path)
    module = root / GATE
    module.write_text("def test_kept(): pass\n", encoding="utf-8")
    manifest = _write_manifest(root)

    module.write_text("def test_kept(): pass\ndef test_added(): pass\n", encoding="utf-8")
    refusal = root / "tests" / "test_new_refusal.py"
    refusal.write_text("from dgx_monarch.refusal import RefusalClass\ndef test_guard(): pass\n", encoding="utf-8")

    errors = guard.check(root, manifest)
    assert f"unrecorded protected test: {GATE}::test_added" in errors
    assert "unrecorded protected test module: tests/test_new_refusal.py" in errors


def test_write_refuses_to_rebaseline_a_missing_hard_suite_or_canary(tmp_path):
    root = _root(tmp_path)
    (root / "tests" / "canary" / "import_canary.py").unlink()

    assert guard.main(["--root", str(root), "--write"]) == 1


def test_check_rejects_a_manifest_that_drops_a_hard_file(tmp_path):
    root = _root(tmp_path)
    (root / GATE).write_text("def test_present(): pass\n", encoding="utf-8")
    manifest = _write_manifest(root)
    manifest.write_text(
        '{"version": 2, "required_files": [], "protected_tests": {}}',
        encoding="utf-8",
    )

    assert "manifest required_files does not match the tool's REQUIRED_FILES list" in guard.check(root, manifest)


def test_a_presence_only_manifest_must_be_rebaselined(tmp_path):
    root = _root(tmp_path)
    manifest = root / "tests" / "protected_test_inventory.json"
    manifest.write_text(json.dumps({
        "version": 1, "required_files": list(guard.REQUIRED_FILES), "protected_tests": {},
    }), encoding="utf-8")

    [error] = guard.check(root, manifest)
    assert "manifest version must be 2" in error


def test_an_edited_protected_body_fails_until_the_same_diff_rebaselines_it(tmp_path, capsys):
    root = _root(tmp_path)
    module = root / GATE
    module.write_text(
        "def test_kept():\n    assert 1 + 1 == 2\n    assert 2 * 2 == 4\n"
        "def test_other():\n    assert True\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(root)
    capsys.readouterr()

    module.write_text(
        "def test_kept():\n    assert 1 + 1 == 2\n"
        "def test_other():\n    assert True\n",
        encoding="utf-8",
    )
    assert guard.check(root, manifest) == [
        f"protected test changed without an inventory re-baseline: {GATE}::test_kept",
    ]

    assert guard.main(["--root", str(root), "--manifest", str(manifest), "--write"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"changed: {GATE}::test_kept"]
    assert guard.check(root, manifest) == []


def test_write_names_every_added_and_removed_protected_test(tmp_path, capsys):
    root = _root(tmp_path)
    module = root / GATE
    module.write_text("def test_old(): pass\n", encoding="utf-8")
    manifest = _write_manifest(root)
    capsys.readouterr()

    module.write_text("def test_new(): pass\n", encoding="utf-8")
    assert guard.main(["--root", str(root), "--manifest", str(manifest), "--write"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"removed: {GATE}::test_old",
        f"added: {GATE}::test_new",
    ]


def test_removing_a_row_from_a_parametrize_table_changes_the_digest():
    template = (
        "import pytest\n"
        "ROWS = [{rows}]\n"
        "EXTRA = ROWS + [(9, 9)]\n"
        "@pytest.mark.parametrize(('a', 'b'), EXTRA, ids=['one', 'two', 'three'])\n"
        "def test_case(a, b):\n"
        "    assert a <= b\n"
    )
    full = _digest(template.format(rows="(1, 2), (3, 4)"))

    assert _digest(template.format(rows="(1, 2)")) != full
    assert _digest(template.replace("'three'", "'third'").format(rows="(1, 2), (3, 4)")) != full


def test_reached_helpers_fixtures_and_marks_count_but_unrelated_code_does_not():
    base = (
        "import pytest\n"
        "from helpers import build, unrelated\n"
        "pytestmark = pytest.mark.slow\n"
        "@pytest.fixture(autouse=True)\n"
        "def _isolated():\n"
        "    yield\n"
        "@pytest.fixture\n"
        "def store():\n"
        "    return build()\n"
        "def _check(value):\n"
        "    assert value.ok\n"
        "    assert value.size == 2\n"
        "def _unused():\n"
        "    return 1\n"
        "def test_case(store):\n"
        "    _check(store)\n"
    )
    digest = _digest(base)

    for edit in (
        ("    assert value.size == 2\n", ""),
        ("    return build()\n", "    return build(lazy=True)\n"),
        ("    yield\n", "    yield None\n"),
        ("pytest.mark.slow", "pytest.mark.skip"),
        ("from helpers import build,", "from other import build,"),
    ):
        assert _digest(base.replace(*edit)) != digest, edit

    for edit in (
        ("    return 1\n", "    return 2\n"),
        ("import build, unrelated", "import build, unrelated, added"),
        ("def test_case(store):\n", "def test_case(store):\n    '''Docstrings and # comments are not semantics.'''\n"),
        ("    _check(store)\n", "    _check(\n        store,  # reformatted\n    )\n"),
    ):
        assert _digest(base.replace(*edit)) == digest, edit


def test_a_renamed_test_keeps_its_digest_so_rename_and_edit_stay_distinct():
    source = "def test_case():\n    assert True\n"

    assert _digest(source) == _digest(source.replace("test_case", "test_other"), "test_other")


def test_class_methods_reach_their_hooks_and_the_members_they_name():
    base = (
        "class TestGroup:\n"
        "    LIMIT = 3\n"
        "    OTHER = 4\n"
        "    def setup_method(self):\n"
        "        self.items = []\n"
        "    def test_case(self):\n"
        "        assert len(self.items) < self.LIMIT\n"
    )
    digest = _digest(base, "TestGroup.test_case")

    assert _digest(base.replace("LIMIT = 3", "LIMIT = 2"), "TestGroup.test_case") != digest
    assert _digest(base.replace("self.items = []", "self.items = [1]"), "TestGroup.test_case") != digest
    assert _digest(base.replace("OTHER = 4", "OTHER = 5"), "TestGroup.test_case") == digest


# Python 3.12 parses a nested f-string format spec with a trailing empty
# literal that 3.11 and 3.13 omit, and 3.12 added an empty type_params field.
# CI runs this file under 3.11 and 3.12, so a pinned value proves the
# normalized digest is the same on both.
_PINNED_SOURCE = '''
import pytest
from helpers import build, ROWS as CASES

LIMIT = 3


def _expect(value, width):
    assert f"{value!r:>{width}}" == f"{value=}"[6:] or value


@pytest.mark.parametrize("row", CASES, ids=lambda row: f"{row:{LIMIT}.{LIMIT}}")
async def test_pinned(row, tmp_path, *args, key: int = 0, **kwargs) -> None:
    """Docstrings never change a digest."""
    match row:
        case {"k": [1, *rest]} | None:
            _expect(build(row, rest), LIMIT)
    return [item async for item in row] if key else None
'''


def test_normalized_digest_is_identical_across_supported_interpreters():
    assert guard.digest_tests(ast.parse(_PINNED_SOURCE)) == {"test_pinned": "4739bcad87b83609"}
