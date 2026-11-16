"""A module-level hook or skip that turns off every protected test changes their digests.

pytest runs ``setup_module``, ``setUpModule`` and ``setup_function`` as autouse
fixtures, reads ``__test__ = False`` as "collect nothing here", and a
``pytest.skip`` or ``pytest.importorskip`` at module level skips the whole
module however its result is named. Each of these silences every test of a
protected module with no test edited, so the guard's ``--check`` must fail on
it. The edits go into copies of two real protected modules, checked against
an inventory written from the unedited copies.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "check_protected_tests.py"
SPEC = importlib.util.spec_from_file_location("check_protected_tests", TOOL)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)

MODULES = ("tests/test_capacity_quote.py", "tests/test_vae_decode_worker.py")
EDITS = {
    "setup_module": "\n\ndef setup_module(module):\n    pytest.skip('module off')\n",
    "setUpModule": "\n\ndef setUpModule():\n    pytest.skip('module off')\n",
    "setup_function": "\n\ndef setup_function(function):\n    pytest.skip('off')\n",
    "teardown_module": "\n\ndef teardown_module(module):\n    raise SystemExit(0)\n",
    "__test__": "\n\n__test__ = False\n",
    "importorskip": "\n\n_absent = pytest.importorskip('no_such_module')\n",
    "bare skip": "\n\npytest.skip('module off', allow_module_level=True)\n",
    "guarded skip": "\n\nif True:\n    pytest.skip('module off', allow_module_level=True)\n",
    "raised SkipTest": "\n\nimport unittest\n\nraise unittest.SkipTest('module off')\n",
    "skip imported by another name": (
        "\n\nfrom pytest import skip as _off\n\n_off('module off', allow_module_level=True)\n"),
}


def _root(tmp_path: Path) -> Path:
    """A checkout with the real required files and protected modules, and its inventory."""
    for rel in {*guard.REQUIRED_FILES, *guard.SUITE_MODULES, *guard.CAPACITY_MODULES}:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, target)
    manifest = tmp_path / "tests" / "protected_test_inventory.json"
    assert guard.main(["--root", str(tmp_path), "--manifest", str(manifest), "--write"]) == 0
    assert guard.check(tmp_path, manifest) == []
    return tmp_path


@pytest.mark.parametrize("edit", sorted(EDITS))
def test_a_module_level_hook_or_skip_fails_the_check_for_every_test_it_silences(tmp_path, edit):
    root = _root(tmp_path)
    manifest = root / "tests" / "protected_test_inventory.json"
    before = {rel: guard.digest_tests(guard._parse(root, rel)) for rel in MODULES}
    for rel in MODULES:
        path = root / rel
        path.write_text(path.read_text(encoding="utf-8") + EDITS[edit], encoding="utf-8")
    errors = guard.check(root, manifest)
    for rel in MODULES:
        after = guard.digest_tests(guard._parse(root, rel))
        unchanged = sorted(name for name, digest in after.items() if before[rel].get(name) == digest)
        assert not unchanged, (edit, rel, unchanged)
        assert any(rel in error for error in errors), (edit, rel, errors)


def test_a_module_level_statement_that_skips_nothing_moves_no_digest():
    source = "import pytest\nLIMIT = 3\n\ndef test_case():\n    assert LIMIT\n\ndef test_other():\n    pass\n"
    tree = guard.ast.parse(source)
    baseline = guard.digest_tests(tree)
    edited = guard.digest_tests(guard.ast.parse(source + "\nprint('collected')\n"))
    assert edited == baseline
    # A script entry point runs only when the file runs as a script, never when pytest imports it.
    script = source + "\nif __name__ == '__main__':\n    raise SystemExit(0)\n"
    assert guard.digest_tests(guard.ast.parse(script)) == baseline
