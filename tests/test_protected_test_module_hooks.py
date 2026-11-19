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
    "skip assigned to another name": "\n\n_off = pytest.skip\n_off('module off', allow_module_level=True)\n",
    "helper that skips, called at import": (
        "\n\ndef _needs():\n    _check()\n\n\ndef _check():\n    pytest.importorskip('no_such_module')\n\n\n_needs()\n"),
    "skip in a class body": "\n\nclass TestLate:\n    pytest.skip('module off', allow_module_level=True)\n",
    "skipping helper as a decorator on a helper": (
        "\n\ndef _needs(f):\n    pytest.importorskip('no_such_module')\n    return f\n\n\n"
        "@_needs\ndef _helper():\n    pass\n"),
    "skipping helper as a decorator on a class": (
        "\n\ndef _needs(cls):\n    pytest.importorskip('no_such_module')\n    return cls\n\n\n"
        "@_needs\nclass _Holder:\n    pass\n"),
    "skipping helper in an argument default": (
        "\n\ndef _needs():\n    pytest.importorskip('no_such_module')\n\n\ndef _helper(value=_needs()):\n    return value\n"),
    "skipping helper in a keyword-only default": (
        "\n\ndef _needs():\n    pytest.importorskip('no_such_module')\n\n\n"
        "def _helper(*, value=_needs()):\n    return value\n"),
    "skip under a not-main guard": (
        "\n\nif __name__ != '__main__':\n    pytest.skip('module off', allow_module_level=True)\n"),
    "skip in a main guard's else": (
        "\n\nif __name__ == '__main__':\n    pass\nelse:\n    pytest.skip('module off', allow_module_level=True)\n"),
    "skipping helper in a class's bases": (
        "\n\ndef _needs():\n    pytest.importorskip('no_such_module')\n    return object\n\n\n"
        "class _Holder(_needs()):\n    pass\n"),
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
    reversed_guard = source + "\nif '__main__' == __name__:\n    raise SystemExit(0)\n"
    assert guard.digest_tests(guard.ast.parse(reversed_guard)) == baseline
    # A helper called at import that skips nothing reaches no test that does not name it.
    helper = source + "\ndef _build():\n    return [1, 2]\n\n\nROWS = _build()\n"
    assert guard.digest_tests(guard.ast.parse(helper)) == baseline
    held = source + "\nclass Holder:\n    rows = [1, 2]\n"
    assert guard.digest_tests(guard.ast.parse(held)) == baseline
    # A skip mark skips one test, never the module, by any name and wherever it is written.
    for marked in (
        "\nclass Holder:\n    ROWS = [pytest.param(1, marks=pytest.mark.skip(reason='x'))]\n",
        "\n_mark = pytest.mark.skip\n_mark(reason='x')\n",
        "\npytest.mark.skip(reason='x')\n",
        "\nfrom pytest import mark as _marks\n\n_marks.skip(reason='x')\n",
    ):
        assert guard.digest_tests(guard.ast.parse(source + marked)) == baseline, marked
    # A decorator whose wrapper skips skips the one test it wraps, when that test runs.
    for gated in (
        "\nimport functools\n\n\ndef _requires_x(fn):\n    @functools.wraps(fn)\n    def wrapper(*a, **k):\n"
        "        pytest.skip('no x')\n    return wrapper\n\n\n@_requires_x\ndef test_gated():\n    pass\n",
        "\ndef _needs(name):\n    def decorate(fn):\n        def wrapper(*a, **k):\n            pytest.skip(name)\n"
        "        return wrapper\n    return decorate\n\n\n@_needs('x')\ndef test_gated():\n    pass\n",
    ):
        digests = guard.digest_tests(guard.ast.parse(source + gated))
        assert {name: digests[name] for name in baseline} == baseline, gated
