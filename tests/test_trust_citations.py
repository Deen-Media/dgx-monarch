"""Unit tests of tools/repin_trust_citations.py on fixtures; CI runs its --check on the real TRUST.md."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
def test_named_spans_include_decorators():
    repin = _repin_tool()
    assert ("target", 1, 3) in repin.named_spans("@decorator\ndef target():\n    pass\n")


def test_citation_range_must_not_escape_its_named_anchor():
    repin = _repin_tool()
    spans = [("target", 10, 20)]
    assert repin.range_within_named_span(spans, "target", 10, 20)
    assert not repin.range_within_named_span(spans, "target", 9, 20)
    assert not repin.range_within_named_span(spans, "target", 10, 21)


def _repin_tool():
    """The repin script as a module, without adding tools/ to the import path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "repin_trust_citations", REPO / "tools" / "repin_trust_citations.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_fails_but_write_repairs_a_moved_citation(tmp_path, monkeypatch):
    repin = _repin_tool()
    trust = tmp_path / "docs" / "TRUST.md"
    source = tmp_path / "src" / "sample.py"
    trust.parent.mkdir()
    source.parent.mkdir()
    citation = '[`sample.py:1-2`](../src/sample.py#L1-L2 "anchor:target")\n'
    document = citation * 100
    trust.write_text(document)
    source.write_text("\n\n\ndef target():\n    pass\n")
    monkeypatch.setattr(repin, "TRUST", trust)
    monkeypatch.setattr(
        repin,
        "baseline_source",
        lambda _base, relative: (
            document if relative == "../docs/TRUST.md"
            else "def target():\n    pass\n" if relative == "../src/sample.py"
            else None
        ),
    )

    assert repin.repin("baseline", check=True) == 1
    assert trust.read_text() == document
    assert repin.repin("baseline", check=False) == 0
    assert trust.read_text() == (
        '[`sample.py:4-5`](../src/sample.py#L4-L5 "anchor:target")\n' * 100
    )


def test_valid_manual_citation_edit_is_not_remapped(tmp_path, monkeypatch):
    repin = _repin_tool()
    trust = tmp_path / "docs" / "TRUST.md"
    source = tmp_path / "src" / "sample.py"
    trust.parent.mkdir()
    source.parent.mkdir()
    baseline = '[`sample.py:1-1`](../src/sample.py#L1-L1 "anchor:target")\n' * 100
    manual = '[`sample.py:2-2`](../src/sample.py#L2-L2 "anchor:target")\n' * 100
    trust.write_text(manual)
    source.write_text("def target():\n    first = 1\n    return first\n")
    monkeypatch.setattr(repin, "TRUST", trust)
    monkeypatch.setattr(
        repin,
        "baseline_source",
        lambda _base, relative: (
            baseline if relative == "../docs/TRUST.md"
            else "def target():\n    return 1\n" if relative == "../src/sample.py"
            else None
        ),
    )

    assert repin.repin("baseline", check=True) == 0
    assert repin.repin("baseline", check=False) == 0
    assert trust.read_text() == manual


def test_check_ignores_baseline_when_citation_count_changes(tmp_path, monkeypatch):
    repin = _repin_tool()
    trust = tmp_path / "docs" / "TRUST.md"
    source = tmp_path / "src" / "sample.py"
    trust.parent.mkdir()
    source.parent.mkdir()
    citation = '[`sample.py:4-5`](../src/sample.py#L4-L5 "anchor:target")\n'
    trust.write_text(citation * 101)
    source.write_text("\n\n\ndef target():\n    pass\n")
    monkeypatch.setattr(repin, "TRUST", trust)
    monkeypatch.setattr(
        repin,
        "baseline_source",
        lambda *_args: (_ for _ in ()).throw(AssertionError("check read a baseline")),
    )

    assert repin.repin("shallow-or-different-baseline", check=True) == 0
    monkeypatch.setattr(repin, "merge_base", lambda: (_ for _ in ()).throw(AssertionError("check read git")))
    monkeypatch.setattr(repin.sys, "argv", ["repin_trust_citations.py", "--check"])
    assert repin.main() == 0


def test_a_narrower_citation_keeps_both_insets_when_its_anchor_grows():
    """A range that crosses an inserted line has to grow with it.

    Keeping only the distance from the anchor start loses a line every time the
    span grows below the range, which left three gate_ceremony citations one line
    short of what they meant until the 2026-09-09 fix.
    """
    repin = _repin_tool()

    grown, clamped = repin.remapped((240, 245), (236, 247), (236, 248))

    assert grown == (240, 246)
    assert clamped is False


def test_a_whole_span_citation_still_means_the_whole_definition():
    repin = _repin_tool()

    assert repin.remapped((236, 247), (236, 247), (240, 260)) == ((240, 260), False)
    # No baseline span to read the range against: the citation covers the span.
    assert repin.remapped((240, 245), None, (236, 248)) == ((236, 248), False)


def test_a_range_the_new_anchor_cannot_hold_reports_the_cut():
    """A range the new anchor cannot hold comes back flagged as clamped, so the
    caller reports the cut instead of writing a silently shortened range."""
    repin = _repin_tool()

    (start, end), clamped = repin.remapped((240, 260), (236, 300), (100, 120))

    assert clamped is True
    assert (start, end) == (104, 104)


def test_citation_ranges_group_by_file_and_anchor_in_document_order():
    """The baseline document is the source of coordinates, so its groups pair
    with the working copy's by position."""
    repin = _repin_tool()
    document = (
        '[`a.py:1-2`](../src/dgx_monarch/a.py#L1-L2 "anchor:one")\n'
        '[`b.py:5-6`](../src/dgx_monarch/b.py#L5-L6 "anchor:two")\n'
        '[`a.py:3-4`](../src/dgx_monarch/a.py#L3-L4 "anchor:one")\n'
    )

    grouped = repin.citation_ranges(document)

    assert grouped[("../src/dgx_monarch/a.py", "one")] == [(1, 2), (3, 4)]
    assert grouped[("../src/dgx_monarch/b.py", "two")] == [(5, 6)]


def test_baseline_follows_main_or_master_without_remote(tmp_path, monkeypatch):
    import subprocess

    repin = _repin_tool()
    monkeypatch.setattr(repin, "REPO", tmp_path)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-b", "main")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test",
        "commit", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-b", "feature")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test",
        "commit", "--allow-empty", "-m", "feature")
    assert repin.merge_base() == base
    git("branch", "-m", "main", "master")
    assert repin.merge_base() == base
    assert repin.merge_base("HEAD") == git("rev-parse", "HEAD")
    git("branch", "-D", "master")
    assert repin.merge_base() == git("rev-parse", "HEAD")


def test_baseline_remote_head_resolves_ambiguous_local_branches(tmp_path, monkeypatch):
    import subprocess

    import pytest

    repin = _repin_tool()
    monkeypatch.setattr(repin, "REPO", tmp_path)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-b", "main")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test",
        "commit", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("branch", "master")
    with pytest.raises(SystemExit, match="pass --base"):
        repin.merge_base()
    git("update-ref", "refs/remotes/origin/main", base)
    git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    assert repin.merge_base() == base
