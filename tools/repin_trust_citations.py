"""Update docs/TRUST.md citation ranges after source lines move.

``--check`` validates Python citations against the current tree: each needs an
``anchor:<name>`` title, matching displayed and linked ranges, and a range
inside the named definition, class, or module-level assignment.

Rewrite mode pairs current and baseline citations by file, anchor, and order
when group sizes match. It preserves each range's inset from both ends of the
anchor. Whole-anchor citations remain whole; narrower ranges grow or shrink
with the anchor. Paired ranges edited since the baseline are left unchanged.

Unpaired ranges remain unchanged while inside their anchor and are reported
if the anchor moved. An unpaired range outside its anchor is relocated using
its current coordinates as the baseline. Any problem, including a range the
new anchor cannot hold, prevents the write.

    python tools/repin_trust_citations.py --check
    python tools/repin_trust_citations.py --base <rev>

The default baseline is the merge base with the locally known default branch.
Without a remote HEAD, a sole local main or master branch is used; otherwise
pass --base to select a baseline. A checkout with neither uses HEAD.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRUST = REPO / "docs" / "TRUST.md"

PYTHON_LINK = re.compile(
    r"\[(?P<label>[^]]+)\]\("
    r"(?P<relative>\.\./(?:src|tests)/[^)#]+\.py)"
    r"#L(?P<start>\d+)-L(?P<end>\d+)"
    r'(?: "anchor:(?P<anchor>[A-Za-z_][A-Za-z0-9_]*)")?'
    r"\)"
)
PYTHON_LABEL = re.compile(r"`[^`]+\.py:(?P<start>\d+)-(?P<end>\d+)`")
CITATION = re.compile(
    r"\[`(?P<label_file>[^`]+\.py):(?P<label_start>\d+)-(?P<label_end>\d+)`\]\("
    r"(?P<relative>\.\./(?:src|tests)/[^)#]+\.py)"
    r"#L(?P<start>\d+)-L(?P<end>\d+)"
    r' "anchor:(?P<anchor>[A-Za-z_][A-Za-z0-9_]*)"\)'
)


def live_citation_problems(text: str) -> list[str]:
    """Return every TRUST citation contract violation, read against the current tree."""
    citations = list(PYTHON_LINK.finditer(text))
    problems: list[str] = []
    if len(citations) < 100:
        problems.append(f"TRUST.md has only {len(citations)} Python citations (expected at least 100)")
    parsed: dict[Path, tuple[int, list[tuple[str, int, int]]]] = {}
    for match in citations:
        relative = match.group("relative")
        start, end = int(match.group("start")), int(match.group("end"))
        anchor = match.group("anchor")
        displayed = PYTHON_LABEL.fullmatch(match.group("label"))
        if anchor is None:
            problems.append(f"{relative}#L{start}-L{end}: citation needs an anchor title")
            continue
        if displayed is None:
            problems.append(f"{relative}#L{start}-L{end}: displayed label needs its line range")
        elif (int(displayed.group("start")), int(displayed.group("end"))) != (start, end):
            problems.append(f"{relative}#L{start}-L{end}: displayed range does not match href")
        path = (TRUST.parent / relative).resolve()
        if not path.is_file():
            problems.append(f"{relative}: cited file does not exist")
            continue
        if path not in parsed:
            source = path.read_text()
            parsed[path] = (len(source.splitlines()), named_spans(source))
        line_count, spans = parsed[path]
        if not 1 <= start <= end <= line_count:
            problems.append(f"{relative}#L{start}-L{end}: range exceeds 1-{line_count}")
        elif not range_within_named_span(spans, anchor, start, end):
            problems.append(
                f"{relative}#L{start}-L{end}: named anchor {anchor!r} does not contain the cited range")
    return problems


def named_spans(source: str) -> list[tuple[str, int, int]]:
    """Return (name, start, end) spans for defs, classes and module assignments."""
    spans: list[tuple[str, int, int]] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorator_lines = [decorator.lineno for decorator in node.decorator_list]
            start = min([node.lineno, *decorator_lines])
            spans.append((node.name, start, node.end_lineno or node.lineno))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                spans.append((target.id, node.lineno, node.end_lineno or node.lineno))
    return spans


def range_within_named_span(
    spans: list[tuple[str, int, int]], anchor: str, start: int, end: int
) -> bool:
    return any(name == anchor and low <= start <= end <= high
               for name, low, high in spans)


def baseline_source(rev: str, relative: str) -> str | None:
    path = relative.replace("../", "", 1)
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return result.stdout if result.returncode == 0 else None


def pick_span(spans: list[tuple[str, int, int]], anchor: str,
              start: int, end: int) -> tuple[int, int] | None:
    candidates = [(a, b) for name, a, b in spans if name == anchor]
    if not candidates:
        return None
    containing = [span for span in candidates if span[0] <= start <= end <= span[1]]
    pool = containing or candidates
    return min(pool, key=lambda span: abs(span[0] - start))


def citation_ranges(text: str) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Every citation range in one TRUST.md, grouped by file and anchor in document order."""
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for match in CITATION.finditer(text):
        key = (match.group("relative"), match.group("anchor"))
        grouped.setdefault(key, []).append(
            (int(match.group("start")), int(match.group("end"))))
    return grouped


def remapped(source: tuple[int, int], old_span: tuple[int, int] | None,
             new_span: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    """Where a baseline range lands in the current anchor, and whether it was cut.

    A range equal to its whole baseline span means "this definition", so it
    becomes the whole current span, as does any range with no baseline span.
    Any narrower range keeps both insets: the lines it left at the top of the
    span, and the lines it left at the bottom. Keeping only the top inset leaves
    a range that crosses an inserted line one line short (seen 2026-09-09), and
    moves a range one line early when the line lands inside the span above it.
    Both insets err in a way the result cannot show: each line inserted inside
    the span but outside the range widens the range by one line, and each such
    line deleted drops one cited line. Nothing reports that drop unless the
    range would hold no line, in which case a clamp moves an end.

    The second value says a clamp moved an end. The caller reports that range
    and does not write it: only an author can say what an anchor that can no
    longer hold the range should cite.
    """
    start, end = source
    if old_span is None or (start, end) == old_span:
        return new_span, False
    wanted_start = new_span[0] + (start - old_span[0])
    wanted_end = new_span[1] - (old_span[1] - end)
    new_start = min(max(wanted_start, new_span[0]), new_span[1])
    new_end = min(max(wanted_end, new_start), new_span[1])
    return (new_start, new_end), (new_start, new_end) != (wanted_start, wanted_end)


def default_branch() -> str:
    remote = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if remote.returncode == 0:
        ref = remote.stdout.strip()
        exists = subprocess.run(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        if exists.returncode == 0:
            return ref
    branches = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", "refs/heads/main", "refs/heads/master"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    refs = [ref for ref in branches.stdout.splitlines() if ref in {"refs/heads/main", "refs/heads/master"}]
    if len(refs) > 1:
        raise SystemExit("both main and master exist without a known remote default; pass --base")
    return refs[0] if refs else "HEAD"


def merge_base(rev: str | None = None) -> str:
    result = subprocess.run(
        ["git", "merge-base", "HEAD", rev or default_branch()],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or "HEAD"


def repin(base: str, check: bool) -> int:
    text = TRUST.read_text()
    if check:
        # Read no baseline: a shallow CI clone can lack the merge base, and a
        # valid document must pass even when its citation count or order
        # changed while an anchor moved.
        live_problems = live_citation_problems(text)
        for line in live_problems:
            print(f"PROBLEM {line}", file=sys.stderr)
        if not live_problems:
            print("every TRUST.md citation is already pinned to its anchor")
        return 1 if live_problems else 0

    sources: dict[str, str] = {}
    baselines: dict[str, str | None] = {}
    changes: list[str] = []
    remap_problems: list[str] = []
    # Source coordinates come from the baseline's TRUST.md, which writing the
    # working copy cannot move, so a second run does not shift a shifted range
    # again. Citations of one file and anchor pair in document order, and only
    # when the group has the same size in both documents.
    baseline_doc = baseline_source(base, "../docs/TRUST.md")
    baseline_ranges = citation_ranges(baseline_doc) if baseline_doc is not None else {}
    current_ranges = citation_ranges(text)
    seen: dict[tuple[str, str], int] = {}

    def replace(match: re.Match[str]) -> str:
        relative = match.group("relative")
        anchor = match.group("anchor")
        start, end = int(match.group("start")), int(match.group("end"))
        key = (relative, anchor)
        index = seen.get(key, 0)
        seen[key] = index + 1
        path = (TRUST.parent / relative).resolve()
        if not path.is_file():
            remap_problems.append(f"{relative}: cited file does not exist")
            return match.group(0)
        if relative not in sources:
            sources[relative] = path.read_text()
            baselines[relative] = baseline_source(base, relative)
        paired = len(baseline_ranges.get(key, ())) == len(current_ranges.get(key, ()))
        baseline_range = (
            baseline_ranges[key][index]
            if paired and key in baseline_ranges else None
        )
        # A paired range that differs from its baseline range changed after the
        # baseline, by hand or through an earlier run: keep it. The tool rewrites
        # a paired range only while it equals its baseline range, and an unpaired
        # one only once it has left its anchor, from its current range.
        if baseline_range is not None and (start, end) != baseline_range:
            return match.group(0)
        source = baseline_range or (start, end)
        new_spans = named_spans(sources[relative])
        new_span = pick_span(new_spans, anchor, *source)
        if new_span is None:
            remap_problems.append(f"{relative}: anchor {anchor!r} no longer exists")
            return match.group(0)
        old_source = baselines[relative]
        old_span = (
            pick_span(named_spans(old_source), anchor, *source)
            if old_source is not None else None
        )
        if not paired and new_span[0] <= start <= end <= new_span[1]:
            # No baseline range pairs with this citation: its group changed size,
            # or the baseline has no TRUST.md. It still sits inside its anchor,
            # so keep it as written, and report it if the anchor moved.
            if old_span is not None and old_span != new_span:
                remap_problems.append(
                    f"{relative}: {anchor!r} moved from {old_span[0]}-{old_span[1]} to "
                    f"{new_span[0]}-{new_span[1]} and this citation has no baseline "
                    f"pair, so {start}-{end} was left alone; check it by hand")
            return match.group(0)
        (new_start, new_end), clamped = remapped(source, old_span, new_span)
        if clamped:
            remap_problems.append(
                f"{relative}: {anchor!r} can no longer hold {source[0]}-{source[1]}, "
                f"which lands outside {new_span[0]}-{new_span[1]}; the range was "
                "not rewritten")
            return match.group(0)
        if (new_start, new_end) == (start, end):
            return match.group(0)
        changes.append(
            f"{relative} {anchor}: {start}-{end} -> {new_start}-{new_end}"
        )
        label_file = match.group("label_file")
        return (
            f"[`{label_file}:{new_start}-{new_end}`]"
            f"({relative}#L{new_start}-L{new_end} \"anchor:{anchor}\")"
        )

    updated = CITATION.sub(replace, text)
    live_problems = live_citation_problems(updated)
    for line in changes:
        print(f"repin {line}")
    for line in [*live_problems, *remap_problems]:
        print(f"PROBLEM {line}", file=sys.stderr)
    if changes and not check and not live_problems and not remap_problems:
        TRUST.write_text(updated)
        print(f"rewrote {len(changes)} citation(s) in {TRUST}")
    elif not changes:
        print("no TRUST.md citation was rewritten")
    return 1 if live_problems or remap_problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None, help="baseline revision (default: merge-base with the locally known default branch)")
    parser.add_argument("--check", action="store_true",
                        help="check every Python citation in TRUST.md against this tree; change nothing")
    args = parser.parse_args()
    if args.check:
        return repin("", True)
    return repin(args.base or merge_base(), False)


if __name__ == "__main__":
    raise SystemExit(main())
