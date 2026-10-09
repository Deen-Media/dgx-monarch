"""Mechanize the single-home doctrine rule (DESIGN.md §7).

Four things are checked here, and nowhere else:

1. Version-anchored prose stays out of evergreen surfaces. A version literal
   used as the subject of a standing claim goes stale at the next bump; a
   dated record does not, so a claim whose own sentence carries a date is
   exempt and the rest is a failure. Sentence scope, not block scope: one
   dated bullet must not exempt the standing claims around it.
2. SECURITY.md's fabric-trust claim points at live code. The two cited call
   sites are pinned by AST span and by the value they pass, so neither moving
   the function nor adding real CA plumbing can leave the claim standing.
3. Every timeout docs/TROUBLESHOOTING.md quotes is formatted from the live
   constant, so changing the constant breaks the doc assertion instead of
   leaving the doc wrong, and the ladder those constants form stays ordered.
   The RDMA return sections of DESIGN.md §5.6 and docs/TROUBLESHOOTING.md #29
   keep the token, ACK and manual-recycle contract.
4. Each doctrine fact has one home. Code counts as a surface: a fact restated
   in a src comment or an operator string is drift too, and so is one restated
   in a benchmark harness.
"""

from __future__ import annotations

import ast
import functools
import io
import re
import tokenize
from pathlib import Path

import pytest

import dgx_monarch
from dgx_monarch import capacity_agreement, mesh_liveness, mesh_runtime, rdma_ownership, transfer

REPO = Path(__file__).resolve().parents[1]

# Part 1: version-anchored prose

# Three dotted parts and no fourth: an IP address is not a version literal.
_SEMVER = r"(?<![\d.])\d+\.\d+\.\d+(?!\.?\d)"

# The pinned dependencies whose version literals go stale in prose. Naming
# only torchmonarch would hide every comfy-aimdo compat note from the first
# PIN_ANCHOR alternative.
_PINNED = r"(?:torchmonarch|comfy-aimdo)"

# A qualifier run between the version and the noun it anchors: "0.6.0 pin",
# "0.6.0 acceptance pin", "0.6.0 queue-dispatch default". Hyphenated words
# count, which `\w+` could not express.
_QUALIFIER = r"(?:[\w-]+\s+){0,2}"

# One alternative per way a pinned release becomes the subject of a claim; the
# last needs no version literal.
PIN_ANCHOR = re.compile(
    rf"{_PINNED}\s+(?!>=|>|version\b)v?{_SEMVER}"       # "torchmonarch 0.6.0 forwards ..."
    rf"|{_SEMVER}\s+and\s+{_SEMVER}\s+pins?\b"          # "the 0.5.0 and 0.6.0 pins"
    rf"|{_SEMVER}\s+{_QUALIFIER}pins?\b"                # "the 0.6.0 pin", "0.6.0 acceptance pin"
    rf"|(?:under|on|at|since|per|against)\s+the\s+{_SEMVER}"  # "against the 0.6.0"
    rf"|{_SEMVER}\s+{_QUALIFIER}(?:introduced|added|dropped|removed|took|takes|"
    rf"killed|forwards|changed|renamed|broke)\b"        # "0.4.10 dropped nvml_pressure"
    rf"|the\s+{_SEMVER}\s+{_QUALIFIER}default\b"        # "the 0.6.0 queue-dispatch default"
    rf"|{_SEMVER}\s*'s\b"                               # "0.5.0's broken link"
    rf"|\b(?:any|every)\s+pin\s+to\s+date\b",           # pin-anchored epistemics, no semver
    re.IGNORECASE,
)

# A minimum requirement stays true when the pin moves. The first alternative's
# lookahead already skips ">= x.y.z"; this also excuses a floor the lookahead
# does not skip, such as "0.6.0 or newer".
VERSION_FLOOR = re.compile(rf"(?:>=|>)\s*v?{_SEMVER}|{_SEMVER}\s+or\s+(?:newer|later)")

# A date in a claim's own sentence marks it as a dated record.
DATED_RECORD = re.compile(r"\d{4}-\d{2}-\d{2}")

# Sentence boundary inside a flattened unit: terminator, space, then something
# that opens a sentence. Version literals carry no space after their dots, so
# "0.6.0" never splits here.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[`*_\"'#|-])")

_WHITESPACE = re.compile(r"\s+")

# Units that must name a release. Each entry names its enclosing unit by a
# marker substring.
ALLOWLIST: tuple[tuple[str, str], ...] = (
    # docs/TROUBLESHOOTING.md #57 diagnoses a console line that only one
    # release emits: the version is the symptom.
    ("docs/TROUBLESHOOTING.md", "## 57."),
    # The absorb-fire soak in docs/TROUBLESHOOTING.md #60 compares the pre-bump
    # baseline with the current pin; the question needs both numbers.
    ("docs/TROUBLESHOOTING.md", "## 60."),
    # The absorb_fire.py module docstring compares two named releases; that
    # comparison is the reason the module exists.
    ("src/dgx_monarch/absorb_fire.py", "was built against"),
    # The surface canary exists because that release made concurrent_endpoint
    # load-bearing, so the comment must name the release.
    ("src/dgx_monarch/monarch_surface.py", "a signature or removal drift here"),
    # Same canary: the dispatch-default flip it catches.
    ("src/dgx_monarch/monarch_surface.py", "a default flip (torchmonarch"),
    # A string-taxonomy audit against a named wheel: the wheel is the evidence.
    ("src/dgx_monarch/mesh_helpers.py", "audited against the"),
    # The named release records the currently verified submodule rewire boundary.
    ("src/dgx_monarch/actor/comfy_dynamic.py", "Re-point comfy_aimdo's submodules"),
)

# A fixed cap makes allowlist growth an explicit review decision.
MAX_ALLOWLIST = 12

# Exclude historical records and machine pin snapshots. Test names also retain
# regression history; scan benchmark harness prose for stale runtime claims.
DATED_RECORD_FILES = ("VALIDATION.md",)


def _flatten(text: str) -> str:
    """One space everywhere, so a claim that wraps across lines (or across a
    comment marker) matches the same as an inline one."""
    return _WHITESPACE.sub(" ", text).strip()


def _markdown_units(path: Path):
    """(line, text) per unit. The unit is the blank-line-delimited block with
    its innermost enclosing heading prepended: small enough that a date in one
    bullet cannot exempt a whole section, and still carrying the heading an
    allowlist entry names."""
    lines = path.read_text().splitlines()
    heading = ""
    block: list[str] = []
    block_start = 1
    for index, line in enumerate(lines, start=1):
        if re.match(r"^#{1,6} ", line):
            if block:
                yield block_start, _flatten(heading + " " + " ".join(block))
                block = []
            heading = line
            yield index, _flatten(line)
            continue
        if not line.strip():
            if block:
                yield block_start, _flatten(heading + " " + " ".join(block))
                block = []
            continue
        if not block:
            block_start = index
        block.append(line)
    if block:
        yield block_start, _flatten(heading + " " + " ".join(block))


def _toml_units(path: Path):
    """One unit per contiguous comment block and one per other non-blank line.
    The file's acknowledgement comment wraps across three lines, so a
    line-scoped scan could not see a claim written that way."""
    block: list[str] = []
    block_start = 1
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if line.lstrip().startswith("#"):
            if not block:
                block_start = index
            block.append(line.lstrip().lstrip("#").strip())
            continue
        if block:
            yield block_start, _flatten(" ".join(block))
            block = []
        if line.strip():
            yield index, _flatten(line)
    if block:
        yield block_start, _flatten(" ".join(block))


def _python_units(path: Path):
    """Contiguous comment blocks plus every string constant. Scanning every
    string constant, not only docstrings, covers operator text: the doctor
    rows, the CLI refusals and the rendered cluster.toml template."""
    source = path.read_text()
    previous = -10
    block: list[str] = []
    block_start = 1
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line = token.start[0]
        if line != previous + 1 and block:
            yield block_start, _flatten(" ".join(block))
            block = []
        if not block:
            block_start = line
        block.append(token.string.lstrip("#").strip())
        previous = line
    if block:
        yield block_start, _flatten(" ".join(block))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, _flatten(node.value)


def _prose_surfaces() -> list[Path]:
    paths = [
        path
        for path in sorted((REPO / "docs").rglob("*.md"))
        if path.name not in DATED_RECORD_FILES
    ]
    paths.append(REPO / "README.md")
    paths.append(REPO / "SECURITY.md")
    paths.append(REPO / "CONTRIBUTING.md")
    paths.append(REPO / "cluster.example.toml")
    paths.extend(sorted((REPO / "skills").rglob("*.md")))
    return [path for path in paths if path.is_file()]


def _scanned_surfaces() -> list[Path]:
    code = sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "benchmark").rglob("*.py"))
    return _prose_surfaces() + code


def _units(path: Path):
    if path.suffix == ".md":
        return _markdown_units(path)
    if path.suffix == ".toml":
        return _toml_units(path)
    return _python_units(path)


def _dated_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences that carry a date of their own."""
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in [*_SENTENCE_BREAK.finditer(text), None]:
        end = boundary.start() if boundary is not None else len(text)
        if DATED_RECORD.search(text, start, end):
            spans.append((start, end))
        if boundary is not None:
            start = boundary.end()
    return spans


def _match_line(path: Path, unit_line: int, matched: str) -> int:
    """The line the match actually sits on. Units span several lines, so the
    unit's first line can be many lines off; report where a reader must look."""
    version = re.search(_SEMVER, matched)
    needle = version.group(0) if version else matched
    lines = path.read_text().splitlines()
    for offset in range(unit_line - 1, len(lines)):
        if needle in lines[offset]:
            return offset + 1
    return unit_line


@functools.cache
def _raw_findings() -> tuple[tuple[str, int, str, str], ...]:
    """(relative path, line, matched text, unit text), before the allowlist."""
    findings: list[tuple[str, int, str, str]] = []
    for path in _scanned_surfaces():
        relative = path.relative_to(REPO).as_posix()
        for line, text in _units(path):
            dated = _dated_spans(text)
            floors = [(m.start(), m.end()) for m in VERSION_FLOOR.finditer(text)]
            for match in PIN_ANCHOR.finditer(text):
                if any(s <= match.start() and match.end() <= e for s, e in dated):
                    continue
                if any(s < match.end() and match.start() < e for s, e in floors):
                    continue
                findings.append(
                    (relative, _match_line(path, line, match.group(0)), match.group(0), text)
                )
    return tuple(findings)


def _excused_by(finding: tuple[str, int, str, str]) -> list[tuple[str, str]]:
    relative, _line, _matched, text = finding
    return [
        entry for entry in ALLOWLIST if entry[0] == relative and entry[1] in text
    ]


def test_no_pin_anchored_prose_outside_dated_records():
    unexcused = [
        f"{relative}:{line}: {matched!r} in {text[:110]!r}"
        for relative, line, matched, text in _raw_findings()
        if not _excused_by((relative, line, matched, text))
    ]
    assert not unexcused, (
        "version-anchored prose in an evergreen surface; state the fact without "
        "the pin, or move it to a dated record:\n" + "\n".join(unexcused)
    )


def test_the_allowlist_stays_small_and_every_entry_still_matches():
    assert len(ALLOWLIST) <= MAX_ALLOWLIST, (
        f"{len(ALLOWLIST)} allowlist entries exceeds MAX_ALLOWLIST={MAX_ALLOWLIST}; "
        "an exemption per bump is the drift this test exists to stop"
    )
    findings = _raw_findings()
    dead = [
        entry
        for entry in ALLOWLIST
        if not any(entry in _excused_by(finding) for finding in findings)
    ]
    assert not dead, (
        "allowlist entries that excuse nothing (the prose they covered is gone, "
        f"or the scanner never saw it): {dead}"
    )


# Part 2: SECURITY.md's claim is pinned to live code

# Same grammar and containment rule as docs/TRUST.md, except the path is
# resolved against the citing document's own directory, so a root-level file
# cites src/... without climbing out of docs/.
CITATION = re.compile(
    r"\[(?P<label>[^]]+)\]\("
    r"(?P<relative>(?:src|tests)/[^)#]+\.py)"
    r"#L(?P<start>\d+)-L(?P<end>\d+)"
    r' "anchor:(?P<anchor>[A-Za-z_][A-Za-z0-9_]*)"\)'
)
CITATION_LABEL = re.compile(r"`[^`]+\.py:(?P<start>\d+)-(?P<end>\d+)`")

SECURITY = REPO / "SECURITY.md"

# The claim is "dgx-monarch never passes anything but trust_all_connections".
# These are the only two places in the tree that pass `ca` at all, so the claim
# is exactly as true as these two spans are.
REQUIRED_ANCHORS: tuple[tuple[str, str], ...] = (
    ("src/dgx_monarch/mesh_runtime.py", "attach_once"),
    ("src/dgx_monarch/cli/worker_loop.py", "main"),
)


def _named_spans(tree: ast.AST) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
            spans.append((node.name, start, node.end_lineno or node.lineno))
    return spans


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path}: no function named {name}")


def test_security_md_cites_live_named_anchors():
    text = SECURITY.read_text()
    citations = list(CITATION.finditer(text))
    cited = {(m.group("relative"), m.group("anchor")) for m in citations}
    assert set(REQUIRED_ANCHORS) <= cited, (
        "SECURITY.md must cite the attach call sites its claim rests on; "
        f"missing {sorted(set(REQUIRED_ANCHORS) - cited)}"
    )

    failures: list[str] = []
    for match in citations:
        relative = match.group("relative")
        start, end = int(match.group("start")), int(match.group("end"))
        anchor = match.group("anchor")
        label = CITATION_LABEL.fullmatch(match.group("label"))
        if label is None:
            failures.append(f"{relative}#L{start}-L{end}: label must carry its line range")
        elif (int(label.group("start")), int(label.group("end"))) != (start, end):
            failures.append(
                f"{relative}#L{start}-L{end}: displayed range "
                f"{label.group('start')}-{label.group('end')} does not match the href"
            )
        path = (SECURITY.parent / relative).resolve()
        if not path.is_file():
            failures.append(f"{relative}: cited file does not exist")
            continue
        source = path.read_text()
        if not 1 <= start <= end <= len(source.splitlines()):
            failures.append(f"{relative}#L{start}-L{end}: range exceeds the file")
            continue
        if not any(
            name == anchor and node_start <= start <= end <= node_end
            for name, node_start, node_end in _named_spans(ast.parse(source))
        ):
            failures.append(
                f"{relative}#L{start}-L{end}: named anchor {anchor!r} "
                "does not contain the cited range"
            )
    assert not failures, "stale SECURITY.md citations:\n" + "\n".join(failures)


def test_the_attach_path_still_passes_only_trust_all_connections():
    # A span pin proves SECURITY.md points at live code; only reading the
    # argument proves that code still does what SECURITY.md claims. Every ca=
    # in the span is read, and a non-constant one is a failure rather than a
    # silent drop: real CA plumbing arrives alongside the constant, not in
    # place of it, and a set built from constants alone would not see it.
    for relative, name in REQUIRED_ANCHORS:
        node = _function(REPO / relative, name)
        passed = [
            keyword.value.value
            if isinstance(keyword.value, ast.Constant)
            else ast.unparse(keyword.value)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            for keyword in call.keywords
            if keyword.arg == "ca"
        ]
        assert passed and set(passed) == {"trust_all_connections"}, (
            f"{relative}:{name} passes ca={sorted(set(passed))}; SECURITY.md's fabric-trust "
            "claim describes a surface that only accepts trust_all_connections"
        )


# Part 3: every quoted timeout is formatted from its live constant

TROUBLESHOOTING = REPO / "docs" / "TROUBLESHOOTING.md"
DESIGN = REPO / "docs" / "DESIGN.md"

# (doc, quoted sentence fragment, live value, formatter). Change a constant and
# the doc assertion fails, so no doc keeps quoting a number the code no longer
# uses.
QUOTED_TIMEOUTS: tuple[tuple[Path, str, object, str], ...] = (
    (TROUBLESHOOTING, "attach timeout to {} before the first monarch import",
     dgx_monarch.ATTACH_CONFIG_TIMEOUT, "{}"),
    (TROUBLESHOOTING, "waits {}s for the attach to initialize",
     mesh_runtime.ATTACH_INIT_WAIT_S, "{}"),
    (TROUBLESHOOTING, "after {} seconds rather than parking every waiter",
     mesh_runtime._DEFAULT_MESH_CREATION_TIMEOUT_S, "{:g}"),
    (TROUBLESHOOTING, "finite positive value up to {} seconds",
     mesh_runtime._MAX_MESH_CREATION_TIMEOUT_S, "{:g}"),
    (TROUBLESHOOTING, "a {} s native-read budget",
     transfer.RDMA_READ_TIMEOUT_S, "{:g}"),
    (TROUBLESHOOTING, "a {} s margin on the future get",
     transfer.RDMA_GET_MARGIN_S, "{:g}"),
    (TROUBLESHOOTING, "one {} s drop budget per descriptor part",
     transfer.RDMA_READ_TIMEOUT_S, "{:g}"),
    (TROUBLESHOOTING, "up to {} ACK broadcast attempts",
     rdma_ownership.RDMA_ACK_ATTEMPTS, "{:g}"),
    (TROUBLESHOOTING, "with a {} s budget per attempt",
     rdma_ownership.RDMA_ACK_TIMEOUT_S, "{:g}"),
    (TROUBLESHOOTING, "a final {} s outer join margin",
     transfer.RDMA_READ_OUTER_MARGIN_S, "{:g}"),
    (TROUBLESHOOTING, "did not answer within {} s",
     capacity_agreement.CAPACITY_QUOTE_DEADLINE_S, "{:g}"),
)


def test_troubleshooting_quotes_the_live_timeout_constants():
    flattened = {path: _flatten(path.read_text()) for path, _, _, _ in QUOTED_TIMEOUTS}
    missing = [
        f"{path.name}: expected the sentence {sentence.format(formatter.format(value))!r}"
        for path, sentence, value, formatter in QUOTED_TIMEOUTS
        if sentence.format(formatter.format(value)) not in flattened[path]
    ]
    assert not missing, (
        "a timeout constant moved and its documented value did not:\n" + "\n".join(missing)
    )


def _bounded_section(path: Path, start: str, end: str) -> str:
    text = path.read_text()
    assert text.count(start) == 1 and text.count(end) == 1
    section = text.split(start, 1)[1].split(end, 1)[0]
    return _flatten(section)


def test_active_rdma_docs_pin_the_token_ack_and_manual_recycle_contract():
    sections = (
        _bounded_section(DESIGN, "### 5.6 Transfers", "### 5.7 ComfyUI node surface"),
        _bounded_section(
            TROUBLESHOOTING,
            "## 29. Large RDMA latent return fails after workers finish rendering",
            "## 30. Sampler Custom fails before model loading with `ModuleNotFoundError`",
        ),
    )
    for section in sections:
        assert "128-bit token" in section
        assert "setup generation" in section or "setup_generation" in section
        assert "exactly one" in section
        assert "zero ACK" in section
        assert "incomplete" in section and "ambiguous drop" in section
    assert "configured world-size responses" in sections[0]
    assert "configured worker count" in sections[1]
    assert all("Production does not recycle automatically" in section
               for section in sections)
    assert "client/operator" in sections[0]
    assert "client-owned ProcMesh" in sections[1]


@pytest.mark.parametrize(
    ("budget", "wait"),
    [("60s", 70), ("180s", 190), ("180", 190), ("30s", 70), ("", 70), ("soon", 70),
     # monarch reads this budget with humantime, so a real budget can arrive
     # spelled in minutes or hours. Reading bare seconds alone would drop `3m`
     # to the floor and put the 70 s wait back inside monarch's window.
     ("3m", 190), ("3 min", 190), ("1h", 3610), ("500ms", 70), ("2m30s", 70)],
)
def test_the_attach_wait_outlives_the_budget_in_force_not_the_shipped_one(
    monkeypatch, budget, wait,
):
    """The constant is the shipped budget; the environment is the real one.

    The ladder below reads the constant, so it could not see an operator who
    raised the budget. The sweep driver launches at 180 s, which put the 70 s
    wait inside monarch's own window and turned a typed per-host error back
    into a bare TimeoutError (2026-09-09). An unreadable value keeps the floor.
    """
    monkeypatch.setenv(mesh_runtime.ATTACH_CONFIG_TIMEOUT_ENV, budget)

    resolved = mesh_runtime._attach_init_wait()

    assert resolved == wait
    seconds = mesh_runtime._budget_seconds(budget)
    if seconds:
        # Every budget this reader can parse is outlived, whatever its unit.
        assert resolved > seconds


def test_the_timeout_ladder_stays_ordered():
    # Each bound is correct only relative to the one it must outlive, and no doc
    # sentence can enforce that, so this test does.
    raw = dgx_monarch.ATTACH_CONFIG_TIMEOUT
    assert re.fullmatch(r"\d+s", raw), (
        f"ATTACH_CONFIG_TIMEOUT is {raw!r}; the ladder below compares it against a "
        "seconds constant, so a change of unit needs a conversion here, not a crash"
    )
    config_push = int(raw[:-1])
    assert mesh_runtime.ATTACH_INIT_WAIT_S > config_push, (
        "the attach-init wait must outlive the config-push budget, or a slow "
        "attach fails with a bare TimeoutError instead of monarch's typed error"
    )
    assert transfer.RDMA_GET_MARGIN_S > 0, (
        "the future get must outlive the read op, or a get-side timeout abandons "
        "a live operation against the registration"
    )
    assert transfer.RDMA_READ_OUTER_MARGIN_S > transfer.RDMA_GET_MARGIN_S, (
        "the scratch-thread join must outlive the get, or a join TimeoutError "
        "replaces the primary error"
    )
    assert 0 < mesh_runtime._DEFAULT_MESH_CREATION_TIMEOUT_S <= mesh_runtime._MAX_MESH_CREATION_TIMEOUT_S
    assert 0 < mesh_liveness.PROBE_HOST_BUDGET_S < mesh_liveness.PROBE_FLEET_BUDGET_S, (
        "one host's probe must fit inside the fleet budget, or the first host "
        "can spend the whole deadline and the rest read unknown for no reason"
    )
    assert (mesh_liveness.PROBE_FLEET_BUDGET_S
            < mesh_runtime._DEFAULT_MESH_CREATION_TIMEOUT_S), (
        "the fleet probe must finish well inside the creation timeout, or the "
        "attach it is trying to unblock misses its own deadline"
    )


# Part 4: one home per doctrine fact

# One signature fragment per doctrine fact, chosen to appear only in that
# fact's canonical home. The one-sentence pointers elsewhere do not contain
# them.
CANONICAL_FRAGMENTS: tuple[tuple[str, str, str], ...] = (
    ("fabric trust", "certificate/CA plumbing", "SECURITY.md"),
    ("RDMA latent return", "hashes each host-memory registration", "docs/DESIGN.md"),
    # The HOLD's terms, as opposed to its status. Operator surfaces state the
    # status and link; whoever restates why the probe arm survives and what
    # would lift the hold has copied the record instead of pointing at it, and
    # that copy carries a version literal that goes stale.
    ("RDMA latent-return HOLD", "retained outside this repository", "docs/DESIGN.md"),
    ("attach timeout and wedge", "a wedged creator raises", "docs/TROUBLESHOOTING.md"),
    # The self-heal's terms. Every operator surface names the action for its own
    # cause; whoever restates what the heal costs and when it refuses has copied
    # the record instead of pointing at it.
    ("attached-mesh self-heal", "pays for fresh actors plus a cold model load",
     "docs/TROUBLESHOOTING.md"),
)


def _surface_text(path: Path) -> str:
    """Every unit of a surface as one string. Python goes through the same unit
    split as the scanner, so a fact restated in a src comment or an operator
    string is as visible here as one restated in a doc."""
    if path.suffix == ".py":
        return " ".join(text for _line, text in _python_units(path))
    return _flatten(path.read_text())


def test_each_doctrine_fact_has_exactly_one_home():
    texts = {
        path.relative_to(REPO).as_posix(): _surface_text(path)
        for path in _scanned_surfaces()
    }
    failures: list[str] = []
    for fact, fragment, home in CANONICAL_FRAGMENTS:
        homes = [name for name, text in texts.items() if fragment in text]
        if homes != [home]:
            failures.append(
                f"{fact}: {fragment!r} should live in {home} alone, found in {homes}"
            )
    assert not failures, (
        "a load-bearing fact has more than one home (or none); restate the copy "
        "as a link (DESIGN.md §7):\n" + "\n".join(failures)
    )
