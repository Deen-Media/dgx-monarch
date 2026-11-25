"""The consent read in an explicit-on sibling, against every short ledger the real writers make.

The rule (consent_quarantine._scope_decision, docs/DESIGN.md "A gate that compared
nothing denies no consent"), in three parts:

1. A no-material check's row in an explicit-on sibling (stamped by slab-mode
   equivalence and marked ``no_material_revocation``), of any build, never decides
   there and never hides what is under it: where it is the newest row of the
   context, the read looks beneath it.
2. Beneath it a liftable copy is taken out: a row another check copied into the
   context (a stamped row, the RETESTING guard of a check that names the context
   from its third place on, or the repair of that guard) that, read alone under
   this build, is a denial. A copied guard goes too when a later terminal row of
   the guard's own version (any source) closed it in that context. The row that
   closed it goes only when it is itself a liftable copy: a repair another
   source wrote reads unknown on its own and stays. A FAIL that no longer reads
   (retired) closes nothing.
3. Nothing else is taken out. A row a check that ran in the context left, a
   stamped PASS, a copy that reads clear alone, and every other row of another
   build read by the plain rules (another build's verdict row reads unknown).

The alphabet is every event the writers can make under each of three identities:
this build, this version from another source, and another version. Each sequence
is written through the real writers (``GateLedger``, ``gate_verdict``, ``dgxm gate
--repair``'s own ``repair``); the reads are compared with an oracle written here,
which applies the three parts from the rows' own fields to a copy of the ledger,
calls no lift code of the reader, strips the no-material marker from every row it
leaves in a sibling so the reader cannot lift on the copy, and reads that copy by
the plain rules. Every ledger is checked as written and again after a check of
this build that compared nothing, on five reads: contextless, for the load, the
panel's offer, and each explicit-on sibling under its own context. Two
consequences are asserted besides. No read is stricter after that check than
just before it unless what now decides, in an explicit-on context, is an own
row of this build that the plain rules refuse: the check's own row can close
another source's guard of this version, which read unknown over that row
(part 2). And an explicit-on unproven or cut-off row of this build, the newest
of what stays, still refuses.

Sequences of one and two events run by default: 1122 sequences, 2244 ledgers
read. Three run when ``DGXM_CONSENT_ENUMERATION_FULL=1`` is set, before every
release (CONTRIBUTING, "Release checks"): 37 059 sequences, 74 118 ledgers read,
about two minutes on one core.

The real gate ledgers of the rig are replayed the same way outside the suite:
every (key, artifacts) read contextless and under every context it names, under
every (protocol, version, source) seen in its rows, and compared with the reader
the change replaced. The reads production makes, and every contextless read, are
the same. The only reads that differ are scoped to an explicit-on context
under an older build, where another build's check that compared nothing is the
newest row and nothing stays beneath it: they read clear where the old reader
read unknown (part 1).
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

from test_consent_wiring import _all_three_reads, _h3_combo, _rig, checkpoint  # noqa: F401  (fixtures)

FULL_ENV = "DGXM_CONSENT_ENUMERATION_FULL"
LEDGER_NAME = "dgxm_gate_ledger.jsonl"
ANOTHER_SOURCE = "another-source"
ANOTHER_VERSION = "an-older-release"
IDENTITIES = ("this build", "another source", "another version")
EVENTS = ("on PASS", "on unproven", "on cut off", "on FAIL", "auto PASS", "auto unproven",
          "auto stock reference", "auto FAIL", "auto cut off", "auto no-material", "repair")
TERMINAL = ("PASS", "FAIL", "INCONCLUSIVE")
STRICTNESS = {"clear": 0, "unknown": 1, "denied": 2, "fail": 3}


def _writer(event: str):
    from dgx_monarch.cli import gate_repair
    from no_material_ledger_rows import open_auto_ceremony, write_ceremony_rows, write_explicit_on_rows

    outcome = {"on PASS": "PASS", "on unproven": "repeat diverged", "on cut off": "cut off",
               "on FAIL": "cross leg failed"}
    kind = {"auto PASS": "PASS", "auto unproven": "unproven", "auto stock reference": "stock reference",
            "auto FAIL": "FAIL", "auto no-material": "no_material"}
    if event in outcome:
        return lambda ledger, key, artifacts, _out: write_explicit_on_rows(ledger, key, artifacts, outcome[event])
    if event in kind:
        return lambda ledger, key, artifacts, _out: write_ceremony_rows(ledger, key, artifacts, kind=kind[event])
    if event == "auto cut off":
        return lambda ledger, key, artifacts, _out: open_auto_ceremony(ledger, key, artifacts)
    return lambda _ledger, _key, _artifacts, out: gate_repair.repair(str(out), apply=True)


def _write(event: str, identity: str, directory: Path, key, artifacts, monkeypatch) -> None:
    from dgx_monarch import gate_ledger, runtime_provenance
    from dgx_monarch.cli import gate_repair

    with monkeypatch.context() as build:
        if identity == "another source":
            build.setattr(runtime_provenance, "cached_dgx_source_manifest_sha256", lambda: ANOTHER_SOURCE)
        elif identity == "another version":
            build.setattr(gate_ledger, "__version__", ANOTHER_VERSION)
            build.setattr(gate_repair, "__version__", ANOTHER_VERSION)
        _writer(event)(gate_ledger.GateLedger(str(directory)), key, artifacts, directory)


# ---- the oracle: the three parts from the rows' own fields --------------------------------


def _version(row: dict) -> tuple:
    return row.get("gate_protocol"), row.get("dgx_monarch")


def _this_build(row: dict, source: str) -> bool:
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    return _version(row) == (GATE_PROTOCOL_VERSION, __version__) and row.get("dgx_source") == source


def _in(row: dict, context: str) -> bool:
    return row.get("capability_context") == context or context in (row.get("blocked_contexts") or ())


def _place(guard: dict, context: str) -> int:
    blocked = guard.get("blocked_contexts") or []
    return blocked.index(context) if context in blocked else -1


def _no_material(row: dict) -> bool:
    return (row.get("verdict") == "INCONCLUSIVE" and row.get("stamped") == "slab-mode-equivalence"
            and row.get("no_material_revocation") is True)


def _copied(row: dict, context: str, rows: list[dict]) -> bool:
    if row.get("stamped") == "slab-mode-equivalence":
        return True
    if row.get("verdict") == "RETESTING":
        return _place(row, context) >= 2
    if row.get("origin") == "gate_repair":
        return _place(rows[int(row["repaired_from_line"]) - 1], context) >= 2
    return False


def _denial_alone(row: dict, source: str) -> bool:
    """Read alone under this build, a denial: this build's RETESTING, or its INCONCLUSIVE that is
    neither a stock reference a typed class C refusal kept from loading nor a check that compared nothing."""
    if not _this_build(row, source):
        return False
    if row.get("verdict") == "RETESTING":
        return True
    stock_reference = (row.get("cross_mode") == "CAPACITY" and row.get("loras") == 0
                       and row.get("quarantine_levers") == [] and row.get("max_abs_latent_diff") == 0
                       and "/class-C " in str(row.get("capacity_detail", "")) + " ")
    nothing_compared = row.get("inconclusive_kind") == "no_material" or row.get("no_material_revocation")
    return row.get("verdict") == "INCONCLUSIVE" and not stock_reference and not nothing_compared


def _taken_out(rows: list[dict], context: str, source: str) -> set[int]:
    """The indices the three parts take out of one sibling context."""
    here = [index for index, row in enumerate(rows) if _in(row, context)]
    if not here or not _no_material(rows[here[-1]]):
        return set()
    out = {index for index in here if _no_material(rows[index])}
    out |= {index for index in here if _copied(rows[index], context, rows) and _denial_alone(rows[index], source)}
    for index in here:
        guard = rows[index]
        if guard.get("verdict") == "RETESTING" and _copied(guard, context, rows) and any(
                later > index and rows[later].get("verdict") in TERMINAL and not _retired(rows[later])
                and _version(rows[later]) == _version(guard) for later in here):
            out.add(index)
    return out


def _retired(row: dict) -> bool:
    """A FAIL that no longer reads: stamped by slab-mode equivalence with a cross leg that did not
    fail, or of another protocol or version. It closes no guard, as it decides nothing."""
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    return row.get("verdict") == "FAIL" and (
        (row.get("stamped") == "slab-mode-equivalence" and row.get("cross_mode") != "FAIL")
        or _version(row) != (GATE_PROTOCOL_VERSION, __version__))


def _oracle(text: str, siblings: tuple[str, ...], source: str) -> tuple[str, dict[str, dict]]:
    """The ledger with what the three parts take out gone from each sibling, and the newest
    row that stays in each sibling (a FAIL aside). A row taken out of every context it names
    keeps its line under another key, so a repair's pointer still holds."""
    rows = [json.loads(line) for line in text.splitlines()]
    out = copy.deepcopy(rows)
    newest: dict[str, dict] = {}
    for context in siblings:
        gone = _taken_out(rows, context, source)
        for index, row in enumerate(out):
            if not _in(rows[index], context):
                continue
            if index in gone:
                if row.get("verdict") == "RETESTING":
                    row["blocked_contexts"] = [value for value in row["blocked_contexts"] if value != context]
                    if not row["blocked_contexts"]:
                        row["key"] = "taken out by the oracle"
                else:
                    row["key"] = "taken out by the oracle"
            else:
                if row.get("stamped"):
                    row.pop("no_material_revocation", None)
                if row.get("verdict") != "FAIL":
                    newest[context] = rows[index]
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in out), newest


# ---- reads ----------------------------------------------------------------------------------


def _reads(directory: Path, checkpoint, key, artifacts, siblings) -> list[tuple[str, str]]:  # noqa: F811
    from dgx_monarch.nodes import consent_quarantine

    sys.modules["folder_paths"].get_output_directory = lambda: str(directory)  # type: ignore[attr-defined]
    reads = [(decision.state, decision.reason) for decision in _all_three_reads(checkpoint)]
    for context in siblings:
        decision = consent_quarantine.quarantine_decision(key, artifacts, context)
        reads.append((decision.state, decision.reason))
    return reads


def _own_refusal(row: dict, context: str, rows: list[dict], source: str) -> bool:
    """This build's explicit-on unproven or cut-off row in the context it ran: one that must refuse."""
    return (not _copied(row, context, rows) and _this_build(row, source)
            and row.get("origin") in ("explicit", "gate_repair")
            and (row.get("verdict") == "RETESTING" or _denial_alone(row, source)))


def _run(depth: int, checkpoint, tmp_path: Path, monkeypatch) -> int:  # noqa: F811
    from dgx_monarch import gate_ledger, runtime_provenance
    from no_material_ledger_rows import SIBLING_CONTEXTS, write_ceremony_rows

    monkeypatch.setattr(os, "fsync", lambda _fd: None)  # durability is not under test, and costs most of the time
    key, artifacts = _h3_combo(checkpoint)
    source = runtime_provenance.cached_dgx_source_manifest_sha256()
    siblings = tuple(gate_ledger.GateLedger._context_value(context) for context in SIBLING_CONTEXTS)
    levels = [tmp_path / f"level{depth_}" for depth_ in range(depth + 1)]
    checked, oracle = tmp_path / "checked", tmp_path / "oracle"
    for directory in (*levels, checked, oracle):
        directory.mkdir()
    alphabet = [(event, identity) for identity in IDENTITIES for event in EVENTS]
    count = 0

    def check(directory: Path, label: list[str], after_check: bool = False):
        text = (directory / LEDGER_NAME).read_text() if (directory / LEDGER_NAME).exists() else ""
        reads = _reads(directory, checkpoint, key, artifacts, siblings)
        expected_text, newest = _oracle(text, siblings, source)
        (oracle / LEDGER_NAME).write_text(expected_text)
        expected = _reads(oracle, checkpoint, key, artifacts, siblings)
        assert reads == expected, (label, reads, expected)
        rows = [json.loads(line) for line in text.splitlines()]
        if after_check:
            # This build's explicit-on unproven or cut-off row, newest of what stays, still refuses.
            on = siblings[0]
            if on in newest and _own_refusal(newest[on], on, rows, source):
                assert all(read[0] in ("denied", "fail") for read in (reads[0], reads[1], reads[3])), (label, reads)
        return reads, newest, rows

    def walk(level: int, label: list[str]) -> None:
        nonlocal count
        parent = levels[level]
        for event, identity in alphabet:
            child = levels[level + 1]
            (child / LEDGER_NAME).write_text((parent / LEDGER_NAME).read_text()
                                             if (parent / LEDGER_NAME).exists() else "")
            _write(event, identity, child, key, artifacts, monkeypatch)
            path = [*label, event if identity == "this build" else f"{event} ({identity})"]
            before, _newest, _rows = check(child, path)
            (checked / LEDGER_NAME).write_text((child / LEDGER_NAME).read_text())
            write_ceremony_rows(gate_ledger.GateLedger(str(checked)), key, artifacts)
            after, newest, rows = check(checked, [*path, "the check"], after_check=True)
            if any(STRICTNESS[now[0]] > STRICTNESS[then[0]] for now, then in zip(after, before, strict=True)):
                # Stricter after the check only where what now decides, in an explicit-on
                # context, is an own row of this build that the plain rules refuse.
                assert any(context in newest and _own_refusal(newest[context], context, rows, source)
                           for context in siblings), (path, before, after)
            count += 1
            if level + 1 < depth:
                walk(level + 1, path)

    walk(0, [])
    return count


@pytest.fixture(autouse=True)
def _quiet_repair(capsys):
    """``dgxm gate --repair`` prints what it closes; nothing here reads it."""
    yield
    capsys.readouterr()


def test_every_ledger_of_one_and_two_events_reads_by_the_rule(checkpoint, _rig, tmp_path, monkeypatch):  # noqa: F811
    size = len(IDENTITIES) * len(EVENTS)
    assert _run(2, checkpoint, tmp_path, monkeypatch) == size + size ** 2


@pytest.mark.skipif(os.environ.get(FULL_ENV) != "1",
                    reason=f"set {FULL_ENV}=1 to read every ledger of up to three events (before a release)")
def test_every_ledger_of_up_to_three_events_reads_by_the_rule(checkpoint, _rig, tmp_path, monkeypatch):  # noqa: F811
    size = len(IDENTITIES) * len(EVENTS)
    assert _run(3, checkpoint, tmp_path, monkeypatch) == size + size ** 2 + size ** 3
