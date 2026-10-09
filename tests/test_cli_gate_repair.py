"""`dgxm gate --repair`: close RETESTING rows no terminal row superseded.

Rows come from `GateLedger.begin_retest_required` and `GateLedger.record`, so
protocol, package version and source manifest are the running ones and the
ledger's own schema guard accepts them. Only the inert-row test edits a field
by hand.
"""
from __future__ import annotations

import json

import pytest

from dgx_monarch import __version__
from dgx_monarch.cli import gate_repair
from dgx_monarch.cli import main as cli
from dgx_monarch.cli.main import main
from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, GateLedger

CONTEXTS = [
    {"kind": "normal", "worker_args": {"slab_weights": "auto"}},
    {"kind": "fleet", "worker_args": {"slab_weights": "auto"}},
    {"kind": "normal", "worker_args": {"slab_weights": True}},
]


def _open_retest(tmp_path, contexts=None, *, model="flux2-dev.safetensors"):
    ledger = GateLedger(str(tmp_path))
    ledger.begin_retest_required(
        "combo", "sig", "commit", contexts or CONTEXTS[:1],
        {"model": model, "loras": 0, "origin": "auto_first_use",
         "run_id": "", "phase": "ceremony-preflight"})
    return ledger


def _run(tmp_path, *flags):
    return main(["gate", "--repair", "--report-dir", str(tmp_path), *flags])


def _rows(tmp_path):
    return GateLedger(str(tmp_path)).entries()


def test_a_dry_run_names_the_orphan_and_writes_nothing(tmp_path, capsys):
    ledger = _open_retest(tmp_path)
    before = open(ledger.path, "rb").read()

    code = _run(tmp_path)

    out = capsys.readouterr().out
    assert code == 1
    assert "flux2-dev.safetensors" in out
    assert "1 open context(s) on 1 RETESTING row(s)" in out
    assert "does not change what is refused" in out
    assert open(ledger.path, "rb").read() == before


def test_apply_appends_one_terminal_row_per_open_context(tmp_path, capsys):
    ledger = _open_retest(tmp_path)
    opened = ledger.entries()[-1]

    code = _run(tmp_path, "--apply")

    assert code == 0
    assert "appended 1 terminal INCONCLUSIVE row(s)" in capsys.readouterr().out
    rows = _rows(tmp_path)
    assert [row["verdict"] for row in rows] == ["RETESTING", "INCONCLUSIVE"]
    repaired = rows[-1]
    assert repaired["origin"] == "gate_repair"
    assert repaired["inconclusive_reasons"] == [gate_repair.REPAIR_REASON]
    assert repaired["repaired_from_time"] == opened["time"]
    assert repaired["capability_context"] == ledger._context_value(CONTEXTS[0])


def test_a_second_apply_finds_nothing(tmp_path, capsys):
    _open_retest(tmp_path)
    assert _run(tmp_path, "--apply") == 0
    capsys.readouterr()

    assert _run(tmp_path, "--apply") == 0
    assert len(_rows(tmp_path)) == 2
    assert "0 open context(s)" in capsys.readouterr().out


def test_a_retest_row_with_a_terminal_successor_is_untouched(tmp_path, capsys):
    ledger = _open_retest(tmp_path)
    ledger.record("combo", "sig", "commit", "PASS", {"model": "m"}, CONTEXTS[0])
    before = open(ledger.path, "rb").read()

    code = _run(tmp_path, "--apply")

    assert code == 0
    assert "0 open context(s)" in capsys.readouterr().out
    assert open(ledger.path, "rb").read() == before


def test_a_partly_closed_row_is_repaired_only_where_it_is_open(tmp_path):
    ledger = _open_retest(tmp_path, CONTEXTS)
    ledger.record("combo", "sig", "commit", "FAIL", {"model": "m"}, CONTEXTS[1])

    assert _run(tmp_path, "--apply") == 0

    rows = _rows(tmp_path)
    repaired = [row for row in rows if row.get("origin") == "gate_repair"]
    assert len(repaired) == 2
    assert {row["capability_context"] for row in repaired} == {
        ledger._context_value(CONTEXTS[0]), ledger._context_value(CONTEXTS[2])}


@pytest.mark.parametrize(
    "field,value",
    [("gate_protocol", GATE_PROTOCOL_VERSION + 1), ("dgx_monarch", "0.0.1")],
)
def test_a_row_from_another_protocol_or_build_is_reported_inert(
    tmp_path, capsys, field, value,
):
    """Inert is not clean: the count has its own line so an operator can tell."""
    ledger = _open_retest(tmp_path)
    rows = [json.loads(line) for line in
            open(ledger.path).read().splitlines() if line.strip()]
    rows[-1][field] = value
    with open(ledger.path, "w") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    before = open(ledger.path, "rb").read()

    code = _run(tmp_path)

    out = capsys.readouterr().out
    assert code == 0
    assert "1 inert rows skipped (protocol/version mismatch)" in out
    assert "0 open context(s)" in out
    assert open(ledger.path, "rb").read() == before


def test_the_repair_is_append_only(tmp_path):
    ledger = _open_retest(tmp_path)
    before = open(ledger.path, "rb").read()

    assert _run(tmp_path, "--apply") == 0

    after = open(ledger.path, "rb").read()
    assert after.startswith(before)
    assert len(after) > len(before)


def test_a_failed_append_names_the_row_and_exits_two(
    tmp_path, capsys, monkeypatch,
):
    _open_retest(tmp_path)
    monkeypatch.setattr(GateLedger, "record", lambda *_args, **_kwargs: False)

    code = _run(tmp_path, "--apply")

    assert code == 2
    assert "flux2-dev.safetensors" in capsys.readouterr().err


def test_apply_without_repair_is_refused(tmp_path, capsys):
    code = main(["gate", "--apply", "--report-dir", str(tmp_path)])

    assert code == 2
    assert "--apply needs --repair" in capsys.readouterr().err


def test_the_repair_never_resolves_a_driver_host(tmp_path, monkeypatch):
    """Disk only: an operator repairs a ledger with no driver running."""
    def refuse(_host):
        raise AssertionError("the repair arm must not probe for a driver")

    monkeypatch.setattr(cli, "_resolve_driver_host", refuse)
    _open_retest(tmp_path)

    assert _run(tmp_path) == 1
    assert _run(tmp_path, "--apply") == 0


def test_a_later_retesting_row_is_not_a_successor(tmp_path, capsys):
    """A second retest opens its own transaction; it closes nothing."""
    ledger = _open_retest(tmp_path)
    ledger.begin_retest_required("combo", "sig", "commit", CONTEXTS[:1])

    code = _run(tmp_path)

    assert code == 1
    assert "2 open context(s) on 2 RETESTING row(s)" in capsys.readouterr().out


def test_a_repaired_row_reads_the_same_state_the_open_row_read(tmp_path):
    """The repair closes the transaction and changes no authorization."""
    ledger = _open_retest(tmp_path)
    before = ledger.lookup("combo", "sig", "commit", CONTEXTS[0])

    assert _run(tmp_path, "--apply") == 0

    after = GateLedger(str(tmp_path)).lookup(
        "combo", "sig", "commit", CONTEXTS[0])
    assert before == after == "inconclusive"
    assert __version__ == GateLedger(str(tmp_path)).entries()[-1]["dgx_monarch"]
