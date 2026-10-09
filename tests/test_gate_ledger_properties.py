"""Property checks for damaged and duplicated gate-ledger JSONL records."""
from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dgx_monarch.gate_ledger import GateLedger

_RECORD = st.fixed_dictionaries({
    "key": st.text(max_size=20),
    "artifacts": st.text(max_size=20),
    "comfy": st.text(max_size=20),
    # The two audit verdicts (gate_audit_vocab.AUDIT_VERDICTS) go through the
    # same damage and duplicate checks as the others. An audit row must scan as
    # intact: lookup reads a PASS older than any damage as "error", and no row
    # older than the damage may back a process PASS.
    "verdict": st.sampled_from([
        "PASS", "FAIL", "INCONCLUSIVE", "unknown", "WAIVER", "CAPACITY_CERTIFIED"]),
})
_DAMAGE_LINE = st.one_of(
    st.sampled_from([b"", b" ", b"null", b"[]", b'"scalar"', b"{", b'{"torn":']),
    st.binary(max_size=64).map(
        lambda payload: b"\xff" + payload.replace(b"\n", b"_").replace(b"\r", b"_")
    ),
)
_PROPERTY_SETTINGS = settings(
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def _ledger_stream(draw):
    chunks: list[bytes] = []
    expected: list[dict] = []
    for _ in range(draw(st.integers(min_value=0, max_value=20))):
        if draw(st.booleans()):
            record = draw(_RECORD)
            encoded = json.dumps(record, separators=(",", ":")).encode() + b"\n"
            repeats = draw(st.integers(min_value=1, max_value=3))
            chunks.extend([encoded] * repeats)
            expected.extend([record] * repeats)
        else:
            chunks.append(draw(_DAMAGE_LINE) + b"\n")
    if draw(st.booleans()):
        chunks.append(b'{"torn_tail":')
    return b"".join(chunks), expected


def _encoded_lines(lines: list[bytes]) -> bytes:
    return b"".join(line + b"\n" for line in lines)


@_PROPERTY_SETTINGS
@given(stream=_ledger_stream())
def test_entries_preserve_exact_valid_dict_rows_amid_damage_and_duplicates(tmp_path, stream):
    payload, expected = stream
    ledger = GateLedger(str(tmp_path))
    (tmp_path / "dgxm_gate_ledger.jsonl").write_bytes(payload)
    assert ledger.entries() == expected


@_PROPERTY_SETTINGS
@given(garbage=st.lists(_DAMAGE_LINE, max_size=20))
def test_garbage_lines_cannot_mint_a_pass_verdict(tmp_path, garbage):
    ledger = GateLedger(str(tmp_path))
    (tmp_path / "dgxm_gate_ledger.jsonl").write_bytes(_encoded_lines(garbage))
    # A stream of only empty or whitespace lines reads a clean unknown. Damage
    # with no intact target row also reads unknown, never PASS; authority
    # callers read the scan's integrity bit (session_pass_safe) separately.
    assert ledger.lookup("target", "sig", "commit") != "pass"


@_PROPERTY_SETTINGS
@given(
    before=st.lists(_DAMAGE_LINE, max_size=10),
    between=st.lists(_DAMAGE_LINE, max_size=10),
    after=st.lists(_DAMAGE_LINE, max_size=10),
    fail_copies=st.integers(min_value=1, max_value=4),
)
def test_intact_fail_stays_sticky_through_damage_and_duplicate_rows(
        tmp_path, before, between, after, fail_copies):
    pass_row = json.dumps({
        "key": "target", "artifacts": "sig", "comfy": "commit", "verdict": "PASS",
    }, separators=(",", ":")).encode()
    fail_row = json.dumps({
        "key": "target", "artifacts": "sig", "comfy": "commit", "verdict": "FAIL",
    }, separators=(",", ":")).encode()
    lines = [*before, pass_row, *between, *([fail_row] * fail_copies), *after]
    ledger = GateLedger(str(tmp_path))
    (tmp_path / "dgxm_gate_ledger.jsonl").write_bytes(_encoded_lines(lines))
    assert ledger.lookup("target", "sig", "different-commit") == "fail"
