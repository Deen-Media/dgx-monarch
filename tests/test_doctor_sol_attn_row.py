"""Classify the advisory sol-attn install row from the probe's wire format.

Cases use the heredoc's literal values, including "True", "False", "MISSING",
and "12.1". No case may return FAIL or mark the row critical: this
off-by-default feature must not change doctor's exit code, even when its
probe is unreadable.
"""
from __future__ import annotations

from dgx_monarch.cli import sol_attn_row
from dgx_monarch.cli.probe_certainty import FAIL, OK, WARN, probe_row

HEAD = "head"
WORKER = "worker"


def _fields(sol=True, tvm=True, cutlass=True, cap="12.1",
            solver="0.5.0", soldsl="4.7.0") -> dict[str, str]:
    """One host's probe output, spelled the way the heredoc prints it."""
    return {
        "sol": "True" if sol else "False",
        "tvm": "True" if tvm else "False",
        "cutlass": "True" if cutlass else "False",
        "solver": solver,
        "soldsl": soldsl,
        "cap": cap,
    }


def _row(per_host) -> dict:
    """The real doctor row, so the wiring is pinned and not just the tuple."""
    return probe_row(*sol_attn_row.doctor_row(per_host))


# The cases the row tells apart, each named by what the probes report.
CASES: dict[str, dict[str, dict[str, str]]] = {
    "absent everywhere": {HEAD: _fields(sol=False, tvm=False, cutlass=False,
                                        solver="MISSING", soldsl="MISSING"),
                          WORKER: _fields(sol=False, tvm=False, cutlass=False,
                                          solver="MISSING", soldsl="MISSING")},
    "installed everywhere": {HEAD: _fields(), WORKER: _fields()},
    "capability elsewhere": {HEAD: _fields(cap="9.0"), WORKER: _fields(cap="9.0")},
    "capability unknown": {HEAD: _fields(cap="unknown"), WORKER: _fields(cap="unknown")},
    "one box only": {HEAD: _fields(), WORKER: _fields(sol=False, tvm=False,
                                                      solver="MISSING")},
    "tvm missing on one": {HEAD: _fields(), WORKER: _fields(tvm=False)},
    "tvm missing everywhere": {HEAD: _fields(tvm=False), WORKER: _fields(tvm=False)},
    "dsl missing on one": {HEAD: _fields(),
                           WORKER: _fields(cutlass=False, soldsl="MISSING")},
    "dsl missing here, capability unreadable there": {
        HEAD: _fields(cutlass=False, soldsl="MISSING"),
        WORKER: _fields(cap="unknown")},
    "one box blind": {HEAD: _fields(), WORKER: {}},
    "every box blind": {HEAD: {}, WORKER: {}},
    "no hosts probed": {},
}


def test_absent_everywhere_reads_ok_and_names_the_install_pointer():
    status, name, detail = sol_attn_row.doctor_row(CASES["absent everywhere"])
    assert (status, name) == (OK, "sol-attn install")
    assert "not installed on any host" in detail
    assert sol_attn_row.ENTRY in detail


def test_full_preconditions_read_ok_and_point_at_the_worker_log():
    status, _name, detail = sol_attn_row.doctor_row(CASES["installed everywhere"])
    assert status == OK
    assert "install state only" in detail
    assert "backend=... cute=..." in detail
    # The versions are the probe's, never literals in the row's prose.
    assert "sol-attn 0.5.0" in detail and "nvidia-cutlass-dsl 4.7.0" in detail


def test_capability_elsewhere_reads_ok_and_defers_to_upstream_dispatch():
    status, _name, detail = sol_attn_row.doctor_row(CASES["capability elsewhere"])
    assert status == OK
    assert "is not (12, 1)" in detail
    assert "upstream's own dispatch decides" in detail


def test_unreadable_capability_softens_rather_than_claiming_a_mismatch():
    status, _name, detail = sol_attn_row.doctor_row(CASES["capability unknown"])
    assert status == OK
    assert "nvidia-smi did not answer" in detail
    assert "is not (12, 1)" not in detail


def test_asymmetric_presence_warns_and_names_the_box_without_it():
    status, _name, detail = sol_attn_row.doctor_row(CASES["one box only"])
    assert status == WARN
    assert f"sol-attn is installed on {HEAD} and missing on {WORKER}" in detail


def test_missing_tvm_warns_in_the_same_shape_as_a_missing_package():
    status, _name, detail = sol_attn_row.doctor_row(CASES["tvm missing on one"])
    assert status == WARN
    assert f"apache-tvm-ffi is installed on {HEAD} and missing on {WORKER}" in detail


def test_missing_tvm_everywhere_names_no_host_as_having_it():
    status, _name, detail = sol_attn_row.doctor_row(CASES["tvm missing everywhere"])
    assert status == WARN
    assert f"apache-tvm-ffi is missing on {HEAD}, {WORKER}" in detail
    assert "installed on " not in detail


def test_missing_dsl_warns_that_nothing_refuses_and_names_the_fix():
    status, _name, detail = sol_attn_row.doctor_row(CASES["dsl missing on one"])
    assert status == WARN
    assert f"nvidia-cutlass-dsl is missing on {WORKER}" in detail
    assert "Nothing refuses" in detail
    assert sol_attn_row.DSL_INSTALL in detail


def test_another_hosts_unreadable_capability_cannot_hide_a_missing_compiler():
    """One host's unreadable capability must not hide a missing compiler on another.

    nvidia-smi can fail or time out on one box for reasons of its own while
    another box at the dispatch capability runs Triton without the DSL. If the
    unreadable answer decided the row, the operator would read OK for the one
    case the row must catch.
    """
    status, _name, detail = sol_attn_row.doctor_row(
        CASES["dsl missing here, capability unreadable there"])
    assert status == WARN
    assert f"nvidia-cutlass-dsl is missing on {HEAD}" in detail
    assert sol_attn_row.DSL_INSTALL in detail


def test_a_host_at_the_dispatch_capability_without_the_compiler_never_reads_ok():
    """The invariant behind the case above, over every answer the peer can give."""
    for peer in (_fields(), _fields(cap="9.0"), _fields(cap="unknown"),
                 _fields(cutlass=False, soldsl="MISSING")):
        status, _name, detail = sol_attn_row.doctor_row(
            {HEAD: _fields(cutlass=False, soldsl="MISSING"), WORKER: peer})
        assert status == WARN, peer
        assert HEAD in detail, peer


def test_a_blind_host_warns_unobserved_without_taking_the_exit_code():
    row = _row(CASES["one box blind"])
    assert row["status"] == WARN
    assert row["reason"] == "unobserved"
    assert "critical" not in row
    assert WORKER in row["detail"]


def test_every_case_emits_exactly_one_row_that_never_fails():
    for label, per_host in CASES.items():
        row = _row(per_host)
        assert row["name"] == "sol-attn install", label
        assert row["status"] != FAIL, label
        assert "critical" not in row, label
        assert row["detail"], label


def test_row_constants_track_the_backend_owner():
    """The dispatch decides these; the row only quotes them.

    A moved capability or a moved compiler pin must fail here, not leave a
    doctor row giving the operator a stale instruction.
    """
    from dgx_monarch.adapters import sol_attention_backend

    major, minor = sol_attention_backend.GB10_CAPABILITY
    assert sol_attn_row.CUTE_CAPABILITY == f"{major}.{minor}"
    assert sol_attn_row.DSL_INSTALL == sol_attention_backend.CUTE_PATH_INSTALL["cutlass"]


def test_the_probe_prints_one_parseable_line_of_every_field():
    """The heredoc prints one line in the whitespace `key=value` form doctor parses."""
    source = sol_attn_row.PROBE_SOURCE
    for field in ("sol=", "tvm=", "cutlass=", "solver=", "soldsl=", "cap="):
        assert field in source
    # find_spec, not import: doctor is passive and can run beside a render.
    assert "find_spec" in source and "import sol_attn" not in source
    # Capability without a CUDA context, and no heredoc terminator in the body.
    assert "nvidia-smi" in source and "torch" not in source
    assert "\nEOF" not in source


def test_the_probe_prints_the_line_doctor_selects_and_the_row_reads_it():
    """Run the payload and take its output the way doctor takes it.

    doctor picks this probe's line out of several by its prefix, so a reordered
    print or a respelled field would leave every host reading blind and the row
    unobserved for good, with nothing to say the payload was the cause. This
    asserts the mechanics, not the answers: the fields are whatever the
    interpreter running the suite has.
    """
    import subprocess
    import sys

    from dgx_monarch.cli import doctor

    result = subprocess.run(
        [sys.executable, "-"], input=sol_attn_row.PROBE_SOURCE,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    printed = result.stdout.strip().splitlines()
    assert len(printed) == 1
    assert printed[0].startswith(sol_attn_row.LINE_PREFIX)
    fields = doctor._env_tokens(printed[0])
    assert set(fields) == {"sol", "tvm", "cutlass", "solver", "soldsl", "cap"}
    row = _row({HEAD: fields})
    assert row["name"] == "sol-attn install"
    assert row["status"] != FAIL
    assert "unobserved" not in row["detail"]
