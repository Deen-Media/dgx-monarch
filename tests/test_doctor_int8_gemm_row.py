"""Classify the INT8 GEMM doctor rows from the probe's wire format.

Cases use the probe's literal values ("installed", "MISSING", "present",
"missing", "unknown"). No case may return FAIL or mark a row critical: without
its prerequisites the kernel falls back to stock CUTLASS, so a missing one
costs speed, never a render, and must not change doctor's exit code.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from dgx_monarch.cli import int8_gemm_row
from dgx_monarch.cli.probe_certainty import FAIL, OK, WARN, probe_row

HOST = "worker"
UNSEEN = "cannot see the kernel that runs"
JOURNAL = "`INT8 GEMM:` line in the Worker journal and in the driver log"


def _fields(kitchen="0.2.40", triton="installed", pyheaders="present") -> dict[str, str]:
    return {"int8gemm": "probe", "kitchen": kitchen, "triton": triton, "pyheaders": pyheaders}


def _host(fields, worker_args=None) -> dict:
    """The real doctor row, so the wiring is pinned and not just the tuple."""
    return probe_row(*int8_gemm_row.host_row(HOST, fields, worker_args))


HOST_CASES = {
    "ready by default": (_fields(), None, OK, "the default selects the Triton kernel, and the Worker venv has "
                                              "comfy-kitchen 0.2.40 (validated), Triton and Python.h"),
    "ready by config": (_fields(), {"int8_gemm": "triton"}, OK, "[worker_args] int8_gemm selects the Triton kernel"),
    "off by config": (_fields(kitchen="MISSING"), {"int8_gemm": "cutlass"}, OK,
                      "cutlass from [worker_args] int8_gemm: the actors there run stock CUTLASS"),
    "unvalidated kitchen": (_fields(kitchen="0.2.41"), None, WARN,
                            "comfy-kitchen 0.2.41 is not in the validated set (0.2.37, 0.2.40)"),
    "no kitchen": (_fields(kitchen="MISSING"), None, WARN,
                   "comfy-kitchen is not installed, so there is no INT8 GEMM to replace"),
    "unreadable kitchen": (_fields(kitchen="unknown"), None, WARN, "the comfy-kitchen version could not be read"),
    "no triton": (_fields(triton="MISSING"), None, WARN, "Triton is not installed"),
    "no headers": (_fields(pyheaders="missing"), None, WARN, "in the Worker venv Python.h is missing"),
    "unreadable headers": (_fields(pyheaders="unknown"), None, WARN, "whether Python.h is present could not be read"),
    "everything missing": (_fields(kitchen="0.2.41", triton="MISSING", pyheaders="missing"), None, WARN,
                           "(0.2.37, 0.2.40); Triton is not installed; Python.h is missing"),
    "blind": ({}, None, WARN, "unobserved: the INT8 GEMM probe did not answer on worker"),
}


@pytest.mark.parametrize("label", sorted(HOST_CASES))
def test_each_host_case_reads_as_it_should_and_never_fails(label):
    fields, worker_args, status, words = HOST_CASES[label]
    row = _host(fields, worker_args)
    assert row["name"] == f"{HOST} INT8 GEMM"
    assert row["status"] == status, row["detail"]
    assert row["status"] != FAIL and "critical" not in row
    assert words in row["detail"], row["detail"]
    # Every row says what doctor cannot observe and where the answer is.
    assert UNSEEN in row["detail"] and JOURNAL in row["detail"]


def test_a_missing_prerequisite_names_the_fallback_and_the_entry():
    detail = _host(_fields(pyheaders="missing"))["detail"]
    assert "The actors there run stock CUTLASS and log one WARNING" in detail
    assert "renders still run" in detail
    assert int8_gemm_row.ENTRY in detail


def test_a_blind_host_warns_unobserved_without_taking_the_exit_code():
    row = _host({})
    assert (row["status"], row["reason"]) == (WARN, "unobserved")
    assert "critical" not in row


def test_the_local_row_reads_this_shell_and_the_local_actors():
    ready = _fields()
    status, name, detail = int8_gemm_row.local_row(None, local_mode=True, environ={}, fields=ready)
    assert (status, name) == (OK, int8_gemm_row.LOCAL_NAME)
    assert "A ComfyUI started from this shell selects the Triton kernel (DGXM_INT8_GEMM unset)" in detail
    assert "The local actors select the Triton kernel (the default)" in detail
    assert "This interpreter has a validated comfy-kitchen, Triton and Python.h" in detail
    assert UNSEEN in detail

    status, _, detail = int8_gemm_row.local_row({"int8_gemm": "cutlass"}, local_mode=True,
                                                environ={"DGXM_INT8_GEMM": "cutlass"},
                                                fields=_fields(pyheaders="missing"))
    assert status == OK, "nothing here selects the kernel, so nothing is missing"
    assert "selects stock CUTLASS (DGXM_INT8_GEMM='cutlass')" in detail
    assert "The local actors select stock CUTLASS ([worker_args] int8_gemm)" in detail

    status, _, detail = int8_gemm_row.local_row(None, local_mode=False, environ={},
                                                fields=_fields(triton="MISSING"))
    assert status == WARN
    assert "local actors" not in detail, "cluster actors have their own host rows"
    assert "In this interpreter Triton is not installed, so that ComfyUI runs stock CUTLASS" in detail

    status, _, detail = int8_gemm_row.local_row(None, local_mode=False, environ={"DGXM_INT8_GEMM": "fast"},
                                                fields=ready)
    assert status == WARN
    assert "DGXM_INT8_GEMM='fast' in this shell names neither" in detail


def test_the_probe_prints_one_parseable_line_without_importing_triton():
    """Run the payload and read its output the way doctor does."""
    from dgx_monarch.cli import doctor

    source = int8_gemm_row.PROBE_SOURCE
    assert "find_spec" in source and "import triton" not in source and "torch" not in source
    assert "\nEOF" not in source
    probe = source + "\nimport sys\nprint('loaded=' + str('triton' in sys.modules))\n"
    result = subprocess.run([sys.executable, "-"], input=probe, capture_output=True, text=True,
                            timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    printed = result.stdout.strip().splitlines()
    assert printed[-1] == "loaded=False"
    assert len(printed) == 2 and printed[0].startswith(int8_gemm_row.LINE_PREFIX)
    fields = doctor._env_tokens(printed[0])
    assert set(fields) == {"int8gemm", "kitchen", "triton", "pyheaders"}
    row = _host(fields)
    assert row["status"] != FAIL and "did not answer" not in row["detail"]
    assert fields == {"int8gemm": "probe", **int8_gemm_row.local_fields()}


def test_doctor_sends_the_probe_and_emits_one_row_per_host():
    import inspect

    from dgx_monarch.cli import doctor

    source = inspect.getsource(doctor)
    assert "{int8_gemm_row.PROBE_SOURCE}" in source
    assert "int8_gemm_row.select(lines)" in source
    assert "int8_gemm_row.host_rows(config, int8_gemm)" in source
    assert "int8_gemm_row.local_row_for(config)" in source


def test_select_reads_only_this_probes_line():
    lines = ["sol=True tvm=True", "int8gemm=probe kitchen=0.2.40 triton=installed pyheaders=present",
             "swap_kib=0"]
    assert int8_gemm_row.select(lines) == _fields()
    assert int8_gemm_row.select(["sol=True"]) == {}


def test_host_rows_follow_host_order_and_read_a_silent_host_as_unobserved():
    from dgx_monarch.config import ClusterConfig, HostConfig

    config = ClusterConfig(hosts=(HostConfig(name="head", address="tcp://192.0.2.1:26600"),
                                  HostConfig(name="worker", address="tcp://192.0.2.2:26600")),
                           worker_args={})
    rows = int8_gemm_row.host_rows(config, {"worker": _fields()})
    assert [name for _status, name, _detail in rows] == ["head INT8 GEMM", "worker INT8 GEMM"]
    assert rows[0][2].startswith("unobserved:") and rows[1][0] == OK


def test_the_local_row_never_raises(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("probe broke")

    monkeypatch.setattr(int8_gemm_row, "local_row", boom)
    status, name, detail = int8_gemm_row.local_row_for(None)
    assert (status, name) == (WARN, int8_gemm_row.LOCAL_NAME)
    assert "probe unavailable (RuntimeError('probe broke'))" in detail


def test_the_validated_set_is_the_option_module_set():
    from dgx_monarch.kernels import int8_gemm

    assert int8_gemm_row.VALIDATED_KITCHEN is int8_gemm.VALIDATED_KITCHEN
