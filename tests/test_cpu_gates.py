"""scripts/test_cpu.sh runs every gate CI's lint-and-unit job runs that can run off CI.

A local green run used to leave CI's gates unchecked: the protected test
inventory, the refusal ledger, the TRUST citations, the leak check, ruff, mypy
and the generator checks ran only as CI steps, so a contributor learned of
them after a paid push. The wrapper now runs them before the suite. Each CI
step is classified here, so a step CI gains must be added to the wrapper or
named as CI-only, with the reason. A green wrapper run covers the local gates
and the suite. It does not promise a green CI run: the CI-only steps below
run nowhere else, and CI's tools can be other versions than a developer's.
"""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (ROOT / "scripts" / "test_cpu.sh").read_text(encoding="utf-8")
CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
SDIST_STEP = "Validate wheel installability and source completeness"

# CI step -> the command text the step and the wrapper both carry.
LOCAL = {
    "Leak assertion (tracked working tree)": ["tools/leak_check.py"],
    "ruff": ["ruff check __init__.py src tests benchmark tools studio/tests"],
    "Trust citations": ["tools/repin_trust_citations.py --check"],
    "Protected test inventory": ["tools/check_protected_tests.py --check"],
    "Refusal ledger is current": ["tests/test_refusal_classes.py --write"],
    "mypy (full package)": ["mypy -p dgx_monarch"],
    "Surface artifact checks": [
        "compileall -q __init__.py src benchmark tests tools studio/tests",
        "tools/gen_templates.py --check", "tools/gen_brand_assets.py --check",
        "tools/gen_template_cards.py --check", "tools/check_artifacts.py --strict",
        "bash -n", "shellcheck scripts/*.sh", "node --check",
        "node tools/check_dgx_monarch_segments.mjs",
    ],
}
# CI steps that cannot run as a local gate, and why.
CI_ONLY = {
    "Install shellcheck": "installs a tool on the runner; the wrapper skips shellcheck where it is absent",
    "Install graphic export dependencies": "installs system packages on the runner",
    "Install (pinned CUDA-linked torch + pinned monarch)": (
        "builds CI's own environment: torch 2.12.0+cu132 with no CUDA device, and "
        "tools/check_dependencies.py, which downloads a wheel to check one install defect"),
    "pytest (no-GPU unit suite)": "is this wrapper",
    SDIST_STEP: (
        "builds the wheel and sdist into dist/ and two virtual environments, and writes a "
        "sentinel into .claude/worktrees; it changes the checkout it checks. The wrapper's "
        "source archive gate runs this step's sdist census on a copy of the tracked files; "
        "the wheel checks, the step's other sdist assertions, the tests it runs from the "
        "extracted archive and the two smoke environments stay with CI"),
    "Wheel ships exactly the hashed Studio resources": "reads the wheel the previous step builds",
    "dgxm smoke": "expects dgxm doctor to fail for want of a CUDA device, which a Spark has",
}


def _steps() -> dict[str, str]:
    return {step["name"]: step.get("run", "") for step in CI["jobs"]["lint-and-unit"]["steps"]
            if "name" in step}


def test_every_ci_step_is_a_local_gate_or_named_ci_only():
    steps = _steps()
    assert set(steps) == set(LOCAL) | set(CI_ONLY), {
        "unclassified": sorted(set(steps) - set(LOCAL) - set(CI_ONLY)),
        "gone": sorted((set(LOCAL) | set(CI_ONLY)) - set(steps)),
    }


def test_the_wrapper_runs_each_local_gate_as_ci_writes_it():
    steps = _steps()
    for name, commands in LOCAL.items():
        for command in commands:
            assert command in steps[name], (name, command)
            assert command in WRAPPER, (name, command)


def test_the_protected_inventory_gate_runs_first_and_the_suite_after_every_gate():
    gates = re.findall(r'^\s*gate "([^"]+)"', WRAPPER, flags=re.MULTILINE)
    assert gates[0] == "protected test inventory"
    assert WRAPPER.index("run_gates\n") < WRAPPER.index('-m pytest tests/')


def test_ci_and_the_opt_out_skip_the_gates():
    assert '[ "${GITHUB_ACTIONS:-}" != "true" ] && [ "${DGXM_SKIP_GATES:-}" != "1" ]' in WRAPPER


def test_the_wrapper_passes_the_shellcheck_ci_runs_as_well_as_a_newer_one():
    # The gate helper calls its check functions by name. ShellCheck 0.9.0, which
    # CI's runner carries, calls that SC2317; 0.11 calls it SC2329 and 0.9.0
    # ignores a code it does not know, so the file-wide directive names both.
    directive = re.search(r"^# shellcheck disable=(\S+)$", WRAPPER, flags=re.MULTILINE)
    assert directive and {"SC2317", "SC2329"} <= set(directive.group(1).split(","))
    assert WRAPPER.index(directive.group(0)) < WRAPPER.index("\nset -euo pipefail\n")


def test_the_source_archive_gate_applies_cis_census_rule():
    # CI holds the sdist to the tracked files plus the files the build writes.
    # The wrapper's gate runs the same comparison on a copy of the tracked tree.
    gates = re.findall(r'^\s*gate "([^"]+)"', WRAPPER, flags=re.MULTILINE)
    assert "source archive" in gates
    assert 'build --sdist --no-isolation --outdir "$scratch/dist" "$scratch/tree"' in WRAPPER

    def generated(text: str) -> set[str]:
        block = re.search(r"generated = \{(.*?)\}", text, flags=re.DOTALL)
        assert block, "no generated-file set"
        return set(re.findall(r'"([^"]+)"', block.group(1)))

    assert generated(WRAPPER) == generated(_steps()[SDIST_STEP])
    assert "expected = tracked | generated" in WRAPPER
    assert "expected_members = tracked | generated" in _steps()[SDIST_STEP]


def test_the_gates_keep_their_scratch_and_caches_out_of_the_checkout():
    # One folder from mktemp holds the saved ledger, the sdist build and
    # compileall's caches, and one EXIT trap removes it.
    assert "  gate_tmp=$(mktemp -d)\n  trap 'rm -rf \"$gate_tmp\"' EXIT\n" in WRAPPER
    assert 'PYTHONPYCACHEPREFIX="$gate_tmp/pycache"' in WRAPPER
    assert "local scratch=$gate_tmp/sdist" in WRAPPER
    assert "saved=$gate_tmp/refusal_class_ledger.json" in WRAPPER


def _ledger_gate(tmp_path: Path, writer: str) -> tuple[Path, Path, subprocess.Popen]:
    """Start the wrapper's ledger gate in a scratch tree, with ``writer`` as its interpreter."""
    function = re.search(r"^refusal_ledger_is_current\(\) \{\n.*?^\}\n", WRAPPER, flags=re.MULTILINE | re.DOTALL)
    assert function, "the wrapper has no refusal_ledger_is_current"
    tree, scratch = tmp_path / "tree", tmp_path / "scratch"
    (tree / "tests").mkdir(parents=True)
    scratch.mkdir()
    ledger = tree / "tests" / "refusal_class_ledger.json"
    ledger.write_text("as found\n", encoding="utf-8")
    stub = tmp_path / "writer.sh"
    stub.write_text(f"#!/usr/bin/env bash\n{writer}\n", encoding="utf-8")
    stub.chmod(0o755)
    script = tmp_path / "gate.sh"
    script.write_text(
        f"set -uo pipefail\ncd {shlex.quote(str(tree))}\ntest_python={shlex.quote(str(stub))}\n"
        "gate_tmp=$(mktemp -d)\ntrap 'rm -rf \"$gate_tmp\"' EXIT\n"  # as run_gates sets them
        f"{function.group(0)}refusal_ledger_is_current\n", encoding="utf-8")

    def default_signals() -> None:  # a suite started in the background inherits SIGINT ignored
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # Its own session, so a signal sent to its group reaches nothing but the gate and its writer.
    process = subprocess.Popen(
        ["bash", str(script)], env={**os.environ, "TMPDIR": str(scratch), "MARK": str(tmp_path / "mark")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True, preexec_fn=default_signals)
    return ledger, scratch, process


@pytest.mark.parametrize(("writer", "status"), [
    ("exit 0", 0),
    ("echo rewritten > tests/refusal_class_ledger.json", 1),
    ("echo half > tests/refusal_class_ledger.json; exit 3", 1),
    ("exit 3", 1),
], ids=["current", "stale", "writer failed after writing", "writer failed before writing"])
def test_the_ledger_gate_fails_on_a_stale_ledger_or_a_failed_writer_and_leaves_the_ledger(tmp_path, writer, status):
    ledger, scratch, process = _ledger_gate(tmp_path, writer)
    output, _ = process.communicate(timeout=60)
    assert process.returncode == status, output
    assert ("rewritten" in output) == ("rewritten" in writer), output  # a stale ledger prints its diff
    assert ledger.read_text(encoding="utf-8") == "as found\n"
    assert not any(scratch.iterdir())


@pytest.mark.parametrize("interrupt", [signal.SIGINT, signal.SIGTERM], ids=["Ctrl-C", "TERM"])
def test_an_interrupted_ledger_gate_puts_the_tracked_ledger_back(tmp_path, interrupt):
    ledger, scratch, process = _ledger_gate(
        tmp_path, 'echo rewritten > tests/refusal_class_ledger.json; touch "$MARK"; exec sleep 60')
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "mark").exists():
            assert process.poll() is None and time.monotonic() < deadline, "the writer never started"
            time.sleep(0.02)
        assert ledger.read_text(encoding="utf-8") == "rewritten\n"
        assert os.getpgid(process.pid) == process.pid  # the group this test started, and no other
        os.killpg(process.pid, interrupt)
        output, _ = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=60)
    assert process.returncode == 130, output
    assert ledger.read_text(encoding="utf-8") == "as found\n"
    assert not any(scratch.iterdir())
