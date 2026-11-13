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
import shutil
import subprocess
import sys
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
    "Refusal ledger is current": ["tests/test_refusal_classes.py --check"],
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


def test_no_gate_writes_into_the_checkout_to_compare():
    # A gate that regenerated a tracked file in place and copied the old one
    # back could cross with a second run in the same checkout, or be cut off
    # between the two. Every gate now compares in memory or works in the one
    # folder from mktemp, which one EXIT trap removes.
    assert "  gate_tmp=$(mktemp -d)\n  trap 'rm -rf \"$gate_tmp\"' EXIT\n" in WRAPPER
    assert 'PYTHONPYCACHEPREFIX="$gate_tmp/pycache"' in WRAPPER
    assert "local scratch=$gate_tmp/sdist" in WRAPPER
    assert "--write" not in WRAPPER, "a gate runs a tool in its write mode"
    assert WRAPPER.count("trap ") == 1, "a gate has something to put back"
    ledger_step = _steps()["Refusal ledger is current"]
    assert "--write" not in ledger_step and "trap" not in ledger_step
    assert '  gate "refusal ledger" "$test_python" tests/test_refusal_classes.py --check\n' in WRAPPER


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *arguments],
        check=True, capture_output=True, text=True, timeout=60).stdout


def _ledger_checkout(tmp_path: Path, *, stale: bool) -> tuple[Path, Path]:
    """A committed scratch checkout: the ledger tool, a ledger, and this tree's src through a link."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    checkout = tmp_path / "checkout"
    (checkout / "tests").mkdir(parents=True)
    shutil.copy(ROOT / "tests" / "test_refusal_classes.py", checkout / "tests")
    text = (ROOT / "tests" / "refusal_class_ledger.json").read_text(encoding="utf-8")
    if stale:
        assert '"waivable": false' in text
        text = text.replace('"waivable": false', '"waivable": true', 1)
    ledger = checkout / "tests" / "refusal_class_ledger.json"
    ledger.write_text(text, encoding="utf-8")
    (checkout / "src").symlink_to(ROOT / "src", target_is_directory=True)
    _git(checkout, "init", "-q")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-q", "-m", "a checkout with a ledger")
    return checkout, ledger


def _ledger_check(checkout: Path) -> subprocess.Popen:
    """Start the command the wrapper's gate and CI's step both run."""
    return subprocess.Popen(
        [sys.executable, "tests/test_refusal_classes.py", "--check"], cwd=checkout,
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _as_written(ledger: Path) -> tuple[bytes, int, int]:
    """Its bytes, and two marks that move whenever anything writes or replaces the file."""
    status = ledger.stat()
    return ledger.read_bytes(), status.st_ino, status.st_mtime_ns


@pytest.mark.parametrize(("stale", "status"), [(False, 0), (True, 1)], ids=["current", "stale"])
def test_the_ledger_check_never_writes_the_ledger_and_names_what_a_stale_one_differs_in(tmp_path, stale, status):
    checkout, ledger = _ledger_checkout(tmp_path, stale=stale)
    as_found = _as_written(ledger)
    process = _ledger_check(checkout)
    output, _ = process.communicate(timeout=300)
    assert process.returncode == status, output
    if stale:
        assert '-      "waivable": true\n+      "waivable": false\n' in output, output
        assert "tests/test_refusal_classes.py --write" in output
    else:
        assert "refusal ledger is current" in output
    assert _as_written(ledger) == as_found
    assert _git(checkout, "status", "--porcelain") == ""


@pytest.mark.parametrize(("stale", "status"), [(False, 0), (True, 1)], ids=["current", "stale"])
def test_two_ledger_checks_at_once_in_one_checkout_both_answer_right_and_leave_it_clean(tmp_path, stale, status):
    checkout, ledger = _ledger_checkout(tmp_path, stale=stale)
    as_found = _as_written(ledger)
    both = [_ledger_check(checkout), _ledger_check(checkout)]
    outputs = [process.communicate(timeout=300)[0] for process in both]
    assert [process.returncode for process in both] == [status, status], outputs
    assert _as_written(ledger) == as_found
    assert _git(checkout, "status", "--porcelain") == ""
