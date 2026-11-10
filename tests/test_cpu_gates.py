"""scripts/test_cpu.sh runs every gate CI's lint-and-unit job runs that can run off CI.

A local green run used to leave CI's gates unchecked: the protected test
inventory, the refusal ledger, the TRUST citations, the leak check, ruff, mypy
and the generator checks ran only as CI steps, so a contributor learned of
them after a paid push. The wrapper now runs them before the suite. Each CI
step is classified here, so a step CI gains must be added to the wrapper or
named as CI-only, with the reason.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (ROOT / "scripts" / "test_cpu.sh").read_text(encoding="utf-8")
CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))

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
    "Validate wheel installability and source completeness": (
        "builds the wheel and sdist into dist/ and two virtual environments, and writes a "
        "sentinel into .claude/worktrees; it changes the checkout it checks"),
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
