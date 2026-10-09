"""Installability and source-completeness contracts."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
# CI reruns this file from the extracted sdist. Keep the complete committed
# benchmark surface explicit so neither omissions nor workstation output ship.
COMMITTED_BENCHMARK_FILES = {
    "benchmark/chroma_numerics_probe.py",
    "benchmark/flux2_cross_topology_matrix.toml",
    "benchmark/gates.py",
    "benchmark/matrix.example.toml",
    "benchmark/rdma_return_bench.py",
    "benchmark/reports/chroma_cfg_amplification_matrix.toml",
    "benchmark/reports/chroma_pad_fidelity_matrix.toml",
    "benchmark/reports/flux_family_pad_fidelity_matrix.toml",
    "benchmark/reports/followon_2026-09_cells.csv.gz",
    "benchmark/reports/pad_fidelity_matrix.toml",
    "benchmark/reports/stable_source_2026-09_cells.csv.gz",
    "benchmark/reports/sweep349_2026-09_cells.csv.gz",
    "benchmark/reports/wan_i2v_fidelity_matrix.toml",
    "benchmark/run_matrix.py",
    "benchmark/sweep/__init__.py",
    "benchmark/sweep/compare.py",
    "benchmark/sweep/convert.py",
    "benchmark/sweep/driver.py",
    "benchmark/sweep/matrix.py",
    "benchmark/sweep/media.py",
    "benchmark/sweep/memhog.py",
    "benchmark/sweep/memsample.py",
    "benchmark/sweep/report.py",
    "benchmark/sweep/run.py",
    "benchmark/sweep/sweep.example.toml",
}
IGNORED_BENCHMARK_OUTPUTS = {
    "benchmark/matrix.toml",
    "benchmark/results.json",
}


def _assert_extracted_sdist_benchmark_membership(repo: Path) -> None:
    # A Git checkout may hold ignored operator inputs and results. There,
    # MANIFEST.in is the packaging contract; only an extracted sdist must match
    # the allowlist file for file on disk.
    if (repo / ".git").exists():
        return

    shipped_benchmark = {
        path.relative_to(repo).as_posix()
        for path in (repo / "benchmark").rglob("*")
        if path.is_file() and path.suffix in {".gz", ".json", ".py", ".toml"}
    }
    assert shipped_benchmark == COMMITTED_BENCHMARK_FILES, (
        "sdist benchmark membership drift: "
        f"missing={sorted(COMMITTED_BENCHMARK_FILES - shipped_benchmark)}, "
        f"extra={sorted(shipped_benchmark - COMMITTED_BENCHMARK_FILES)}"
    )


def _write_benchmark_fixture(repo: Path, relative_paths: set[str]) -> None:
    for relative_path in relative_paths:
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_changelog_has_a_clean_1_0_boundary():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(
        r"^## (?:1\.0\.0|\[1\.0\.0\])(?: - \d{4}-\d{2}-\d{2})?$",
        changelog, flags=re.MULTILINE,
    )
    assert not re.search(r"^## \[?0\.", changelog, flags=re.MULTILINE)
    assert not re.search(r"^## Path to 1\.0", changelog, flags=re.MULTILINE)
    assert "Initial release of DGX Monarch" in changelog
    for target in (
        "skills/dgx-monarch/SKILL.md",
        "README.md#set-up-with-a-coding-agent",
        "docs/QUICKSTART.md#first-distributed-render",
        "docs/VALIDATION.md#fresh-agent-installation",
    ):
        assert f"]({target})" in changelog, target
        assert (REPO / target.split("#", 1)[0]).is_file(), target


def test_first_use_gate_latency_and_release_invalidation_are_prominent():
    for relative_path in ("README.md", "docs/TROUBLESHOOTING.md"):
        text = (REPO / relative_path).read_text()
        normalized = " ".join(
            line.lstrip("> ").strip() for line in text.splitlines()
        )
        assert "auto_gate=first_use" in normalized
        assert "2-4 minutes" in normalized
        assert re.search(r"before (?:(?:the |showing )?first (?:pixels|render)|producing your result)", normalized)
        assert re.search(r"(?:an upgrade requires new checks before those results can be reused|Each combination is checked again on first use after an upgrade or source change)", normalized)
        assert "Fleet" in normalized
        assert "slab" in normalized and "low-RSS" in normalized


def test_github_actions_are_pinned_to_immutable_commits():
    uses = []
    for workflow in (REPO / ".github" / "workflows").glob("*.yml"):
        for line_number, line in enumerate(workflow.read_text().splitlines(), 1):
            match = re.search(r"\buses:\s*([^\s#]+)", line)
            if match:
                uses.append((workflow.name, line_number, match.group(1)))
    assert uses
    mutable = [entry for entry in uses if not re.search(r"@[0-9a-f]{40}$", entry[2])]
    assert not mutable, f"actions must use immutable commit SHAs: {mutable}"


def test_ci_push_trigger_covers_private_and_public_default_branches():
    workflow_path = REPO / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    triggers = workflow.get("on", workflow.get(True))

    assert isinstance(triggers, dict)
    assert set(triggers["push"]["branches"]) == {"master", "main"}


def test_public_repository_links_follow_the_remote_default_branch():
    assert "/tree/HEAD/docs" in (REPO / "pyproject.toml").read_text()
    assert "../blob/HEAD/SECURITY.md" in (
        REPO / ".github" / "ISSUE_TEMPLATE" / "bug-report.yml"
    ).read_text()


# Import gates whose package stays out of the dev extra, each with its reason.
# Any `comfy.*` gate joins them: ComfyUI is never a dependency, the host
# application owns it, so those suites are expected to skip off a rig.
GATES_OUTSIDE_THE_DEV_EXTRA = {
    "monarch.actor": "torchmonarch is a runtime dependency, declared above",
    "numpy": "torch brings it; a direct pin would only drift",
    "sol_attn.interface": (
        "sol-attn is an optional lever, never a wheel requirement; the two "
        "tests behind this gate assert the CuTe dispatch entry lands in the "
        "real table, and every other sol test runs without the package"),
    "comfy_kitchen.tensor": (
        "comfy-kitchen ships with ComfyUI, which the host application owns; "
        "the quantized-shard suite behind this gate runs on a rig venv, and "
        "the wrapper's structural ops are exercised there against the real "
        "QuantizedTensor rather than a stand-in"),
    "comfy.model_base": (
        "ComfyUI itself, which the host application owns; the render-memory "
        "canary behind this gate pins the mirrored formula against comfy's "
        "own memory_required on a rig where comfy resolves"),
    "comfy.model_management": (
        "ComfyUI itself; the same render-memory canary reads the flash-path "
        "switch it mirrors"),
}
# `importorskip` names the module; the extra names the distribution that ships it.
IMPORT_NAME_TO_DISTRIBUTION = {"yaml": "pyyaml", "PIL": "pillow"}


def test_dev_extra_declares_every_import_gated_test_dependency():
    """CI installs pinned torch, then `-e .[dev]`, and no other test package. A
    suite that guards its import with `pytest.importorskip` skips in silence
    when the package is missing, so every gate is either declared in the dev
    extra or named in GATES_OUTSIDE_THE_DEV_EXTRA with its reason."""
    dev = re.search(r"^dev = \[(.*?)\]$", (REPO / "pyproject.toml").read_text(),
                    re.MULTILINE | re.DOTALL)
    assert dev, "the dev extra moved; this contract follows it"
    declared = set(re.findall(r'"([A-Za-z0-9_.-]+)', dev.group(1)))

    gated = set()
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        gated.update(
            IMPORT_NAME_TO_DISTRIBUTION.get(name, name)
            for name in re.findall(r'importorskip\(\s*"([A-Za-z0-9_.-]+)"',
                                   path.read_text())
        )
    assert {"aiohttp", "rich"} <= gated, (
        "the HTTP and TUI suites lost their import gates; if they now import "
        "unconditionally, drop them from this contract"
    )
    unaccounted = {
        name for name in gated - declared - set(GATES_OUTSIDE_THE_DEV_EXTRA)
        if not name.startswith("comfy.")
    }
    assert not unaccounted, (
        f"import-gated test dependencies neither declared in the dev extra nor "
        f"listed as deliberately absent: {sorted(unaccounted)}. Undeclared, "
        "those suites skip in CI and the run still reports green."
    )


def test_sdist_manifest_carries_every_repository_surface():
    manifest = (REPO / "MANIFEST.in").read_text()
    expected = {
        "include __init__.py",
        "include .gitignore",
        "include LICENSE",
        "include NOTICE",
        "include THIRD-PARTY-NOTICES.txt",
        "include SECURITY.md",
        "recursive-include .github *.md *.yml",
        "recursive-include docs *",
        "recursive-include example_workflows *.jpg *.json",
        "include tests/fixtures/workflows/README.md",
        "recursive-include scripts *.md *.sh",
        "recursive-include tests *.json *.py",
        "recursive-include tools *.py",
        "recursive-include tools *.mjs",
        "include tools/leak_allowlist.toml",
        "recursive-include web *.js *.svg",
    } | {f"include {path}" for path in COMMITTED_BENCHMARK_FILES}
    assert expected <= set(manifest.splitlines())


def test_sdist_includes_only_the_two_reviewed_skills():
    manifest = set((REPO / "MANIFEST.in").read_text().splitlines())
    included = {
        line.removeprefix("include ")
        for line in manifest
        if line.startswith("include skills/")
    }
    assert included == {
        "skills/dgx-monarch/SKILL.md",
        "skills/dgx-monarch-dev/SKILL.md",
    }
    assert "prune .claude/worktrees" in manifest
    assert not any(line.startswith("recursive-include .claude") for line in manifest)


def test_sdist_limits_benchmark_to_committed_files():
    manifest = set((REPO / "MANIFEST.in").read_text().splitlines())
    broad_benchmark_rules = {
        line for line in manifest
        if line.startswith("recursive-include benchmark ")
        or line.startswith("recursive-include benchmark/")
    }
    assert not broad_benchmark_rules, (
        "recursive benchmark rules can include ignored or untracked local files: "
        f"{sorted(broad_benchmark_rules)}"
    )
    included_benchmark = {
        line.removeprefix("include ")
        for line in manifest
        if line.startswith("include benchmark/")
    }
    assert included_benchmark == COMMITTED_BENCHMARK_FILES, (
        "benchmark manifest allowlist drift: "
        f"missing={sorted(COMMITTED_BENCHMARK_FILES - included_benchmark)}, "
        f"extra={sorted(included_benchmark - COMMITTED_BENCHMARK_FILES)}"
    )
    assert IGNORED_BENCHMARK_OUTPUTS <= set(
        (REPO / ".gitignore").read_text().splitlines()
    )
    assert IGNORED_BENCHMARK_OUTPUTS.isdisjoint(included_benchmark)


def test_checkout_benchmark_allowlist_matches_git_index():
    """A tracked benchmark cannot disappear from both packaging allowlists."""
    import os
    import subprocess

    import pytest

    if not (REPO / ".git").exists():
        pytest.skip("Git index is unavailable in an extracted sdist")
    output = subprocess.check_output([
        "git", "-C", os.fspath(REPO), "ls-files", "--cached", "-z", "--",
        "benchmark",
    ])
    tracked = {os.fsdecode(path) for path in output.split(b"\0") if path}
    assert tracked == COMMITTED_BENCHMARK_FILES, (
        "benchmark allowlist differs from Git index: "
        f"missing_from_allowlist={sorted(tracked - COMMITTED_BENCHMARK_FILES)}, "
        f"not_tracked={sorted(COMMITTED_BENCHMARK_FILES - tracked)}"
    )


def test_sdist_benchmark_membership_matches_committed_allowlist():
    """The extracted archive carries all and only reviewed benchmark inputs."""
    _assert_extracted_sdist_benchmark_membership(REPO)


def test_benchmark_membership_distinguishes_checkout_outputs_from_archive_leaks(
    tmp_path,
):
    import pytest

    checkout = tmp_path / "checkout"
    _write_benchmark_fixture(
        checkout,
        COMMITTED_BENCHMARK_FILES | IGNORED_BENCHMARK_OUTPUTS,
    )
    (checkout / ".git").mkdir()
    _assert_extracted_sdist_benchmark_membership(checkout)

    extracted_sdist = tmp_path / "extracted-sdist"
    _write_benchmark_fixture(
        extracted_sdist,
        COMMITTED_BENCHMARK_FILES | IGNORED_BENCHMARK_OUTPUTS,
    )
    with pytest.raises(AssertionError, match="sdist benchmark membership drift"):
        _assert_extracted_sdist_benchmark_membership(extracted_sdist)


def test_ci_runs_the_committed_worktree_leak_check():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "python tools/leak_check.py" in workflow
    assert workflow.index("python tools/leak_check.py") < workflow.index("Install shellcheck")
    assert (REPO / "tools" / "leak_check.py").is_file()
    assert (REPO / "tools" / "leak_allowlist.toml").is_file()


def test_ci_lints_and_compiles_the_root_custom_node_entrypoint():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "ruff check __init__.py src tests benchmark tools" in workflow
    assert (
        "python -m compileall -q __init__.py src benchmark tests tools" in workflow
    )


def test_ci_requires_exact_sdist_and_python_module_membership():
    workflow_path = REPO / ".github" / "workflows" / "ci.yml"
    workflow = workflow_path.read_text()

    assert "assert members == expected_members" in workflow
    assert "assert source_modules == sdist_modules == wheel_modules" in workflow
    assert '"LICENSE"' in workflow

    parsed = yaml.safe_load(workflow)
    steps = parsed["jobs"]["lint-and-unit"]["steps"]
    run = next(
        step["run"]
        for step in steps
        if step.get("name") == "Validate wheel installability and source completeness"
    )
    inline = run.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(inline, str(workflow_path), "exec")
    assert ".claude/worktrees/sdist-ignore-sentinel.md" in workflow
    assert "ignored Claude worktree content leaked into sdist" in workflow


def test_repository_tool_caches_are_ignored():
    ignored = set((REPO / ".gitignore").read_text().splitlines())
    assert {".venv/", ".mypy_cache/", ".hypothesis/"} <= ignored


def test_ci_uses_production_torch_build_for_fake_cuda_contracts():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    setup = (REPO / "scripts" / "setup_env.sh").read_text()
    requirement = "torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132"

    assert f"pip install --no-cache-dir {requirement}" in workflow
    assert requirement in setup
    assert '("2.12.0+cu132", "13.2", False)' in workflow
    assert "python -m pip check" in workflow
    assert "Install (pinned CUDA-linked torch + pinned monarch)" in workflow
    assert "pytest (no-GPU unit suite)" in workflow
    assert "/whl/cpu" not in workflow


def test_comfy_canary_covers_entrypoint_package_and_browser_paths():
    workflow = (REPO / ".github" / "workflows" / "comfy-canary.yml").read_text()
    assert 'python tests/canary/comfy_entrypoint_canary.py' in workflow
    assert "assert Comfy touchpoint manifest" in workflow
    canary = (REPO / "tests" / "canary" / "comfy_entrypoint_canary.py").read_text()
    assert "assert_comfy_surface()" in canary
    assert "_headless_prompt_server(server.PromptServer, web.RouteTableDef)" in canary
    assert "object.__new__(prompt_server_type)" in canary
    assert "route_table_def()" in canary
    assert '"dgx_monarch_segments.js"' in canary
    assert '"dgx_monarch_readiness_model.js"' in canary
    for path_filter in ('"__init__.py"', '"src/**"', '"web/**"'):
        assert path_filter in workflow


def test_comfy_canary_runs_the_seam_behavioral_contracts():
    # Signatures are the import canary's job. This step is the behavioral half:
    # without it, a change behind an unchanged signature reaches a render first
    # (docs/DESIGN.md 5.10).
    workflow = (REPO / ".github" / "workflows" / "comfy-canary.yml").read_text()
    assert "python tests/canary/comfy_seam_contracts.py" in workflow
    assert "Seam behavioral contracts vs comfy master (CPU)" in workflow
    trigger_block = workflow.split("jobs:", 1)[0]
    assert "schedule:" in trigger_block, (
        "the seam suite only earns its keep on a schedule; a pull-request-only "
        "run cannot page on upstream drift"
    )
    suite = REPO / "tests" / "canary" / "comfy_seam_contracts.py"
    assert suite.is_file()
    # A seam that reads a repo module outside src is only guarded on a pull
    # request if that module is a trigger too. The media load inputs seam
    # imports the sweep matrix, so a change to the map it holds must run this job.
    for imported in ("benchmark.sweep.matrix",):
        if imported in suite.read_text():
            path_filter = '"' + imported.replace(".", "/") + '.py"'
            assert path_filter in trigger_block, (
                f"{imported} feeds a seam but does not trigger the canary")
    assert "tests/canary/comfy_seam_contracts.py" in (
        REPO / ".github" / "workflows" / "ci.yml"
    ).read_text(), "the seam suite must ship in the reviewable source distribution"


def test_public_source_has_no_hardware_route():
    workflows = REPO / ".github" / "workflows"
    assert not (workflows / "hardware-smoke.yml").exists()
    assert not (workflows / "hardware-smoke.yaml").exists()
    workflow_paths = (*workflows.glob("*.yml"), *workflows.glob("*.yaml"))
    assert workflow_paths, "no workflows found; the route guard would pass vacuously"
    for workflow_path in workflow_paths:
        workflow = yaml.safe_load(workflow_path.read_text())
        assert isinstance(workflow, dict)
        jobs = workflow.get("jobs")
        assert isinstance(jobs, dict) and jobs, workflow_path
        for job in jobs.values():
            assert job.get("runs-on") == "ubuntu-24.04"
            assert "uses" not in job
        workflow_text = workflow_path.read_text()
        assert "self-hosted" not in workflow_text
        assert "dgxm-hw-" not in workflow_text
        assert "generate-jitconfig" not in workflow_text
    ci_text = (workflows / "ci.yml").read_text()
    assert "retired validation/ tree reappeared in sdist" in ci_text


def test_torchmonarch_canary_is_read_only_artifacted_and_never_auto_bumps():
    workflow_path = REPO / ".github" / "workflows" / "torchmonarch-canary.yml"
    workflow = workflow_path.read_text()
    trigger_block = workflow.split("jobs:", 1)[0]
    assert "schedule:" in trigger_block
    assert 'cron: "29 7 * * 1"' in trigger_block
    assert "workflow_dispatch:" in trigger_block
    assert "pull_request:" not in trigger_block
    assert "contents: read" in trigger_block
    assert "persist-credentials: false" in workflow
    assert '"torchmonarch==$PIN"' in workflow
    assert 'pip install --upgrade torchmonarch' in workflow
    assert "--pre" not in workflow
    assert workflow.count("dgxm-torchmonarch-canary") >= 2
    assert "tests/fixtures/torchmonarch_pin_0_6_0.json" in workflow
    assert "snapshot --output \"$ARTIFACTS/pin.json\"" in workflow
    assert "snapshot --output \"$ARTIFACTS/latest.json\"" in workflow
    assert "latest-vs-pin-report.md" in workflow
    assert workflow.count("if: always()") >= 7
    assert "Ensure failure diagnostics are artifactable" in workflow
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow
    assert workflow.index("Upload pin/latest snapshots and reports") < workflow.index(
        "Final compatibility and semantic gate"
    )
    assert "git push" not in workflow
    assert "gh pr" not in workflow
    assert "pyproject.toml" in workflow  # read to resolve the source-of-truth pin
    assert (REPO / "src" / "dgx_monarch" / "monarch_semantics.py").is_file()
    assert (REPO / "src" / "dgx_monarch" / "monarch_surface.py").is_file()
    assert (REPO / "src" / "dgx_monarch" / "monarch_surface_report.py").is_file()
    assert (REPO / "tests" / "canary" / "monarch_surface_canary.py").is_file()
    assert (REPO / "tests" / "fixtures" / "torchmonarch_pin_0_6_0.json").is_file()


def test_torchmonarch_probe_surfaces_are_carried_by_sdist_patterns():
    manifest = set((REPO / "MANIFEST.in").read_text().splitlines())
    assert "recursive-include .github *.md *.yml" in manifest
    assert "recursive-include tests *.json *.py" in manifest


def test_header_accepts_alignment_padded_layouts(tmp_path):
    """Producers may pad between and after tensors; overlap stays a hard error.
    Strict contiguity would refuse legal files outright, because a structural
    header error never falls back to the stock loader."""
    import json
    import struct

    import pytest

    from dgx_monarch.safetensors_header import (
        SafetensorsHeaderError,
        read_safetensors_header,
    )

    def write(path, header, payload):
        blob = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)

    padded = tmp_path / "padded.safetensors"
    write(padded, {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        # 8-byte alignment hole before b, 4 trailing pad bytes after it
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]},
    }, bytes(28))
    parsed = read_safetensors_header(str(padded))
    assert set(parsed.tensors) == {"a", "b"}

    overlapping = tmp_path / "overlap.safetensors"
    write(overlapping, {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
    }, bytes(12))
    with pytest.raises(SafetensorsHeaderError, match="overlaps"):
        read_safetensors_header(str(overlapping))


def test_extracted_sdist_pytest_run_covers_the_surface_contracts():
    """The extracted-sdist pytest run executes these contracts, not only ships
    them: a merge of two CI edits can drop a file from the run while it stays
    in the archive-member set."""
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    sdist_run = workflow.split('[sys.executable, "-m", "pytest", "-q",', 1)[1]
    sdist_run = sdist_run.split("]", 1)[0]
    for required in (
        "tests/test_comfy_surface.py",
        "tests/test_monarch_surface.py",
        "tests/test_leak_check.py",
        "tests/test_node_contract.py",
        "tests/test_benchmark_provenance.py",
        "tests/test_release_surfaces.py",
    ):
        assert required in sdist_run, f"sdist run lost {required}"
