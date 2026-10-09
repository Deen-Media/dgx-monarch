"""Public documentation contracts for the source-checkout release."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    REPO / "README.md",
    REPO / "docs" / "INSTALL.md",
    REPO / "docs" / "QUICKSTART.md",
    REPO / "docs" / "FAQ.md",
    REPO / "docs" / "CONCEPTS.md",
    REPO / "docs" / "CLUSTER.md",
    REPO / "docs" / "MODELS.md",
    REPO / "docs" / "ADAPTERS.md",
    REPO / "docs" / "TUI.md",
    REPO / "docs" / "VALIDATION.md",
    REPO / "docs" / "BENCHMARKS.md",
)
AI_INSTRUCTIONS = (
    REPO / "skills" / "dgx-monarch" / "SKILL.md",
    REPO / "skills" / "dgx-monarch-dev" / "SKILL.md",
)
INSTALL_SURFACES = (
    REPO / "docs" / "INSTALL.md",
    REPO / "skills" / "dgx-monarch" / "SKILL.md",
)
PREREQUISITE_SURFACES = (
    *INSTALL_SURFACES,
    REPO / "docs" / "QUICKSTART.md",
)
ALL_RELEASE_SURFACES = (*PUBLIC_DOCS, *AI_INSTRUCTIONS, REPO / "requirements.txt")
OFFICIAL_COMFY_INSTALL = "https://docs.comfy.org/installation/manual_install"
FORBIDDEN_AVAILABILITY_CLAIMS = re.compile(
    r"comfyui[- ]manager|\bmanager installation\b|comfy-cli|\bpypi\b"
    r"|published\s+(?:wheel|sdist|source archive)"
    r"|installing\s+(?:a\s+|the\s+)?(?:wheel|sdist)"
    r"|registry\s+(?:publish|install|availability)|wheel-only\s+install",
    re.IGNORECASE,
)
PIP_INSTALL = re.compile(r"\bpip(?:3)?\s+install\b", re.IGNORECASE)
EM_DASH = chr(0x2014)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _bash_blocks(text: str) -> str:
    return "\n".join(re.findall(r"```(?:bash|sh)\n(.*?)```", text, flags=re.DOTALL))


def test_primary_install_surfaces_use_one_comfy_interpreter_and_source_checkout() -> None:
    install = _text(INSTALL_SURFACES[0])
    assert 'export COMFY_PYTHON="/absolute/path/to/python-used-by-ComfyUI"' in install
    assert '"$COMFY_PYTHON" -m pip install -e' in install
    assert '"$COMFY_DIR/custom_nodes/dgx-monarch"' in install
    skill = _text(INSTALL_SURFACES[1])
    assert "../../docs/INSTALL.md#install-from-source" in skill
    assert "COMFY_PYTHON" in skill and "COMFY_DIR" in skill
    assert '"$COMFY_PYTHON" -m pip install' not in _bash_blocks(skill)


def test_missing_comfy_prerequisite_points_to_official_manual_install() -> None:
    for path in PREREQUISITE_SURFACES:
        assert OFFICIAL_COMFY_INSTALL in _text(path), path


def test_public_release_surfaces_reject_old_install_claims() -> None:
    for path in ALL_RELEASE_SURFACES:
        text = _text(path)
        assert FORBIDDEN_AVAILABILITY_CLAIMS.search(text) is None, path
        assert EM_DASH not in text, path
        for line_number, line in enumerate(text.splitlines(), 1):
            if PIP_INSTALL.search(line):
                assert '"$COMFY_PYTHON" -m pip install' in line, (
                    f"{path}:{line_number}: install must use COMFY_PYTHON"
                )


def test_shell_examples_never_background_comfyui() -> None:
    for path in (*PUBLIC_DOCS, *AI_INSTRUCTIONS):
        text = _text(path)
        blocks = _bash_blocks(text)
        assert re.search(r"(?mi)^\s*nohup\s+.*comfy", blocks) is None, path
        assert re.search(r"(?mi)^\s*setsid\s+.*comfy", blocks) is None, path
        assert re.search(r"(?mi)^.*(?:comfy|main\.py).*(?:\s&|;\s*disown)\s*$", blocks) is None, path


def test_onboarding_explains_foreground_and_worker_service_lifecycle() -> None:
    surfaces = (
        REPO / "README.md",
        REPO / "docs" / "INSTALL.md",
        REPO / "docs" / "QUICKSTART.md",
        REPO / "skills" / "dgx-monarch" / "SKILL.md",
    )
    for path in surfaces:
        text = _text(path)
        assert re.search(r"\bforeground\b", text, re.IGNORECASE), path
        assert re.search(r"\bworker[- ]services?\b", text, re.IGNORECASE), path
        assert "dgxm status" in text, path


def test_operator_skill_reports_persistent_worker_services() -> None:
    text = _text(REPO / "skills" / "dgx-monarch" / "SKILL.md")
    assert "Never launch it with" in text
    assert "run\n`dgxm status` and tell the user exactly which services remain running" in text
    assert "Do not stop services merely\nbecause ComfyUI exits" in text


def test_removed_prompt_weight_feature_is_absent_from_public_model_docs() -> None:
    for path in (REPO / "README.md", REPO / "docs" / "MODELS.md"):
        text = _text(path).lower()
        assert "prompt weight" not in text, path
        assert "(word:weight)" not in text, path


def test_provenance_and_skill_boundaries_use_current_contracts() -> None:
    readme = _text(REPO / "README.md")
    assert "written from scratch" not in readme
    assert "repository does not vendor their code" in readme

    operator_skill = _text(
        REPO / "skills" / "dgx-monarch" / "SKILL.md"
    )
    assert "since 0.2.0" not in operator_skill
    assert "or a symlink to the working checkout" in operator_skill
    assert "Do not install from a second source tree" in " ".join(operator_skill.split())
    assert "The in-process attach retry is separate from `auto_heal`" in " ".join(operator_skill.split())

    dev_skill = _text(
        REPO / "skills" / "dgx-monarch-dev" / "SKILL.md"
    )
    assert "hardware-smoke.yml" not in dev_skill
    assert "Public-source CI" in dev_skill
    assert "separately reviewed private control repository" in dev_skill
    dev_skill_flat = " ".join(dev_skill.split())
    for phrase in (
        "never routes a job to protected hardware",
        "exact public source commit",
        "hand-run scripts",
        "No CI workflow, Actions runner, or scheduled job",
        "source manifest on every host",
        "explicit maintenance authorization",
        "report the hardware gate as **NOT RUN**",
        "Do not arm a source-repository hardware route",
    ):
        assert phrase in dev_skill_flat, phrase

    threat_model_flat = " ".join(
        _text(REPO / "docs" / "THREAT_MODEL.md").split()
    )
    for phrase in (
        "No executable hardware workflow may sit there",
        "generate-jitconfig",
    ):
        assert phrase in threat_model_flat, phrase


def test_source_archive_includes_repo_ai_instructions() -> None:
    manifest = _text(REPO / "MANIFEST.in").splitlines()
    assert {
        "include skills/dgx-monarch/SKILL.md",
        "include skills/dgx-monarch-dev/SKILL.md",
    } <= set(manifest)
    assert "prune .claude/worktrees" in manifest
    assert not any(line.startswith("recursive-include .claude") for line in manifest)


def test_operational_docs_have_existing_relative_link_targets() -> None:
    link = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
    for path in PUBLIC_DOCS:
        for raw_target in link.findall(_text(path)):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                continue
            resolved = (path.parent / unquote(target)).resolve()
            assert resolved.exists(), f"{path}: missing link target {raw_target!r}"
