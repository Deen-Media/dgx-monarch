"""The INT8 GEMM kernel's validated comfy-kitchen set against what an install gets.

The Triton kernel runs only on the comfy-kitchen releases it was proven exact
against (``VALIDATED_KITCHEN`` in src/dgx_monarch/kernels/int8_gemm_triton.py);
any other release keeps stock CUTLASS and logs one WARNING. comfy-kitchen
reaches an environment through ComfyUI's requirements.txt, which
scripts/setup_env.sh and the comfy canary both install; dgx-monarch declares
no requirement of its own. A new pin there would turn the kernel off on every
box that follows it, with nothing but a log line to say so. These tests fail
instead, so the set is re-validated (docs/MODELS.md, "INT8 GEMM kernel").

The comfy canary runs this file against ComfyUI master and the newest ComfyUI
release, where the requirements and the installed package both exist. Ordinary
CI has neither, so there only the documentation check runs and the other two
skip with their reasons; on a rig the installed-package check runs too.
"""
from __future__ import annotations

import importlib.metadata
import os
import re
from pathlib import Path

import pytest

from dgx_monarch.kernels.int8_gemm_triton import VALIDATED_KITCHEN

REPO = Path(__file__).resolve().parents[1]
_REQUIREMENT = re.compile(r"^[ \t]*comfy[-_.]kitchen\b(?P<spec>[^#;\n]*)", re.IGNORECASE | re.MULTILINE)
_PIN = re.compile(r"^\s*==\s*(?P<version>[0-9][0-9A-Za-z.+-]*)\s*$")


def _comfy_checkout() -> Path | None:
    for name in ("COMFYUI_DIR", "COMFY_DIR"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def _spoken(versions: tuple[str, ...]) -> str:
    return versions[0] if len(versions) == 1 else ", ".join(versions[:-1]) + " and " + versions[-1]


def _requirements_problem(text: str) -> str | None:
    """What is wrong with the comfy-kitchen requirement in a requirements.txt, or None."""
    found = [match.group("spec").strip() for match in _REQUIREMENT.finditer(text)]
    if not found:
        return ("ComfyUI's requirements.txt no longer names comfy-kitchen, so this test no longer "
                "sees the release an install gets; re-anchor it where comfy-kitchen now comes from")
    if len(found) > 1:
        return f"ComfyUI's requirements.txt names comfy-kitchen {len(found)} times: {found}"
    pin = _PIN.match(found[0])
    if pin is None:
        return (f"ComfyUI's requirements.txt asks for comfy-kitchen{found[0]}, not one exact release, "
                "so the release an install gets is open; check the installed one on every box")
    if pin.group("version") not in VALIDATED_KITCHEN:
        return (f"ComfyUI pins comfy-kitchen {pin.group('version')}, outside the INT8 GEMM kernel's "
                f"validated set ({', '.join(VALIDATED_KITCHEN)}). Every box that installs it runs stock "
                "CUTLASS. Re-validate the kernel on that release (docs/MODELS.md, INT8 GEMM kernel) "
                "before adding it")
    return None


def test_comfyuis_requirements_pin_a_validated_comfy_kitchen():
    checkout = _comfy_checkout()
    if checkout is None or not (checkout / "requirements.txt").is_file():
        pytest.skip("no ComfyUI checkout named by COMFYUI_DIR or COMFY_DIR")
    problem = _requirements_problem((checkout / "requirements.txt").read_text(encoding="utf-8"))
    assert problem is None, problem


@pytest.mark.parametrize("text, words", [
    ("torch\ncomfy-kitchen==0.2.40\nsafetensors>=0.4.2\n", None),
    ("comfy_kitchen == 0.2.37  # pinned\n", None),
    ("comfy-kitchen==0.2.41\n", "pins comfy-kitchen 0.2.41, outside"),
    ("comfy-kitchen>=0.2.40\n", "not one exact release"),
    ("comfy-kitchen\n", "not one exact release"),
    ("torch\nnumpy\n", "no longer names comfy-kitchen"),
    ("comfy-kitchen==0.2.40\ncomfy-kitchen==0.2.41\n", "names comfy-kitchen 2 times"),
    ("# comfy-kitchen==0.2.41\ncomfy-kitchen==0.2.40\n", None),
])
def test_the_requirements_check_reads_one_exact_pin(text, words):
    """The canary's parser, on requirement lines it can meet, in ordinary CI too."""
    problem = _requirements_problem(text)
    assert (problem is None) if words is None else (words in str(problem)), problem


def test_the_installed_comfy_kitchen_is_validated():
    try:
        version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("comfy-kitchen is not installed in this environment")
    assert version in VALIDATED_KITCHEN, (
        f"comfy-kitchen {version} is installed here, outside the INT8 GEMM kernel's validated set "
        f"({', '.join(VALIDATED_KITCHEN)}), so this environment runs stock CUTLASS. Re-validate the "
        "kernel on that release (docs/MODELS.md, INT8 GEMM kernel) before adding it")


def test_the_docs_name_the_validated_set():
    models = " ".join((REPO / "docs" / "MODELS.md").read_text(encoding="utf-8").split())
    assert f"The validated comfy-kitchen releases are {_spoken(VALIDATED_KITCHEN)}." in models
