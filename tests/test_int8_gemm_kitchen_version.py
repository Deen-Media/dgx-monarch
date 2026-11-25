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
import sys
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
    ("comfy-kitchen==0.2.42\n", "pins comfy-kitchen 0.2.42, outside"),
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


def _kitchen_module(name: str) -> bool:
    return name == "comfy_kitchen" or name.startswith("comfy_kitchen.")


@pytest.fixture
def kitchen_modules_put_back():
    """Take out of sys.modules whatever comfy-kitchen module the test imported."""
    before = {name for name in sys.modules if _kitchen_module(name)}
    try:
        yield
    finally:
        for name in [name for name in sys.modules if _kitchen_module(name) and name not in before]:
            del sys.modules[name]


@pytest.mark.parametrize("compiling", [False, True])
def test_the_compile_safe_wrapper_takes_the_installed_comfy_kitchens_own_calls(
        monkeypatch, kitchen_modules_put_back, compiling):
    """The installed comfy-kitchen's INT8 layout through the pack's ``int8_linear`` wrapper, on CPU.

    The wrapper stands in for ``comfy_kitchen.int8_linear`` wherever the Triton
    kernel is on, which is on a GPU only, so no CPU run ever met it. 0.2.42's
    layout began to pass ``weight_tiled`` on every ``linear_input_act`` call,
    and the wrapper of the day would have raised TypeError on each one, in
    every INT8 family with a fused activation. Here comfy-kitchen's own layout
    makes its own call through the wrapper, eager and as the custom op the
    wrapper takes while torch.compile traces, and the result must be stock's.
    The comfy canary installs each ComfyUI's comfy-kitchen and runs this file,
    so a release that changes the call fails here before any GPU sees it.
    """
    import torch

    comfy_kitchen = pytest.importorskip("comfy_kitchen", reason="comfy-kitchen is not installed here")
    from comfy_kitchen.tensor import QuantizedTensor

    from dgx_monarch.kernels import int8_gemm_triton as vendored

    stock_function = comfy_kitchen.int8_linear
    if getattr(stock_function, "_ck_compile_safe", False):
        pytest.skip("another test left the wrapper in place")
    generator = torch.Generator().manual_seed(42)
    weight = QuantizedTensor.from_float(
        (torch.randn(64, 256, generator=generator) * 0.02).to(torch.bfloat16),
        "TensorWiseINT8Layout", is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
    x = torch.randn(130, 256, generator=generator).to(torch.bfloat16)
    residual = torch.randn(130, 64, generator=generator).to(torch.bfloat16)
    scale = torch.ones(64, dtype=torch.bfloat16)
    cases = ({}, {"input_act": "swiglu"}, {"residual": residual, "residual_scale": scale})
    wide = torch.randn(130, 512, generator=generator).to(torch.bfloat16)

    def run(kwargs):
        return weight.linear_input_act(wide if kwargs.get("input_act") == "swiglu" else x, **kwargs)

    stock = [run(kwargs) for kwargs in cases]
    # The wrapper goes on for a validated release only; this checks the call, not the set.
    monkeypatch.setattr(vendored, "kitchen_version", lambda: vendored.VALIDATED_KITCHEN[-1])
    monkeypatch.setattr(comfy_kitchen, "int8_linear", stock_function)  # restored after the test
    assert vendored.compile_safe_int8_linear(comfy_kitchen) is True
    assert comfy_kitchen.int8_linear.__wrapped__ is stock_function
    if compiling:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    for kwargs, expected in zip(cases, stock, strict=True):
        assert torch.equal(run(kwargs), expected), sorted(kwargs)


def test_the_docs_name_the_validated_set():
    models = " ".join((REPO / "docs" / "MODELS.md").read_text(encoding="utf-8").split())
    assert f"The validated comfy-kitchen releases are {_spoken(VALIDATED_KITCHEN)}." in models
