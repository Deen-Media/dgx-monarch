"""The update tool's dependency contract for the pinned torchmonarch.

Both update modes read the candidate wheel's own METADATA into a contract and
check its closure on every interpreter before they change anything, then
install the pin with ``--no-deps``. torchmonarch 0.8.0 added two unconditional
requirements, ``nvidia-ml-py`` and ``psutil``, that 0.7.0 did not declare, so a
host without them must be refused by name rather than updated into a runtime
that pip check flags.
"""
from __future__ import annotations

import importlib.metadata as metadata

import pytest
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name

from dgx_monarch import TORCHMONARCH_PIN
from dgx_monarch.cli.update_dependency_contract import check_contract, dependency_contract

ADDED_IN_080 = {"nvidia-ml-py", "psutil"}


def _pinned_metadata() -> bytes:
    try:
        distribution = metadata.distribution("torchmonarch")
    except metadata.PackageNotFoundError:
        pytest.skip("torchmonarch is not installed")
    if distribution.version != TORCHMONARCH_PIN:
        pytest.skip(f"installed torchmonarch {distribution.version} is not the pin {TORCHMONARCH_PIN}")
    text = distribution.read_text("METADATA")
    assert text is not None
    return text.encode()


def test_the_pinned_wheel_requires_nvidia_ml_py_and_psutil_unconditionally():
    contract = dependency_contract(_pinned_metadata(), TORCHMONARCH_PIN, ValueError)
    unconditional = {canonicalize_name(Requirement(raw).name)
                     for raw in contract["requirements"] if Requirement(raw).marker is None}
    assert ADDED_IN_080 <= unconditional


def test_the_closure_check_names_each_missing_one(monkeypatch):
    contract = dependency_contract(_pinned_metadata(), TORCHMONARCH_PIN, ValueError)
    real = metadata.distribution

    def without_added(name):
        if canonicalize_name(name) in ADDED_IN_080:
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "distribution", without_added)
    errors = check_contract(contract, [f"torchmonarch=={TORCHMONARCH_PIN}"])
    assert {"nvidia-ml-py: missing", "psutil: missing"} <= set(errors), errors
