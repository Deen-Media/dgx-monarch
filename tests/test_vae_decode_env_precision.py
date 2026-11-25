"""Adopting the driver's float32 matmul precision keeps torch's state readable.

``torch.backends.cuda.matmul.allow_tf32`` is a legacy alias of the precision
that ``torch.set_float32_matmul_precision`` sets. On torch 2.12 and later a
write to the alias after the new setter, even one that writes the value the
alias already reads, leaves the backends mixed, and
``torch.get_float32_matmul_precision()`` then raises. The adoption writes the
precision last, both when it adopts the driver's flags and when it puts this
host's back, and reads every adopted flag back.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from dgx_monarch import vae_decode_env

PRECISIONS = ("highest", "high", "medium")


class _NoModules:
    def modules(self):
        return []


@pytest.fixture(autouse=True)
def _restore_torch_precision():
    """Leave the process as it started: these flags are process-global."""
    cudnn = torch.backends.cudnn.allow_tf32
    yield
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = cudnn


def _flags_at(precision: str) -> dict:
    torch.set_float32_matmul_precision(precision)
    return vae_decode_env.backend_flags()


@pytest.mark.parametrize("here,driver", list(itertools.product(PRECISIONS, PRECISIONS)))
def test_adoption_reads_back_the_drivers_precision_and_restores_this_hosts(here, driver):
    driver_flags = _flags_at(driver)
    before = _flags_at(here)
    with vae_decode_env.adopted(_NoModules(), {"flags": driver_flags}):
        # backend_flags() calls the getter, which raises on a mixed state.
        assert vae_decode_env.backend_flags() == driver_flags
    assert vae_decode_env.backend_flags() == before
    assert torch.get_float32_matmul_precision() == here


def test_the_precision_moves_with_another_flag_in_both_directions():
    """A second differing flag beside the precision changes neither order."""
    driver_flags = dict(_flags_at("medium"), **{"cudnn.allow_tf32": False})
    torch.backends.cudnn.allow_tf32 = True
    before = _flags_at("highest")
    with vae_decode_env.adopted(_NoModules(), {"flags": driver_flags}):
        assert vae_decode_env.backend_flags() == driver_flags
    assert vae_decode_env.backend_flags() == before


def test_a_flag_this_host_cannot_hold_is_a_decline_and_restores(monkeypatch):
    """Readback is the proof: a flag that does not take is a decline, so the
    driver decodes the unit, and the host's own flags come back first."""
    torch.backends.cudnn.allow_tf32 = True
    before = _flags_at("highest")
    driver_flags = dict(_flags_at("high"), **{"cudnn.allow_tf32": False})
    torch.set_float32_matmul_precision("highest")
    real_set = vae_decode_env._set_flag

    def ignore_cudnn_tf32(name, value):
        if name != "cudnn.allow_tf32":
            real_set(name, value)

    monkeypatch.setattr(vae_decode_env, "_set_flag", ignore_cudnn_tf32)
    with pytest.raises(vae_decode_env.Decline, match=r"cudnn\.allow_tf32"):
        with vae_decode_env.adopted(_NoModules(), {"flags": driver_flags}):
            pytest.fail("the decode ran under flags that did not take")
    assert vae_decode_env.backend_flags() == before
