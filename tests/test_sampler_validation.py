"""Worker-side sampler/scheduler name validation.

Stock comfy silently substitutes euler for unknown sampler names; on a worker
whose custom node pack failed to preload that is a silent fidelity loss
(res_2s rendering as euler). require_known_sampler refuses instead, with a
ValueError that names the likely cause and the fix. comfy is faked through
sys.modules because the unit suite runs without a ComfyUI checkout.
"""
import sys
import types

import pytest

from dgx_monarch.actor.sampling import require_known_sampler

STOCK_SAMPLERS = ["euler", "heun", "dpmpp_2m", "ddim", "uni_pc"]
STOCK_SCHEDULERS = ["normal", "simple", "karras"]


@pytest.fixture
def fake_comfy(monkeypatch):
    comfy_mod = types.ModuleType("comfy")
    samplers_mod = types.ModuleType("comfy.samplers")

    class KSampler:
        SAMPLERS = list(STOCK_SAMPLERS)
        SCHEDULERS = list(STOCK_SCHEDULERS)

    samplers_mod.KSampler = KSampler
    comfy_mod.samplers = samplers_mod
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers_mod)
    return KSampler


def test_stock_names_pass(fake_comfy):
    require_known_sampler("euler", "simple")
    require_known_sampler("heun", "karras")


def test_registered_custom_name_passes(fake_comfy):
    # A pack that preloaded successfully registers its names (RES4LYF inserts
    # into KSampler.SAMPLERS at import time); those must pass.
    fake_comfy.SAMPLERS.append("res_2s")
    require_known_sampler("res_2s", "simple")


def test_unknown_sampler_raises_instead_of_silent_euler(fake_comfy):
    with pytest.raises(ValueError) as excinfo:
        require_known_sampler("res_2s", "simple")
    message = str(excinfo.value)
    assert "res_2s" in message
    assert "euler" in message                    # names the silent fallback it prevents
    assert "failed to preload" in message        # names the likely cause
    assert "TROUBLESHOOTING" in message          # points at the runbook


def test_unknown_scheduler_raises(fake_comfy):
    with pytest.raises(ValueError) as excinfo:
        require_known_sampler("euler", "bong_tangent")
    assert "bong_tangent" in str(excinfo.value)
    assert "TROUBLESHOOTING" in str(excinfo.value)
