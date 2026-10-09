"""bake_key/verify_key pin their math to the model's load_device.

lora_compute_dtype depends on the device (fp16 on GPU-class devices, fp32 on
cpu), and comfy's own load-time bake computes at device_to=load_device. A lazy
swap can land while weights are still cpu-offloaded (the first swap after an
unload, before load_models_gpu); a bake there at weight.device computes in
fp32 on cpu, a different precision from every other bake, from stock comfy,
and from slab mode (whose offload pin keeps weights GPU-resident). Measured
2026-07-09: deterministic cross-state render divergence, max pixel diff 240.
"""
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor import store_bake

KEY = "diffusion_model.blocks.0.attn.gate.weight"
LOAD_DEV = torch.device("cpu")  # sentinel identity matters, not its kind


@pytest.fixture
def comfy_stub(monkeypatch):
    seen = SimpleNamespace(cast_args=None, dtype_queries=[], oracle_device=None)

    mm = types.ModuleType("comfy.model_management")

    def lora_compute_dtype(device):
        seen.dtype_queries.append(device)
        # device-dependent, as on real hardware (fp16 on GPU, fp32 on cpu)
        return torch.float16 if device is LOAD_DEV else torch.float32

    def cast_to_device(weight, device, dtype, copy=False):
        seen.cast_args = (device, dtype)
        return weight.detach().clone().to(dtype)

    mm.lora_compute_dtype = lora_compute_dtype
    mm.cast_to_device = cast_to_device

    mp = types.ModuleType("comfy.model_patcher")
    weight = torch.ones(4, 4, dtype=torch.bfloat16)  # "cpu-offloaded" residence
    mp.get_key_weight = lambda model, key: (weight, None, None)

    lora = types.ModuleType("comfy.lora")
    lora.calculate_weight = lambda patches, temp, key: temp * 2

    cfloat = types.ModuleType("comfy.float")
    cfloat.stochastic_rounding = lambda out, dtype, seed=0: out.to(dtype)

    cutils = types.ModuleType("comfy.utils")
    cutils.string_to_seed = lambda key: 0
    cutils.copy_to_param = lambda model, key, out: None

    comfy = types.ModuleType("comfy")
    comfy.model_management, comfy.model_patcher = mm, mp
    comfy.lora, comfy.float, comfy.utils = lora, cfloat, cutils
    for name, mod in [("comfy", comfy), ("comfy.model_management", mm),
                      ("comfy.model_patcher", mp), ("comfy.lora", lora),
                      ("comfy.float", cfloat), ("comfy.utils", cutils)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return seen


def test_bake_key_computes_on_load_device_not_weight_residence(comfy_stub):
    active = SimpleNamespace(model=object(), load_device=LOAD_DEV,
                             patches={KEY: ["patch"]})
    store_bake.bake_key(active, KEY)
    device, dtype = comfy_stub.cast_args
    assert device is LOAD_DEV, "bake must pin to load_device, not weight.device"
    assert dtype == torch.float16, "compute dtype must come from load_device"


def test_bake_key_falls_back_to_weight_device_without_load_device(comfy_stub):
    active = SimpleNamespace(model=object(), patches={KEY: ["patch"]})
    store_bake.bake_key(active, KEY)
    device, dtype = comfy_stub.cast_args
    assert device == torch.device("cpu") and dtype == torch.float32


def test_verify_oracle_uses_the_same_pinned_device(comfy_stub, monkeypatch):
    baked = torch.full((4, 4), 2.0, dtype=torch.float16)
    monkeypatch.setattr("dgx_monarch.actor.unbake.live_tensor",
                        lambda model, key: baked)

    def oracle(key, device_to=None, return_weight=False):
        comfy_stub.oracle_device = device_to
        return baked.clone()

    active = SimpleNamespace(model=object(), load_device=LOAD_DEV,
                             patches={KEY: ["patch"]},
                             patch_weight_to_device=oracle)
    store = SimpleNamespace(_bake_key=lambda active, key: None, verify_failures=0)
    store_bake.verify_key(store, active, KEY)
    assert comfy_stub.oracle_device is LOAD_DEV


def test_verify_oracle_mismatch_raises_and_counts(comfy_stub, monkeypatch):
    """A baked weight that does not bit-match the oracle must raise (the slot
    drops for a clean reload) and increment the failure counter."""
    from dgx_monarch.actor.unbake import UnbakeError

    baked = torch.full((4, 4), 2.0, dtype=torch.float16)
    monkeypatch.setattr("dgx_monarch.actor.unbake.live_tensor",
                        lambda model, key: baked)
    active = SimpleNamespace(
        model=object(), load_device=LOAD_DEV, patches={KEY: ["patch"]},
        patch_weight_to_device=lambda key, device_to=None, return_weight=False:
            torch.full((4, 4), 3.0, dtype=torch.float16))
    store = SimpleNamespace(_bake_key=lambda active, key: None, verify_failures=0)
    with pytest.raises(UnbakeError, match="ambient bake verify MISMATCH"):
        store_bake.verify_key(store, active, KEY)
    assert store.verify_failures == 1
