"""The identity gate's stock reference frees the driver's models before it gives up.

The 2026-11-04 release test refused the Flux 2 template on a fresh pair of
Sparks. Its two slab proof renders matched; the third, the stock reference the
gate compares them with, priced 74.3 GiB against 63.0 GiB on the head, because
the driver still held the 33 GiB Flux 2 text encoder there. The verdict came
out INCONCLUSIVE, the session turned slab residency off for the combination,
and the user's render was priced as the same stock load and refused. ComfyUI
reloads the encoder on demand and the conditioning was already encoded, so the
reference leg now frees what the driver can get back and runs once more
(nodes/gate_cross_mode.py). CPU only, on the shared ceremony rig.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

from dgx_monarch import driver_unload
from dgx_monarch.mesh_safety import StockLoadCapacityError
from gate_orchestration_helpers import _cross_mode_rig, _run

# What every rank's swap cycle reports for a LoRA-free Flux 2 slab ceremony.
NO_STACK_SLAB = [{"conclusive": False, "no_material": True,
                  "reason": "no lora stack: lazy swap is not applicable",
                  "family": "flux2", "slab_active": True}]


def _refusal() -> StockLoadCapacityError:
    return StockLoadCapacityError(
        "[dgxm:C guard=cross_rank_capacity waivable=0] rank 0 priced a stock load of "
        "flux2_dev_fp8mixed.safetensors: the load needs 74.3 GiB and 63.0 GiB of "
        "unified memory is available, 11.4 GiB short.")


class _ComfyModels:
    """ComfyUI's model management with the Flux 2 text encoder loaded."""

    def __init__(self):
        encoder = SimpleNamespace(model=SimpleNamespace(model=type("Flux2TEModel_", (), {})()))
        self.current_loaded_models = [encoder]
        self.unloads = 0

    def unload_all_models(self):
        self.unloads += 1
        self.current_loaded_models = []

    def soft_empty_cache(self):
        pass


def _rig(monkeypatch, tmp_path, renders):
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, renders, {"lora_low_rss": False},
        cycle_response=NO_STACK_SLAB, driver_unload=True)
    model.loras = ()
    model.unet_name = "flux2_dev_fp8mixed.safetensors"
    comfy = _ComfyModels()
    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: comfy)
    monkeypatch.setattr(driver_unload.time, "sleep", lambda _s: None)
    return handle, ledger, model, comfy


def _renders(handle):
    return [call for call in handle.calls if call[0] == "render"]


def test_the_stock_reference_loads_after_the_driver_frees_its_encoder(monkeypatch, tmp_path):
    """The refused 2026-11-04 ceremony: one refusal, an unload, then a real PASS."""
    handle, ledger, model, comfy = _rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), _refusal(), torch.zeros(1)])
    result = _run(model)
    assert comfy.unloads == 1
    assert len(_renders(handle)) == 4
    assert result["cross_mode"]["verdict"] == "PASS"
    assert result["verdict"] == "PASS"
    assert [record[3] for record in ledger.records][:2] == ["PASS", "PASS"]
    # The slab policy the ceremony borrowed for the leg is back.
    applies = [args[0] for method, args in handle.calls if method == "apply_worker_args"]
    assert applies[0]["slab_weights"] is False and "slab_weights" not in applies[-1]


def test_a_reference_that_still_cannot_load_names_itself_as_the_reason(
        monkeypatch, tmp_path, caplog):
    """After the unload and the actors' trim the leg refuses for good.

    The verdict stays INCONCLUSIVE with cross_mode CAPACITY, as before, and the
    reason is the missing stock reference, not the swap cycle's "no lora stack",
    which says nothing about a slab proof.
    """
    handle, _ledger, model, comfy = _rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), _refusal(), _refusal(), _refusal()])
    with caplog.at_level(logging.WARNING):
        result = _run(model)
    assert comfy.unloads == 1
    assert ("clear_vram", ("soft",)) in handle.calls
    assert len(_renders(handle)) == 5
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "CAPACITY"
    assert result["proof_kind"] == "slab_cross_residency"
    reasons = result["inconclusive_reasons"]
    assert not any("no lora stack" in reason for reason in reasons)
    assert any("stock reference cannot load here" in reason for reason in reasons)
    lines = [record.getMessage() for record in caplog.records
             if "identity gate INCONCLUSIVE" in record.getMessage()]
    assert lines and not any("lazy path" in line for line in lines)


def test_a_busy_driver_keeps_its_models_and_the_leg_refuses_once(monkeypatch, tmp_path):
    handle, _ledger, model, comfy = _rig(
        monkeypatch, tmp_path, [torch.zeros(1), torch.zeros(1), _refusal()])
    monkeypatch.setattr(driver_unload, "busy_reason",
                        lambda _handle: "a render is still running in this driver")
    result = _run(model)
    assert comfy.unloads == 0
    assert len(_renders(handle)) == 3
    assert result["cross_mode"]["verdict"] == "CAPACITY"
    assert result["verdict"] == "INCONCLUSIVE"


def test_the_off_switch_keeps_one_stock_attempt(monkeypatch, tmp_path):
    handle, _ledger, model, comfy = _rig(
        monkeypatch, tmp_path, [torch.zeros(1), torch.zeros(1), _refusal()])
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")
    result = _run(model)
    assert comfy.unloads == 0 and len(_renders(handle)) == 3
    assert result["cross_mode"]["verdict"] == "CAPACITY"
