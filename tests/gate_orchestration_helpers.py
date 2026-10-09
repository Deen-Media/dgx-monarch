"""Process-verdict reset, artifact identity and the cross-mode ceremony rig shared by the gate suites."""
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
import dgx_monarch.mesh_setup as mesh_setup_mod
import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
import dgx_monarch.runtime_provenance as runtime_provenance
from dgx_monarch.actor import model_store as model_store_mod
from dgx_monarch.mesh_setup import SetupToken
from dgx_monarch.nodes import gate_process_state


@pytest.fixture(autouse=True)
def _clear_process_gate_verdicts():
    common._AUTO_GATE_SESSION.clear()
    gate_process_state._PROCESS_GATE_DENIALS = {}
    yield
    common._AUTO_GATE_SESSION.clear()
    gate_process_state._PROCESS_GATE_DENIALS = {}


def _identity(unet_name, loras):
    artifacts = [{"kind": "diffusion_models", "name": unet_name,
                  "signature": "model-sig"}]
    artifacts.extend(
        {"kind": "loras", "name": entry["name"],
         "signature": f"lora-sig-{index}"}
        for index, entry in enumerate(loras or [])
    )
    return {
        "digest": ledger_mod.artifact_set_signature(
            item["signature"] for item in artifacts).current,
        "comfy": "known",
        "artifacts": artifacts,
    }


def _cross_mode_rig(
    monkeypatch,
    tmp_path,
    renders,
    worker_args,
    cycle_response=None,
    *,
    cycle_error=None,
    initial_unload_response=None,
    cleanup_unload_error=None,
    cleanup_unload_response=None,
    cleanup_latch_errors=(),
):
    events = []

    class Handle:
        setup_generation = 4
        setup_key = ("setup",)
        worker_args_key = ("worker",)
        world = 1

        def __init__(self):
            self.calls = []
            self.unload_calls = 0
            self.dirty_latches = []
            self.latch_attempts = 0
            self.cleanup_latch_errors = list(cleanup_latch_errors)
            self.defunct = False
            self.provenance_calls = 0
            self.provenance_kwargs = []
            self.provenance_response = None
            self.provenance_source = (
                runtime_provenance.cached_dgx_source_manifest_sha256()
            )

        def _latch_ambiguous_mutation(self, endpoint, timeout_s, exc):
            self.latch_attempts += 1
            if self.cleanup_latch_errors:
                raise self.cleanup_latch_errors.pop(0)
            self.dirty_latches.append((endpoint, timeout_s, exc))

        def call_all(self, method, *args, **kwargs):
            events.append(("call", method))
            self.calls.append((method, args))
            if method == "provenance_baseline":
                mesh_setup_mod._require_setup_token(
                    self, kwargs.get("setup_token")
                )
                self.provenance_calls += 1
                self.provenance_kwargs.append(dict(kwargs))
                if callable(self.provenance_response):
                    return self.provenance_response(
                        self.provenance_calls, args[0]
                    )
                if self.provenance_response is not None:
                    return self.provenance_response
                world = int(self.world)
                return [
                    {
                        "rank": rank,
                        "world": world,
                        "setup_generation": args[0],
                        "source_manifest_sha256": self.provenance_source,
                    }
                    for rank in range(world)
                ]
            if method == "unload":
                self.unload_calls += 1
                if initial_unload_response is not None and self.unload_calls == 1:
                    return initial_unload_response
                if cleanup_unload_error is not None and self.unload_calls > 1:
                    raise cleanup_unload_error
                if cleanup_unload_response is not None and self.unload_calls > 1:
                    return cleanup_unload_response
                world = int(getattr(self, "world", 1))
                return [{"unloaded": True} for _rank in range(world)]
            if method in {"gate_swap_cycle", "gate_fsdp_reload_cycle"}:
                if cycle_error is not None:
                    raise cycle_error
                if cycle_response is not None:
                    return [
                        {
                            "rank": rank,
                            "world": self.world,
                            "setup_generation": self.setup_generation,
                            **dict(entry),
                        }
                        for rank, entry in enumerate(cycle_response)
                    ]
                slab_active = worker_args.get("slab_weights") is True
                return [{"rank": 0, "world": self.world,
                         "setup_generation": self.setup_generation,
                         "conclusive": True, "transitions": ["lazy"],
                         "family": "krea2" if slab_active else "unvouched",
                         "slab_active": slab_active}]
            if method == "apply_worker_args":
                return list(getattr(self, "apply_response", [{"slab_weights": False}]))
            return [{"ok": True}]

    class Ledger:
        def __init__(self):
            self.records = []
            self.retests = []
            self.events = events

        def begin_retest_required(self, *args):
            self.events.append(("retest", args[0]))
            self.retests.append(args)

        def record(self, *args):
            self.records.append(args)

    handle, ledger = Handle(), Ledger()
    model = SimpleNamespace(
        unet_name="model.safetensors", options={},
        loras=({"name": "test.safetensors", "strength": 1.0},),
        mesh=SimpleNamespace(handle=handle, worker_args=worker_args),
    )
    queue = iter(renders)

    def render(*args, **kwargs):
        handle.calls.append(("render", args))
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return {"samples": item}

    monkeypatch.setattr(gate_mod, "run_render", render)
    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        gate_mod,
        "render_setup_token",
        lambda *_args: SetupToken(
            int(handle.setup_generation),
            tuple(handle.setup_key),
            tuple(handle.worker_args_key),
        ),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda value: ("combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gate_mod, "GateLedger", lambda directory: ledger)
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)
    monkeypatch.setattr("dgx_monarch.telemetry.emit", lambda *args, **kwargs: None)
    return handle, ledger, model


def _run(model, latent=None):
    return gate_mod.run_identity_ceremony(
        model, {"noise_seed": 1, "steps": 2, "cfg": 1.0},
        {"samples": torch.zeros(1)} if latent is None else latent,
        1.0, 2, "explicit", run_id="x",
    )
