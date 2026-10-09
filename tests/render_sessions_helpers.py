"""Handle, progress and direct-submit doubles shared by the render-session suites."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.pending as pending_mod
import dgx_monarch.nodes.render_submit as submit_mod
from dgx_monarch import mesh_setup, telemetry
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.topology import Topology


def _real_handle() -> MeshHandle:
    """Make an identity-valid handle without constructing Monarch actors."""
    handle = object.__new__(MeshHandle)
    handle.lock = threading.RLock()
    handle.config = SimpleNamespace(worker_args={})
    return handle


def _ready_real_handle() -> tuple[MeshHandle, mesh_setup.SetupToken]:
    handle = _real_handle()
    handle.lock = threading.RLock()
    handle.defunct = False
    handle.setup_cleanup_state = None
    handle.setup_generation = 1
    handle.setup_key = ("ready",)
    handle.worker_args_key = ("policy",)
    handle.sample_leases = {}
    handle.abandoned_sample_leases = {}
    handle.deferred_supervision_error = None
    return handle, mesh_setup.current_setup_token(handle)


class _Progress:
    port = 1234

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _stub_direct_submit(monkeypatch, handle, submit_sample):
    handle.world = 1
    handle.cancel_sample = lambda *_args, **_kwargs: None
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=handle, topology_preset="uly1", attention="TORCH_FLASH",
            sync_ulysses=False, worker_args={}),
        unet_name="model.safetensors",
    )
    topology = Topology(world=1)
    monkeypatch.setattr(
        submit_mod, "_enforce_persisted_quarantine", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        common, "resolve_topology", lambda *_args: (topology, False, "test"))
    monkeypatch.setattr(submit_mod, "validate_render_topology", lambda *_args: None)
    monkeypatch.setattr(submit_mod, "pack_latent", lambda value: value)
    monkeypatch.setattr(submit_mod, "ProgressReceiver", _Progress)
    monkeypatch.setattr(
        mesh_setup, "ensure_request_setup", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mesh_setup, "submit_sample", submit_sample)
    monkeypatch.setattr(submit_mod, "_finish_render", lambda *_args: {"samples": "done"})
    monkeypatch.setattr(pending_mod, "_throw_if_comfy_interrupted", lambda: None)
    monkeypatch.setattr(
        telemetry.render_progress, "start", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda *_args: None)
    return spec
