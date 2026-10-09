"""The strict FSDP launch preflight: header proof, inode recheck and reload."""
import os
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import model_store as ms
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.constants import TRANSITION_LOAD
from store_ensure_helpers import STACK, _FakeBase, _write_one_tensor_safetensors
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.


def test_non_fsdp_store_does_not_run_strict_launch_preflight(rig, monkeypatch):
    store, calls = rig
    monkeypatch.setattr(
        ms.store_detect,
        "validate_fsdp_request_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("ordinary store ran FSDP preflight"),
    )

    _active, transition = store.ensure("m.safetensors", None, STACK)
    assert transition == TRANSITION_LOAD
    assert calls.load == 1


def test_fsdp_store_preflights_worker_file_before_comfy_load(rig, monkeypatch):
    store, calls = rig

    def reject(*_args, **_kwargs):
        raise UnsupportedModelError("worker-local FSDP header refused")

    monkeypatch.setattr(ms.store_detect, "validate_fsdp_request_checkpoint", reject)
    with pytest.raises(UnsupportedModelError, match="worker-local FSDP header refused"):
        store.ensure(
            "m.safetensors",
            None,
            None,
            fsdp_launch=True,
        )
    assert calls.load == 0


def test_fsdp_header_proof_is_bound_to_the_artifact_identity(rig, monkeypatch):
    from dgx_monarch.mesh_safety import ArtifactBindingError

    store, calls = rig
    identities = iter(("before", "after"))
    callbacks = []

    def identity(unet_name, _lora_stack):
        value = next(identities)
        return {
            "digest": value,
            "comfy": "commit",
            "artifacts": [{
                "kind": "diffusion_models",
                "name": unet_name,
                "signature": value,
            }],
        }

    monkeypatch.setattr(ms, "request_artifact_identity", identity)
    monkeypatch.setattr(
        ms.store_detect,
        "validate_fsdp_request_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(
        ArtifactBindingError,
        match="identity changed while proving the worker-local FSDP checkpoint header",
    ):
        store.ensure(
            "m.safetensors",
            None,
            None,
            on_base_loaded=lambda *_args: callbacks.append("entered"),
            fsdp_launch=True,
        )
    assert callbacks == []
    assert calls.load == 0


def test_fsdp_exact_header_inode_is_rechecked_after_comfy_load(
    rig,
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.mesh_safety import ArtifactBindingError

    store, calls = rig
    path = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    _write_one_tensor_safetensors(replacement, "F16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)

    def replace_during_load(_path, model_options=None):
        calls.load += 1
        calls.model_options.append(model_options)
        os.replace(replacement, path)
        return _FakeBase()

    monkeypatch.setattr(
        sys.modules["comfy.sd"],
        "load_diffusion_model",
        replace_during_load,
    )

    with pytest.raises(
        ArtifactBindingError,
        match="changed while finishing the Comfy model load",
    ):
        store.ensure(
            "model.safetensors",
            None,
            None,
            fsdp_launch=True,
        )

    assert calls.load == 1
    assert store.current is None


def test_fsdp_comfy_load_uses_proven_inode_across_path_aba(rig, monkeypatch, tmp_path):
    store, calls = rig
    path = tmp_path / "model.safetensors"
    proven_file = tmp_path / "proven-bytes.safetensors"
    replacement_file = tmp_path / "replacement-bytes.safetensors"
    replacement = tmp_path / "replacement-link.safetensors"
    parked = tmp_path / "proven-link.safetensors"
    _write_one_tensor_safetensors(proven_file, "BF16")
    _write_one_tensor_safetensors(replacement_file, "F16")
    path.symlink_to(proven_file)
    replacement.symlink_to(replacement_file)
    expected_bytes = path.read_bytes()
    expected_inode = os.stat(path).st_ino
    loaded = SimpleNamespace(path=None, payload=None, inode=None)
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)

    def swap_restore_during_load(load_path, model_options=None):
        calls.load += 1
        calls.model_options.append(model_options)
        os.replace(path, parked)
        os.replace(replacement, path)
        try:
            loaded.path = load_path
            loaded.payload = open(load_path, "rb").read()
            loaded.inode = os.stat(load_path).st_ino
        finally:
            os.replace(path, replacement)
            os.replace(parked, path)
        return _FakeBase()

    monkeypatch.setattr(
        sys.modules["comfy.sd"],
        "load_diffusion_model",
        swap_restore_during_load,
    )

    _active, transition = store.ensure(
        "model.safetensors",
        None,
        None,
        fsdp_launch=True,
    )

    assert transition == TRANSITION_LOAD
    assert loaded.path != str(path)
    assert loaded.path.endswith(".safetensors")
    assert loaded.payload == expected_bytes
    assert loaded.inode == expected_inode
    assert os.path.exists(loaded.path)
    store.unload_all()
    assert not os.path.exists(loaded.path)


def test_fsdp_reloads_same_fingerprint_when_checkpoint_inode_changes(
    rig,
    monkeypatch,
    tmp_path,
):
    store, calls = rig
    path = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    replacement.write_bytes(path.read_bytes())
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)

    _active, first_transition = store.ensure(
        "model.safetensors",
        None,
        None,
        fsdp_launch=True,
    )
    first_pin = store.current.fsdp_checkpoint_pin
    first_alias = first_pin.loader_path
    assert first_transition == TRANSITION_LOAD

    os.replace(replacement, path)
    assert os.stat(path).st_ino != first_pin.file_identity.file_ino

    _active, second_transition = store.ensure(
        "model.safetensors",
        None,
        None,
        fsdp_launch=True,
    )

    assert second_transition == TRANSITION_LOAD
    assert calls.load == 2
    assert store.current.fsdp_checkpoint_pin is not first_pin
    assert store.current.fsdp_checkpoint_pin.file_identity.file_ino == os.stat(path).st_ino
    assert not os.path.exists(first_alias)
