"""The pread gate must feature-detect, not version-parse. The shim it installs
also has to hand the file header on to the loader."""
import json
import subprocess
import sys
import types


def _fake_safetensors(monkeypatch, version, with_backend):
    fake = types.ModuleType("safetensors")
    fake.__version__ = version
    if with_backend:
        def safe_open(filename, framework, device="cpu", backend="mmap"):
            ...
    else:
        def safe_open(filename, framework, device="cpu"):  # 0.8.0rc0 shape
            ...
    fake.safe_open = safe_open
    monkeypatch.setitem(sys.modules, "safetensors", fake)
    return fake


def _gate_result(monkeypatch, version, with_backend):
    from dgx_monarch.actor import comfy_bridge

    fake = _fake_safetensors(monkeypatch, version, with_backend)
    comfy_bridge._enable_pread_backend()
    return getattr(fake.safe_open, "_dgxm_pread", False)


def test_rc_without_backend_stays_on_mmap(monkeypatch):
    assert _gate_result(monkeypatch, "0.8.0rc0", with_backend=False) is False


def test_final_with_backend_enables_pread(monkeypatch):
    assert _gate_result(monkeypatch, "0.8.0", with_backend=True) is not False


def test_old_version_stays_on_mmap(monkeypatch):
    assert _gate_result(monkeypatch, "0.7.9", with_backend=False) is False


def test_future_prerelease_with_backend_still_enables(monkeypatch):
    # Feature detection trusts the callable surface rather than a version string.
    assert _gate_result(monkeypatch, "0.9.0b1", with_backend=True) is not False


def test_bridge_reexports_move_only_backend_controls():
    from dgx_monarch.actor import comfy_bridge, pread_backend

    assert comfy_bridge._enable_pread_backend is pread_backend._enable_pread_backend
    assert comfy_bridge._disable_pread_backend is pread_backend._disable_pread_backend


def test_backend_helper_keeps_safetensors_import_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dgx_monarch.actor.pread_backend; "
            "assert 'safetensors' not in sys.modules",
        ],
        check=True,
    )


def test_disable_restores_exact_original_and_is_idempotent(monkeypatch):
    from dgx_monarch.actor import pread_backend

    fake = _fake_safetensors(monkeypatch, "0.8.0", with_backend=True)
    original = fake.safe_open

    pread_backend._enable_pread_backend()
    first_shim = fake.safe_open
    assert first_shim is not original
    assert first_shim._dgxm_orig is original

    pread_backend._disable_pread_backend()
    assert fake.safe_open is original
    pread_backend._disable_pread_backend()
    assert fake.safe_open is original

    pread_backend._enable_pread_backend()
    assert fake.safe_open is not first_shim
    assert fake.safe_open._dgxm_orig is original


def test_pread_reads_still_carry_the_file_metadata_to_the_loader(tmp_path, monkeypatch):
    # The worker loads a DiT with no mmap, and comfy builds that model from the
    # file's __metadata__["config"] (LTX carries connector depth and timestep
    # scale there). Both safetensors and the pread backend are real.
    import safetensors
    import safetensors.torch
    import torch

    from dgx_monarch.actor import pread_backend

    config = {"transformer": {"connector_num_layers": 8}}
    path = str(tmp_path / "dit.safetensors")
    safetensors.torch.save_file({"w": torch.zeros(2, 2)}, path,
                                metadata={"config": json.dumps(config)})
    # Records the live attribute so teardown puts the process back on it.
    monkeypatch.setattr(safetensors, "safe_open", safetensors.safe_open)

    pread_backend._enable_pread_backend()
    assert getattr(safetensors.safe_open, "_dgxm_pread", False) is True
    with safetensors.safe_open(path, framework="pt") as handle:
        assert json.loads(handle.metadata()["config"]) == config


def _installed_shim(monkeypatch):
    fake = _fake_safetensors(monkeypatch, "0.8.0", with_backend=True)
    from dgx_monarch.actor import comfy_bridge

    comfy_bridge._enable_pread_backend()
    return fake


def test_mmap_window_restores_pread_after_the_load(monkeypatch):
    from dgx_monarch.actor import pread_backend

    fake = _installed_shim(monkeypatch)
    shim = fake.safe_open
    assert getattr(shim, "_dgxm_pread", False)
    with pread_backend.mmap_load_window():
        assert fake.safe_open is shim._dgxm_orig  # stock mmap during the load
    assert fake.safe_open is shim  # pread back for every later load


def test_mmap_window_restores_pread_when_the_load_raises(monkeypatch):
    import pytest

    from dgx_monarch.actor import pread_backend

    fake = _installed_shim(monkeypatch)
    shim = fake.safe_open
    with pytest.raises(RuntimeError):
        with pread_backend.mmap_load_window():
            raise RuntimeError("load died")
    assert fake.safe_open is shim


def test_mmap_window_is_a_noop_without_the_pread_shim(monkeypatch):
    from dgx_monarch.actor import pread_backend

    fake = _fake_safetensors(monkeypatch, "0.8.0", with_backend=True)
    stock = fake.safe_open
    with pread_backend.mmap_load_window():
        assert fake.safe_open is stock
    assert fake.safe_open is stock
