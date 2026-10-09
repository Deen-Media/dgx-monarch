"""Activation footprint preflight regressions on the render dispatch path."""
import sys
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.nodes.common as common
from gate_orchestration_helpers import _clear_process_gate_verdicts  # noqa: F401


def _scail_render_rig(monkeypatch, tmp_path, *, avail_bytes):
    """Stub folder_paths and the UMA memory reads so run_render's
    activation-footprint preflight can resolve a checkpoint; each test sets
    the family the sniff returns."""
    import types

    from dgx_monarch import mesh_safety as mesh_safety_mod
    from dgx_monarch.nodes import render_preflight as render_preflight_mod

    ckpt = tmp_path / "scail.safetensors"
    ckpt.write_bytes(b"x" * 4096)
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda _kind, _name: str(ckpt)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setattr(mesh_safety_mod, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety_mod, "mem_available_bytes", lambda: avail_bytes)
    return render_preflight_mod


def test_run_render_activation_preflight_blocks_before_dispatch(monkeypatch, tmp_path):
    from dgx_monarch import mesh_safety as mesh_safety_mod

    render_preflight_mod = _scail_render_rig(monkeypatch, tmp_path, avail_bytes=4096 + 1)
    monkeypatch.setattr(
        render_preflight_mod, "sniff_checkpoint",
        lambda _path: ("wan_scail", "bf16"))
    monkeypatch.setattr(
        common, "submit_render",
        lambda *_a, **_k: pytest.fail("submit_render called"))
    monkeypatch.setattr(
        common, "_maybe_auto_gate",
        lambda *_a, **_k: pytest.fail("_maybe_auto_gate called"))
    model = SimpleNamespace(unet_name="scail.safetensors")
    request = {"positive": [(torch.zeros(1), {
        "reference_latents": [torch.zeros(1, 16, 1, 64, 64)],
        "pose_video_latent": torch.zeros(1, 16, 4, 32, 32),
    })]}
    latent = {"samples": torch.zeros(1, 16, 20, 64, 64)}
    with pytest.raises(mesh_safety_mod.StockLoadCapacityError):
        common.run_render(model, request, latent, 1.0, 2)


def test_run_render_activation_preflight_ignores_non_scail_family(monkeypatch, tmp_path):
    render_preflight_mod = _scail_render_rig(monkeypatch, tmp_path, avail_bytes=1)
    monkeypatch.setattr(
        render_preflight_mod, "sniff_checkpoint",
        lambda _path: ("wan", "bf16"))
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={
        "lora_low_rss": True, "slab_weights": True}), unet_name="wan.safetensors")
    request = {"positive": [(torch.zeros(1), {})]}
    latent = {"samples": torch.zeros(1, 16, 20, 64, 64)}

    class Pending:
        def result(self):
            return {"samples": torch.zeros(1)}

    monkeypatch.setattr(common, "submit_render", lambda *_a, **_k: Pending())
    monkeypatch.setattr(common, "_maybe_auto_gate", lambda *_a, **_k: None)
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_RUNNING.clear()
    # Even with a MemAvailable of 1 byte, an unregistered family never fires the
    # reference/pose activation preflight; run_render reaches the mocked dispatch.
    result = common.run_render(model, request, latent, 1.0, 2)
    assert result == {"samples": torch.zeros(1)}


def test_activation_preflight_for_request_never_crashes_on_wrong_suffix(
    monkeypatch, tmp_path,
):
    """A real, unmocked non-safetensors checkpoint never lets
    CheckpointSniffError out of the preflight, for any preset or family.
    resolve_topology sniffs only under the "auto" preset, but this preflight
    runs on every render, so it skips a non-safetensors file and fails open on
    an unreadable header when no family is forced."""
    import types

    from dgx_monarch.nodes import render_preflight as render_preflight_mod

    ckpt = tmp_path / "not-safetensors.gguf"
    ckpt.write_bytes(b"garbage, not a safetensors header at all")
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda _kind, _name: str(ckpt)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    model = SimpleNamespace(unet_name="not-safetensors.gguf")
    request = {"positive": [(torch.zeros(1), {})]}
    latent = {"samples": torch.zeros(1, 16, 20, 64, 64)}
    render_preflight_mod.activation_footprint_preflight_for_request(
        model, request, latent)  # must not raise


def test_activation_preflight_for_request_never_crashes_on_malformed_header(
    monkeypatch, tmp_path,
):
    """The same for a .safetensors file whose header does not parse: the real
    sniff raises CheckpointSniffError, and the preflight catches it instead of
    failing every render whatever the family."""
    import types

    from dgx_monarch.nodes import render_preflight as render_preflight_mod

    ckpt = tmp_path / "scail.safetensors"
    ckpt.write_bytes(b"\x00" * 8 + b"not a real json header")
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda _kind, _name: str(ckpt)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    model = SimpleNamespace(unet_name="scail.safetensors")
    request = {"positive": [(torch.zeros(1), {})]}
    latent = {"samples": torch.zeros(1, 16, 20, 64, 64)}
    render_preflight_mod.activation_footprint_preflight_for_request(
        model, request, latent)  # must not raise


def test_activation_preflight_for_request_skips_ref_latent_missing_shape(
    monkeypatch, tmp_path,
):
    """A reference latent with no real .shape is skipped, not counted, so the
    estimate cannot raise mid-way (mesh_safety._token_count_5d unpacks exactly
    five shape elements)."""
    render_preflight_mod = _scail_render_rig(monkeypatch, tmp_path, avail_bytes=2**40)
    monkeypatch.setattr(
        render_preflight_mod, "sniff_checkpoint",
        lambda _path: ("wan_scail", "bf16"))
    model = SimpleNamespace(unet_name="scail.safetensors")
    request = {"positive": [(torch.zeros(1), {
        "reference_latents": [object()],  # no .shape attribute at all
    })]}
    latent = {"samples": torch.zeros(1, 16, 20, 64, 64)}
    render_preflight_mod.activation_footprint_preflight_for_request(
        model, request, latent)  # must not raise


def test_the_render_site_reports_a_configless_ltx_checkpoint(monkeypatch, tmp_path):
    """Both preflight sites carry the same report. Wiring only one would make
    its appearance depend on which site ran first for a given render."""
    import logging

    from dgx_monarch import family_select
    from dgx_monarch.adapters import detect

    render_preflight_mod = _scail_render_rig(monkeypatch, tmp_path, avail_bytes=2**40)
    monkeypatch.setattr(
        render_preflight_mod, "sniff_checkpoint", lambda _path: ("ltx", "bf16"))
    monkeypatch.setattr(detect, "sniff_metadata_keys", lambda _path: ("license",))
    monkeypatch.setattr(family_select, "_auto_warned", set())
    model = SimpleNamespace(unet_name="scail.safetensors")
    request = {"positive": [(torch.zeros(1), {})]}
    latent = {"samples": torch.zeros(1, 16, 20, 64, 64)}

    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    family_select.log.addHandler(handler)
    try:
        render_preflight_mod.activation_footprint_preflight_for_request(
            model, request, latent)
    finally:
        family_select.log.removeHandler(handler)
    assert [line for line in records if "carries no 'config' metadata" in line]
