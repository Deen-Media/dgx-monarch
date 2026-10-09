"""activation_footprint_preflight: the driver-side capacity refusal for families
that append reference or pose tokens (docs/TROUBLESHOOTING.md #47)."""
import pytest

from dgx_monarch import mesh_safety

# About 512x512 at 77 frames (20 latent frames): (B, C, T, H, W) = (1, 16, 20, 64, 64).
_VIDEO_SHAPE = (1, 16, 20, 64, 64)


def test_activation_footprint_refuses_when_estimate_exceeds_memavailable(
    monkeypatch, tmp_path,
):
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 4096 + 1)
    with pytest.raises(mesh_safety.StockLoadCapacityError,
                        match="activation footprint preflight refuses"):
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(f), "scail.safetensors",
            video_shape=_VIDEO_SHAPE)


def test_activation_footprint_fits_when_memavailable_is_generous(monkeypatch, tmp_path):
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 2**40)
    mesh_safety.activation_footprint_preflight(
        "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE)  # no raise


def test_activation_footprint_skips_unregistered_family(monkeypatch, tmp_path):
    f = tmp_path / "wan.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 1)
    mesh_safety.activation_footprint_preflight(
        "wan", str(f), "wan.safetensors", video_shape=_VIDEO_SHAPE)  # no raise


def test_activation_footprint_skips_discrete_gpu(monkeypatch, tmp_path):
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 1)
    mesh_safety.activation_footprint_preflight(
        "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE)  # no raise


def test_activation_footprint_skips_off_linux(monkeypatch, tmp_path):
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: None)
    mesh_safety.activation_footprint_preflight(
        "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE)  # no raise


def test_activation_footprint_handles_no_ref_or_pose(monkeypatch, tmp_path):
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 2**40)
    mesh_safety.activation_footprint_preflight(
        "wan_scail", str(f), "scail.safetensors",
        video_shape=_VIDEO_SHAPE, ref_shapes=(), pose_shape=None)  # no raise

    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 4096 + 1)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(f), "scail.safetensors",
            video_shape=_VIDEO_SHAPE, ref_shapes=(), pose_shape=None)


def test_activation_footprint_sp_degree_reduces_estimate(monkeypatch, tmp_path):
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)

    # Compute both estimates to pick a MemAvailable strictly between them:
    # sp_degree=2 halves the activation term (rounding up), never the weights.
    hidden_dim = mesh_safety.REF_POSE_TOKEN_FAMILIES["wan_scail"]
    bytes_per_token = (hidden_dim * mesh_safety._ACTIVATION_BF16_BYTES
                       * mesh_safety._ACTIVATION_BLOCK_FACTOR)
    tokens = mesh_safety._token_count_5d(_VIDEO_SHAPE)
    estimate_sp1 = 4096 + tokens * bytes_per_token
    estimate_sp2 = 4096 + (-(-tokens // 2)) * bytes_per_token
    assert estimate_sp2 < estimate_sp1
    between = (estimate_sp1 + estimate_sp2) // 2

    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: between)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(f), "scail.safetensors",
            video_shape=_VIDEO_SHAPE, sp_degree=1)
    mesh_safety.activation_footprint_preflight(
        "wan_scail", str(f), "scail.safetensors",
        video_shape=_VIDEO_SHAPE, sp_degree=2)  # no raise


def test_activation_footprint_error_is_recognized_as_capacity(monkeypatch, tmp_path):
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 4096 + 1)
    try:
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE)
    except mesh_safety.StockLoadCapacityError as exc:
        assert mesh_safety.is_stock_load_capacity_error(exc)
    else:
        pytest.fail("expected StockLoadCapacityError")


def test_activation_footprint_escape_hatch_env_var_bypasses_refusal(monkeypatch, tmp_path):
    """docs/TROUBLESHOOTING.md #47's operator escape hatch: a false-positive
    refusal must never hard-block a feasible run."""
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 4096 + 1)
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, "1")
    mesh_safety.activation_footprint_preflight(
        "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE)  # no raise


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "", "  "])
def test_an_off_spelling_keeps_the_activation_footprint_guard(
        monkeypatch, tmp_path, value):
    """The kill switch takes only an explicit on value: the refusal tells the
    operator to set `=1`, so `=0` must not disable it."""
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 4096 + 1)
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, value)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE)


@pytest.mark.parametrize("value", ["true", "ON", " yes "])
def test_an_on_spelling_takes_the_activation_footprint_bypass(
        monkeypatch, tmp_path, value):
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 4096 + 1)
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, value)
    mesh_safety.activation_footprint_preflight(
        "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE)  # no raise


def test_token_count_5d_ceiling_rounds_odd_spatial_dims():
    # 5x5 spatial with patch (1,2,2) -> ceil(5/2) = 3 per axis.
    assert mesh_safety._token_count_5d((1, 16, 4, 5, 5)) == 4 * 3 * 3


def test_weights_resident_credit_passes_warm_rerender(monkeypatch, tmp_path):
    """A checkpoint that rendered earlier has already reduced MemAvailable, so
    the credit must pass the same config warm where weights plus activations
    refuse cold (a false positive seen on hardware, 2026-07-28)."""
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    # Enough for activations alone, not for weights + activations.
    hidden = mesh_safety.REF_POSE_TOKEN_FAMILIES["wan_scail"]
    tokens = mesh_safety._token_count_5d(_VIDEO_SHAPE)
    activations = tokens * hidden * mesh_safety._ACTIVATION_BF16_BYTES * mesh_safety._ACTIVATION_BLOCK_FACTOR
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: activations + 1)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE)
    mesh_safety.activation_footprint_preflight(
        "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE,
        weights_resident=True)


def test_weights_resident_still_refuses_oversized_activations(monkeypatch, tmp_path):
    """The credit removes only the weight term, so a warm config whose
    activations alone do not fit still refuses."""
    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 4096)
    with pytest.raises(mesh_safety.StockLoadCapacityError,
                        match="resident, credited"):
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(f), "scail.safetensors", video_shape=_VIDEO_SHAPE,
            weights_resident=True)


def test_note_successful_render_memo_roundtrip(monkeypatch):
    """The driver memo credits only the exact last-rendered unet_name."""
    from dgx_monarch.nodes import render_preflight

    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", None)
    render_preflight.note_successful_render("a.safetensors")
    if render_preflight._LAST_RENDERED_UNET != "a.safetensors":
        pytest.fail("memo did not record the rendered unet")
    render_preflight.note_successful_render(1234)
    if render_preflight._LAST_RENDERED_UNET is not None:
        pytest.fail("non-string unet must clear the memo, not store junk")


def test_request_wrapper_tolerates_non_dict_conditioning_extras(monkeypatch, tmp_path):
    """A None, tensor or string extras slot degrades to shape-only estimation
    instead of aborting the render (`.get` on a non-dict raises)."""
    import sys
    import types

    import torch

    from dgx_monarch.nodes import render_preflight

    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(f)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("wan_scail", "bf16"))
    seen = []
    monkeypatch.setattr(
        mesh_safety, "activation_footprint_preflight",
        lambda *a, **k: seen.append(k))
    model = types.SimpleNamespace(unet_name="scail.safetensors")
    latent = {"samples": torch.zeros(1, 16, 5, 8, 8)}
    for extras_slot in (None, torch.zeros(2), "junk"):
        request = {"positive": [[torch.zeros(1), extras_slot]]}
        render_preflight.activation_footprint_preflight_for_request(
            model, request, latent)
    if len(seen) != 3:
        pytest.fail(f"estimator should run for all 3 malformed shapes, ran {len(seen)}")
    for k in seen:
        if k.get("ref_shapes") != [] or k.get("pose_shape") is not None:
            pytest.fail("malformed extras must degrade to shape-only estimation")


def test_token_count_multiplies_by_batch():
    """The batch axis multiplies the activation token count one-for-one."""
    single = mesh_safety._token_count_5d((1, 16, 5, 8, 8))
    double = mesh_safety._token_count_5d((2, 16, 5, 8, 8))
    if double != 2 * single:
        pytest.fail(f"batch 2 must double the token count ({double} vs {single})")


def test_request_wrapper_scans_all_cond_entries(monkeypatch, tmp_path):
    """A pose set through a timestep-range entry (a later cond entry) and refs
    spread across entries are both seen, not only those in positive[0]."""
    import sys
    import types

    import torch

    from dgx_monarch.nodes import render_preflight

    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(f)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("wan_scail", "bf16"))
    seen = []
    monkeypatch.setattr(mesh_safety, "activation_footprint_preflight",
                        lambda *a, **k: seen.append(k))
    model = types.SimpleNamespace(unet_name="scail.safetensors")
    latent = {"samples": torch.zeros(1, 16, 5, 8, 8)}
    ref = torch.zeros(1, 16, 1, 8, 8)
    pose = torch.zeros(1, 16, 5, 4, 4)
    request = {"positive": [
        [torch.zeros(1), {"reference_latents": [ref]}],
        [torch.zeros(1), {"pose_video_latent": pose}],
    ]}
    render_preflight.activation_footprint_preflight_for_request(model, request, latent)
    if len(seen) != 1:
        pytest.fail("estimator must run exactly once")
    k = seen[0]
    if len(k.get("ref_shapes") or []) != 1 or k.get("pose_shape") is None:
        pytest.fail(f"multi-entry extras must be seen: {k.get('ref_shapes')}, {k.get('pose_shape')}")


def test_in_flight_submission_credits_weights(monkeypatch, tmp_path):
    """A second pipelined push of the same checkpoint before the first collect
    carries weights_resident=True."""
    import sys
    import types

    import torch

    from dgx_monarch.nodes import render_preflight

    f = tmp_path / "scail.safetensors"
    f.write_bytes(b"x" * 4096)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(f)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("wan_scail", "bf16"))
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", None)
    monkeypatch.setattr(render_preflight, "_LAST_SUBMITTED_UNET", None)
    seen = []
    monkeypatch.setattr(mesh_safety, "activation_footprint_preflight",
                        lambda *a, **k: seen.append(k.get("weights_resident")))
    model = types.SimpleNamespace(unet_name="scail.safetensors")
    latent = {"samples": torch.zeros(1, 16, 5, 8, 8)}
    request = {"positive": [[torch.zeros(1), {}]]}
    render_preflight.activation_footprint_preflight_for_request(model, request, latent)
    render_preflight.note_submitted_render("scail.safetensors")
    render_preflight.activation_footprint_preflight_for_request(model, request, latent)
    if seen != [False, True]:
        pytest.fail(f"expected cold then in-flight-credited, saw {seen}")
