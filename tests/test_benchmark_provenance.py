"""Benchmark results must carry enough source state to reproduce a table."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dgxm_benchmark_run_matrix", REPO / "benchmark" / "run_matrix.py"
)
assert SPEC and SPEC.loader
run_matrix = importlib.util.module_from_spec(SPEC)
_argv_before = list(sys.argv)
_path_before = list(sys.path)
SPEC.loader.exec_module(run_matrix)
assert sys.argv == _argv_before
assert sys.path == _path_before


def _load_rdmabench(monkeypatch):
    """Load the standalone benchmark through its real import boundary."""
    actor_api = ModuleType("monarch.actor")
    actor_api.Actor = object
    actor_api.endpoint = lambda function: function
    actor_api.attach_to_workers = lambda **_kwargs: None
    actor_api.enable_transport = lambda _bind: None
    monarch = ModuleType("monarch")
    monarch.__path__ = []
    monarch.actor = actor_api
    monkeypatch.setitem(sys.modules, "monarch", monarch)
    monkeypatch.setitem(sys.modules, "monarch.actor", actor_api)

    spec = importlib.util.spec_from_file_location(
        "dgxm_rdmabench_under_test", REPO / "benchmark" / "rdma_return_bench.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # The bench script inserts the repo's src into sys.path at import. Restore
    # the path per load, or each load leaves one more copy for the whole run.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("primary", "cleanup", "winner"),
    [
        (RuntimeError("direct RDMA read failed"), None, "primary"),
        (None, RuntimeError("owned ProcMesh stop failed"), "cleanup"),
        (RuntimeError("direct RDMA read failed"), RuntimeError("owned ProcMesh stop failed"),
         "primary"),
        (RuntimeError("direct RDMA read failed"), KeyboardInterrupt("stop cancelled"), "cleanup"),
    ],
)
def test_rdma_benchmark_stops_only_its_owned_procmesh_and_preserves_failures(
    monkeypatch, primary, cleanup, winner,
):
    """A microbenchmark attach must never stop persistent Worker services."""
    module = _load_rdmabench(monkeypatch)
    calls: list[object] = []

    class Future:
        def __init__(self, value=None, error=None):
            self.value, self.error = value, error

        def get(self, *args, **kwargs):
            calls.append(("get", args, kwargs))
            if self.error is not None:
                raise self.error
            return self.value

    class Hosts:
        initialized = Future()

        def __init__(self):
            self.shutdown_calls = 0

        def spawn_procs(self, **kwargs):
            calls.append(("spawn_procs", kwargs))
            return procs

        def shutdown(self):
            self.shutdown_calls += 1
            raise AssertionError("benchmark must not stop persistent workers")

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def spawn(self, name, actor):
            calls.append(("spawn", name, actor))
            return actors

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return Future(error=cleanup)

    class Actors:
        extent = SimpleNamespace(labels={"hosts": 2})
        info = SimpleNamespace(call=lambda: Future([
            ({"hosts": 0}, {"host": "spark-1", "ibverbs": True,
                            "backend": "ibverbs", "ib_devices": ["mlx5_0"]}),
        ]))
        pack = SimpleNamespace(call_one=lambda *_args: Future((
            {"kind": "rdma"}, hashlib.sha256(b"payload").hexdigest(), "rdma", 1,
        )))

        def slice(self, **kwargs):
            calls.append(("slice", kwargs))
            return self

    hosts, procs, actors = Hosts(), Procs(), Actors()
    monkeypatch.setattr(module, "find_config_path", lambda _value: Path("cluster.toml"))
    monkeypatch.setattr(module, "load_cluster_config", lambda _path: SimpleNamespace(
        client_bind="10.0.0.1", hosts=[SimpleNamespace(address="spark-1")],
    ))
    monkeypatch.setattr(module, "enable_transport", lambda bind: calls.append(("transport", bind)))
    monkeypatch.setattr(module, "attach_to_workers", lambda **kwargs: hosts)
    if primary is None:
        payload = SimpleNamespace(numpy=lambda: SimpleNamespace(tobytes=lambda: b"payload"))
        monkeypatch.setattr(module, "read_latent_result", lambda _desc: payload)
    else:
        monkeypatch.setattr(
            module,
            "read_latent_result",
            lambda _desc: (_ for _ in ()).throw(primary),
        )
    monkeypatch.setattr(sys, "argv", ["rdma_return_bench.py", "--sizes-mib", "1",
                                       "--splits", "1", "--repeats", "1"])

    expected = primary if winner == "primary" else cleanup
    if expected is None:
        assert module.main() == 0
    else:
        with pytest.raises(type(expected)) as failure:
            module.main()
        assert failure.value is expected
        if primary is not None and cleanup is not None:
            expected_cause = cleanup if winner == "primary" else primary
            assert failure.value.__cause__ is expected_cause

    assert procs.stop_reasons == ["dgx-monarch rdma return benchmark"]
    assert hosts.shutdown_calls == 0
    assert ("spawn_procs", {"per_host": {"gpus": 1}}) in calls
    assert ("get", (), {"timeout": 60}) in calls


def test_matrix_provenance_embeds_hash_and_parsed_content(tmp_path):
    raw = b'# campaign note\r\n[defaults]\r\nsteps = 1\r\n\r\n[[case]]\r\nname = "tiny"\r\n'
    path = tmp_path / "matrix.toml"
    path.write_bytes(raw)
    actual = run_matrix._matrix_provenance(str(path))

    assert actual == {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content": raw.decode(),
    }


def test_git_provenance_records_dirty_state_and_changed_paths(monkeypatch):
    responses = iter([
        SimpleNamespace(returncode=0, stdout="abc123\n"),
        SimpleNamespace(returncode=0, stdout=" M benchmark/run_matrix.py\n?? local.toml\n"),
    ])
    monkeypatch.setattr(run_matrix.subprocess, "run", lambda *_args, **_kwargs: next(responses))

    assert run_matrix._git_provenance(REPO) == {
        "commit": "abc123",
        "dirty": True,
        "changes": [" M benchmark/run_matrix.py", "?? local.toml"],
    }


def test_case_provenance_covers_unet_text_encoder_and_loras(monkeypatch):
    calls = []

    def fake_artifact(name, category):
        calls.append((name, category))
        return {"name": name, "category": category}

    monkeypatch.setattr(run_matrix, "_artifact_provenance", fake_artifact)
    cases = [{
        "name": "cell",
        "unet": "model.safetensors",
        "te": {"name": "text.safetensors", "type": "krea2"},
        "loras": [{"name": "style.safetensors", "strength": 0.7}],
    }]

    actual = run_matrix._case_artifact_provenance(cases, {"steps": 8})

    assert calls == [
        ("model.safetensors", "diffusion_models"),
        ("text.safetensors", "text_encoders"),
        ("style.safetensors", "loras"),
    ]
    assert actual["cell"]["text_encoder"] == {
        "name": "text.safetensors",
        "category": "text_encoders",
    }
    assert actual["cell"]["loras"][0]["name"] == "style.safetensors"


def test_case_provenance_uses_list_only_for_multiple_text_encoders(monkeypatch):
    monkeypatch.setattr(
        run_matrix,
        "_artifact_provenance",
        lambda name, category: {"name": name, "category": category},
    )
    cases = [{
        "name": "cell",
        "unet": "model.safetensors",
        "te": {"name": ["clip.safetensors", "t5.safetensors"], "type": "sd3"},
    }]

    actual = run_matrix._case_artifact_provenance(cases, {})

    assert actual["cell"]["text_encoder"] == [
        {"name": "clip.safetensors", "category": "text_encoders"},
        {"name": "t5.safetensors", "category": "text_encoders"},
    ]


def test_case_settings_preserve_defaults_and_per_cell_overrides():
    settings = run_matrix._case_settings(
        {"name": "cell", "steps": 1, "topology": "uly2"},
        {"steps": 8, "seed": 42, "mode": "cluster"},
    )
    assert settings == {
        "steps": 1,
        "seed": 42,
        "mode": "cluster",
        "name": "cell",
        "topology": "uly2",
    }


def test_sampling_path_defaults_to_ksampler_and_accepts_flux2():
    assert run_matrix._sampling_path({}) == "ksampler"
    assert run_matrix._sampling_path({"sampling_path": "ksampler"}) == "ksampler"
    assert run_matrix._sampling_path({"sampling_path": "flux2"}) == "flux2"


@pytest.mark.parametrize("value", [None, 2, "custom", "FLUX2"])
def test_sampling_path_rejects_unknown_or_non_string_values(value):
    with pytest.raises(ValueError, match="sampling_path must be 'ksampler' or 'flux2'"):
        run_matrix._sampling_path({"sampling_path": value})


@pytest.mark.parametrize("path", ["ksampler", "flux2"])
def test_sampling_path_routes_only_to_selected_callable(path):
    calls = []

    def route(name):
        return lambda: calls.append(name) or f"{name}-result"

    actual = run_matrix._route_sampling(
        {"sampling_path": path},
        ksampler=route("ksampler"),
        flux2=route("flux2"),
    )
    assert actual == f"{path}-result"
    assert calls == [path]


def test_init_options_match_node_defaults():
    assert run_matrix._init_options({}) == {
        "auto_gate": "first_use",
        "lora_low_rss": "auto",
        "slab_weights": "auto",
    }


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("auto_gate", ("first_use", "off")),
        ("lora_low_rss", ("auto", "on", "off")),
        ("slab_weights", ("auto", "on", "off")),
    ],
)
def test_init_options_accept_exact_node_choices(field, values):
    for value in values:
        assert run_matrix._init_options({field: value})[field] == value


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("auto_gate", None, "auto_gate must be one of: 'first_use', 'off'"),
        ("auto_gate", True, "auto_gate must be one of: 'first_use', 'off'"),
        ("auto_gate", "on", "auto_gate must be one of: 'first_use', 'off'"),
        ("auto_gate", "FIRST_USE", "auto_gate must be one of: 'first_use', 'off'"),
        ("lora_low_rss", 1, "lora_low_rss must be one of: 'auto', 'on', 'off'"),
        ("lora_low_rss", "true", "lora_low_rss must be one of: 'auto', 'on', 'off'"),
        ("lora_low_rss", " on", "lora_low_rss must be one of: 'auto', 'on', 'off'"),
        ("slab_weights", False, "slab_weights must be one of: 'auto', 'on', 'off'"),
        ("slab_weights", "first_use", "slab_weights must be one of: 'auto', 'on', 'off'"),
        ("slab_weights", "OFF", "slab_weights must be one of: 'auto', 'on', 'off'"),
    ],
)
def test_init_options_reject_invalid_values(field, value, message):
    with pytest.raises(ValueError) as exc_info:
        run_matrix._init_options({field: value})
    assert str(exc_info.value) == message


def test_model_sampling_sd3_shift_is_optional_and_normalized():
    assert run_matrix._model_sampling_sd3_shift({}) is None
    assert run_matrix._model_sampling_sd3_shift({"model_sampling_sd3_shift": 0}) == 0.0
    assert run_matrix._model_sampling_sd3_shift({"model_sampling_sd3_shift": 5.25}) == 5.25
    assert run_matrix._model_sampling_sd3_shift({"model_sampling_sd3_shift": 100}) == 100.0


@pytest.mark.parametrize(
    "value",
    [None, True, "5", float("nan"), float("inf"), -0.01, 100.01],
)
def test_model_sampling_sd3_shift_rejects_invalid_values(value):
    with pytest.raises(
        ValueError,
        match="model_sampling_sd3_shift must be a finite number from 0 through 100",
    ):
        run_matrix._model_sampling_sd3_shift({"model_sampling_sd3_shift": value})


def test_model_sampling_sd3_patch_is_applied_once_or_left_absent():
    model = object()
    calls = []

    def patch(current, shift):
        calls.append((current, shift))
        return ("patched",)

    assert run_matrix._apply_model_sampling_sd3(model, {}, patch=patch) is model
    assert calls == []
    assert run_matrix._apply_model_sampling_sd3(
        model, {"model_sampling_sd3_shift": 5}, patch=patch
    ) == "patched"
    assert calls == [(model, 5.0)]


def test_latent_shape_defaults_to_current_image_layout():
    settings = {"height": 1024, "width": 768}
    expected = (
        2,
        4,
        128,
        96,
    )
    assert run_matrix._latent_shape(settings, 2) == expected
    assert run_matrix._latent_shape({**settings, "latent_t": 0}, 2) == expected


def test_latent_shape_supports_flux2_channels_and_downscale():
    settings = {
        "height": 1024,
        "width": 768,
        "latent_channels": 128,
        "latent_downscale": 16,
    }
    assert run_matrix._latent_shape(settings, 1) == (1, 128, 64, 48)


def test_benchmark_latent_carries_exact_downscale_into_sampler_path():
    from dgx_monarch.nodes.common import LATENT_DOWNSCALE_METADATA_KEY

    settings = {
        "height": 1024,
        "width": 768,
        "latent_channels": 128,
        "latent_downscale": 16,
    }
    shapes = []

    def zeros(shape):
        shapes.append(shape)
        return "samples"

    latent, downscale = run_matrix._benchmark_latent(
        settings,
        2,
        zeros=zeros,
        metadata_key=LATENT_DOWNSCALE_METADATA_KEY,
    )

    assert shapes == [(2, 128, 64, 48)]
    assert downscale == 16
    assert latent == {
        "samples": "samples",
        LATENT_DOWNSCALE_METADATA_KEY: 16,
    }


def test_latent_shape_supports_video_temporal_axis():
    settings = {
        "height": 720,
        "width": 1280,
        "latent_channels": 16,
        "latent_downscale": 8,
        "latent_t": 25,
    }
    assert run_matrix._latent_shape(settings, 3) == (3, 16, 25, 90, 160)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("latent_channels", True, "latent_channels must be an integer"),
        ("latent_channels", 4.0, "latent_channels must be an integer"),
        ("latent_channels", 0, "latent_channels must be a positive integer"),
        ("latent_channels", -1, "latent_channels must be a positive integer"),
        ("latent_downscale", False, "latent_downscale must be an integer"),
        ("latent_downscale", "8", "latent_downscale must be an integer"),
        ("latent_downscale", 0, "latent_downscale must be a positive integer"),
        ("latent_downscale", -8, "latent_downscale must be a positive integer"),
        ("latent_t", True, "latent_t must be an integer"),
        ("latent_t", 1.5, "latent_t must be an integer"),
        ("latent_t", -1, "latent_t must be a non-negative integer"),
    ],
)
def test_latent_shape_rejects_invalid_settings(field, value, message):
    settings = {"height": 1024, "width": 1024, field: value}
    with pytest.raises(ValueError, match=message):
        run_matrix._latent_shape(settings, 1)


def test_latent_sha256_is_stable_for_equal_logical_contiguous_bytes():
    import torch

    samples = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    noncontiguous = samples.transpose(-1, -2)
    expected = hashlib.sha256(
        noncontiguous.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()

    assert run_matrix._latent_sha256(noncontiguous) == expected
    assert run_matrix._latent_sha256(noncontiguous.clone()) == expected


def test_latent_sha256_is_sensitive_to_raw_dtype_bytes():
    import torch

    values = [0.0, 1.0, -2.5]
    assert run_matrix._latent_sha256(torch.tensor(values, dtype=torch.float32)) != (
        run_matrix._latent_sha256(torch.tensor(values, dtype=torch.float64))
    )


def test_flux2_schedule_forwards_exact_dimensions_and_unwraps_first_output():
    calls = []
    sentinel = object()

    def get_schedule(**kwargs):
        calls.append(kwargs)
        return (sentinel, "ignored")

    actual = run_matrix._flux2_schedule(
        get_schedule, steps=20, width=1024, height=768
    )
    assert actual is sentinel
    assert calls == [{"steps": 20, "width": 1024, "height": 768}]


def test_recycle_mesh_handles_deduplicates_and_reports_success():
    calls = []

    class Handle:
        def recycle(self):
            calls.append("recycle")
            return True

    handle = Handle()
    assert run_matrix._recycle_mesh_handles([handle, handle]) == [
        {"recycled": True}
    ]
    assert calls == ["recycle"]


def test_tracked_mesh_is_owned_before_init_setup_can_fail():
    calls = []
    handle = object()
    handles = []

    def get_mesh_fn(**kwargs):
        calls.append(("get_mesh", kwargs))
        return handle

    class Init:
        def init(self, **kwargs):
            calls.append(("init", kwargs))
            raise RuntimeError("setup failed after handle acquisition")

    settings = {
        "topology": "uly2+fsdp",
        "mode": "cluster",
        "config": "cluster.toml",
        "auto_gate": "off",
        "lora_low_rss": "off",
        "slab_weights": "off",
    }
    with pytest.raises(RuntimeError, match="setup failed"):
        run_matrix._init_tracked_mesh(
            settings,
            handles,
            get_mesh_fn=get_mesh_fn,
            init_node=Init(),
        )

    assert handles == [handle]
    assert calls == [
        (
            "get_mesh",
            {
                "config_path": "cluster.toml",
                "mode": "cluster",
                "gpus_per_host": 0,
            },
        ),
        (
            "init",
            {
                "topology": "uly2+fsdp",
                "mode": "cluster",
                "config_path": "cluster.toml",
                "auto_gate": "off",
                "lora_low_rss": "off",
                "slab_weights": "off",
            },
        ),
    ]


def test_tracked_mesh_records_unexpected_replacement_handle():
    acquired = object()
    replacement = object()
    handles = []

    class Init:
        def init(self, **_kwargs):
            return (SimpleNamespace(handle=replacement),)

    mesh = run_matrix._init_tracked_mesh(
        {"topology": "auto"},
        handles,
        get_mesh_fn=lambda **_kwargs: acquired,
        init_node=Init(),
    )

    assert mesh.handle is replacement
    assert handles == [acquired, replacement]


def test_recycle_mesh_handles_continues_and_sanitizes_failures():
    calls = []

    class Broken:
        def recycle(self):
            calls.append("broken")
            raise RuntimeError("private endpoint must not enter the report")

    class Refused:
        def recycle(self):
            calls.append("refused")
            return False

    assert run_matrix._recycle_mesh_handles([Broken(), Refused()]) == [
        {"recycled": False, "error_type": "RuntimeError"},
        {"recycled": False},
    ]
    assert calls == ["broken", "refused"]


def test_effective_topology_records_degrees_attention_and_reason():
    topology = SimpleNamespace(
        describe=lambda: "uly2+cfg2",
        ulysses=2,
        ring=1,
        cfg=2,
        dp=1,
        fsdp=False,
        world=4,
    )
    actual = run_matrix._topology_provenance(
        topology, sage=True, reason="auto table row 10", attention="TORCH_FLASH"
    )
    assert actual == {
        "name": "uly2+cfg2",
        "ulysses": 2,
        "ring": 1,
        "cfg": 2,
        "dp": 1,
        "fsdp": False,
        "world": 4,
        "attention": "SAGE_AUTO",
        "reason": "auto table row 10",
    }


def test_keep_latent_npy_is_a_noop_when_unset(tmp_path):
    import torch

    results = {"case-a": {"name": "case-a", "latent": torch.zeros(2), "status": "ok"}}
    run_matrix._save_latent_npy(results, "")

    assert list(tmp_path.iterdir()) == []
    assert results["case-a"]["latent"] is not None


def test_keep_latent_npy_writes_expected_arrays_before_the_json_strip(tmp_path):
    import numpy as np
    import torch

    samples = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    results = {
        "case-a": {"name": "case-a", "latent": samples, "status": "ok"},
        "case-b": {"name": "case-b", "status": "N/A"},
    }
    keep_dir = tmp_path / "npy"

    run_matrix._save_latent_npy(results, str(keep_dir))

    saved = np.load(keep_dir / "case-a_latent.npy")
    assert np.array_equal(saved, samples.numpy())
    assert not (keep_dir / "case-b_latent.npy").exists()
    # The function only reads "latent"; the later row.pop("latent", None) in
    # main() strips it regardless of whether the flag was set.
    assert results["case-a"]["latent"] is samples


def test_latent_statistics_accepts_only_finite_tensors():
    import torch

    assert run_matrix._latent_statistics(torch.tensor([1.0, 3.0])) == {
        "mean": 2.0,
        "std": pytest.approx(2**0.5),
    }
    with pytest.raises(ValueError, match="non-finite"):
        run_matrix._latent_statistics(torch.tensor([1.0, float("nan")]))


def test_latent_statistics_rejects_nonfinite_derived_statistics():
    import torch

    with pytest.raises(ValueError, match="statistics are non-finite"):
        run_matrix._latent_statistics(torch.tensor([3.4e38, -3.4e38]))


def test_not_available_result_never_renders_exception_text():
    class PrivateFailure(RuntimeError):
        def __str__(self):
            raise AssertionError("exception text must not be rendered")

        def __repr__(self):
            raise AssertionError("exception repr must not be rendered")

    row = run_matrix._not_available_result(
        "cell",
        {"name": "cell"},
        {"topology": "cfg2"},
        PrivateFailure(),
    )

    assert row == {
        "name": "cell",
        "status": "N/A",
        "reason": "case execution failed",
        "error_type": "PrivateFailure",
        "error_site": "",
        "settings": {"topology": "cfg2", "name": "cell"},
    }


def test_not_available_result_names_the_raise_site_and_nothing_else():
    # Four gate legs of the 2026-09-07 cfg matrix read "N/A: ValueError" and
    # nothing more, so the scorer's own raise could not be told from a
    # worker's. The row names the innermost frame's file and line beside the
    # type, never the message.
    def inner():
        raise ValueError("private text")

    try:
        inner()
    except ValueError as exc:
        row = run_matrix._not_available_result("cell", {"name": "cell"}, {}, exc)
    assert row["error_type"] == "ValueError"
    assert row["error_site"].startswith("test_benchmark_provenance.py:")
    assert "private text" not in json.dumps(row)
