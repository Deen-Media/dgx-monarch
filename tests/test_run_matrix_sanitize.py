"""sanitize_report: the auto-redaction gate for non-gitignored benchmark reports."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dgxm_benchmark_run_matrix_sanitize", REPO / "benchmark" / "run_matrix.py"
)
assert SPEC and SPEC.loader
run_matrix = importlib.util.module_from_spec(SPEC)
_argv_before = list(sys.argv)
_path_before = list(sys.path)
SPEC.loader.exec_module(run_matrix)
assert sys.argv == _argv_before
assert sys.path == _path_before


def _representative_payload() -> dict:
    return {
        "provenance": {
            "generated_at": "2026-07-11T00:00:00Z",
            "host": "examplebox-f00f",
            "comfy_dir": "/home/user/ComfyUI",
            "matrix": {
                "path": "/home/user/dgx-monarch/benchmark/matrix.toml",
                "size_bytes": 42,
                "sha256": "abc123",
            },
            "config": {"path": "/home/user/dgx-monarch/cluster.toml", "size_bytes": 10},
            "gpu_hardware": [
                "NVIDIA GB10, GPU-deadbeef-0000-0000-0000-000000000000, 122880",
            ],
            "case_artifacts": {
                "cell": {
                    "unet": {
                        "name": "model.safetensors",
                        "path": "/home/user/ComfyUI/models/diffusion_models/model.safetensors",
                    },
                    "text_encoder": {
                        "name": "text.safetensors",
                        "path": "/home/user/ComfyUI/models/text_encoders/text.safetensors",
                    },
                    "loras": [
                        {"name": "style.safetensors",
                         "path": "/home/user/ComfyUI/models/loras/style.safetensors"},
                    ],
                },
            },
        },
        "results": [{
            "name": "cell",
            "status": "ok",
            "median_s": 1.23,
            "settings": {"config": "/home/user/dgx-monarch/cluster.toml"},
        }],
    }


def test_sanitize_report_redacts_host_uuid_and_paths_without_mutating_input():
    payload = _representative_payload()
    original = copy.deepcopy(payload)

    sanitized = run_matrix.sanitize_report(payload)

    assert payload == original
    prov = sanitized["provenance"]
    assert prov["host"] == "[hostname redacted]"
    assert prov["gpu_hardware"] == ["NVIDIA GB10, [UUID redacted], 122880"]
    assert prov["comfy_dir"] == "ComfyUI"
    assert prov["matrix"]["path"] == "matrix.toml"
    assert prov["matrix"]["sha256"] == "abc123"  # untouched fields survive
    assert prov["config"]["path"] == "cluster.toml"
    artifacts = prov["case_artifacts"]["cell"]
    assert artifacts["unet"]["path"] == "model.safetensors"
    assert artifacts["unet"]["name"] == "model.safetensors"  # names are not paths
    assert artifacts["text_encoder"]["path"] == "text.safetensors"
    assert artifacts["loras"][0]["path"] == "style.safetensors"
    assert sanitized["results"][0]["settings"]["config"] == "cluster.toml"
    assert sanitized["results"][0]["median_s"] == 1.23
    assert sanitized["sanitization"] == (
        "auto-redacted (host/uuid/private paths) before non-gitignored write")


def test_sanitize_report_leaves_unknown_sentinels_alone():
    payload = _representative_payload()
    payload["provenance"]["config"] = {"path": "unknown", "size_bytes": "unknown"}
    payload["provenance"]["gpu_hardware"] = "unknown"

    sanitized = run_matrix.sanitize_report(payload)

    assert sanitized["provenance"]["config"]["path"] == "unknown"
    assert sanitized["provenance"]["gpu_hardware"] == "unknown"


def test_sanitize_report_redacts_generic_absolute_and_relative_path_fields():
    payload = _representative_payload()
    payload["provenance"]["matrix"]["content"] = (
        'checkpoint = "/mnt/private/models/model.safetensors"\n'
    )
    payload["provenance"]["custom"] = {
        "cache_dir": r"C:\\Users\\user\\private-cache",
        "unexpected": "/srv/private/operator-only.json",
        "diagnostic": "failed under /home/user/ComfyUI while probing",
        "public_url": "https://example.com/public/report.json",
    }
    payload["results"][0]["settings"]["config"] = "private/rig/cluster.toml"

    sanitized = run_matrix.sanitize_report(payload)

    custom = sanitized["provenance"]["custom"]
    assert custom["cache_dir"] == "private-cache"
    assert custom["unexpected"] == "operator-only.json"
    assert custom["diagnostic"] == "failed under ComfyUI while probing"
    assert custom["public_url"] == "https://example.com/public/report.json"
    assert sanitized["provenance"]["matrix"]["content"] == (
        'checkpoint = "model.safetensors"\n'
    )
    assert sanitized["results"][0]["settings"]["config"] == "cluster.toml"


def test_private_path_secrets_covers_comfy_config_artifacts_and_generic_paths():
    payload = _representative_payload()
    payload["extra"] = {"value": "/opt/private/campaign.json"}

    secrets = run_matrix.private_path_secrets(payload)

    assert "/home/user/ComfyUI" in secrets
    assert "/home/user/dgx-monarch/cluster.toml" in secrets
    assert "/home/user/ComfyUI/models/loras/style.safetensors" in secrets
    assert "/opt/private/campaign.json" in secrets


def test_gpu_uuids_extracts_the_uuid_field_only():
    rows = [
        "NVIDIA GB10, GPU-aaaa, 122880",
        "NVIDIA GB10, GPU-bbbb, 122880",
    ]
    assert run_matrix._gpu_uuids(rows) == ["GPU-aaaa", "GPU-bbbb"]
    assert run_matrix._gpu_uuids("unknown") == []
    assert run_matrix._gpu_uuids([]) == []
    assert run_matrix._gpu_uuids(["no-comma-row"]) == []


def test_leaked_secrets_flags_present_values_and_ignores_absent_or_falsy():
    assert run_matrix.leaked_secrets(
        "report body mentions examplebox-f00f in passing", ["examplebox-f00f"]
    ) == ["examplebox-f00f"]
    assert run_matrix.leaked_secrets(
        "clean report, nothing sensitive here", ["examplebox-f00f", "GPU-aaaa"]
    ) == []
    assert run_matrix.leaked_secrets("anything at all", ["", None]) == []


def test_sanitize_report_output_carries_no_leaked_secrets():
    payload = _representative_payload()
    host = payload["provenance"]["host"]
    uuids = run_matrix._gpu_uuids(payload["provenance"]["gpu_hardware"])

    sanitized = run_matrix.sanitize_report(payload)
    serialized = json.dumps(sanitized)

    assert run_matrix.leaked_secrets(
        serialized,
        [host, *uuids, *run_matrix.private_path_secrets(payload)],
    ) == []


def test_serialize_report_rejects_a_path_leak_before_the_write_call():
    payload = _representative_payload()
    config_path = payload["results"][0]["settings"]["config"]

    with pytest.raises(run_matrix.ReportRedactionError) as exc_info:
        run_matrix._serialize_report(payload, [config_path])

    assert exc_info.value.count == 1


def test_serialize_report_allows_explicit_full_fidelity_policy_without_forbidden_values():
    payload = _representative_payload()

    serialized = run_matrix._serialize_report(payload, [])

    assert json.loads(serialized) == payload


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_serialize_report_rejects_nonfinite_json_numbers(value):
    payload = _representative_payload()
    payload["results"][0]["median_s"] = value

    with pytest.raises(run_matrix.ReportSerializationError, match="non-finite"):
        run_matrix._serialize_report(payload, [])
