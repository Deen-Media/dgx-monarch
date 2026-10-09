"""Explicit-host and disclosure contracts for guided setup discovery."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dgx_monarch.cli import setup_config_io, setup_probe, setup_probe_script
from dgx_monarch.config import ClusterConfig, HostConfig


def _payload(*, artifact_hash: str = "a" * 64) -> dict[str, object]:
    return {
        "schema": 1,
        "python_available": True,
        "python_version": "3.12.3",
        "torch_version": "2.8.0+cu128",
        "cuda_available": True,
        "gpu_count": 1,
        "integrated": True,
        "gpu_models": ["NVIDIA GB10"],
        "torchmonarch_version": "0.6.0",
        "comfy_exists": True,
        "comfy_runtime_marker": True,
        "comfy_git": True,
        "comfy_dirty": False,
        "comfy_commit": "b" * 40,
        "fabric_interface_count": 2,
        "link_layers": ["Ethernet"],
        "rsync_available": True,
        "systemd_user_available": True,
        "linger_enabled": True,
        "service_installed": False,
        "systemd_service_active": False,
        "worker_process_active": False,
        "worker_listener_active": False,
        "service_active": False,
        "artifacts": [
            {
                "ordinal": 1,
                "exists": True,
                "regular": True,
                "size": 123,
                "sha256": artifact_hash,
            }
        ],
    }


def _readiness(probe: setup_probe.HostProbe) -> list[str]:
    blockers, _warnings = setup_config_io.setup_readiness(
        probes=(probe,),
        expected_gpus=(1,),
        artifacts=(),
        fabric_profile="single-node",
        install_service=False,
        start_workers=False,
        verify=False,
        transport_security="trusted_fabric",
    )
    return blockers


@pytest.mark.parametrize(
    "value",
    ["/etc/passwd", "../model.safetensors", "models/../secret", "~/model", "a\\b", "a//b", "a/./b"],
)
def test_artifact_paths_must_be_canonical_relative_names(value):
    with pytest.raises(ValueError, match="artifact paths"):
        setup_probe.validate_artifact_paths([value])


def test_artifact_paths_are_bounded_and_deduplicated():
    assert setup_probe.validate_artifact_paths(["models/checkpoint.safetensors", "models/checkpoint.safetensors"]) == (
        "models/checkpoint.safetensors",
    )
    with pytest.raises(ValueError, match="at most"):
        setup_probe.validate_artifact_paths([f"models/{index}" for index in range(65)])


def test_probe_contacts_exact_candidates_once_and_public_rows_are_sanitized():
    hosts = (
        HostConfig("secret-spark-a", "tcp://10.20.30.40:26600"),
        HostConfig("secret-spark-b", "tcp://10.20.30.41:26600"),
    )
    config = ClusterConfig(
        hosts=hosts,
        comfy_dir="/home/operator/private/ComfyUI",
        python_bin="/home/operator/venv/bin/python",
    )
    contacted: list[str] = []

    def runner(_config, host, script, *, timeout):
        contacted.append(host.name)
        assert timeout == 300
        assert "git" in script and "infiniband" in script and "systemctl" in script
        assert "--untracked-files=all" in script and '"ls-files", "-v"' in script
        assert 'GIT_OPTIONAL_LOCKS"] = "0"' in script and "core.fsmonitor=false" in script
        stdout = setup_probe._MARKER + json.dumps(_payload()) + "\n"
        return subprocess.CompletedProcess([], 0, stdout, "secret stderr")

    probes = setup_probe.probe_hosts(config, artifacts=["models/checkpoint.safetensors"], runner=runner)

    assert contacted == [host.name for host in hosts]
    assert [probe.ordinal for probe in probes] == [1, 2]
    assert all(probe.reachable for probe in probes)
    public = json.dumps([probe.as_dict() for probe in probes], sort_keys=True)
    for secret in ("secret-spark", "10.20.30", "/home/operator", "secret stderr"):
        assert secret not in public
    assert probes[0].model_fingerprint == probes[1].model_fingerprint


def test_transport_exception_and_malformed_output_collapse_to_stable_failures():
    host = HostConfig("secret-node", "tcp://10.0.0.8:26600")
    config = ClusterConfig(hosts=(host,))

    def unavailable(*_args, **_kwargs):
        raise OSError("ssh to secret-node at 10.0.0.8 failed")

    failed = setup_probe.probe_hosts(config, runner=unavailable)[0]
    assert failed.failure == "transport_unavailable"
    assert "secret" not in json.dumps(failed.as_dict())

    malformed = setup_probe.parse_probe_output(
        setup_probe._MARKER + '{"schema":1,"python_available":true,"gpu_models":"bad"}',
        1,
        0,
    )
    assert malformed.failure == "invalid_response"

    malformed_models = _payload()
    malformed_models["gpu_models"] = "NVIDIA GB10"
    parsed = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(malformed_models), 1, 1)
    assert parsed.failure == "invalid_response"

    missing_version = _payload()
    missing_version["python_version"] = None
    parsed = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(missing_version), 1, 1)
    assert parsed.failure == "invalid_response"


def test_python_unavailable_is_reachable_but_readiness_specific():
    raw = setup_probe_script._unavailable_payload(0)
    parsed = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(raw), 1, 0)
    assert parsed.failure == "python_unavailable"
    assert parsed.reachable is True
    assert parsed.python_available is False
    assert _readiness(parsed) == ["host_1_python_unavailable"]


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("3.10.14", ["host_1_python_version_unsupported"]),
        ("3.11.0", []),
    ],
)
def test_readiness_requires_python_3_11_or_newer(version: str, expected: list[str]):
    raw = _payload()
    raw["python_version"] = version
    parsed = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(raw), 1, 1)

    assert _readiness(parsed) == expected


def test_probe_rejects_incoherent_aggregate_worker_activity():
    raw = _payload()
    raw["worker_process_active"] = True
    raw["service_active"] = False

    parsed = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(raw), 1, 1)

    assert parsed.failure == "invalid_response"


def test_unknown_listener_state_keeps_aggregate_activity_unknown():
    raw = _payload()
    raw["worker_listener_active"] = None
    raw["service_active"] = None

    parsed = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(raw), 1, 1)

    assert parsed.worker_listener_active is None
    assert parsed.service_active is None


def test_artifact_comparison_requires_every_host_and_exact_size_hash():
    first = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(_payload()), 1, 1)
    second = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(_payload(artifact_hash="c" * 64)), 2, 1)
    mismatch = setup_probe.compare_artifacts((first, second), 1)[0]
    assert mismatch.state == "mismatch"
    assert mismatch.sha256 is None

    same = setup_probe.compare_artifacts((first, first), 1)[0]
    assert same.state == "match"
    assert same.sha256 == "a" * 64


def test_fabric_recommendation_recognizes_the_validated_dual_spark_cohort():
    probes = tuple(
        setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(_payload()), ordinal, 1) for ordinal in (1, 2)
    )
    assert setup_config_io.recommend_fabric(probes) == "dgx-spark-pair"

    generic_payload = _payload()
    generic_payload["gpu_models"] = ["Other integrated GPU"]
    generic = setup_probe.parse_probe_output(setup_probe._MARKER + json.dumps(generic_payload), 2, 1)
    assert setup_config_io.recommend_fabric((probes[0], generic)) == "generic-roce"


def test_probe_script_never_embeds_an_artifact_as_shell_syntax():
    selected = setup_probe.validate_artifact_paths(["models/a $(touch nope).safetensors"])
    script = setup_probe.build_probe_script("~/venv/bin/python", "~/Comfy UI", selected)
    assert "$(touch nope)" not in script
    assert "DGXM_SETUP_ARTIFACTS=" in script
    assert "bash -c" not in script
    assert 'value.startswith("tcp://")' in script
    assert 'exec "$PYBIN" -I -B -' in script


def test_generated_probe_script_executes_and_hashes_only_named_artifact(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    artifact = comfy / "models" / "selected.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"selected bytes")
    (comfy / "models" / "not-selected.bin").write_bytes(b"ignore me")
    script = setup_probe.build_probe_script(sys.executable, str(comfy), ("models/selected.bin",))

    completed = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=30)
    parsed = setup_probe.parse_probe_output(completed.stdout, 1, 1)

    assert completed.returncode == 0, completed.stderr
    assert parsed.reachable is True
    assert parsed.python_available is True
    assert parsed.comfy_exists is True
    assert parsed.comfy_runtime_marker is False
    assert parsed.artifacts[0].size == len(b"selected bytes")
    assert parsed.artifacts[0].sha256 == hashlib.sha256(b"selected bytes").hexdigest()


def test_generated_probe_ignores_inherited_git_repository_overrides(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    alternate = tmp_path / "alternate"
    for root, content in ((comfy, "real\n"), (alternate, "alternate\n")):
        root.mkdir()
        (root / "tracked.txt").write_text(content, encoding="utf-8")
        (root / "comfy").mkdir()
        (root / "comfy" / "sd.py").write_text("# minimal runtime marker\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "tracked.txt", "comfy/sd.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Setup Test",
                "-c",
                "user.email=setup" + "@example.invalid",
                "commit",
                "-qm",
                "initial",
            ],
            check=True,
        )
    expected = subprocess.run(
        ["git", "-C", str(comfy), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    alternate_head = subprocess.run(
        ["git", "-C", str(alternate), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    assert expected != alternate_head
    environment = dict(os.environ, GIT_DIR=str(alternate / ".git"), GIT_WORK_TREE=str(comfy))

    completed = subprocess.run(
        ["bash", "-s"],
        input=setup_probe.build_probe_script(sys.executable, str(comfy), ()),
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    parsed = setup_probe.parse_probe_output(completed.stdout, 1, 0)

    assert completed.returncode == 0, completed.stderr
    assert parsed.comfy_runtime_marker is True
    assert parsed.comfy_git is True
    assert parsed.comfy_commit == expected


def test_generated_probe_rejects_an_arbitrary_clean_git_repo_as_comfy(tmp_path: Path):
    root = tmp_path / "not-comfy"
    root.mkdir()
    (root / "tracked.txt").write_text("not ComfyUI\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Setup Test",
            "-c",
            "user.email=setup@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )

    completed = subprocess.run(
        ["bash", "-s"],
        input=setup_probe.build_probe_script(sys.executable, str(root), ()),
        capture_output=True,
        text=True,
        timeout=30,
    )
    parsed = setup_probe.parse_probe_output(completed.stdout, 1, 0)

    assert completed.returncode == 0, completed.stderr
    assert parsed.comfy_exists is True
    assert parsed.comfy_git is True
    assert parsed.comfy_dirty is False
    assert parsed.comfy_runtime_marker is False
    blockers = _readiness(parsed)
    assert "host_1_comfy_runtime_marker_missing" in blockers
    assert "host_1_comfy_missing" not in blockers
    assert "host_1_comfy_not_git" not in blockers


def test_generated_probe_treats_a_wildcard_bind_as_worker_port_activity(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    port = 26600
    proc_net = tmp_path / "tcp"
    proc_net.write_text(
        f"  sl  local_address rem_address   st\n   0: 00000000:{port:04X} 00000000:0000 0A\n",
        encoding="utf-8",
    )
    script = setup_probe.build_probe_script(
        sys.executable,
        str(comfy),
        (),
        worker_address=f"tcp://127.0.0.1:{port}",
    ).replace('"/proc/net/tcp"', repr(str(proc_net)))
    completed = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=30)

    parsed = setup_probe.parse_probe_output(completed.stdout, 1, 0)

    assert completed.returncode == 0, completed.stderr
    assert parsed.worker_listener_active is True
    assert parsed.service_active is True


def test_generated_probe_treats_dual_stack_wildcard_as_ipv4_activity(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    port = 26600
    proc4 = tmp_path / "tcp"
    proc6 = tmp_path / "tcp6"
    proc4.write_text("  sl  local_address rem_address   st\n", encoding="utf-8")
    proc6.write_text(
        f"  sl  local_address rem_address   st\n   0: {'0' * 32}:{port:04X} {'0' * 32}:0000 0A\n",
        encoding="utf-8",
    )
    script = setup_probe.build_probe_script(
        sys.executable,
        str(comfy),
        (),
        worker_address=f"tcp://127.0.0.1:{port}",
    )
    script = script.replace('"/proc/net/tcp6"', repr(str(proc6))).replace('"/proc/net/tcp"', repr(str(proc4)))
    completed = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=30)

    parsed = setup_probe.parse_probe_output(completed.stdout, 1, 0)

    assert completed.returncode == 0, completed.stderr
    assert parsed.worker_listener_active is True
    assert parsed.service_active is True
