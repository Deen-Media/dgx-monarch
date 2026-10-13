"""Capture real policy-aware Worker argv using only a temporary procfs fixture."""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import pytest

from dgx_monarch.cli import update_installation as install
from dgx_monarch.cli import update_installation_remote as remote
from dgx_monarch.cli.lifecycle_systemd import systemd_worker_unit
from dgx_monarch.cli.systemd_unit import quote
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.monarch_launch import worker_policy_args

TOKEN = "u-111111111111-2222222222222222"  # noqa: S105


@pytest.fixture
def capture(tmp_path, monkeypatch):
    unit = tmp_path / "systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    source = tmp_path / "site"
    (source / "dgx_monarch").mkdir(parents=True)
    proc = tmp_path / "proc/123"
    proc.mkdir(parents=True)
    (proc / "exe").symlink_to(sys.executable)
    (proc / "stat").write_text("123 (worker) " + " ".join(["S", *(["0"] * 18), "42"]))
    (proc / "environ").write_bytes(f"PYTHONPATH={source}\0".encode())
    (proc / "task").mkdir()
    group = tmp_path / "cgroup/user.slice/test"
    group.mkdir(parents=True)
    (group / "cgroup.procs").write_text("123\n")
    state = {"MainPID": "123", "ActiveState": "active", "SubState": "running", "Job": "", "ControlGroup": "/user.slice/test"}
    monkeypatch.setattr(remote, "unit_state", lambda *_args: dict(state))

    def mapped_path(value):
        if str(value) == "/proc":
            return tmp_path / "proc"
        if str(value) == "/sys/fs/cgroup":
            return tmp_path / "cgroup"
        return Path(value)

    mapped_path.home = Path.home
    monkeypatch.setattr(remote, "Path", mapped_path)

    def run(unit_text, argv):
        unit.write_text(unit_text)
        unit.chmod(0o600)
        (proc / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
        return remote.snapshot(unit, tmp_path / "absent", sys.executable, "tcp://127.0.0.1:26600")

    return run, unit, proc, group


def setup(native=False):
    host = HostConfig(name="head", address="tcp://127.0.0.1:26600")
    return ClusterConfig(hosts=[host], python_bin=sys.executable, rdma_latent_return=native), host


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("hardened", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_snapshot_accepts_only_unit_bound_worker_command(capture, native, hardened, legacy):
    run, _, _, _ = capture
    config, host = setup(native)
    unit = systemd_worker_unit(config, host, managed_source=True, update_token=TOKEN,
                               site_packages="/venv/lib/python3.12/site-packages" if hardened else None)
    policy = worker_policy_args(config, host)
    if legacy:
        unit = unit.replace(" " + " ".join(quote(arg) for arg in policy) + "\n", "\n")
        policy = []
    argv = [sys.executable, *(["-S", "-P", "-s", "-B"] if hardened else []),
            "-m", "dgx_monarch.cli.worker_loop", "--address", host.address, *policy]
    result = run(unit, argv)
    assert result["pid"] == "123" and result["birth"] == "42"
    assert result["sha256"] == hashlib.sha256(unit.encode()).hexdigest()


@pytest.mark.parametrize("drift", ["policy", "native", "missing", "extra", "address", "python_flags", "exe"])
def test_snapshot_rejects_process_drift_from_exact_policy_unit(capture, drift):
    run, _, proc, _ = capture
    config, host = setup()
    unit = systemd_worker_unit(config, host, managed_source=True, update_token=TOKEN)
    argv = [sys.executable, "-m", "dgx_monarch.cli.worker_loop", "--address", host.address,
            *worker_policy_args(config, host)]
    if drift == "policy":
        argv[-1] = "e30="
    elif drift == "native":
        argv.append("--native-rdma")
    elif drift == "missing":
        del argv[-2:]
    elif drift == "extra":
        argv.extend(["--unknown", "value"])
    elif drift == "address":
        argv[4] = "tcp://127.0.0.1:26601"
    elif drift == "python_flags":
        argv.insert(1, "-O")
    else:
        (proc / "exe").unlink()
        (proc / "exe").symlink_to("/bin/false")
    with pytest.raises(ValueError, match="Worker command"):
        run(unit, argv)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("hardened", [False, True])
def test_exact_policy_and_legacy_update_unit_recognition(monkeypatch, native, legacy, hardened):
    config, host = setup(native)
    monkeypatch.setattr(install.lifecycle, "_is_local", lambda _host: False)
    site = "/home/user/.local/share/dgx-monarch/src"
    dependencies = "/venv/lib/python3.12/site-packages" if hardened else ""
    unit = systemd_worker_unit(config, host, managed_source=True, update_token=TOKEN, site_packages=dependencies or None)
    if legacy:
        unit = unit.replace(" " + " ".join(quote(arg) for arg in worker_policy_args(config, host)) + "\n", "\n")
    row = {"site": site, "site_packages": dependencies, "home": "/home/user", "user": "user",
           "live": {"kind": "symlink", "target": f"/home/user/.local/share/dgx-monarch/releases/{TOKEN}/site"}}
    def recognize(text):
        row.update(unit=base64.b64encode(text.encode()).decode(), sha256=hashlib.sha256(text.encode()).hexdigest())
        return install.recognized_unit(config, host, row)
    assert recognize(unit)
    assert not recognize(unit.replace("RestartSec=3", "RestartSec=8"))
    if not legacy:
        token = worker_policy_args(config, host)[1]
        # "e30=" parses as the default policy but is not its canonical encoding.
        assert not recognize(unit.replace(token, "e30="))
        assert not recognize(unit.replace(' "--runtime-policy"', ' "--native-rdma" "--runtime-policy"'))
        # A flipped native flag is an earlier policy, not tampering: capture
        # accepts it so the update can render the current one.
        flipped = unit.replace(' "--native-rdma"', '') if native else unit.replace(f'"{token}"', f'"{token}" "--native-rdma"')
        assert recognize(flipped)
        assert not recognize(flipped.replace("RestartSec=3", "RestartSec=8"))


@pytest.mark.parametrize("rdma", [
    pytest.param({"peer_device_affinity": "groups:a,b"}, id="single_group"),
    pytest.param({"target": "gpu:07"}, id="leading_zero"),
    pytest.param({"peer_device_affinity": "groups:a|b"}, id="still_valid"),
])
def test_a_unit_rendered_from_a_value_config_loading_now_refuses_is_stale(monkeypatch, rdma):
    """rev-pr-c 2: the update must re-render such a unit, or the operator's fix loops on
    prior_worker_drift. Recognition checks the token's shape and encoding, never its values."""
    import json
    from dataclasses import replace

    from dgx_monarch.monarch_policy import MonarchPolicy, RdmaPolicy

    config, host = setup()
    monkeypatch.setattr(install.lifecycle, "_is_local", lambda _host: False)
    # The dataclass skips parse_policy, as the earlier release that rendered this unit did.
    earlier = replace(config, monarch=MonarchPolicy(rdma=RdmaPolicy(**rdma)))
    unit = systemd_worker_unit(earlier, host, managed_source=True, update_token=TOKEN)
    row = {"site": "/home/user/.local/share/dgx-monarch/src", "site_packages": "", "home": "/home/user",
           "user": "user", "live": {"kind": "symlink",
                                    "target": f"/home/user/.local/share/dgx-monarch/releases/{TOKEN}/site"}}

    def recognize(text):
        row.update(unit=base64.b64encode(text.encode()).decode(), sha256=hashlib.sha256(text.encode()).hexdigest())
        return install.recognized_unit(config, host, row)

    assert recognize(systemd_worker_unit(config, host, managed_source=True, update_token=TOKEN))
    assert recognize(unit)
    token = worker_policy_args(earlier, host)[1]
    payload = json.loads(base64.urlsafe_b64decode(token))
    spaced = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
    extra = base64.urlsafe_b64encode(json.dumps({**payload, "extra": 1}, sort_keys=True,
                                                separators=(",", ":")).encode()).decode()
    for tampered in (spaced, extra):
        assert not recognize(unit.replace(token, tampered))


def test_capture_binds_live_argv_to_its_own_unit_across_a_policy_rerender(capture, monkeypatch):
    """The update captures a stale unit with the argv it started; after the
    re-render the next capture binds the new argv to the new unit, never one
    unit to the other's argv."""
    from dataclasses import replace

    from dgx_monarch.monarch_policy import parse_policy

    run, _, _, _ = capture
    monkeypatch.setattr(install.lifecycle, "_is_local", lambda _host: False)
    old, host = setup()
    new = replace(old, rdma_latent_return=True, monarch=parse_policy({"diagnostics": {"metrics": True}}))
    stale = systemd_worker_unit(old, host, managed_source=True, update_token=TOKEN)
    rendered = systemd_worker_unit(new, host, managed_source=True, update_token="u-333333333333-4444444444444444")  # noqa: S106

    def argv(config):
        return [sys.executable, "-m", "dgx_monarch.cli.worker_loop", "--address", host.address,
                *worker_policy_args(config, host)]

    captured = run(stale, argv(old))
    row = {**captured, "site": "/home/user/.local/share/dgx-monarch/src", "home": "/home/user", "user": "user",
           "live": {"kind": "symlink", "target": f"/home/user/.local/share/dgx-monarch/releases/{TOKEN}/site"}}
    assert install.recognized_unit(new, host, row)
    assert run(rendered, argv(new))["sha256"] == hashlib.sha256(rendered.encode()).hexdigest()
    for unit, command in ((rendered, argv(old)), (stale, argv(new))):
        with pytest.raises(ValueError, match="Worker command"):
            run(unit, command)


def test_snapshot_preserves_children_and_cgroup_refusals(capture):
    run, _, proc, group = capture
    config, host = setup()
    unit = systemd_worker_unit(config, host, managed_source=True)
    argv = [sys.executable, "-m", "dgx_monarch.cli.worker_loop", "--address", host.address, *worker_policy_args(config, host)]
    (group / "cgroup.procs").write_text("123\n124\n")
    with pytest.raises(ValueError, match="cgroup contains"):
        run(unit, argv)
    (group / "cgroup.procs").write_text("123\n")
    task = proc / "task/123"
    task.mkdir()
    (task / "children").write_text("124")
    with pytest.raises(ValueError, match="child actors"):
        run(unit, argv)


def test_snapshot_refuses_unit_change_during_capture(capture, monkeypatch):
    run, _, _, _ = capture
    config, host = setup()
    unit = systemd_worker_unit(config, host, managed_source=True)
    argv = [sys.executable, "-m", "dgx_monarch.cli.worker_loop", "--address", host.address, *worker_policy_args(config, host)]
    read = remote.read
    calls = 0
    def changing(path):
        nonlocal calls
        calls += 1
        return read(path) + (b"# changed\n" if calls > 1 else b"")
    monkeypatch.setattr(remote, "read", changing)
    with pytest.raises(ValueError, match="changed during capture"):
        run(unit, argv)


@pytest.mark.parametrize("tail", [
    ' "--native-rdma"',
    ' "--runtime-policy" "e30=" "--runtime-policy" "e30="',
    ' "--runtime-policy" "e30=" "--unknown"',
    ' "--runtime-policy" "bad token"',
    ' "--runtime-policy" "e30=" "--native-rdma" "--native-rdma"',
])
def test_snapshot_policy_parser_refuses_noncanonical_unit_suffix(tail):
    with pytest.raises(ValueError, match="policy arguments"):
        remote.worker_suffix(('ExecStart="python"' + tail + '\n').encode(), "tcp://127.0.0.1:26600")


@pytest.mark.parametrize("unit", [b"", b"ExecStart=one\nExecStart=two\n"])
def test_snapshot_policy_parser_refuses_ambiguous_execstart(unit):
    with pytest.raises(ValueError, match="command is ambiguous"):
        remote.worker_suffix(unit, "tcp://127.0.0.1:26600")
