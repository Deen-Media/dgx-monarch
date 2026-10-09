"""Regressions for config, lifecycle, and telemetry security boundaries."""
from __future__ import annotations

import io
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.actor.comfy_bridge import _uma_memory_defaults
from dgx_monarch.actor.worker_env import setup_impl
from dgx_monarch.cli import lifecycle, lifecycle_systemd, worker_loop
from dgx_monarch.cli import main as cli
from dgx_monarch.cli.lifecycle_generation import (
    generation_prestart_argv,
    generation_quarantine_argv,
    require_generation_shell,
    systemd_generation_invalidator,
)
from dgx_monarch.cli.lifecycle_lock import locked_script
from dgx_monarch.config import ClusterConfig, ClusterConfigError, HostConfig, load_cluster_config
from dgx_monarch.config_schema import validate_fabric_env
from dgx_monarch.nodes.fleet import build_jobs
from dgx_monarch.nodes.samplers import DGXMonarchKSamplerPipeline


@pytest.fixture(autouse=True)
def _systemd_unit_renderer_uses_the_declared_remote_host(monkeypatch):
    """Answer the ``worker`` fixture host as remote without a lookup.

    Unit-byte tests render units for that configured remote fabric host and do
    not test hostname resolution, so a slow resolver must not delay every
    ownership test. Every other host reaches the real ``is_local``.
    """
    real_is_local = lifecycle_systemd.is_local

    def locality(host):
        if host.name == "worker" and host.address.startswith("tcp://10.0.0.2:"):
            return False
        return real_is_local(host)

    monkeypatch.setattr(lifecycle_systemd, "is_local", locality)


@pytest.mark.parametrize("local", [False, True])
def test_systemd_renderer_preserves_local_source_and_home_python(monkeypatch, local):
    from dgx_monarch.cli import lifecycle_host

    monkeypatch.setattr(lifecycle_host.socket, "gethostname", lambda: "unit-test-local")
    monkeypatch.setattr(lifecycle_host.socket, "getfqdn", lambda: "unit-test-local")
    monkeypatch.setattr(
        lifecycle_host.socket, "getaddrinfo",
        lambda *_a, **_k: pytest.fail("unit rendering must not query DNS for these hosts"),
    )
    monkeypatch.setattr(lifecycle_systemd, "package_src_dir", lambda: Path("/opt/node pack/src"))
    host = HostConfig("unit-test-local" if local else "worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), python_bin="~/venv dir/python%x",
                           transport_security="trusted_fabric")

    unit = lifecycle._systemd_worker_unit(config, host)

    source = "/opt/node pack/src" if local else "%h/.local/share/dgx-monarch/src"
    assert f'Environment="PYTHONPATH={source}"' in unit
    assert 'ExecStart="%h/venv dir/python%%x" "-m"' in unit


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
        '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n'
        f"{extra}"
    )
    return path


@pytest.mark.parametrize("key", [
    "lora_low_rss", "mmap_fallback", "load_profile", "compile_dit",
    "disable_pinned_memory", "disable_async_offload", "disable_smart_memory",
])
def test_worker_boolean_strings_are_rejected(tmp_path: Path, key: str):
    path = _write_config(tmp_path, f'\n[worker_args]\n{key} = "off"\n')
    with pytest.raises(ClusterConfigError, match="boolean true/false"):
        load_cluster_config(path)


def test_worker_slab_off_string_is_rejected_at_config_and_actor(tmp_path: Path):
    path = _write_config(tmp_path, '\n[worker_args]\nslab_weights = "off"\n')
    with pytest.raises(ClusterConfigError, match="true, false"):
        load_cluster_config(path)
    with pytest.raises(ClusterConfigError, match="true, false"):
        _uma_memory_defaults({"slab_weights": "off"}, integrated=False)


def test_worker_numeric_policy_is_bounded_and_normalized(tmp_path: Path):
    path = _write_config(
        tmp_path,
        "\n[worker_args]\nreserve_vram_gb = 8\nswap_verify = -1\n"
        'slab_weights = "auto"\n',
    )
    args = load_cluster_config(path).worker_args
    assert args == {"reserve_vram_gb": 8.0, "swap_verify": -1, "slab_weights": "auto"}


# test_config.py's test_cluster_toml_refusals table holds the unspecified- and
# multicast-bind cases for worker and client binds; this file keeps the CLI-argv case.
def test_worker_loop_direct_cli_rejects_unspecified_bind(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["dgxm-worker", "--address", "tcp://0.0.0.0:26600"])
    with pytest.raises(SystemExit):
        worker_loop.main()


# The IPv4-mapped spellings of an unspecified, a loopback and a multicast address.
# Python 3.12.3's IPv6 objects answer False to is_unspecified/is_loopback/is_multicast
# for all three; on 3.11.14 and 3.13.11 each answers True where its IPv4 form does
# (checked 2026-10-07). So the bind checks unwrap them (config.unmapped_bind_ip) and
# refuse any mapped literal.
MAPPED_BINDS = ("::ffff:0.0.0.0", "::ffff:127.0.0.1", "::ffff:224.0.0.1")


# test_cluster_toml_refusals holds the IPv4-mapped worker- and client-bind
# cases; MAPPED_BINDS serves the CLI-argv case below.
@pytest.mark.parametrize("host", MAPPED_BINDS)
def test_worker_loop_direct_cli_rejects_ipv4_mapped_binds(monkeypatch, host: str):
    monkeypatch.setattr(
        sys, "argv", ["dgxm-worker", "--address", f"tcp://[{host}]:26600"])
    with pytest.raises(SystemExit):
        worker_loop.main()


@pytest.mark.parametrize("address", ["tcp://10.0.0.2:26600", "tcp://[2001:db8::2]:26600"])
def test_plain_fabric_binds_still_load(tmp_path: Path, address: str):
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
        f'[[hosts]]\nname = "worker"\naddress = "{address}"\n')
    assert [host.address for host in load_cluster_config(path).hosts] == [address]


def test_heterogeneous_gpu_counts_are_rejected_during_load(tmp_path: Path):
    path = _write_config(
        tmp_path,
        '\ngpus = 1\n\n[[hosts]]\nname = "worker2"\n'
        'address = "tcp://10.0.0.3:26600"\ngpus = 2\n',
    )
    with pytest.raises(ClusterConfigError, match="heterogeneous"):
        load_cluster_config(path)


def test_cluster_world_size_is_capped_before_mesh_spawn(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    hosts = "".join(
        f'[[hosts]]\nname = "w{i}"\naddress = "tcp://10.0.0.{i + 2}:26600"\n'
        "gpus = 64\n\n"
        for i in range(5)
    )
    path.write_text(f'[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n{hosts}')
    with pytest.raises(ClusterConfigError, match="world size"):
        load_cluster_config(path)


@pytest.mark.parametrize("key", ["LD_PRELOAD", "NCCL_NET_PLUGIN", "UCX_LOG_FILE"])
def test_fabric_env_rejects_process_injection_and_arbitrary_paths(tmp_path: Path, key: str):
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n'
        'fabric_profile = "custom"\n\n'
        '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n\n'
        f'[fabric.custom]\n{key} = "/tmp/x"\n'
    )
    with pytest.raises(ClusterConfigError, match=r"allowed|permitted"):
        load_cluster_config(path)
    with pytest.raises(ClusterConfigError, match=r"allowed|permitted"):
        validate_fabric_env({key: "/arbitrary/path"}, "fabric_env")


def test_fabric_dangerous_key_matching_uses_complete_tokens():
    assert validate_fabric_env(
        {"UCX_PROFILE_MODE": "accum"}, "fabric_env"
    ) == {"UCX_PROFILE_MODE": "accum"}


@pytest.mark.parametrize("value", ["LL", "Simple", "LL128,Simple"])
def test_fabric_env_refuses_every_nccl_proto_override(value: str):
    with pytest.raises(ClusterConfigError, match=r"NCCL_PROTO.*unset"):
        validate_fabric_env({"NCCL_PROTO": value}, "fabric_env")


@pytest.mark.parametrize("override", [
    {"world": 257}, {"rank": 2}, {"gpus_per_host": 65}, {"local_gpu_index": 2},
])
def test_actor_setup_rejects_oversized_or_invalid_gpu_shape(override: dict):
    env = {
        "setup_generation": 1,
        "world": 2,
        "rank": 0,
        "gpus_per_host": 2,
        "local_gpu_index": 0,
    }
    env.update(override)
    worker = SimpleNamespace(_setup_cleanup_failed=False, _setup_key=None)
    with pytest.raises(ValueError, match="worker setup"):
        setup_impl(worker, env)


@pytest.mark.parametrize("name", ["-oProxyCommand=touch_/tmp/pwn", "worker\nProxyCommand=x"])
def test_ssh_destination_option_and_control_injection_is_rejected(tmp_path: Path, name: str):
    path = tmp_path / "cluster.toml"
    escaped = json.dumps(name)
    path.write_text(
        f'[[hosts]]\nname = {escaped}\naddress = "tcp://10.0.0.2:26600"\n')
    with pytest.raises(ClusterConfigError, match=r"start with '-'|control"):
        load_cluster_config(path)


@pytest.mark.parametrize(
    ("name", "ssh_user"),
    [
        ("alice@redirect.example", ""),
        ("worker:redirect", ""),
        ("worker", "alice@redirect.example"),
        ("worker", "alice:redirect"),
    ],
)
def test_ssh_destination_delimiters_are_rejected(
    tmp_path: Path, name: str, ssh_user: str,
):
    path = tmp_path / "cluster.toml"
    path.write_text(
        f'[[hosts]]\nname = {json.dumps(name)}\n'
        'address = "tcp://10.0.0.2:26600"\n'
        f'ssh_user = {json.dumps(ssh_user)}\n'
    )
    with pytest.raises(
        ClusterConfigError, match=r"destination|SSH login|must not contain"
    ):
        load_cluster_config(path)


@pytest.mark.parametrize("sink", ["ssh", "rsync"])
@pytest.mark.parametrize(
    ("name", "ssh_user"),
    [
        ("alice@redirect.example", ""),
        ("worker:redirect", ""),
        ("worker", "alice@redirect.example"),
        ("worker", "alice:redirect"),
    ],
)
def test_programmatic_host_destination_is_revalidated_at_every_sink(
    sink: str, name: str, ssh_user: str,
):
    host = HostConfig(name, "tcp://10.0.0.2:26600", ssh_user=ssh_user)
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")

    with pytest.raises(
        ClusterConfigError, match=r"destination|SSH login|must not contain"
    ):
        if sink == "ssh":
            lifecycle._ssh_base(config, host)
        else:
            lifecycle._rsync_target(host)


def test_ssh_and_rsync_use_option_boundaries_and_ipv6_safe_target(monkeypatch, tmp_path):
    host = HostConfig("fd00::2", "tcp://[fd00::2]:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    commands = []
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(lifecycle, "_pkg_src_dir", lambda: tmp_path / "src")
    monkeypatch.setattr(
        lifecycle, "run_on_host",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(
        lifecycle.subprocess, "run",
        lambda command, **_kwargs: commands.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    assert lifecycle._ssh_base(config, host)[-2:] == ["--", "fd00::2"]
    assert lifecycle.sync_package(config, host)
    command = commands[0]
    assert "--" in command
    assert "--delete-excluded" in command
    assert "--exclude=__pycache__/" in command
    assert "--exclude=*.pyc" in command
    assert "--include=/dgx_monarch/***" in command
    assert "--exclude=*" in command
    assert command[-2].endswith("/src/")
    assert command[-1] == "[fd00::2]:~/.local/share/dgx-monarch/src/"


def test_ipv6_loop_process_regex_matches_literal_argv_only():
    host = HostConfig("worker", "tcp://[fd00::2]:26600")
    pattern = lifecycle._loop_regex(host)
    argv = "/venv/python -m dgx_monarch.cli.worker_loop --address tcp://[fd00::2]:26600"
    assert re.search(pattern, argv)
    assert not re.search(pattern, argv.replace("fd00::2", "fd00::3"))


def test_start_and_install_refuse_unacknowledged_transport_before_sync(monkeypatch):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,))
    monkeypatch.setattr(
        lifecycle, "sync_package",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must fail before sync")),
    )
    assert not lifecycle.up(config)
    assert not lifecycle.install_systemd(config)


def test_systemd_unit_quotes_python_without_sed_interpolation(monkeypatch):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(
        hosts=(host,), python_bin='~/venv dir/py"thon%x',
        transport_security="trusted_fabric",
    )
    scripts = []
    monkeypatch.setattr(lifecycle, "sync_package", lambda *_args: True)
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    results = iter((
        subprocess.CompletedProcess([], 0, "UNIT_ABSENT\n", ""),
        subprocess.CompletedProcess([], 0, "STATE=active\nLINGER=yes\n", ""),
    ))
    monkeypatch.setattr(
        lifecycle, "run_on_host",
        lambda _config, _host, script, timeout=60: scripts.append(script) or next(results),
    )
    assert lifecycle.install_systemd(config)
    assert "sed -i" not in scripts[0]
    assert 'ExecStart="%h/venv dir/py\\"thon%%x"' in scripts[1]
    assert systemd_generation_invalidator() in scripts[1]
    assert "/usr/bin/flock" not in systemd_generation_invalidator()
    assert scripts[1].index("ExecStartPre=") < scripts[1].index("ExecStart=")
    assert "UMask=0077" in scripts[1]
    assert scripts[1].index("worker-generation") < scripts[1].index("enable --now")


def test_nohup_start_sets_private_umask_before_log_redirect(monkeypatch):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    scripts = []
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda _config, _host, script, timeout=60: scripts.append(script)
        or subprocess.CompletedProcess([], 0, "STARTED_NOHUP\n", ""),
    )

    assert lifecycle.up(config, sync=False)
    assert scripts[0].index("umask 077") < scripts[0].index("worker-loop.log")
    assert 'chmod 700 "$(dirname "$HOME/.local/state/dgx-monarch/worker-loop.log")"' in scripts[0]
    assert 'chmod 600 "$HOME/.local/state/dgx-monarch/worker-loop.log"' in scripts[0]
    assert 'export PYTHONPATH="$HOME/.local/share/dgx-monarch/src"' in scripts[0]
    assert "export PYTHONDONTWRITEBYTECODE=1" in scripts[0]
    assert "PYTHONPYCACHEPREFIX=/proc/self/fd/2147483647" in scripts[0]
    assert "${PYTHONPATH:-}" not in scripts[0]
    assert "dgxm_owned_listener_ready" in scripts[0]
    assert 'readlink "$DGXM_LOOP_FD"' in scripts[0]
    assert scripts[0].index("worker-generation") < scripts[0].index("nohup")


def _run_owned_listener_probe(tmp_path: Path, loop_inode: int) -> subprocess.CompletedProcess:
    host = HostConfig("worker", "tcp://127.0.0.1:26600")
    proc_root = tmp_path / "proc"
    (proc_root / "net").mkdir(parents=True)
    (proc_root / "4242" / "fd").mkdir(parents=True)
    (proc_root / "9999" / "fd").mkdir(parents=True)
    proc_stat = "S " + "0 " * 18 + "117707\n"
    (proc_root / "4242" / "stat").write_text(
        f"4242 (python) {proc_stat}", encoding="ascii"
    )
    (proc_root / "9999" / "stat").write_text(
        f"9999 (python) {proc_stat}", encoding="ascii"
    )
    listener_inode = 117707
    endpoint = f"0100007F:{26600:04X}"
    (proc_root / "net" / "tcp").write_text(
        "sl local_address rem_address st tx_queue tr tm->when retrnsmt uid timeout inode\n"
        f"0: {endpoint} 00000000:0000 0A 00000000:00000000 "
        f"00:00000000 00000000 1000 0 {listener_inode} 1\n",
        encoding="ascii",
    )
    (proc_root / "4242" / "fd" / "3").symlink_to(f"socket:[{loop_inode}]")
    (proc_root / "9999" / "fd" / "3").symlink_to(
        f"socket:[{listener_inode}]"
    )
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    pgrep = tool_dir / "pgrep"
    pgrep.write_text("#!/bin/sh\nprintf '4242\\n'\n", encoding="ascii")
    pgrep.chmod(0o700)
    script = lifecycle._listener_probe_shell(
        host,
        lifecycle._loop_regex(host),
        proc_root=str(proc_root),
    )
    return subprocess.run(
        ["/bin/bash", "-c", f"{script}\ndgxm_owned_listener_ready"],
        capture_output=True,
        text=True,
        env={"PATH": f"{tool_dir}:/usr/bin:/bin"},
        check=False,
    )


def test_matching_loop_does_not_claim_a_foreign_listener(tmp_path):
    result = _run_owned_listener_probe(tmp_path, loop_inode=229922)

    assert result.returncode != 0


def test_listener_proof_accepts_only_an_inode_owned_by_the_matching_loop(tmp_path):
    result = _run_owned_listener_probe(tmp_path, loop_inode=117707)

    assert result.returncode == 0, result.stderr


def test_local_nohup_and_sweep_preserve_literal_dollar_in_source_path(monkeypatch, tmp_path):
    source = tmp_path / "$DGXM_UNSET" / "src"
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    scripts: list[str] = []
    monkeypatch.delenv("DGXM_UNSET", raising=False)
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: True)
    monkeypatch.setattr(lifecycle, "_pkg_src_dir", lambda: source)

    def run_host(_config, _host, script, timeout=60):
        scripts.append(script)
        output = "STARTED_NOHUP\n" + SWEEP_OK
        return subprocess.CompletedProcess([], 0, output, "")

    monkeypatch.setattr(lifecycle, "run_on_host", run_host)
    assert lifecycle.up(config, sync=False)
    assert len(scripts) == 1
    for script in scripts:
        syntax = subprocess.run(
            ["/bin/bash", "-n"], input=script, capture_output=True, text=True, check=False)
        assert syntax.returncode == 0, syntax.stderr
        export = next(
            line.strip() for line in script.splitlines()
            if line.strip().startswith("export PYTHONPATH=")
        )
        assert export == f"export PYTHONPATH='{source}'"
        result = subprocess.run(
            ["/bin/bash", "-uc", f"{export}\nprintf '%s' \"$PYTHONPATH\""],
            capture_output=True, text=True, env={}, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == str(source)


def test_explicit_update_host_does_not_hide_another_local_driver(monkeypatch):
    import urllib.request

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def urlopen(url, timeout=1):
        if "127.0.0.1:8199/object_info/DGXMonarchInit" in url:
            return Response(json.dumps({"DGXMonarchInit": {
                "name": "DGXMonarchInit", "category": "DGX Monarch",
                "output": ["DGXM_MESH"],
            }}).encode())
        if "127.0.0.1:8199/dgxm/telemetry" in url:
            return Response(b'{"workers": []}')
        raise OSError("stopped")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(cli.doctor_mod, "_find_comfy_ports", lambda: {8199: 1})
    assert cli._driver_probe("127.0.0.1:8188")[0] == "127.0.0.1:8199"


def test_uninstall_without_config_does_not_claim_removal(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_config", lambda _args: None)
    assert cli.cmd_uninstall(SimpleNamespace()) == 0
    assert "no worker services or units were changed" in capsys.readouterr().out


def test_uninstall_failure_reports_incomplete(monkeypatch, capsys):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    monkeypatch.setattr(cli, "_load_config", lambda _args: ClusterConfig(hosts=(host,)))
    events: list[str] = []
    monkeypatch.setattr(cli.lifecycle, "units_are_removable", lambda _config: True)
    monkeypatch.setattr(
        cli.lifecycle, "down", lambda _config: events.append("down") or False
    )
    monkeypatch.setattr(
        cli.lifecycle,
        "uninstall_systemd",
        lambda _config: events.append("uninstall") or True,
    )
    assert cli.cmd_uninstall(SimpleNamespace()) == 1
    assert events == ["down"]
    assert "uninstall incomplete" in capsys.readouterr().err


def test_uninstall_refuses_a_foreign_unit_before_stopping_or_clearing_anything(
    monkeypatch, capsys
):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    monkeypatch.setattr(cli, "_load_config", lambda _args: ClusterConfig(hosts=(host,)))
    events: list[str] = []
    monkeypatch.setattr(
        cli.lifecycle, "down", lambda _config: events.append("down") or True)
    monkeypatch.setattr(
        cli.lifecycle, "uninstall_systemd", lambda _config: events.append("uninstall") or True)
    monkeypatch.setattr(
        cli.lifecycle,
        "run_on_host",
        lambda _config, _host, _script, timeout=30: events.append("probe")
        or subprocess.CompletedProcess([], 1, "REFUSED_FOREIGN_UNIT\n", ""),
    )

    assert cli.cmd_uninstall(SimpleNamespace()) == 1

    # `down` clears the generation record and stops the worker. Neither may run
    # before the unit is proved ours, or the refusal is not a no-change refusal.
    assert events == ["probe"]
    out = capsys.readouterr()
    assert "REFUSED_FOREIGN_UNIT (the existing unit" in out.out
    assert "uninstall incomplete" in out.err


def test_uninstall_refuses_config_lock_contention_before_lifecycle_mutation(
    monkeypatch, tmp_path, capsys
):
    from dgx_monarch.cli.setup_config_lock import SetupConfigLock

    source = tmp_path / "cluster.toml"
    source.write_text(
        '[cluster]\ntransport_security = "trusted_fabric"\n\n'
        '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n'
    )
    config = load_cluster_config(source)
    monkeypatch.setattr(cli, "_load_config", lambda _args: config)
    events: list[str] = []
    monkeypatch.setattr(cli.lifecycle, "units_are_removable", lambda _config: True)
    monkeypatch.setattr(cli.lifecycle, "down", lambda _config: events.append("down") or True)
    monkeypatch.setattr(
        cli.lifecycle, "uninstall_systemd", lambda _config: events.append("uninstall") or True
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    with SetupConfigLock(source):
        assert cli.cmd_uninstall(SimpleNamespace()) == 1

    assert events == []
    assert source.exists()
    assert "serialization is unavailable" in capsys.readouterr().err


def test_uninstall_removes_only_the_exact_lock_bound_config(monkeypatch, tmp_path):
    source = tmp_path / "cluster.toml"
    source.write_text(
        '[cluster]\ntransport_security = "trusted_fabric"\n\n'
        '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n'
    )
    config = load_cluster_config(source)
    monkeypatch.setattr(cli, "_load_config", lambda _args: config)
    monkeypatch.setattr(cli.lifecycle, "units_are_removable", lambda _config: True)
    monkeypatch.setattr(cli.lifecycle, "down", lambda _config: True)
    monkeypatch.setattr(cli.lifecycle, "uninstall_systemd", lambda _config: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert cli.cmd_uninstall(SimpleNamespace()) == 0
    assert not source.exists()


def test_uninstall_systemd_requires_removal_readback(monkeypatch):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    scripts = []
    monkeypatch.setattr(
        lifecycle, "run_on_host",
        lambda _config, _host, script, timeout=30: scripts.append(script)
        or subprocess.CompletedProcess([], 1, "FAILED_STILL_PRESENT\n", ""),
    )
    assert not lifecycle.uninstall_systemd(ClusterConfig(hosts=(host,)))
    assert '[ -e "$UNIT" ]' in scripts[0]
    assert "systemctl --user is-active --quiet" in scripts[0]
    assert scripts[0].index("worker-generation") < scripts[0].index("disable --now")


def test_install_systemd_refuses_foreign_unit_before_source_sync(monkeypatch, capsys):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    sync_calls: list[object] = []
    monkeypatch.setattr(lifecycle, "sync_package", lambda *_args: sync_calls.append(object()) or True)
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "REFUSED_FOREIGN_UNIT\n", ""),
    )

    assert not lifecycle.install_systemd(config)
    assert not sync_calls
    assert "REFUSED" in capsys.readouterr().out


def test_install_systemd_allows_exact_canonical_reinstall(monkeypatch):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    scripts: list[str] = []
    responses = iter((
        subprocess.CompletedProcess([], 0, "UNIT_OWNED\n", ""),
        subprocess.CompletedProcess([], 0, "STATE=active\nLINGER=yes\n", ""),
    ))
    monkeypatch.setattr(lifecycle, "sync_package", lambda *_args: True)
    monkeypatch.setattr(
        lifecycle, "run_on_host", lambda *_args, **_kwargs: scripts.append(_args[2]) or next(responses)
    )

    assert lifecycle.install_systemd(config)
    assert lifecycle._UNIT_OWNERSHIP_MARKER in scripts[1]
    assert "cmp -s" in scripts[1]
    assert 'cat > "$UNIT_TMP"' in scripts[1]
    assert 'mv -T -- "$UNIT_TMP" "$UNIT"' in scripts[1]
    assert f"cat > ~/.config/systemd/user/{lifecycle._UNIT_NAME}" not in scripts[1]


def test_uninstall_systemd_refuses_foreign_unit_before_generation_mutation(monkeypatch, capsys):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    scripts: list[str] = []
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda _config, _host, script, timeout=30: scripts.append(script)
        or subprocess.CompletedProcess([], 1, "REFUSED_FOREIGN_UNIT\n", ""),
    )

    assert not lifecycle.uninstall_systemd(ClusterConfig(hosts=(host,)))
    assert "cmp -s" in scripts[0]
    assert "REFUSED_FOREIGN_UNIT" in scripts[0]
    # Inside the removal script too, the refusal exits before the removal.
    assert scripts[0].index("REFUSED_FOREIGN_UNIT") < scripts[0].index('rm -f -- "$UNIT"')
    assert "REFUSED_FOREIGN_UNIT (the existing unit" in capsys.readouterr().out


def _install_unit_the_way_the_release_wrote_it(home: Path, body: str) -> None:
    """Write the unit through the pre-marker installer's own heredoc shape.

    That writer wrapped the rendered body as `cat > ... <<'UNIT'` / body / `UNIT`,
    so the file it left carries one newline the renderer never emitted. A
    fixture written with `write_text` lacks that byte, and the test would miss
    the only migration path an existing installation has.
    """
    unit_dir = home / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    written = subprocess.run(
        ["bash", "-s"],
        input=f'cat > "$HOME/.config/systemd/user/dgxm-worker.service" <<\'UNIT\'\n{body}\nUNIT\n',
        capture_output=True,
        text=True,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
    )
    assert written.returncode == 0, written.stderr


def _probe_verdict(home: Path, legacy: str, runtime_dir: str | None = None,
                   **kwargs: bool) -> subprocess.CompletedProcess[str]:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    if runtime_dir is not None:
        env["XDG_RUNTIME_DIR"] = runtime_dir
    return subprocess.run(
        ["bash", "-s"],
        input="set -eu\n" + lifecycle._unit_ownership_probe(legacy, **kwargs),
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize("shape", ["marked", "rendered", "as-installed"])
def test_systemd_ownership_probe_accepts_both_pre_marker_shapes(tmp_path, shape):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    current = lifecycle._systemd_worker_unit(config, host)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    # Install and uninstall share this ownership probe. The marker survives a
    # legitimate config change; a pre-marker unit does not. "as-installed" is
    # the shape a released `dgxm install-service` left on disk.
    if shape == "as-installed":
        _install_unit_the_way_the_release_wrote_it(tmp_path, legacy)
        assert unit.read_text(encoding="utf-8") == legacy + "\n"
    else:
        unit.write_text(
            current.replace("RestartSec=3", "RestartSec=4") if shape == "marked" else legacy,
            encoding="utf-8",
        )

    result = _probe_verdict(tmp_path, legacy)

    assert result.returncode == 0
    assert result.stdout.strip() == "UNIT_OWNED"


def test_systemd_ownership_probe_accepts_exactly_two_pre_marker_shapes(tmp_path):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    # One more trailing newline than the installer wrote is not an accepted
    # shape: the acceptance is two byte-exact files, not a trailing-blank rule.
    unit.write_text(legacy + "\n\n", encoding="utf-8")

    result = _probe_verdict(tmp_path, legacy)

    assert result.returncode != 0
    assert result.stdout.strip() == "REFUSED_FOREIGN_UNIT"


def test_pre_marker_unit_is_migrated_to_the_marked_form_by_one_install(tmp_path):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    current = lifecycle._systemd_worker_unit(config, host)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    _install_unit_the_way_the_release_wrote_it(tmp_path, legacy)
    # The probe and the publish helper run as one shell, so the publish helper
    # may not read a variable the probe stopped defining.
    script = (
        "set -eu\n"
        + lifecycle._unit_ownership_probe(legacy)
        + lifecycle._unit_publish_script(current)
    )

    result = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "UNIT_OWNED"
    assert unit.read_text(encoding="utf-8") == current
    assert [path.name for path in unit.parent.iterdir()] == [unit.name]


def _setup_written_unit(tmp_path: Path) -> str:
    from dgx_monarch.cli import setup_services_platform
    from dgx_monarch.cli.setup_services import SetupServiceRequest

    source = tmp_path / "source" / "dgx_monarch"
    source.mkdir(parents=True)
    config = ClusterConfig(
        hosts=(HostConfig("node-1", "tcp://10.0.0.2:26600"),),
        python_bin=sys.executable,
        transport_security="trusted_fabric",
    )
    request = SetupServiceRequest(config, source, "d" * 64, True, False)
    ops = setup_services_platform.SystemSetupServiceOps(
        request,
        local_detector=lambda _host: False,
        token_factory=lambda: "s-" + "a" * 16,
    )
    return ops._unit(config.hosts[0], 1)


def test_guided_setup_unit_is_removable_but_never_republishable(tmp_path):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    unit.write_text(_setup_written_unit(tmp_path), encoding="utf-8")

    removal = _probe_verdict(tmp_path, legacy, allow_setup_token=True)
    install = _probe_verdict(tmp_path, legacy)

    # Uninstall is the only teardown guided setup has, so it must take this
    # unit. Install may not: the canonical body carries no setup token, and
    # remote source sync keys its managed short-circuit on that token.
    assert (removal.returncode, removal.stdout.strip()) == (0, "UNIT_OWNED")
    assert (install.returncode, install.stdout.strip()) == (1, "REFUSED_SETUP_UNIT")
    # The asymmetry is in the rendered script, not only in this run's verdict.
    assert "REFUSED_SETUP_UNIT" in lifecycle._unit_ownership_probe(legacy)
    assert "REFUSED_SETUP_UNIT" not in lifecycle._unit_ownership_probe(
        legacy, allow_setup_token=True)


@pytest.mark.parametrize("append_marker", [False, True])
def test_systemd_ownership_probe_refuses_one_byte_edited_legacy_unit(
    tmp_path, append_marker,
):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    edited = legacy.replace("RestartSec=3", "RestartSec=4")
    if append_marker:
        edited += lifecycle._UNIT_OWNERSHIP_MARKER + "\n"
    unit.write_text(edited, encoding="utf-8")

    result = subprocess.run(
        ["bash", "-s"],
        input="set -eu\n" + lifecycle._unit_ownership_probe(legacy),
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode != 0
    assert result.stdout.strip() == "REFUSED_FOREIGN_UNIT"


@pytest.mark.parametrize("unit_shape", ["owned", "absent"])
@pytest.mark.parametrize("dropin_dir", [
    # The unit's own directory and the dash-truncated one systemd also reads for
    # a dashed unit name (systemd.unit(5)), in each of the two search paths under
    # the operator's home. A probe that reads only the first of the four proves
    # nothing about the command systemd runs.
    ".config/systemd/user/dgxm-worker.service.d",
    ".config/systemd/user/dgxm-.service.d",
    ".local/share/systemd/user/dgxm-worker.service.d",
    ".local/share/systemd/user/dgxm-.service.d",
])
def test_systemd_ownership_probe_refuses_a_drop_in_it_never_wrote(
        tmp_path, unit_shape, dropin_dir):
    """A drop-in can replace ExecStart while the unit stays byte-canonical."""
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    if unit_shape == "owned":
        unit.write_text(lifecycle._systemd_worker_unit(config, host), encoding="utf-8")
    dropins = tmp_path / dropin_dir
    dropins.mkdir(parents=True)
    (dropins / "override.conf").write_text(
        "[Service]\nExecStart=\nExecStart=/usr/bin/other\n", encoding="utf-8")

    result = _probe_verdict(tmp_path, legacy)

    assert result.returncode != 0
    # The name is reported home-relative: four directories can hold the same
    # basename, and the operator has to know which one to clear.
    assert result.stdout.strip() == f"REFUSED_FOREIGN_DROPIN {dropin_dir}/override.conf"
    refusal = lifecycle._unit_refusal(result.stdout)
    assert f"{dropin_dir}/override.conf" in refusal and "ExecStart" in refusal


def test_systemd_ownership_probe_refuses_a_runtime_drop_in(tmp_path):
    """`systemctl --user edit --runtime` writes under XDG_RUNTIME_DIR.

    That directory is the user's, and a drop-in in any search path overrides
    the unit file wherever it lives (systemd.unit(5)), so an override left
    there keeps replacing ExecStart after an install publishes the canonical
    unit. The name comes back in full, since it is not under the operator's home.
    """
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / "home/.config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    unit.write_text(lifecycle._systemd_worker_unit(config, host), encoding="utf-8")
    runtime = tmp_path / "run/user/1000"
    dropins = runtime / "systemd/user/dgxm-worker.service.d"
    dropins.mkdir(parents=True)
    (dropins / "override.conf").write_text(
        "[Service]\nExecStart=\nExecStart=/usr/bin/other\n", encoding="utf-8")

    result = _probe_verdict(tmp_path / "home", legacy, runtime_dir=str(runtime))

    assert result.returncode != 0
    assert result.stdout.strip() == (
        f"REFUSED_FOREIGN_DROPIN {dropins / 'override.conf'}")


def test_systemd_ownership_probe_runs_with_no_runtime_dir_set(tmp_path):
    """An unset XDG_RUNTIME_DIR is a missing directory, not a broken probe."""
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    unit.write_text(lifecycle._systemd_worker_unit(config, host), encoding="utf-8")

    result = _probe_verdict(tmp_path, legacy)

    assert (result.returncode, result.stdout.strip()) == (0, "UNIT_OWNED")


def test_systemd_ownership_probe_ignores_the_box_wide_service_drop_in(tmp_path):
    """`service.d` applies to every user service, not only this one.

    systemd reads it for this unit too, but a file there says nothing about
    who edited this service, and refusing would stop an install over an
    unrelated setting the operator meant to keep.
    """
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    unit.write_text(lifecycle._systemd_worker_unit(config, host), encoding="utf-8")
    box_wide = tmp_path / ".config/systemd/user/service.d"
    box_wide.mkdir()
    (box_wide / "10-limits.conf").write_text("[Service]\nLimitNOFILE=8192\n",
                                             encoding="utf-8")

    result = _probe_verdict(tmp_path, legacy)

    assert (result.returncode, result.stdout.strip()) == (0, "UNIT_OWNED")


def test_systemd_ownership_probe_accepts_an_empty_drop_in_directory(tmp_path):
    """An empty drop-in directory holds no override, so the scan names nothing."""
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    unit.write_text(lifecycle._systemd_worker_unit(config, host), encoding="utf-8")
    (tmp_path / ".config/systemd/user/dgxm-worker.service.d").mkdir()

    result = _probe_verdict(tmp_path, legacy)

    assert (result.returncode, result.stdout.strip()) == (0, "UNIT_OWNED")


def test_unit_removal_names_the_drop_in_it_left_behind(tmp_path):
    """Uninstall removes the unit only, so the override has to be named.

    The removal path does not refuse a drop-in: refusing there would leave both
    the unit and the override in place. It reports instead, and the next
    install refuses.
    """
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    dropins = tmp_path / ".config/systemd/user/dgxm-worker.service.d"
    dropins.mkdir(parents=True)
    (dropins / "override.conf").write_text("[Service]\nExecStart=\n", encoding="utf-8")
    (dropins / "90-extra.conf").write_text("[Service]\n", encoding="utf-8")
    truncated = tmp_path / ".config/systemd/user/dgxm-.service.d"
    truncated.mkdir(parents=True)
    (truncated / "50-prefix.conf").write_text("[Service]\n", encoding="utf-8")
    script = lifecycle_systemd.unit_remove_script(legacy)
    report = 'if [ -n "$DGXM_DROPINS" ]; then\n  echo "LEFT_DROPIN$DGXM_DROPINS"\nfi\n'

    # The scan and the report the removal ends with, run on their own: the rest
    # of that script drives systemctl, which no test may run.
    scanned = subprocess.run(
        ["bash", "-s"],
        input="set -eu\n" + lifecycle_systemd.unit_dropin_scan() + report,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert scanned.returncode == 0, scanned.stderr
    left = lifecycle_systemd.unit_removal_leftovers(scanned.stdout)
    assert "dgxm-worker.service.d/override.conf" in left
    assert "dgxm-worker.service.d/90-extra.conf" in left
    # The dash-truncated directory reattaches to the next unit just as the
    # unit's own does, so removal has to name it too.
    assert "dgxm-.service.d/50-prefix.conf" in left
    assert "refuses" in left
    # Both halves are in the shipped removal script, not only in this rehearsal.
    assert lifecycle_systemd.unit_dropin_scan() in script
    assert report in script
    assert "REFUSED_FOREIGN_DROPIN" not in script
    assert lifecycle_systemd.unit_removal_leftovers("DONE\n") == ""


def test_the_drop_in_refusal_is_written_down_where_the_operator_looks():
    """A refusal an operator meets is a runbook row, not only a string.

    docs/TROUBLESHOOTING.md #90 names every refusal this probe can print. The
    drop-in one stops an install and names files, and the operator has to know
    that removing them is the whole fix and that uninstall reports rather than
    refuses.
    """
    text = (Path(__file__).resolve().parents[1]
            / "docs" / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
    entry = text.split("\n## 90. ", 1)[1].split("\n## ", 1)[0]

    assert "REFUSED_FOREIGN_DROPIN" in entry
    assert "dgxm-worker.service.d" in entry
    assert "ExecStart=" in entry
    assert "left behind:" in entry


@pytest.mark.parametrize("target_kind", ["marked", "legacy", "dangling"])
def test_systemd_ownership_probe_refuses_symlink_without_touching_target(
    tmp_path: Path, target_kind: str,
) -> None:
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    current = lifecycle._systemd_worker_unit(config, host)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    target = tmp_path / "outside.service"
    if target_kind != "dangling":
        target.write_text(current if target_kind == "marked" else legacy, encoding="utf-8")
        before = target.read_bytes()
    unit.symlink_to(target)

    result = subprocess.run(
        ["bash", "-s"],
        input="set -eu\n" + lifecycle._unit_ownership_probe(legacy),
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode != 0
    assert result.stdout.strip() == "REFUSED_FOREIGN_UNIT"
    assert unit.is_symlink()
    if target_kind != "dangling":
        assert target.read_bytes() == before
    else:
        assert not target.exists()


def test_systemd_unit_publish_replaces_racing_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    unit = tmp_path / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    outside = tmp_path / "outside.service"
    outside.write_text("foreign bytes\n", encoding="utf-8")
    current = lifecycle._systemd_worker_unit(config, host)
    legacy = lifecycle._systemd_worker_unit(config, host, marked=False)
    script = (
        "set -eu\n"
        + lifecycle._unit_ownership_probe(legacy)
        + f'ln -s -- {shlex.quote(str(outside))} "$UNIT"\n'
        + lifecycle._unit_publish_script(current)
    )

    result = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert not unit.is_symlink()
    assert unit.read_text(encoding="utf-8") == current
    assert outside.read_text(encoding="utf-8") == "foreign bytes\n"


def test_whole_telemetry_payload_is_cached(monkeypatch):
    from dgx_monarch.nodes import routes

    calls = []
    monkeypatch.setattr(routes, "_TELEMETRY_CACHE", {"t": 0.0, "data": None, "inflight": False})
    monkeypatch.setattr(
        routes, "_telemetry_uncached",
        lambda: calls.append(1) or {"t": 1.0, "workers": []},
    )
    assert routes._telemetry() == {"t": 1.0, "workers": []}
    assert routes._telemetry() == {"t": 1.0, "workers": []}
    assert calls == [1]


def test_whole_telemetry_refresh_failure_keeps_last_good(monkeypatch):
    from dgx_monarch.nodes import routes

    good = {"workers": [{"host": "good"}]}
    monkeypatch.setattr(
        routes, "_TELEMETRY_CACHE", {"t": 0.0, "data": good, "inflight": False})
    monkeypatch.setattr(
        routes, "_telemetry_uncached",
        lambda: (_ for _ in ()).throw(RuntimeError("transient")),
    )
    stale = routes._telemetry()
    assert stale is not good
    assert stale["workers"] is good["workers"]
    assert stale["telemetry_error"] == "RuntimeError"
    assert stale["readiness"]["overall"] == "unknown"
    assert good == {"workers": [{"host": "good"}]}
    assert routes._TELEMETRY_CACHE["data"] is stale
    assert routes._telemetry() is stale


def test_whole_telemetry_first_refresh_failure_has_fail_closed_readiness(monkeypatch):
    from dgx_monarch.nodes import routes

    monkeypatch.setattr(
        routes, "_TELEMETRY_CACHE", {"t": 0.0, "data": None, "inflight": False})
    monkeypatch.setattr(
        routes, "_telemetry_uncached",
        lambda: (_ for _ in ()).throw(RuntimeError("private detail")),
    )
    fallback = routes._telemetry()
    assert fallback["telemetry_error"] == "RuntimeError"
    assert fallback["readiness"]["overall"] == "unknown"
    assert "private detail" not in str(fallback)


def test_whole_telemetry_refresh_is_single_flight(monkeypatch):
    from dgx_monarch.nodes import routes

    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        routes, "_TELEMETRY_CACHE",
        {"t": 0.0, "data": {"workers": ["stale"]}, "inflight": False},
    )

    def refresh():
        started.set()
        release.wait(timeout=3)
        return {"workers": ["fresh"]}

    monkeypatch.setattr(routes, "_telemetry_uncached", refresh)
    thread = threading.Thread(target=routes._telemetry)
    thread.start()
    assert started.wait(timeout=3)
    stale = routes._telemetry()
    assert stale["workers"] == ["stale"]
    assert stale["telemetry_error"] == "RefreshInFlight"
    assert stale["readiness"]["overall"] == "unknown"
    assert routes._TELEMETRY_CACHE["data"] == {"workers": ["stale"]}
    release.set()
    thread.join(timeout=3)
    assert routes._TELEMETRY_CACHE["data"] == {"workers": ["fresh"]}


def test_prometheus_labels_are_escaped_and_non_numeric_values_skipped():
    from dgx_monarch.nodes.routes import _metrics_text

    text = _metrics_text({"render": {}, "workers": [{"host": {
        "host": 'node"\\x\ninjected 1',
        "gpu": {"util_pct": "2\nevil_metric 9", "power_w": 12.5},
        "mem_gib": {}, "rails": {},
    }}]})
    assert 'host="node\\"\\\\x\\ninjected 1"' in text
    assert "evil_metric" not in text
    assert "\ninjected 1" not in text


def test_render_job_cardinality_is_bounded():
    with pytest.raises(ValueError, match="1024"):
        build_jobs("\n".join(f"job {i}" for i in range(1025)), None, ["negative"], 0)
    with pytest.raises(ValueError, match="1024"):
        DGXMonarchKSamplerPipeline._parse_seeds(" ".join(str(i) for i in range(1025)))


def test_setup_script_sends_quoted_values_over_stdin():
    script = (Path(__file__).parents[1] / "scripts" / "setup_env.sh").read_text()
    assert "printf 'TORCHAUDIO_DONOR=%q" in script
    assert '| ssh "${SSH_ARGS[@]}" -- "$SIB" bash -s' in script
    assert "$(declare -f" not in script


SWEEP_OK = "ACTOR_SWEEP reaped=1 killed=0 left=0 tracked=2 gib=49.1 pids=117707\n"
UPDATE_GENERATION = "u-0123456789ab-0123456789abcdef"


def _sweep_capture(monkeypatch, replies, host_name="worker", address="tcp://10.0.0.2:26600"):
    """Record every script `down` and `up` send.

    Each script gets the first reply. One that runs the actor reaper gets the
    second reply appended, with the second reply's exit code.
    """
    host = HostConfig(host_name, address)
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    scripts: list[str] = []

    def run_host(_config, _host, script, timeout=60):
        scripts.append(script)
        first = replies[0]
        if "dgx_monarch.cli.actor_reaper" in script and len(replies) > 1:
            sweep = replies[1]
            return subprocess.CompletedProcess(
                [], sweep.returncode, first.stdout + sweep.stdout, first.stderr + sweep.stderr
            )
        return first

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(lifecycle, "run_on_host", run_host)
    return config, scripts


def test_down_sweeps_actor_procs_after_the_verified_loop_stop(monkeypatch, capsys):
    config, scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 0, "DONE_SYSTEMD\n", ""),
        subprocess.CompletedProcess([], 0, SWEEP_OK, ""),
    ])
    assert lifecycle.down(config)
    # Stop and sweep are one payload under the same lifecycle lock.
    assert "FAILED_SYSTEMD_ACTIVE" in scripts[0]
    assert scripts[0].index("worker-generation") < scripts[0].index("systemctl --user stop")
    assert "os.fsync(fd)" in scripts[0]
    sweep = scripts[0]
    assert "-m dgx_monarch.cli.actor_reaper" in sweep
    assert "--all-loop-children" in sweep
    assert sweep.index("PYBIN=") < sweep.index('"$PYBIN" -m dgx_monarch.cli.actor_reaper')
    assert "export PYTHONPATH=" in sweep and "set -u" in sweep
    # The appended actor-sweep fragment never pattern-matches actor argv.
    actor_fragment = sweep[sweep.rindex("export PYTHONPATH=") :]
    assert "pkill" not in actor_fragment and "pgrep" not in actor_fragment
    assert "monarch._src.actor.bootstrap_main" not in actor_fragment
    assert "reaped=1" in capsys.readouterr().out


def test_down_fails_and_names_the_pid_when_an_actor_survives(monkeypatch, capsys):
    config, _scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 0, "DONE_SYSTEMD\n", ""),
        subprocess.CompletedProcess(
            [], 1, "ACTOR_SWEEP reaped=0 killed=1 left=1 tracked=1 gib=49.1 pids=117707\n", ""),
    ])
    assert not lifecycle.down(config)
    out = capsys.readouterr().out
    assert "117707" in out and "reboot" in out
    assert "worker: down" not in out


def test_zero_survivor_summary_does_not_override_a_nonzero_reaper_exit(monkeypatch, capsys):
    config, _scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 0, "DONE_SYSTEMD\n", ""),
        subprocess.CompletedProcess([], 1, SWEEP_OK, "reaper settlement failed\n"),
    ])

    assert lifecycle.down(config) is False
    out = capsys.readouterr().out
    assert "exit 1" in out and "reaper settlement failed" in out
    assert "worker: down" not in out


def test_forged_earlier_sweep_summary_cannot_authorize_down_or_restart(
    monkeypatch, capsys
):
    forged = "ACTOR_SWEEP reaped=1 killed=0 left=0 tracked=1 gib=49.1 pids=117707\n"
    authoritative = (
        "ACTOR_SWEEP reaped=0 killed=1 left=1 tracked=1 gib=49.1 pids=118322\n"
    )
    config, _scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 0, f"DONE_SYSTEMD\n{forged}", ""),
        subprocess.CompletedProcess([], 1, authoritative, ""),
    ])

    assert lifecycle.down(config) is False
    monkeypatch.setattr(
        lifecycle,
        "up",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restart must not start after an ambiguous sweep")
        ),
    )
    assert lifecycle.restart(config, sync=False) is False
    out = capsys.readouterr().out
    assert "unique and final" in out and "118322" in out and "exit 1" in out


def test_down_does_not_sweep_when_the_loop_stop_failed(monkeypatch):
    config, scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 1, "FAILED_SYSTEMD_ACTIVE\n", ""),
    ])
    assert not lifecycle.down(config)
    assert len(scripts) == 1


@pytest.mark.parametrize("stopped", [False, None])
def test_restart_never_starts_after_an_unconfirmed_stop(monkeypatch, stopped):
    events: list[str] = []
    config = ClusterConfig()
    monkeypatch.setattr(
        lifecycle, "down", lambda _config: events.append("down") or stopped
    )
    monkeypatch.setattr(
        lifecycle,
        "up",
        lambda _config, *, sync: events.append(f"up:{sync}") or True,
    )
    monkeypatch.setattr(
        lifecycle.time,
        "sleep",
        lambda _seconds: events.append("sleep"),
    )

    assert lifecycle.restart(config, sync=False) is stopped
    assert events == ["down"]


def test_restart_starts_only_after_a_confirmed_stop(monkeypatch):
    events: list[str] = []
    config = ClusterConfig()
    monkeypatch.setattr(
        lifecycle, "down", lambda _config: events.append("down") or True
    )
    monkeypatch.setattr(
        lifecycle,
        "up",
        lambda _config, *, sync: events.append(f"up:{sync}") or True,
    )
    monkeypatch.setattr(
        lifecycle.time,
        "sleep",
        lambda _seconds: events.append("sleep"),
    )

    assert lifecycle.restart(config, sync=False) is True
    assert events == ["down", "sleep", "up:False"]


def test_update_generation_is_recorded_only_after_both_readiness_paths(
    monkeypatch,
):
    config, scripts = _sweep_capture(
        monkeypatch,
        [subprocess.CompletedProcess([], 0, "STARTED_NOHUP\n" + SWEEP_OK, "")],
    )

    assert lifecycle.up(config, sync=False, generation=UPDATE_GENERATION) is True
    assert scripts[0].count("if dgxm_record_generation; then") == 2
    assert "dgxm_quarantine_failed_start systemd" in scripts[0]
    assert "dgxm_quarantine_failed_start nohup" in scripts[0]
    assert '[ "$DGXM_RECORD_STATUS" -eq 10 ] || exit "$DGXM_RECORD_STATUS"' in scripts[0]
    assert 'exit "$DGXM_SETTLE_STATUS"' in scripts[0]
    assert "/usr/bin/flock" not in scripts[0]
    assert scripts[0].index("dgxm_owned_listener_ready") < scripts[0].rindex(
        "dgxm_record_generation"
    )


def _record_host_work(monkeypatch) -> list[str]:
    """Record each lookup, command and sync a lifecycle call would send out.

    The resolver stand-in in ``conftest.py`` answers a placeholder host at
    once, so elapsed time proves nothing about a refusal. Every seam the start
    and stop paths reach is a recorder here instead. Each one answers as a
    reachable remote host would, so a call that refuses late leaves the seams
    it passed in the list.
    """
    touched: list[str] = []

    def seam(name: str, answer: object):
        def record(*_args, **_kwargs):
            touched.append(name)
            return answer

        return record

    done = subprocess.CompletedProcess([], 0, "", "")
    monkeypatch.setattr(lifecycle, "_is_local", seam("locality", False))
    monkeypatch.setattr(lifecycle_systemd, "is_local", seam("locality", False))
    monkeypatch.setattr(lifecycle, "run_on_host", seam("host command", done))
    monkeypatch.setattr(lifecycle.subprocess, "run", seam("subprocess", done))
    monkeypatch.setattr(lifecycle.socket, "getaddrinfo", seam("lookup", []))
    monkeypatch.setattr(lifecycle.socket, "create_connection", seam("socket", None))
    monkeypatch.setattr(lifecycle.time, "sleep", seam("sleep", None))
    return touched


@pytest.mark.parametrize("generation", ["", "u-short", "u-0123456789AB-0123456789abcdef"])
def test_invalid_update_generation_is_rejected_before_host_mutation(monkeypatch, generation: str):
    config = ClusterConfig(
        hosts=(HostConfig("worker", "tcp://10.0.0.2:26600"),),
        transport_security="trusted_fabric",
    )
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ValueError, match="generation"):
        lifecycle.up(config, sync=False, generation=generation)
    assert touched == []


@pytest.mark.parametrize("generation", ["", "u-short", "u-0123456789AB-0123456789abcdef"])
def test_invalid_update_generation_is_refused_before_the_default_package_sync(
    monkeypatch, generation: str,
):
    """`up` syncs unless told not to, and the sync writes the managed tree.

    So the refusal comes before the locality lookup and, with the default sync,
    before the probe, the mkdir and the rsync.
    """
    config = ClusterConfig(
        hosts=(
            HostConfig("worker", "tcp://10.0.0.2:26600"),
            HostConfig("worker-b", "tcp://10.0.0.3:26600"),
        ),
        transport_security="trusted_fabric",
    )
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ValueError, match=r"^worker generation is invalid$"):
        lifecycle.up(config, generation=generation)
    assert touched == []


def test_invalid_update_generation_is_refused_when_no_host_would_render_it(monkeypatch):
    """The refusal does not depend on a host reaching the start script."""
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ValueError, match=r"^worker generation is invalid$"):
        lifecycle.up(ClusterConfig(), generation="u-short")
    assert touched == []


def test_malformed_listener_address_is_refused_before_any_host_is_started(monkeypatch, capsys):
    """A later host's bad address must not let an earlier host restart first."""
    config = ClusterConfig(
        hosts=(
            HostConfig("worker", "tcp://10.0.0.2:26600"),
            HostConfig("worker-b", "tcp://worker-b:26600"),
        ),
        transport_security="trusted_fabric",
    )
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ValueError, match=r"^not a complete tcp endpoint: 'tcp://worker-b:26600'$"):
        lifecycle.up(config, sync=False)
    assert touched == []
    assert capsys.readouterr().out == ""


def test_unacknowledged_transport_still_wins_over_an_invalid_generation(monkeypatch, capsys):
    """Both refusals need no host; the transport refusal comes first."""
    config = ClusterConfig(hosts=(HostConfig("worker", "tcp://10.0.0.2:26600"),))
    touched = _record_host_work(monkeypatch)

    assert lifecycle.up(config, generation="u-short") is False
    assert capsys.readouterr().out == (
        "refusing to start the worker service: peer authentication is unavailable at the "
        "torchmonarch attach API dgx-monarch calls. Source-restrict the dedicated "
        'fabric, then set cluster.transport_security = "trusted_fabric" '
        "(SECURITY.md).\n"
    )
    assert touched == []


@pytest.mark.parametrize("generation", ["", "u-short", "u-0123456789AB-0123456789abcdef"])
def test_invalid_stop_generation_is_refused_before_any_host_work(monkeypatch, generation: str):
    config = ClusterConfig(
        hosts=(HostConfig("worker", "tcp://10.0.0.2:26600"),),
        transport_security="trusted_fabric",
    )
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ValueError, match=r"^worker generation is invalid$"):
        lifecycle.down_if_generation(config, generation)
    assert touched == []


def test_invalid_stop_generation_is_refused_when_no_host_would_check_it(monkeypatch):
    """The stop half agrees with the start half: the token is bad whatever the host list."""
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ValueError, match=r"^worker generation is invalid$"):
        lifecycle.down_if_generation(ClusterConfig(), "u-short")
    assert touched == []


_BAD_SSH_IDENTITIES = [
    (HostConfig("worker:redirect", "tcp://10.0.0.2:26600"),
     "hosts.name must not contain ':' unless it is a literal IPv6 address"),
    (HostConfig("user@evil", "tcp://10.0.0.2:26600"),
     "hosts.name must not contain '@', which rewrites the SSH login target"),
    (HostConfig("-oProxyCommand=x", "tcp://10.0.0.2:26600"),
     "hosts.name must be a non-empty string, must not start with '-', "
     "and must not contain whitespace/control characters"),
    (HostConfig("worker", "tcp://10.0.0.2:26600", ssh_user="alice:redirect"),
     "hosts.ssh_user must not contain '@' or ':' destination delimiters"),
]
_SSH_IDENTITY_IDS = ["name-colon", "name-at", "name-dash", "user-colon"]


@pytest.mark.parametrize(("host", "refusal"), _BAD_SSH_IDENTITIES, ids=_SSH_IDENTITY_IDS)
@pytest.mark.parametrize("entry", ["up", "up-no-sync", "down", "down_if_generation"])
def test_bad_ssh_identity_is_refused_before_any_host_is_resolved(
    monkeypatch, entry: str, host: HostConfig, refusal: str,
):
    """The SSH checks are string checks, so they need no locality lookup.

    The config loader runs the same two checks on every host whatever its
    locality, and the lifecycle entry points run them too, before the lookup.
    """
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    touched = _record_host_work(monkeypatch)
    calls = {
        "up": lambda: lifecycle.up(config),
        "up-no-sync": lambda: lifecycle.up(config, sync=False),
        "down": lambda: lifecycle.down(config),
        "down_if_generation": lambda: lifecycle.down_if_generation(config, "s-0123456789abcdef"),
    }

    with pytest.raises(ClusterConfigError, match=re.escape(refusal)):
        calls[entry]()
    assert touched == []


def test_bad_ssh_identity_on_a_later_host_is_refused_before_an_earlier_one_restarts(
    monkeypatch, capsys,
):
    """Host one must not be probed, synced or restarted before host two is refused."""
    config = ClusterConfig(
        hosts=(
            HostConfig("worker", "tcp://10.0.0.2:26600"),
            HostConfig("-oProxyCommand=x", "tcp://10.0.0.3:26600"),
        ),
        transport_security="trusted_fabric",
    )
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ClusterConfigError, match="must not start with '-'"):
        lifecycle.up(config)
    assert touched == []
    assert capsys.readouterr().out == ""


def test_package_sync_refuses_a_bad_ssh_identity_before_its_lookup(monkeypatch):
    touched = _record_host_work(monkeypatch)
    config = ClusterConfig(transport_security="trusted_fabric")

    with pytest.raises(ClusterConfigError, match=r"hosts\.name must not contain '@'"):
        lifecycle.sync_package(config, HostConfig("user@evil", "tcp://10.0.0.2:26600"))
    assert touched == []


def test_restart_refuses_an_unacknowledged_transport_before_stopping_anything(monkeypatch, capsys):
    """A refusal after the stop would leave every worker stopped.

    The refusal needs only the config, so it comes first and the workers stay
    as they were.
    """
    config = ClusterConfig(hosts=(HostConfig("worker", "tcp://10.0.0.2:26600"),))
    touched = _record_host_work(monkeypatch)

    assert lifecycle.restart(config, sync=False) is False
    out = capsys.readouterr().out
    assert out.startswith("refusing to start the worker service: peer authentication is unavailable")
    assert "down" not in out
    assert touched == []


def test_restart_refuses_a_malformed_later_address_before_stopping_anything(monkeypatch, capsys):
    config = ClusterConfig(
        hosts=(
            HostConfig("worker", "tcp://10.0.0.2:26600"),
            HostConfig("worker-b", "tcp://worker-b:26600"),
        ),
        transport_security="trusted_fabric",
    )
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ValueError, match=r"^not a complete tcp endpoint: 'tcp://worker-b:26600'$"):
        lifecycle.restart(config, sync=False)
    assert touched == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("command", ["install_systemd", "units_are_removable", "uninstall_systemd"])
def test_systemd_commands_refuse_a_control_character_before_any_host_is_resolved(
    monkeypatch, command: str,
):
    """The unit renderer's refusal comes before any host's locality lookup."""
    config = ClusterConfig(
        hosts=(HostConfig("worker", "tcp://10.0.0.2:26600"),),
        transport_security="trusted_fabric",
        python_bin="py\tthon",
    )
    touched = _record_host_work(monkeypatch)

    with pytest.raises(ValueError, match=r"^systemd arguments must not contain control characters$"):
        getattr(lifecycle, command)(config)
    assert touched == []


@pytest.mark.parametrize("command", ["install_systemd", "units_are_removable", "uninstall_systemd"])
@pytest.mark.parametrize(
    ("later_host", "refusal"),
    [
        (HostConfig("worker-b", "tcp://10.0.0.3:26600\n"),
         r"^systemd arguments must not contain control characters$"),
        (HostConfig("user@evil", "tcp://10.0.0.3:26600"),
         "hosts.name must not contain '@', which rewrites the SSH login target"),
    ],
    ids=["address-newline", "name-at"],
)
def test_systemd_commands_refuse_a_bad_later_host_before_touching_an_earlier_one(
    monkeypatch, capsys, command: str, later_host: HostConfig, refusal: str,
):
    """Host one's unit must not be probed, synced, installed or removed before host two is refused."""
    config = ClusterConfig(
        hosts=(HostConfig("worker", "tcp://10.0.0.2:26600"), later_host),
        transport_security="trusted_fabric",
    )
    touched = _record_host_work(monkeypatch)

    with pytest.raises((ValueError, ClusterConfigError), match=refusal):
        getattr(lifecycle, command)(config)
    assert touched == []
    assert capsys.readouterr().out == ""


def test_generation_start_refuses_a_legacy_unfenced_systemd_unit_before_mutation(
    monkeypatch, tmp_path: Path,
):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    units = tmp_path / ".config/systemd/user"
    units.mkdir(parents=True)
    unit = units / "dgxm-worker.service"
    unit.write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")
    state = tmp_path / ".local/state/dgx-monarch"
    state.mkdir(parents=True)
    marker = state / "worker-generation"
    marker.write_text("prior-authority\n", encoding="ascii")

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle, "locked_script", lambda _python, script, **_kwargs: script
    )
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda _config, _host, script, timeout=60: subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        ),
    )

    assert lifecycle.up(config, sync=False, generation=UPDATE_GENERATION) is None
    assert marker.read_text(encoding="ascii") == "prior-authority\n"
    assert unit.is_file()


def test_generation_replacement_refuses_stale_stop_without_mutation(
    monkeypatch, tmp_path: Path, capsys,
):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    state = tmp_path / ".local" / "state" / "dgx-monarch"
    state.mkdir(parents=True)
    for directory in (tmp_path / ".local", tmp_path / ".local/state", state):
        directory.chmod(0o700)
    replacement = "u-fedcba987654-fedcba9876543210"
    (state / "worker-generation").write_text(replacement + "\n", encoding="ascii")
    mutated = tmp_path / "mutated"

    def run_host(_config, _host, script, timeout=60):
        functions = (
            f'systemctl() {{ : > "{mutated}"; return 0; }}\n'
            f'pkill() {{ : > "{mutated}"; return 0; }}\n'
        )
        return subprocess.run(
            ["bash", "-c", functions + script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        )

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle, "locked_script", lambda _python, script, **_kwargs: script
    )
    monkeypatch.setattr(lifecycle, "run_on_host", run_host)

    assert lifecycle.down_if_generation(config, UPDATE_GENERATION) is None
    assert not mutated.exists()
    output = capsys.readouterr().out
    assert "UNKNOWN (worker generation changed)" in output
    assert UPDATE_GENERATION not in output and replacement not in output


def test_same_generation_with_restarted_process_refuses_aba_stop(
    monkeypatch, tmp_path: Path,
):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    state = tmp_path / ".local" / "state" / "dgx-monarch"
    state.mkdir(parents=True)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    (state / "worker-generation").write_text(
        f"DGXM_WORKER_GENERATION_V1\n{UPDATE_GENERATION}\n{boot_id}\n{os.getpid()}:0\n",
        encoding="ascii",
    )
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    pgrep = tool_dir / "pgrep"
    pgrep.write_text(f"#!/bin/sh\nprintf '%s\\n' {os.getpid()}\n", encoding="ascii")
    pgrep.chmod(0o700)
    mutated = tmp_path / "mutated"

    def run_host(_config, _host, script, timeout=60):
        functions = f'systemctl() {{ : > "{mutated}"; return 0; }}\n'
        return subprocess.run(
            ["bash", "-c", functions + script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"HOME": str(tmp_path), "PATH": f"{tool_dir}:/usr/bin:/bin"},
        )

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle, "locked_script", lambda _python, script, **_kwargs: script
    )
    monkeypatch.setattr(lifecycle, "run_on_host", run_host)

    assert lifecycle.down_if_generation(config, UPDATE_GENERATION) is None
    assert not mutated.exists()


def test_generation_guard_accepts_the_exact_boot_pid_and_birth_identity(tmp_path: Path):
    state = tmp_path / ".local" / "state" / "dgx-monarch"
    state.mkdir(parents=True)
    for directory in (tmp_path / ".local", tmp_path / ".local/state", state):
        directory.chmod(0o700)
    pid = os.getpid()
    stat_tail = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    starttime = stat_tail[19]
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    (state / "worker-generation").write_text(
        f"DGXM_WORKER_GENERATION_V1\n{UPDATE_GENERATION}\n{boot_id}\n{pid}:{starttime}\n",
        encoding="ascii",
    )
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    pgrep = tool_dir / "pgrep"
    pgrep.write_text(f"#!/bin/sh\nprintf '%s\\n' {pid}\n", encoding="ascii")
    pgrep.chmod(0o700)

    result = subprocess.run(
        ["bash", "-c", require_generation_shell(UPDATE_GENERATION, "exact-pattern")],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": f"{tool_dir}:/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr


def test_generation_guard_accepts_confirmed_inactive_start_settlement(tmp_path: Path):
    state = tmp_path / ".local" / "state" / "dgx-monarch"
    state.mkdir(parents=True)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    (state / "worker-generation").write_text(
        f"DGXM_WORKER_GENERATION_V1\n{UPDATE_GENERATION}\n{boot_id}\n0:0\n",
        encoding="ascii",
    )
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    pgrep = tool_dir / "pgrep"
    pgrep.write_text("#!/bin/sh\nexit 1\n", encoding="ascii")
    pgrep.chmod(0o700)

    result = subprocess.run(
        ["bash", "-c", require_generation_shell(UPDATE_GENERATION, "exact-pattern")],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": f"{tool_dir}:/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr


def test_actual_prestart_refuses_while_generation_bound_stop_holds_fence(
    tmp_path: Path,
):
    state = tmp_path / ".local" / "state" / "dgx-monarch"
    state.mkdir(parents=True)
    for directory in (tmp_path / ".local", tmp_path / ".local/state", state):
        directory.chmod(0o700)
    pid = os.getpid()
    stat_tail = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    starttime = stat_tail[19]
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    (state / "worker-generation").write_text(
        f"DGXM_WORKER_GENERATION_V1\n{UPDATE_GENERATION}\n{boot_id}\n{pid}:{starttime}\n",
        encoding="ascii",
    )
    ready = tmp_path / "fence-ready"
    release = tmp_path / "fence-release"
    # The holder keeps the fence until the prestart below has answered, 5 s at most.
    holder = subprocess.Popen(
        ["bash", "-c", locked_script(
            sys.executable,
            f'touch "{ready}"; i=0; '
            f'while [ ! -e "{release}" ] && [ "$i" -lt 500 ]; do sleep 0.01; i=$((i + 1)); done',
            generation_fence=True,
        )],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    deadline = time.monotonic() + 2
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    result = subprocess.run(
        generation_prestart_argv(),
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 75
    assert result.stdout.strip() == "UNKNOWN_GENERATION_FENCE"
    assert (state / "worker-generation").exists()
    release.touch()
    assert holder.wait(timeout=5) == 0


def test_actual_prestart_durably_revokes_an_uncontended_marker(tmp_path: Path):
    state = tmp_path / ".local" / "state" / "dgx-monarch"
    state.mkdir(parents=True)
    for directory in (tmp_path / ".local", tmp_path / ".local/state", state):
        directory.chmod(0o700)
    marker = state / "worker-generation"
    marker.write_text("stale\n", encoding="ascii")
    marker.chmod(0o600)

    result = subprocess.run(
        generation_prestart_argv(),
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("show_output", "show_status"),
    [
        ("ActiveState=activating\nSubState=start\nMainPID=0\nJob=1\n", 0),
        ("ActiveState=deactivating\nSubState=stop\nMainPID=1\nJob=2\n", 0),
        ("ActiveState=inactive\n", 0),
        ("", 1),
    ],
)
def test_failed_start_quarantine_requires_exact_inactive_systemd_readback(
    tmp_path: Path, show_output: str, show_status: int,
):
    state = tmp_path / ".local/state/dgx-monarch"
    state.mkdir(parents=True)
    for directory in (tmp_path / ".local", tmp_path / ".local/state", state):
        directory.chmod(0o700)
    tools = tmp_path / "bin"
    tools.mkdir()
    systemctl = tools / "systemctl"
    systemctl.write_text(
        "#!/bin/sh\n"
        "if [ \"$2\" = stop ]; then exit 0; fi\n"
        f"printf %s {shlex.quote(show_output)}\nexit {show_status}\n",
        encoding="utf-8",
    )
    systemctl.chmod(0o700)
    pkill = tools / "pkill"
    pkill.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    pkill.chmod(0o700)

    result = subprocess.run(
        generation_quarantine_argv(
            UPDATE_GENERATION, "definitely-no-worker-process$", "systemd",
            "dgxm-worker.service",
        ),
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": f"{tools}:/usr/bin:/bin"},
    )

    assert result.returncode == 75
    assert result.stdout.strip() == "UNKNOWN_START_SETTLEMENT"
    assert not (state / "worker-generation").exists()


def test_failed_start_quarantine_records_only_stable_inactive_settlement(tmp_path: Path):
    state = tmp_path / ".local/state/dgx-monarch"
    state.mkdir(parents=True)
    for directory in (tmp_path / ".local", tmp_path / ".local/state", state):
        directory.chmod(0o700)
    tools = tmp_path / "bin"
    tools.mkdir()
    systemctl = tools / "systemctl"
    systemctl.write_text(
        "#!/bin/sh\n"
        "if [ \"$2\" = stop ]; then exit 0; fi\n"
        "printf 'ActiveState=inactive\\nSubState=dead\\nMainPID=0\\nJob=0\\n'\n",
        encoding="utf-8",
    )
    systemctl.chmod(0o700)
    pkill = tools / "pkill"
    pkill.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    pkill.chmod(0o700)

    result = subprocess.run(
        generation_quarantine_argv(
            UPDATE_GENERATION, "definitely-no-worker-process$", "systemd",
            "dgxm-worker.service",
        ),
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": f"{tools}:/usr/bin:/bin"},
    )

    marker = state / "worker-generation"
    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="ascii").splitlines()[1:] == [
        UPDATE_GENERATION,
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "0:0",
    ]


def test_generation_refusal_dominates_confirmed_peer_settlement(monkeypatch):
    hosts = (
        HostConfig("owned", "tcp://10.0.0.2:26600"),
        HostConfig("replaced", "tcp://10.0.0.3:26600"),
    )
    config = ClusterConfig(hosts=hosts, transport_security="trusted_fabric")
    replies = iter(
        (
            subprocess.CompletedProcess([], 0, "DONE_NOHUP\n" + SWEEP_OK, ""),
            subprocess.CompletedProcess([], 75, "REFUSED_GENERATION\n", ""),
        )
    )
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle, "run_on_host", lambda *_args, **_kwargs: next(replies)
    )

    assert lifecycle.down_if_generation(config, UPDATE_GENERATION) is None


def test_up_sweeps_orphans_only_after_the_start_script(monkeypatch):
    config, scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 0, "STARTED_NOHUP\n", ""),
        subprocess.CompletedProcess([], 0, SWEEP_OK, ""),
    ])
    assert lifecycle.up(config, sync=False)
    assert "nohup" in scripts[0]
    assert "-m dgx_monarch.cli.actor_reaper" in scripts[0]
    assert "--all-loop-children" not in scripts[0]


def test_up_start_failure_still_sweeps_orphans_inside_the_lock(monkeypatch, capsys):
    config, scripts = _sweep_capture(
        monkeypatch,
        [
            subprocess.CompletedProcess([], 1, "FAILED_NOHUP\n", ""),
            subprocess.CompletedProcess([], 0, SWEEP_OK, ""),
        ],
    )
    assert lifecycle.up(config, sync=False) is False
    assert len(scripts) == 1 and "dgx_monarch.cli.actor_reaper" in scripts[0]
    out = capsys.readouterr().out
    assert "actor sweep reaped=1" in out
    assert "worker: up" not in out


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("FAILED_GENERATION_RECORD", False),
        ("UNKNOWN_START_SETTLEMENT", None),
    ],
)
def test_generation_record_settlement_preserves_start_certainty(
    monkeypatch, marker: str, expected: bool | None,
):
    config, _scripts = _sweep_capture(
        monkeypatch,
        [subprocess.CompletedProcess([], 0, marker + "\n" + SWEEP_OK, "")],
    )

    assert lifecycle.up(config, sync=False, generation=UPDATE_GENERATION) is expected


def test_a_host_without_the_reaper_module_does_not_fail_down(monkeypatch):
    """In a mixed-version fleet during rollout, a sweep that cannot run is a note."""
    config, _scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 0, "DONE_SYSTEMD\n", ""),
        subprocess.CompletedProcess(
            [], 1, "", "No module named dgx_monarch.cli.actor_reaper\n"),
    ])
    assert lifecycle.down(config)


def test_a_sweep_that_prints_no_contract_line_does_not_fail_the_host(monkeypatch):
    config, _scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 0, "DONE_NOHUP\n", ""),
    ])
    assert lifecycle.down(config)


def test_a_sweep_that_crashes_after_the_ladder_fails_the_host(monkeypatch, capsys):
    """Exit 0 with no summary is a stub; a non-zero exit with none is a crash.

    The module prints its one contract line only after the whole SIGTERM then
    SIGKILL ladder has run, so a traceback, an OOM kill or truncated output
    before that line leaves an unknown number of survivors, and the host must
    not report `down`.
    """
    config, _scripts = _sweep_capture(monkeypatch, [
        subprocess.CompletedProcess([], 0, "DONE_SYSTEMD\n", ""),
        subprocess.CompletedProcess(
            [], 1, "", "Traceback (most recent call last):\nOSError: /proc vanished\n"),
    ])
    assert lifecycle.down(config) is False
    assert "unaccounted for" in capsys.readouterr().out


def test_a_sweep_transport_timeout_is_unknown_without_exception_detail(
    monkeypatch, capsys
):
    """A timeout is not a version problem, so it is not a note.

    The kill ladder takes at most SWEEP_TERM_WAIT_S plus SWEEP_KILL_WAIT_S
    (15 s), well inside the 75 s timeout `down` gives its stop-and-sweep
    payload, so reaching it means the host is wedged or a target is stuck in
    uninterruptible sleep: the state that must not print `down`.
    """
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    calls: list[str] = []

    def run_host(_config, _host, script, timeout=60):
        calls.append(script)
        raise subprocess.TimeoutExpired("bash", timeout)

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(lifecycle, "run_on_host", run_host)
    assert lifecycle.down(config) is None
    out = capsys.readouterr().out
    assert "UNKNOWN (lifecycle transport unavailable)" in out
    assert "timed out" not in out


@pytest.mark.parametrize("operation", ["up", "down"])
def test_lifecycle_response_without_a_terminal_marker_is_unknown(
    monkeypatch, capsys, operation
):
    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "ssh banner without settlement\n", ""
        ),
    )

    result = (
        lifecycle.up(config, sync=False)
        if operation == "up"
        else lifecycle.down(config)
    )

    assert result is None
    assert "UNKNOWN (no terminal lifecycle marker)" in capsys.readouterr().out


@pytest.mark.parametrize("operation", ["up", "down"])
def test_unknown_host_dominates_a_definite_failure_in_mixed_host_results(
    monkeypatch, capsys, operation
):
    hosts = (
        HostConfig("definite", "tcp://10.0.0.2:26600"),
        HostConfig("ambiguous", "tcp://10.0.0.3:26600"),
    )
    config = ClusterConfig(hosts=hosts, transport_security="trusted_fabric")
    replies = iter(
        (
            subprocess.CompletedProcess(
                [],
                1,
                ("FAILED_NOHUP\n" + SWEEP_OK)
                if operation == "up"
                else "FAILED_NOHUP_ACTIVE\n",
                "",
            ),
            subprocess.CompletedProcess([], 255, "", "connection lost"),
        )
    )
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda *_args, **_kwargs: next(replies),
    )

    result = (
        lifecycle.up(config, sync=False)
        if operation == "up"
        else lifecycle.down(config)
    )

    assert result is None
    assert "UNKNOWN (lifecycle transport exited 255)" in capsys.readouterr().out


def test_the_sweep_quotes_the_loop_address(monkeypatch):
    config, scripts = _sweep_capture(
        monkeypatch,
        [subprocess.CompletedProcess([], 0, "DONE_SYSTEMD\n", ""),
         subprocess.CompletedProcess([], 0, SWEEP_OK, "")],
        host_name="fd00::2", address="tcp://[fd00::2]:26600",
    )
    assert lifecycle.down(config)
    assert "--loop-address 'tcp://[fd00::2]:26600'" in scripts[0]
