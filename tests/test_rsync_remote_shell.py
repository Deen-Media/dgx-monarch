"""Exercise how rsync inserts its SSH arguments, without a network connection."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from dgx_monarch.cli import lifecycle
from dgx_monarch.config import ClusterConfig, HostConfig


@pytest.mark.parametrize("name", ["worker.example", "2001:db8::2"])
def test_rsync_inserts_explicit_login_before_the_validated_host(tmp_path, name):
    rsync = shutil.which("rsync")
    if rsync is None:
        pytest.skip("rsync is required for the real transport argument regression")
    binary = tmp_path / "bin"
    binary.mkdir()
    trace = tmp_path / "ssh-argv.json"
    fake = binary / "ssh"
    fake.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ['DGXM_SSH_TRACE']).write_text(json.dumps(sys.argv[1:]))\nsys.exit(77)\n")
    fake.chmod(0o700)
    key = str(tmp_path / "key with spaces")
    host = HostConfig(name, "tcp://192.0.2.2:26600", ssh_user="operator")
    config = ClusterConfig(hosts=(host,), ssh_key=key)
    source = tmp_path / "source.txt"
    source.write_text("fixture")
    result = subprocess.run(
        [rsync, "--dry-run", "-a", "-e", lifecycle._rsync_remote_shell(config, host),
         "--", str(source), lifecycle._rsync_target(host) + ":fixture-destination/"],
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "DGXM_SSH_TRACE": str(trace)},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0  # The fake SSH exits 77 and never starts a receiver.
    arguments = json.loads(trace.read_text())
    assert arguments[arguments.index("-i") + 1] == key
    login = arguments.index("-l")
    assert arguments[login + 1:login + 3] == ["operator", name]
    assert "--" not in arguments[:login + 1]


@pytest.mark.parametrize("name,user", [("-oProxyCommand=x", ""), ("worker.example", "-bad")])
def test_rsync_shell_still_validates_host_and_user_before_building_options(name, user):
    host = HostConfig(name, "tcp://192.0.2.2:26600", ssh_user=user)
    with pytest.raises(ValueError):
        lifecycle._rsync_remote_shell(ClusterConfig(hosts=(host,)), host)
