"""Test the xFuser step of scripts/setup_env.sh --comfy with fake python, pip and ssh binaries only."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "setup_env.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _fake_home(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "home with spaces"
    bindir = home / "monarch-env" / "bin"
    bindir.mkdir(parents=True)
    comfy = home / "ComfyUI"
    comfy.mkdir()
    (comfy / "requirements.txt").write_text("")
    log = tmp_path / "pip.log"
    _write_executable(
        bindir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *'import torch; print(torch.__version__)'* ]]; then
  [[ "${FAIL_TORCH_PROBE:-}" != 1 ]] || exit 7
  printf '2.12.0+cu132\\n'
elif [[ "$*" == *'sysconfig.get_paths'* ]]; then
  printf '%s\\n' "$HOME/site-packages"
fi
""",
    )
    _write_executable(
        bindir / "pip",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'pip'
while (($#)); do
  if [[ "$1" == --constraint ]]; then
    printf ' constraint=%s' "$(cat "$2")"
    shift 2
  else
    printf ' arg=%s' "$1"
    shift
  fi
done
printf '\\n'
""",
    )
    return home, log


def _run(home: Path, log: Path, *, fail_probe: bool) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"HOME": str(home), "PIP_LOG": str(log)}
    if fail_probe:
        env["FAIL_TORCH_PROBE"] = "1"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--comfy"], cwd=REPO, env=env,
        text=True, capture_output=True,
    )
    # The fake pip prints its calls to stdout. Write the log from stdout after
    # the run, so it keeps the earlier pip calls when the torch probe fails.
    log.write_text(result.stdout)
    return result


def test_comfy_bootstrap_constrains_xfuser_resolution_to_existing_cuda_torch(tmp_path):
    home, log = _fake_home(tmp_path)
    result = _run(home, log, fail_probe=False)

    assert result.returncode == 0, result.stderr
    closure = next(
        line for line in log.read_text().splitlines()
        if "xfuser==0.7.0+dgxm.npuimport1" in line
    )
    assert "constraint=torch==2.12.0+cu132" in closure
    assert "arg=--find-links" in closure
    assert "arg=xfuser==0.7.0+dgxm.npuimport1" in closure
    assert "arg=yunchang==0.6.4" in closure


def test_torch_probe_failure_stops_before_xfuser_resolution(tmp_path):
    home, log = _fake_home(tmp_path)
    result = _run(home, log, fail_probe=True)

    assert result.returncode != 0
    assert "xfuser==0.7.0+dgxm.npuimport1" not in log.read_text()


def test_worker_stream_carries_the_canonical_builder_without_a_peer_checkout(tmp_path):
    local_home, _ = _fake_home(tmp_path / "local")
    remote_home, _ = _fake_home(tmp_path / "remote")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    stream = tmp_path / "worker-stream.sh"
    remote_output = tmp_path / "remote-output.log"
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
set -euo pipefail
cat > "$STREAM_CAPTURE"
env HOME="$REMOTE_HOME" bash -s < "$STREAM_CAPTURE" > "$REMOTE_OUTPUT" 2>&1
""",
    )
    env = os.environ | {
        "HOME": str(local_home),
        "DGXM_SIBLING": "192.0.2.12",
        "STREAM_CAPTURE": str(stream),
        "REMOTE_HOME": str(remote_home),
        "REMOTE_OUTPUT": str(remote_output),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT), "--comfy", "--worker"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = stream.read_text()
    assert "OFFICIAL_SHA256" in payload
    assert "PATCHED_RING" in payload
    remote_pip = remote_output.read_text()
    closure = next(
        line for line in remote_pip.splitlines()
        if "xfuser==0.7.0+dgxm.npuimport1" in line
    )
    assert "constraint=torch==2.12.0+cu132" in closure
    assert "arg=--find-links" in closure
