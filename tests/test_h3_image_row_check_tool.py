"""tools/h3_image_row_check.py runs on CPU against a real ComfyUI and writes its record.

The hardware run that vouches for the three row-keyed calls at a rank's rows is
the maintainer's (docs/VALIDATION.md); this smoke holds the tool to ComfyUI's
calls, so the hardware run does not fail on a renamed one. It runs in a
subprocess, leaving this process's ComfyUI imports untouched.

Bits per row at a rank's row count hold on the rig's aarch64 CPU; x86 BLAS may pick
another kernel for the shorter call, so the comfy canary's x86 runner compares within
2e-5 (tests/test_ltx_comfy_exactness.py, #163). Bit identity on GB10 is the hardware run.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
EXACT = platform.machine() == "aarch64"
REAL_CALLS = ("qkv_proj", "rms_rope_split_half_", "out_proj", "mlp")


def _equal_or_close(record):
    rows = [row for row in record["calls"] if row["call"] in REAL_CALLS]
    return all(row["equal"] for row in rows) if EXACT else all(
        row["max_abs_diff"] <= 2e-5 for row in rows)


def _tiny_run(tmp_path, *extra):
    model = Path(COMFY_DIR, "comfy", "ldm", "minimax", "model.py")
    if not model.is_file() or "class MiniMaxH3ImageModel" not in model.read_text():
        pytest.skip("no ComfyUI 0.45.0 or newer on this host (comfy-canary job covers it)")
    out = tmp_path / "record.json"
    # No CUDA_VISIBLE_DEVICES here: --tiny must keep itself on CPU.
    env = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")})
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "h3_image_row_check.py"), "--comfy", COMFY_DIR,
         "--tiny", "--latent", "8", "8", "--text-rows", "7", "--world", "2", "--out", str(out),
         *extra],
        env=env, capture_output=True, text=True, timeout=300)
    return done, out


def test_the_tiny_run_records_every_call_on_every_rank(tmp_path):
    done, out = _tiny_run(tmp_path)
    assert done.returncode in ((0,) if EXACT else (0, 1)), done.stdout[-2000:] + done.stderr[-2000:]
    record = json.loads(out.read_text())
    assert record["device"] == "cpu" and record["rows"] == 23 and record["local_rows"] == 12
    assert _equal_or_close(record)
    calls = {(row["call"], row["rank"]) for row in record["calls"]}
    assert calls == {(name, rank) for name in ("qkv_proj", "rms_rope_split_half_", "out_proj", "mlp")
                     for rank in (0, 1)}
    assert [row["real_rows"] for row in record["calls"][:2]] == [12, 11]


def test_the_negative_control_and_only_it_differs(tmp_path):
    """A row-dependent call must compare unequal, or the comparison proves nothing."""
    done, out = _tiny_run(tmp_path, "--self-check")
    assert done.returncode in ((0,) if EXACT else (0, 1)), done.stdout[-2000:] + done.stderr[-2000:]
    record = json.loads(out.read_text())
    assert record["self_check_seen"] and _equal_or_close(record)
    differs = {row["call"] for row in record["calls"] if not row["equal"]}
    assert "self_check_row_dependent" in differs
    if EXACT:
        assert differs == {"self_check_row_dependent"} and record["all_equal"]
