"""tools/h3_image_row_check.py runs on CPU against a real ComfyUI and writes its record.

The hardware run that vouches for the three row-keyed calls at a rank's rows is
the maintainer's (docs/VALIDATION.md); this smoke holds the tool to ComfyUI's
calls, so the hardware run does not fail on a renamed one. It runs in a
subprocess, leaving this process's ComfyUI imports untouched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))


def test_the_tiny_run_records_every_call_on_every_rank(tmp_path):
    model = Path(COMFY_DIR, "comfy", "ldm", "minimax", "model.py")
    if not model.is_file() or "class MiniMaxH3ImageModel" not in model.read_text():
        pytest.skip("no ComfyUI 0.45.0 or newer on this host (comfy-canary job covers it)")
    out = tmp_path / "record.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "h3_image_row_check.py"), "--comfy", COMFY_DIR,
         "--tiny", "--latent", "8", "8", "--text-rows", "7", "--world", "2", "--out", str(out)],
        env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    record = json.loads(out.read_text())
    assert record["rows"] == 23 and record["local_rows"] == 12 and record["all_equal"]
    calls = {(row["call"], row["rank"]) for row in record["calls"]}
    assert calls == {(name, rank) for name in ("qkv_proj", "rms_rope_split_half_", "out_proj", "mlp")
                     for rank in (0, 1)}
    assert [row["real_rows"] for row in record["calls"][:2]] == [12, 11]
