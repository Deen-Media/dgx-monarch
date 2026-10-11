"""Explicit model-free Sol kernel smoke; never used by passive Doctor."""
from __future__ import annotations

import inspect
import json
import math
import shlex
import subprocess
from collections.abc import Callable
from typing import Any, TypedDict, cast

from ..config import ClusterConfig
from .lifecycle import _pybin_shell, run_on_host
from .lifecycle_lock import lifecycle_lock_source
from .probe_certainty import UNKNOWN_EXIT


class SmokeResult(TypedDict):
    backend: str
    capability: list[int]
    relative_l2: float
    shape: list[int]
    dtype: str
    scope: str


RESULT_PREFIX = "DGXM_SOL_SMOKE="


def kernel_control(torch: Any, sol_attn: Callable[..., Any], get_backend: Callable[..., str],
                   enable_cute: Callable[[], str]) -> SmokeResult:
    """Execute one BF16 full-KV control; injectable dependencies keep tests CPU-only."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(0)
    capability = tuple(torch.cuda.get_device_capability())
    enable_cute()
    backend = str(get_backend(torch.cuda.current_device()))
    if capability == (12, 1) and backend != "cute_sm120":
        raise RuntimeError("GB10 requires the CuTe backend")
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.inference_mode():
        q, k, v = (torch.randn((1, 4096, 1, 128), device="cuda", dtype=torch.bfloat16,
                               generator=generator).contiguous() for _ in range(3))
        output = sol_attn(q, k, v, tau=1.0, thresh_type="exact", kv_splits=1,
                          sink_start=0, sink_tokens=4096)
        torch.cuda.synchronize()
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2).contiguous()
        torch.cuda.synchronize()
        if output.dtype != q.dtype or output.device != q.device:
            raise RuntimeError("Sol output dtype or device differs from BF16 CUDA input")
        if output.shape != reference.shape or not bool(torch.isfinite(output).all()):
            raise RuntimeError("Sol output shape or finiteness check failed")
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError("SDPA reference is nonfinite")
        denominator = reference.float().norm().item()
        if denominator <= 0:
            raise RuntimeError("SDPA reference has zero norm")
        error = (output.float() - reference.float()).norm().item() / denominator
        if not 0 <= error <= 0.01:
            raise RuntimeError("Sol dense control exceeds relative L2 tolerance")
    return {"backend": backend, "capability": list(capability), "relative_l2": error,
            "shape": [1, 4096, 1, 128], "dtype": "bfloat16", "scope": "dense-kernel-control"}


def _parse_result(stdout: str) -> SmokeResult:
    lines = [line[len(RESULT_PREFIX):] for line in stdout.splitlines()
             if line.startswith(RESULT_PREFIX)]
    if len(lines) != 1:
        raise ValueError("missing or ambiguous smoke result")
    result = json.loads(lines[0])
    if not isinstance(result, dict):
        raise ValueError("invalid smoke result")
    error = result.get("relative_l2")
    capability = result.get("capability")
    backend = result.get("backend")
    if (not isinstance(error, (int, float)) or isinstance(error, bool) or not math.isfinite(error) or not 0 <= error <= 0.01
            or not isinstance(capability, list) or len(capability) != 2
            or not all(type(part) is int and 0 <= part < 100 for part in capability)
            or not isinstance(backend, str) or not backend or len(backend) > 80
            or not all(char.isalnum() or char == "_" for char in backend)
            or (capability == [12, 1] and backend != "cute_sm120")
            or result.get("shape") != [1, 4096, 1, 128]
            or result.get("dtype") != "bfloat16"
            or result.get("scope") != "dense-kernel-control"):
        raise ValueError("invalid smoke evidence")
    return cast(SmokeResult, result)


def smoke_script(python_bin: str) -> str:
    """Guard and kernel share a process under the ordinary host lifecycle lock."""
    from .sol_install_guard import PROBE_SOURCE

    source = ("from __future__ import annotations\n" + lifecycle_lock_source() + "\n" + PROBE_SOURCE + "\n"
              + inspect.getsource(kernel_control) + '''
import json, os
from pathlib import Path
lock_fd = acquire_lifecycle_lock(Path.home(), nonblocking=True)
require_idle()
from dgx_monarch.cli.worker_process_env import filter_worker_environment
filter_worker_environment(os.environ)
import torch
from sol_attn import sol_attn, get_sol_attn_backend
from dgx_monarch.adapters.sol_attention_backend import enable_cute_on_gb10
result = kernel_control(torch, sol_attn, get_sol_attn_backend, enable_cute_on_gb10)
print("DGXM_SOL_SMOKE=" + json.dumps(result, allow_nan=False))
''')
    return "set -eu\n" + _pybin_shell(python_bin) + '\n"$PYBIN" -I -c ' + shlex.quote(source)


def cmd_sol_smoke(config: ClusterConfig, *, idle_maintenance: bool = False,
                  timeout_s: int = 300) -> int:
    """Run explicit kernel controls sequentially, with no mesh or Worker changes."""
    if not idle_maintenance:
        print("Sol smoke refused: --idle-maintenance is required; close ComfyUI and settle actors first.")
        return 2
    if not config.hosts or any(host.gpus != 1 for host in config.hosts):
        print("Sol smoke requires configured single-GPU hosts.")
        return 2
    if not 30 <= timeout_s <= 1800:
        print("Sol smoke timeout must be between 30 and 1800 seconds.")
        return 2
    script = smoke_script(config.python_bin)
    for host in config.hosts:
        try:
            completed = run_on_host(config, host, script, timeout=timeout_s,
                                    require_known_locality=True)
            if completed.returncode:
                print(f"{host.name}: Sol smoke refused or failed (exit {completed.returncode}); "
                      "check idle state and optional dependencies. Remaining hosts untested.")
                return 1
            result = _parse_result(completed.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            print(f"{host.name}: Sol smoke outcome unknown; settle this host before retrying. "
                  "Remaining hosts untested.")
            return UNKNOWN_EXIT
        print(f"{host.name}: dense kernel control PASS backend={result['backend']} "
              f"relative_l2={result['relative_l2']:.6g}")
    print("Model-free kernel execution only; no render, sparse-attention fidelity or model accuracy evidence.")
    return 0
