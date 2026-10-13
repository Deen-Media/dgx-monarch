"""Helpers shared by the actor and driver tensor-transfer paths."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
from typing import Any

import torch

from .error_utils import (
    failure_summary,
    failure_text,
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
    safe_call,
    safe_note,
)

__all__ = [
    "failure_summary",
    "failure_text",
    "prefer_error",
    "raise_with_distinct_cause",
    "reconcile_error",
    "safe_call",
    "safe_note",
]


def return_threshold(mode: str, min_bytes: int | None) -> int:
    from .constants import DEFAULT_RDMA_MIN_BYTES

    if mode not in ("message", "rdma"):
        raise ValueError(f"unknown latent return mode {mode!r}")
    if mode == "rdma":
        from .actor import comfy_dynamic  # lazy: the driver has no actor stack

        comfy_dynamic.assert_rdma_compatible()
    if min_bytes is None:
        min_bytes = int(os.environ.get("DGXM_RDMA_MIN_BYTES", DEFAULT_RDMA_MIN_BYTES))
    return max(1, int(min_bytes))


def native_rdma_buffer(tensor: torch.Tensor) -> Any:
    """Construct monarch.rdma's RDMABuffer, importing it on first use.

    Node registration imports transfer.py. A module-level Monarch import there
    loads the native runtime before prepare_driver installs the transport
    policy and OTLP endpoint, so both native names resolve here, at call time.
    """
    from monarch.rdma import RDMABuffer

    return RDMABuffer(tensor)


def ibverbs_available() -> bool:
    """monarch.rdma.is_ibverbs_available, imported on first use like native_rdma_buffer."""
    from monarch.rdma import is_ibverbs_available

    return bool(is_ibverbs_available())


def native_policy_ready(log: Any) -> bool:
    from .monarch_policy import verify_native_policy

    try:
        verify_native_policy()
    except Exception:
        safe_call(log.error, "RDMA policy readback unavailable; using messaging before registration")
        return False
    return True


def rdma_drop_phase():
    from .runtime_diagnostics import phase

    return phase("rdma_drop", return_mode="rdma")


def rdma_register_phase():
    from .runtime_diagnostics import phase

    return phase("rdma_register", return_mode="rdma")


def qp_split() -> int:
    from .monarch_policy import application_chunks

    admitted = application_chunks()
    if admitted is not None:
        return admitted
    try:
        return max(1, min(8, int(os.environ.get("DGXM_RDMA_QP_SPLIT", "1"))))
    except ValueError:
        return 1


def native_rdma_preflight() -> bool:
    """Probe native registration in a child so a native abort stays isolated."""
    code = (
        "import os\n"
        "from dgx_monarch.monarch_policy import POLICY_ENV, policy_from_token, apply_policy, verify_native_policy\n"
        "apply_policy(policy_from_token(os.environ[POLICY_ENV]), role='preflight', native_enabled=True)\n"
        "verify_native_policy()\n"
        "import torch\n"
        "from monarch._rust_bindings import rdma as native_rdma\n"
        "from monarch.rdma import RDMABuffer, is_ibverbs_available\n"
        "assert is_ibverbs_available()\n"
        "buffer = RDMABuffer(torch.zeros(4096, dtype=torch.uint8))\n"
        "buffer.drop().get(timeout=5)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def rdma_outer_timeout(
    desc: dict, read_timeout: float, get_margin: float,
    ack_timeout: float, ack_attempts: int, outer_margin: float,
) -> float:
    ack_budget = ack_timeout * ack_attempts if (
        desc.get("owner_token") is not None
        or desc.get("setup_generation") is not None
    ) else 0.0
    return read_timeout * (len(desc["parts"]) + 1) + get_margin + ack_budget + outer_margin


def byte_split(nbytes: int, count: int) -> list[tuple[int, int]]:
    count = max(1, min(count, nbytes))
    base = nbytes // count
    ranges, offset = [], 0
    for index in range(count):
        size = nbytes - offset if index == count - 1 else base
        ranges.append((offset, size))
        offset += size
    return ranges


def read_parts_concurrent(
    parts: list[dict], out_bytes: torch.Tensor, timeout: float, margin: float
) -> None:
    def read_one(part: dict) -> None:
        from .runtime_diagnostics import phase

        offset, size = part["offset"], part["nbytes"]
        with phase("rdma_read", return_mode="rdma"):
            part["buffer"].read_into(
                out_bytes[offset:offset + size], timeout=timeout
            ).get(timeout=timeout + margin)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(parts)) as executor:
        for future in [executor.submit(read_one, part) for part in parts]:
            future.result()


def is_nested_tensor(value: Any) -> bool:
    return type(value).__name__ == "NestedTensor" and hasattr(value, "unbind")


def tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=False).contiguous()
    if is_nested_tensor(value):
        return type(value)(tuple(tree_to_cpu(t) for t in value.unbind()))
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    return value


def pack_conditioning(conds: Any) -> Any:
    return tree_to_cpu(conds)


def pack_latent(latent: dict) -> dict:
    return tree_to_cpu(latent)
