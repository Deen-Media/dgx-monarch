"""The ``dgxm doctor`` row for the torch the actors import and their NCCL backend.

torch 2.15 maps the "nccl" backend name to its nccl2 implementation, which
aborts an actor at its first process group under ComfyUI's cudaMallocAsync
allocator, so the launcher pins torch's legacy ProcessGroupNCCL in every
actor's start environment (docs/TROUBLESHOOTING.md #143). The row builds that
environment with the launcher's own function, so it reads what an actor would
start with and needs no mesh, SSH or CUDA.
"""
from __future__ import annotations

import re

from ..config import ClusterConfig
from .probe_certainty import FAIL, OK, WARN

NAME = "actor NCCL backend"
ENTRY = "docs/TROUBLESHOOTING.md #143"
# The first torch whose "nccl" name resolves to the nccl2 backend.
NCCL2_DEFAULT_FROM = (2, 15)


def _torch_build() -> str:
    try:
        import torch
    except Exception as exc:
        return f"not importable ({type(exc).__name__})"
    return str(torch.__version__)


def _needs_pin(torch_build: str) -> bool | None:
    """Whether this torch resolves "nccl" to nccl2; None when the version cannot be read."""
    match = re.match(r"(\d+)\.(\d+)", torch_build)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2))) >= NCCL2_DEFAULT_FROM


def doctor_row(config: ClusterConfig | None, torch_build: str | None = None) -> tuple[str, str, str]:
    """``(status, name, detail)`` for the actor start environment and this interpreter's torch.

    Local actors start from this interpreter. On a cluster each Worker host's
    actors import the torch its env row shows, and ``torch version match``
    holds them to this one. A missing pin fails under torch 2.15 or later and
    warns before it, where the legacy backend is the only one.
    """
    from ..monarch_launch import LEGACY_NCCL_ENV, actor_environment

    build = _torch_build() if torch_build is None else torch_build
    try:
        environment = actor_environment(config or ClusterConfig(), "")
    except Exception as exc:
        return (WARN, NAME, f"unobserved: the actor start environment could not be built "
                            f"({type(exc).__name__}); torch {build} here")
    where = ("local actors import it" if config is None or not config.hosts
             else "each Worker host's actors import the torch its env row shows")
    setting = " ".join(f"{key}={value}" for key, value in LEGACY_NCCL_ENV.items())
    needs = _needs_pin(build)
    if all(environment.get(key) == value for key, value in LEGACY_NCCL_ENV.items()):
        effect = ("which torch before 2.15 ignores" if needs is False
                  else 'so "nccl" keeps torch\'s legacy ProcessGroupNCCL')
        return (OK, NAME, f"torch {build} here, and {where}; every actor starts with "
                          f"{setting}, {effect}")
    return (FAIL if needs is not False else WARN, NAME,
            f"torch {build} here, and {where}; the actor start environment lacks {setting}, "
            f"so torch 2.15 or later aborts the first process group under cudaMallocAsync ({ENTRY})")
