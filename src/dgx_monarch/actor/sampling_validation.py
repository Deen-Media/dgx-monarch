"""Fail-closed validation of worker-side sampler registry names."""
from __future__ import annotations

import socket


def require_known_sampler(sampler_name: str, scheduler: str) -> None:
    """Refuse names this worker's ComfyUI would otherwise substitute."""
    import comfy.samplers

    host = socket.gethostname()
    if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        raise ValueError(
            f"sampler {sampler_name!r} is not registered with ComfyUI on worker host {host}; "
            "stock comfy would silently render euler instead, which under-computes multi-stage "
            "samplers. The custom node pack that provides it failed to preload in the worker "
            "or is missing on this host: find 'custom node pack ... failed to import' in this "
            "host's Worker service log, then install the pack and its python requirements into "
            "this host's worker env. docs/TROUBLESHOOTING.md #11."
        )
    if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
        raise ValueError(
            f"scheduler {scheduler!r} is not registered with ComfyUI on worker host {host}; "
            "stock comfy would silently substitute the default scheduler. The custom node pack "
            "that provides it failed to preload in the worker or is missing on this host "
            "(docs/TROUBLESHOOTING.md #11)."
        )
