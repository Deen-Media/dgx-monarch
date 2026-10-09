"""Header-only gate for artifacts this release has no model forward for.

World size, topology, budget, and consent cannot change this refusal. The
driver checks it at the loader before capacity, mesh healing, setup, or RPC;
Init may already have created the fleet. The worker uses the same message at
its load entry point.

Only standard-library modules load eagerly. Detection imports at call time
so headless callers can import the gate.
"""
from __future__ import annotations

TROUBLESHOOTING = 91


# The body only: the class tag is built at the raise, never here. An
# import-time tag binds to a stale refusal module in a long-lived process
# (see the comment above actor/partial_load_guard.REFUSAL_TEXT).
ZIMAGE_L2P_REFUSAL = (
    "this Z-Image checkpoint is the L2P pixel-space surface (local_decoder "
    "weights, no dec_net head), and this release carries no L2P forward. "
    "Current ComfyUI has no L2P model contract, so it reads the header as "
    "latent Z-Image with a memory_usage_factor of 2.8, asks for about 129 GiB "
    "of sample-time memory at render size on a 121 GiB box, and streams every "
    "block on every step until the host runs out. Even if memory held, the latent forward "
    "would run on pixel-space weights and the image would be wrong. Nothing "
    "was loaded and nothing was quarantined. No consent can supply a missing "
    "forward, so no card will appear and there is no waiver for this refusal. "
    "Use the latent Z-Image checkpoint or the DCT PixelSpace checkpoint "
    "instead; both are hardware validated. This refusal retires when ComfyUI "
    "carries the L2P model contract (upstream draft Comfy-Org/ComfyUI#14055) "
    "and the L2P forward lands (issue #279 section 1)."
)


def refuses(path: str | None) -> bool:
    """True when this release carries no forward for the artifact at ``path``.

    Total: an unnamed, unreadable or non-safetensors path answers False, so a
    failed read degrades to a normal load rather than to a refusal.
    """
    try:
        if not path or not str(path).lower().endswith((".safetensors", ".sft")):
            return False
        from .adapters import detect

        return detect.sniff_zimage_l2p(str(path))
    except Exception:  # a broken read never refuses a load
        return False
