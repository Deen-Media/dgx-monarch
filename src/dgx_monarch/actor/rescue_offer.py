"""Build capacity rescue descriptors shared by pricing and load decisions.

``store_residency`` raises the offer; ``capacity_quote`` returns it as data.
Both use the same descriptor, blocker pair, and local hostname.

The driver compares only rank-invariant fields: ``consent_id``, ``kind``,
``file_identity``, ``context_fingerprint``, ``memo_context``, and ``family_hint``.
``measured`` and ``human_reason`` describe one rank and are never compared.
This module calls no torch or ComfyUI code and reads no GPU state.
"""
from __future__ import annotations

import socket
from collections.abc import Callable

from .. import consent_descriptor
from ..capacity_fit import SlabFit, StockFit
from ..consent_descriptor import ConsentDescriptor
from ..consent_kinds import KIND_SPECS
from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call
from . import store_slab_admit

log = get_logger(__name__)

_RESCUE_SPEC = KIND_SPECS[consent_descriptor.KIND_RESCUE_SLAB]
_PANEL_ACTION = _RESCUE_SPEC.primary_label

_GENERIC_OPTIONS = (
    "use a pruned or quantized artifact, or free unified memory on this host"
)

# The clause a cold box's rescue card carries, set only where the memo was
# read and came back empty. It goes in ``reason`` rather than a seventeenth
# wire field, which every reader of a quote row would have to learn.
COLD_MEMO_CLAUSE = (
    "This host has no record of loading this file, so residency could not be "
    "resolved from the family memo here; another rank that has loaded it may "
    "resolve differently."
)


def host_name() -> str:
    """This box's own name. No rank ever composes a sentence about a peer."""
    try:
        return socket.gethostname()
    except OSError:
        return "this host"


def memo_context(unet_name: str, node_options: dict | None,
                 lora_stack: list | None) -> dict[str, str]:
    """Build the narrow residency context used by driver consent memos.

    The worker knows which request hit the capacity wall. Matching the loader's
    key format lets the driver find the grant without inferring request details
    from prose. Trust context must stay out of it.
    """
    from ..gate_ledger import combo_key

    options = dict(node_options or {})
    loras = [str(entry.get("name", "")) for entry in (lora_stack or [])
             if isinstance(entry, dict)]
    return {
        "combo_key": combo_key(unet_name, options, loras),
        "weight_dtype": str(options.get("weight_dtype", "default")),
    }


def rescue_blocker(
    *,
    slab_capable_path: bool,
    lora_low_rss: bool,
    fsdp_launch: bool,
    compile_dit: bool,
    explicit_stock: bool,
    blocked_reason: str,
    slab_fit: SlabFit | None = None,
) -> tuple[str, str] | None:
    """Return the slab blocker and a fitting alternative, or ``None``.

    Explicit stock is checked first because identity quarantine disables both
    residency levers. A stale consent must not override that stronger decision
    or suggest re-enabling a quarantined lever. Capacity is checked last, after
    every policy answer: a rescue no policy allows is refused for that reason
    and not for a price nobody would have paid.
    """
    if explicit_stock:
        return (blocked_reason or
                "slab residency is off for this render: the Init node's slab_weights=off "
                "or an identity-gate quarantine turned it off",
                "clear the quarantine with an identity ceremony, or set slab_weights=on when "
                "the combination is not quarantined, or " + _GENERIC_OPTIONS)
    if not slab_capable_path:
        return ("the checkpoint is not a .safetensors or .sft file, which the zero-copy "
                "slab loader requires",
                "convert or re-stage the checkpoint as safetensors, or " + _GENERIC_OPTIONS)
    if fsdp_launch:
        return ("FSDP is active and reshards weights into DTensors, so a slab would be "
                "an unused extra copy",
                "run this shape without FSDP, or " + _GENERIC_OPTIONS)
    if compile_dit:
        return ("compile_dit is on and compiled fp8 GEMM templates reject slab-resident "
                "weights",
                "turn compile_dit off for this render, or " + _GENERIC_OPTIONS)
    if not lora_low_rss:
        return ("lora_low_rss is off, and slab residency requires it (baked mode keeps "
                "comfy's full weight backup, which would cost another copy)",
                "turn lora_low_rss on, or " + _GENERIC_OPTIONS)
    if slab_fit is not None and slab_fit.applies and not slab_fit.fits:
        return store_slab_admit.rescue_blocker(slab_fit)
    return None


def build_descriptor(
    *,
    unet_name: str,
    path: str,
    fit: StockFit,
    model_options: dict,
    file_identity: Callable[[str], str],
    family_hint: str | None,
    node_options: dict | None = None,
    lora_stack: list | None = None,
    context: Callable[..., dict[str, str]] = memo_context,
    host: Callable[[], str] = host_name,
) -> ConsentDescriptor:
    """The rescue offer's machine half, identical on every rank but its numbers.

    ``context`` and ``host`` are injected so the ladder keeps its own module
    attributes as the seam its diagnostics tests replace.
    """
    # Preserve the ComfyUI-relative subpath for driver resolution without
    # exposing a host path.
    name = unet_name
    try:
        identity = file_identity(path)
    except OSError:
        identity = ""
    dtype = model_options.get("weight_dtype") or model_options.get("dtype")
    fingerprint = consent_descriptor.memo_context_fingerprint(
        unet_name=name, weight_dtype=None if dtype is None else str(dtype))
    kind = consent_descriptor.KIND_RESCUE_SLAB
    measured = dict(fit.measured())
    measured["host"] = host()
    try:
        narrow = context(name, node_options, lora_stack)
    except Exception as exc:  # An unkeyed card still communicates the refusal.
        safe_call(log.warning, "consent memo context could not be computed for %s (%s)", name, failure_summary(exc))
        narrow = {}
    return ConsentDescriptor(
        version=consent_descriptor.CONSENT_DESCRIPTOR_VERSION,
        kind=kind,
        consent_id=consent_descriptor.consent_id(kind, identity, fingerprint),
        unet_name=name,
        memo_context=narrow,
        file_identity=identity or "unreadable",
        context_fingerprint=fingerprint,
        family_hint=family_hint,
        measured=measured,
        human_reason=(
            f"{name} needs {fit.required_gib} GiB of unified memory under stock residency "
            f"(a {fit.size_gib} GiB file plus the host copy it is placed from) and only "
            f"{fit.avail_gib} GiB is available. Zero-copy slab residency fits it. Image "
            "identity on the slab path is not proven for this family yet, so this load is "
            "availability-only evidence, never an accuracy verdict."
        ),
        evidence=consent_descriptor.EVIDENCE_BYTE_VERIFIED,
        panel_action=_PANEL_ACTION,
        env_fallback=consent_descriptor.ENV_SLAB_RESCUE,
    )
