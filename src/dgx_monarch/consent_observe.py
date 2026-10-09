"""Turn worker refusal descriptors into panel cards.

Pending collection, bound RPC calls and the cross-rank capacity agreement all
route here. A refusal with no consent descriptor raises no card. The
cross-residency gate's stock leg loads under explicit stock, so its capacity
refusal carries none; the ceremony records it in the gate ledger instead, as a
CAPACITY cross-mode verdict and, when the slab leg's certificates and the
refusal's measured numbers are both complete, a certificate row
(``gate_audit.record_ceremony_certificate``). Card registration
is best effort and must never replace the typed refusal; incomplete local
identity grants nothing.
"""
from __future__ import annotations

from typing import Any

from . import consent_descriptor, consent_pending
from .log import get_logger

log = get_logger(__name__)


def _local_path(unet_name: str) -> str | None:
    """Resolve the refused checkpoint in the driver's local model tree."""
    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", unet_name)
    return str(path) if path else None


def _artifacts(path: str) -> Any:
    """Build the checkpoint-only artifact digest named by the refusal.

    The memo combination already binds LoRA names; the descriptor cannot safely
    identify their files, so this function does not invent an artifact set.
    """
    from .gate_ledger import artifact_set_signature, artifact_signature

    return artifact_set_signature([artifact_signature(path)])


def refusal_text(exc: object) -> str:
    """Read the inner refusal without formatting the ActorError wrapper."""
    inner = getattr(exc, "exception", None)
    target = exc if inner is None else inner
    try:
        return target if isinstance(target, str) else str(target)
    except Exception:  # An unreadable message cannot carry usable authority.
        return ""


def _already_granted(kind: str, path: str, memo_context: dict[str, Any]) -> bool:
    """Return whether this exact class-K question already has a memo.

    Class-K grants are resolved per dispatch (nodes/consent_waiver.py) and a
    gate ceremony leg never renders under one, so a class-K refusal can arrive
    under a live memo.
    Capacity refusals under a memo remain visible because they indicate a
    different fault. A store read failure also remains unanswered so it cannot
    swallow the card.
    """
    from . import consent_store
    from .consent_kinds import KIND_SPECS

    spec = KIND_SPECS.get(kind)
    if spec is None or spec.refusal_class != "K":
        return False
    try:
        return consent_store.lookup(kind, path, memo_context) is not None
    except Exception as exc:
        log.warning("consent memo lookup failed while raising a card (%r)", exc)
        return False


def observe_refusal(exc: object) -> dict[str, Any] | None:
    """Register and return the requested card, or return None."""
    try:
        text = refusal_text(exc)
        if consent_descriptor.DESCRIPTOR_BEGIN not in text:
            return None
        descriptor = consent_descriptor.parse(text)
        if descriptor is None:
            return None
        if not consent_descriptor.valid_kind(descriptor.kind):
            return None
        path = _local_path(descriptor.unet_name)
        if path is None:
            log.warning("consent offer names %r, which this driver cannot resolve",
                        descriptor.unet_name)
            return None
        memo_context = dict(descriptor.memo_context or {})
        if not memo_context:
            # Options, LoRA names, and loader dtype are unavailable here. A
            # guessed key would never resolve, so version skew fails closed.
            log.warning(
                "a worker raised a consent offer for %s with no memo context, so this "
                "driver cannot key a grant; the refusal stands and no card was raised. "
                "This is version skew: restart the Worker service (`dgxm restart`) so both "
                "sides run one build", descriptor.unet_name)
            return None
        if _already_granted(descriptor.kind, path, memo_context):
            # Repeating the invitation would only rewrite the same memo.
            log.warning("%s is already granted for %s in this context; the refusal "
                        "stands for this render and no card was raised",
                        descriptor.kind, descriptor.unet_name)
            return None
        artifacts = _artifacts(path)
        card = consent_pending.register_pending(
            consent_pending.descriptor_from_refusal(
                descriptor,
                path=path,
                combo_key=str(memo_context.get("combo_key", "")),
                artifacts=artifacts.current,
                artifacts_legacy=artifacts.legacy,
                artifacts_legacy_complete=artifacts.legacy_complete,
                memo_context=memo_context,
            ))
        if card is not None:
            log.info("consent card raised for %s (%s)", descriptor.unet_name, descriptor.kind)
        return card
    except Exception as exc_info:  # Card registration must not replace the refusal.
        log.warning("worker consent offer not turned into a card (%r)", exc_info)
        return None
