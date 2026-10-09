"""Consent kinds, their permitted grants, and their user-facing copy.

The frozen ``gate_audit_vocab`` supplies each kind's refusal class and guard
allowlist, keeping panel cards consistent with permanent audit rows.
``title``, ``risk``, ``evidence`` and ``primary_label`` are user-facing text.

Dependencies are limited to the standard library, audit vocabulary, and store
canonicalizer.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from . import consent_store
from .gate_audit_vocab import KIND_CLASS


@dataclass(frozen=True, slots=True)
class KindSpec:
    """Describe a consent kind, its UI text, and permitted authorization channels.

    ``default_guard`` must belong to the frozen vocabulary. ``wired`` indicates
    whether this release can offer the kind; unused and retired kinds remain in
    the vocabulary to preserve existing ledger records. Vocabulary changes
    require a protocol version change.

    Class-K waivers travel with each request, not cached worker policy. Their
    memos remain valid until revoked or their identity changes, covering later
    renders of the same combination, topology, and world.
    """

    kind: str
    style: str
    title: str
    risk: str
    evidence: str | None
    primary_label: str
    env_var: str
    auto_eligible: bool
    default_guard: str
    context_fields: tuple[str, ...]
    wired: bool
    measured: str | None = None

    @property
    def refusal_class(self) -> str:
        return KIND_CLASS[self.kind]


_SPECS: tuple[KindSpec, ...] = (
    KindSpec(
        kind="rescue-slab",
        style="capacity",
        title="Load {artifact} with slab residency",
        risk=("Stock residency cannot fit this checkpoint on this box, so the load "
              "would be killed out of memory."),
        evidence=("Slab residency maps the weights from the file instead of copying "
                  "them, and every byte is verified against the checkpoint while it "
                  "loads."),
        primary_label="Load with slab residency",
        env_var="DGXM_ALLOW_SLAB_RESCUE",
        auto_eligible=True,
        default_guard="stock_load_preflight",
        context_fields=("combo_key", "weight_dtype"),
        wired=True,
    ),
    KindSpec(
        kind="waive-unvouched-slab",
        style="capacity",
        title="Use slab residency for {artifact} before it is vouched",
        risk=("This model family has not passed an identity gate under slab "
              "residency, so image identity on this path is unproven."),
        evidence=("The load is still byte-verified against the checkpoint, and a "
                  "later gate failure still quarantines slab for this combination."),
        primary_label="Use slab residency anyway",
        env_var="DGXM_ALLOW_UNVOUCHED_SLAB",
        auto_eligible=False,
        default_guard="slab_vouched_families",
        context_fields=("combo_key", "weight_dtype"),
        # Unwired: explicit slab_weights=on already selects slab for any family.
        # The rescue-slab offer includes the unvouched-family condition; it does
        # not need a separate consent card.
        wired=False,
    ),
    KindSpec(
        kind="waive-first-load-stock",
        style="capacity",
        title="Skip the stock first load for {artifact}",
        risk=("The first load of an unknown checkpoint normally runs stock so the "
              "family is learned from the loaded model, and skipping it means the "
              "family is guessed from the file header."),
        evidence=("The family memo is still written from the loaded model once the "
                  "load finishes."),
        primary_label="Skip the stock first load",
        env_var="DGXM_SKIP_FIRST_LOAD_STOCK",
        auto_eligible=False,
        default_guard="first_load_stock_memo",
        context_fields=("combo_key", "weight_dtype"),
        # Not wired in this release: a rescue-slab grant covers this bypass for
        # its load (consent_descriptor.KIND_RESCUE_SLAB), so the operator sees
        # one card and clicks once, not twice.
        wired=False,
    ),
    KindSpec(
        kind="waive-known-wrong:ring-pad",
        style="accuracy",
        # The guard applies at every Ring degree, including hybrids and odd-world
        # auto Ring. Keep the title degree-neutral; the memo binds exact topology.
        title="Render {artifact} with a padded sequence on a ring topology",
        risk=("This combination produces measured wrong math, so the output will be "
              "visibly incorrect."),
        evidence=("Output is stamped rendered-under-waiver and the waiver is recorded "
                  "in the gate ledger."),
        primary_label="Render under waiver (output stamped)",
        env_var="DGXM_WAIVE_KNOWN_WRONG",
        auto_eligible=False,
        default_guard="ring_pad",
        context_fields=("combo_key", "topology", "world"),
        # Wired: the adapter base's backstop and the H3-scoped guard both raise
        # this card.
        wired=True,
        measured="left-edge artifacts, measured 2026-07-10",
    ),
    KindSpec(
        kind="waive-known-wrong:sol-attn",
        style="accuracy",
        # The title names the kernel rather than a tau: the tau is part of the
        # kernel name the operator picked, and one card per tau would be a card
        # per widget value for the same measured wrongness.
        title="Render {artifact} with the sol-attn sparse kernel",
        risk=("This kernel is approximate by construction and is not identity "
              "preserving: the same seed gives a different composition than it "
              "does under exact attention."),
        evidence=("Output is stamped rendered-under-waiver and the waiver is recorded "
                  "in the gate ledger."),
        primary_label="Render under waiver (output stamped)",
        env_var="DGXM_WAIVE_KNOWN_WRONG",
        auto_eligible=False,
        default_guard="sol_attn",
        context_fields=("combo_key", "topology", "world"),
        # Wired: the kernel wrapper raises this card on every render that has
        # no grant. Scoped to the resolved topology and world because routing
        # density depends on the shape each rank sees.
        wired=True,
        measured="a different composition, measured 2026-08-15",
    ),
    KindSpec(
        kind="waive-known-wrong:pixeldit-sp",
        style="accuracy",
        title="Render {artifact} with PixelDiT sequence parallel",
        # One sentence, no competing number: `measured` below carries the
        # figures, and the permanent WAIVER row's `reason` is this sentence
        # alone, so a rounded copy here would understate ring2 on the audit.
        risk=("This topology diverges from the certified dp2/single reference by more "
              "than the fidelity floor, so the image will not match."),
        evidence=("Output is stamped rendered-under-waiver and the waiver is recorded "
                  "in the gate ledger."),
        primary_label="Render under waiver (output stamped)",
        env_var="DGXM_WAIVE_KNOWN_WRONG",
        auto_eligible=False,
        # The ledger's guard prefix for this kind, not a `refusal.GUARDS` id:
        # `gate_audit_vocab.check_guard` validates a class-K guard as
        # `<prefix>` or `<prefix>:<scope>`, so a family-scoped guard needs no
        # vocabulary bump.
        default_guard="sp_unvalidated",
        context_fields=("combo_key", "topology", "world"),
        # Kept for protocol-v9 ledger compatibility. The PixelDiT guard retired
        # after exact-gather passed world-2 hardware parity.
        wired=False,
        measured=("1-step NRMS 0.265 on uly2 and 0.338 on ring2 against a certified "
                  "dp2/single reference, floor 0.10, measured 2026-07-13"),
    ),
    KindSpec(
        kind="waive-known-wrong:shard-quant",
        style="accuracy",
        # The title names the checkpoint's quantization rather than a topology:
        # the same wrongness reads on uly2 and on cfg2, and the card would name
        # a topology the operator is not rendering.
        title="Render {artifact} with sharded nvfp4 activations",
        risk=("On a sharded topology this checkpoint's nvfp4 activations diverge "
              "from a one-GPU render by more than the fidelity floor, whether or "
              "not every layer quantizes against one shared scale."),
        evidence=("Even with the shared activation scale on and exact, a waived "
                  "render diverges by the measured amount. It is stamped "
                  "rendered-under-waiver and recorded in the gate ledger. The grant "
                  "stays live until you revoke it in the panel, so every later render "
                  "of this checkpoint on this topology and world takes it and is "
                  "stamped too."),
        primary_label="Render under waiver (output stamped)",
        env_var="DGXM_WAIVE_KNOWN_WRONG",
        auto_eligible=False,
        default_guard="shard_quant_scale",
        context_fields=("combo_key", "topology", "world"),
        # Wired: the sample path raises this card once per dispatch when a family
        # measured past the floor renders nvfp4 on a sharded topology.
        wired=True,
        measured=("1-step NRMS 0.120 on uly2 and 0.129 on cfg2 with the shared scale "
                  "on, 0.131 and 0.134 without it, against a bf16 baseline of 0.032, "
                  "floor 0.10, measured 2026-09-05"),
    ),
)

KIND_SPECS: Mapping[str, KindSpec] = MappingProxyType({spec.kind: spec for spec in _SPECS})


def consent_context(kind: str, **fields: Any) -> dict[str, Any]:
    """The narrow memo context for a kind, validated field by field."""
    spec = KIND_SPECS.get(kind)
    if spec is None:
        raise ValueError(f"unknown consent kind {kind!r}")
    missing = [name for name in spec.context_fields if name not in fields]
    extra = [name for name in fields if name not in spec.context_fields]
    if missing or extra:
        raise ValueError(
            f"consent context for {kind!r} needs exactly {list(spec.context_fields)}"
            f" (missing {missing}, unexpected {extra})")
    context = {name: fields[name] for name in spec.context_fields}
    consent_store.canonical_context(context)  # fail closed on non-JSON values
    return context
