"""Refusal taxonomy: the class every typed refusal declares, in its own text.

Four classes, defined in docs/DESIGN.md section 5.9:

* ``P`` physics: the model or ComfyUI itself cannot do it. No bypass exists,
  so the text must name the working alternative.
* ``C`` capacity-protective: a preflight. Never a bare refusal. It offers the
  fitting strategy first and refuses outright only after the supported alternatives cannot fit.
* ``U`` unproven correctness: vouching, ceremonies, unvalidated degrees.
  Always clearable by consent.
* ``K`` known-wrong math: measured wrongness. Refuse by default; a waiver may
  exist, and where it never will the text says so.

The class rides in the message, not on the exception object: a worker raise
reaches the driver as Monarch-wrapped text and only the text survives. One
helper builds every tagged message, so no site can drift.

Leaf module: standard library only. It must stay importable from the adapters,
the actor, the nodes and the driver endpoints without dragging a runtime in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class RefusalClass(StrEnum):
    """The four refusal classes. The value is what the tag carries."""

    PHYSICS = "P"
    CAPACITY = "C"
    UNPROVEN = "U"
    KNOWN_WRONG = "K"


CLASS_TITLES: Final[dict[str, str]] = {
    "P": "physics",
    "C": "capacity-protective",
    "U": "unproven correctness",
    "K": "known-wrong math",
}


@dataclass(frozen=True, slots=True)
class GuardSpec:
    """One waivable or waiver-tracked boundary.

    ``waivable_now`` is True only when a consent for this guard can be granted
    in this release. A guard whose waiver is designed but not wired stays
    False, so no refusal names a card that does not exist (``_validate``).
    """

    refusal_class: RefusalClass
    waivable_now: bool
    consent_kind: str
    note: str


# The frozen guard vocabulary (DESIGN.md section 5.9). Renaming a string after
# a ledger row carries it requires a protocol version change.
GUARDS: Final[dict[str, GuardSpec]] = {
    "stock_load_preflight": GuardSpec(
        RefusalClass.CAPACITY,
        True,
        "rescue-slab",
        "worker-side stock residency wall; a rescue consent loads slab instead",
    ),
    "driver_footprint_preflight": GuardSpec(
        RefusalClass.CAPACITY,
        False,
        "rescue-slab",
        "no card at the render site in this release: the driver stack is "
        "already spent there, so the loader site offers the rescue instead",
    ),
    "loader_footprint_preflight": GuardSpec(
        RefusalClass.CAPACITY,
        True,
        "rescue-slab",
        "the driver-stack projection at the loader site, where the rescue offer is raised",
    ),
    "activation_footprint_preflight": GuardSpec(
        RefusalClass.CAPACITY,
        False,
        "rescue-slab",
        "no card at any of its sites: the escape stays the documented env var "
        "DGXM_DISABLE_ACTIVATION_PREFLIGHT=1 (#47 for wan_scail, #56 for "
        "minimax_h3), which the render memory preflight under this guard reads too",
    ),
    "partial_load_divergence": GuardSpec(
        RefusalClass.CAPACITY,
        False,
        "rescue-slab",
        "cross-rank weight-residency agreement, read after ComfyUI has placed "
        "the weights: no card, because the ladder is spent by then and the "
        "loader site owns the rescue offer; none is designed, because a waiver "
        "would authorize ranks that compute different weights",
    ),
    "slab_load_preflight": GuardSpec(
        RefusalClass.CAPACITY,
        False,
        "rescue-slab",
        "worker-side slab residency wall: no card, because slab is the rescue "
        "itself and no grant makes unified memory appear",
    ),
    "cross_rank_capacity": GuardSpec(
        RefusalClass.CAPACITY,
        False,
        "rescue-slab",
        "cross-rank capacity agreement, read before any rank allocates: no card, "
        "because no consent makes a refusing rank fit and one refusing rank refuses "
        "the whole fleet; when no rank refuses, the rescue offer goes out under the "
        "waivable stock guard, priced at the leanest rank that slab would fit",
    ),
    "slab_lora_bake_preflight": GuardSpec(
        RefusalClass.CAPACITY,
        False,
        "rescue-slab",
        "the exact-patch backstop before a slab-resident LoRA bake: no card, "
        "because slab is already the residency and the ladder already admitted "
        "the load the bake runs on",
    ),
    "first_load_stock_memo": GuardSpec(
        RefusalClass.UNPROVEN,
        False,
        "waive-first-load-stock",
        "no card this release: a rescue grant subsumes it for that load",
    ),
    "slab_vouched_families": GuardSpec(
        RefusalClass.UNPROVEN,
        False,
        "waive-unvouched-slab",
        "no card this release: an explicit slab request still loads unvouched",
    ),
    "ring_pad": GuardSpec(
        RefusalClass.KNOWN_WRONG,
        True,
        "waive-known-wrong:ring-pad",
        "the adapter-base backstop for pad rows that ring or hybrid attention cannot "
        "exclude; the expert waiver is wired and every render under it is stamped",
    ),
    "ring_pad:minimax_h3": GuardSpec(
        RefusalClass.KNOWN_WRONG,
        True,
        "waive-known-wrong:ring-pad",
        "the family-scoped ring pad guard for MiniMax H3; it shares the "
        "backstop's card, and the permanent waiver row names this scope",
    ),
    "sol_attn": GuardSpec(
        RefusalClass.KNOWN_WRONG,
        True,
        "waive-known-wrong:sol-attn",
        "the unscoped spelling the card and the headless env var use; every "
        "raise site is family scoped",
    ),
    "sol_attn:minimax_h3": GuardSpec(
        RefusalClass.KNOWN_WRONG,
        True,
        "waive-known-wrong:sol-attn",
        "the sol-attn sparse kernel is not identity preserving; the waiver is "
        "wired and every render under it is stamped",
    ),
    "shard_quant_scale": GuardSpec(
        RefusalClass.KNOWN_WRONG,
        True,
        "waive-known-wrong:shard-quant",
        "the unscoped spelling the card and the headless env var use; every "
        "raise site is family scoped",
    ),
    "shard_quant_scale:chroma": GuardSpec(
        RefusalClass.KNOWN_WRONG,
        True,
        "waive-known-wrong:shard-quant",
        "a sharded nvfp4 render of a family measured past the fidelity floor; "
        "the waiver is wired and every render under it is stamped",
    ),
}

WAIVABLE_GUARDS: Final[frozenset[str]] = frozenset(
    name for name, spec in GUARDS.items() if spec.waivable_now
)


@dataclass(frozen=True, slots=True)
class PanelAction:
    """The one click that clears a refusal, and its headless equivalent."""

    label: str
    env: str
    env_value: str = "1"


@dataclass(frozen=True, slots=True)
class RefusalTag:
    """A parsed tag. ``guard`` is None when the site declared none."""

    refusal_class: RefusalClass
    guard: str | None
    waivable: bool


TAG_OPEN: Final = "[dgxm:"
_TAG_RE: Final = re.compile(
    r"\[dgxm:(?P<cls>[PCUK])"
    r"(?: guard=(?P<guard>[A-Za-z0-9_:.-]{1,64}))?"
    r"(?: waivable=(?P<waivable>[01]))?\]"
)
_MAX_TEXT_CHARS: Final = 4000


def _parsed_tag(match: re.Match[str]) -> RefusalTag | None:
    """Build a tag from one regex match without trusting inconsistent fields."""
    guard = match.group("guard")
    waivable_raw = match.group("waivable")
    if guard is None and waivable_raw is not None:
        return None
    if guard is not None and waivable_raw is None:
        return None
    try:
        parsed_class = RefusalClass(match.group("cls"))
    except ValueError:
        return None
    return RefusalTag(parsed_class, guard, waivable_raw == "1")


def escape_body_tags(text: str) -> str:
    """Escape tag-shaped text in a body so no parser reads it as a tag.

    Every match in ``text`` is body data, including one at byte zero; only the
    outer tag built by :func:`refusal` is authoritative, and interpolated
    artifact names and diagnostics may not write tags. The semicolon keeps the
    text recognizable and the same length, and only ``[dgxm:`` opens a tag.
    """
    return _TAG_RE.sub(
        lambda match: f"[dgxm;{match.group(0)[len(TAG_OPEN):]}", text)


def _panel_sentences(action: PanelAction, troubleshooting: int | None) -> str:
    tail = (
        f' Open the DGX Monarch panel and click "{action.label}" on the card '
        "for this model, then queue the render again."
        f" Headless: set {action.env}={action.env_value} on the driver and run again."
    )
    if troubleshooting is not None:
        tail = f"{tail} See docs/TROUBLESHOOTING.md #{troubleshooting}."
    return tail


def _validate(
    refusal_class: RefusalClass,
    guard: str | None,
    waivable: bool,
    panel_action: PanelAction | None,
) -> None:
    if guard is not None:
        spec = GUARDS.get(guard)
        if spec is None:
            raise ValueError(
                f"unknown refusal guard {guard!r}: add it to refusal.GUARDS "
                "before a site declares it"
            )
        if spec.refusal_class is not refusal_class:
            raise ValueError(
                f"guard {guard!r} is class {spec.refusal_class.value}, "
                f"but the site declared class {refusal_class.value}"
            )
        if waivable and not spec.waivable_now:
            raise ValueError(
                f"guard {guard!r} has no consent wired in this release "
                f"({spec.note}), so a site may not declare it waivable"
            )
    if waivable:
        if guard is None:
            raise ValueError("a waivable refusal must declare its guard")
        if panel_action is None:
            raise ValueError(
                "a waivable refusal must name the panel action that clears it"
            )
    elif panel_action is not None:
        raise ValueError(
            "a refusal that is not waivable must not name a panel action"
        )
    if refusal_class is RefusalClass.PHYSICS and guard is not None:
        raise ValueError("class P refusals have no guard: no bypass exists")
    if refusal_class in (RefusalClass.CAPACITY, RefusalClass.UNPROVEN) and guard is None:
        raise ValueError(
            f"class {refusal_class.value} refusals must declare a guard, "
            "even where the ladder is spent"
        )


def refusal(
    refusal_class: RefusalClass,
    text: str,
    *,
    guard: str | None = None,
    waivable: bool = False,
    panel_action: PanelAction | None = None,
    troubleshooting: int | None = None,
) -> str:
    """Return ``text`` carrying its class tag, and its panel sentence when waivable.

    Raises ``ValueError`` on empty or blank text and on text over 4,000
    characters. It also raises on an inconsistent declaration, rather than
    emit a message that misstates what the user can do.
    """
    if not text or not text.strip():
        raise ValueError("a refusal must say what happened")
    if len(text) > _MAX_TEXT_CHARS:
        raise ValueError(f"refusal text exceeds {_MAX_TEXT_CHARS} characters")
    body = escape_body_tags(text.strip())
    _validate(refusal_class, guard, waivable, panel_action)
    parts = [f"[dgxm:{refusal_class.value}"]
    if guard is not None:
        parts.append(f" guard={guard}")
        parts.append(f" waivable={1 if waivable else 0}")
    parts.append("] ")
    if panel_action is not None:
        body = f"{body}{_panel_sentences(panel_action, troubleshooting)}"
    elif troubleshooting is not None:
        body = f"{body} See docs/TROUBLESHOOTING.md #{troubleshooting}."
    return "".join(parts) + body


def parse_refusal_tag(text: str) -> RefusalTag | None:
    """Recover the tag from a message, however it was wrapped on the wire.

    Total: it returns None for anything it cannot read, and never raises. Two
    tags in one text also return None, not the first, so no class is ever
    chosen silently.
    """
    if not isinstance(text, str):
        return None
    matches = _TAG_RE.findall(text)
    if len(matches) != 1:
        return None
    match = _TAG_RE.search(text)
    if match is None:
        return None
    return _parsed_tag(match)


def parse_leading_refusal_tag(text: str) -> RefusalTag | None:
    """Recover only a canonical tag at the start of the text, followed by a space.

    Ownership decisions use this stricter parser: a tag-shaped artifact name or
    crash diagnostic inside an otherwise untyped exception cannot prove that a
    worker refused before collective or result packing. Diagnostic messages
    that Monarch or another exception wrapper prefixed on the wire use
    :func:`parse_refusal_tag`.
    """
    if not isinstance(text, str):
        return None
    match = _TAG_RE.match(text)
    if match is None or not text.startswith(" ", match.end()):
        return None
    return _parsed_tag(match)


def untagged(text: str) -> str:
    """The body of an already-classified refusal, ready to be tagged once.

    Re-tagging a tagged message escapes the inner tag and prints the escape to
    the operator, and storing one in a field another card interpolates puts a
    second tag mid-sentence. Whoever carries a refusal's words rather than its
    exception takes the tag off here.
    """
    if parse_leading_refusal_tag(text) is None:
        return text
    return text.split("] ", 1)[-1]


def is_waivable(text: str) -> bool:
    """True when the message declares a bypass a consent in this release can grant."""
    tag = parse_refusal_tag(text)
    return bool(tag and tag.waivable and tag.guard in WAIVABLE_GUARDS)
