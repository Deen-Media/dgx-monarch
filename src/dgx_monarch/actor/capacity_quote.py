"""Price one rank's load without allocating or raising a capacity refusal.

``price`` returns a ``RankPrice`` for two callers. ``store_residency.resolve``
uses it to choose a residency or raise locally while preserving ledger rows.
The worker's ``capacity_quote`` endpoint returns one row per slot before any
rank allocates; the driver owns fleet-wide refusal so it can name the rank
instead of receiving an actor traceback.

This module owns the shared rung vocabulary and pricing helpers. It imports
no torch, ComfyUI, or ``store_residency`` code, takes no GPU lock, and performs
no mutation. The endpoint supplies the ComfyUI-derived ``pinned_staging`` input.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .. import capacity_fit, capacity_lora_bake, capacity_memory, compile_policy, mesh_safety, residency_mode
from ..capacity_fit import (
    SlabFit,
    StockFit,
    managed_required_bytes,
    preflight_wall_measured,
)
from ..log import get_logger
from . import store_family
from .rescue_offer import COLD_MEMO_CLAUSE, build_descriptor, rescue_blocker

log = get_logger(__name__)

# Only ``refuse`` and ``rescue`` stop a load. ``unpriced`` and ``no_claim`` admit
# and are recorded, so a fitting rank and one that made no claim never read alike.
VERDICT_FITS = "fits"
VERDICT_REFUSE = "refuse"
VERDICT_RESCUE = "rescue"
VERDICT_UNPRICED = "unpriced"          # reserved for a rung in UNPRICED_RUNGS
VERDICT_NO_CLAIM = "no_claim"          # the probe could not run: no claim either way
VERDICTS = frozenset({
    VERDICT_FITS, VERDICT_REFUSE, VERDICT_RESCUE, VERDICT_UNPRICED, VERDICT_NO_CLAIM,
})

# What the load would do to the slot. A reuse places nothing and admits; swap
# and unknown are charged the file, because the quote prices as held and never
# models the drop: a stock resident's arena is not returned by an in-process
# unload and a slab resident barely moves MemAvailable. ``unknown`` reads as
# ``fresh``: a quote can land between a slot's drop and its repopulation.
WOULD_LOAD_FRESH = "fresh"
WOULD_LOAD_REUSE = "reuse"
WOULD_LOAD_SWAP = "swap"
WOULD_LOAD_UNKNOWN = "unknown"
WOULD_LOAD = frozenset({
    WOULD_LOAD_FRESH, WOULD_LOAD_REUSE, WOULD_LOAD_SWAP, WOULD_LOAD_UNKNOWN,
})

# The name a reuse block answers under, and neither probe's: naming one of
# those is the inverse's cue to rebuild an applying fit from a probe that never
# ran. It stays outside the frozen ``gate_audit_vocab.MEASURED_PROBES``, because
# a reuse admits and its block reaches no refusal and no audit row.
PROBE_REUSE = "worker_reuse"

SLOT_COND = "cond"
SLOT_UNCOND = "uncond"
SLOTS = (SLOT_COND, SLOT_UNCOND)

# Reported for a slot whose load is in flight or whose cleanup failed. Never a
# residency a decision can report: it says the box held what the store cannot.
RESIDENCY_IN_FLIGHT = "in_flight"

# The ladder's rungs. One home: ``store_residency`` imports them back.
RUNG_COMFY_MANAGED = "comfy_managed"
RUNG_EXPLICIT = "explicit"
RUNG_EXPLICIT_STOCK = "explicit_stock"
RUNG_VOUCHED_AUTO = "vouched_auto"
RUNG_STOCK_FITS = "stock_fits"
RUNG_CONSENTED_RESCUE = "consented_rescue"
RUNGS = frozenset({
    RUNG_COMFY_MANAGED, RUNG_EXPLICIT, RUNG_EXPLICIT_STOCK,
    RUNG_VOUCHED_AUTO, RUNG_STOCK_FITS, RUNG_CONSENTED_RESCUE,
})

# Rungs that return without a priced verdict. Empty: the rung coverage tests in
# tests/test_capacity_quote.py fail on any member. A member added with those
# tests needs a comment naming what the machine cannot price there.
UNPRICED_RUNGS: frozenset[str] = frozenset()


def _no_residents() -> dict[str, str | None]:
    return {SLOT_COND: None, SLOT_UNCOND: None}


@dataclass(frozen=True, slots=True)
class RankPrice:
    """One rank's slot price against a single snapshot.

    Fields describe only this host; the driver composes fleet-wide messages.
    ``measured`` holds stock, managed, or FSDP evidence under the relevant probe
    name. ``slab_measured`` holds slab evidence. Rescue rows carry both because
    both probes ran. ``priced`` is false for reuse, which performs no load.

    ``reason`` describes the selected rung. For a ``rescue`` verdict it instead
    holds the cold-memo clause, or is empty, so the UI can distinguish missing
    family evidence from insufficient capacity.
    """

    verdict: str
    rung: str
    use_slab: bool = False
    priced: bool = False
    measured: dict[str, Any] | None = None
    slab_measured: dict[str, Any] | None = None
    blocker: tuple[str, str] | None = None
    descriptor: Any = None
    would_load: str = WOULD_LOAD_UNKNOWN
    holds_resident: dict[str, str | None] = field(default_factory=_no_residents)
    checkpoint_bytes: int = 0
    slot: str = SLOT_COND
    shmem_bytes: int | None = None
    retained_failed_load_slabs: int = 0
    failed_load_cleanup_pending: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown capacity verdict {self.verdict!r}")
        if self.would_load not in WOULD_LOAD:
            raise ValueError(f"unknown load transition {self.would_load!r}")

    def refuses(self) -> bool:
        """Whether this rank's own answer stops the load."""
        return self.verdict == VERDICT_REFUSE

    def to_row(self) -> dict[str, Any]:
        """The wire row: JSON-native, descriptor as its public mapping."""
        descriptor = self.descriptor
        if descriptor is not None and not isinstance(descriptor, dict):
            descriptor = descriptor.public()
        return {
            "verdict": self.verdict,
            "rung": self.rung,
            "use_slab": bool(self.use_slab),
            "priced": bool(self.priced),
            "measured": None if self.measured is None else dict(self.measured),
            "slab_measured": (None if self.slab_measured is None
                              else dict(self.slab_measured)),
            "blocker": None if self.blocker is None else list(self.blocker),
            "descriptor": descriptor,
            "would_load": self.would_load,
            "holds_resident": dict(self.holds_resident),
            "checkpoint_bytes": int(self.checkpoint_bytes),
            "slot": self.slot,
            "shmem_bytes": (None if self.shmem_bytes is None else int(self.shmem_bytes)),
            "retained_failed_load_slabs": int(self.retained_failed_load_slabs),
            "failed_load_cleanup_pending": bool(self.failed_load_cleanup_pending),
            "reason": self.reason,
        }

    @classmethod
    def from_row(cls, row: Any) -> RankPrice:
        """Parse one wire row, or raise ``ValueError``.

        An unparseable row refuses like a silent rank, so a partial row must
        never resolve into a price that admits.
        """
        if not isinstance(row, dict):
            raise ValueError(f"a quote row must be a mapping, got {type(row).__name__}")
        missing = sorted(ROW_KEYS - set(row))
        if missing:
            raise ValueError(f"quote row is missing {missing}")
        blocker = row["blocker"]
        if blocker is not None:
            blocker = tuple(blocker)
            if len(blocker) != 2:
                raise ValueError("a blocker is a pair of why and what would fit")
        holds = row["holds_resident"]
        if not isinstance(holds, dict):
            raise ValueError("holds_resident is a mapping of slot to residency")
        return cls(
            verdict=row["verdict"],
            rung=row["rung"],
            use_slab=bool(row["use_slab"]),
            priced=bool(row["priced"]),
            measured=row["measured"],
            slab_measured=row["slab_measured"],
            blocker=blocker,
            descriptor=row["descriptor"],
            would_load=row["would_load"],
            holds_resident=dict(holds),
            checkpoint_bytes=int(row["checkpoint_bytes"]),
            slot=row["slot"],
            shmem_bytes=row["shmem_bytes"],
            retained_failed_load_slabs=int(row["retained_failed_load_slabs"]),
            failed_load_cleanup_pending=bool(row["failed_load_cleanup_pending"]),
            reason=row["reason"],
        )


ROW_KEYS: frozenset[str] = frozenset({
    "verdict", "rung", "use_slab", "priced", "measured", "slab_measured",
    "blocker", "descriptor", "would_load", "holds_resident", "checkpoint_bytes",
    "slot", "shmem_bytes", "retained_failed_load_slabs",
    "failed_load_cleanup_pending", "reason",
})


def comfy_managed_active() -> bool:
    """Whether this worker brought ComfyUI's own DynamicVRAM up, from the
    environment ``comfy_dynamic.stage_b`` exports it through."""
    return residency_mode.active()


def compile_dit_active() -> bool:
    """Whether this worker process has compiled DiT enabled."""
    return os.environ.get("DGXM_COMPILE_DIT") == "1"


def default_resolve_lora_path(name: str) -> str:
    """This host's path for a LoRA named as a loader widget shows it.

    Local import: neither this module nor ``capacity_fit`` may import comfy.
    """
    from . import comfy_bridge

    return comfy_bridge.resolve_model_path("loras", name)


def _charge(fit: Any, charged: int) -> Any:
    """Re-price a slot after subtracting earlier slots' requirements.

    Two 30 GiB checkpoints can each fit a 75 GiB budget when priced alone but
    exceed it together. Use ``capacity_fit.stock_required_bytes`` for each
    stock load, including its placement overhead.
    """
    if charged <= 0 or not fit.applies or fit.avail_bytes is None:
        return fit
    left = fit.avail_bytes - charged
    return replace(fit, avail_bytes=left, fits=fit.required_bytes <= left)


def _resident_word(entry: Any) -> str | None:
    return None if not isinstance(entry, dict) else entry.get("residency")


def _holds_this_checkpoint(entry: Any, request_key: str) -> bool:
    """Whether the slot's resident is the key this load would ask for.

    ``ModelStore.snapshot`` publishes the request key as a repr and ``ensure``
    reuses on the whole key, so a match on the checkpoint name alone is a dtype
    or LoRA change: a swap, which is charged, not a reuse, which places nothing.
    A caller that states no key gets the charged reading.
    """
    key = entry.get("request_key") if isinstance(entry, dict) else None
    return bool(request_key) and isinstance(key, str) and key == request_key


def _slot_context(snapshot: dict, slot: str, request_key: str) -> dict[str, Any]:
    """The half of a row that describes the box rather than the price."""
    holds = {name: _resident_word(snapshot.get(name)) for name in SLOTS}
    entry = snapshot.get(slot)
    failed = set(snapshot.get("cleanup_failed_slots") or ())
    if slot in failed:
        # A load is in flight or its cleanup failed: the store says empty while
        # the box sits at the load high water. Read it as unknown, price fresh.
        holds[slot] = RESIDENCY_IN_FLIGHT
        would = WOULD_LOAD_UNKNOWN
    elif entry is None:
        would = WOULD_LOAD_FRESH
    elif _holds_this_checkpoint(entry, request_key):
        would = WOULD_LOAD_REUSE
    else:
        would = WOULD_LOAD_SWAP
    meminfo = capacity_memory.read_meminfo()
    return {
        "slot": slot,
        "would_load": would,
        "holds_resident": holds,
        "shmem_bytes": None if meminfo is None else meminfo.get("Shmem"),
        "retained_failed_load_slabs": int(snapshot.get("retained_failed_load_slabs") or 0),
        "failed_load_cleanup_pending": bool(snapshot.get("failed_load_cleanup_pending")),
    }


def _managed(path: str, model_options: dict, pinned_staging: bool | None,
             charged: int, size: int, context: dict[str, Any]) -> RankPrice:
    """Price rung zero, which the ladder returns unpriced and its wall prices after.

    A quote that skipped it would invent the managed verdict. The one ComfyUI
    flag it needs is read by the endpoint and passed in; without it this rank
    makes no claim rather than guessing which staging path comfy will take.
    """
    # One sentence for the rung, whatever the verdict: the ladder returns this
    # decision unchanged, so the word it logs must not move. ``no_claim`` beside
    # it is what says the price was not taken.
    common = dict(context, rung=RUNG_COMFY_MANAGED, checkpoint_bytes=size,
                  reason="this worker runs comfy-managed residency (ComfyUI's DynamicVRAM)")
    if (pinned_staging is None or model_options.get("dtype") is not None
            or not mesh_safety.gpu_is_integrated()):
        return RankPrice(VERDICT_NO_CLAIM, **common)
    avail = mesh_safety.mem_available_bytes()
    if avail is None:
        return RankPrice(VERDICT_NO_CLAIM, **common)
    avail -= charged
    required = managed_required_bytes(size, pinned_staging)
    fits = required <= avail
    if not fits:
        # The rung's sentence says which residency this rank runs, which is as
        # true of a load that fits. A refusal owes the operator the shortfall.
        common["reason"] += _shortfall(required, avail)
    return RankPrice(VERDICT_FITS if fits else VERDICT_REFUSE, priced=True, **common,
                     measured=preflight_wall_measured(size, avail, required, fits))


def _shortfall(required: int, avail: int) -> str:
    """What a wall-priced refusal adds to its rung's own sentence."""
    return (f", and this load needs {capacity_fit.gib(required)} GiB against "
            f"{capacity_fit.gib(avail)} GiB of available unified memory, "
            f"{capacity_fit.gib(required - avail)} GiB short")


def _slab_rung(rung: str, reason: str, slab_fit: SlabFit, size: int,
               context: dict[str, Any]) -> RankPrice:
    """One of the three rungs that return slab, priced by ``slab_load_fit``."""
    if not slab_fit.applies:
        # The rung's sentence is the row's reason on every rung; a probe that
        # made no claim says so in ``priced`` and reports why to the journal,
        # rather than rewriting the ladder's decision text.
        log.debug("the slab probe made no claim on the %s rung: %s", rung,
                  slab_fit.skipped_reason)
        return RankPrice(VERDICT_NO_CLAIM, rung, use_slab=True,
                         checkpoint_bytes=size, reason=reason, **context)
    return RankPrice(
        VERDICT_FITS if slab_fit.fits else VERDICT_REFUSE, rung, use_slab=True,
        priced=True, slab_measured=slab_fit.measured(), checkpoint_bytes=size,
        reason=reason, **context)


def _reuse_measured(size: int, charged: int) -> dict[str, Any] | None:
    """Report available headroom for a reuse without claiming a load probe ran.

    Reuse returns before both probes, so ``priced`` and ``applies`` stay false
    and ``probe`` names neither real probe. The UI still needs ``headroom_gib``
    to display and sort the rank correctly. Keys match ``StockFit.measured``;
    ``required_bytes`` is zero because the weights are already resident, and
    ``worker_capacity._charged_bytes`` charges subsequent slots nothing.

    The probe name prevents ``store_residency`` from reconstructing a load fit
    from this display-only row.
    """
    avail = mesh_safety.mem_available_bytes()
    if avail is None:
        return None                     # no reading, so no claim in either half
    avail = max(int(avail) - charged, 0)
    return {"probe": PROBE_REUSE,
            "checkpoint_bytes": int(size), "mem_available_bytes": avail,
            "required_bytes": 0, "headroom_bytes": avail,
            "weights_gib": capacity_fit.gib(size), "required_gib": 0.0,
            "mem_available_gib": capacity_fit.gib(avail),
            "headroom_gib": capacity_fit.gib(avail), "fits": True,
            "applies": False}


def price(
    *,
    path: str,
    unet_name: str,
    model_options: dict,
    slab_weights: bool | str,
    slab_capable_path: bool,
    lora_low_rss: bool,
    fsdp_launch: bool,
    blocked_reason: str,
    world: int | None = None,
    authoritative_slab_retry: bool,
    memoized_family: Callable[[str], str | None],
    vouched_families: frozenset[str],
    rescue_consent: dict | None = None,
    fit_probe: Callable[[str, dict], StockFit] | None = None,
    file_identity: Callable[[str], str] | None = None,
    compile_dit: Callable[[], bool] | None = None,
    comfy_managed: Callable[[], bool] | None = None,
    resolve_lora_path: Callable[[str], str] | None = None,
    family_hint: str | None = None,
    family_override: str | None = None,
    node_options: dict | None = None,
    lora_stack: list | None = None,
    reserve_bytes: int = 0,
    slot: str = SLOT_COND,
    pinned_staging: bool | None = None,
    store_snapshot: dict | None = None,
    budget_bytes_already_charged: int = 0,
    request_key: str = "",
) -> RankPrice:
    """Price one slot's load on this rank and return the answer.

    Takes ``store_residency.resolve``'s arguments unchanged, plus five: the
    ``slot`` priced, the ``pinned_staging`` flag the endpoint reads from ComfyUI
    so this module need not import it, one ``store_snapshot`` every slot of a
    request is priced against, the bytes earlier slots already charged, and the
    ``request_key`` ``ModelStore.ensure`` would build, which the endpoint
    composes because this module may not import the store to compose it.

    It returns on every path, refusing ones included: the ladder raises locally,
    the driver for the fleet.
    """
    fit_probe = fit_probe or capacity_fit.stock_load_fit
    file_identity = file_identity or store_family.file_identity
    compile_dit = compile_dit or compile_dit_active
    comfy_managed = comfy_managed or comfy_managed_active
    resolve_lora_path = resolve_lora_path or default_resolve_lora_path
    snapshot = store_snapshot if isinstance(store_snapshot, dict) else {}
    charged = max(int(budget_bytes_already_charged or 0), 0)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    context = _slot_context(snapshot, slot, request_key)
    if context["would_load"] == WOULD_LOAD_REUSE and not authoritative_slab_retry:
        # ``ensure`` reuses this exact key above the ladder, so no load runs.
        # A pending slab retry instead reloads the file and is not a reuse.
        held = snapshot[slot]           # a reuse verdict proves it is a mapping
        rung = held.get("residency_rung")
        slab = held.get("residency") == residency_mode.MODE_SLAB
        block = _reuse_measured(size, charged)
        return RankPrice(
            VERDICT_FITS, rung if rung in RUNGS else RUNG_STOCK_FITS, priced=False,
            use_slab=slab, checkpoint_bytes=size,
            measured=None if slab else block,
            slab_measured=block if slab else None,
            reason="this slot already holds these bytes; no load would run",
            **context)

    auto = slab_weights == "auto"
    if family_override:
        # A forced family never inherits the memo's slab vouching.
        auto = False
        authoritative_slab_retry = False

    if comfy_managed():
        return _managed(path, model_options, pinned_staging, charged, size, context)

    slab_fit = _charge(
        capacity_fit.slab_load_fit(
            # Only the low-RSS bake needs this term.
            path, model_options, reserve_bytes=reserve_bytes, lora_stack=lora_stack if lora_low_rss else None,
            resolve_lora_path=resolve_lora_path, lora_credit_bytes=capacity_lora_bake.swap_credit_bytes(
                snapshot.get(slot), request_key, size, context, None if authoritative_slab_retry else slab_weights)),
        charged)

    # Only a detected eager no-op may retain slab; unknown models stay stock.
    memo = None
    compile_active = bool(compile_dit())
    if not family_override and ((auto and not authoritative_slab_retry)
                                or (compile_active and bool(slab_weights))):
        memo = memoized_family(path)
    compile_blocks_slab = compile_policy.compile_dit_blocks_slab(memo, compile_active)
    compile_block_reason = (
        "compile_dit is on and this model is not known to be a family that compile leaves eager; "
        "compiled fp8 GEMM templates reject slab-resident weights"
    )

    if slab_weights is True and slab_capable_path and not compile_blocks_slab:
        return _slab_rung(RUNG_EXPLICIT, "slab_weights=on was requested",
                          slab_fit, size, context)

    cold_memo = ""
    if auto and slab_capable_path:
        if (authoritative_slab_retry or memo in vouched_families) and not compile_blocks_slab:
            return _slab_rung(
                RUNG_VOUCHED_AUTO,
                "the first stock load of this file discovered a vouched family"
                if authoritative_slab_retry
                else "the family memo names a ledger-vouched family",
                slab_fit, size, context)
        if not memo:
            cold_memo = COLD_MEMO_CLAUSE

    if fsdp_launch:
        return _fsdp_rung(path, unet_name, model_options, world, size, charged, context)

    fit = _charge(fit_probe(path, model_options), charged)
    explicit_stock = slab_weights is False or compile_blocks_slab
    if not fit.applies or fit.fits:
        rung = RUNG_EXPLICIT_STOCK if explicit_stock else RUNG_STOCK_FITS
        return RankPrice(
            VERDICT_FITS if fit.applies else VERDICT_NO_CLAIM, rung,
            priced=fit.applies,
            measured=fit.measured() if fit.applies else None,
            checkpoint_bytes=size,
            reason=fit.skipped_reason or "stock residency fits this checkpoint",
            **context)

    blocker = rescue_blocker(
        slab_capable_path=slab_capable_path, lora_low_rss=lora_low_rss,
        fsdp_launch=fsdp_launch, compile_dit=compile_blocks_slab,
        explicit_stock=explicit_stock,
        blocked_reason=blocked_reason or (compile_block_reason if compile_blocks_slab else ""),
        slab_fit=slab_fit,
    )
    both = {"measured": fit.measured(), "checkpoint_bytes": size, "priced": True,
            "slab_measured": slab_fit.measured() if slab_fit.applies else None}
    if blocker is not None:
        return RankPrice(VERDICT_REFUSE, RUNG_STOCK_FITS, blocker=blocker,
                         reason=blocker[0], **both, **context)

    descriptor = build_descriptor(
        unet_name=unet_name, path=path, fit=fit, model_options=model_options,
        file_identity=file_identity, family_hint=family_hint,
        node_options=node_options, lora_stack=lora_stack,
    )
    if rescue_consent:
        # Driver-projected grants arrive as explicit slab policy on every rank.
        return RankPrice(
            VERDICT_FITS if slab_fit.fits or not slab_fit.applies else VERDICT_REFUSE,
            RUNG_CONSENTED_RESCUE, use_slab=True, descriptor=descriptor,
            reason="a consent memo authorizes slab residency here",
            **both, **context)
    return RankPrice(VERDICT_RESCUE, RUNG_STOCK_FITS, descriptor=descriptor,
                     reason=cold_memo, **both, **context)


def _fsdp_rung(path: str, unet_name: str, model_options: dict, world: int | None,
               size: int, charged: int, context: dict[str, Any]) -> RankPrice:
    """Delegate FSDP pricing without a module-import cycle."""
    from .capacity_quote_fsdp import fsdp_rung

    return fsdp_rung(path, unet_name, model_options, world, size, charged, context)
