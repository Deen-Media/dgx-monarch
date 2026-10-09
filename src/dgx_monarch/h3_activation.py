"""MiniMax H3 packed-sequence activation preflight.

H3 pricing counts video, audio, text, keyframe, and reference rows. Sequence
rows divide by sequence-parallel degree; replicated staging and nonresident DiT
weights do not. The total is compared with ``MemAvailable`` minus reserve before
worker RPC. The guard is non-waivable and refuses only on a complete over-budget
estimate; all incomplete estimates fail open.

Calibration lives in ``h3_calibration``. Keep this leaf free of runtime-heavy
dependencies.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from . import driver_footprint, h3_rows, mesh_safety
from .h3_calibration import (
    H3_GUIDE_CLIP_ENCODE_BYTES,
    H3_GUIDE_CLIP_FLOOR_FRAMES,
    H3_GUIDE_CLIP_LENGTH_BYTES,
    H3_GUIDE_LENGTH_MEASURED_PIXELS,
    H3_GUIDE_STILL_ENCODE_BYTES,
    H3_GUIDE_STILL_MEASURED_PIXELS,
    H3_REPLICATED_ROW_BYTES,
    H3_SHARDED_ROW_BYTES,
)
from .log import get_logger
from .refusal import RefusalClass, refusal

log = get_logger(__name__)

_GIB = 2 ** 30

H3_FAMILY = "minimax_h3"

# The generic activation guard's id. H3 counts its rows in h3_rows and prices them here.
_GUARD = "activation_footprint_preflight"

SMALLER_CANVAS = (
    "shrink the canvas or the clip length: packed rows grow linearly in latent "
    "frames and in canvas area, and at 1344x768 every latent frame is 1008 rows")
SPLIT_SEQUENCE = (
    "run this on world 2 with a uly* preset: sequence parallel is the only "
    "degree that shards the packed row axis for this family, and cfg and dp are "
    "refused for it")
FEWER_REF_ROWS = (
    "cut the number, resolution or length of reference blocks and keyframes: "
    "every reference row stays in the packed sequence for every sampling step")

# Ulysses shards activation rows but replicates weights.
H3_FITTING_OPTIONS = (SMALLER_CANVAS, SPLIT_SEQUENCE, FEWER_REF_ROWS,
                      driver_footprint.PRUNED_DIT, driver_footprint.FREE_MEMORY)

_RESIDENT_NOTE = "resident, credited"
_PENDING_NOTE = "not yet loaded"
CAST_NOTE = ("weight_dtype cast: file size cannot price cast weights, so they are "
             "not charged; activation bytes do not depend on the cast")


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


@dataclass(frozen=True, slots=True)
class H3ActivationEstimate:
    """Reproducible per-rank bytes and budget with explicit residency credit."""

    rows: h3_rows.H3Rows
    sp_degree: int = 1
    weight_bytes: int = 0        # 0 when residency is already credited
    mem_available: int = 0
    reserve: int = 0
    weights_note: str = ""

    @property
    def rows_per_rank(self) -> int:
        return self.rows.per_rank(self.sp_degree)

    @property
    def sharded_bytes(self) -> int:
        return self.rows_per_rank * H3_SHARDED_ROW_BYTES

    @property
    def replicated_bytes(self) -> int:
        return self.rows.total * H3_REPLICATED_ROW_BYTES

    @property
    def activation_bytes(self) -> int:
        return self.sharded_bytes + self.replicated_bytes

    @property
    def projected(self) -> int:
        return self.weight_bytes + self.activation_bytes

    @property
    def usable(self) -> int:
        return max(self.mem_available - self.reserve, 0)

    @property
    def fits(self) -> bool:
        return self.projected <= self.usable

    @property
    def shortfall(self) -> int:
        """Measure against the unclamped budget to preserve reserve shortfall."""
        return max(self.projected + self.reserve - self.mem_available, 0)

    @property
    def max_total_rows(self) -> int:
        """Largest reachable packed-row total that clears this budget.

        The closed form omits ``ceil(total/N)``. A bounded walk of at most
        ``N - 1`` rows corrects the floor-division result.
        """
        budget = self.usable - self.weight_bytes
        if budget <= 0:
            return 0
        degree = max(self.sp_degree, 1)
        rows = (budget * degree
                // (H3_SHARDED_ROW_BYTES + H3_REPLICATED_ROW_BYTES * degree))
        while rows > 0 and (-(-rows // degree) * H3_SHARDED_ROW_BYTES
                            + rows * H3_REPLICATED_ROW_BYTES) > budget:
            rows -= 1
        return rows


def estimate_h3_activation(
    *,
    rows: h3_rows.H3Rows,
    sp_degree: int,
    weight_bytes: int,
    weights_resident: bool,
    mem_available: int,
    reserve: int,
    weights_note: str = "",
) -> H3ActivationEstimate:
    """Estimate from explicit values without I/O; note omitted weight charges."""
    charged = 0 if weights_resident else max(int(weight_bytes), 0)
    note = weights_note or (_RESIDENT_NOTE if weights_resident else _PENDING_NOTE)
    return H3ActivationEstimate(
        rows=rows, sp_degree=max(int(sp_degree), 1), weight_bytes=charged,
        mem_available=max(int(mem_available), 0), reserve=max(int(reserve), 0),
        weights_note=note)


def h3_refusal(estimate: H3ActivationEstimate, unet_name: str) -> str:
    """Build a class C refusal with budget, charges, remedies, and bypass.

    Put budget values first so 200-character gate diagnostics retain them.
    """
    rows = estimate.rows
    floor = driver_footprint.reserve_bytes(None)
    reserve_source = ("your uma_reserve_gb" if estimate.reserve > floor
                      else "the driver floor")
    # When the weight charge, the reserve, or both leave no room for one row,
    # no length fits: offer no zero-row target.
    fit = (f"a packed total of at most {estimate.max_total_rows} rows at this "
           f"topology, against this render's {rows.total}"
           if estimate.max_total_rows else
           "no packed total of any length at this topology (the DiT weight "
           "charge and the reserve leave no room for one row)")
    return (
        f"minimax_h3 activation preflight: {_gib(estimate.projected)} GiB "
        f"needed per rank, {_gib(estimate.usable)} GiB usable "
        f"({_gib(estimate.mem_available)} GiB MemAvailable minus a "
        f"{_gib(estimate.reserve)} GiB reserve, {reserve_source}). "
        f"Refuses {unet_name}. Charged: DiT weights "
        f"{_gib(estimate.weight_bytes)} GiB ({estimate.weights_note}); packed "
        f"activations {_gib(estimate.activation_bytes)} GiB for {rows.total} "
        f"rows [{rows.stream_summary()}] at {estimate.rows_per_rank} rows/rank "
        f"over {estimate.sp_degree} rank(s), of which "
        f"{_gib(estimate.sharded_bytes)} GiB shards with the sequence and "
        f"{_gib(estimate.replicated_bytes)} GiB is replicated staging that no "
        f"topology divides. What would fit here: {fit}, or "
        f"{_gib(estimate.shortfall)} GiB more free "
        f"memory. Options: {'; '.join(H3_FITTING_OPTIONS)}. "
        f"Set {mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV}=1 to bypass this "
        "preflight (docs/TROUBLESHOOTING.md #56).")


_DISABLE_LOGGED = False


def preflight_disabled() -> bool:
    """Return the shared override and log the disabled boundary once."""
    global _DISABLE_LOGGED
    if not mesh_safety.env_enabled(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV):
        return False
    if not _DISABLE_LOGGED:
        _DISABLE_LOGGED = True
        log.warning(
            "minimax_h3 activation preflight DISABLED by %s; packed-sequence "
            "capacity refusals are off for this driver process",
            mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV)
    return True


def h3_activation_preflight(estimate: H3ActivationEstimate, *,
                            unet_name: str) -> None:
    """Raise only for a complete over-budget estimate on an active UMA guard.

    This class C refusal is non-waivable because residency cannot reduce a
    shape's activation bytes.
    """
    if estimate.fits or preflight_disabled() or not mesh_safety.gpu_is_integrated():
        return
    raise mesh_safety.StockLoadCapacityError(refusal(
        RefusalClass.CAPACITY, h3_refusal(estimate, unet_name),
        guard=_GUARD, waivable=False))


def sp_degree_for_mesh(mesh: Any) -> int:
    """Return the sequence-parallel degree, conservatively defaulting to one.

    H3 refuses DP and CFG parallelism, so dispatchable ``ulysses * ring`` equals
    world. Unreadable state returns one to preserve the larger charge.
    """
    try:
        world = getattr(mesh, "world", None)
        if isinstance(world, bool) or not isinstance(world, int) or world < 1:
            world = getattr(getattr(mesh, "handle", None), "world", 1)
        if isinstance(world, bool) or not isinstance(world, int) or world < 1:
            return 1
        return world
    except Exception:
        return 1


def _co_resident_enough(mesh: Any) -> bool:
    """Treat unknown rank-0 placement as local to avoid undercharging."""
    if getattr(mesh, "handle", None) is None:
        return True
    return driver_footprint.rank0_co_resident(mesh)


def preflight_h3_activation_for_request(
    model: Any, request: Any, latent: Any, *, family: str, path: str,
    unet_name: str, memo_credit: bool, latent_shape: Any = None,
) -> None:
    """The render-site entry point: refuse this dispatch before any worker RPC.

    Sample ``memo_credit`` before driver-footprint delegation, which may record
    an in-flight load. Otherwise first render could receive false residency
    credit. Only a completed over-budget estimate fails closed.
    ``latent_shape`` is the grid the workers will sample.
    """
    try:
        if family != H3_FAMILY:
            return                       # no-op for every other family
        if preflight_disabled() or not mesh_safety.gpu_is_integrated():
            return
        mesh = getattr(model, "mesh", None)
        if not _co_resident_enough(mesh):
            return
        avail = mesh_safety.mem_available_bytes()
        if avail is None:
            return  # off Linux: host MemAvailable bounds nothing here
        rows = h3_rows.rows_from_request(request, latent, video_shape=latent_shape)
        if not rows.resolved:
            return
        # Explicit presets load weights before this reading.
        resident = memo_credit or str(
            getattr(mesh, "topology_preset", "auto") or "auto") != "auto"
        weight_bytes = driver_footprint.file_size_bytes(path)
        note = ""
        if driver_footprint.dtype_cast_requested(getattr(model, "options", None)):
            # Cast invalidates file-based weight pricing, not activation pricing.
            resident, weight_bytes, note = True, 0, CAST_NOTE
        # Price the workers' effective reserve; a failed merge keeps the live args.
        args = dict(getattr(mesh, "worker_args", None) or {})
        merge = getattr(getattr(mesh, "handle", None),
                        "effective_worker_args", None)
        try:
            args = dict(merge(args)) if merge is not None else args
        except Exception:  # advisory merge only
            pass
        estimate = estimate_h3_activation(
            rows=rows, sp_degree=sp_degree_for_mesh(mesh),
            weight_bytes=weight_bytes, weights_resident=resident,
            mem_available=avail, reserve=driver_footprint.reserve_bytes(args),
            weights_note=note)
        h3_activation_preflight(estimate, unet_name=unet_name)
    except mesh_safety.StockLoadCapacityError:
        raise
    except Exception:
        return  # fail open: a broken estimator must never block a render


GUIDE_FRAMES_ENV = "DGXM_H3_GUIDE_FRAMES"

GUIDE_UNKNOWN_LENGTH = (
    f"length unknown on a link, priced as a clip; set {GUIDE_FRAMES_ENV} to "
    "the frame count to price a still instead (docs/TROUBLESHOOTING.md #82)")


def declared_guide_frames() -> int:
    """Return the declared guide length, or zero when unset.

    A linked batch has no statically readable length. The declaration supplies
    that term while the remaining capacity checks, including canvas cost, stay
    active. It applies process-wide, so a driver that also queues clips must
    leave it unset (docs/TROUBLESHOOTING.md #82).
    """
    raw = os.environ.get(GUIDE_FRAMES_ENV, "").strip()
    return int(raw) if raw.isdigit() else 0


def guide_encode_bytes(canvas_pixels: int, frames: int = 0) -> int:
    """Price one guide encode at this canvas from the measured probe points.

    ``frames`` of zero means the length is unknown, which is the reachable
    case. An unknown length is priced as a clip: comfy's tooltip on that input
    advertises clips, so a clip is what an operator can wire.
    """
    if canvas_pixels <= 0:
        return 0
    if 0 < frames < H3_GUIDE_CLIP_FLOOR_FRAMES:
        return max(H3_GUIDE_STILL_ENCODE_BYTES,
                   H3_GUIDE_STILL_ENCODE_BYTES * canvas_pixels
                   // H3_GUIDE_STILL_MEASURED_PIXELS)
    charged = H3_GUIDE_CLIP_ENCODE_BYTES[-1][1]
    for pixels, cost in H3_GUIDE_CLIP_ENCODE_BYTES:
        if canvas_pixels <= pixels:
            charged = cost
            break
    if canvas_pixels > H3_GUIDE_LENGTH_MEASURED_PIXELS:
        return charged           # the frame sweep does not reach up here
    for length, cost in H3_GUIDE_CLIP_LENGTH_BYTES:
        if frames and frames <= length:
            return max(charged, cost)
    # No declaration, or one past the longest measured clip: an unread link can
    # carry any legal length, so it costs the longest length a run measured.
    return max(charged, H3_GUIDE_CLIP_LENGTH_BYTES[-1][1])


def graph_guide_encode_bytes(prompt: Any, guides: tuple[tuple[Any, int], ...],
                             *, family: str) -> tuple[int, str]:
    """Return the bytes a graph's anchored guides encode before the sampler.

    Each guide arrives as the node whose canvas its latent link names and the
    frame count its source file proved, zero when nothing proved one. Evidence
    outranks the operator's declaration, which outranks the worst case, so a
    file that proves a clip is charged as a clip whatever the environment
    claims. An unresolvable canvas falls back to the largest in the graph.

    Guides encode one at a time and keep only a frame grid each, so the peak is
    one encode however many are chained. This is a driver-host transient: no
    topology divides it, unlike the packed rows the same guide later adds.
    Never raises, like every other loader-site charge helper.
    """
    try:
        if family != H3_FAMILY or not guides:
            return 0, "no guide image linked in this graph"
        if driver_footprint.preflight_disabled():
            return 0, f"{driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV} is set"
        widest = h3_rows.canvas_pixels_from_prompt_graph(prompt)
        declared = declared_guide_frames()
        charged, evidence, pixels = 0, "", 0
        for node, proven in guides:
            canvas = h3_rows.canvas_pixels_of_node(prompt, node) or widest
            if canvas <= 0:
                continue
            frames = proven or declared
            cost = guide_encode_bytes(canvas, frames)
            if cost > charged:
                charged, pixels = cost, canvas
                evidence = (
                    f"{proven} frame(s) proved from the named file" if proven
                    else f"{declared} frame(s) declared by {GUIDE_FRAMES_ENV}"
                    if declared else GUIDE_UNKNOWN_LENGTH)
        if not pixels:
            return 0, "no MiniMax H3 canvas node in this graph"
        return charged, (f"{len(guides)} guide image link(s) at {pixels} "
                         f"canvas pixels, {evidence}, probe of 2026-08-14")
    except Exception:
        return 0, "guide geometry unreadable; encode not charged"


def graph_activation_bytes(prompt: Any, sp_degree: int, *,
                           family: str) -> tuple[int, str]:
    """Return graph-declared bytes for the loader's settled memory window.

    Both guard overrides apply. This helper never raises, because propagation
    would discard the complete loader-site footprint refusal.
    """
    try:
        if family != H3_FAMILY:
            return 0, "not a packed-sequence family"
        if preflight_disabled():
            return 0, f"{mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV} is set"
        if driver_footprint.preflight_disabled():
            return 0, f"{driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV} is set"
        degree = int(sp_degree)
        if degree < 1:
            return 0, "no sequence-parallel degree resolved"
        rows = h3_rows.rows_from_prompt_graph(prompt)
        if not rows.resolved:
            return 0, "no MiniMax H3 canvas node in this graph"
        return (rows.per_rank(degree) * H3_SHARDED_ROW_BYTES
                + rows.total * H3_REPLICATED_ROW_BYTES,
                f"{rows.total} declared rows over {degree} rank(s); video and "
                "audio rows only, keyframe and reference rows are not visible "
                "in the graph")
    except Exception:
        return 0, "row extraction failed; activations not charged"
