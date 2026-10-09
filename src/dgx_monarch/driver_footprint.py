"""Driver-side UMA footprint estimation and capacity refusal.

At render time the driver stack is already reflected in ``MemAvailable``.
This module therefore charges only a pending DiT load. Explicit-topology
eager loads are checked earlier by ``nodes.loader_preflight``. The estimator
is pure, reports its terms and options, and refuses only a complete
over-budget result. See docs/VALIDATION.md and docs/TROUBLESHOOTING.md #52.
"""
from __future__ import annotations

import functools
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import capacity_fit, mesh_safety, residency_mode
from .capacity_floor import ABSOLUTE_HOST_FLOOR_BYTES
from .loader_options import weight_dtype_from_options
from .log import get_logger
from .refusal import RefusalClass, refusal

log = get_logger(__name__)

_GIB = 2 ** 30

# Separate escape hatch: disabling the activation estimator must not disable
# this driver-side check (docs/TROUBLESHOOTING.md #52).
DRIVER_PREFLIGHT_DISABLE_ENV = "DGXM_DISABLE_DRIVER_PREFLIGHT"

# Shared worker host floor; a larger operator ``uma_reserve_gb`` wins.
# A separate driver floor could admit a load that its local worker refuses.
_DEFAULT_RESERVE_BYTES = ABSOLUTE_HOST_FLOOR_BYTES

# What a stock load holds above the file while it places the weights (the host
# copy beside the device copy), shared with the worker's stock price in
# ``capacity_fit`` so both charge one placement once. Charged only when
# ``slab_weights`` is explicitly off: under auto and comfy-managed residency
# this module cannot prove which placement path ran.
_STOCK_PLACEMENT_RATIO = capacity_fit.STOCK_PLACEMENT_RATIO

# What this host commits while it builds one FSDP shard: 1.05 of the shard
# (docs/VALIDATION.md, the 2026-09-09 shard-build price). It sits below the
# worker's ``capacity_fit.fsdp_materialize_factor``, measured on 2026-08-26,
# before the copy-on-write fix (docs/VALIDATION.md, 2026-09-07), when the build's
# device copy broke the checkpoint mapping and a rank paid its shard twice.
# With the bounce copy a rank holds its shard once; the 0.05 covers allocator
# slack and the fp32 islands a rank replicates (0.1 and 0.045 percent of the
# file for H3 and LTX 2.5). A family near the admitted 5 percent island cap
# (Krea2's islands are 4.9 percent of its file) would sit about 3 percent of
# its shard above this estimate; the worker capacity check can refuse it.
# This assumes the bounce copy is deployed on every box that loads; without it
# a rank pays about twice its shard. The worker's walls keep the higher factor,
# so nothing here admits a 60 GiB-class load a worker would refuse.
FSDP_SHARD_BUILD_FACTOR = 1.05

# The one pinned bounce buffer a build allocates for the row copy and frees
# when it closes (``adapters/fsdp_shard_build.DeviceCopier``).
FSDP_BOUNCE_BUFFER_BYTES = 256 * 1024 * 1024


def fsdp_shard_fraction(world: int) -> float:
    """The share of the file one rank holds, never under this rig's world 2.

    No leg has measured a world past 2, and the clamp keeps a stale handle's
    world from charging under what the pair itself costs.
    """
    return max(1.0 / max(int(world), 1), 0.5)


def fsdp_shard_build_share(world: int) -> float:
    """The multiple of the file one rank commits: its share times the build factor."""
    return fsdp_shard_fraction(world) * FSDP_SHARD_BUILD_FACTOR


def fsdp_shard_build_bytes(size_bytes: int, world: int) -> int:
    """One rank's shard, what the build holds above it, and the bounce buffer."""
    return (int(max(size_bytes, 0) * fsdp_shard_build_share(world))
            + FSDP_BOUNCE_BUFFER_BYTES)


SMALLER_TE = "use a smaller text-encoder artifact"
PRUNED_DIT = "use a pruned or quantized DiT artifact"
FEWER_REFS = (
    "cut the number or size of reference blocks, the canvas, or the clip "
    "length (the reference node truncates every reference video to the "
    "generation frame count, so submitting fewer frames alone changes nothing)"
)
FREE_MEMORY = (
    "free host memory (drop caches, stop co-resident LLM servers, dgxm reap)"
)
SMALLER_CANVAS = (
    "shrink the canvas or the clip length: video rows are latent frames times "
    "(height//32) times (width//32), so both axes are linear"
)
SPLIT_SEQUENCE = (
    "run this on world 2 with a uly* or ring* preset: sequence parallel is the "
    "only degree that shards a row axis, and cfg and dp shard none of it"
)


class DriverFootprintCapacityError(mesh_safety.StockLoadCapacityError):
    """Capacity refusal recognized by the stock-load gate classification.

    Gate details are truncated to 200 characters, so refusal messages put the
    budget before the artifact name and leave the full options in the log.
    """


@dataclass(frozen=True, slots=True)
class DriverStackProfile:
    """One family's driver-side row. A new family is one DRIVER_STACK_FAMILIES
    entry, as in REF_POSE_TOKEN_FAMILIES; an unregistered family is unchecked."""

    family: str
    ref_extras_keys: tuple[str, ...]
    fitting_options: tuple[str, ...]
    dit_note: str = ""


DRIVER_STACK_FAMILIES: dict[str, DriverStackProfile] = {
    # H3 options omit multi-host splitting: Ulysses and Ring replicate the
    # weights. FSDP, which does not, has admitted the bf16 release through the
    # fp32-islands profile since 2026-08-26, but its one render (2026-09-09,
    # docs/VALIDATION.md) had no reference comparison, so it is not offered here.
    "minimax_h3": DriverStackProfile(
        family="minimax_h3",
        ref_extras_keys=("minimax_refs", "minimax_keyframes"),
        fitting_options=(SMALLER_TE, PRUNED_DIT, FEWER_REFS, FREE_MEMORY),
        dit_note="the pruned int8 and fp8 reference-to-video artifacts are "
                 "19.5 GiB each against bf16's 61.7",
    ),
    # LTX prices two token streams rather than reference blocks, so it declares
    # no ref_extras_keys: the row charge arrives from ltx25_activation through
    # the loader-site driver-stack window.
    "ltx": DriverStackProfile(
        family="ltx",
        ref_extras_keys=(),
        fitting_options=(SMALLER_CANVAS, SPLIT_SEQUENCE, PRUNED_DIT,
                         SMALLER_TE, FREE_MEMORY),
        dit_note="the int8-convrot 22B artifact is 20.0 GiB against bf16's "
                 "39.1, and its gemma4 text encoder is 14.3 GiB at int8 "
                 "against 24.5 at bf16",
    ),
}


@dataclass(frozen=True, slots=True)
class DriverFootprintEstimate:
    """Pending driver bytes, available budget, and already-encoded reference geometry."""

    weight_bytes: int = 0
    arena_bytes: int = 0
    ref_pixels: int = 0
    ref_blocks: int = 0
    ref_frames: int = 0
    mem_available: int = 0
    reserve: int = 0
    co_resident: bool = False
    # The multiple of the file this host commits: 1.0 for a whole-file
    # placement, one rank's share times the build factor for an FSDP shard
    # build. dit_ceiling divides by the same number, or it would offer an
    # artifact size that does not clear the budget.
    weight_share: float = 1.0
    fsdp_world: int = 0
    notes: Mapping[str, str] = field(default_factory=dict)

    @property
    def projected(self) -> int:
        return self.weight_bytes + self.arena_bytes

    @property
    def usable(self) -> int:
        return max(self.mem_available - self.reserve, 0)

    @property
    def fits(self) -> bool:
        return self.projected <= self.usable

    @property
    def shortfall(self) -> int:
        """Shortfall against the unclamped budget, including the reserve."""
        return max(self.projected + self.reserve - self.mem_available, 0)

    @property
    def dit_ceiling(self) -> int:
        """Largest DiT artifact that would clear this budget, honoring the
        arena multiplier when the arena term is charged and the shard share
        when this host commits only part of the file. A non-positive share
        falls back to the whole-file divisor, which never invents headroom.
        """
        share = self.weight_share if self.weight_share > 0 else 1.0
        head = self.usable
        if self.fsdp_world >= 2:
            # The bounce buffer does not scale with the artifact, so it comes
            # off the budget before the share divides what is left.
            head = max(head - FSDP_BOUNCE_BUFFER_BYTES, 0)
        if self.weight_bytes and self.arena_bytes:
            return int(head / (share * (1.0 + _STOCK_PLACEMENT_RATIO)))
        return int(head / share)


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


def estimate_driver_footprint(
    *,
    profile: DriverStackProfile,
    mem_available: int,
    weight_bytes: int = 0,
    weights_resident: bool = False,
    co_resident: bool = True,
    slab_weights: bool | str | None = None,
    ref_pixels: int = 0,
    ref_blocks: int = 0,
    ref_frames: int = 0,
    reserve: int = _DEFAULT_RESERVE_BYTES,
    fsdp_world: int = 0,
) -> DriverFootprintEstimate:
    """Return a pure, replayable estimate from caller-supplied evidence.

    ``co_resident`` is a flag, not a rank count. Multiple local ranks can make
    this estimate pass too readily, but cannot create a false refusal.

    ``fsdp_world`` is the world of a shard build this host will run; the caller
    owns the proof that the build is the placement (``nodes/loader_preflight``).
    At world 2 or more the weight term becomes the shard price and the legacy
    arena drops. Anything else keeps the whole-file price: a share applied to
    any other placement would admit a load the box cannot hold.
    """
    notes: dict[str, str] = {}
    charged_weights = 0
    file_bytes = max(weight_bytes, 0)
    sharded = co_resident and not weights_resident and int(fsdp_world) >= 2
    share = fsdp_shard_build_share(fsdp_world) if sharded else 1.0
    if not co_resident:
        notes["weights"] = "rank 0 is not on the driver host, not charged"
    elif weights_resident:
        notes["weights"] = "already loaded, inside this MemAvailable reading"
    elif sharded:
        charged_weights = fsdp_shard_build_bytes(
            file_bytes, fsdp_world)
        notes["weights"] = (
            f"co-resident rank 0, FSDP shard build: this rank holds "
            f"{fsdp_shard_fraction(fsdp_world):.2f} of the "
            f"{_gib(file_bytes)} GiB file at world {int(fsdp_world)}, charged "
            f"at {FSDP_SHARD_BUILD_FACTOR:.2f}x that shard plus a "
            f"{FSDP_BOUNCE_BUFFER_BYTES // 2 ** 20} MiB pinned bounce buffer "
            f"(this price assumes the bounce copy is deployed)")
    else:
        # Charge the file size, an upper bound over the one measured settled
        # cost of 0.87x (docs/VALIDATION.md, 2026-08-04): one leg is too few to
        # apply that ratio, and with no catchable OOM the high side is safer.
        charged_weights = file_bytes
        notes["weights"] = "co-resident rank 0, not yet loaded"
    arena = 0
    if charged_weights and sharded:
        notes["arena"] = residency_mode.arena_omitted_note(
            residency_mode.MODE_FSDP_SHARD)
    elif charged_weights and residency_mode.charges_legacy_arena(slab_weights):
        arena = int(_STOCK_PLACEMENT_RATIO * charged_weights)
        notes["arena"] = "stock load host copy while it places the weights, slab_weights is off"
    elif charged_weights:
        notes["arena"] = residency_mode.arena_omitted_note(slab_weights)

    blocks = ref_blocks or (1 if ref_pixels > 0 else 0)
    if blocks:
        # Encoded, never submitted: the node truncates every reference video
        # to the generation frame count, so reporting the submitted count
        # would invite the operator to cut frames and see no change.
        notes["ref"] = (f"{blocks} reference block(s), {ref_frames} encoded "
                        f"frame(s), {ref_pixels} encoded pixels")
    else:
        notes["ref"] = "no reference blocks in this request"

    return DriverFootprintEstimate(
        weight_bytes=charged_weights, arena_bytes=arena,
        ref_pixels=max(ref_pixels, 0), ref_blocks=blocks,
        ref_frames=max(ref_frames, 0), mem_available=max(mem_available, 0),
        reserve=max(reserve, 0), co_resident=bool(co_resident),
        weight_share=share, fsdp_world=int(fsdp_world) if sharded else 0,
        notes=notes)


def what_would_fit(estimate: DriverFootprintEstimate,
                   profile: DriverStackProfile) -> str:
    """Describe a reachable artifact or memory adjustment for this budget."""
    if estimate.dit_ceiling < _GIB // 10:   # anything that prints as 0.0 GiB
        return (f"What would fit here: no DiT artifact of any size clears a "
                f"{_gib(estimate.usable)} GiB budget, so this load needs "
                f"{_gib(estimate.shortfall)} GiB more free memory. ")
    dit_note = f" ({profile.dit_note})" if profile.dit_note else ""
    # A ceiling from a shard price holds only for a shard build: staged as a
    # whole-file load, an artifact that size refuses. The text names the
    # charged share (fraction times build factor), not the weights note's
    # shard fraction, so the two numbers never share a name.
    sharded = (f" charged at {estimate.weight_share:.2f} of the file across "
               f"world {estimate.fsdp_world}"
               if estimate.fsdp_world >= 2 else "")
    return (f"What would fit here: a DiT artifact of at most "
            f"{_gib(estimate.dit_ceiling)} GiB{sharded}{dit_note}, or "
            f"{_gib(estimate.shortfall)} GiB more free memory. ")


def render_refusal(estimate: DriverFootprintEstimate, unet_name: str,
                   profile: DriverStackProfile) -> str:
    """Build the complete capacity message, with budget first for Gate truncation."""
    n = estimate.notes
    return (
        f"driver-side footprint preflight: {_gib(estimate.projected)} GiB to "
        f"place on the driver host, {_gib(estimate.usable)} GiB usable "
        f"({_gib(estimate.mem_available)} GiB MemAvailable minus a {_gib(estimate.reserve)} GiB reserve, "
        f"{'your uma_reserve_gb' if estimate.reserve > _DEFAULT_RESERVE_BYTES else 'the driver floor'}). "
        f"Refuses {unet_name}. Charged: DiT weights {_gib(estimate.weight_bytes)} GiB "
        f"({n.get('weights', 'not charged')}); stock host copy "
        f"{_gib(estimate.arena_bytes)} GiB ({n.get('arena', 'not charged')}). "
        f"Already spent by the driver, and inside that MemAvailable reading: "
        f"its text encoder, its VAEs and this request's reference encodes "
        f"({n.get('ref', '')}). "
        f"{what_would_fit(estimate, profile)}"
        f"Options: {'; '.join(profile.fitting_options)}. "
        f"Set {DRIVER_PREFLIGHT_DISABLE_ENV}=1 to bypass this "
        "preflight (docs/TROUBLESHOOTING.md #52).")


_DISABLE_LOGGED = False


def preflight_disabled() -> bool:
    """Return the escape-hatch state and warn once per driver process."""
    global _DISABLE_LOGGED
    if not mesh_safety.env_enabled(DRIVER_PREFLIGHT_DISABLE_ENV):
        return False
    if not _DISABLE_LOGGED:
        _DISABLE_LOGGED = True
        log.warning(
            "driver footprint preflight DISABLED by %s; driver-side capacity "
            "refusals are off for this driver process",
            DRIVER_PREFLIGHT_DISABLE_ENV)
    return True


def driver_footprint_preflight(estimate: DriverFootprintEstimate, *,
                               unet_name: str,
                               profile: DriverStackProfile) -> None:
    """Raise only on a successfully computed, over-budget estimate. The env
    escape and integrated-device guard are re-checked here so no caller can
    reach the raise around them."""
    if estimate.fits or preflight_disabled() or not mesh_safety.gpu_is_integrated():
        return
    # Class C, non-waivable here: the stack is already allocated, so
    # slab prices the same. The rescue card belongs to the loader site (GUARDS).
    raise DriverFootprintCapacityError(refusal(
        RefusalClass.CAPACITY, render_refusal(estimate, unet_name, profile),
        guard="driver_footprint_preflight", waivable=False))


def dtype_cast_requested(options: Mapping[str, Any] | None) -> bool:
    """Whether a cast makes file size an unsafe refusing estimate."""
    try:
        return weight_dtype_from_options(options) != "default"
    except (TypeError, ValueError):
        return True


def weights_owned_by_activation_preflight(family: str) -> bool:
    """Whether activation_footprint_preflight already charges the weight term:
    a family in both registries is charged once, not twice
    (tests/test_driver_footprint_preflight.py guards the double charge)."""
    return family in mesh_safety.REF_POSE_TOKEN_FAMILIES


def _address_is_local(address: str) -> bool:
    """Whether routing to a literal address selects that same local address.

    The connectionless UDP probe sends no packet. DNS and hostname comparison
    cannot reliably identify a separately named fabric interface.
    """
    try:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.connect((address, 9))
            return str(probe.getsockname()[0]) == address
    except (OSError, ValueError, IndexError):
        return False


@functools.lru_cache(maxsize=32)
def host_is_local(name: str | None) -> bool:
    """Whether a configured host resolves to this box.

    Resolution is cached once per name. An unresolved name returns false so an
    uncertain locality result cannot invent a capacity charge.
    """
    if not isinstance(name, str) or not name.strip():
        return False
    host = name.strip()
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        if host in {socket.gethostname(), socket.getfqdn()}:
            return True
        addresses = {str(info[4][0]) for info in socket.getaddrinfo(host, None)}
    except (OSError, UnicodeError, IndexError):
        return False
    return any(_address_is_local(address) for address in addresses)


def rank0_co_resident(mesh: Any) -> bool:
    """Does rank 0 share the driver host? Resolved from the handle with no
    probing beyond one cached local-address check: a this_host() local mesh
    (owns_hosts False) puts every rank in a child of the driver process, so
    co-residency is certain; a cluster puts rank 0 on config.hosts[0];
    anything unresolvable charges nothing."""
    handle = getattr(mesh, "handle", None)
    if handle is None:
        return False
    if getattr(handle, "owns_hosts", True) is False:
        return True
    hosts = getattr(getattr(handle, "config", None), "hosts", None) or ()
    if not hosts:
        return False
    return host_is_local(getattr(hosts[0], "name", None))


def reserve_bytes(worker_args: Mapping[str, Any] | None) -> int:
    """Apply the larger of the driver floor and operator ``uma_reserve_gb``."""
    try:
        requested = float((worker_args or {}).get("uma_reserve_gb") or 0.0)
    except (TypeError, ValueError, AttributeError):
        requested = 0.0
    return max(_DEFAULT_RESERVE_BYTES, int(requested * _GIB))


def file_size_bytes(path: object) -> int:
    try:
        return os.path.getsize(str(path)) if path else 0
    except OSError:
        return 0


def _entry_geometry(entry: Mapping[str, Any]) -> tuple[int, int, int]:
    """(frames, pixel_height, pixel_width) for one reference or keyframe
    block. Prefers the declared latent_h/latent_w, falls back to the encoded
    latent's trailing dims, and inverts comfy's video_latent_t for the frame
    count: video_latent_t(n) = 2 for n <= 5 else ((n - 5) // 17) * 5 + 2, and
    the node trims every reference video to n % 17 == 5, so it is exact."""
    shape = getattr(entry.get("latent"), "shape", None)
    h, w = entry.get("latent_h"), entry.get("latent_w")
    if not isinstance(h, int) or not isinstance(w, int):
        if shape is None or len(shape) < 2:
            return 0, 0, 0   # an audio-only reference block has no pixels
        h, w = int(shape[-2]), int(shape[-1])
    latent_t = entry.get("latent_t")
    if not isinstance(latent_t, int):
        return 1, int(h) * 16, int(w) * 16   # image and keyframe blocks
    return (5 if latent_t <= 2 else 17 * (latent_t - 2) // 5 + 5,
            int(h) * 16, int(w) * 16)


def ref_pixel_volume_from_latents(
    request: Mapping[str, Any] | None, profile: DriverStackProfile,
) -> tuple[int, int, int]:
    """Return encoded reference pixels, blocks, and frames from a packed request.

    ``pack_conditioning`` preserves shape through ``tree_to_cpu``. These bytes
    are already allocated, so report them without charging them again. Malformed
    extras contribute zero and do not abort the render.
    """
    positive = request.get("positive") if isinstance(request, Mapping) else None
    pixels = blocks = frames = 0
    for cond in positive if isinstance(positive, (list, tuple)) else ():
        extra = cond[1] if isinstance(cond, (list, tuple)) and len(cond) > 1 else None
        if not isinstance(extra, Mapping):
            continue
        for key in profile.ref_extras_keys:
            for entry in extra.get(key) or ():
                if not isinstance(entry, Mapping):
                    continue
                blocks += 1
                n, height, width = _entry_geometry(entry)
                frames += n
                pixels += n * height * width
    return pixels, blocks, frames
