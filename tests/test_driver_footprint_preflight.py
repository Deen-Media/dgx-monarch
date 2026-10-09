"""driver_footprint: the driver-side footprint preflight (docs/TROUBLESHOOTING.md #52).

CPU only: no tensors, no comfy, no CUDA. The pure estimator takes ints, so the
2026-08-04 window replays here from the real staged artifact sizes and the
recorded MemAvailable series.

Provenance for the incident constants: the driver host's MemAvailable series for
2026-08-04, the H3 candidate-leg logs from that day, and `ls --block-size=1`
on the staged artifact set (docs/VALIDATION.md, "Driver-side footprint window,
2026-08-04"). The four reference-to-video legs are the acceptance surface: one
froze the host, three completed, and the estimator has to separate them.

Every test here runs a shipped call shape. At render time the driver stack is
already inside MemAvailable, so this site charges only a pending DiT load. One
test pins the gap that leaves: the recorded incident ran an explicit topology
preset, whose eager load this site cannot see; nodes/loader_preflight.py
prices that load.
"""
from __future__ import annotations

import ast
import contextlib
import socket
import sys
import types
from pathlib import Path

import pytest

from dgx_monarch import capacity_fit, capacity_floor, driver_footprint, mesh_safety
from dgx_monarch.nodes import gate_cross_mode, render_preflight

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"

_GIB = 2 ** 30

# Staged artifact sizes, byte exact. 5.2 GB and 0.6 GB are the decimal
# readings of the two VAEs; in GiB the pair is 5.41, not 5.8.
DIT_REF2VA_BF16 = 66_280_487_368          # 61.73 GiB, the leg that froze the box
DIT_REF2VA_INT8 = 34_038_894_550          # 31.70 GiB, ran clean, dipped to 0.7 GiB free
DIT_REF2VA_INT8_PRUNED = 20_970_379_616   # 19.53 GiB, ran clean
DIT_REF2VA_FP8_PRUNED = 20_958_205_608    # 19.52 GiB, ran clean

# MemAvailable from the recorded series (GiB), 2026-08-04.
MEM_0722_PEAK = int(114.5 * _GIB)         # 07:17:25, before the eager load
MEM_0722_SETTLED = int(60.9 * _GIB)       # 07:18:40, after that load settled
MEM_LEG5 = int(114.7 * _GIB)
MEM_LEG9_FP8 = int(115.1 * _GIB)
MEM_LEG11 = int(114.3 * _GIB)
MEM_WORLD2 = int(106.0 * _GIB)
# Below the 5 GiB reserve. Reachable on the render path for the same reason
# the check exists: dispatch happens after the driver stack is spent.
MEM_BELOW_RESERVE = int(3.0 * _GIB)

# The driver's own stack for a Config A reference-to-video render, measured as
# a lower bound: after the eager load settled at 60.9 GiB free, the driver
# spent all of it over the next 110 seconds and the host died, so the stack is
# at least that big. Under the `auto` preset the order reverses: the driver
# stack is spent first and the weight load is still pending at dispatch, so
# dispatch sees at most this much free. That is the reading the render path
# would have refused on, and it is derived from the curve, not recorded at a
# dispatch that never happened.
DRIVER_STACK_SPENT = MEM_0722_SETTLED
MEM_AUTO_DISPATCH = MEM_0722_PEAK - DRIVER_STACK_SPENT   # 53.6 GiB

# The Config A reference set, as the node encoded it: one reference image at
# 512x320 and one reference video whose 48 submitted frames are truncated to
# the 5-frame generation count before the VAE sees them.
# 6 encoded frames x 512 x 320 = 983,040 pixels, not 8,028,160.
REF_PIXELS_CONFIG_A = (1 + 5) * 512 * 320
REF_BLOCKS_CONFIG_A = 2
REF_FRAMES_CONFIG_A = 6

CONFIG_A_REFS = [
    {"kind": "image", "latent_h": 20, "latent_w": 32},
    {"kind": "video", "latent_t": 2, "latent_h": 20, "latent_w": 32},
]

# 65 characters. `unet_name` is comfy's relative path under diffusion_models,
# so a subfoldered, quant-suffixed artifact name is the ordinary case and the
# only length worth pinning the 200-character Gate truncation against.
LONG_UNET_NAME = (
    "MiniMax-H3/minimax_h3_ref2va_bf16_convrot_int8_pruned.safetensors")

H3 = driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"]


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Every cache and once-only log flag is process state; reset it so test
    order can never decide a verdict. gpu_is_integrated is forced on here
    because the probe returns False on any GPU-less CI runner, and a test that
    expects a refusal would otherwise pass locally and fail there."""
    driver_footprint.host_is_local.cache_clear()
    monkeypatch.setattr(driver_footprint, "_DISABLE_LOGGED", False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV,
                       raising=False)
    yield
    driver_footprint.host_is_local.cache_clear()


def _estimate(dit: int, mem: int, *, refs: bool = True, co_resident: bool = True,
              slab_weights=None, weights_resident: bool = False,
              reserve: int | None = None):
    """Exactly the argument shape nodes/render_preflight.py passes.

    The reserve defaults to the shipped floor, not a literal, so a change to
    that constant is re-proved against every leg that ran clean.
    """
    if reserve is None:
        reserve = driver_footprint.reserve_bytes(None)
    return driver_footprint.estimate_driver_footprint(
        profile=H3,
        mem_available=mem,
        weight_bytes=dit,
        weights_resident=weights_resident,
        co_resident=co_resident,
        slab_weights=slab_weights,
        ref_pixels=REF_PIXELS_CONFIG_A if refs else 0,
        ref_blocks=REF_BLOCKS_CONFIG_A if refs else 0,
        ref_frames=REF_FRAMES_CONFIG_A if refs else 0,
        reserve=reserve)


# Acceptance: what this boundary refuses, what it must never refuse, and the
# shape it cannot see.

def test_the_recorded_incident_shape_is_not_intercepted_here(monkeypatch, tmp_path):
    """A known gap, pinned so no edit can claim this site closes it.

    The 2026-08-04 07:22 leg (`leg10_bf16_ref2va_ref` in the record) ran an
    explicit topology preset (`local mesh (1 x 1)` in its log), so the loader
    node loaded the 61.73 GiB checkpoint eagerly at 07:17:40 while 114.5 GiB
    was free, and the host froze at 07:20:30 during the driver's own
    text-encoder and reference work, before the sampler ran. By the time the
    render path sees a request the weights are already inside the MemAvailable
    reading, so this site reports them and charges nothing.
    nodes/loader_preflight.py prices the eager load.
    """
    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                            mem=MEM_0722_SETTLED, preset="single")
    seen: list = []
    monkeypatch.setattr(driver_footprint, "driver_footprint_preflight",
                        lambda estimate, **_kw: seen.append(estimate))
    render_preflight.activation_footprint_preflight_for_request(
        model, {"positive": [[object(), {"minimax_refs": CONFIG_A_REFS}]]},
        latent)
    if len(seen) != 1:
        pytest.fail(f"the driver preflight must run exactly once, ran {len(seen)}")
    if seen[0].projected != 0 or not seen[0].fits:
        pytest.fail("an eagerly loaded checkpoint must be reported, not charged")
    if "already loaded" not in seen[0].notes["weights"]:
        pytest.fail(f"the credit must be stated: {seen[0].notes['weights']}")


def test_incident_counterfactual_under_auto_refuses(monkeypatch, tmp_path):
    """The same configuration under the `auto` preset, which is the default.

    Auto defers the load to the first render, so the driver stack is spent
    first and the 61.73 GiB load is still pending when the render path runs.
    Dispatch then sees at most 53.6 GiB (the curve's own arithmetic, see
    MEM_AUTO_DISPATCH) and the load is refused before any worker RPC.
    """
    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                            mem=MEM_AUTO_DISPATCH)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as excinfo:
        render_preflight.activation_footprint_preflight_for_request(
            model, {"positive": [[object(), {"minimax_refs": CONFIG_A_REFS}]]},
            latent)
    message = str(excinfo.value)
    for token in ("driver-side footprint preflight:",
                  "Refuses minimax_h3_ref2va_bf16.safetensors",
                  "61.7", "53.6", "48.6", "5.0",
                  str(REF_PIXELS_CONFIG_A), "6 encoded frame(s)",
                  driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV,
                  "docs/TROUBLESHOOTING.md #52"):
        if token not in message:
            pytest.fail(f"refusal must name {token!r}: {message}")


@pytest.mark.parametrize(("name", "dit"), [
    ("leg5_cand_ref2va (pruned int8)", DIT_REF2VA_INT8_PRUNED),
    ("leg9_fp8_ref2va_cand (pruned fp8)", DIT_REF2VA_FP8_PRUNED),
    ("leg11_int8np_ref2va_cand (int8)", DIT_REF2VA_INT8),
])
@pytest.mark.parametrize("mem", [MEM_LEG5, MEM_LEG9_FP8, MEM_LEG11,
                                 MEM_AUTO_DISPATCH])
def test_clean_reference_legs_still_pass(monkeypatch, tmp_path, name, dit, mem):
    """The three reference-to-video legs that completed cleanly on the same
    box on the same day, with the identical Config A reference set, at every
    recorded MemAvailable peak and at the counterfactual dispatch reading that
    refuses the bf16 artifact.

    These tests keep a later edit from turning this preflight into a refusal
    of runs that fit. The margin is thin: the 31.70 GiB leg dipped to 0.7 GiB
    of MemAvailable, so this is a go/no-go classifier on the DiT artifact
    axis, not a model of true demand. An estimator calibrated to true peak
    demand would refuse all three.
    """
    model, latent = _h3_rig(monkeypatch, tmp_path, size=dit, mem=mem)
    render_preflight.activation_footprint_preflight_for_request(
        model, {"positive": [[object(), {"minimax_refs": CONFIG_A_REFS}]]},
        latent)   # no raise


def test_config_b_t2va_uly2_world2_still_passes():
    """Config B, text-to-video under uly2 at world 2, both artifacts. Rank 0
    sits on the head, which is the driver box, so the weight term is charged
    and must still fit."""
    for dit in (DIT_REF2VA_BF16, DIT_REF2VA_INT8_PRUNED):
        estimate = _estimate(dit, MEM_WORLD2, refs=False)
        if not estimate.fits:
            pytest.fail(f"a world-2 leg that ran clean was refused: "
                        f"{estimate.projected / _GIB:.2f} GiB")


@pytest.mark.parametrize("dit", [DIT_REF2VA_BF16, DIT_REF2VA_INT8,
                                 DIT_REF2VA_INT8_PRUNED, DIT_REF2VA_FP8_PRUNED])
def test_the_boundary_is_the_artifact_plus_the_reserve(dit):
    """Written as arithmetic so a constant edit fails here with the reason, not
    inside one of the leg tests. With the driver stack already spent, only the
    artifact is left to place, so the verdict flips exactly at file size plus
    reserve."""
    floor = driver_footprint.reserve_bytes(None)
    if not _estimate(dit, dit + floor).fits:
        pytest.fail("exactly enough room must pass")
    if _estimate(dit, dit + floor - 1).fits:
        pytest.fail("one byte short of the reserve must refuse")


def test_remote_rank0_is_charged_no_weight_term():
    """Negative control for the co-residency rule: when rank 0 does not share
    the driver host its weights are not on this pool, so charging them would
    invent a refusal out of a guess."""
    estimate = _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH, co_resident=False)
    if estimate.weight_bytes != 0:
        pytest.fail("a remote rank 0 must contribute no weight term")
    if "not on the driver host" not in estimate.notes["weights"]:
        pytest.fail(f"the omission must be stated: {estimate.notes['weights']}")


def test_env_escape_bypasses_and_logs_once(monkeypatch):
    """A false positive from a coarse estimator must never block a run that
    fits, and the bypassed state must be greppable."""
    records: list[tuple[str, tuple]] = []
    monkeypatch.setattr(driver_footprint, "log", types.SimpleNamespace(
        warning=lambda msg, *args: records.append((msg, args))))
    monkeypatch.setenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, "1")
    estimate = _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH)
    if estimate.fits:
        pytest.fail("this fixture must be over budget for the test to mean anything")
    for _ in range(3):
        driver_footprint.driver_footprint_preflight(
            estimate, unet_name="x.safetensors", profile=H3)  # no raise
    if len(records) != 1:
        pytest.fail(f"expected exactly one WARNING, got {len(records)}")
    if driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV not in records[0][1]:
        pytest.fail(f"the WARNING must name the variable: {records[0]}")


# The kill switch honors only an explicit on value.

@pytest.mark.parametrize("value", ["0", "false", "off", "no", "", "  "])
def test_an_off_spelling_keeps_the_driver_preflight(monkeypatch, value):
    """The refusal text says set `=1`. `=0` must not disable the guard."""
    monkeypatch.setenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, value)
    estimate = _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH)
    if estimate.fits:
        pytest.fail("this fixture must be over budget for the test to mean anything")
    with pytest.raises(driver_footprint.DriverFootprintCapacityError):
        driver_footprint.driver_footprint_preflight(
            estimate, unet_name="x.safetensors", profile=H3)


@pytest.mark.parametrize("value", ["true", "ON", " yes "])
def test_an_on_spelling_takes_the_driver_preflight_bypass(monkeypatch, value):
    monkeypatch.setenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, value)
    estimate = _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH)
    driver_footprint.driver_footprint_preflight(
        estimate, unet_name="x.safetensors", profile=H3)  # no raise


def test_discrete_gpu_never_refuses(monkeypatch):
    """Off unified memory, host MemAvailable does not bound model capacity."""
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    estimate = _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH)
    driver_footprint.driver_footprint_preflight(
        estimate, unet_name="x.safetensors", profile=H3)  # no raise


def test_arena_omitted_when_slab_unknown_charged_when_off():
    """Three slab_weights readings, three arena terms. Under auto or unset the
    term is omitted and the omission is stated, because a false refusal is
    forbidden while a missed charge falls back to pricing the weights alone."""
    expected = {None: 0, True: 0, "auto": 0,
                False: int(capacity_fit.STOCK_PLACEMENT_RATIO * DIT_REF2VA_INT8_PRUNED)}
    for value, want in expected.items():
        estimate = _estimate(DIT_REF2VA_INT8_PRUNED, MEM_LEG5,
                             slab_weights=value)
        if estimate.arena_bytes != want:
            pytest.fail(f"slab_weights={value!r} charged "
                        f"{estimate.arena_bytes} arena bytes, expected {want}")
    unknown = _estimate(DIT_REF2VA_INT8_PRUNED, MEM_LEG5)
    if "omitted" not in unknown.notes.get("arena", ""):
        pytest.fail("an omitted term must carry its provenance note")


@pytest.mark.parametrize(("dit", "mem"), [
    (DIT_REF2VA_INT8_PRUNED, MEM_LEG5),
    (DIT_REF2VA_FP8_PRUNED, MEM_LEG9_FP8),
    (DIT_REF2VA_INT8, MEM_LEG11),
])
def test_clean_legs_pass_with_slab_weights_explicitly_off(dit, mem):
    """The arena term adds STOCK_PLACEMENT_RATIO times the file, and the gate
    ceremony, the cross-mode leg and the A/B harness all run with slab_weights
    off. Every leg that ran clean must still pass in that mode, or the term
    turns a diagnostic switch into a refusal."""
    estimate = _estimate(dit, mem, slab_weights=False)
    if not estimate.fits:
        pytest.fail(f"a clean leg was refused with slab_weights off: "
                    f"{estimate.projected / _GIB:.2f} GiB projected against "
                    f"{estimate.usable / _GIB:.2f} GiB usable")


@pytest.mark.parametrize(("dit", "fits"), [
    (DIT_REF2VA_INT8_PRUNED, True),
    (DIT_REF2VA_FP8_PRUNED, True),
    (DIT_REF2VA_INT8, False),
])
def test_slab_off_at_the_auto_dispatch_reading(dit, fits):
    """slab_weights explicitly off at the `auto` dispatch reading. Both halves
    are default paths: `auto` is the default preset, and a persisted
    identity-gate FAIL writes slab_weights=False into the graph's worker_args
    before the session's first render, so this combination ships whether or
    not anyone chose it.

    The verdict is a split, not a blanket pass. At 53.6 GiB the arena term
    makes 31.70 GiB of weights cost 2.1 times the file, 66.57 GiB at the
    placement ratio of 1.1 in force since 2026-10-01, against a 48.60 GiB
    budget, and that is not a false refusal: with the driver stack already
    spent, a legacy load of that artifact needs more than the box has left.
    `leg11_int8np_ref2va_cand` completed with it because it ran an
    explicit preset and loaded the artifact before the driver stack existed.
    The refusal must still leave the operator a move, so the offered ceiling
    is checked against the artifacts the same message recommends."""
    estimate = _estimate(dit, MEM_AUTO_DISPATCH, slab_weights=False)
    if estimate.fits != fits:
        pytest.fail(
            f"slab off at the auto dispatch reading: expected fits={fits}, got "
            f"{estimate.fits} ({estimate.projected / _GIB:.2f} GiB projected "
            f"against {estimate.usable / _GIB:.2f} GiB usable)")
    if fits:
        return
    if estimate.dit_ceiling < DIT_REF2VA_INT8_PRUNED:
        pytest.fail(
            f"the refusal offers a {estimate.dit_ceiling / _GIB:.2f} GiB "
            "ceiling that excludes the artifacts its own message recommends")


# The FSDP shard-build price (2026-09-09). A rank under a `*+fsdp` preset
# holds one shard, and the streaming build assigns file-backed rows rather
# than copying the file, so the whole file plus a legacy arena is the price of
# a load this placement never runs.

# The M6 A leg's reading, boundary 4 (2026-09-09, docs/VALIDATION.md): a
# 61.7 GiB bf16 family at uly2+fsdp, world 2, on a head with 92.9 GiB
# MemAvailable.
MEM_M6_A = int(92.9 * _GIB)
SHARD_BF16 = driver_footprint.fsdp_shard_build_bytes(DIT_REF2VA_BF16, 2)


def _sharded(dit: int, mem: int, *, world: int = 2, slab_weights=None,
             weights_resident: bool = False, co_resident: bool = True):
    return driver_footprint.estimate_driver_footprint(
        profile=H3, mem_available=mem, weight_bytes=dit,
        weights_resident=weights_resident, co_resident=co_resident,
        slab_weights=slab_weights, fsdp_world=world,
        reserve=driver_footprint.reserve_bytes(None))


def test_the_shard_price_is_one_ranks_shard_the_build_and_the_bounce_buffer():
    """The constant, read back through the arithmetic that charges it.

    The band is the M6 A leg's acceptance: the leg proceeds to the shard build
    when the loader prices it near 31 to 34 GiB. The whole-file price it
    refused at on 2026-09-09 was 114.2 GiB. A factor that drifted either way
    would leave that band.
    """
    assert SHARD_BF16 == (int(DIT_REF2VA_BF16 * 0.5
                              * driver_footprint.FSDP_SHARD_BUILD_FACTOR)
                          + driver_footprint.FSDP_BOUNCE_BUFFER_BYTES)
    assert 31.0 <= SHARD_BF16 / _GIB <= 34.0


def test_a_shard_build_charges_the_share_and_no_arena():
    """Slab is not a lever under FSDP (worker_env.slab_mode_effective), so
    every slab value takes the one price and none of them adds an arena."""
    for slab in (None, False, True, "auto"):
        estimate = _sharded(DIT_REF2VA_BF16, MEM_M6_A, slab_weights=slab)
        assert estimate.weight_bytes == SHARD_BF16, slab
        assert estimate.arena_bytes == 0, slab
        assert "FSDP" in estimate.notes["arena"], slab


def test_the_weights_note_names_the_share_and_the_world():
    estimate = _sharded(DIT_REF2VA_BF16, MEM_M6_A)
    note = estimate.notes["weights"]
    assert "0.50 of the 61.7 GiB file at world 2" in note
    assert "1.05x that shard" in note and "256 MiB pinned bounce buffer" in note


def test_the_m6_a_leg_admits_at_the_shard_price_and_refuses_at_the_whole_file():
    """The leg the boundary 4 record refused: the whole file at the stock
    placement price against 87.9 GiB usable. One rank's shard fits the same
    box."""
    whole_file = _estimate(DIT_REF2VA_BF16, MEM_M6_A, refs=False,
                           slab_weights=False)
    assert not whole_file.fits
    expected_gib = round(
        DIT_REF2VA_BF16 * (1.0 + capacity_fit.STOCK_PLACEMENT_RATIO) / _GIB, 1)
    assert round(whole_file.projected / _GIB, 1) == expected_gib
    assert _sharded(DIT_REF2VA_BF16, MEM_M6_A).fits


@pytest.mark.parametrize("world", [0, 1, True])
def test_a_world_below_two_keeps_the_whole_file_price(world):
    """Worlds 0 and 1 shard nothing, and bool is an int: True must not read as
    a world. Each keeps the whole-file price."""
    estimate = _sharded(DIT_REF2VA_BF16, MEM_M6_A, world=world,
                        slab_weights=False)
    assert estimate.weight_bytes == DIT_REF2VA_BF16
    assert estimate.arena_bytes == int(
        capacity_fit.STOCK_PLACEMENT_RATIO * DIT_REF2VA_BF16)
    assert estimate.weight_share == 1.0 and estimate.fsdp_world == 0


def test_a_shard_already_built_is_not_charged_a_second_time():
    estimate = _sharded(DIT_REF2VA_BF16, MEM_M6_A, weights_resident=True)
    assert estimate.weight_bytes == 0 and estimate.fsdp_world == 0
    assert "already loaded" in estimate.notes["weights"]


def test_the_ceiling_offered_under_a_shard_price_is_a_whole_file_size():
    """The ceiling is what the operator would stage, which is a file, not a
    shard. Offering a shard-sized file would refuse on the next queue."""
    estimate = _sharded(DIT_REF2VA_BF16, int(50.0 * _GIB))
    at_the_ceiling = _sharded(estimate.dit_ceiling, int(50.0 * _GIB))
    assert at_the_ceiling.fits
    assert estimate.dit_ceiling > estimate.usable    # a shard of it, not it
    message = driver_footprint.what_would_fit(estimate, H3)
    # The charged share, named as such: the weights note's 0.50 is the shard
    # fraction inside it, and one message must not give two numbers one name.
    assert "charged at 0.53 of the file across world 2" in message
    assert "0.50 of the file" not in message


def test_refusal_names_every_fitting_option():
    """Class C forbids a bare refusal, so the message keeps every fitting
    option."""
    estimate = _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH)
    message = driver_footprint.render_refusal(estimate, "x.safetensors", H3)
    for option in H3.fitting_options:
        if option not in message:
            pytest.fail(f"missing fitting option {option!r}")
    for phrase in ("What would fit here", "GiB more free memory",
                   "a DiT artifact of at most", "Already spent by the driver",
                   "reference block(s)"):
        if phrase not in message:
            pytest.fail(f"missing {phrase!r} from a class C refusal")


@pytest.mark.parametrize("slab", [None, False])
@pytest.mark.parametrize("mem", [MEM_AUTO_DISPATCH, MEM_BELOW_RESERVE])
def test_every_number_the_refusal_offers_is_reachable(slab, mem):
    """Every offered remediation must be reachable. Each number in the message
    is checked against the estimator.

    MEM_BELOW_RESERVE covers dispatch after the driver stack is spent, when
    MemAvailable can sit under the reserve. There `usable` is 0 and no
    artifact ceiling exists, so printing one next to the family's 19.5 GiB
    artifacts would offer a fix that cannot work."""
    estimate = _estimate(DIT_REF2VA_BF16, mem, slab_weights=slab)
    message = driver_footprint.render_refusal(estimate, "x.safetensors", H3)
    if "at most 0.0 GiB" in message:
        pytest.fail(f"a zero-valued lever is not a lever: {message}")
    if estimate.dit_ceiling:
        at_the_ceiling = _estimate(estimate.dit_ceiling, mem, slab_weights=slab)
        if not at_the_ceiling.fits:
            pytest.fail(
                f"the offered DiT ceiling {estimate.dit_ceiling / _GIB:.2f} GiB "
                f"does not fit: {at_the_ceiling.projected / _GIB:.2f} GiB against "
                f"{at_the_ceiling.usable / _GIB:.2f} GiB usable")
    elif H3.dit_note in message:
        pytest.fail(f"no artifact fits here, so none may be offered: {message}")
    with_the_shortfall = _estimate(
        DIT_REF2VA_BF16, mem + estimate.shortfall, slab_weights=slab)
    if not with_the_shortfall.fits:
        pytest.fail("the offered shortfall must be the amount that fixes it")


def test_h3_options_do_not_offer_world_2_or_fsdp():
    """Do not offer two-host H3 as a capacity remedy. Ulysses replicates weights;
    the recorded BF16 FSDP render has no reference comparison (docs/VALIDATION.md).
    """
    message = driver_footprint.render_refusal(
        _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH), "x.safetensors", H3)
    for banned in ("both boxes", "world 2", "fsdp", "second Spark"):
        if banned.lower() in message.lower():
            pytest.fail(f"refusal offered {banned!r}, which provably refuses")


def test_error_is_recognized_as_capacity():
    """Subclassing keeps the ceremony's capacity classification working with
    no edit to mesh_safety."""
    estimate = _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH)
    try:
        driver_footprint.driver_footprint_preflight(
            estimate, unet_name="x.safetensors", profile=H3)
    except driver_footprint.DriverFootprintCapacityError as exc:
        if not mesh_safety.is_stock_load_capacity_error(exc):
            pytest.fail("the refusal must classify as a capacity error")
        if not isinstance(exc, mesh_safety.StockLoadCapacityError):
            pytest.fail("the refusal must remain a StockLoadCapacityError")
    else:
        pytest.fail("expected DriverFootprintCapacityError")


def test_a_gate_leg_reports_it_as_capacity_with_the_numbers_intact():
    """A side effect of the subclass: raised inside a first-use Gate leg, the
    refusal becomes a CAPACITY verdict whose detail is cut at 200 characters.
    The leading numbers must survive the cut and the full message must reach
    the log.

    LONG_UNET_NAME has the length of a real name (see its comment). A
    13-character stand-in would pass this assertion under a message layout
    that fails for every real name on this rig."""
    estimate = _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH)
    message = driver_footprint.render_refusal(estimate, LONG_UNET_NAME, H3)

    @contextlib.contextmanager
    def _policy(_handle, _original, _override):
        yield [{"slab_weights": False}]

    logged: list = []
    runtime = {
        "_temporary_worker_policy": _policy,
        "_model_with_worker_overrides": lambda model, _o, handle=None: model,
        "copy_transaction": dict,
        "is_artifact_binding_error": lambda _exc: False,
        "is_stock_load_capacity_error": mesh_safety.is_stock_load_capacity_error,
        "is_memory_exhaustion": lambda _exc: False,
        "log": types.SimpleNamespace(warning=lambda msg, *a: logged.append(a)),
    }

    def _refuse(*_args, **_kwargs):
        raise driver_footprint.DriverFootprintCapacityError(message)

    verdict, latent = gate_cross_mode.run_cross_residency_reference(
        runtime=runtime, ceremony_model=object(),
        handle=types.SimpleNamespace(call_all=lambda *_a, **_kw: None),
        original_worker_args={},
        slab_proof=types.SimpleNamespace(error=None, expected=True),
        frozen_request={}, frozen_latent={}, artifact_binding={},
        cfg_value=1.0, steps_hint=1, stock_latent={},
        transaction_render=_refuse, bind_request=lambda request, _b: request)
    if latent is not None or verdict is None or verdict["verdict"] != "CAPACITY":
        pytest.fail(f"a driver footprint refusal must land as CAPACITY: {verdict}")
    detail = verdict["detail"]
    if len(detail) != 200:
        pytest.fail(f"this test exists for the truncation; detail is {len(detail)}")
    if len(LONG_UNET_NAME) < 60:
        pytest.fail("this test is only worth running at a realistic name length")
    for token in ("driver-side footprint preflight:", "61.7", "48.6", "53.6", "5.0"):
        if token not in detail:
            pytest.fail(f"the truncated detail must still carry {token!r}: {detail}")
    if not logged:
        pytest.fail("the full message must reach the log before the cut")


def test_no_double_charge_with_activation_registry(monkeypatch):
    """activation_footprint_preflight owns the weight term for its registered
    families, so a family in both registries would be charged twice without a
    word."""
    if driver_footprint.weights_owned_by_activation_preflight("minimax_h3"):
        pytest.fail("minimax_h3 must stay out of REF_POSE_TOKEN_FAMILIES")
    monkeypatch.setitem(mesh_safety.REF_POSE_TOKEN_FAMILIES, "minimax_h3", 5376)
    if not driver_footprint.weights_owned_by_activation_preflight("minimax_h3"):
        pytest.fail("a family in both registries must yield the weight term")


def test_reserve_honors_a_larger_operator_reserve():
    """uma_reserve_gb is the operator's co-residency headroom, which the
    worker-side warning also reads: a larger value wins, and the 5 GiB floor
    still applies."""
    if driver_footprint.reserve_bytes({}) != capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES:
        pytest.fail("the default floor must be 5 GiB")
    if driver_footprint.reserve_bytes({"uma_reserve_gb": 2.0}) != 5 * _GIB:
        pytest.fail("a smaller operator value must not lower the floor")
    if driver_footprint.reserve_bytes({"uma_reserve_gb": 8.0}) != 8 * _GIB:
        pytest.fail("a larger operator value must win")
    if driver_footprint.reserve_bytes(None) != 5 * _GIB:
        pytest.fail("a missing worker_args must not raise")
    if driver_footprint.reserve_bytes({"uma_reserve_gb": "junk"}) != 5 * _GIB:
        pytest.fail("a junk operator value must degrade to the floor")


# Reference geometry: read from the encoded latents, where it is exact.

def test_ref_pixel_volume_from_encoded_latents():
    """The reference node writes latent_h/latent_w per block and a video's
    latent_t inverts back to its encoded frame count. latent_t=2 is the
    5-frame case, which is what Config A encoded."""
    request = {"positive": [[object(), {"minimax_refs": CONFIG_A_REFS}]]}
    pixels, blocks, frames = driver_footprint.ref_pixel_volume_from_latents(
        request, H3)
    if (pixels, blocks, frames) != (REF_PIXELS_CONFIG_A, 2, 6):
        pytest.fail(f"expected the Config A volume, got "
                    f"{(pixels, blocks, frames)}")


def test_latent_t_inversion_matches_comfys_forward_formula():
    """The node trims every reference video to n % 17 == 5, so inverting
    comfy's video_latent_t is exact on every frame count it keeps."""
    for frames in (5, 22, 39, 56, 107):   # points on the n % 17 == 5 grid
        latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        entry = {"kind": "video", "latent_t": latent_t,
                 "latent_h": 20, "latent_w": 32}
        recovered, height, width = driver_footprint._entry_geometry(entry)
        if (recovered, height, width) != (frames, 320, 512):
            pytest.fail(f"latent_t {latent_t} recovered {recovered} frames, "
                        f"expected {frames}")


def test_keyframe_and_audio_blocks_count_correctly():
    """A keyframe carries no latent_t and is one frame at the canvas; an
    audio-only reference block carries no spatial geometry at all."""
    request = {"positive": [[object(), {
        "minimax_keyframes": [{"resolved_frame_index": 0,
                               "latent": types.SimpleNamespace(
                                   shape=(1, 24, 1, 20, 32))}],
        "minimax_refs": [{"kind": "audio", "ref_audio_t": 80}],
    }]]}
    pixels, blocks, frames = driver_footprint.ref_pixel_volume_from_latents(
        request, H3)
    if (pixels, blocks, frames) != (512 * 320, 2, 1):
        pytest.fail(f"keyframe/audio geometry wrong: {(pixels, blocks, frames)}")


def test_malformed_extras_degrade_to_no_ref_term():
    """A None, a tensor-like or a string in the extras slot must degrade to
    zero reference geometry, never abort a render."""
    for extras in (None, types.SimpleNamespace(shape=(1, 2)), "junk", 7):
        request = {"positive": [[object(), extras]]}
        if driver_footprint.ref_pixel_volume_from_latents(request, H3) != (0, 0, 0):
            pytest.fail(f"extras slot {extras!r} must degrade to no ref term")
    for request in (None, {}, {"positive": None}, {"positive": "junk"},
                    {"positive": [None, 7, []]}):
        if driver_footprint.ref_pixel_volume_from_latents(request, H3) != (0, 0, 0):
            pytest.fail(f"malformed request {request!r} must degrade")


# Co-residency: a route question, not a name question.

def _cluster_mesh(name: str):
    return types.SimpleNamespace(handle=types.SimpleNamespace(
        owns_hosts=True, config=types.SimpleNamespace(
            hosts=[types.SimpleNamespace(name=name)])))


def test_cluster_rank0_on_the_drivers_own_fabric_ip_is_co_resident(monkeypatch):
    """cluster.toml names hosts[0] by a private fabric address, which is one
    of this box's own interface addresses and resolves to nothing the driver's
    hostname resolves to (the addresses below stand in for the ones
    cluster.example.toml documents). Comparing names would call rank 0 remote
    on the box it runs on, so cluster mode would never charge the weight term
    and the check could not refuse anything."""
    monkeypatch.setattr(socket, "gethostname", lambda: "driver-box")
    monkeypatch.setattr(socket, "getfqdn", lambda *_a: "localhost")
    monkeypatch.setattr(driver_footprint, "_address_is_local",
                        lambda address: address == "10.10.0.1")
    driver_footprint.host_is_local.cache_clear()
    if not driver_footprint.rank0_co_resident(_cluster_mesh("10.10.0.1")):
        pytest.fail("the driver's own fabric address is this box")
    driver_footprint.host_is_local.cache_clear()
    if driver_footprint.rank0_co_resident(_cluster_mesh("10.10.0.2")):
        pytest.fail("the sibling's fabric address is not this box")


def test_address_probe_answers_loopback_without_a_name_lookup():
    """The probe is a route lookup, not a name lookup: it sends no packet and
    it answers for a literal address that DNS knows nothing about."""
    if not driver_footprint._address_is_local("127.0.0.1"):
        pytest.fail("loopback is always an address on this box")
    if driver_footprint._address_is_local("203.0.113.7"):   # RFC 5737 TEST-NET
        pytest.fail("a documentation-range address is not on this box")


def test_co_residency_cases(monkeypatch):
    """owns_hosts False is a this_host() mesh, so every rank is a child of
    the driver process: certain co-residency. A cluster resolves rank 0 from
    config.hosts[0]. Anything unresolvable charges nothing."""
    local = types.SimpleNamespace(handle=types.SimpleNamespace(owns_hosts=False))
    if not driver_footprint.rank0_co_resident(local):
        pytest.fail("a local mesh always shares the driver host")
    # Stub the resolver so the unresolvable case does not wait out DNS.
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *_args, **_kwargs: (_ for _ in ()).throw(socket.gaierror())
    )
    if driver_footprint.rank0_co_resident(
            _cluster_mesh("a-host-that-does-not-resolve.invalid")):
        pytest.fail("an unresolvable host must charge nothing")
    if not driver_footprint.rank0_co_resident(_cluster_mesh("localhost")):
        pytest.fail("loopback is this box")
    if driver_footprint.rank0_co_resident(types.SimpleNamespace(handle=None)):
        pytest.fail("no handle must charge nothing")
    if driver_footprint.host_is_local(None) or driver_footprint.host_is_local(""):
        pytest.fail("an unnamed host is not local")


# Wiring: the render chain carries the check, and its callers do not name it.

def _h3_rig(monkeypatch, tmp_path, *, size: int, mem: int, preset="auto",
            worker_args=None, handle=None,
            unet_name="minimax_h3_ref2va_bf16.safetensors"):
    """The shipped path, with the artifact's real size instead of its bytes:
    the recorded checkpoints are 19 to 61 GiB and no test writes those."""
    checkpoint = tmp_path / "minimax_h3_ref2va_bf16.safetensors"
    checkpoint.write_bytes(b"x")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(checkpoint)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("minimax_h3", "bf16"))
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda _path: size)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: mem)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", None)
    monkeypatch.setattr(render_preflight, "_LAST_SUBMITTED_UNET", None)
    monkeypatch.setattr(render_preflight, "_LAST_DRIVER_CHARGED_UNET", None)
    mesh = types.SimpleNamespace(
        topology_preset=preset, worker_args=dict(worker_args or {}),
        handle=handle or types.SimpleNamespace(owns_hosts=False))
    model = types.SimpleNamespace(unet_name=unet_name, mesh=mesh, options={})
    latent = {"samples": types.SimpleNamespace(shape=(1, 24, 2, 20, 32))}
    return model, latent


def test_render_site_reads_the_operators_cluster_toml_reserve(monkeypatch, tmp_path):
    """The reserve is the operator's own co-residency headroom, and a
    cluster.toml [worker_args] entry never reaches the graph's worker_args.
    Resolving it through the canonical merger keeps this refusal and the
    worker-side co-residency warning in agreement."""
    handle = types.SimpleNamespace(
        owns_hosts=False,
        config=types.SimpleNamespace(worker_args={"uma_reserve_gb": 24.0}))
    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_INT8_PRUNED,
                            mem=int(40 * _GIB), handle=handle)
    seen: list = []
    monkeypatch.setattr(driver_footprint, "driver_footprint_preflight",
                        lambda estimate, **_kw: seen.append(estimate))
    render_preflight.activation_footprint_preflight_for_request(
        model, {"positive": [[object(), {}]]}, latent)
    if not seen or seen[0].reserve != 24 * _GIB:
        pytest.fail(f"the cluster.toml reserve must win: {seen}")


def test_a_retry_after_a_failed_render_is_not_charged_twice(monkeypatch, tmp_path):
    """A class C double charge. Both submission-path memos are stamped too
    late to cover a failed render: note_successful_render only after
    `pending.result()` returns, note_submitted_render only from the pipelined
    push. On the sequential path (every stock sampler node) a render that dies
    after the worker loads the checkpoint leaves the slot resident with both
    memos unset. Without the preflight's own memo the retry would charge the
    whole file again against a MemAvailable that already holds it, and refuse
    a config that just loaded.

    Driven through the shipped entry point twice under `auto`, with the render
    failing in between, at a MemAvailable that fits the load exactly once."""
    mem = DIT_REF2VA_INT8 + 6 * _GIB
    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_INT8, mem=mem)
    request = {"positive": [[object(), {"minimax_refs": CONFIG_A_REFS}]]}
    charged: list = []
    real = driver_footprint.estimate_driver_footprint

    def _record(**kwargs):
        charged.append(real(**kwargs))
        return charged[-1]

    monkeypatch.setattr(driver_footprint, "estimate_driver_footprint", _record)

    render_preflight.activation_footprint_preflight_for_request(
        model, request, latent)
    # The render dispatches, the worker loads, the sampler raises. Neither
    # submission memo is stamped, and the worker keeps the slot loaded.
    monkeypatch.setattr(mesh_safety, "mem_available_bytes",
                        lambda: mem - DIT_REF2VA_INT8)
    render_preflight.activation_footprint_preflight_for_request(
        model, request, latent)   # must not raise

    if len(charged) != 2:
        pytest.fail(f"expected two estimates, got {len(charged)}")
    if not charged[0].weight_bytes:
        pytest.fail("the first pass must charge the pending load")
    if charged[1].weight_bytes:
        pytest.fail(
            "the retry charged the resident checkpoint a second time: "
            f"{charged[1].weight_bytes} bytes, note "
            f"{charged[1].notes['weights']!r}")
    if "already loaded" not in charged[1].notes["weights"]:
        pytest.fail(f"the credit must be stated: {charged[1].notes['weights']}")


def test_a_weight_dtype_cast_stands_the_check_down(monkeypatch, tmp_path):
    """A cast's resident size is not the file's, which is why
    mesh_safety.stock_load_preflight skips one too. Charging the 61.7 GiB file
    for an fp8 cast that lands near 31 GiB resident would refuse a run that
    fits, and the cast already does what the message advises."""
    calls: list = []   # hoisted: a per-iteration list would close over the loop
    monkeypatch.setattr(driver_footprint, "estimate_driver_footprint",
                        lambda **kw: calls.append(kw))
    for dtype in ("fp8_e4m3fn", "fp8_e5m2", "bf16"):
        model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_AUTO_DISPATCH)
        model.options = {"weight_dtype": dtype}
        render_preflight.activation_footprint_preflight_for_request(
            model, {"positive": [[object(), {"minimax_refs": CONFIG_A_REFS}]]},
            latent)   # no raise
        if calls:
            pytest.fail(f"weight_dtype={dtype} must stand the check down")
    if driver_footprint.dtype_cast_requested({"weight_dtype": "default"}):
        pytest.fail("the default is not a cast")
    if driver_footprint.dtype_cast_requested(None):
        pytest.fail("no options is not a cast")
    if not driver_footprint.dtype_cast_requested({"weight_dtype": "junk"}):
        pytest.fail("an unreadable dtype must stand the check down too")


def test_the_reserve_knob_documents_that_it_now_refuses():
    """actor/worker_status.check_uma_reserve only logs and emits, but
    reserve_bytes also subtracts uma_reserve_gb from a refusal budget. An
    operator who raises it to protect a co-resident LLM can be refused a
    render, so the tooltip and the troubleshooting entry must say so. The text
    is pinned to the behavior because the two have drifted apart before."""
    from dgx_monarch.nodes.init import DGXMonarchInit

    tooltip = DGXMonarchInit.INPUT_TYPES()["optional"]["uma_reserve_gb"][1][
        "tooltip"]
    for phrase in ("warn", "refuse"):
        if phrase not in tooltip.lower():
            pytest.fail(f"the tooltip must still say it {phrase}s: {tooltip}")
    troubleshooting = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    if "uma_reserve_gb warns when renders eat into co-resident headroom" in troubleshooting:
        pytest.fail("TROUBLESHOOTING still calls uma_reserve_gb warn-only")
    if driver_footprint.reserve_bytes({"uma_reserve_gb": 24.0}) != 24 * _GIB:
        pytest.fail("the documented behavior is that a larger value wins")
    # The refusal also names what set the reserve, the operator's value or the
    # driver floor, because an operator may never open the tooltip.
    operator = driver_footprint.render_refusal(
        _estimate(DIT_REF2VA_INT8_PRUNED, int(40 * _GIB), reserve=24 * _GIB),
        "x.safetensors", H3)
    if "your uma_reserve_gb" not in operator:
        pytest.fail(f"an operator-set reserve must be named: {operator}")
    if "the driver floor" not in driver_footprint.render_refusal(
            _estimate(DIT_REF2VA_BF16, MEM_AUTO_DISPATCH), "x.safetensors", H3):
        pytest.fail("the default reserve must be named as the floor")


def test_pipelined_submission_inherits_the_refusal(monkeypatch, tmp_path):
    """The delegation lives inside the shared entry point, so the depth>1
    pipeline path inherits it. Moving the delegation into run_render would
    reopen the pipeline bypass closed for the activation preflight on
    2026-07-28, and every other test here would stay green."""
    from dgx_monarch.nodes.pipeline import RenderPipeline

    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                            mem=MEM_AUTO_DISPATCH)
    pipe = RenderPipeline(depth=2)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError):
        pipe.push(model, {"kind": "ksampler",
                          "positive": [[object(), {"minimax_refs": CONFIG_A_REFS}]]},
                  latent, 4.5, 10)


def test_unregistered_family_is_a_hard_no_op(monkeypatch, tmp_path):
    """Every family without a DRIVER_STACK_FAMILIES row keeps its behavior
    exactly, the way minimax_h3 is absent from REF_POSE_TOKEN_FAMILIES."""
    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16, mem=1)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("krea2", "bf16"))
    calls: list = []
    monkeypatch.setattr(driver_footprint, "estimate_driver_footprint",
                        lambda **kw: calls.append(kw))
    render_preflight.activation_footprint_preflight_for_request(
        model, {"positive": []}, latent)
    if calls:
        pytest.fail("an unregistered family must never reach the estimator")


def test_off_linux_never_refuses(monkeypatch, tmp_path):
    """mem_available_bytes returns None off Linux; there is nothing to
    compare against, so the check stands down."""
    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16, mem=1)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: None)
    calls: list = []
    monkeypatch.setattr(driver_footprint, "estimate_driver_footprint",
                        lambda **kw: calls.append(kw))
    render_preflight.activation_footprint_preflight_for_request(
        model, {"positive": []}, latent)
    if calls:
        pytest.fail("off Linux the estimator must not run")


def test_estimator_exception_fails_open(monkeypatch, tmp_path):
    """A broken estimator must never block a render.

    The H3 activation guard shares this entry point and is a separate boundary
    with its own estimator and its own kill switch. On this fixture it refuses
    for its own reason (61.7 GiB of pending weights plus the packed rows
    against a 48.6 GiB budget), which would mask what this test is about, so
    the test stands it down instead of retuning the shape: every other test in
    this file depends on that shape.
    """
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, "1")
    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                            mem=MEM_AUTO_DISPATCH)

    def _boom(**_kw):
        raise ValueError("estimator is broken")

    monkeypatch.setattr(driver_footprint, "estimate_driver_footprint", _boom)
    render_preflight.activation_footprint_preflight_for_request(
        model, {"positive": []}, latent)  # no raise


def test_genuine_refusal_still_reaches_the_caller(monkeypatch, tmp_path):
    """The fail-open handler re-raises the typed refusal before its generic
    `except Exception`, or no refusal would ever reach the caller."""
    model, latent = _h3_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                            mem=MEM_AUTO_DISPATCH)

    def _refuse(_estimate, **_kw):
        raise driver_footprint.DriverFootprintCapacityError("a genuine refusal")

    monkeypatch.setattr(driver_footprint, "driver_footprint_preflight", _refuse)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError,
                       match="a genuine refusal"):
        render_preflight.activation_footprint_preflight_for_request(
            model, {"positive": []}, latent)


def test_tag_shaped_unet_name_keeps_render_capacity_fail_closed(monkeypatch, tmp_path):
    """An artifact label cannot turn a completed capacity verdict into fail-open."""
    from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

    name = "MiniMax-H3/[dgxm:P] ref2va_bf16.safetensors"
    model, latent = _h3_rig(
        monkeypatch, tmp_path, size=DIT_REF2VA_BF16, mem=MEM_AUTO_DISPATCH,
        unet_name=name)

    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as excinfo:
        render_preflight.activation_footprint_preflight_for_request(
            model, {"positive": []}, latent)

    message = str(excinfo.value)
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert "Refuses MiniMax-H3/[dgxm;P] ref2va_bf16.safetensors" in message


def test_delegation_runs_before_the_activation_registry_guard():
    """The driver-stack families (minimax_h3, ltx) are absent from
    REF_POSE_TOKEN_FAMILIES, so a call placed after that guard would never run
    for them."""
    source = (SRC / "nodes" / "render_preflight.py").read_text()
    tree = ast.parse(source)
    host = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "activation_footprint_preflight_for_request")
    delegation = [node.lineno for node in ast.walk(host)
                  if isinstance(node, ast.Call)
                  and getattr(node.func, "id", "") == "_driver_footprint_preflight_for_request"]
    guard = [node.lineno for node in ast.walk(host)
             if isinstance(node, ast.Attribute)
             and node.attr == "REF_POSE_TOKEN_FAMILIES"]
    if not delegation or not guard:
        pytest.fail(f"expected both the delegation and the guard: {delegation}, {guard}")
    if min(delegation) > min(guard):
        pytest.fail("the delegation must precede the activation registry guard")


def test_neither_caller_can_route_around_the_shared_entry_point():
    """nodes/common.py and nodes/pipeline.py are the two callers of the shared
    entry point. Delegating from inside that function means neither can route
    around it."""
    common = (SRC / "nodes" / "common.py").read_text()
    pipeline = (SRC / "nodes" / "pipeline.py").read_text()
    for name, source in (("common.py", common), ("pipeline.py", pipeline)):
        if "driver_footprint" in source:
            pytest.fail(f"{name} must not name the new module; the delegation "
                        "lives inside render_preflight")
        if "activation_footprint_preflight_for_request" not in source:
            pytest.fail(f"{name} must still call the shared entry point")


def test_driver_footprint_is_a_leaf_module():
    """driver_footprint imports no mesh, actor, subprocess, torch or ComfyUI
    module anywhere in the file, and stays at 500 lines or fewer."""
    path = SRC / "driver_footprint.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.append(base)
            imported.extend(f"{base}.{alias.name}" if base else alias.name
                            for alias in node.names)
    banned = {"mesh", "actor", "subprocess", "dgx_monarch.mesh",
              "dgx_monarch.actor"}
    roots = {"torch", "comfy", "comfy_extras", "folder_paths", "nodes"}
    hits = [name for name in imported
            if name in banned or (name and name.split(".")[0] in roots)]
    if hits:
        pytest.fail(f"driver_footprint must stay a leaf, imports {hits}")
    if len(path.read_text().splitlines()) > 500:
        pytest.fail("driver_footprint must stay under the 500-line rule")
