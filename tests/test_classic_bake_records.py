"""The classic bake's price held against every hardware record of a load with a LoRA stack.

Each row is a load with ``lora_low_rss`` off that ran on a DGX Spark between
2026-11-11 and 2026-11-12 (the first Spark under its clock cap, max 2100 MHz;
one row is the second Spark's Worker of a pair). A row gives:

- the stock price the Worker printed for the file with no stack;
- the weights the stack patches, by class, as the build counts them from the
  real headers;
- the stack's files, as often as the stack lists them;
- which reading the price was held against, read from the run's own log:
  ``before`` the Worker had read the stack's files (one load line, into an
  empty slot) or ``after`` (a bare load, then "a LoRA change ... would copy
  the model off the GPU first", then the stack's price line: the hot swap
  had read the stack and declined);
- the readings at which the load is on record, and what it took from the box
  after that reading (the reading less the lowest reading that followed).

Two things must hold for every row, whatever a later change does to a
constant or a rule in ``capacity_classic_bake``:

- the price is no more than a reading at which the load rendered, or at
  which it was wrongly refused, so it is admitted;
- the price less the 5 GiB floor is no less than what the load took, so an
  admitted load does not take the box under the floor.

Two earlier versions of the charge fail rows of this table: the line and the
files for every format on both readings; then no per-weight share for INT8
and no files for scaled FP8. Each is a test below. One row carries readings
the price is known to refuse, and one a shortfall it is known not to cover;
both are pinned so they cannot grow unseen. CPU only: the weights are
synthetic lists, not files.
"""
from __future__ import annotations

import dataclasses

import pytest

from dgx_monarch import capacity_classic_bake as price
from dgx_monarch import capacity_fit

GIB, MIB, KIB = 2 ** 30, 2 ** 20, 2 ** 10
FLOOR = capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
BEFORE, AFTER = "before", "after"


def _gib(value: float) -> int:
    return int(value * GIB)


@dataclasses.dataclass(frozen=True)
class Record:
    name: str
    stock_gib: float                    # the Worker's price for the file with no stack
    weights: tuple                      # (class, stored bytes) for each key the stack patches
    files_gib: float                    # the stack's LoRA files, each as often as it is listed
    basis: str                          # BEFORE or AFTER the Worker read the stack's files
    readings_gib: tuple                 # readings at which this load must be admitted
    took_gib: float | None = None       # the reading less the lowest reading after the price line
    spare_gib: float = 0.0              # room the price must keep above take plus floor
    short_gib: float = 0.0              # a shortfall on record that the price is known not to cover
    refused_gib: tuple = ()             # readings on record that the price is known to refuse

    @property
    def charge_bytes(self) -> int:
        return price.charge_from_weights(self.weights, _gib(self.files_gib), stack_on_host=self.basis == AFTER)

    @property
    def price_bytes(self) -> int:
        return _gib(self.stock_gib) + self.charge_bytes


def _of(count: int, kind: str, nbytes: int) -> tuple:
    return ((kind, nbytes),) * count


# By the real headers: 480 INT8 weights, 774 BF16 biases and norm weights of at most 640 KiB, and 7
# BF16 embedding weights of 1.4 MiB and up. The shipped dgx-monarch-wan-animate2 template's own pair.
WAN_INT8 = (_of(480, price.INT8_QUANTIZED, 30 * MIB) + _of(774, price.FULL_PRECISION, 16 * KIB)
            + _of(7, price.FULL_PRECISION, 50 * MIB))
H3_INT8 = _of(200, price.INT8_QUANTIZED, 90 * MIB) + _of(8, price.FULL_PRECISION, 100 * MIB)
QWEN_720 = _of(720, price.PLAIN_FP8, 26 * MIB)
LTX_OWN = (_of(1344, price.FP8_QUANTIZED, 13 * MIB) + _of(288, price.FP8_QUANTIZED, 100 * KIB)
           + _of(26, price.FULL_PRECISION, 20 * MIB) + _of(2, price.FULL_PRECISION, 100 * KIB))
LTX_SMALL = _of(480, price.FP8_QUANTIZED, 25 * MIB)

RECORDS = (
    # INT8 ConvRot Wan-Animate 2, 1261 patched keys, the base held (block C). At a reading of 44.6
    # ComfyUI placed it partly and the partial-load route baked it whole: it took 33.6.
    Record("int8 wan-animate 2, base held, partial load and the route", 37.57, WAN_INT8, 0.687, AFTER,
           (44.6,), took_gib=33.6),
    # The same stack at 46.7, placed WHOLE (low 7.29): ComfyUI's classic bake keeps the host copy of
    # the patched weights. The second version's price promised 35.3 and it took 39.4.
    Record("int8 wan-animate 2, base held, whole", 37.57, WAN_INT8, 0.687, AFTER, (46.7,), took_gib=39.4),
    # The same stack into an empty slot, one load (blocks A and B): partly, with the route, at a reading
    # of 45.0, where it took 33.1; 44.8 is the same state, refused at 49.5 by the first version. The
    # price that covers the whole load above refuses both, by 0.2 and 0.4 GiB: read before the file it
    # is 45.2. Recorded as refused, for the lead's ruling; this row fails if the gap grows.
    Record("int8 wan-animate 2, empty slot, partial load and the route", 37.57, WAN_INT8, 0.687, BEFORE,
           (), took_gib=33.1, refused_gib=(45.0, 44.8)),
    # INT8 ConvRot MiniMax H3, 208 patched keys, whole. The base held (block C): 42.3 from 51.5.
    Record("int8 minimax h3, base held, whole", 46.0, H3_INT8, 1.822, AFTER, (51.5,), took_gib=42.3),
    # Into an empty slot (block A): 43.2 from 50.9.
    Record("int8 minimax h3, empty slot, whole", 46.0, H3_INT8, 1.822, BEFORE, (50.9,), took_gib=43.2),
    # Plain FP8 Qwen Image, 720 patched keys. The base held (block C): whole 37.7 from 48.9; partly,
    # with the route, 38.0 from 47.8.
    Record("plain fp8 qwen image, 720 keys, base held", 44.96, QWEN_720, 0.791, AFTER,
           (48.9, 47.8), took_gib=38.0),
    # Into an empty slot (block A): whole, 38.9 from 49.1. 48.7 was refused at 53.0 by the first version.
    Record("plain fp8 qwen image, 720 keys, empty slot", 44.96, QWEN_720, 0.791, BEFORE,
           (49.1, 48.7), took_gib=38.9),
    # The same model with a second stack, 240 patched keys, the base held, partly with the route.
    Record("plain fp8 qwen image, 240 keys, base held", 44.96, _of(240, price.PLAIN_FP8, 26 * MIB), 0.088, AFTER,
           (50.0,), took_gib=39.3),
    # Scaled FP8 LTX 2.3 with its own 7.08 GiB stack, the base held: 55.6 (block B, 75.9), 55.5 (67.9)
    # and 55.1 (66.8) in block C. 68.5 is the state whose decode relief was proven the same day (low
    # 13.04), refused at 72.1 by the first version.
    Record("scaled fp8 ltx 2.3, own stack, base held", 50.9, LTX_OWN, 7.083, AFTER,
           (75.9, 67.9, 66.8, 68.5), took_gib=55.6, spare_gib=2.0),
    # The same load measured from an empty Worker, before the base or the file is read: 62.5 from 74.2.
    # The second version charged no file and promised 60.1.
    Record("scaled fp8 ltx 2.3, own stack, empty worker", 50.9, LTX_OWN, 7.083, BEFORE,
           (74.2,), took_gib=62.5, spare_gib=2.0),
    # The same file listed twice at 0.4 each, one load into an empty Worker: 69.5 from 82.4, both copies
    # held (33.8 GiB of the Worker's memory against 27.2 for one). The second version promised 64.8.
    Record("scaled fp8 ltx 2.3, own stack twice, empty worker", 50.9, LTX_OWN, 2 * 7.083, BEFORE,
           (82.4,), took_gib=69.5, spare_gib=2.0),
    # The same file with its 0.305 GiB stack, the base held: 47.4 (82.6, 2026-11-11) and 47.7 (58.2).
    Record("scaled fp8 ltx 2.3, small stack, base held", 50.9, LTX_SMALL, 0.305, AFTER,
           (82.6, 58.2), took_gib=47.7, spare_gib=2.0),
    # BF16 Wan-Animate 2 with the same LoRA: the shipped memory-saver template, one load into an empty
    # slot on a pair. 97.8 is the first Spark's Worker, 113.6 the second's, whose low was 36.21: it took
    # 1.3 to 1.4 GiB more than the price covers. That shortfall is on record and waits for a measured
    # pass; this row fails if a change makes it larger. 95.0: the cleared first Spark of 2026-11-05
    # ("about 95 GiB" in DESIGN 5.9), where that template's clear and retry rendered.
    Record("bf16 wan-animate 2, 1261 keys, empty slot", 69.13, _of(1261, price.FULL_PRECISION, 24 * MIB), 0.687,
           BEFORE, (97.8, 113.6, 95.0), took_gib=77.4, short_gib=1.5),
    # BF16 Klein with its 2.27 GiB stack, 121 patched weights, the base held and roomy. No low was read
    # after its price line.
    Record("bf16 klein, 121 keys, base held", 40.51, _of(121, price.FULL_PRECISION, 140 * MIB), 2.27, AFTER,
           (105.0,)),
)
NAMES = [record.name for record in RECORDS]
TAKEN = [record for record in RECORDS if record.took_gib is not None]


def _named(name: str) -> Record:
    return next(record for record in RECORDS if record.name == name)


@pytest.mark.parametrize("record", RECORDS, ids=NAMES)
def test_a_recorded_load_is_admitted_at_every_reading_it_is_on_record_at(record):
    for reading in record.readings_gib:
        assert record.price_bytes <= _gib(reading), (
            f"{record.name}: priced {record.price_bytes / GIB:.2f} GiB, on record at a reading of {reading} GiB")


@pytest.mark.parametrize("record", TAKEN, ids=[record.name for record in TAKEN])
def test_the_price_covers_what_the_recorded_load_took_and_the_floor(record):
    need = _gib(record.took_gib) + FLOOR
    covered = record.price_bytes + _gib(record.short_gib)
    assert covered >= need + _gib(record.spare_gib), (
        f"{record.name}: priced {record.price_bytes / GIB:.2f} GiB, took {record.took_gib} GiB "
        f"and the floor is {FLOOR / GIB:.0f}")


def test_what_each_recorded_stack_is_charged():
    """The charges, pinned: a change to a constant or a rule that moves one shows here first."""
    charged = {record.name: round(record.charge_bytes / GIB, 2) for record in RECORDS}
    assert charged == {
        "int8 wan-animate 2, base held, partial load and the route": 6.95,
        "int8 wan-animate 2, base held, whole": 6.95,
        "int8 wan-animate 2, empty slot, partial load and the route": 7.64,
        "int8 minimax h3, base held, whole": 2.84,
        "int8 minimax h3, empty slot, whole": 4.66,
        "plain fp8 qwen image, 720 keys, base held": 2.0,
        "plain fp8 qwen image, 720 keys, empty slot": 2.79,
        "plain fp8 qwen image, 240 keys, base held": 2.0,
        "scaled fp8 ltx 2.3, own stack, base held": 14.16,
        "scaled fp8 ltx 2.3, own stack, empty worker": 21.24,
        "scaled fp8 ltx 2.3, own stack twice, empty worker": 28.32,
        "scaled fp8 ltx 2.3, small stack, base held": 5.52,
        "bf16 wan-animate 2, 1261 keys, empty slot": 11.92,
        "bf16 klein, 121 keys, base held": 2.89,
    }


def test_the_int8_charge_sits_between_its_two_bounds():
    """On a held base the whole load must be covered and the same stack admitted at 44.6: 0.2 GiB apart.

    Recorded as it is: the price of the shipped template's own stack may move
    by about a tenth of a GiB either way before one of those two rows fails.
    """
    whole = _named("int8 wan-animate 2, base held, whole")
    tight = _named("int8 wan-animate 2, base held, partial load and the route")
    assert whole.price_bytes == tight.price_bytes
    assert _gib(39.4) + FLOOR <= whole.price_bytes <= _gib(44.6)
    assert whole.price_bytes / GIB == pytest.approx(44.52, abs=0.02)


def test_the_readings_the_int8_price_is_known_to_refuse():
    """Read before the file, that price is 0.7 GiB more, above two readings at which the stack was run.

    Both loads were placed partly and baked by the route (33.1 GiB taken). A
    price that covers the whole load cannot admit them. The gap is pinned so
    it cannot grow unseen.
    """
    refused = [record for record in RECORDS if record.refused_gib]
    assert [record.name for record in refused] == ["int8 wan-animate 2, empty slot, partial load and the route"]
    (record,) = refused
    assert record.price_bytes / GIB == pytest.approx(45.21, abs=0.02)
    for reading in record.refused_gib:
        assert _gib(reading) < record.price_bytes <= _gib(reading + 0.45)


def test_the_first_version_refused_six_of_these_rows():
    """The line and the files for every format, on both readings (the build of 2026-11-12 morning)."""
    def first(record):
        return (_gib(record.stock_gib) + price.CLASSIC_BAKE_FIXED_BYTES
                + len(record.weights) * price.CLASSIC_BAKE_BYTES_PER_WEIGHT + _gib(record.files_gib))

    refused = [record.name for record in RECORDS
               if any(first(record) > _gib(reading) for reading in record.readings_gib)]
    assert refused == ["int8 wan-animate 2, base held, partial load and the route",
                       "int8 wan-animate 2, base held, whole",
                       "int8 minimax h3, empty slot, whole",
                       "plain fp8 qwen image, 720 keys, base held",
                       "plain fp8 qwen image, 720 keys, empty slot",
                       "scaled fp8 ltx 2.3, own stack, base held"]


def test_the_second_version_came_up_short_on_three_recorded_loads():
    """No per-weight share for INT8, and no files for scaled FP8 on either reading (the afternoon's build)."""
    light = (price.INT8_QUANTIZED, price.PLAIN_FP8)

    def second(record):
        kinds = {kind for kind, _nbytes in record.weights}
        if kinds & set(light):
            large = sum(1 for kind, nbytes in record.weights
                        if kind not in light and nbytes >= price.SMALL_KEY_BYTES)
            return (_gib(record.stock_gib) + price.CLASSIC_BAKE_FIXED_BYTES
                    + large * price.CLASSIC_BAKE_BYTES_PER_WEIGHT + _gib(record.files_gib))
        line = price.CLASSIC_BAKE_FIXED_BYTES + len(record.weights) * price.CLASSIC_BAKE_BYTES_PER_WEIGHT
        if price.FP8_QUANTIZED in kinds:
            return _gib(record.stock_gib) + max(line, _gib(record.files_gib) + _gib(4.7))
        return _gib(record.stock_gib) + line + _gib(record.files_gib)

    short = [record.name for record in TAKEN
             if second(record) + _gib(record.short_gib) < _gib(record.took_gib) + FLOOR]
    assert short == ["int8 wan-animate 2, base held, whole", "scaled fp8 ltx 2.3, own stack, empty worker",
                     "scaled fp8 ltx 2.3, own stack twice, empty worker"]


def test_full_precision_is_charged_as_the_workers_printed_it():
    """The line and the files, to the figure a Worker printed for a load into an empty slot: 81.1 GiB."""
    assert _named("bf16 wan-animate 2, 1261 keys, empty slot").price_bytes / GIB == pytest.approx(81.1, abs=0.08)
    # Klein's Worker printed 45.7 with its 2.27 GiB of files charged on a held base; they are read by then.
    klein = _named("bf16 klein, 121 keys, base held")
    assert (klein.price_bytes + _gib(klein.files_gib)) / GIB == pytest.approx(45.7, abs=0.08)
