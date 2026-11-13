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

One row holds the second by another rule, and says so (``whole_from_gib``):
the INT8 Wan-Animate 2 stack loaded WHOLE takes 39.4 GiB against a price of
39.6. Every row ran at the Worker's default ComfyUI reserve, 8 GiB, and at
that reserve ComfyUI itself places this model whole only from a reading of
45.1 (under it the load is placed partly and the partial-load route carries
it, which the price does cover), so the lowest whole load leaves 5.7. That
row asserts exactly that, and that with a reserve too low for it the price
carries the share and covers the whole load by itself.

Two earlier versions of the charge fail rows of this table: the line and the
files for every format on both readings; then no per-weight share for INT8
at any reserve and no files for scaled FP8. Each is a test below. One row
carries a shortfall the price is known not to cover, pinned so it cannot
grow unseen. CPU only: the weights are synthetic lists, not files.
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
    reserve_gib: float = 8.0            # the Worker's ComfyUI reserve the load ran at: the default, every row
    whole_from_gib: float | None = None  # ComfyUI's lowest whole-load reading at that reserve, where a row leans on it

    def charge_at(self, reserve_gib: float | None) -> int:
        return price.charge_from_weights(self.weights, _gib(self.files_gib), stack_on_host=self.basis == AFTER,
                                         reserve_bytes=None if reserve_gib is None else _gib(reserve_gib))

    def price_at(self, reserve_gib: float | None) -> int:
        return _gib(self.stock_gib) + self.charge_at(reserve_gib)

    @property
    def charge_bytes(self) -> int:
        return self.charge_at(self.reserve_gib)

    @property
    def price_bytes(self) -> int:
        return self.price_at(self.reserve_gib)


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
    # the patched weights, and it took 39.4. At the reserve of 8 it ran at, ComfyUI places this model
    # whole only from a reading of 45.1 (cells 1 and 2 and block A's 45.0, all placed partly under it).
    Record("int8 wan-animate 2, base held, whole", 37.57, WAN_INT8, 0.687, AFTER, (46.7,), took_gib=39.4,
           whole_from_gib=45.1),
    # The same stack into an empty slot, one load (blocks A and B): partly, with the route, at a reading
    # of 45.0, where it took 33.1; 44.8 is the same state, refused at 49.5 by the first version.
    Record("int8 wan-animate 2, empty slot, partial load and the route", 37.57, WAN_INT8, 0.687, BEFORE,
           (45.0, 44.8), took_gib=33.1),
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
    # and 55.1 (66.8) in block C. 68.5 is the state whose decode relief was proven the same day,
    # refused at 72.1 by the first version; block C rebuilt that state (another model first, the text
    # encoder held) and the second version admitted it at 65.1 against 67.9: whole, 54.9 taken, a low
    # of 13.04. It must stay admitted.
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
# The rows whose price covers take plus floor by itself, and the one that leans on ComfyUI's condition.
COVERED = [record for record in TAKEN if record.whole_from_gib is None]
INT8 = [record for record in TAKEN if any(kind == price.INT8_QUANTIZED for kind, _nbytes in record.weights)]


def _named(name: str) -> Record:
    return next(record for record in RECORDS if record.name == name)


@pytest.mark.parametrize("record", RECORDS, ids=NAMES)
def test_a_recorded_load_is_admitted_at_every_reading_it_is_on_record_at(record):
    for reading in record.readings_gib:
        assert record.price_bytes <= _gib(reading), (
            f"{record.name}: priced {record.price_bytes / GIB:.2f} GiB, on record at a reading of {reading} GiB")


@pytest.mark.parametrize("record", COVERED, ids=[record.name for record in COVERED])
def test_the_price_covers_what_the_recorded_load_took_and_the_floor(record):
    need = _gib(record.took_gib) + FLOOR
    covered = record.price_bytes + _gib(record.short_gib)
    assert covered >= need + _gib(record.spare_gib), (
        f"{record.name}: priced {record.price_bytes / GIB:.2f} GiB, took {record.took_gib} GiB "
        f"and the floor is {FLOOR / GIB:.0f}")


def test_every_row_ran_at_the_default_reserve_and_one_leans_on_comfyuis_condition():
    assert {record.reserve_gib for record in RECORDS} == {8.0}
    assert [record.name for record in TAKEN if record.whole_from_gib is not None] == [
        "int8 wan-animate 2, base held, whole"]


def test_the_whole_int8_load_keeps_the_floor_by_comfyuis_own_condition_at_the_default_reserve():
    """What holds for the whole load: not "the price covers take plus floor", which is false at a reserve of 8.

    ComfyUI placed it whole at 46.7, it took 39.4 and the low was 7.29. The
    price of 39.6 promises 34.6. The floor holds because ComfyUI places this
    model whole only from a reading of 45.1 at that reserve: the lowest whole
    load leaves 5.7, which is the reserve less what a whole load takes beyond
    ComfyUI's own accounting (``WHOLE_LOAD_BEYOND_COMFYUI_BYTES``, 2.3 GiB).
    """
    whole = _named("int8 wan-animate 2, base held, whole")
    assert whole.price_bytes / GIB == pytest.approx(39.62, abs=0.02)
    assert whole.price_bytes < _gib(whole.took_gib) + FLOOR                      # the price alone does not cover it
    assert _gib(46.7) - _gib(whole.took_gib) == pytest.approx(_gib(7.29), abs=_gib(0.06))   # the low on record
    lowest_whole_low = _gib(whole.whole_from_gib) - _gib(whole.took_gib)
    assert lowest_whole_low >= FLOOR                                             # 5.7 GiB
    beyond = _gib(whole.reserve_gib) - lowest_whole_low
    assert beyond == pytest.approx(price.WHOLE_LOAD_BEYOND_COMFYUI_BYTES, abs=_gib(0.01))
    share_keys = sum(1 for kind, nbytes in whole.weights
                     if kind == price.INT8_QUANTIZED or nbytes < price.SMALL_KEY_BYTES)
    assert share_keys == price.RECORDED_WHOLE_LOAD_SHARE_KEYS
    assert price.reserve_keeps_floor(_gib(whole.reserve_gib), share_keys)
    # Under 45.1 the same stack is placed partly and the route carries it: the price covers that take.
    for name in ("int8 wan-animate 2, base held, partial load and the route",
                 "int8 wan-animate 2, empty slot, partial load and the route"):
        route = _named(name)
        assert route.price_bytes >= _gib(route.took_gib) + FLOOR
        assert all(_gib(reading) < _gib(whole.whole_from_gib) + (_gib(route.files_gib) if route.basis == BEFORE else 0)
                   for reading in route.readings_gib)


@pytest.mark.parametrize("reserve_gib", [7.0, 3.0, 0.0, None])
def test_with_a_reserve_that_does_not_keep_the_floor_the_int8_price_covers_every_take_by_itself(reserve_gib):
    """A lower reserve lets ComfyUI place the model whole at lower readings, so the price carries the share."""
    for record in INT8:
        assert record.price_at(reserve_gib) >= _gib(record.took_gib) + FLOOR, record.name
    whole = _named("int8 wan-animate 2, base held, whole")
    assert whole.price_at(reserve_gib) / GIB == pytest.approx(44.52, abs=0.02)
    assert whole.price_at(reserve_gib) <= _gib(44.6)        # still admitted where the template's own cell ran
    empty = _named("int8 wan-animate 2, empty slot, partial load and the route")
    assert empty.price_at(reserve_gib) / GIB == pytest.approx(45.21, abs=0.02)
    # What a lowered reserve costs: read before the file, the two tight readings of blocks A and B are refused.
    assert all(_gib(reading) < empty.price_at(reserve_gib) <= _gib(reading + 0.45) for reading in empty.readings_gib)


@pytest.mark.parametrize("reserve_gib", [7.3, 8.0, 16.0])
def test_with_a_reserve_that_keeps_the_floor_the_int8_rows_are_priced_as_at_the_default(reserve_gib):
    for record in INT8:
        assert record.price_at(reserve_gib) == record.price_bytes, record.name


def test_what_each_recorded_stack_is_charged():
    """The charges, pinned: a change to a constant or a rule that moves one shows here first."""
    charged = {record.name: round(record.charge_bytes / GIB, 2) for record in RECORDS}
    assert charged == {
        "int8 wan-animate 2, base held, partial load and the route": 2.05,
        "int8 wan-animate 2, base held, whole": 2.05,
        "int8 wan-animate 2, empty slot, partial load and the route": 2.74,
        "int8 minimax h3, base held, whole": 2.06,
        "int8 minimax h3, empty slot, whole": 3.88,
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


def test_the_first_version_refused_seven_of_these_rows():
    """The line and the files for every format, on both readings (the build of 2026-11-12 morning)."""
    def first(record):
        return (_gib(record.stock_gib) + price.CLASSIC_BAKE_FIXED_BYTES
                + len(record.weights) * price.CLASSIC_BAKE_BYTES_PER_WEIGHT + _gib(record.files_gib))

    refused = [record.name for record in RECORDS
               if any(first(record) > _gib(reading) for reading in record.readings_gib)]
    assert refused == ["int8 wan-animate 2, base held, partial load and the route",
                       "int8 wan-animate 2, base held, whole",
                       "int8 wan-animate 2, empty slot, partial load and the route",
                       "int8 minimax h3, empty slot, whole",
                       "plain fp8 qwen image, 720 keys, base held",
                       "plain fp8 qwen image, 720 keys, empty slot",
                       "scaled fp8 ltx 2.3, own stack, base held"]


def test_the_second_version_came_up_short_on_three_recorded_loads():
    """No per-weight share for INT8, and no files for scaled FP8 on either reading (the afternoon's build).

    Its price came up short of take plus floor on these three. The two scaled
    FP8 rows are covered now by the files. The INT8 row is still short by its
    price at the default reserve: ComfyUI's own whole-load condition keeps
    the floor there, and under that reserve the share does
    (``test_the_whole_int8_load_keeps_the_floor_by_comfyuis_own_condition_at_the_default_reserve``).
    """
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
