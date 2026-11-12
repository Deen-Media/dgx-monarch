"""The classic bake's price held against every hardware record of a load with a LoRA stack.

Each row is a load with ``lora_low_rss`` off that ran on a DGX Spark on
2026-11-11 or 2026-11-12 (the first Spark under its clock cap, max 2100 MHz;
one row is the second Spark's Worker of a pair). It gives the stock price
the Worker printed for the file with no stack, the weights the stack patches
by class, the stack's files, the readings at which that load is on record,
and what the load and its render took from the box after the price line
(the reading less the lowest reading that followed).

Two things must hold for every row, whatever a later change does to a
constant in ``capacity_classic_bake``:

- the price is no more than a reading at which the load is on record, so a
  recorded render is not refused;
- the price is no less than what the load took plus the 5 GiB floor, so an
  admitted load does not take the box under the floor.

The first version of the term charged every format the scaled FP8 line and
the stack's files, and refused four of these states. No test held the
records then. CPU only: the weights are synthetic lists, not files.
"""
from __future__ import annotations

import dataclasses

import pytest

from dgx_monarch import capacity_classic_bake as price
from dgx_monarch import capacity_fit

GIB, MIB, KIB = 2 ** 30, 2 ** 20, 2 ** 10
FLOOR = capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES


def _gib(value: float) -> int:
    return int(value * GIB)


@dataclasses.dataclass(frozen=True)
class Record:
    name: str
    stock_gib: float                    # the Worker's price for the file with no stack
    weights: tuple                      # (class, stored bytes) for each key the stack patches
    files_gib: float                    # the stack's LoRA files
    readings_gib: tuple                 # readings at which this load is on record
    took_gib: float | None = None       # the reading less the lowest reading after the price line
    spare_gib: float = 0.0              # room the price must keep above take plus floor
    short_gib: float = 0.0              # a shortfall on record that the price is known not to cover

    @property
    def price_bytes(self) -> int:
        return _gib(self.stock_gib) + price.charge_from_weights(self.weights, _gib(self.files_gib))


def _of(count: int, kind: str, nbytes: int) -> tuple:
    return ((kind, nbytes),) * count


RECORDS = (
    # INT8 ConvRot Wan-Animate 2 with its LoRA: the shipped dgx-monarch-wan-animate2 template's pair. The
    # file has 480 INT8 weights; the other patched keys are biases and norm weights, and a few embeddings.
    # 45.0: loaded partly, baked whole by the partial-load route, low 11.93. 44.8: the same state, which
    # the first term refused at 49.5.
    Record("int8 wan-animate 2, 1261 patched keys", 37.6,
           _of(480, price.INT8_QUANTIZED, 30 * MIB) + _of(776, price.FULL_PRECISION, 16 * KIB)
           + _of(5, price.FULL_PRECISION, 50 * MIB),
           0.69, (45.0, 44.8), took_gib=33.1),
    # INT8 ConvRot MiniMax H3 with its LoRA, whole. 50.9: low 7.73. 50.4: refused by the first term at 51.4.
    Record("int8 minimax h3, 208 patched keys", 46.0, _of(208, price.INT8_QUANTIZED, 90 * MIB),
           1.82, (50.9, 50.4), took_gib=43.2),
    # Plain FP8 Qwen Image with its LoRA, whole. 49.1: low 10.19, less than the stock price promised.
    # 48.7: refused by the first term at 53.0. One weight count only.
    Record("plain fp8 qwen image, 720 patched keys", 45.0, _of(720, price.PLAIN_FP8, 26 * MIB),
           0.79, (49.1, 48.7), took_gib=38.9),
    # Scaled FP8 LTX 2.3 with its own 7.08 GiB stack, whole. 75.9: low 20.32 (a second run took 54.6).
    # 68.5: the state whose decode relief was proven the same day, which the first term refused at 72.1.
    Record("scaled fp8 ltx 2.3, 1660 patched keys", 50.9, _of(1660, price.FP8_QUANTIZED, 11 * MIB),
           7.08, (75.9, 68.5), took_gib=55.6, spare_gib=2.0),
    # The same file with its 0.305 GiB stack, whole. 82.6: low 35.22.
    Record("scaled fp8 ltx 2.3, 480 patched keys", 50.9, _of(480, price.FP8_QUANTIZED, 25 * MIB),
           0.305, (82.6,), took_gib=47.4, spare_gib=2.0),
    # BF16 Wan-Animate 2 with the same LoRA: the shipped memory-saver template, on a pair. 97.8 is the
    # first Spark's Worker, 113.6 the second's, whose low was 36.21: it took 1.3 to 1.4 GiB more than
    # the price covers. That shortfall is on record and waits for a measured pass; this row fails if a
    # change makes it larger.
    Record("bf16 wan-animate 2, 1261 patched keys", 69.13, _of(1261, price.FULL_PRECISION, 24 * MIB),
           0.69, (97.8, 113.6), took_gib=77.4, short_gib=1.5),
    # BF16 Klein with its 2.27 GiB stack, 121 patched weights by the headers, whole and roomy. The Worker
    # printed a price of 45.7 GiB. No low was read after its price line.
    Record("bf16 klein, 121 patched keys", 40.51, _of(121, price.FULL_PRECISION, 140 * MIB),
           2.27, (105.0,)),
)


@pytest.mark.parametrize("record", RECORDS, ids=[record.name for record in RECORDS])
def test_a_recorded_render_is_not_refused(record):
    for reading in record.readings_gib:
        assert record.price_bytes <= _gib(reading), (
            f"{record.name}: priced {record.price_bytes / GIB:.2f} GiB, on record at a reading of {reading} GiB")


@pytest.mark.parametrize("record", [r for r in RECORDS if r.took_gib is not None],
                         ids=[r.name for r in RECORDS if r.took_gib is not None])
def test_the_price_covers_what_the_recorded_load_took_and_the_floor(record):
    need = _gib(record.took_gib) + FLOOR
    covered = record.price_bytes + _gib(record.short_gib)
    assert covered >= need + _gib(record.spare_gib), (
        f"{record.name}: priced {record.price_bytes / GIB:.2f} GiB, took {record.took_gib} GiB "
        f"and the floor is {FLOOR / GIB:.0f}")


def test_the_figures_the_workers_printed_for_the_formats_left_as_they_were():
    """Full precision keeps the line and the files: 81.1 GiB and 45.7 GiB, as two Workers printed them."""
    by_name = {record.name: record for record in RECORDS}
    assert by_name["bf16 wan-animate 2, 1261 patched keys"].price_bytes / GIB == pytest.approx(81.1, abs=0.08)
    assert by_name["bf16 klein, 121 patched keys"].price_bytes / GIB == pytest.approx(45.7, abs=0.08)


def test_the_first_term_refused_four_of_these_states():
    """What this table would have caught: the line and the files for every format."""
    def first_price(record):
        count = len(record.weights)
        return (_gib(record.stock_gib) + price.CLASSIC_BAKE_FIXED_BYTES
                + count * price.CLASSIC_BAKE_BYTES_PER_WEIGHT + _gib(record.files_gib))

    refused = [record.name for record in RECORDS
               if any(first_price(record) > _gib(reading) for reading in record.readings_gib)]
    assert refused == ["int8 wan-animate 2, 1261 patched keys", "int8 minimax h3, 208 patched keys",
                       "plain fp8 qwen image, 720 patched keys", "scaled fp8 ltx 2.3, 1660 patched keys"]
