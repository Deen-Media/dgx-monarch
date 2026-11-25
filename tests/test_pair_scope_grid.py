"""The pair scopes never turn a render that had a route into a refusal.

A pair scope sends an equal pair from a cfg2 row to the row after it, and a
lift (a kind the scope leaves out, or a pair the sequence-parallel forward
refuses) keeps the pair on the cfg2 row. At world 4 a cfg2 row leaves two
data-parallel ranks that an odd batch cannot split, where auto refuses rather
than build a cfg-plus-sequence composite. A lift that kept the pair there
would turn the scope's route into that refusal, so the walk takes the scope's
row instead (topology.cfg_row_strands). The one exception is a pair the
sequence forward refuses: the scope's row is a sequence route that refused it
in the forward, so it now meets auto's refusal at once, named cell by cell
below.

The oracle is the walk itself under the scopes this branch started from
(dev 143acbe): rows 19, 20 and 79 scoped every kind, rows 15 and 68 granted
every pair, and nothing read the file's kind or the sequence tag. Under those
scopes this tree reproduces that tree's decision in every cell of a wider grid
(1,285,200 cells: five quant kinds, 53 megapixel values, batch 1 to 3, cfg++
on and off, both pair tags, worlds 2 to 4), so a cell here compares the change
with the tree before it.
"""
from __future__ import annotations

import dataclasses
import itertools

import pytest

from dgx_monarch import topology, topology_cfg_evidence
from dgx_monarch.topology import choose_auto_topology
from dgx_monarch.topology_cfg_evidence import BLOCK_SCALED_FP8, CAST_FP8
from dgx_monarch.topology_table import AUTO_TABLE

_BEFORE = {
    15: {"pairs": "any", "pair_quants": ()},
    19: {"pair_quants": ()},
    20: {"pair_quants": ()},
    68: {"pairs": "any", "pair_quants": ()},
    79: {"pair_quants": ()},
}
_FAMILIES = sorted({rule.family for rule in AUTO_TABLE} | {"unknown"})
_KINDS = (("bf16", None), ("fp8", None), ("fp8", BLOCK_SCALED_FP8), ("fp8", CAST_FP8),
          ("int8", None))
_MEGAPIXELS = (0.26, 0.79, 1.05, 1.19, 1.5, 2.4, 8.0)

# Every (world, family, quant, kind) whose cells went from a refusal to a route,
# each at world 4 with an odd batch and an equal pair; nothing went the other way.
# Every (world, family, quant, kind, sequence tag) whose cells changed outcome,
# each at world 4 with an odd batch and an equal pair.
_NOW_ROUTED = {
    # The scope sends an equal Krea2 BF16 or Qwen-Image FP8 pair to rows 12
    # and 71, which fold the spare ranks into ring2.
    (4, "krea2", "bf16", None, False),
    (4, "qwen_image", "fp8", None, False),
    # Rows 15 and 68 keep these kinds by the file's kind, which does not hold
    # where the row strands ranks: the pair takes rows 11 and 71 as an equal
    # pair of a scoped kind does.
    (4, "krea2", "fp8", None, False),
    (4, "krea2", "fp8", BLOCK_SCALED_FP8, False),
    (4, "krea2", "fp8", CAST_FP8, False),
    (4, "qwen_image", "fp8", BLOCK_SCALED_FP8, False),
    (4, "qwen_image", "fp8", CAST_FP8, False),
}
# A pair the sequence forward refuses (the tag) no longer takes the scope's
# Ulysses and ring row, which loaded the model and then refused in its
# forward; it meets auto's refusal at once, which names what runs it. A header
# the tag cannot read never gets here: auto refuses it itself.
_NOW_REFUSED_AT_ONCE = {
    (4, "chroma", quant, kind, True)
    for quant, kind in (("bf16", None), ("int8", None), ("fp8", None),
                        ("fp8", BLOCK_SCALED_FP8), ("fp8", CAST_FP8))
} | {(4, "boogu", "fp8", kind, True) for kind in (None, BLOCK_SCALED_FP8, CAST_FP8)}


def _outcome(family, quant, megapixels, world, **tags):
    try:
        decision = choose_auto_topology(family, quant, megapixels, world, **tags)
    except Exception as exc:  # the kind of outcome is the datum
        return f"raise {type(exc).__name__}"
    return f"route {decision.topology.describe()} row{decision.rule.row}"


def _grid():
    """Every cell of the grid, as (key, the tags auto reads, the tags before)."""
    flags = (False, True)
    for world, family, (quant, kind), megapixels, cfg, batch, keeps, pair, refused in (
            itertools.product((2, 3, 4), _FAMILIES, _KINDS, _MEGAPIXELS, (1.0, 3.5),
                              (1, 2, 3), flags, flags, flags)):
        before = {"cfg_value": cfg, "batch_size": batch, "keeps_uncond": keeps,
                  "pair_differs": pair}
        now = {**before, "pair_kind": kind, "sequence_refused": refused}
        yield ((world, family, quant, kind, megapixels, cfg, batch, keeps, pair, refused),
               now, before)


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(topology.log, "info", lambda *_args, **_kwargs: None)


def test_no_cell_of_the_world_2_3_and_4_grids_turns_a_route_into_a_refusal(monkeypatch,
                                                                            quiet):
    cells = list(_grid())
    now = {key: _outcome(key[1], key[2], key[4], key[0], **tags) for key, tags, _ in cells}
    for row, fields in _BEFORE.items():
        record = topology_cfg_evidence.CFG2_EVIDENCE[row]
        monkeypatch.setitem(topology_cfg_evidence.CFG2_EVIDENCE, row,
                            dataclasses.replace(record, **fields))
    memo: dict = {}
    refused, routed = set(), set()
    for key, _tags, tags in cells:
        plain = (key[0], key[1], key[2], *key[4:9])  # before, nothing read the kind or the tag
        if plain not in memo:
            memo[plain] = _outcome(key[1], key[2], key[4], key[0], **tags)
        was, is_now = memo[plain], now[key]
        if was.split()[0] != is_now.split()[0]:
            assert key[0] == 4 and key[6] % 2 and not key[8], (key, was, is_now)
            (refused if is_now.startswith("raise") else routed).add((*key[:4], key[9]))
    assert refused == _NOW_REFUSED_AT_ONCE
    assert routed == _NOW_ROUTED


@pytest.mark.parametrize("world", [2, 3, 4])
def test_a_lift_keeps_an_equal_pair_on_cfg2_wherever_the_row_runs(quiet, world):
    """Where the cfg2 row runs (world 2, an odd world as pure ring, an even
    batch at world 4), a lift keeps the pair on it as it keeps a differing
    pair; only the spare ranks an odd batch strands move it."""
    for batch in (1, 2):
        equal = choose_auto_topology("chroma", "fp8", 1.5, world, cfg_value=3.5,
                                     batch_size=batch)
        if world == 4 and batch == 1:
            assert (equal.topology.describe(), equal.rule.row) == ("uly2+ring2", 21)
            continue
        differing = choose_auto_topology("chroma", "fp8", 1.5, world, cfg_value=3.5,
                                         batch_size=batch, pair_differs=True)
        assert equal.rule.row == differing.rule.row == 19
        assert equal.topology == differing.topology
        assert equal.sage is differing.sage is False
