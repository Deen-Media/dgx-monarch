"""Ulysses pad exclusion at degree 4, end to end in float64 on CPU (issue #188).

``auto`` folds MiniMax H3's spare ranks into uly4 at world 4. The pad algebra
is written for any degree, but the oracle in tests/test_usp_pad_exact.py runs
the all-to-all as an identity, so nothing on CPU had sharded a padded stream
four ways and put it back. This file does, for H3's 56 heads: it shards a
stream across four emulated ranks with random values in its pad rows (a pad
row is a real key after the first modulation), scatters heads with yunchang's
own ``all_to_all_4D`` over an emulated exchange, drops the rows
``padded_row_indices`` names with ``attend_without_pads``, scatters back,
gathers and trims, and compares with attention over the real rows alone.
Totals of 0 to 3 modulo 4 cover one, two and three pad rows, and a stream
short enough that the pads span several ranks.

It proves the row algebra and the all-to-all's layout at degree 4. It does not
prove a GPU kernel's accumulation order, NCCL, or anything about one GPU.
"""
from __future__ import annotations

import math

import pytest
import torch
from yunchang.comm import all_to_all  # a runtime dependency, so no import gate

import dgx_monarch.adapters.base as base
from dgx_monarch.adapters.usp_pad_exclusion import attend_without_pads

DEGREE = 4
HEADS = 56   # MiniMax H3's one sharded attention
DIM = 3


def _attend(q, k, v, groups=None):
    q_h, k_h, v_h = (t.transpose(1, 2) for t in (q, k, v))
    scores = q_h @ k_h.transpose(-1, -2) / math.sqrt(q_h.shape[-1])
    return (torch.softmax(scores, dim=-1) @ v_h).transpose(1, 2)


class _Exchange:
    """``all_to_all_single`` among DEGREE ranks that run one after another.

    yunchang's ``all_to_all_4D`` reshapes on the calling rank and hands
    ``all_to_all_single`` a tensor split into DEGREE equal chunks on dim 0;
    rank r's chunk s goes to rank s's slot r. The first pass records every
    rank's input, the second fills each rank's output from them.
    """

    def __init__(self):
        self.rank = 0
        self.inputs: dict[int, torch.Tensor] = {}
        self.recording = True

    def get_world_size(self, group=None):
        return DEGREE

    def all_to_all_single(self, output, tensor, group=None):
        if self.recording:
            self.inputs[self.rank] = tensor.clone()
            return
        for source in range(DEGREE):
            output[source] = self.inputs[source][self.rank]

    def run(self, locals_, scatter_idx, gather_idx):
        self.inputs, self.recording = {}, True
        for rank, value in enumerate(locals_):
            self.rank = rank
            all_to_all.all_to_all_4D(value, scatter_idx, gather_idx)
        self.recording = False
        results = []
        for rank, value in enumerate(locals_):
            self.rank = rank
            results.append(all_to_all.all_to_all_4D(value, scatter_idx, gather_idx))
        return results


@pytest.fixture
def exchange(monkeypatch):
    fake = _Exchange()
    monkeypatch.setattr(all_to_all, "dist", fake)
    monkeypatch.setattr(base, "sp_world", lambda: DEGREE)
    return fake


@pytest.mark.parametrize("rows", [20, 21, 22, 23, 1, 2, 5])
def test_degree_4_drops_every_pad_and_reproduces_unsharded_attention(exchange, rows):
    padded_rows = -(-rows // DEGREE) * DEGREE
    chunk = padded_rows // DEGREE
    generator = torch.Generator().manual_seed(rows)
    # Every row random, the pad rows too: after the first modulation a zero
    # embedding is not a zero key, so the test must not lean on zeros.
    q, k, v = (torch.randn(1, padded_rows, HEADS, DIM, generator=generator, dtype=torch.float64)
               for _ in range(3))
    drop = base.padded_row_indices([(rows, chunk)])
    assert drop == list(range(rows, padded_rows))

    def shard(t):
        return [t[:, rank * chunk:(rank + 1) * chunk].contiguous() for rank in range(DEGREE)]

    full_q, full_k, full_v = (exchange.run(shard(t), 2, 1) for t in (q, k, v))
    for value in full_q:
        assert value.shape == (1, padded_rows, HEADS // DEGREE, DIM)
    outs = [attend_without_pads(full_q[r], full_k[r], full_v[r], drop_rows=drop,
                                kv_drop_rows=drop, groups=None, attend=_attend)
            for r in range(DEGREE)]
    back = exchange.run(outs, 1, 2)
    gathered = torch.cat(back, dim=1).narrow(1, 0, rows)

    reference = _attend(q[:, :rows], k[:, :rows], v[:, :rows])
    assert torch.allclose(gathered, reference, atol=1e-12, rtol=0)
    if drop:
        attended = _attend(q, k, v)[:, :rows]
        assert (attended - reference).abs().max() > 1e-6


def test_the_exchange_is_yunchangs_head_scatter(exchange):
    """Rank r ends up with the whole token axis in rank order and heads
    r*14 to r*14+13, the layout padded_row_indices numbers rows in."""
    rows, chunk = 8, 2
    tokens = torch.arange(rows, dtype=torch.float64).view(1, rows, 1, 1).expand(1, rows, HEADS, 1)
    heads = torch.arange(HEADS, dtype=torch.float64).view(1, 1, HEADS, 1).expand(1, rows, HEADS, 1)
    value = (tokens * 1000 + heads).contiguous()
    locals_ = [value[:, rank * chunk:(rank + 1) * chunk].contiguous() for rank in range(DEGREE)]
    for rank, full in enumerate(exchange.run(locals_, 2, 1)):
        share = HEADS // DEGREE
        assert torch.equal(full, value[:, :, rank * share:(rank + 1) * share])
