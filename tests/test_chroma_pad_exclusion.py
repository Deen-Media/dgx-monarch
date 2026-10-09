"""Chroma's divisibility pad rows: where they land, and that none is attended.

Two halves, both CPU-only and float64.

The first is row algebra. `ChromaAdapter`'s two loops carry two layouts, text
first in both: a double block cats `[txt_local, img_local]` per rank, and the
single loop shards one re-concatenated `[txt, img]` stream. Reading either
segment order backwards drops real tokens and the render still finishes, so the
coordinates are pinned against a hand-rebuilt gathered layout.

The second drives the real bound forward, one thread a rank, through the
real `base.make_usp_attention` dispatch, `usp_options`, `assert_ulysses_only_padding`
and `attend_without_pads`, against the same forward at world 1. The kernel is a
recording float64 softmax, so `torch.equal` is the row algebra rather than a
kernel's accumulation order.
"""
from __future__ import annotations

import math
import sys
import threading
import types
from unittest.mock import patch

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from dgx_monarch.adapters import base, flux_family
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.flux_family import ChromaAdapter


def _matches(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    """The sharded and unsharded toy paths reduce in a different order, and
    on x86 CPUs that order shows in the last bits where it does not on this
    rig's ARM cores. Bit equality is the claim between ranks; against the
    unsharded reference the claim is agreement within atol 1e-6."""
    return torch.equal(candidate, reference) or torch.allclose(
        candidate, reference, rtol=0.0, atol=1e-6)

# Disjoint nonzero marker ranges, so a misrouted row reads as a wrong number
# rather than a wrong shape, and neither ever collides with shard_seq's own
# zero pad.
_TXT_BASE = 1
_IMG_BASE = 1000


# First half: the row algebra, sequential, no threads.
def _markers(base_value: int, rows: int) -> torch.Tensor:
    return torch.arange(base_value, base_value + rows,
                        dtype=torch.float64).reshape(1, rows, 1)


def _shard_both_layouts(txt_len: int, img_len: int, world: int):
    """Shard every rank the way ChromaAdapter's two loops do.

    Returns the gathered double-loop stream and its ids, the gathered
    single-loop stream and its ids, and the drop set each loop computes.
    """
    txt, img = _markers(_TXT_BASE, txt_len), _markers(_IMG_BASE, img_len)
    txt_ids, img_ids = txt.repeat(1, 1, 3), img.repeat(1, 1, 3)
    joint, joint_ids = torch.cat((txt, img), dim=1), torch.cat((txt_ids, img_ids), dim=1)
    double, double_ids, single, single_ids = [], [], [], []
    with patch.object(base, "sp_world", return_value=world):
        for rank in range(world):
            with patch.object(base, "sp_rank", return_value=rank):
                txt_local, txt_orig = base.shard_seq(txt, dim=1)
                img_local, img_orig = base.shard_seq(img, dim=1)
                txt_ids_local, _ = base.shard_seq(txt_ids, dim=1)
                img_ids_local, _ = base.shard_seq(img_ids, dim=1)
                joint_local, joint_orig = base.shard_seq(joint, dim=1)
                joint_ids_local, _ = base.shard_seq(joint_ids, dim=1)
            # Chroma's double block cats text first (comfy DoubleStreamBlock:
            # q = cat((txt_q, img_q))), unlike lens, whose block cats image
            # first. The single loop shards one already joined stream.
            double.append(torch.cat((txt_local, img_local), dim=1))
            double_ids.append(torch.cat((txt_ids_local, img_ids_local), dim=1))
            single.append(joint_local)
            single_ids.append(joint_ids_local)
        double_drop = base.padded_row_indices(
            [(txt_orig, txt_local.shape[1]), (img_orig, img_local.shape[1])])
        single_drop = base.padded_row_indices([(joint_orig, joint_local.shape[1])])
    return (torch.cat(double, dim=1), torch.cat(double_ids, dim=1), double_drop,
            torch.cat(single, dim=1), torch.cat(single_ids, dim=1), single_drop)


def _kept(stream: torch.Tensor, drop: list[int]) -> list[float]:
    dropped = set(drop)
    return [float(stream[0, row, 0]) for row in range(stream.shape[1])
            if row not in dropped]


_LENGTHS = st.integers(1, 12)
_WORLDS = st.sampled_from([1, 2, 4])


@given(txt_len=_LENGTHS, img_len=_LENGTHS, world=_WORLDS)
@settings(max_examples=60)
def test_the_drop_set_addresses_every_pad_row_and_no_real_token(txt_len, img_len, world):
    # Attention is permutation-invariant over keys as long as q/k/v share the
    # permutation, so what correctness needs is the real values present once
    # each after the drop, not the original order.
    double, double_ids, double_drop, single, single_ids, single_drop = \
        _shard_both_layouts(txt_len, img_len, world)
    expected = sorted([float(v) for v in range(_TXT_BASE, _TXT_BASE + txt_len)]
                      + [float(v) for v in range(_IMG_BASE, _IMG_BASE + img_len)])
    for stream, ids, drop in ((double, double_ids, double_drop),
                              (single, single_ids, single_drop)):
        assert sorted(_kept(stream, drop)) == expected
        # The id table takes the same pad and the same chunk: that is the RoPE
        # alignment the forward's assert holds inside the block loops.
        assert sorted(_kept(ids, drop)) == expected
        for row in drop:
            assert float(stream[0, row, 0]) == 0.0
            assert float(ids[0, row, 0]) == 0.0
        assert len(drop) == len(set(drop))
        assert stream.shape[1] - len(drop) == txt_len + img_len


def test_world_one_pads_nothing_in_either_layout():
    _, _, double_drop, single, _, single_drop = _shard_both_layouts(5, 8, 1)
    assert (double_drop, single_drop) == ([], [])
    assert single.shape[1] == 13


def test_the_segment_order_is_text_first_and_reading_it_backwards_drops_a_token():
    """The one mistake that renders instead of raising.

    Chroma's double block cats text first, so `padded_row_indices` must name
    the text segment first. Named the other way round, the set addresses a row
    holding a real image marker, and every render would quietly lose it.
    """
    double, _, double_drop, _, _, _ = _shard_both_layouts(txt_len=5, img_len=8, world=2)
    # 5 text rows pad to 6 (3 a rank), 8 image rows split 4 and 4, so the
    # gathered layout is [txt_r0(0..2), img_r0(3..6), txt_r1(7..9), img_r1(10..13)]
    # and the single pad row is the last of rank 1's text slice.
    assert double_drop == [9]
    assert float(double[0, 9, 0]) == 0.0
    with patch.object(base, "sp_world", return_value=2):
        reversed_order = base.padded_row_indices([(8, 4), (5, 3)])
    assert reversed_order == [13]
    assert float(double[0, 13, 0]) == _IMG_BASE + 7  # a real image token


# Second half: the toy forward, one thread a rank, against the same forward at world 1.
BATCH, HEADS, HEAD_DIM = 1, 1, 4
DIM = HEADS * HEAD_DIM
# Channel 0 of the toy hidden state is a tag every block copies through
# untouched: real text rows 1..T, real image rows T+1..T+I, and a pad row keeps
# shard_seq's own 0. A tag survives the projection bias that a zero row does
# not, so the kernel can name exactly which rows it was handed. It is a label
# and not a feature: the query side carries a zero there, so the tag never
# enters a score and never decides an output value.
_TAG = 0

_CONTEXT = threading.local()   # which block the calling thread is inside
_LOCAL = threading.local()     # this thread's sequence-parallel rank
_ACTIVE = None                 # the harness the installed stubs answer for


def _rank() -> int:
    return getattr(_LOCAL, "rank", 0)


def test_a_zero_row_after_norm_and_modulation_is_not_a_zero_key():
    """The premise of the exclusion, asserted directly.

    `img_norm1` is a LayerNorm with no affine, which maps an all-zero row to an
    all-zero row; `apply_mod` then returns the modulation shift, which is not
    zero, and the qkv map turns that into a real key at RoPE (0, 0, 0). Chroma
    runs its blocks at modulation=False, so the shift is a (B, 1, D) slice of
    the distilled table and broadcasts over every row, pad rows included.
    """
    row = torch.zeros(1, 1, 6, dtype=torch.float64)
    shift = torch.full((1, 1, 6), 0.25, dtype=torch.float64)
    scale = torch.full((1, 1, 6), 0.5, dtype=torch.float64)
    normed = torch.nn.functional.layer_norm(row, (6,))
    assert torch.equal(normed, row)
    modulated = normed * (1 + scale) + shift
    assert not torch.equal(modulated, torch.zeros_like(modulated))


class _Fabric:
    """One exchange per call site, indexed per rank so no write outruns a read.

    A single barrier with one slot list would race: the faster thread returns,
    reaches the next exchange and overwrites its slot while the slower one is
    still reading. Each rank counts its own exchanges instead, and the k-th
    exchange on every rank shares slot list k.
    """

    def __init__(self, world: int) -> None:
        self.world = world
        self.barrier = threading.Barrier(world)
        self.lock = threading.Lock()
        self.slots: dict[int, list] = {}
        self.counts: dict[int, int] = {}

    def exchange(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        rank = _rank()
        with self.lock:
            index = self.counts.get(rank, 0)
            self.counts[rank] = index + 1
            slot = self.slots.setdefault(index, [None] * self.world)
        slot[rank] = tensor
        self.barrier.wait(timeout=30)
        return list(slot)


class _Harness:
    """The fabric, the recording kernel, and what both of them saw."""

    def __init__(self, world: int, *, strict: bool = True) -> None:
        self.world = world
        self.fabric = _Fabric(world)
        self.strict = strict
        self.expected_tags: list[float] = []
        self.calls: list[tuple] = []      # (entry, block_type, index, kept tags)
        self.dispatch: list = []          # drop_rows every usp_attention saw

    def attend(self, entry: str, q, k, v):
        """Attend over a key axis sorted by tag, recording the rows it kept.

        Only the key axis is reduced over, so query row order cannot change a
        per-row value: sorting the keys removes the accumulation-order term and
        leaves `torch.equal` meaning the row algebra.
        """
        tags = k[0, :, 0, _TAG].tolist()
        where = getattr(_CONTEXT, "where", (None, None))
        self.calls.append((entry, where[0], where[1], tags))
        if self.strict:
            assert 0.0 not in tags, "a divisibility pad row reached attention"
            assert sorted(tags) == self.expected_tags, "attention lost a real row"
        order = torch.argsort(k[0, :, 0, _TAG])
        k, v = k.index_select(1, order), v.index_select(1, order)
        q_h, k_h, v_h = (t.transpose(1, 2) for t in (q, k, v))
        scores = q_h @ k_h.transpose(-1, -2) / math.sqrt(q_h.shape[-1])
        return (torch.softmax(scores, dim=-1) @ v_h).transpose(1, 2)


def _install_stubs(monkeypatch, harness: _Harness) -> None:
    """Rank accessors and the xfuser/yunchang leaves, bound to this harness.

    The real xfuser probes for an accelerator at import, so it is stubbed in
    sys.modules the way the other adapter tests do it. Only three leaves are
    fake: the all-to-all, the sequence-parallel all-gather, and the kernel.
    Everything between them is the shipped code.
    """
    global _ACTIVE
    _ACTIVE = harness

    class _StubUSP:
        ulysses_pg = ring_pg = attn_type = attn_processor = None
        q_descale = k_descale = v_descale = None

        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, attn, q_l, k_l, v_l, **kwargs):
            # xfuser's own callable, which an unpadded stream must still reach:
            # exchange the keys, attend this rank's rows against the whole axis.
            keys = torch.cat(_ACTIVE.fabric.exchange(k_l), dim=1)
            values = torch.cat(_ACTIVE.fabric.exchange(v_l), dim=1)
            return _ACTIVE.attend("xfuser", q_l, keys, values)

        @staticmethod
        def ring_attn_fn(q, k, v, **kwargs):
            # Reached only at the Ulysses full-sequence point, past the pads.
            return _ACTIVE.attend("exact", q, k, v)

    class _StubAllToAll:
        @staticmethod
        def apply(_group, tensor, scatter_idx, gather_idx):
            if (scatter_idx, gather_idx) == (2, 1):
                # The head scatter is skipped: every rank keeps all heads and
                # runs the same kernel, which is what makes every rank
                # deterministic and identical.
                return torch.cat(_ACTIVE.fabric.exchange(tensor), dim=1)
            return tensor.chunk(_ACTIVE.world, dim=1)[_rank()]

    class _StubSpGroup:
        @staticmethod
        def all_gather(tensor, dim):
            return torch.cat(_ACTIVE.fabric.exchange(tensor), dim=dim)

    modules = {name: types.ModuleType(name) for name in (
        "xfuser", "xfuser.core", "xfuser.core.distributed",
        "xfuser.core.long_ctx_attention", "yunchang", "yunchang.kernels",
        "yunchang.comm", "yunchang.comm.all_to_all")}
    modules["xfuser.core.distributed"].get_ring_parallel_world_size = lambda: 1
    modules["xfuser.core.distributed"].get_sp_group = lambda: _StubSpGroup()
    modules["xfuser.core.long_ctx_attention"].xFuserLongContextAttention = _StubUSP
    modules["yunchang.kernels"].AttnType = {"TORCH_FLASH": "TORCH_FLASH"}
    modules["yunchang.comm.all_to_all"].SeqAllToAll4D = _StubAllToAll
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    # flux_family imports sp_rank by value, so all three names need patching
    # (the `sp` fixture in test_flux_adapter.py explains why).
    monkeypatch.setattr(base, "sp_world", lambda: _ACTIVE.world)
    monkeypatch.setattr(base, "sp_rank", _rank)
    monkeypatch.setattr(flux_family, "sp_rank", _rank)


@pytest.fixture
def comfy_layers_stub(monkeypatch):
    """The one comfy leaf chroma's forward imports before the block loops."""
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    flux_pkg = types.ModuleType("comfy.ldm.flux")
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.timestep_embedding = lambda t, dim: torch.zeros(*t.shape, dim)
    comfy.ldm, ldm.flux, flux_pkg.layers = ldm, flux_pkg, layers
    for name, module in (("comfy", comfy), ("comfy.ldm", ldm),
                         ("comfy.ldm.flux", flux_pkg),
                         ("comfy.ldm.flux.layers", layers)):
        monkeypatch.setitem(sys.modules, name, module)


class _Modulation:
    """A (B, 1, D) shift/scale/gate triple, chroma's ChromaModulationOut shape.

    Per batch, never per token: the distilled table is built from timesteps and
    guidance before any shard, so there is nothing in that path to shard or pad.
    """

    def __init__(self, seed: int) -> None:
        self.shift = torch.full((1, 1, DIM), 0.125 + 0.01 * seed, dtype=torch.float64)
        self.scale = torch.full((1, 1, DIM), 0.25 - 0.01 * seed, dtype=torch.float64)
        self.gate = torch.full((1, 1, DIM), 0.5 + 0.02 * seed, dtype=torch.float64)


def _keep_tag(mapped: torch.Tensor, tag: torch.Tensor) -> torch.Tensor:
    return torch.cat((tag, mapped[..., 1:]), dim=-1)


def _row_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-row, so a differing row count cannot change a per-row value."""
    body = x[..., 1:]
    scaled = body / (1.0 + body.abs().sum(dim=-1, keepdim=True))
    return torch.cat((x[..., :1], scaled), dim=-1)


def _modulate(x: torch.Tensor, mod: _Modulation) -> torch.Tensor:
    normed = _row_norm(x)
    return _keep_tag(normed * (1 + mod.scale) + mod.shift, x[..., :1])


def _channel_map(x: torch.Tensor, seed: int) -> torch.Tensor:
    weight = torch.linspace(0.7, 1.3, DIM, dtype=torch.float64) + 0.05 * seed
    bias = torch.linspace(-0.2, 0.2, DIM, dtype=torch.float64) - 0.03 * seed
    return x * weight + bias


def _add_pe(q: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Fold the row's position into the query, so a misaligned id shard shows."""
    return torch.cat((q[..., :1], q[..., 1:] + pe), dim=-1)


def _attention(q, k, v, pe, options):
    """Call the installed override the way comfy's wrap_attn does."""
    _CONTEXT.where = (options.get("block_type"), options.get("block_index"))
    q = _add_pe(q, pe)
    shaped = [t.reshape(t.shape[0], t.shape[1], HEADS, -1).transpose(1, 2)
              for t in (q, k, v)]
    override = options["optimized_attention_override"]
    return override(None, *shaped, HEADS, skip_reshape=True, mask=None,
                    transformer_options=options, _inside_attn_wrapper=True)


class _ToyBlock:
    """Elementwise apart from the attention: a per-row normalization, a
    per-channel qkv map, a per-channel output map and a gated residual. No gemm
    over the token axis, so a differing row count cannot move a per-row value.
    """

    def __init__(self, index: int) -> None:
        self.index = index

    def qkv(self, x, mod):
        modulated = _modulate(x, mod)
        tag, blank = x[..., :1], torch.zeros_like(x[..., :1])
        # The tag is a label the kernel reads off the key axis, never a
        # feature. The query carries a zero there, so channel 0 contributes
        # nothing to any score and a modulated pad row competes for softmax
        # weight on the same terms as a real row, which is the thing being
        # measured. A tag inside the dot product would make every score a
        # near-hard max and hide the pad in the eleventh decimal.
        return (_keep_tag(_channel_map(modulated, self.index + 1), blank),
                _keep_tag(_channel_map(modulated, self.index + 2), tag),
                _keep_tag(_channel_map(modulated, self.index + 3), blank))

    def residual(self, x, attended, mod):
        # The projection zeroes the tag channel, so the residual carries the
        # row's tag untouched into every later block.
        projected = _channel_map(attended, self.index + 4)
        return x + mod.gate * _keep_tag(projected, torch.zeros_like(projected[..., :1]))


class _ToyDoubleBlock(_ToyBlock):
    def __call__(self, img, txt, vec, pe, attn_mask, transformer_options):
        img_mod, txt_mod = vec
        img_q, img_k, img_v = self.qkv(img, img_mod)
        txt_q, txt_k, txt_v = self.qkv(txt, txt_mod)
        # Text first, exactly as comfy's DoubleStreamBlock cats it.
        attended = _attention(torch.cat((txt_q, img_q), dim=1),
                              torch.cat((txt_k, img_k), dim=1),
                              torch.cat((txt_v, img_v), dim=1), pe, transformer_options)
        # Stock splits at the local txt row count, which includes this rank's
        # share of the text pad; attend_without_pads returns the full query
        # layout, so the split stays correct.
        txt_attn, img_attn = attended[:, :txt.shape[1]], attended[:, txt.shape[1]:]
        return self.residual(img, img_attn, img_mod), self.residual(txt, txt_attn, txt_mod)


class _ToySingleBlock(_ToyBlock):
    def __call__(self, img, vec, pe, attn_mask, transformer_options):
        q, k, v = self.qkv(img, vec)
        return self.residual(img, _attention(q, k, v, pe, transformer_options), vec)


class _ToyChroma:
    """Chroma-shaped where it matters: two block loops, a per-batch modulation
    table, an identity position embedder and a final head."""

    def __init__(self, doubles: int = 2, singles: int = 2) -> None:
        self.img_in = lambda x: x
        self.txt_in = lambda x: x
        self.distilled_guidance_layer = lambda x: x
        self.pe_embedder = lambda ids: ids
        self.double_blocks = [_ToyDoubleBlock(i) for i in range(doubles)]
        self.single_blocks = [_ToySingleBlock(10 + i) for i in range(singles)]
        self.skip_mmdit: tuple = ()
        self.skip_dit: tuple = ()

    def get_modulations(self, vectors, kind, idx=None):
        # Deterministic on purpose: hash() is salted per process, and a toy
        # whose weights move between runs cannot reproduce its own failure.
        return _Modulation((sum(map(ord, kind)) + 3 * (idx or 0)) % 7)

    def final_layer(self, img, vec):
        return _channel_map(img, 9)


def _streams(txt_len: int, img_len: int):
    """Tagged text and image streams plus chroma's own id tables."""
    txt = torch.zeros(BATCH, txt_len, DIM, dtype=torch.float64)
    img = torch.zeros(BATCH, img_len, DIM, dtype=torch.float64)
    txt[0, :, _TAG] = torch.arange(1, txt_len + 1, dtype=torch.float64)
    img[0, :, _TAG] = torch.arange(txt_len + 1, txt_len + img_len + 1,
                                   dtype=torch.float64)
    for column in range(1, DIM):
        txt[0, :, column] = 0.1 * column + 0.03 * torch.arange(txt_len, dtype=torch.float64)
        img[0, :, column] = 0.2 * column - 0.05 * torch.arange(img_len, dtype=torch.float64)
    # Chroma builds txt_ids as zeros, so every text position is (0, 0, 0); the
    # image ids are distinct, which is where a misaligned shard would show.
    txt_ids = torch.zeros(BATCH, txt_len, DIM - 1, dtype=torch.float64)
    img_ids = torch.arange(1, img_len + 1, dtype=torch.float64).reshape(
        1, img_len, 1).repeat(1, 1, DIM - 1)
    return txt, txt_ids, img, img_ids


def _drive(harness: _Harness, model: _ToyChroma, streams) -> list[torch.Tensor]:
    """Run the real bound forward once per rank and return what each returned."""
    txt, txt_ids, img, img_ids = streams
    kernel = base.make_usp_attention("TORCH_FLASH")

    def attn(*args, **kwargs):
        harness.dispatch.append(kwargs.get("drop_rows"))
        return kernel(*args, **kwargs)

    ChromaAdapter().inject_usp(
        model, InjectionContext(topology_sp=harness.world, usp_attention=attn))
    outputs: list = [None] * harness.world
    errors: list = [None] * harness.world

    def body(rank: int) -> None:
        try:
            _LOCAL.rank = rank
            outputs[rank] = model.forward_orig(
                img, img_ids, txt, txt_ids, torch.zeros(1, dtype=torch.float64),
                guidance=torch.zeros(1, dtype=torch.float64))
        except BaseException as exc:   # re-raised in the test body
            errors[rank] = exc
            # Kill the other thread's wait now rather than after the timeout.
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,))
               for rank in range(harness.world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    for error in errors:
        if error is not None:
            raise error
    return outputs


def _reference(monkeypatch, model, streams, txt_len, img_len):
    """The same bound forward at world 1: no pad, no drop, natural order."""
    harness = _Harness(1)
    harness.expected_tags = [float(tag) for tag in range(1, txt_len + img_len + 1)]
    _install_stubs(monkeypatch, harness)
    return _drive(harness, model, streams)[0], harness


def _sharded(monkeypatch, model, streams, txt_len, img_len, *, strict=True, world=2):
    harness = _Harness(world, strict=strict)
    harness.expected_tags = [float(tag) for tag in range(1, txt_len + img_len + 1)]
    _install_stubs(monkeypatch, harness)
    return _drive(harness, model, streams), harness


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("txt_len,img_len", [(5, 8), (1, 8)])
def test_the_sharded_render_reproduces_the_unsharded_one_row_for_row(
    monkeypatch, comfy_layers_stub, txt_len, img_len, world
):
    """The claim, on an odd text stream and on the one-token blank.

    At world 2, 5 text rows pad to 6 in the double loop and the 13-row joint
    stream pads to 14 in the single loop; 1 text row is the PixArt T5 blank
    negative a cfg_pp sampler evaluates even at cfg 1.0. World 4 is the wider
    mesh this rig cannot run: 1 text row pads to 4, so three ranks hold a text
    slice of pad alone, and the drop set spans them.
    """
    model = _ToyChroma()
    streams = _streams(txt_len, img_len)
    reference, _ = _reference(monkeypatch, model, streams, txt_len, img_len)
    outputs, harness = _sharded(monkeypatch, model, streams, txt_len, img_len, world=world)

    assert reference.shape == (BATCH, img_len, DIM)
    # Every rank returns the whole image, and it is the unsharded answer.
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference)

    # Every attention call saw exactly the real rows, in both loops, and the
    # recorder met one call per rank per block, so none slipped past it.
    for entry, _kind, _index, tags in harness.calls:
        assert entry == "exact"
        assert 0.0 not in tags
        assert sorted(tags) == harness.expected_tags
    for kind, blocks in (("double", model.double_blocks), ("single", model.single_blocks)):
        seen = [call for call in harness.calls if call[1] == kind]
        assert len(seen) == harness.world * len(blocks)
    # Two loops, two drop sets: the double loop names the text pad inside a
    # text slice, the single loop names a tail of the joined stream.
    double_drop, single_drop = harness.dispatch[0], harness.dispatch[-1]
    assert double_drop and single_drop and double_drop != single_drop
    joint = txt_len + img_len
    assert single_drop == list(range(joint, math.ceil(joint / world) * world))


def test_an_even_stream_keeps_the_untouched_xfuser_call(monkeypatch, comfy_layers_stub):
    """The even template is unchanged by construction, and this is the proof.

    Nothing pads, so `padded_row_indices` returns [], `base.usp_attention` reads
    `padded == False`, and every call reaches xfuser's own callable rather than
    the exclusion path.
    """
    model = _ToyChroma()
    streams = _streams(4, 8)
    reference, _ = _reference(monkeypatch, model, streams, 4, 8)
    outputs, harness = _sharded(monkeypatch, model, streams, 4, 8)

    assert _matches(outputs[0], reference)
    assert harness.dispatch and all(drop == [] for drop in harness.dispatch)
    assert {call[0] for call in harness.calls} == {"xfuser"}


def test_without_the_drop_sets_the_pad_rows_change_the_answer(
    monkeypatch, comfy_layers_stub
):
    """The negative control: without it this file would pass on a forward that
    never padded at all. The pads stay, the drop sets are forced empty, and the
    real rows must now differ from the unsharded answer."""
    model = _ToyChroma()
    streams = _streams(5, 8)
    reference, _ = _reference(monkeypatch, model, streams, 5, 8)
    monkeypatch.setattr(flux_family, "_family_drop_rows",
                        lambda pad_vouched, segments: [])
    outputs, harness = _sharded(monkeypatch, model, streams, 5, 8, strict=False)

    # Materially, not in the last bit: a toy whose pad key wins no softmax
    # weight would pass `not torch.equal` on rounding alone and prove nothing.
    assert (outputs[0] - reference).norm() / reference.norm() > 1e-4
    # The pad row reached the kernel: its tag is shard_seq's own zero.
    assert any(0.0 in tags for _entry, _kind, _index, tags in harness.calls)
