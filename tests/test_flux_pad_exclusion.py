"""Flux, Flux2 and LongCat divisibility pad rows: none of them is attended.

The three run one bound forward, so one file drives all three. It is the same
claim tests/test_chroma_pad_exclusion.py makes for chroma, on the sibling
forward: one thread a rank, through the real `base.make_usp_attention`
dispatch, `usp_options`, `assert_ulysses_only_padding` and
`attend_without_pads`, against the same forward at world 1. The kernel is a
recording float64 softmax, so `torch.equal` is the row algebra rather than a
kernel's accumulation order.

Two branches of that forward differ and both run here. Flux 1.x and LongCat
take their block modulation from one per-batch vector; Flux2 sets
`global_modulation`, which computes the double-block modulation pair once and
copies it through, and carries a per-token `txt_norm` before the shard. Neither
touches the token axis, which is what these runs check rather than assume.

The gathered coordinates themselves are pinned in the chroma file, because both
forwards build their two drop sets from the same `padded_row_indices` call on
the same two layouts. The hardware legs are in docs/VALIDATION.md (Flux-family
pad-row exclusion, 2026-09-09).
"""
from __future__ import annotations

import math
import sys
import threading
import types

import pytest
import torch

from dgx_monarch.adapters import base, flux_family
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.flux_family import (
    Flux2Adapter,
    FluxAdapter,
    LongCatAdapter,
)

ADAPTERS = [FluxAdapter, Flux2Adapter, LongCatAdapter]


def _matches(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    """The sharded and unsharded toy paths reduce in a different order, and on
    x86 CPUs that order shows in the last bits of these float64 values where it
    does not on this rig's ARM cores. Bit equality is the claim between ranks;
    against the unsharded reference the claim is agreement within atol 1e-6."""
    return torch.equal(candidate, reference) or torch.allclose(
        candidate, reference, rtol=0.0, atol=1e-6)


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
    """Why pad rows must leave attention, on this family's own block.

    `img_norm1` in comfy's flux DoubleStreamBlock is a LayerNorm with
    `elementwise_affine=False`, which maps an all-zero row to an all-zero row;
    `modulated_norm` then adds the modulation shift, which is not zero, and the
    qkv map turns that into a real key at RoPE (0, 0, 0). Flux's Modulation is
    per batch, a (B, 1, D) triple, so the shift broadcasts over every row, pad
    rows included.
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
        self.orders: list = []            # (block_type, sequence_order) per call

    def attend(self, entry: str, q, k, v):
        """Attend over a key axis sorted by tag, recording the rows it kept.

        Only the key axis is reduced over, so query row order cannot change a
        per-row value: sorting the keys removes the accumulation-order term and
        leaves `torch.equal` meaning the row algebra.
        """
        tags = k[0, :, 0, _TAG].tolist()
        # Every batch item reaches the kernel in the same key order.
        assert all(k[b, :, 0, _TAG].tolist() == tags for b in range(k.shape[0]))
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
    sys.modules. Only three leaves are fake: the all-to-all, the sequence-parallel
    all-gather, and the kernel. Everything between them is the shipped code.
    """
    global _ACTIVE
    _ACTIVE = harness

    class _StubUSP:
        ulysses_pg = ring_pg = attn_type = attn_processor = None
        q_descale = k_descale = v_descale = None

        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, attn, q_l, k_l, v_l, **kwargs):
            # xfuser's own callable, the route of an unpadded call with no order
            # descriptor: exchange the keys, attend this rank's rows against the whole axis.
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
    """The one comfy leaf the flux forward imports before the block loops."""
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
    """A (B, 1, D) shift/scale/gate triple, flux's ModulationOut shape.

    Per batch, never per token: `vec` is built from the timestep and the
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
        # The query's tag channel is zero (see _TAG), so a modulated pad row
        # competes for softmax weight on the same terms as a real row, which
        # is what this file measures.
        return (_keep_tag(_channel_map(modulated, self.index + 1), blank),
                _keep_tag(_channel_map(modulated, self.index + 2), tag),
                _keep_tag(_channel_map(modulated, self.index + 3), blank))

    def residual(self, x, attended, mod):
        # The projection zeroes the tag channel, so the residual carries the
        # row's tag untouched into every later block.
        projected = _channel_map(attended, self.index + 4)
        return x + mod.gate * _keep_tag(projected, torch.zeros_like(projected[..., :1]))


def _split_vec(vec):
    """Stock reads one modulation for both streams, or the Flux2 pair."""
    return vec if isinstance(vec, tuple) else (vec, vec)


class _ToyDoubleBlock(_ToyBlock):
    def __call__(self, img, txt, vec, pe, attn_mask, transformer_options):
        img_mod, txt_mod = _split_vec(vec)
        img_q, img_k, img_v = self.qkv(img, img_mod)
        txt_q, txt_k, txt_v = self.qkv(txt, txt_mod)
        # Text first, as comfy's DoubleStreamBlock cats it.
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
        return self.residual(
            img, _attention(*self.qkv(img, vec), pe, transformer_options), vec)


class _ToyFlux:
    """Flux-shaped where it matters: two block loops, a per-batch modulation
    vector, an identity position embedder and a final head. `global_modulation`
    picks the Flux2 branch, which also carries the pre-shard `txt_norm`.
    """

    def __init__(self, *, global_modulation: bool, doubles: int = 2,
                 singles: int = 2) -> None:
        self.img_in = lambda x: x
        self.txt_in = lambda x: x
        self.time_in = lambda embedded: _Modulation(1)
        self.guidance_in = None
        self.vector_in = None
        self.txt_norm = None
        if global_modulation:
            self.txt_norm = lambda x: _keep_tag(_row_norm(x), x[..., :1])
        self.pe_embedder = lambda ids: ids
        self.params = types.SimpleNamespace(
            guidance_embed=False, vec_in_dim=DIM,
            global_modulation=global_modulation,
        )
        self.double_stream_modulation_img = lambda vec: _Modulation(2)
        self.double_stream_modulation_txt = lambda vec: _Modulation(3)
        self.single_stream_modulation = lambda vec: (_Modulation(4), _Modulation(5))
        self.double_blocks = [_ToyDoubleBlock(i) for i in range(doubles)]
        self.single_blocks = [_ToySingleBlock(10 + i) for i in range(singles)]

    def final_layer(self, img, vec):
        return _channel_map(img, 9)


def _streams(txt_len: int, img_len: int):
    """Tagged text and image streams plus the flux id tables."""
    txt = torch.zeros(BATCH, txt_len, DIM, dtype=torch.float64)
    img = torch.zeros(BATCH, img_len, DIM, dtype=torch.float64)
    txt[0, :, _TAG] = torch.arange(1, txt_len + 1, dtype=torch.float64)
    img[0, :, _TAG] = torch.arange(txt_len + 1, txt_len + img_len + 1,
                                   dtype=torch.float64)
    for column in range(1, DIM):
        txt[0, :, column] = 0.1 * column + 0.03 * torch.arange(txt_len, dtype=torch.float64)
        img[0, :, column] = 0.2 * column - 0.05 * torch.arange(img_len, dtype=torch.float64)
    # Flux builds txt_ids as zeros unless the model config names txt_ids_dims,
    # as Flux2 and LongCat do (comfy/model_detection.py); the ids run 1..T here
    # so a misaligned text shard shows on every family. Image ids are distinct.
    txt_ids = torch.arange(1, txt_len + 1, dtype=torch.float64).reshape(
        1, txt_len, 1).repeat(1, 1, DIM - 1)
    img_ids = torch.arange(1, img_len + 1, dtype=torch.float64).reshape(
        1, img_len, 1).repeat(1, 1, DIM - 1)
    return txt, txt_ids, img, img_ids


def _drive(harness: _Harness, adapter_cls, model: _ToyFlux, streams,
           pure_ulysses: bool = False) -> list:
    """Run the real bound forward once per rank and return what each returned."""
    txt, txt_ids, img, img_ids = streams
    kernel = base.make_usp_attention("TORCH_FLASH")

    def attn(*args, **kwargs):
        harness.dispatch.append(kwargs.get("drop_rows"))
        harness.orders.append((_CONTEXT.where[0], kwargs.get("sequence_order")))
        return kernel(*args, **kwargs)

    adapter_cls().inject_usp(
        model, InjectionContext(topology_sp=harness.world, usp_attention=attn,
                                pure_ulysses=pure_ulysses))
    outputs: list = [None] * harness.world
    errors: list = [None] * harness.world

    def body(rank: int) -> None:
        try:
            _LOCAL.rank = rank
            outputs[rank] = model.forward_orig(
                img, img_ids, txt, txt_ids, torch.zeros(1, dtype=torch.float64),
                torch.zeros(1, DIM, dtype=torch.float64))
        except BaseException as exc:   # re-raised after the join below
            errors[rank] = exc
            # Abort the other threads' barrier wait now rather than at the timeout.
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


def _model_for(adapter_cls) -> _ToyFlux:
    return _ToyFlux(global_modulation=adapter_cls is Flux2Adapter)


def _reference(monkeypatch, adapter_cls, model, streams, txt_len, img_len):
    """The same bound forward at world 1: no pad, no drop, natural order."""
    harness = _Harness(1)
    harness.expected_tags = [float(tag) for tag in range(1, txt_len + img_len + 1)]
    _install_stubs(monkeypatch, harness)
    return _drive(harness, adapter_cls, model, streams)[0], harness


def _sharded(monkeypatch, adapter_cls, model, streams, txt_len, img_len,
             *, strict=True, world=2, pure_ulysses=False):
    harness = _Harness(world, strict=strict)
    harness.expected_tags = [float(tag) for tag in range(1, txt_len + img_len + 1)]
    _install_stubs(monkeypatch, harness)
    return _drive(harness, adapter_cls, model, streams, pure_ulysses), harness


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("txt_len,img_len", [(5, 8), (4, 9)])
def test_the_sharded_render_reproduces_the_unsharded_one_row_for_row(
    monkeypatch, comfy_layers_stub, adapter_cls, txt_len, img_len, world
):
    """The whole claim, on an odd text stream and on an odd image stream.

    (5, 8) pads the text in the double loop and the 13-row joint stream in the
    single loop. (4, 9) is the case these three families meet: their text
    arrives at a floor that divides, and the image row count is odd because
    the canvas made it odd. World 4 is the wider mesh this rig cannot
    run, where a pad can span more than one rank.
    """
    model = _model_for(adapter_cls)
    streams = _streams(txt_len, img_len)
    reference, _ = _reference(monkeypatch, adapter_cls, model, streams,
                              txt_len, img_len)
    outputs, harness = _sharded(monkeypatch, adapter_cls, model, streams,
                                txt_len, img_len, world=world)

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
    for kind, blocks in (("double", model.double_blocks),
                         ("single", model.single_blocks)):
        seen = [call for call in harness.calls if call[1] == kind]
        assert len(seen) == harness.world * len(blocks)
    # Two loops, two drop sets: the double loop names the text pad and the
    # image pad in the order its blocks concatenate them, the single loop names
    # a tail of the joined stream.
    double_drop, single_drop = harness.dispatch[0], harness.dispatch[-1]
    assert double_drop and single_drop
    joint = txt_len + img_len
    assert single_drop == list(range(joint, math.ceil(joint / world) * world))


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
def test_an_even_stream_keeps_the_untouched_xfuser_call(
    monkeypatch, comfy_layers_stub, adapter_cls
):
    """A divisible stream with no order descriptor keeps xfuser's call.

    Nothing pads, so `padded_row_indices` returns [], `base.usp_attention` reads
    `padded == False`, and every call reaches xfuser's own callable rather than
    the exclusion path. Under pure Ulysses the double blocks take the full-axis
    path for key order instead (tests/test_flux_family_order.py).
    """
    model = _model_for(adapter_cls)
    streams = _streams(4, 8)
    reference, _ = _reference(monkeypatch, adapter_cls, model, streams, 4, 8)
    outputs, harness = _sharded(monkeypatch, adapter_cls, model, streams, 4, 8)

    assert _matches(outputs[0], reference)
    assert harness.dispatch and all(drop == [] for drop in harness.dispatch)
    assert {call[0] for call in harness.calls} == {"xfuser"}


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
def test_without_the_drop_sets_the_pad_rows_change_the_answer(
    monkeypatch, comfy_layers_stub, adapter_cls
):
    """The negative control: without it this file would pass on a forward that
    never padded at all. The pads stay, the drop sets are forced empty, and the
    real rows must now differ from the unsharded answer."""
    model = _model_for(adapter_cls)
    streams = _streams(4, 9)
    reference, _ = _reference(monkeypatch, adapter_cls, model, streams, 4, 9)
    monkeypatch.setattr(flux_family, "_family_drop_rows",
                        lambda pad_vouched, segments: [])
    outputs, harness = _sharded(monkeypatch, adapter_cls, model, streams, 4, 9,
                                strict=False)

    # Materially, not in the last bit: a toy whose pad key wins no softmax
    # weight would pass `not torch.equal` on rounding alone and prove nothing.
    assert (outputs[0] - reference).norm() / reference.norm() > 1e-4
    # The pad row reached the kernel, which the exclusion prevents: its tag is
    # shard_seq's own zero.
    assert any(0.0 in tags for _entry, _kind, _index, tags in harness.calls)


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
def test_a_member_that_hands_its_probe_back_refuses_the_same_render(
    monkeypatch, comfy_layers_stub, adapter_cls
):
    """One attribute turns the exclusion on and off.

    A family whose fidelity leg fails sets `usp_pad_exclusion_probe` back to
    None, and the same forward refuses the same odd stream again, so a failed
    leg cannot leave an unvouched render in the tree.
    """
    unvouched = type(f"_Unvouched{adapter_cls.__name__}", (adapter_cls,),
                     {"usp_pad_exclusion_probe": None})
    model = _model_for(adapter_cls)
    with pytest.raises(base.UnsupportedModelError,
                       match=rf"{adapter_cls.family} image-token stream"):
        _sharded(monkeypatch, unvouched, model, _streams(4, 9), 4, 9)
