"""Flux-family adapter logic on CPU: control-add span overlap, exact-type
dispatch, the attention-patch contract and the pad-exclusion probe.

No comfy import at module scope: matches() imports comfy.model_base lazily, so
a fake module stands in for it. The forward tests run the real bound
forward_orig with a fake of its one comfy import
(comfy.ldm.flux.layers.timestep_embedding) and stop at a refusal or right
after the double-block loop (`_StopAfterDoubleBlocks`), so none needs xfuser's
collectives.
"""
import pathlib
import sys
import types

import pytest
import torch

from dgx_monarch.adapters import (
    attention_patches,
    base,
    detect_family,
    flux_family,
    flux_shared,
)
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.flux_family import (
    ChromaAdapter,
    Flux2Adapter,
    FluxAdapter,
    LongCatAdapter,
    _control_span_overlap,
)


def _apply_global(total, span_start, add):
    """Stock semantics: out[span_start:span_start+len(add)] += add."""
    out = [0] * total
    for j, v in enumerate(add):
        out[span_start + j] += v
    return out


def _apply_sharded(world, rows, span_start, add):
    """Every rank applies its overlap to its own window; concat reconstructs."""
    out = [0] * (world * rows)
    for rank in range(world):
        row0 = rank * rows
        ov = _control_span_overlap(row0, rows, span_start, len(add))
        if ov is None:
            continue
        local_lo, local_hi, add_lo, add_hi = ov
        assert local_hi - local_lo == add_hi - add_lo
        for j in range(local_hi - local_lo):
            out[row0 + local_lo + j] += add[add_lo + j]
    return out


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize(
    "span_start,add_len,total",
    [
        (0, 16, 16),    # double loop: add covers the whole img stream
        (0, 10, 16),    # double loop: add shorter than img (Kontext ref tail)
        (7, 9, 16),     # single loop: img span after a 7-token txt block
        (7, 5, 16),     # single loop: shorter add, ends mid-stream
        (13, 3, 16),    # span entirely inside the last rank's window
        (0, 1, 16),     # single-row add on rank 0 only
    ],
)
def test_sharded_adds_reconstruct_stock_global_add(world, span_start, add_len, total):
    assert total % world == 0  # shard_seq pads to a multiple of world or refuses
    rows = total // world
    add = list(range(1, add_len + 1))  # distinct values so misalignment is caught
    assert _apply_sharded(world, rows, span_start, add) == _apply_global(total, span_start, add)


def test_overlap_none_when_rank_holds_only_txt():
    # Single loop, rank 0 entirely inside the txt block: no add applies.
    assert _control_span_overlap(0, 8, span_start=8, span_len=4) is None


def test_overlap_none_when_rank_past_span_end():
    # Last rank holds only rows beyond the add.
    assert _control_span_overlap(12, 4, span_start=0, span_len=10) is None


def test_overlap_bounds_on_straddling_rank():
    # Rank window [4, 8), span [6, 11): overlap is global [6, 8).
    assert _control_span_overlap(4, 4, span_start=6, span_len=5) == (2, 4, 0, 2)


@pytest.fixture
def model_base(monkeypatch):
    """Fake comfy.model_base mirroring the real hierarchy: Flux is the
    isinstance root of Flux2, LongCatImage and Chroma (model_base.py)."""
    mb = types.ModuleType("comfy.model_base")

    class Flux: ...

    class Flux2(Flux): ...

    class LongCatImage(Flux): ...

    class Chroma(Flux): ...

    class ChromaRadiance(Chroma): ...

    class FluxFutureVariant(Flux): ...  # an exotic subclass comfy might grow

    for cls in (Flux, Flux2, LongCatImage, Chroma, ChromaRadiance, FluxFutureVariant):
        setattr(mb, cls.__name__, cls)
    comfy = types.ModuleType("comfy")
    comfy.model_base = mb
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_base", mb)
    return mb


# Registry slice in the required order: most specific first, root FluxAdapter
# last (it owns the typed reject for unclaimed flux subclasses).
def _flux_registry():
    return (ChromaAdapter(), Flux2Adapter(), LongCatAdapter(), FluxAdapter())


def _first_match(model):
    for adapter in _flux_registry():
        if adapter.matches(model):
            return adapter
    return None


def test_exact_flux_binds_root_adapter(model_base):
    # Dev, schnell and inpaint all instantiate the exact model_base.Flux.
    assert type(_first_match(model_base.Flux())) is FluxAdapter


def test_flux2_and_longcat_bind_their_subclass_adapters(model_base):
    assert type(_first_match(model_base.Flux2())) is Flux2Adapter
    assert type(_first_match(model_base.LongCatImage())) is LongCatAdapter


def test_chroma_family_never_reaches_flux_adapter(model_base):
    assert type(_first_match(model_base.Chroma())) is ChromaAdapter
    assert type(_first_match(model_base.ChromaRadiance())) is ChromaAdapter


def test_unknown_flux_subclass_raises_typed(model_base):
    with pytest.raises(UnsupportedModelError, match="FluxFutureVariant"):
        _first_match(model_base.FluxFutureVariant())


def test_detect_family_preserves_unsupported_subclass_refusal(model_base):
    with pytest.raises(UnsupportedModelError, match="FluxFutureVariant"):
        detect_family(model_base.FluxFutureVariant())


def test_store_family_detection_propagates_registry_refusal(model_base):
    from dgx_monarch.actor.store_detect import _detect_family

    assert _detect_family(types.SimpleNamespace(model=model_base.Flux())) == "flux"
    with pytest.raises(UnsupportedModelError, match="FluxFutureVariant"):
        _detect_family(
            types.SimpleNamespace(model=model_base.FluxFutureVariant())
        )


def test_non_flux_model_matches_nothing(model_base):
    class SomethingElse: ...

    assert _first_match(SomethingElse()) is None


def test_subclass_adapters_decline_parent_and_sibling_types(model_base):
    # No raise, no match: exact-type adapters must pass plain Flux (and each
    # other's types) down the registry instead of claiming or rejecting them.
    assert Flux2Adapter().matches(model_base.Flux()) is False
    assert LongCatAdapter().matches(model_base.Flux()) is False
    assert Flux2Adapter().matches(model_base.LongCatImage()) is False


def test_root_adapter_alone_rejects_subclasses(model_base):
    # Direct call, bypassing registry order: the root adapter's allowlist is
    # exact-type, so registry ordering is what routes subclasses correctly.
    with pytest.raises(UnsupportedModelError):
        FluxAdapter().matches(model_base.Flux2())


def test_cfg_cond_padding_declarations():
    # The cfg pad rules (docs/ADAPTERS.md, What an adapter provides): flux and
    # longcat honor an additive key mask (pad+mask); comfy front-pads flux2
    # conds to 512, so zero-pad suffices.
    assert FluxAdapter.cfg_cond_padding == "pad+mask"
    assert Flux2Adapter.cfg_cond_padding == "pad"
    assert LongCatAdapter.cfg_cond_padding == "pad+mask"
    assert (FluxAdapter.family, Flux2Adapter.family, LongCatAdapter.family) == ("flux", "flux2", "longcat")


def _attention_patch(*capabilities):
    def patch(value):
        return value

    setattr(
        patch,
        attention_patches.ATTENTION_PATCH_CAPABILITIES_ATTR,
        frozenset(capabilities),
    )
    return patch


@pytest.mark.parametrize("hook", ["attn1_patch", "attn1_output_patch"])
def test_usp_attention_patch_contract_allows_explicit_sequence_local_hook(hook):
    patch = _attention_patch(attention_patches.SEQUENCE_SHARD_LOCAL_CAPABILITY)
    attention_patches.assert_usp_attention_patches_safe(
        {"patches": {hook: [patch]}}, "flux"
    )


def test_usp_attention_patch_contract_requires_every_installed_hook_to_opt_in():
    marked = _attention_patch(attention_patches.SEQUENCE_SHARD_LOCAL_CAPABILITY)
    unmarked = _attention_patch()
    with pytest.raises(UnsupportedModelError, match="sequence_shard_local"):
        attention_patches.assert_usp_attention_patches_safe(
            {"patches": {"attn1_output_patch": [marked, unmarked]}}, "flux"
        )


@pytest.mark.parametrize(
    "options",
    [
        {"patches": ["not", "a", "mapping"]},
        {
            "patches": _attention_patch(
                attention_patches.SEQUENCE_SHARD_LOCAL_CAPABILITY
            )
        },
        {
            "patches": {
                "attn1_patch": [
                    types.SimpleNamespace(
                        _dgxm_attention_patch_capabilities=frozenset(
                            {attention_patches.SEQUENCE_SHARD_LOCAL_CAPABILITY}
                        )
                    )
                ]
            }
        },
    ],
)
def test_usp_attention_patch_contract_rejects_malformed_patch_surfaces(options):
    with pytest.raises(UnsupportedModelError, match="attn1_patch"):
        attention_patches.assert_usp_attention_patches_safe(options, "flux")


def test_attention_patch_can_declare_both_distributed_locality_contracts():
    patch = _attention_patch(
        attention_patches.SEQUENCE_SHARD_LOCAL_CAPABILITY,
        attention_patches.CONDITION_SHARD_LOCAL_CAPABILITY,
    )
    options = {"patches": {"attn1_output_patch": [patch]}}
    attention_patches.assert_usp_attention_patches_safe(options, "flux")
    attention_patches.assert_cfg_attention_output_patches_safe(options, "flux")


class _FakeFluxGuardOnly:
    """Enough model surface to bind Flux's forward and hit its first guard."""

    def __init__(self):
        self.double_blocks = []
        self.single_blocks = []


class _FakeFluxPadGuard(_FakeFluxGuardOnly):
    """Parent Flux forward surface through its first sequence shard."""

    def __init__(self):
        super().__init__()
        self.img_in = lambda value: value
        self.time_in = lambda value: value
        self.params = types.SimpleNamespace(
            guidance_embed=False,
            vec_in_dim=4,
            global_modulation=False,
        )
        self.vector_in = None
        self.txt_norm = None
        self.txt_in = lambda value: value


@pytest.mark.parametrize("adapter_cls", [FluxAdapter, Flux2Adapter, LongCatAdapter])
@pytest.mark.parametrize(
    "img_len,txt_len,stream",
    [(3, 2, "image-token"), (2, 1, "text-token")],
)
def test_a_flux_family_member_without_a_probe_refuses_padding(
    monkeypatch, sp, chroma_layers_stub, adapter_cls, img_len, txt_len, stream
):
    """The refusal is keyed off the probe attribute, never off the family name.

    All three shipped members name a probe since 2026-09-09 and pad. A member
    that arrives with no measured evidence, or one that hands its probe back
    after a failed leg, keeps the refusal of docs/TROUBLESHOOTING.md #77 on the
    same forward.
    """
    unvouched = type(
        f"_Unvouched{adapter_cls.__name__}", (adapter_cls,),
        {"usp_pad_exclusion_probe": None},
    )
    sp(2, 0)
    model = _FakeFluxPadGuard()
    unvouched().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=None)
    )

    with pytest.raises(
        UnsupportedModelError,
        match=rf"{adapter_cls.family} {stream} stream.*requires exact divisibility",
    ):
        model.forward_orig(
            torch.zeros(1, img_len, 4),
            torch.zeros(1, img_len, 3),
            torch.zeros(1, txt_len, 4),
            torch.zeros(1, txt_len, 3),
            torch.zeros(1),
            torch.zeros(1, 4),
        )


@pytest.mark.parametrize("adapter_cls", [FluxAdapter, Flux2Adapter, LongCatAdapter])
def test_every_shipped_flux_family_member_pads_and_names_one_probe(
    sp, chroma_layers_stub, probe_sp_gather, adapter_cls
):
    """All three members carry the one attribute that makes them pad.

    The three run one forward, so they name one probe file and each member's
    own leg is a case inside it. Flux2 and LongCat inherit the attribute from
    FluxAdapter rather than restating it.
    """
    assert adapter_cls.usp_pad_exclusion_probe == flux_shared.PAD_FIDELITY_PROBE
    sp(2, 0)
    model = _FakeFluxPadGuard()
    adapter_cls().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=None))
    with pytest.raises(_StopAfterDoubleBlocks) as caught:
        model.forward_orig(
            torch.zeros(1, 3, 4), None, torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 3), torch.zeros(1), torch.zeros(1, 4),
        )
    # 3 image rows pad to 4 and halve: the forward shards them, no refusal.
    assert caught.value.img.shape[1] == 2


def test_every_declared_pad_exclusion_probe_names_a_file_in_the_tree():
    """A probe attribute vouches nothing while the file it names is missing.

    The attribute alone decides between padding and refusing, so the evidence
    it names must be readable from the checkout.
    """
    from dgx_monarch.adapters import ADAPTERS

    repo = pathlib.Path(__file__).resolve().parents[1]
    named = {adapter.usp_pad_exclusion_probe for adapter in ADAPTERS
             if adapter.usp_pad_exclusion_probe is not None}
    assert named
    for probe in sorted(named):
        assert (repo / probe).is_file(), probe


@pytest.mark.parametrize("adapter_cls", [FluxAdapter, Flux2Adapter, LongCatAdapter])
@pytest.mark.parametrize("hook", ["attn1_patch", "attn1_output_patch"])
def test_flux_family_rejects_uncontracted_attention_patch_under_usp(
    adapter_cls, hook
):
    model = _FakeFluxGuardOnly()
    adapter_cls().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=None)
    )
    with pytest.raises(UnsupportedModelError, match=hook):
        model.forward_orig(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 3),
            torch.zeros(1),
            torch.zeros(1, 4),
            transformer_options={"patches": {hook: [_attention_patch()]}},
        )


@pytest.fixture
def sp(monkeypatch):
    """Patch base's xfuser rank accessors, which shard_seq consults, plus
    flux_family's own `sp_rank` name: it is imported by value (`from .base
    import sp_rank`), so patching base.sp_rank alone would not reach the direct
    sp_rank() calls the control-add remap makes."""

    def set_rank(world, rank):
        monkeypatch.setattr(base, "sp_world", lambda: world)
        monkeypatch.setattr(base, "sp_rank", lambda: rank)
        monkeypatch.setattr(flux_family, "sp_rank", lambda: rank)

    return set_rank


@pytest.fixture
def chroma_layers_stub(monkeypatch):
    """Stub the one comfy leaf the Chroma and Flux forwards import before the
    double-block loop (comfy.ldm.flux.layers.timestep_embedding)."""
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    flux_pkg = types.ModuleType("comfy.ldm.flux")
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.timestep_embedding = lambda t, dim: torch.zeros(*t.shape, dim)
    comfy.ldm = ldm
    ldm.flux = flux_pkg
    flux_pkg.layers = layers
    for name, mod in [("comfy", comfy), ("comfy.ldm", ldm),
                      ("comfy.ldm.flux", flux_pkg), ("comfy.ldm.flux.layers", layers)]:
        monkeypatch.setitem(sys.modules, name, mod)


class _StopAfterDoubleBlocks(Exception):
    """Raised by the fake sp_gather to capture `img` right after the
    double-block loop, before the single blocks, which would need xfuser's
    collectives."""

    def __init__(self, img):
        self.img = img


@pytest.fixture
def probe_sp_gather(monkeypatch):
    def fake_sp_gather(t, orig_len, dim=1):
        raise _StopAfterDoubleBlocks(t)

    monkeypatch.setattr(flux_family, "sp_gather", fake_sp_gather)


def _identity_double_block(img, txt, vec, pe, attn_mask, transformer_options):
    return img, txt


class _FakeChromaDoubleBlockOnly:
    """Enough of a Chroma diffusion_model to reach the double-block loop's
    control-add through the real bound forward_orig. single_blocks exists only
    for the len(...) in inject_usp's log line; the probe stops the forward
    before the single blocks."""

    def __init__(self):
        self.img_in = lambda x: x
        self.txt_in = lambda x: x
        self.distilled_guidance_layer = lambda x: x
        self.pe_embedder = lambda x: x
        self.get_modulations = lambda *a, **k: None
        self.double_blocks = [_identity_double_block]
        self.skip_mmdit = ()
        self.single_blocks = []


@pytest.mark.parametrize("hook", ["attn1_patch", "attn1_output_patch"])
def test_chroma_rejects_uncontracted_attention_patch_under_usp(hook):
    model = _FakeChromaDoubleBlockOnly()
    ChromaAdapter().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=None)
    )
    with pytest.raises(UnsupportedModelError, match=hook):
        model.forward_orig(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 3),
            torch.zeros(1),
            guidance=torch.zeros(1),
            transformer_options={"patches": {hook: [_attention_patch()]}},
        )


def test_chroma_rejects_effective_attention_mask_before_block_work():
    model = _FakeChromaDoubleBlockOnly()
    ChromaAdapter().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=None)
    )
    with pytest.raises(UnsupportedModelError, match="effective attention mask") as caught:
        model.forward_orig(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 3),
            torch.zeros(1),
            guidance=torch.zeros(1),
            attn_mask=torch.tensor([[0.0, -1e4]]),
        )
    message = str(caught.value)
    assert "mode=local" in message and "gpus_per_host=1" in message
    assert "topology 'single'" not in message


class _RecordingBlock:
    """A double block that keeps the options dict the loop handed it."""

    def __init__(self):
        self.options = None

    def __call__(self, img, txt, vec, pe, attn_mask, transformer_options):
        self.options = transformer_options
        return img, txt


def _drop_rows_the_double_loop_baked(model, adapter, img_len, txt_len):
    """Call the installed override and return the row set it carries."""
    seen = []

    def usp_attention(*args, drop_rows=None, **kwargs):
        seen.append(drop_rows)
        return args[0]

    adapter.inject_usp(model, InjectionContext(topology_sp=2, usp_attention=usp_attention))
    with pytest.raises(_StopAfterDoubleBlocks):
        model.forward_orig(
            torch.zeros(1, img_len, 4),
            torch.zeros(1, img_len, 3),
            torch.zeros(1, txt_len, 4),
            torch.zeros(1, txt_len, 3),
            torch.zeros(1),
            guidance=torch.zeros(1),
        )
    block = model.double_blocks[0]
    block.options["optimized_attention_override"](None, torch.zeros(1, 2, 4), heads=1)
    return seen


def test_chroma_pads_an_odd_stream_and_names_its_pad_rows(
    sp, chroma_layers_stub, probe_sp_gather
):
    """Chroma names a pad-exclusion probe, so an odd stream shards.

    The drop set is hand-computed: 3 text and 3 image rows over 2 ranks pad to
    4 each, so the gathered layout is [txt_r0(0,1), img_r0(2,3), txt_r1(4,5),
    img_r1(6,7)] and the two pads sit at 5 and 7. Equal segments name the same
    set in either order; tests/test_chroma_pad_exclusion.py pins text first.
    """
    sp(2, 0)
    model = _FakeChromaDoubleBlockOnly()
    model.double_blocks = [_RecordingBlock()]
    assert _drop_rows_the_double_loop_baked(model, ChromaAdapter(), 3, 3) == [[5, 7]]


def test_chroma_shards_an_even_stream_with_nothing_to_drop(
    sp, chroma_layers_stub, probe_sp_gather
):
    # An even stream has no pad row, so the override carries an empty set. With
    # no order descriptor (this context is not pure Ulysses), base.usp_attention
    # keeps xfuser's call; under pure Ulysses the double blocks take the full-axis
    # path (test_chroma_double_order_descriptor_is_pure_ulysses_only).
    sp(2, 0)
    model = _FakeChromaDoubleBlockOnly()
    model.double_blocks = [_RecordingBlock()]
    assert _drop_rows_the_double_loop_baked(model, ChromaAdapter(), 4, 2) == [[]]


@pytest.mark.parametrize("pure_ulysses,expected", [(True, "RankMajorJointOrder"), (False, None)])
def test_chroma_double_order_descriptor_is_pure_ulysses_only(
    sp, chroma_layers_stub, probe_sp_gather, pure_ulysses, expected
):
    sp(2, 0)
    model = _FakeChromaDoubleBlockOnly()
    model.double_blocks = [_RecordingBlock()]
    seen = []

    def usp_attention(*args, sequence_order=None, **kwargs):
        seen.append(None if sequence_order is None else type(sequence_order).__name__)
        return args[0]

    ChromaAdapter().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=usp_attention,
                                pure_ulysses=pure_ulysses))
    with pytest.raises(_StopAfterDoubleBlocks):
        model.forward_orig(
            torch.zeros(1, 4, 4), torch.zeros(1, 4, 3),
            torch.zeros(1, 2, 4), torch.zeros(1, 2, 3),
            torch.zeros(1), guidance=torch.zeros(1),
        )
    model.double_blocks[0].options["optimized_attention_override"](
        None, torch.zeros(1, 2, 4), heads=1)
    assert seen == [expected]


@pytest.mark.parametrize(
    "img_len,txt_len,stream",
    [(3, 2, "image-token"), (2, 1, "text-token")],
)
def test_a_family_without_a_probe_still_refuses_divisibility_padding(
    sp, chroma_layers_stub, img_len, txt_len, stream
):
    """The refusal is keyed off the probe attribute, never off the family name.

    A family that arrives without measured pad-exclusion evidence keeps the
    refusal of docs/TROUBLESHOOTING.md #77, whatever forward it borrows.
    """

    class _UnvouchedChroma(ChromaAdapter):
        usp_pad_exclusion_probe = None

    sp(2, 0)
    model = _FakeChromaDoubleBlockOnly()
    _UnvouchedChroma().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=None)
    )
    with pytest.raises(
        UnsupportedModelError,
        match=rf"chroma {stream} stream.*requires exact divisibility",
    ):
        model.forward_orig(
            torch.zeros(1, img_len, 4),
            torch.zeros(1, img_len, 3),
            torch.zeros(1, txt_len, 4),
            torch.zeros(1, txt_len, 3),
            torch.zeros(1),
            guidance=torch.zeros(1),
        )


def _double_block_img_after_control(add):
    model = _FakeChromaDoubleBlockOnly()
    ChromaAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=None))
    img = torch.arange(6).float().view(1, 6, 1)
    img_ids = torch.zeros(1, 6, 3)
    txt = torch.zeros(1, 2, 1)
    txt_ids = torch.zeros(1, 2, 3)
    with pytest.raises(_StopAfterDoubleBlocks) as exc_info:
        model.forward_orig(img, img_ids, txt, txt_ids, torch.zeros(1),
                           guidance=torch.zeros(1), control={"input": [add]})
    return exc_info.value.img[0, :, 0]


@pytest.mark.parametrize(
    "rank,expected",
    [(0, [100.0, 201.0, 2.0]), (1, [3.0, 4.0, 5.0])],
)
def test_chroma_double_block_control_shorter_than_stream(
    sp, chroma_layers_stub, probe_sp_gather, rank, expected
):
    # add covers only global rows [0, 2), shorter than the 6-row stream (the
    # Kontext ref-tail case in _control_span_overlap's docstring). Rank 0's
    # window [0, 3) overlaps rows [0, 2); rank 1's window [3, 6) does not
    # overlap at all and must be left untouched. Re-sharding the 2-row add and
    # adding the shard to the window would broadcast one value across all 3
    # of rank 1's rows, and nothing would raise.
    sp(2, rank)
    add = torch.tensor([[[100.0], [200.0]]])
    result = _double_block_img_after_control(add)
    assert torch.equal(result, torch.tensor(expected))


@pytest.mark.parametrize(
    "rank,expected",
    [(0, [100.0, 201.0, 302.0]), (1, [403.0, 504.0, 605.0])],
)
def test_chroma_double_block_control_full_span_matches_stock(
    sp, chroma_layers_stub, probe_sp_gather, rank, expected
):
    # add spans the full 6-row stream, the common case: the per-rank overlap
    # must equal stock's img[:, :add.shape[1]] += add exactly.
    sp(2, rank)
    add = torch.arange(100, 700, 100).float().view(1, 6, 1)
    result = _double_block_img_after_control(add)
    assert torch.equal(result, torch.tensor(expected))
