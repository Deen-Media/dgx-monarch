"""CPU contracts for the minimax_h3 adapter; no ComfyUI, xFuser or hardware.

Coverage includes registry/detection guards, shard and pad math, modulation and
timestep layouts, attention and output-head calls, injection, typed refusals,
topology remedies and frame_count compatibility. The exact-class allowlist
rejects unknown subclasses. H3's hidden stream shards on dim 0; using dim 1
would silently change the result.
"""
import ast
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch.adapters import ADAPTERS, base, get_adapter
from dgx_monarch.adapters import minimax_h3 as h3
from dgx_monarch.adapters import minimax_h3_packing as h3_packing
from dgx_monarch.adapters.base import (
    InjectionContext,
    UnsupportedModelError,
    assert_ulysses_only_padding,
    padded_row_indices,
    usp_options,
)
from dgx_monarch.adapters.minimax_h3 import (
    MINIMAX_H3_BATCH_CAPPED_MESSAGE,
    MiniMaxH3Adapter,
    minimax_h3_topology_would_reject,
)
from dgx_monarch.adapters.minimax_h3_packing import (
    h3_modulation_plan,
    h3_timestep_embedding,
    pad_row_index,
    row_runs,
)
from dgx_monarch.comfy_forward_contracts import REWRITTEN_FORWARD_CALLEE_TOUCHPOINTS
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from fake_model_base_helpers import install_fake_model_base


@pytest.fixture
def sp(monkeypatch):
    """Patch the xfuser rank accessors shard_seq consults.

    Both the base module and the adapter module are patched: the adapter
    imports `sp_world` by value, so patching base alone would not reach the
    forward's own padded-length arithmetic."""

    def set_rank(world, rank):
        monkeypatch.setattr(base, "sp_world", lambda: world)
        monkeypatch.setattr(base, "sp_rank", lambda: rank)
        monkeypatch.setattr(h3, "sp_world", lambda: world)

    return set_rank


@pytest.fixture
def fake_model_base(monkeypatch):
    """A minimal fake comfy.model_base holding only MiniMaxH3 and an unvetted
    subclass. Safe to run through the whole registry: every other adapter's
    exact-type raise sits behind its own isinstance gate, which returns False
    when its class name is absent from this module."""
    return install_fake_model_base(
        monkeypatch, {"MiniMaxH3": None, "MiniMaxH3Exotic": "MiniMaxH3"}
    )


def test_adapter_contract_declared():
    adapter = MiniMaxH3Adapter()
    assert adapter.family == "minimax_h3"
    assert adapter.model_base_classes == ("MiniMaxH3",)
    assert adapter.exact_model_base_classes == ("MiniMaxH3",)
    assert adapter.cfg_cond_padding == "none"
    # dp is structurally refused, so the generic dp slicer is unreachable and
    # the exempt-key registry invariant must keep seeing an empty set here.
    assert adapter.dp_cond_exempt_keys == frozenset()


def test_adapter_defines_no_cfg_pad_forward():
    # The worker installs a cfg-pad forward only when the adapter defines one.
    # H3 must not: cfg-parallel never gets a batched call to split.
    assert not hasattr(MiniMaxH3Adapter, "inject_cfg_pad_forward")


# The matches() trio (exact-type accept, typed reject on an unclaimed
# subclass, foreign-model decline) runs from the shared table in
# tests/test_adapter_matches.py.
def test_registered_at_most_once():
    assert sum(isinstance(a, MiniMaxH3Adapter) for a in ADAPTERS) <= 1


def test_no_other_adapter_claims_minimax_h3(fake_model_base):
    model = fake_model_base.MiniMaxH3()
    for adapter in ADAPTERS:
        if isinstance(adapter, MiniMaxH3Adapter):
            continue
        assert adapter.matches(model) is False


def test_registry_dispatch_is_correct_before_and_after_the_row_lands(fake_model_base):
    model = fake_model_base.MiniMaxH3()
    if any(isinstance(a, MiniMaxH3Adapter) for a in ADAPTERS):
        assert type(get_adapter(model)) is MiniMaxH3Adapter
    else:
        with pytest.raises(UnsupportedModelError, match="no dgx-monarch adapter"):
            get_adapter(model)


def test_state_dict_keys_never_mis_detect_as_another_family():
    # H3's checkpoint keys must not match another family's signature. The
    # closest near-miss is the top-level `token_refiner.` stack against
    # hunyuan's `txt_in.individual_token_refiner.`; `condition_proj.` and the
    # bare `blocks.0.attn.qkv_proj.` are the wan near-misses. Keys carry the
    # `model.diffusion_model.` prefix so the prefix stripper is exercised too.
    # Either answer passes: the test guards only against another family's name.
    from dgx_monarch.adapters.detect import detect_family_from_keys

    keys = [
        "model.diffusion_model.video_patch_proj.weight",
        "model.diffusion_model.audio_patch_proj.weight",
        "model.diffusion_model.condition_proj.weight",
        "model.diffusion_model.rope.inv_freq",
        "model.diffusion_model.token_refiner.blocks.0.attn.qkv_proj.weight",
        "model.diffusion_model.token_refiner.final_norm.weight",
        "model.diffusion_model.blocks.0.attn.qkv_proj.weight",
        "model.diffusion_model.blocks.0.adaln_proj.linear.weight",
        "model.diffusion_model.blocks.0.mlp.fc1.weight",
        "model.diffusion_model.final_layer.video_out.weight",
        "model.diffusion_model.final_layer.audio_out.weight",
    ]
    assert detect_family_from_keys(keys) in ("unknown", "minimax_h3")


def _packed_fixture(seq_len=7, hidden=3):
    """A tiny stand-in for the packed sequence and its aligned per-token
    tensors: [text(0:2) | audio(2:5) | video(5:7)]."""
    stream = torch.arange(seq_len * hidden, dtype=torch.float32).reshape(seq_len, hidden)
    position_ids = torch.arange(seq_len * 3, dtype=torch.float64).reshape(seq_len, 3)
    rows = torch.tensor([4, 4, 1, 1, 1, 0, 0], dtype=torch.long)
    return stream, position_ids, rows


def test_odd_packed_total_shards_and_pads_every_per_token_tensor_identically(sp):
    stream, position_ids, rows = _packed_fixture()
    seq_len, world = 7, 2
    padded_len = -(-seq_len // world) * world
    assert padded_len == 8
    row_index = pad_row_index(rows, padded_len)

    streams, positions, rows = [], [], []
    for rank in (0, 1):
        sp(world, rank)
        s_local, s_orig = base.shard_seq(stream, dim=0)
        p_local, _ = base.shard_seq(position_ids, dim=0)
        r_local, _ = base.shard_seq(row_index, dim=0)
        # every aligned tensor keeps the same rows on this rank
        assert s_local.shape[0] == p_local.shape[0] == r_local.shape[0] == 4
        assert s_orig == seq_len
        streams.append(s_local)
        positions.append(p_local)
        rows.append(r_local)

    # Reassembly: rank order, then the pad tail trims back to the real length.
    assert torch.equal(torch.cat(streams, dim=0).narrow(0, 0, seq_len), stream)
    assert torch.equal(torch.cat(positions, dim=0).narrow(0, 0, seq_len), position_ids)
    assert torch.equal(torch.cat(rows, dim=0).narrow(0, 0, seq_len), row_index[:seq_len])

    # The pad row is the last row of the last rank on every aligned tensor.
    assert float(streams[1][-1].abs().sum()) == 0.0
    assert float(positions[1][-1].abs().sum()) == 0.0


def test_position_id_shard_is_what_rope_must_be_built_from(sp):
    # Shard the ids, then embed locally. Chunking an already embedded full
    # table is only equal on real rows, and differs on the pad row (identity
    # rotation from position (0,0,0) vs an all-zero, non-rotation block).
    _, position_ids, _ = _packed_fixture()
    sp(2, 1)
    p_local, _ = base.shard_seq(position_ids, dim=0)
    # rank 1 holds global rows 4..6 plus the zero pad row
    assert torch.equal(p_local[:3], position_ids[4:7])
    assert float(p_local[3].abs().sum()) == 0.0


def test_pad_row_index_pads_with_the_final_target_video_row():
    _, _, rows = _packed_fixture()
    padded = pad_row_index(rows, 8)
    assert padded.dtype == torch.long
    assert padded.tolist() == [4, 4, 1, 1, 1, 0, 0, 0]
    # The last packed segment is always the target video segment, so the pad
    # row inherits the video modulation row and the local table stays contiguous.
    assert int(padded[7]) == int(rows[-1])


def test_pad_row_index_is_a_noop_when_nothing_is_padded():
    _, _, rows = _packed_fixture()
    assert pad_row_index(rows, 7).tolist() == [4, 4, 1, 1, 1, 0, 0]


def test_row_runs_rebuild_a_contiguous_local_table(sp):
    _, _, rows = _packed_fixture()
    row_index = pad_row_index(rows, 8)
    covered = []
    for rank in (0, 1):
        sp(2, rank)
        r_local, _ = base.shard_seq(row_index, dim=0)
        runs = row_runs(r_local)
        # contiguous cover of exactly this rank's rows, in order
        assert runs[0][0] == 0
        assert runs[-1][1] == r_local.shape[0]
        assert all(runs[i][1] == runs[i + 1][0] for i in range(len(runs) - 1))
        covered.extend(int(r_local[a]) for a, b, _row in runs for _ in range(b - a))
    assert covered == row_index.tolist()


def test_row_runs_merge_adjacent_equal_rows():
    # Merging two adjacent segments that share a modulation row is elementwise
    # identical: modulation scales, shifts and gates each row by its own vector.
    assert row_runs(torch.tensor([1, 1, 1, 1])) == [(0, 4, 1)]
    assert row_runs(torch.tensor([2, 2, 5])) == [(0, 2, 2), (2, 3, 5)]
    assert row_runs(torch.zeros(0, dtype=torch.long)) == []


def test_padded_row_indices_names_the_tail_pad_rows_at_world_2(sp):
    sp(2, 0)
    # one contiguous packed segment: chunk == local and offset == 0, so the
    # gathered pad rows are exactly the tail of the padded sequence.
    assert padded_row_indices([(7, 4)]) == [7]
    assert padded_row_indices([(5, 3)]) == [5]
    assert padded_row_indices([(8, 4)]) == []   # even total: nothing to drop


def test_ring_refuses_a_padded_packed_sequence():
    # ulysses excludes the pad rows exactly; ring has no full-sequence point at
    # which to do it, so a padded H3 sequence on ring/hybrid must refuse.
    with pytest.raises(UnsupportedModelError, match="ulysses-only"):
        assert_ulysses_only_padding(2, 1)
    assert_ulysses_only_padding(1, 1)   # pure ulysses: fine
    assert_ulysses_only_padding(2, 0)   # unpadded on ring: fine


class _Layout:
    def __init__(self, segments):
        self.segments = segments
        self.seq_len = segments[-1][1]


_T2VA = _Layout([(0, 3, "text"), (3, 7, "audio"), (7, 15, "video")])


def test_modulation_plan_handles_a_single_unique_timestep():
    # Step 0 of a plain text-to-video render: sigma_max is exactly 1.0, so
    # t_video == t_audio == 0.0 and M is 1. An adapter that assumed M >= 2
    # would fail on the default path's first step.
    unique_t, rows, video_row, audio_row = h3_modulation_plan(
        _T2VA, {}, 0.0, 0.0, 0.999, 1.0)
    assert unique_t == [0.0]
    assert rows.dtype == torch.long
    assert row_runs(rows) == [(0, 3, 1), (3, 7, 2), (7, 15, 0)]
    assert (video_row, audio_row) == (0, 0)


def test_modulation_plan_splits_rows_by_stream_time():
    unique_t, rows, video_row, audio_row = h3_modulation_plan(
        _T2VA, {}, 0.2, 0.5, 0.999, 1.0)
    assert unique_t == [0.2, 0.5]
    # DiT rows are t_row * 3 + modality tag (text 1, video 0, audio 2).
    assert row_runs(rows) == [(0, 3, 1), (3, 7, 5), (7, 15, 0)]
    # The final layer's rows are plain t_row, no `* 3`: its adaLN projection
    # declares a single modality.
    assert (video_row, audio_row) == (0, 1)


def test_modulation_plan_pins_condition_rows_near_one():
    layout = _Layout([(0, 2, "text"), (2, 6, "cond"), (6, 10, "audio"), (10, 18, "video")])
    unique_t, rows, video_row, audio_row = h3_modulation_plan(
        layout, {}, 0.2, 0.5, 0.999, 1.0)
    assert unique_t == [0.2, 0.5, 0.999]
    assert row_runs(rows) == [(0, 2, 1), (2, 6, 6), (6, 10, 5), (10, 18, 0)]
    assert (video_row, audio_row) == (0, 1)


def test_modulation_plan_tags_the_text_span_per_token():
    layout = _Layout([(0, 4, "text"), (4, 6, "audio"), (6, 10, "video")])
    payload = {"text_token_tags": torch.tensor([1, 1, 0, 0])}
    _unique_t, rows, _v, _a = h3_modulation_plan(layout, payload, 0.0, 0.0, 0.999, 1.0)
    # vision-pad positions inside the prompt carry the video modality (tag 0).
    assert rows[:4].tolist() == [1, 1, 0, 0]
    assert row_runs(rows)[:2] == [(0, 2, 1), (2, 4, 0)]


def test_modulation_plan_refuses_an_unvetted_segment_kind():
    # The single vetting point for segment kinds: a future comfy kind must
    # refuse with the family's typed error, never fall through into a stream.
    layout = _Layout([(0, 2, "text"), (2, 6, "haptics"), (6, 10, "video")])
    with pytest.raises(UnsupportedModelError, match="haptics"):
        h3_modulation_plan(layout, {}, 0.0, 0.0, 0.999, 1.0)


def test_unvetted_segment_kind_refusal_carries_the_class_p_tag():
    """The layout is rebuilt from the request on every rank, so this arm fires
    identically everywhere before any collective and packs no latent. The tag
    retires the sample lease CONSUMED, so acting on the refusal and re-queueing
    works on the same fleet with no recycle (docs/TROUBLESHOOTING.md #68)."""
    layout = _Layout([(0, 2, "text"), (2, 6, "haptics"), (6, 10, "video")])
    with pytest.raises(UnsupportedModelError) as caught:
        h3_modulation_plan(layout, {}, 0.0, 0.0, 0.999, 1.0)
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None, "an unvetted segment kind must carry a class tag"
    assert tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False


def test_anchored_audio_guide_rides_the_audio_condition_schedule():
    """An anchored guide's soundtrack packs as `cond_audio`, which comfy gives
    the audio modality tag and the audio condition time. Comfy e01fb4c56
    introduced the kind."""
    layout = _Layout([(0, 2, "text"), (2, 6, "cond"), (6, 12, "cond_audio"),
                      (12, 16, "audio"), (16, 24, "video")])
    unique_t, rows, video_row, audio_row = h3_modulation_plan(
        layout, {}, 0.2, 0.5, 0.999, 1.0)
    # the guide's frames pin near one on the video augmentation time, its
    # soundtrack on the audio one, and the two are distinct rows.
    assert unique_t == [0.2, 0.5, 0.999, 1.0]
    assert row_runs(rows) == [(0, 2, 1), (2, 6, 6), (6, 12, 11),
                              (12, 16, 5), (16, 24, 0)]
    assert (video_row, audio_row) == (0, 1)


def test_every_vetted_segment_kind_has_a_replicated_row_source():
    """The two maps must not drift: a kind the modulation plan admits but the
    assembly loop cannot source would raise KeyError mid-forward instead of
    refusing, and a kind sourced but never vetted is an entry nothing reads,
    because the plan refuses it before the loop reaches it. Text is assembled
    from the encoder output, not a row stream, so it is the one kind the stream
    map leaves out."""
    assert set(h3.H3_STREAM_OF) | {"text"} == set(h3_packing.H3_SEG_MODALITY)
    assert set(h3.H3_STREAM_OF.values()) == {"video", "audio"}


def test_modulation_plan_honours_payload_noise_aug_overrides():
    layout = _Layout([(0, 2, "text"), (2, 6, "ref_audio"), (6, 10, "audio"), (10, 18, "video")])
    unique_t, _mod, _v, _a = h3_modulation_plan(
        layout, {"audio_cond_noise_aug": 1.0}, 0.2, 0.5, 0.999, 1.0)
    assert unique_t == [0.2, 0.5, 1.0]


def _fake_comfy_model_management(monkeypatch):
    mm = types.ModuleType("comfy.model_management")
    mm.cast_to = lambda t, device=None: t
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)


def test_timestep_embedding_curve_regime_stays_fp32(monkeypatch):
    _fake_comfy_model_management(monkeypatch)
    model = types.SimpleNamespace(
        use_adaln_curves=True,
        adaln_t_table=torch.tensor([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]),
    )
    out = h3_timestep_embedding(model, torch.tensor([0.0, 0.5, 1.0]), torch.bfloat16)
    # t = 1.0 must clamp onto the last interval, not read past the table.
    assert torch.equal(out, torch.tensor([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]))
    # Curve checkpoints hold fp32 adaLN linears: casting here would change math.
    assert out.dtype == torch.float32


def test_timestep_embedding_curve_regime_without_a_table_raises():
    model = types.SimpleNamespace(use_adaln_curves=True)
    with pytest.raises(UnsupportedModelError, match="adaln_t_table"):
        h3_timestep_embedding(model, torch.tensor([0.0]), torch.bfloat16)


def test_timestep_embedding_mlp_regime_casts_to_compute_dtype():
    model = types.SimpleNamespace(
        use_adaln_curves=False,
        time_embedder=lambda t: t.unsqueeze(1).repeat(1, 2),
    )
    out = h3_timestep_embedding(model, torch.tensor([0.25, 0.75]), torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 2)


def test_timestep_embedding_matching_neither_regime_raises():
    with pytest.raises(UnsupportedModelError, match="neither"):
        h3_timestep_embedding(types.SimpleNamespace(), torch.tensor([0.0]), torch.bfloat16)


def test_override_threads_drop_rows_through_h3s_exact_call_shape():
    # H3 calls optimized_attention(q, k, v, self.heads, mask=None,
    # skip_reshape=True, transformer_options=...); comfy c194dd00 adds a
    # preferred_attention keyword, which wrap_attn pops. wrap_attn then hands
    # the override (func, *args, **kwargs) with _inside_attn_wrapper injected.
    seen = {}

    def fake_attn(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "attn-out"

    opts = usp_options({"stock": 1}, fake_attn, drop_rows=[7])
    assert opts["stock"] == 1
    q, k, v = object(), object(), object()
    out = opts["optimized_attention_override"](
        object(), q, k, v, 56, mask=None, skip_reshape=True,
        transformer_options={"stock": 1}, _inside_attn_wrapper=True)
    assert out == "attn-out"
    assert seen["args"] == (q, k, v, 56)
    assert seen["kwargs"] == {"mask": None, "skip_reshape": True, "drop_rows": [7]}


def test_unpadded_sequence_passes_an_empty_drop_rows(sp):
    # An even packed total must take the plain xfuser path unchanged: an
    # empty list is falsy inside usp_attention.
    sp(2, 0)
    seen = {}

    def fake_attn(*args, **kwargs):
        seen.update(kwargs)
        return "attn-out"

    drop = padded_row_indices([(8, 4)])
    opts = usp_options({}, fake_attn, drop_rows=drop)
    opts["optimized_attention_override"](object(), 1, 2, 3, 56, mask=None, skip_reshape=True)
    assert seen["drop_rows"] == []
    assert not seen["drop_rows"]


def test_final_layer_call_matches_the_pinned_positional_shape():
    """The pinned FinalLayer row and the single production call must agree.

    Comfy 2504e68d made the current sigma, the sampler's whole sigma table and
    the two flow shifts required on `FinalLayer.forward`: a PDD LoRA stacks
    head row blocks and the head blends the ones the step spans. All three
    values are the sampler's, so the rebound forward has to hand them down.
    Repinning the row without moving the call, or the reverse, leaves a render
    raising a TypeError that no other CPU test would see. The `minimax h3 pdd
    head` seam, which needs comfy, checks what those arguments do; this half
    runs everywhere.
    """
    row = next(
        touchpoint
        for touchpoint in REWRITTEN_FORWARD_CALLEE_TOUCHPOINTS
        if touchpoint.path == "comfy.ldm.minimax.model.FinalLayer.forward"
    )
    assert row.positional == (
        "self", "x", "t_emb", "video_seg", "audio_seg",
        "sigma", "sample_sigmas", "shifts",
    )
    calls = [
        node
        for node in ast.walk(ast.parse(Path(h3.__file__).read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "final_layer"
    ]
    assert len(calls) == 1
    # `self` is the bound receiver; the call carries every other name in the row.
    assert not calls[0].keywords
    assert len(calls[0].args) == len(row.positional) - 1
    # The table is the sampler's own, read off the stock transformer_options
    # rather than the sequence-parallel copy the block loop runs under.
    assert [ast.unparse(arg) for arg in calls[0].args] == [
        "h", "t_emb", "video_seg", "audio_seg", "sigma_v",
        "transformer_options.get('sample_sigmas')", "(shift_v, shift_a)",
    ]


_STOCK = object()


class _FakeH3:
    def __init__(self, blocks=3):
        self.blocks = [object() for _ in range(blocks)]
        self.forward = _STOCK
        self._forward = _STOCK
        self.token_refiner = _STOCK
        self.final_layer = _STOCK


def test_inject_usp_binds_only_the_inner_forward():
    model = _FakeH3()
    MiniMaxH3Adapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    assert model._forward is not _STOCK      # the sharded forward
    assert model.forward is _STOCK           # stock wrapper-executor shell
    # No attention module is patched: H3 threads transformer_options into
    # comfy's dispatcher, so the usp_options override reaches every block.
    assert model.token_refiner is _STOCK
    assert model.final_layer is _STOCK


def test_inject_usp_is_idempotent():
    model = _FakeH3()
    adapter = MiniMaxH3Adapter()
    ctx = InjectionContext(topology_sp=2, usp_attention=object())
    adapter.inject_usp(model, ctx)
    first = model._forward
    adapter.inject_usp(model, ctx)
    assert model._forward is not first       # rebound
    assert model._forward.__func__.__name__ == first.__func__.__name__


def _current_outer_forward(
    self, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs
):
    """Stand-in outer forward holding the constants and names that
    `_has_current_audio_carry_outer` fingerprints, for the no-comfy tests."""
    transformer_options = transformer_options or {}
    scale = float((minimax_payload or {}).get("audio_scale", 1.0))
    shift_v = transformer_options.get("minimax_h3_sigma_shift_video", 12.0)
    shift_a = transformer_options.get("minimax_h3_sigma_shift_audio", 3.0)
    if scale != 1.0:
        time_shift_sigma(timestep, shift_v, shift_a)
    return self._forward(
        x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs
    )


def time_shift_sigma(sigma, _from_shift, _to_shift):
    return sigma


def _bound_forward():
    model = types.SimpleNamespace(blocks=[], forward=None)
    model.forward = types.MethodType(_current_outer_forward, model)
    MiniMaxH3Adapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    return model


def _av_latents(batch=1):
    return [torch.zeros(batch, 24, 1, 4, 4), torch.zeros(batch, 32, 2, 8)]


def test_forward_refuses_a_missing_payload():
    # Stock always publishes at least a seed, so an absent payload is a
    # transport failure that would render a plausible but wrong t2va clip.
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError, match="minimax_payload"):
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), {})


@pytest.mark.parametrize("payload", [None, {}])
def test_forward_refuses_an_empty_payload_too(payload):
    # comfy writes payload["seed"] unconditionally, so a keyless payload is
    # the same wire failure an absent one is. Letting it through would build a
    # bare t2va layout and silently drop the keyframes or reference blocks.
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError, match="minimax_payload"):
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), {},
                       minimax_payload=payload)


def test_forward_refuses_payload_without_current_audio_carry_marker():
    # Pre-bdcb886a comfy publishes the seed but not audio_scale, and its outer
    # forward does not convert this adapter's raw audio velocity back to the
    # sampler-carried variable. Running that pairing would silently integrate
    # the wrong audio ODE, so the marker is a required compatibility contract.
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError, match="audio_scale"):
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), {},
                       minimax_payload={"seed": 0})


@pytest.mark.parametrize(
    "audio_scale",
    [
        None,
        "4.0",
        True,
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        pytest.param(10**10000, id="overflowing-int"),
    ],
)
def test_forward_refuses_invalid_audio_carry_marker(audio_scale):
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError, match="audio_scale"):
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), {},
                       minimax_payload={"seed": 0, "audio_scale": audio_scale})


def test_forward_refuses_a_real_whose_float_conversion_raises():
    class ExplosiveReal:
        def __float__(self):
            raise RuntimeError("conversion must stay behind the typed gate")

    h3.Real.register(ExplosiveReal)
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError, match="audio_scale"):
        model._forward(
            _av_latents(),
            torch.zeros(1),
            torch.zeros(1, 4, 8),
            {},
            minimax_payload={"seed": 0, "audio_scale": ExplosiveReal()},
        )


def test_forward_refuses_new_payload_marker_around_a_legacy_outer_method():
    def legacy_outer(
        self, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs
    ):
        return self._forward(
            x,
            timestep,
            context,
            transformer_options or {},
            minimax_payload=minimax_payload,
            **kwargs,
        )

    model = _bound_forward()
    model.forward = types.MethodType(legacy_outer, model)
    with pytest.raises(UnsupportedModelError, match="partial or mixed ComfyUI update") as caught:
        model.forward(
            _av_latents(),
            torch.zeros(1),
            torch.zeros(1, 4, 8),
            {},
            minimax_payload={"seed": 0, "audio_scale": 4.0},
        )
    # Host-decided arm: on a mixed fleet the sibling rank may already sit in a
    # collective, so this refusal carries no class tag and retires its lease
    # ABANDONED, one Recycle before the next dispatch
    # (docs/TROUBLESHOOTING.md #68; hardware-measured 2026-08-07).
    assert parse_refusal_tag(str(caught.value)) is None


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(None, id="absent-payload"),
        pytest.param({"seed": 0}, id="missing-marker"),
        pytest.param({"seed": 0, "audio_scale": -1.0}, id="malformed-marker"),
    ],
)
def test_request_decided_payload_refusals_carry_the_class_p_tag(payload):
    """The three payload arms are decided from the request, fire identically
    on every rank before any collective, and pack no latent, so they carry the
    class-P tag and retire their sample lease CONSUMED
    (nodes/pending.typed_worker_refusal): the operator re-queues on the same
    fleet with no recycle (docs/TROUBLESHOOTING.md #68)."""
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError) as caught:
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), {},
                       minimax_payload=payload)
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None, "payload refusal must carry a refusal-class tag"
    assert tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False


def test_forward_refuses_batch_greater_than_one():
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError, match="batch 2"):
        model._forward(_av_latents(batch=2), torch.zeros(1), torch.zeros(1, 4, 8), {},
                       minimax_payload={"seed": 0, "audio_scale": 4.0})


@pytest.mark.parametrize("name", ["attention_mask", "attn_mask"])
def test_forward_refuses_an_attention_mask(name):
    # H3's stock attention is maskless and fully bidirectional; the sharded
    # kernel has no way to apply a mask, and dropping one silently is worse.
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError, match=name):
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), {},
                       minimax_payload={"seed": 0, "audio_scale": 4.0},
                       **{name: torch.ones(1, 4)})


@pytest.mark.parametrize("name", ["denoise_mask", "audio_denoise_mask"])
def test_forward_refuses_a_latent_noise_mask(name):
    # Comfy ff6c8a8 answers a mask by giving masked rows their own timestep,
    # which makes the modulation row a per-token index tensor. This forward
    # carries one row per packed segment, so a dropped mask would renoise the
    # rows the operator preserved.
    model = _bound_forward()
    with pytest.raises(UnsupportedModelError, match=name) as caught:
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), {},
                       minimax_payload={"seed": 0, "audio_scale": 4.0},
                       **{name: torch.ones(1, 1, 1, 4, 4)})
    # Request-decided and rank-symmetric: every rank refuses before the first
    # collective, so the tag retires the sample lease CONSUMED and the operator
    # re-queues on the same fleet.
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False
    assert "mode=local" in str(caught.value)


@pytest.mark.parametrize("name", ["denoise_mask", "audio_denoise_mask"])
def test_forward_accepts_an_unset_latent_noise_mask(name):
    # Comfy before ff6c8a8 never sends these; from it on, every unmasked
    # render sends None, and that is nearly every render.
    model = _bound_forward()
    try:
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), {},
                       minimax_payload={"seed": 0, "audio_scale": 4.0},
                       **{name: None})
    except UnsupportedModelError as exc:
        pytest.fail(f"an unset {name} must not trip a refusal: {exc}")
    except Exception:
        # Anything else is the comfy runtime this CPU-only test does not have.
        pass


def test_the_mask_parameters_are_named_rather_than_absorbed():
    """The rebound forward must declare what the newer stock forward passes.

    Absorbed into ``**kwargs``, a mask would be dropped without a refusal.
    """
    model = _bound_forward()
    parameters = inspect.signature(model._forward).parameters
    assert ("denoise_mask", "audio_denoise_mask") == tuple(
        name for name in parameters if name.endswith("denoise_mask")
    )
    assert all(
        parameters[name].default is None
        for name in ("denoise_mask", "audio_denoise_mask")
    )


def test_forward_refuses_a_dit_block_replacement():
    # The replacement callback receives the stream, the rotation table and the
    # modulation table, all rank-local under sharding.
    model = _bound_forward()
    options = {"patches_replace": {"dit": {("double_block", 0): object()}}}
    with pytest.raises(UnsupportedModelError, match="block replacement"):
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8), options,
                       minimax_payload={"seed": 0, "audio_scale": 4.0})


def test_forward_accepts_an_empty_patches_replace():
    # An empty dict is the common case and must not trip the refusal.
    model = _bound_forward()
    try:
        model._forward(_av_latents(), torch.zeros(1), torch.zeros(1, 4, 8),
                       {"patches_replace": {}},
                       minimax_payload={"seed": 0, "audio_scale": 4.0})
    except UnsupportedModelError as exc:
        pytest.fail(f"an empty patches_replace must not trip a refusal: {exc}")
    except Exception:
        # The guards are as far as this CPU-only test can drive the forward.
        pass


@pytest.mark.parametrize(
    ("cfg", "dp", "rejected"),
    [(1, 1, False), (2, 1, True), (1, 2, True), (2, 2, True), (1, 4, True)],
)
def test_topology_predicate_truth_table(cfg, dp, rejected):
    assert minimax_h3_topology_would_reject(cfg, dp) is rejected


def test_refusal_message_names_the_supported_topologies():
    # The remedy must be reachable: mode=local, not the `single` preset (the
    # next test says why).
    assert "mode=local" in MINIMAX_H3_BATCH_CAPPED_MESSAGE
    assert "uly" in MINIMAX_H3_BATCH_CAPPED_MESSAGE
    assert "docs/MODELS.md" in MINIMAX_H3_BATCH_CAPPED_MESSAGE


@pytest.mark.parametrize("module", [h3, h3_packing])
def test_no_refusal_prescribes_the_unreachable_single_preset(module):
    # Every refusal H3 owns fires at world > 1 (the preflight only at cfg > 1
    # or dp > 1, the forward guards only once usp_forward is bound at sp > 1),
    # and at world > 1 the `single` preset derives dp and hits the
    # packed-latent refusal, so naming it would send the operator into a
    # second refusal. The remedy is mode=local with gpus_per_host=1. Both
    # modules are checked: the row algebra carries refusals too.
    source = inspect.getsource(module)
    assert "topology 'single'" not in source
    assert "on 'single'" not in source
    assert "mode=local, gpus_per_host=1" in source


def test_padded_ring_refusal_is_h3s_own_and_reachable():
    # The generic guard in base.py raises the right refusal with the wrong
    # remedy for this family (it names the `single` preset, which derives dp
    # above world 1). H3 therefore reads the ring degree itself at the seam.
    # Ulysses-only and unpadded both stay silent.
    ring = {"n": 2}
    fake = types.ModuleType("xfuser.core.distributed")
    fake.get_ring_parallel_world_size = lambda: ring["n"]
    sys.modules["xfuser"] = types.ModuleType("xfuser")
    sys.modules["xfuser.core"] = types.ModuleType("xfuser.core")
    sys.modules["xfuser.core.distributed"] = fake
    try:
        with pytest.raises(UnsupportedModelError) as excinfo:
            h3._reject_padded_ring(1)
        message = str(excinfo.value)
        assert "minimax_h3" in message
        assert "mode=local, gpus_per_host=1" in message
        assert "uly*" in message
        assert "'single'" not in message
        ring["n"] = 1
        h3._reject_padded_ring(1)          # pure ulysses: fine
    finally:
        for name in ("xfuser.core.distributed", "xfuser.core", "xfuser"):
            sys.modules.pop(name, None)


# The frame_count seam: comfy e01fb4c56 removed PackedLayout's frame_count
# parameter; older comfy still takes it and anchors last-frame keyframes with
# it. The builder passes it only to a class that accepts it.


class _LayoutTakesFrameCount:
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=None, refs=None, frame_count=None):
        self.received_frame_count = frame_count


class _LayoutWithoutFrameCount:
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=None, refs=None):
        pass


def test_frame_count_is_passed_only_where_the_installed_class_accepts_it():
    from dgx_monarch.adapters.minimax_h3 import _build_packed_layout

    payload = {"frame_count": 121, "keyframes": None, "refs": None}
    old = _build_packed_layout(_LayoutTakesFrameCount, payload, 7, 16, 17, 30, 126)
    assert old.received_frame_count == 121

    new = _build_packed_layout(_LayoutWithoutFrameCount, payload, 7, 16, 17, 30, 126)
    assert isinstance(new, _LayoutWithoutFrameCount)


def test_a_missing_payload_frame_count_stays_none_on_the_old_class():
    from dgx_monarch.adapters.minimax_h3 import _build_packed_layout

    old = _build_packed_layout(_LayoutTakesFrameCount, {}, 7, 16, 17, 30, 126)
    assert old.received_frame_count is None
