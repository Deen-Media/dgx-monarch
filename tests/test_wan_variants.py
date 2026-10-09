"""CPU tests for the Wan variant adapters: SCAIL, SCAIL-2, WanDancer and Animate2.

Nothing imports comfy or xfuser at module scope; the rank accessors are
patched (the `sp` fixture in tests/conftest.py, or wan_animate2's own names).
Facts pinned here:

  * WanDancer's audio cross-attention has one key, so softmax is 1 and the
    output is exactly o(v(audio)), which the adapter adds as a residual
    indexed per local token;
  * group(token) = global_pos // (h*w/8) shards with the tokens by construction;
  * SCAIL trims pose off the token axis, then the reference off the temporal axis;
  * WanDancer refuses reference_latent with a typed error through the bound forward;
  * only exact SCAIL-2 takes the Ring2 treatment, and SCAIL-2 precedes SCAIL in
    the registry;
  * Animate2's spatial gather equals stock over a two-rank Gloo group.
"""
import math
import os
import sys
import types

import pytest
import torch

from dgx_monarch.adapters import base, wan_animate2, wan_variants
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.wan_variants import SCAIL2Adapter, SCAILAdapter, WanDancerAdapter
from fake_model_base_helpers import install_fake_model_base


def _animate2_gloo_worker(rank: int, rendezvous: str) -> None:
    """Run the adapter's real spatial gather against a two-rank Gloo group."""
    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        class _Group:
            @staticmethod
            def all_gather(value, dim=1):
                parts = [torch.empty_like(value) for _ in range(2)]
                dist.all_gather(parts, value)
                return torch.cat(parts, dim=dim)

        xfuser = types.ModuleType("xfuser")
        core = types.ModuleType("xfuser.core")
        distributed = types.ModuleType("xfuser.core.distributed")
        distributed.get_sp_group = lambda: _Group()
        sys.modules.update({"xfuser": xfuser, "xfuser.core": core,
                            "xfuser.core.distributed": distributed})

        adapter = wan_variants.WanAnimate2Adapter()
        original_world, original_rank = wan_animate2.sp_world, wan_animate2.sp_rank
        wan_animate2.sp_world = lambda: 2
        wan_animate2.sp_rank = lambda: rank
        try:
            def attend(query, keys, values):
                query = query.transpose(1, 2)
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)
                return (torch.softmax(query @ keys.transpose(-2, -1) / math.sqrt(dim), -1) @ values).transpose(1, 2)

            # [reference, video 0, video 1], four spatial tokens/frame.
            # Values make a reference-strength mistake or j -> j-1 pose
            # mismatch visible in the stock comparison.
            torch.manual_seed(17)
            frames, pose_frames, pixels, heads, dim = 3, 2, 4, 1, 2
            q = torch.randn(1, frames * pixels, heads, dim)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            kp = torch.randn(1, pose_frames * pixels, heads, dim)
            vp = torch.randn_like(kp)
            ref_strength = 0.37
            q_local = adapter._spatial_local(q, frames, pixels, "generation")
            k_local = adapter._spatial_local(k, frames, pixels, "generation")
            v_local = adapter._spatial_local(v, frames, pixels, "generation")
            kp_local = adapter._spatial_local(kp, pose_frames, pixels, "pose")
            vp_local = adapter._spatial_local(vp, pose_frames, pixels, "pose")
            kg = adapter._gather_spatial(k_local, frames, pixels)
            vg = adapter._gather_spatial(v_local, frames, pixels).clone()
            kpg = adapter._gather_spatial(kp_local, pose_frames, pixels)
            vpg = adapter._gather_spatial(vp_local, pose_frames, pixels)
            vg[:, :pixels] *= ref_strength

            local_pixels = pixels // 2
            injected = torch.empty_like(q_local)
            for j in range(frames):
                lo, hi = j * local_pixels, (j + 1) * local_pixels
                keys, values = (kg, vg) if j == 0 else (
                    torch.cat((kg, kpg[:, (j - 1) * pixels:j * pixels]), 1),
                    torch.cat((vg, vpg[:, (j - 1) * pixels:j * pixels]), 1),
                )
                injected[:, lo:hi] = attend(q_local[:, lo:hi], keys, values)
            injected = adapter._gather_spatial(injected, frames, pixels)

            stock_v = v.clone()
            stock_v[:, :pixels] *= ref_strength
            stock = torch.empty_like(q)
            for j in range(frames):
                lo, hi = j * pixels, (j + 1) * pixels
                keys, values = (k, stock_v) if j == 0 else (
                    torch.cat((k, kp[:, (j - 1) * pixels:j * pixels]), 1),
                    torch.cat((stock_v, vp[:, (j - 1) * pixels:j * pixels]), 1),
                )
                stock[:, lo:hi] = attend(q[:, lo:hi], keys, values)
            torch.testing.assert_close(injected, stock, rtol=0, atol=0)
        finally:
            wan_animate2.sp_world, wan_animate2.sp_rank = original_world, original_rank
    finally:
        dist.destroy_process_group()


def test_wan_animate2_two_rank_gloo_spatial_gather_matches_stock(tmp_path):
    """Real Gloo collectives preserve stock reference and driving-frame order."""
    import torch.multiprocessing as mp

    rendezvous = tmp_path / "animate2-gloo"
    mp.spawn(_animate2_gloo_worker, args=(str(rendezvous),), nprocs=2, join=True)
    if rendezvous.exists():
        os.unlink(rendezvous)


def _animate2_real_comfy_gloo_worker(rank: int, rendezvous: str, comfy_dir: str) -> None:
    """Compare the real native model and injected model in isolated CPU children."""
    import torch.distributed as dist

    sys.path.insert(0, comfy_dir)
    import comfy.options

    sys.argv = [sys.argv[0], "--cpu"]
    comfy.options.enable_args_parsing()
    from comfy import ops
    from comfy.ldm.wan.model_animate2 import PoseBranchCache, WanAnimate2Model

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        class _Group:
            @staticmethod
            def all_gather(value, dim=1):
                parts = [torch.empty_like(value) for _ in range(2)]
                dist.all_gather(parts, value)
                return torch.cat(parts, dim=dim)

        xfuser = types.ModuleType("xfuser")
        core = types.ModuleType("xfuser.core")
        distributed = types.ModuleType("xfuser.core.distributed")
        distributed.get_sp_group = lambda: _Group()
        sys.modules.update({"xfuser": xfuser, "xfuser.core": core,
                            "xfuser.core.distributed": distributed})

        def tiny_model():
            model = WanAnimate2Model(
                patch_size=(1, 2, 2), text_len=4, in_dim=36, dim=8,
                ffn_dim=16, freq_dim=2, text_dim=4, out_dim=1,
                num_heads=1, num_layers=1, device="cpu", dtype=torch.float32,
                operations=ops.disable_weight_init,
            ).eval()
            for parameter in model.parameters():
                torch.nn.init.uniform_(parameter, -0.1, 0.1)
            return model

        torch.manual_seed(91)
        stock = tiny_model()
        injected = tiny_model()
        injected.load_state_dict(stock.state_dict())
        adapter = wan_variants.WanAnimate2Adapter()
        original_world, original_rank = wan_animate2.sp_world, wan_animate2.sp_rank
        wan_animate2.sp_world = lambda: 2
        wan_animate2.sp_rank = lambda: rank
        try:
            adapter.inject_usp(
                injected, InjectionContext(topology_sp=2, usp_attention=types.SimpleNamespace(
                    effective_kernel="TORCH_FLASH")))
            # Three generation frames retain the reference at index zero and
            # exercise both pose j -> generation j + 1 edges. A distinct
            # context_pose exercises the override branch of the pose context.
            x = torch.randn(1, 36, 3, 2, 4)
            pose = torch.randn(1, 16, 2, 2, 4)
            context = torch.randn(1, 4, 4)
            context_pose = torch.randn(1, 4, 4)
            timestep = torch.tensor([0.7])

            def compare(options_stock, options_injected):
                with torch.no_grad():
                    expected = stock._forward(
                        x, timestep, context, pose_latents=pose,
                        context_pose=context_pose, pose_strength=0.63,
                        reference_strength=0.41, transformer_options=options_stock,
                    )
                    actual = injected._forward(
                        x, timestep, context, pose_latents=pose,
                        context_pose=context_pose, pose_strength=0.63,
                        reference_strength=0.41, transformer_options=options_injected,
                    )
                assert expected.shape == (1, 1, 3, 2, 4)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            compare({}, {})  # cache-off ordinary native forward
            # The first call's context-window prepass stores pose block inputs;
            # the second reads the filled cache through kv_from_input, on both ranks.
            stock_cache = PoseBranchCache(store_device="cpu")
            injected_cache = PoseBranchCache(store_device="cpu")
            stock_options = {"context_window": object(), "animate2_cache": stock_cache}
            injected_options = {"context_window": object(), "animate2_cache": injected_cache}
            compare(stock_options, injected_options)
            compare(stock_options, injected_options)
            stock_cache.free()
            injected_cache.free()
        finally:
            wan_animate2.sp_world, wan_animate2.sp_rank = original_world, original_rank
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not os.path.isdir(os.environ.get("COMFY_DIR", "")),
                    reason="set COMFY_DIR to an exact ComfyUI checkout for the real-Comfy differential")
def test_wan_animate2_real_comfy_two_rank_gloo_matches_stock(tmp_path):
    """Native WAN_Animate2 and the adapter agree with Gloo collectives."""
    import torch.multiprocessing as mp

    rendezvous = tmp_path / "animate2-real-comfy-gloo"
    mp.spawn(_animate2_real_comfy_gloo_worker,
             args=(str(rendezvous), os.environ["COMFY_DIR"]), nprocs=2, join=True)
    if rendezvous.exists():
        os.unlink(rendezvous)


def test_wan_animate2_rechecks_a_mutable_selector_before_patch_embedding():
    dispatch = types.SimpleNamespace(effective_kernel="TORCH_FLASH")
    patch_calls = []
    fake = types.SimpleNamespace(
        blocks=[],
        patch_embedding=lambda _x: patch_calls.append("patch") or (_ for _ in ()).throw(AssertionError()),
    )
    wan_animate2.WanAnimate2Adapter().inject_usp(
        fake, InjectionContext(topology_sp=2, usp_attention=dispatch))
    dispatch.effective_kernel = "SAGE_AUTO"
    with pytest.raises(UnsupportedModelError, match="requires TORCH_FLASH"):
        fake.forward_orig(None, None, None)
    assert patch_calls == []


def test_wan_animate2_rejects_foreign_attention_override_before_patch_embedding():
    patch_calls = []
    fake = types.SimpleNamespace(
        blocks=[],
        patch_embedding=lambda _x: patch_calls.append("patch") or (_ for _ in ()).throw(AssertionError()),
    )
    wan_animate2.WanAnimate2Adapter().inject_usp(
        fake, InjectionContext(topology_sp=2, usp_attention=types.SimpleNamespace(
            effective_kernel="TORCH_FLASH")))

    with pytest.raises(UnsupportedModelError, match="optimized_attention_override"):
        fake.forward_orig(
            None, None, None,
            transformer_options={"optimized_attention_override": lambda *_args, **_kwargs: None},
        )
    assert patch_calls == []


@pytest.mark.parametrize(
    "patches",
    [
        {"attn1_patch": [lambda _args: _args]},
        {"attn2_patch": [lambda _args: _args]},
        {"attn1_patch": object()},
    ],
)
def test_wan_animate2_refuses_unqualified_attention_patches_before_patch_embedding(patches):
    patch_calls = []
    fake = types.SimpleNamespace(
        blocks=[],
        patch_embedding=lambda _x: patch_calls.append("patch") or (_ for _ in ()).throw(AssertionError()),
    )
    wan_animate2.WanAnimate2Adapter().inject_usp(
        fake, InjectionContext(topology_sp=2, usp_attention=types.SimpleNamespace(
            effective_kernel="TORCH_FLASH")))

    with pytest.raises(UnsupportedModelError, match="sequence_shard_local"):
        fake.forward_orig(None, None, None, transformer_options={"patches": patches})
    assert patch_calls == []


def test_wan_animate2_allows_explicitly_sequence_local_attn2_patch():
    from dgx_monarch.adapters import attention_patches

    def patch(args):
        return args

    setattr(
        patch, attention_patches.ATTENTION_PATCH_CAPABILITIES_ATTR,
        frozenset({attention_patches.SEQUENCE_SHARD_LOCAL_CAPABILITY}),
    )
    attention_patches.assert_usp_attention_patches_safe(
        {"patches": {"attn2_patch": [patch]}}, "wan_animate2",
        ("attn1_patch", "attn1_output_patch", "attn2_patch"),
    )


def test_single_key_attention_returns_value_regardless_of_query():
    # Softmax over a single key is exactly 1.0 for every query, so scaled
    # dot-product attention returns the value vector unchanged. That lets the
    # WanDancer injector drop the attention.
    torch.manual_seed(0)
    b, heads, n, hd = 2, 4, 6, 8
    q = torch.randn(b, heads, n, hd)
    k = torch.randn(b, heads, 1, hd)   # one key (one audio frame)
    v = torch.randn(b, heads, 1, hd)
    attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)  # (b, heads, n, 1)
    assert torch.equal(attn, torch.ones_like(attn))          # exact: one-logit softmax == 1
    assert torch.equal(attn @ v, v.expand(b, heads, n, hd))  # output == value, any q/k


def test_audio_injector_collapses_to_o_of_v():
    # Stock WanT2VCrossAttention with one key (query: a frame's spatial tokens;
    # key and value: that frame's one audio embedding) equals o(v(audio)) on
    # every spatial token. The adapter computes that per frame and indexes it
    # per local token, so it never needs stock's (b (t n) c) rearrange, which
    # needs every frame's tokens on one rank.
    torch.manual_seed(0)
    dim, heads, n = 16, 4, 6
    hd = dim // heads
    v_proj = torch.nn.Linear(dim, dim)
    o_proj = torch.nn.Linear(dim, dim)
    q_proj = torch.nn.Linear(dim, dim)
    k_proj = torch.nn.Linear(dim, dim)
    audio = torch.randn(1, 1, dim)     # one frame's audio (b=1, 1 token)
    hidden = torch.randn(1, n, dim)    # the frame's n spatial query tokens

    # stock path: attention over a single key, then o_proj
    q = q_proj(hidden).view(1, n, heads, hd).transpose(1, 2)
    k = k_proj(audio).view(1, 1, heads, hd).transpose(1, 2)
    val = v_proj(audio).view(1, 1, heads, hd).transpose(1, 2)
    attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
    ref = o_proj((attn @ val).transpose(1, 2).reshape(1, n, dim))

    # adapter forward: o(v(audio)) broadcast to the n tokens
    got = o_proj(v_proj(audio)).expand(1, n, dim)
    assert torch.allclose(ref, got, atol=1e-6)


@pytest.mark.parametrize("latent_frames,hw", [(2, 16), (3, 24)])
def test_group_index_shards_aligned_to_tokens(sp, latent_frames, hw):
    # group(token) = global_pos // (h*w/8). Built full length and sharded with
    # the same shard_seq the tokens take, each rank's group ids align with its
    # token chunk by construction, with no per-rank offset math.
    num_frames = latent_frames * 8
    seq_len = latent_frames * hw
    assert seq_len % num_frames == 0
    tokens_per_group = seq_len // num_frames          # == hw/8
    tokens = torch.arange(seq_len).view(1, seq_len, 1).float()
    group_full = (torch.arange(seq_len) // tokens_per_group).view(1, seq_len, 1)

    recon, t_orig = [], None
    for rank in (0, 1):
        sp(2, rank)
        t_local, t_orig = base.shard_seq(tokens, dim=1)
        g_local, _ = base.shard_seq(group_full, dim=1)
        assert g_local.shape[1] == t_local.shape[1]
        # each local token's group id == its global position // tokens_per_group
        assert torch.equal(g_local[0, :, 0].long(), t_local[0, :, 0].long() // tokens_per_group)
        recon.append(g_local)
    assert torch.equal(torch.cat(recon, dim=1).narrow(1, 0, t_orig), group_full)


def test_group_index_pad_rows_get_no_residual(sp):
    # Pad-path defence. WanDancer's seq_len is a multiple of num_frames (stock's
    # rearrange requires it), hence of 8, so two ranks never pad; the gather must
    # still stay in bounds. An odd length zero-pads on the last rank; those rows
    # clamp to a valid group index and the validity mask zeroes their residual.
    seq_len, num_frames, tpg = 5, 3, 2
    group_full = (torch.arange(seq_len) // tpg).view(1, seq_len, 1)
    valid_full = torch.ones(1, seq_len, 1)
    sp(2, 1)  # last rank carries the divisibility pad
    g_local, _ = base.shard_seq(group_full, dim=1)
    v_local, _ = base.shard_seq(valid_full, dim=1)
    g_idx = g_local.reshape(-1).clamp_(max=num_frames - 1).long()
    assert int(g_idx.max()) <= num_frames - 1
    assert float(v_local.reshape(-1)[-1]) == 0.0


@pytest.mark.parametrize("ref_t,video_t", [(2, 3), (1, 4)])
def test_scail_two_trim_slice_bookkeeping(ref_t, video_t):
    # After the gather the head output is [reference and video grid tokens, then
    # pose tokens]. Trim pose off the token axis before unpatchify, then the
    # reference frames off the temporal axis after it, in stock's order
    # (SCAILWanModel.forward_orig in comfy/ldm/wan/model.py).
    H, W = 2, 2
    grid_t = ref_t + video_t
    main_len = grid_t * H * W
    pose_len = 5
    head_out = torch.arange(main_len + pose_len).view(1, -1, 1).float()  # distinct per token

    # (1) pose trim on the token axis
    after_pose = head_out[:, :-pose_len]
    assert after_pose.shape[1] == main_len
    assert torch.equal(after_pose, head_out[:, :main_len])

    # (2) unpatchify maps main_len tokens -> [B, C, F, H, W] row-major over
    #     (frame, h, w); emulate the frame split, then ref-trim the temporal axis.
    frames = after_pose.view(1, grid_t, H * W)
    after_ref = frames[:, ref_t:]
    assert after_ref.shape[1] == video_t
    # first surviving row is the first token of the first video frame
    assert float(after_ref[0, 0, 0]) == float(ref_t * H * W)


def test_scail2_inherits_scail_forward():
    # SCAIL-2 adds only additive mask embeddings, which the same forward's kwargs
    # branches handle, so it inherits inject_usp unchanged. Its own adapter still
    # carries the exact-type allowlist and the Ring2 treatment.
    assert SCAIL2Adapter.inject_usp is SCAILAdapter.inject_usp
    assert SCAIL2Adapter._EXACT == ("WAN21_SCAIL2",)
    assert SCAIL2Adapter().family == "wan_scail"


@pytest.fixture
def wan_model_stub(monkeypatch):
    """Provide the one lazy Comfy symbol the bound SCAIL forward imports."""
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    wan = types.ModuleType("comfy.ldm.wan")
    model = types.ModuleType("comfy.ldm.wan.model")
    model.sinusoidal_embedding_1d = (
        lambda dim, t: t.reshape(-1, 1).expand(-1, dim)
    )
    comfy.ldm = ldm
    ldm.wan = wan
    wan.model = model
    for name, module in (
        ("comfy", comfy),
        ("comfy.ldm", ldm),
        ("comfy.ldm.wan", wan),
        ("comfy.ldm.wan.model", model),
    ):
        monkeypatch.setitem(sys.modules, name, module)


class _FakeScailBlock:
    def __init__(self):
        self.self_attn = types.SimpleNamespace()
        self.e_shapes = []

    def __call__(self, x, *, e, freqs, context, context_img_len,
                 transformer_options):
        self.e_shapes.append(tuple(e.shape))
        return x


class _FakeScailForward:
    """Minimal model that executes the real injected SCAIL forward on CPU."""

    def __init__(self):
        self.blocks = [_FakeScailBlock()]
        self.freq_dim = 2
        self.dim = 1
        self.img_emb = None
        self.patch_embedding = lambda value: value
        self.patch_embedding_pose = lambda value: value
        self.patch_embedding_mask = lambda value: value
        self.time_embedding = lambda value: value
        self.time_projection = lambda value: value.new_zeros(
            (*value.shape[:-1], 6 * self.dim)
        )
        self.text_embedding = lambda value: value
        self.head = lambda value, _e: value
        self.unpatchify = lambda value, _grid_sizes: value


def _call_scail_forward(model, t):
    return model.forward_orig(
        torch.zeros(1, 1, 1, 1, 4),
        t,
        torch.zeros(1, 2, 1),
        freqs=torch.zeros(1, 4, 1),
    )


@pytest.mark.parametrize("adapter_cls", [SCAILAdapter, SCAIL2Adapter])
@pytest.mark.parametrize("t", [torch.zeros(1), torch.zeros(1, 1)])
def test_scail_scalar_timestep_shapes_run_through_bound_forward(
    monkeypatch, sp, wan_model_stub, adapter_cls, t
):
    sp(2, 0)
    monkeypatch.setattr(
        wan_variants,
        "sp_gather",
        lambda value, orig_len, dim=1: torch.cat((value, value), dim=dim).narrow(
            dim, 0, orig_len
        ),
    )
    model = _FakeScailForward()
    adapter_cls().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=object())
    )

    result = _call_scail_forward(model, t)

    assert result.shape == (1, 4, 1)
    assert model.blocks[0].e_shapes == [(1, 1, 6, 1)]


@pytest.mark.parametrize("adapter_cls", [SCAILAdapter, SCAIL2Adapter])
def test_scail_variants_reject_multirow_timestep_before_sharding(
    monkeypatch, wan_model_stub, adapter_cls
):
    def unexpected_shard(*_args, **_kwargs):
        pytest.fail("multi-row SCAIL timestep reached sequence sharding")

    monkeypatch.setattr(wan_variants, "shard_seq", unexpected_shard)
    model = _FakeScailForward()
    adapter_cls().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=object())
    )

    with pytest.raises(
        UnsupportedModelError,
        match=r"SCAIL/SCAIL-2 sequence parallelism supports only scalar timesteps.*got 2",
    ):
        _call_scail_forward(model, torch.tensor([[0.2, 0.8]]))


def test_worker_selects_ring2_treatment_only_for_exact_scail2(monkeypatch):
    from dgx_monarch.actor import worker as worker_mod

    captured = []
    adapters = [
        SCAIL2Adapter(),
        SCAILAdapter(),
        WanDancerAdapter(),
        type("FutureSCAIL2Adapter", (SCAIL2Adapter,), {})(),
    ]
    for adapter in adapters:
        adapter.inject_usp = (
            lambda _model, ctx, adapter=adapter:
            captured.append((type(adapter), ctx.usp_attention))
        )

    monkeypatch.setattr(worker_mod.store_fsdp, "validate_injection", lambda *_args: None)
    monkeypatch.setattr(worker_mod, "_maybe_compile_dit", lambda _model: None)
    adapter_slot = {"value": adapters[0]}
    monkeypatch.setattr(
        "dgx_monarch.adapters.get_adapter", lambda _model: adapter_slot["value"]
    )

    treatment_calls = []
    dispatch = types.SimpleNamespace(
        for_wan=lambda: treatment_calls.append("for_wan") or "wan-ring2-treatment",
        kernel="TORCH_FLASH", effective_kernel="TORCH_FLASH",
        bind_capability=lambda *_a: None,
    )
    worker = types.SimpleNamespace(
        topology={"dp": 1, "cfg": 1, "ulysses": 1, "ring": 2, "fsdp": False},
        _attn=dispatch,
    )
    patcher = types.SimpleNamespace(
        model=types.SimpleNamespace(diffusion_model=object()), model_options={}
    )

    for adapter in adapters:
        adapter_slot["value"] = adapter
        worker_mod.GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)

    assert treatment_calls == ["for_wan"]
    assert [attention for _adapter_type, attention in captured] == [
        "wan-ring2-treatment",
        dispatch,
        dispatch,
        dispatch,
    ]


def test_wandancer_rejects_reference_latent_typed():
    # Drive the bound forward: with ref_conv present, a reference_latent would
    # prepend full_ref and misalign the injector's pre-captured seq_len, so the
    # adapter refuses it with a typed error.
    model = types.SimpleNamespace(
        blocks=[], ref_conv=object(),
        music_injector=types.SimpleNamespace(injector=[], injected_block_id={}))
    WanDancerAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    with pytest.raises(UnsupportedModelError, match="reference_latent"):
        model.forward_orig(
            torch.zeros(1, 16, 1, 4, 4), torch.zeros(1), torch.zeros(1, 8, 16),
            reference_latent=torch.zeros(1, 16, 1, 4, 4))


@pytest.fixture
def fake_model_base(monkeypatch):
    """SCAIL-2 subclasses SCAIL upstream; WanDancer is a separate WAN21 child."""
    return install_fake_model_base(
        monkeypatch,
        {
            "WAN21": None,
            "WAN21_SCAIL": "WAN21",
            "WAN21_SCAIL2": "WAN21_SCAIL",
            "WAN22_WanDancer": "WAN21",
        },
    )


def _first_match(adapters, model):
    """Mirror of dgx_monarch.adapters.get_adapter: the first True wins; a typed reject propagates."""
    for adapter in adapters:
        if adapter.matches(model):
            return adapter
    return None


# tests/test_adapter_matches.py tables the exact-type accept and the typed
# reject of a future subclass for all three adapters, each declining its real
# parent or sibling (WAN21 for SCAIL, WAN21_SCAIL for SCAIL2 and WanDancer).
def test_scail2_must_precede_scail_in_registry(fake_model_base):
    scail2 = fake_model_base.WAN21_SCAIL2()
    # correct order (most specific first): SCAIL-2 adapter claims it
    assert _first_match([SCAIL2Adapter(), SCAILAdapter()], scail2).family == "wan_scail"
    # Reversed, SCAILAdapter sees SCAIL-2 first: isinstance of WAN21_SCAIL holds
    # but the exact-type check fails, so it raises a typed reject. The registry
    # must list SCAIL-2 first.
    with pytest.raises(UnsupportedModelError):
        _first_match([SCAILAdapter(), SCAIL2Adapter()], scail2)


def test_cfg_cond_padding_is_none():
    assert SCAILAdapter.cfg_cond_padding == "none"
    assert SCAIL2Adapter.cfg_cond_padding == "none"
    assert WanDancerAdapter.cfg_cond_padding == "none"


def test_variant_keys_detect_as_wan_or_variant():
    # The SCAIL, SCAIL-2 and WanDancer state-dict keys also carry the plain-wan
    # needles (text_embedding., self_attn.). detect._SIGNATURES lists the
    # wan_scail and wan_dancer rows before the wan row, so they resolve to the
    # exact family; "wan" is accepted too, because its auto rule picks the same
    # uly2 topology.
    from dgx_monarch.adapters.detect import detect_family_from_keys

    scail = [
        "diffusion_model.text_embedding.0.weight",
        "diffusion_model.blocks.0.self_attn.q.weight",
        "diffusion_model.patch_embedding_pose.weight",
    ]
    scail2 = [*scail, "diffusion_model.patch_embedding_mask.weight"]
    wandancer = [
        "diffusion_model.text_embedding.0.weight",
        "diffusion_model.blocks.0.self_attn.q.weight",
        "diffusion_model.patch_embedding_global.weight",
    ]
    assert detect_family_from_keys(scail) in ("wan", "wan_scail")
    assert detect_family_from_keys(scail2) in ("wan", "wan_scail")
    assert detect_family_from_keys(wandancer) in ("wan", "wan_dancer")


def test_wandancer_per_frame_e0_expansion_arithmetic():
    """Pin the token-level expansion of per-frame e0 that the dancer forward
    applies before shard_seq. If e0 stays unexpanded under uly2, each rank
    stretches the full frame schedule over its local chunk; on 2026-07-28 that gave
    non-finite sampler output with the audio injector disabled. Expansion runs
    on the full sequence, so each rank's slice carries global frame ids, as in
    wan_family's WanAdapter."""
    import torch

    frames, tokens_per_frame, world = 9, 1560, 2
    seq_len = frames * tokens_per_frame
    e0 = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1)

    tokens_per_e = -(-seq_len // e0.shape[1])
    if tokens_per_e != tokens_per_frame:
        pytest.fail("ceil-div must recover tokens-per-frame for divisible grids")
    expanded = torch.repeat_interleave(e0, tokens_per_e, dim=1)[:, :seq_len]
    if expanded.shape[1] != seq_len:
        pytest.fail("expanded e0 must cover every token exactly once")

    half = seq_len // world
    # Rank 1's local table must carry the global frame ids of its tokens.
    rank1 = expanded[:, half:]
    first_global_frame = half // tokens_per_frame
    if int(rank1[0, 0, 0, 0]) != first_global_frame:
        pytest.fail("rank slice must see global frame ids, not a restarted schedule")
    # A non-divisible tail is truncated, never wrapped.
    ragged = torch.repeat_interleave(e0, -(-(seq_len + 5) // frames), dim=1)[:, :seq_len + 5]
    if ragged.shape[1] != seq_len + 5 or int(ragged[0, -1, 0, 0]) != frames - 1:
        pytest.fail("ragged expansion must truncate the last frame, not wrap")

    # Scalar-t models (e0 seq length 1) skip the expansion.
    scalar = torch.zeros(1, 1, 1, 1)
    if scalar.shape[1] > 1:
        pytest.fail("scalar e0 must take the broadcast path")
