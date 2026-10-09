"""Z-Image adapter logic on CPU, with a fake comfy.model_base and no distributed init.

Z-Image is comfy's maskless NextDiT (comfy/ldm/lumina) in latent and pixel-space forms, so
attention would read any zero row. The tests hold:
  * learned x32 padding keeps the [cap | img] concat divisible by sp in {2, 4, 8}, so no
    zero pad row is added;
  * the Ulysses degree must divide the 30 heads (uly2 passes, uly4 refuses);
  * omni/edit, Ming-Image and effective caption-mask inputs refuse through the bound
    forward, before any comfy import;
  * the pixel-space head decodes only the real image tokens, in B*N order;
  * matches() claims ZImagePixelSpace by exact type and Lumina2 only with learned pad
    tokens, and returns False, never raising, for real Lumina 2.0, which shares that type;
  * cfg-pad trims a uniform caption batch and refuses a ragged one.
"""
import sys
import types

import pytest
import torch

import dgx_monarch.adapters.zimage as zimage
from dgx_monarch.actor.sampling import equalize_cond_lengths
from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.zimage import (
    ZImageAdapter,
    _assert_concat_divisible,
    _assert_ulysses_divides_heads,
    _pixel_decode,
    _trim_context_or_raise,
)


def _pad32(n: int) -> int:
    """pad_zimage's rule: pad a length up to the next multiple of 32."""
    return n + ((-n) % 32)


@pytest.mark.parametrize("sp", [2, 4, 8])
@pytest.mark.parametrize("cap_real,img_real", [(1, 1), (7, 6), (33, 120), (300, 4096), (512, 1024)])
def test_learned_pad_makes_concat_divisible_no_zero_pad(sp, cap_real, img_real):
    cap_p, img_p = _pad32(cap_real), _pad32(img_real)
    concat = cap_p + img_p
    # Each stream is a multiple of 32, so the concat is too, and every sp in {2,4,8} divides it.
    assert cap_p % 32 == 0 and img_p % 32 == 0
    assert concat % 32 == 0 and concat % sp == 0
    _assert_concat_divisible(concat, sp)  # no raise
    # base.shard_seq's pad path is a no-op on a divisible length: nothing added.
    padded, orig = base.pad_seq_to_multiple(torch.zeros(1, concat, 4), sp)
    assert orig == concat
    assert padded.shape[1] == concat


def test_concat_divisible_guard_rejects_indivisible_and_names_x_pad_token():
    # x32 padding never yields 30; sp=4 does not divide it, so the guard fires. Its
    # message names x_pad_token, not zeros, because no mask hides a zero pad row.
    with pytest.raises(UnsupportedModelError, match="x_pad_token"):
        _assert_concat_divisible(30, 4)
    _assert_concat_divisible(30, 1)  # sp==1 (single) never raises


def test_shard_seq_roundtrip_has_no_pad_rows(monkeypatch):
    concat = _pad32(7) + _pad32(6)  # 32 + 32 = 64
    t = torch.arange(concat).view(1, concat, 1).float()
    chunks, orig = [], 0
    for rank in range(2):
        monkeypatch.setattr(base, "sp_world", lambda: 2)
        monkeypatch.setattr(base, "sp_rank", lambda r=rank: r)
        local, orig = base.shard_seq(t, dim=1)
        assert local.shape[1] == concat // 2  # exact split, no pad chunk
        chunks.append(local)
    gathered = torch.cat(chunks, dim=1).narrow(1, 0, orig)
    assert torch.equal(gathered, t)


@pytest.mark.parametrize("deg,ok", [(1, True), (2, True), (3, True), (5, True), (6, True),
                                    (4, False), (7, False), (8, False)])
def test_ulysses_degree_must_divide_30_heads(deg, ok):
    if ok:
        _assert_ulysses_divides_heads(30, deg)  # no raise
    else:
        with pytest.raises(UnsupportedModelError, match="Ulysses"):
            _assert_ulysses_divides_heads(30, deg)


class _FakeZImage:
    """Model double: injection and the forward's input refusals never touch a real block."""

    def __init__(self, n_layers=30, pixel=False, n_heads=30):
        self.n_heads = n_heads
        self.patch_size = 2
        self.layers = [object() for _ in range(n_layers)]
        if pixel:
            self.dec_net = object()


def _inject(monkeypatch, pixel=False, ulysses=2):
    monkeypatch.setattr(zimage, "_ulysses_degree", lambda: ulysses)
    model = _FakeZImage(pixel=pixel)
    ZImageAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=None))
    return model


@pytest.mark.parametrize("pixel", [False, True])
def test_omni_ref_latents_rejected_through_forward(monkeypatch, pixel):
    model = _inject(monkeypatch, pixel=pixel)
    assert "_forward" in model.__dict__  # inject bound the right instance forward
    with pytest.raises(UnsupportedModelError, match="omni/edit"):
        model._forward(torch.zeros(1, 16, 8, 8), torch.ones(1), torch.zeros(1, 4, 16), 4,
                       ref_latents=[torch.zeros(1, 16, 8, 8)])
    # ref_contexts / siglip_feats (the omni-edit signals) also trip the gate.
    with pytest.raises(UnsupportedModelError):
        model._forward(torch.zeros(1, 16, 8, 8), torch.ones(1), torch.zeros(1, 4, 16), 4,
                       siglip_feats=[torch.zeros(1, 3, 3, 3)])


@pytest.mark.parametrize("pixel", [False, True])
def test_ming_image_rejected_through_forward(monkeypatch, pixel):
    """Comfy 3b4c0b0e's Ming-Image inputs refuse before the sharded backbone runs:
    a frame axis, a direct-context extension, or reference frames.

    Upstream NextDiTPixelSpace._forward names no direct_context or ref_frames
    parameter, so the adapter's pixel forward passes the guard None and [] for
    them. The pixel variant can trip the guard only with a 5-D x, which the guard
    checks first.
    """
    model = _inject(monkeypatch, pixel=pixel)
    with pytest.raises(UnsupportedModelError, match="Ming-Image"):
        model._forward(torch.zeros(1, 16, 1, 8, 8), torch.ones(1), torch.zeros(1, 4, 16), 4)
    if not pixel:
        with pytest.raises(UnsupportedModelError, match="Ming-Image"):
            model._forward(torch.zeros(1, 16, 8, 8), torch.ones(1), torch.zeros(1, 4, 16), 4,
                           direct_context=torch.zeros(1, 2, 16))
        with pytest.raises(UnsupportedModelError, match="Ming-Image"):
            model._forward(torch.zeros(1, 16, 8, 8), torch.ones(1), torch.zeros(1, 4, 16), 4,
                           ref_frames=[torch.zeros(1, 16, 8, 8)])


@pytest.mark.parametrize("pixel", [False, True])
def test_usp_effective_attention_mask_refuses_before_comfy_work(monkeypatch, pixel):
    model = _inject(monkeypatch, pixel=pixel)
    with pytest.raises(
        UnsupportedModelError, match="effective caption attention mask"
    ) as caught:
        model._forward(
            torch.zeros(1, 16, 8, 8),
            torch.ones(1),
            torch.zeros(1, 4, 16),
            4,
            attention_mask=torch.tensor([[True, True, False, False]]),
        )
    message = str(caught.value)
    assert "pure cfg2" in message and "no supported topology" in message
    assert "single topology" not in message


def test_inject_rejects_uly4_for_30_heads(monkeypatch):
    # inject_usp runs the head guard itself, before it binds any forward.
    monkeypatch.setattr(zimage, "_ulysses_degree", lambda: 4)
    with pytest.raises(UnsupportedModelError, match="Ulysses"):
        ZImageAdapter().inject_usp(_FakeZImage(), InjectionContext(topology_sp=4, usp_attention=None))


def test_pixel_decode_slices_real_image_tokens_and_preserves_order():
    dim = 5
    cap_len, n_patches, pad = 32, 6, 26  # image padded 6 -> 32 => 26 pad tokens beyond N
    total = cap_len + n_patches + pad
    # Gathered joint stream [cap | img_real | img_pad]; each row holds its index
    # in every channel, so the test can trace every token.
    img = torch.arange(total).view(1, total, 1).float().repeat(1, 1, dim)
    pixel_values = torch.zeros(n_patches, 1, 12)  # ignored by the echo dec_net

    def dec_net(pv, cond):  # cond: (B*N, dim) -> (B*N, 1, dim): echo the token id
        return cond.unsqueeze(1)

    captured = {}

    def unpatchify_fn(full, img_size, cap_size, return_tensor):
        captured["full"] = full
        captured["cap_size"] = cap_size
        return torch.zeros(full.shape[0], 3, 4, 4)  # dummy image (B, C, h, w)

    out = _pixel_decode(img, cap_len, n_patches, dim, pixel_values, dec_net, unpatchify_fn,
                        img_size=[(8, 8)], x_is_tensor=True, h=4, w=4)
    full = captured["full"]
    assert full.shape == (1, cap_len + n_patches, dim)
    # The cap placeholder is zeros, and the image rows are exactly the real image
    # tokens [cap_len, cap_len+N), so the 26 pad tokens beyond N never appear.
    assert torch.all(full[:, :cap_len] == 0)
    assert torch.equal(full[0, cap_len:], img[0, cap_len:cap_len + n_patches])
    assert not torch.any(full[0, cap_len:] >= cap_len + n_patches)
    assert captured["cap_size"] == [cap_len]
    assert out.shape == (1, 3, 4, 4)  # comfy's outer forward() applies the x0 residual


@pytest.fixture
def model_base(monkeypatch):
    """Fake comfy.model_base: as in comfy, Lumina 2.0 and latent Z-Image share the Lumina2
    type (there is no model_base.ZImage), and ZImagePixelSpace subclasses Lumina2."""
    mb = types.ModuleType("comfy.model_base")

    class BaseModel: ...

    class Lumina2(BaseModel): ...

    class ZImagePixelSpace(Lumina2): ...

    class SomethingElse(BaseModel): ...

    for cls in (BaseModel, Lumina2, ZImagePixelSpace, SomethingElse):
        setattr(mb, cls.__name__, cls)
    comfy = types.ModuleType("comfy")
    comfy.model_base = mb
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_base", mb)
    return mb


def _mk(cls, pad_tokens_multiple):
    model = cls()
    model.diffusion_model = types.SimpleNamespace(
        pad_tokens_multiple=pad_tokens_multiple, n_heads=30, dim=3840)
    return model


def test_latent_zimage_matches_via_learned_pad_tokens(model_base):
    # The Lumina2 type with pad_tokens_multiple set is latent Z-Image.
    assert ZImageAdapter().matches(_mk(model_base.Lumina2, 32)) is True


def test_real_lumina2_is_declined_not_raised(model_base):
    # Lumina2 with no pad tokens is real Lumina 2.0: matches() returns False rather
    # than raising, so get_adapter falls through to its generic "no adapter" error.
    assert ZImageAdapter().matches(_mk(model_base.Lumina2, None)) is False


def test_pixel_zimage_matches_by_exact_type(model_base):
    assert ZImageAdapter().matches(_mk(model_base.ZImagePixelSpace, 32)) is True


def test_unrelated_model_declined(model_base):
    assert ZImageAdapter().matches(_mk(model_base.SomethingElse, 32)) is False


def test_cfg_trim_uniform_batch_trims_clean():
    # A batch-1 cfg rank with trailing zero rows trims to its real length and
    # leaves no residual pad, so nothing raises.
    ctx = torch.zeros(1, 10, 4)
    ctx[:, :6] = 1.0
    assert _trim_context_or_raise(ctx).shape[1] == 6


def test_cfg_trim_ragged_batch_raises_maskless():
    # Two captions of different real lengths leave residual pad, which the trim refuses because the model is maskless.
    ctx = torch.zeros(2, 10, 4)
    ctx[0, :6] = 1.0
    ctx[1, :9] = 1.0
    with pytest.raises(UnsupportedModelError, match="maskless"):
        _trim_context_or_raise(ctx)


class _CfgZImageProbe:
    def __init__(self):
        self.observed = None

    def _forward(self, x, timesteps, context, num_tokens, attention_mask=None,
                 transformer_options=None, **kwargs):
        self.observed = (context, num_tokens, attention_mask)
        return context


def test_pure_cfg2_asymmetric_prompt_uses_family_trim_path():
    adapter = ZImageAdapter()
    positive = [[torch.ones(1, 7, 4), {}]]
    negative = [[torch.ones(1, 3, 4), {}]]
    _, padded_negative = equalize_cond_lengths(
        adapter, positive, negative, torch.zeros(1, 4, 8, 8)
    )
    assert padded_negative[0][0].shape[1] == 7

    model = _CfgZImageProbe()
    adapter.inject_cfg_pad_forward(model)
    model._forward(
        torch.zeros(1, 4, 8, 8), torch.ones(1), padded_negative[0][0], 7
    )
    context, num_tokens, attention_mask = model.observed
    assert context.shape[1] == num_tokens == 3
    assert attention_mask is None


def test_ming_image_rejected_through_cfg_pad_forward_before_any_trim():
    """cfg-pad refuses Ming-Image instead of trimming it: the driver-pad trim was
    validated only on Z-Image's plain trailing padding, and Ming-Image's
    masked_pad_multiple can make some zero rows load-bearing."""
    model = _CfgZImageProbe()
    ZImageAdapter().inject_cfg_pad_forward(model)
    with pytest.raises(UnsupportedModelError, match="Ming-Image"):
        model._forward(torch.zeros(1, 4, 1, 8, 8), torch.ones(1), torch.zeros(1, 4, 16), 4)
    with pytest.raises(UnsupportedModelError, match="Ming-Image"):
        model._forward(torch.zeros(1, 4, 8, 8), torch.ones(1), torch.zeros(1, 4, 16), 4,
                       direct_context=torch.zeros(1, 2, 16))
    assert model.observed is None  # refused before the stock forward ran


def test_adapter_contract_declared():
    adapter = ZImageAdapter()
    assert adapter.family == "zimage"
    assert adapter.model_base_classes == ("Lumina2", "ZImagePixelSpace")
    # Maskless model: the cfg pad forward trims a uniform batch exactly and refuses a ragged one.
    assert adapter.cfg_cond_padding == "pad"
