"""HunyuanAdapter: exact-type dispatch, the typed guards through the real bound
forward, and the img-slice bounds.

CPU only, with no comfy import at module scope. matches() imports
comfy.model_base lazily, so a fake module with the real class hierarchy stands
in; the guards fire before the forward's first comfy import, so plain fakes
reach the real bound forward_orig.
"""
import sys
import types

import pytest
import torch

from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.hunyuan import (
    _HUNYUAN_SUPPORTED_EXACT,
    HunyuanAdapter,
    _img_slice_bounds,
)


@pytest.mark.parametrize(
    "txt_len,img_len,ref_len,expected",
    [
        (7, 10, 0, (7, 17)),   # no ref: [txt_len : txt_len + img_len]
        (7, 12, 2, (9, 19)),   # ref: also drop the 2 leading ref rows
        (0, 5, 0, (0, 5)),     # no text (all-zero pos ids still shard fine)
        (4, 4, 4, (8, 8)),     # img is all ref: the real-image slice is empty
    ],
)
def test_img_slice_bounds(txt_len, img_len, ref_len, expected):
    assert _img_slice_bounds(txt_len, img_len, ref_len) == expected


def test_img_slice_recovers_real_image_from_joint_stream():
    # The gathered joint [txt, ref, real_img] stream: the bounds must slice out
    # exactly the real-image rows, trimming both txt and ref.
    txt_len, ref_len, real = 3, 2, 6
    img_len = ref_len + real  # stock captures img_len after the ref concat
    joint = (
        [("txt", i) for i in range(txt_len)]
        + [("ref", i) for i in range(ref_len)]
        + [("img", i) for i in range(real)]
    )
    lo, hi = _img_slice_bounds(txt_len, img_len, ref_len)
    assert joint[lo:hi] == [("img", i) for i in range(real)]


class _FakeHunyuan:
    """Minimal stand-in: inject_usp binds forward_orig and logs block counts;
    the guards read only guiding_frame_index, txt_mask, byt5_in with txt_byt5,
    and transformer_options, all before the forward's comfy import."""

    def __init__(self, byt5=False):
        self.double_blocks = []
        self.single_blocks = []
        self.byt5_in = object() if byt5 else None


def _bound(byt5=False):
    model = _FakeHunyuan(byt5=byt5)
    HunyuanAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=None))
    return model


def _call(model, **over):
    args = {
        "img": torch.zeros(1, 4, 1, 2, 2), "img_ids": torch.zeros(1, 4, 3),
        "txt": torch.zeros(1, 3, 8), "txt_ids": torch.zeros(1, 3, 3),
        "txt_mask": None, "timesteps": torch.zeros(1),
    }
    args.update(over)
    return model.forward_orig(**args)


def test_inject_binds_forward_orig():
    model = _bound()
    assert "forward_orig" in model.__dict__


def test_guard_guiding_frame_index():
    with pytest.raises(UnsupportedModelError, match="guiding_frame_index"):
        _call(_bound(), guiding_frame_index=torch.zeros(1))


def test_guard_real_attention_mask():
    with pytest.raises(UnsupportedModelError, match="attention mask"):
        _call(_bound(byt5=False), txt_mask=torch.ones(1, 3))


def test_guard_mask_plus_byt5():
    with pytest.raises(UnsupportedModelError, match="byt5"):
        _call(_bound(byt5=True), txt_mask=torch.ones(1, 3), txt_byt5=torch.zeros(1, 5, 8))


def test_guard_byt5_without_mask_does_not_reject():
    # byt5 without a mask must not trip the mask+byt5 guard: it passes every
    # guard and reaches the forward's first comfy import, which fails in the CPU
    # test env. ModuleNotFoundError is the one exception this test tolerates;
    # any other, UnsupportedModelError (a guard that wrongly fired) included,
    # fails it.
    with pytest.raises(ModuleNotFoundError, match="comfy"):
        _call(_bound(byt5=True), txt_byt5=torch.zeros(1, 5, 8))


def test_guiding_frame_index_checked_before_mask():
    # guiding_frame_index is checked first; beside a mask and byt5 it still wins.
    with pytest.raises(UnsupportedModelError, match="guiding_frame_index"):
        _call(_bound(byt5=True), guiding_frame_index=torch.zeros(1),
              txt_mask=torch.ones(1, 3), txt_byt5=torch.zeros(1, 5, 8))


@pytest.mark.parametrize("hook", ["attn1_patch", "attn1_output_patch"])
def test_hunyuan_rejects_uncontracted_attention_patch_under_usp(hook):
    def patch(value):
        return value

    with pytest.raises(UnsupportedModelError, match=hook):
        _call(
            _bound(),
            transformer_options={"patches": {hook: [patch]}},
        )


@pytest.fixture
def model_base(monkeypatch):
    """Fake comfy.model_base with the real Hunyuan hierarchy (model_base.py):
    HunyuanImage21 is a direct BaseModel child (refiner subclasses it);
    HunyuanVideo15 subclasses the HunyuanVideo 1.0 base (SR subclasses 1.5)."""
    mb = types.ModuleType("comfy.model_base")

    class BaseModel: ...

    class HunyuanVideo(BaseModel): ...            # 1.0

    class HunyuanVideoI2V(HunyuanVideo): ...      # 1.0 i2v

    class HunyuanVideoSkyreelsI2V(HunyuanVideo): ...

    class HunyuanImage21(BaseModel): ...

    class HunyuanImage21Refiner(HunyuanImage21): ...

    class HunyuanVideo15(HunyuanVideo): ...

    class HunyuanVideo15_SR_Distilled(HunyuanVideo15): ...

    for cls in (BaseModel, HunyuanVideo, HunyuanVideoI2V, HunyuanVideoSkyreelsI2V,
                HunyuanImage21, HunyuanImage21Refiner, HunyuanVideo15,
                HunyuanVideo15_SR_Distilled):
        setattr(mb, cls.__name__, cls)
    comfy = types.ModuleType("comfy")
    comfy.model_base = mb
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_base", mb)
    return mb


def test_supported_types_all_match(model_base):
    adapter = HunyuanAdapter()
    for name in _HUNYUAN_SUPPORTED_EXACT:
        assert adapter.matches(getattr(model_base, name)()) is True


def test_refiner_subclass_matches(model_base):
    # HunyuanImage21Refiner subclasses HunyuanImage21 with no forward divergence
    # (patch [1,1,1], a channel-concat cond, disable_time_r); the exact-type list names it.
    assert HunyuanAdapter().matches(model_base.HunyuanImage21Refiner()) is True


def test_sr_distilled_matches_not_swallowed_as_10(model_base):
    # SR subclasses 1.5, which subclasses the 1.0 base. The exact-type list binds
    # it while 1.0 stays out (next test), so the list routes, not isinstance.
    assert HunyuanAdapter().matches(model_base.HunyuanVideo15_SR_Distilled()) is True


def test_hunyuan_video_10_family_raises_typed(model_base):
    adapter = HunyuanAdapter()
    for name in ("HunyuanVideo", "HunyuanVideoI2V", "HunyuanVideoSkyreelsI2V"):
        with pytest.raises(UnsupportedModelError, match="launch set"):
            adapter.matches(getattr(model_base, name)())


def test_unrelated_model_declines_without_raise(model_base):
    class SomethingElse: ...

    assert HunyuanAdapter().matches(SomethingElse()) is False


def test_contract_declared():
    adapter = HunyuanAdapter()
    assert adapter.family == "hunyuan"
    assert adapter.cfg_cond_padding == "pad+text-mask"
    assert adapter.model_base_classes == ("HunyuanImage21", "HunyuanVideo")
