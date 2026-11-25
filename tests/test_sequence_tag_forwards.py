"""The sequence tag says a pair refuses exactly where the sharded forward refuses it.

For each mask branch of reference_preflight.sequence_split_refused, ComfyUI's
own extra_conds (comfy/model_base.py) turns the conditioning into what the
model hands its forward, and the real sharded forward then runs on or refuses.
The tag must answer True exactly where that forward refuses: a tag that says
less sends a pair to a route that refuses it, one that says more keeps cfg2
where Ulysses would run the pair bytes equal. Real ComfyUI only: the file
skips where none is importable, and the comfy-canary job runs it against
master and the newest release.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import test_boogu_adapter as boogu_test
import test_flux_adapter as flux_test
import test_qwen_image_adapter as qwen_test
from dgx_monarch.adapters import base
from dgx_monarch.adapters import boogu as bg
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.flux_family import ChromaAdapter
from dgx_monarch.nodes.reference_preflight import sequence_split_refused
from real_comfy_helpers import real_comfy
from test_pair_scope_fallback_runs import _model, _pair, checkpoint_files

_NOISE = torch.zeros(1, 16, 2, 4)  # 1 x 2 image tokens at patch 2


@pytest.fixture
def files(monkeypatch, tmp_path):
    return checkpoint_files(monkeypatch, tmp_path)


class _RanOn(Exception):
    """The first step after a forward's guards: the forward ran on."""


def _ran_on(*_args, **_kwargs):
    raise _RanOn


def _forward_refuses(model_class: str, entry: dict, forward) -> bool:
    """Whether ``forward`` refuses what ComfyUI's extra_conds hands it for one entry.

    Runs under real ComfyUI, which the forwards import; the tag is read before,
    against the test's own checkpoint reader.
    """
    with real_comfy("sequence-tag", "comfy.model_base") as (_ops, model_base):
        cls = getattr(model_base, model_class)
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model.diffusion_model = SimpleNamespace(patch_size=2)
        model.concat_cond = lambda **_kwargs: None
        conds = model.extra_conds(noise=_NOISE, device="cpu",
                                  cross_attn=torch.zeros(1, 3, 4), **entry)
        published = {name: cond.cond for name, cond in conds.items()}
        try:
            forward(published)
        except UnsupportedModelError:
            return True
        except _RanOn:
            return False
        return False


def _tag(family, extra) -> bool:
    return sequence_split_refused(_model(family), *_pair(extra, extra, rows=3))


@pytest.mark.parametrize("extra", [
    {},
    {"attention_mask": torch.ones(1, 3)},
    {"attention_mask": torch.tensor([[1, 1, 0]])},
])
def test_qwen_image_tags_exactly_the_masks_its_sharded_forward_refuses(files, extra):
    files("F8_E4M3")
    tagged = _tag("qwen_image", extra)
    sharded = qwen_test._usp_fake()
    qwen_test._inject_usp(sharded)
    sharded.process_img = _ran_on  # its first step after the mask guard
    refused = _forward_refuses("QwenImage", extra, lambda published: sharded._forward(
        torch.zeros(1, 16, 1, 8, 8), torch.zeros(1), torch.zeros(1, 3, 16),
        attention_mask=published.get("attention_mask")))
    assert refused is tagged is ("attention_mask" in extra)


def _hiding(size: int, row: int, col: int) -> torch.Tensor:
    mask = torch.zeros(1, size, size)
    mask[0, row, col] = float("-inf")
    return mask


@pytest.mark.parametrize(("extra", "refuses"), [
    # A text encoder's keep-mask: ComfyUI hands the flux family no mask.
    ({"attention_mask": torch.tensor([[1, 1, 0]])}, False),
    # A regional mask at the latent's token grid, hiding a key or nothing.
    ({"attention_mask": _hiding(5, 0, 4), "attention_mask_img_shape": (1, 2)}, True),
    ({"attention_mask": torch.zeros(1, 5, 5), "attention_mask_img_shape": (1, 2)}, False),
    # Drawn on a 2 x 2 grid, so ComfyUI resizes it to 1 x 2 first.
    ({"attention_mask": _hiding(7, 0, 1), "attention_mask_img_shape": (2, 2)}, True),
    ({"attention_mask": torch.zeros(1, 7, 7), "attention_mask_img_shape": (2, 2)}, False),
])
def test_chroma_tags_exactly_the_masks_its_sharded_forward_refuses(files, extra, refuses):
    files("BF16")
    tagged = _tag("chroma", extra)
    model = flux_test._FakeChromaDoubleBlockOnly()
    ChromaAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=None))
    model.img_in = _ran_on  # its first step after the mask guard
    # Chroma.forward hands the published mask to forward_orig as attn_mask.
    refused = _forward_refuses("Chroma", extra, lambda published: model.forward_orig(
        torch.zeros(1, 2, 4), torch.zeros(1, 2, 3), torch.zeros(1, 3, 4), torch.zeros(1, 3, 3),
        torch.zeros(1), guidance=torch.zeros(1), attn_mask=published.get("attention_mask")))
    assert refused is tagged is refuses


@pytest.mark.parametrize("mask", [torch.ones(1, 3), torch.tensor([[1, 1, 0]])])
def test_boogu_tags_exactly_the_caption_masks_its_sharded_forward_refuses(files, monkeypatch,
                                                                         mask):
    files("F8_E4M3")
    extra = {"attention_mask": mask}
    tagged = _tag("boogu", extra)
    # One rank: identity shards, as test_boogu_adapter's comfy_stubs set them.
    monkeypatch.setattr(bg, "shard_seq", lambda t, dim=1: (t, t.shape[dim]))
    monkeypatch.setattr(bg, "sp_gather", lambda t, orig, dim=1: t)
    monkeypatch.setattr(base, "sp_world", lambda: 1)
    model = boogu_test._FakeBoogu([], with_ref=False)
    boogu_test._inject(model)
    # Omnigen2.extra_conds sets num_tokens to the mask's kept rows.
    refused = _forward_refuses("Boogu", extra, lambda published: model.forward(
        torch.zeros(1, 1, 4, 4), torch.zeros(1), torch.ones(1, 3, 4), published["num_tokens"],
        attention_mask=published.get("attention_mask"), transformer_options={}))
    assert refused is tagged is bool((mask == 0).any())
