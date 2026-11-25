"""ComfyUI's sample-time load is priced at the rows the forward holds (real ComfyUI, CPU).

The fault, on two DGX Sparks on 2026-11-11: UltraTex graphs at 2048 a view on
FLUX.2 Klein Base 9B rendered on one GPU and refused class C on uly2
(``partial_load_divergence``) with 75 and 81 GiB available. ComfyUI's estimate
counted every token of every view, of the normal atlas and of the reference,
212,992 latent cells and 81.1 GiB at six views, while stock's own forward
dropped the background tokens and held 90,597 rows. Each rank then offloaded
most of a 17 GiB model.

``actor/held_rows_price.py`` hands ComfyUI's estimate the kept counts for the
length of its own ``prepare_sampling``. These tests run ComfyUI's real
``prepare_sampling``, ``estimate_memory`` and ``memory_required`` on a real
``model_base`` Flux 2, Flux 1 and LongCat at toy width, with only
``load_models_gpu`` replaced by a recorder, and hold the figure it is handed
equal to stock's own formula on the counts stock's own ``extra_conds`` keeps
(``token_keep``). They also hold what the correction must never do: price
above stock, touch a render with no masks, outlive the call, or stop the
partial-load guard refusing a rank that cannot hold the model.
"""
from __future__ import annotations

import ast
import math
import types
from pathlib import Path

import pytest
import torch

import test_flux_views_comfy_exactness as views
from dgx_monarch.actor import held_rows_price as price
from dgx_monarch.actor import partial_load_guard as guard
from test_flux_views_comfy_exactness import (
    TEXT_DIM,
    _base,
    _base_flux1,
    _case_kwargs,
    _forward_inputs,
)

# The real-ComfyUI fixture of the view tests, under its own name in this module.
comfy_flux = views.comfy_flux

WORKER = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch" / "actor" / "worker.py"
SIZE = 5  # tokens a view side
GIB = 2 ** 30


def _model(comfy, family: str):
    """A real BaseModel of ``family``, its channels, patch size and text width."""
    if family == "flux2":
        return _base(comfy), 128, 1, TEXT_DIM
    base, text_dim = _base_flux1(comfy, family)
    return base, 16, 2, text_dim


def _patcher(base_model):
    import comfy.model_patcher

    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(base_model, load_device=cpu, offload_device=cpu)


def _render(comfy, family: str, case: str, batch: int = 1, cut: int = 1):
    """One render's patcher, latent shape, converted conds and stock's own kept counts."""
    import comfy.sampler_helpers as helpers

    base, channels, patch, text_dim = _model(comfy, family)
    grid, extras = _case_kwargs(case, SIZE, cut, channels=channels, patch=patch)
    _, _, conds = _forward_inputs(base, grid, extras, SIZE, batch, channels=channels,
                                  patch=patch, text_dim=text_dim, pooled=family == "flux")
    noise_shape = (len(grid) * batch, channels, patch * SIZE, patch * SIZE)
    positive = helpers.convert_cond([[torch.randn(1, 7, text_dim), dict(extras)]])
    keep = conds["token_keep"].cond if "token_keep" in conds else None
    return _patcher(base), noise_shape, positive, extras, keep, patch


def _kept_counts(extras, keep, patch):
    """Cells stock's forward keeps: the target's, and the references' summed."""
    target = len(keep["target"]) * patch * patch
    refs = 0
    for index, latent in enumerate(extras["reference_latents"]):
        kept = keep["refs"][index] if index < len(keep["refs"]) else None
        refs += math.prod(latent.shape[2:]) if kept is None else len(kept) * patch * patch
    return target, refs


def _prepare(monkeypatch, patcher, noise_shape, conds, on_load=None):
    """Run ComfyUI's own prepare_sampling and return what it hands load_models_gpu."""
    import comfy.model_management as mm
    import comfy.sampler_helpers as helpers

    seen: dict = {}

    def load_models_gpu(models, memory_required=0, force_patch_weights=False,
                        minimum_memory_required=None, force_full_load=False):
        seen.update(memory_required=memory_required, minimum=minimum_memory_required,
                    force_full_load=force_full_load,
                    shadowed=[name for name in price._SHADOWED if name in patcher.model.__dict__])
        if on_load is not None:
            on_load(models[0], minimum_memory_required)

    monkeypatch.setattr(mm, "load_models_gpu", load_models_gpu)
    helpers.prepare_sampling(patcher, noise_shape, conds, model_options=patcher.model_options)
    return seen


def _stock(monkeypatch, comfy, family, case, batch=1, cut=1):
    patcher, noise_shape, positive, *_ = _render(comfy, family, case, batch, cut)
    return _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})


# The masked-only cases whose held rows stay under stock's count: every one at one sample,
# and at two samples those whose token drop outweighs the references held a second time.
# ("token_keep" at two samples holds more than stock counts: the fallback test below.)
PRICED = [(1, "all"), (1, "four_views"), (1, "token_keep"), (1, "view_refs"),
          (2, "all"), (2, "four_views"), (2, "view_refs")]


@pytest.mark.parametrize("family", ["flux2", "flux", "longcat"])
@pytest.mark.parametrize(("batch", "case"), PRICED)
def test_the_load_is_priced_at_the_cells_stocks_forward_keeps(monkeypatch, comfy_flux, family, case, batch):
    patcher, noise_shape, positive, extras, keep, patch = _render(comfy_flux, family, case, batch)
    price.install(patcher, family)
    seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})

    target, refs = _kept_counts(extras, keep, patch)
    base = patcher.model
    # The minimum is one call, which holds the kept target and every kept reference once a
    # sample. Stock's own formula on those counts.
    assert seen["minimum"] == base.memory_required(
        [batch, noise_shape[1], target], cond_shapes={"ref_latents": [[1, 16, refs * batch]]})
    # Cond and uncond in one call only decides what to evict: the kept references once.
    assert seen["memory_required"] == base.memory_required(
        [2 * batch, noise_shape[1], target], cond_shapes={"ref_latents": [[1, 16, refs]]})
    stock = _stock(monkeypatch, comfy_flux, family, case, batch)
    assert seen["minimum"] < stock["minimum"] and seen["memory_required"] < stock["memory_required"]
    assert seen["shadowed"] == list(price._SHADOWED) and not seen["force_full_load"]
    assert not any(name in base.__dict__ for name in price._SHADOWED), "the correction outlived the call"


@pytest.mark.parametrize("case", ["all", "four_views", "view_refs"])
@pytest.mark.parametrize("samples", [1, 2, 3])
def test_the_largest_call_counts_the_rows_stocks_forward_holds(comfy_flux, case, samples):
    """Stock's Flux._forward holds every kept reference once a sample (it takes r[::n]), so the
    count must equal the image rows forward_orig receives, at any number of samples."""
    import comfy.sampler_helpers as helpers

    base = _base(comfy_flux)
    grid, extras = _case_kwargs(case, SIZE, 1, channels=128, patch=1)
    (x, timestep, context), processed, _ = _forward_inputs(
        base, grid, extras, SIZE, samples, channels=128, patch=1)
    dm, held = base.diffusion_model, {}
    original = dm.forward_orig

    def record(img, *args, **kwargs):
        held["rows"] = int(img.shape[0] * img.shape[1])
        return original(img, *args, **kwargs)

    dm.forward_orig = record
    with torch.no_grad():
        dm(x, timestep, context, **processed)
    positive = helpers.convert_cond([[torch.randn(1, 7, TEXT_DIM), dict(extras)]])
    cells = price._call_cells(base, tuple(x.shape), {"positive": positive})
    assert sum(cells.largest) == held["rows"]


def _bare_negative(positive):
    """ConditioningZeroOut of the bare prompt: no masks, no grid, no reference."""
    import comfy.sampler_helpers as helpers

    return helpers.convert_cond([[torch.zeros(1, 7, positive[0]["cross_attn"].shape[-1]), {}]])


@pytest.mark.parametrize(("family", "case", "negative"), [
    ("flux2", "token_keep", False), ("flux", "token_keep", False), ("longcat", "token_keep", False),
    ("flux2", "all", True), ("flux", "all", True)])
def test_a_call_that_holds_more_than_stock_counts_keeps_stocks_figure(
        monkeypatch, comfy_flux, caplog, family, case, negative):
    """Two samples a call. Stock counts each reference once a call and the forward holds it
    once a sample, so with little dropped the held rows pass stock's own count. This price
    never goes above stock's: both figures are stock's, nothing is bound on the model, and
    the journal says why. No warning: this is no fault."""
    cut = 0 if negative else 1
    patcher, noise_shape, positive, extras, keep, patch = _render(comfy_flux, family, case, 2, cut)
    conds = {"positive": positive, **({"negative": _bare_negative(positive)} if negative else {})}
    stock = _prepare(monkeypatch, _patcher(patcher.model), noise_shape, conds)
    target, refs = _kept_counts(extras, keep, patch)
    held_rows = 2 * (target + refs)
    stock_cells = price._cells(noise_shape) + sum(
        math.prod(latent.shape[2:]) for latent in extras["reference_latents"])
    assert held_rows > stock_cells, "the case must be one whose forward holds more than stock counts"
    price.install(patcher, family)
    with caplog.at_level("INFO", logger="dgx_monarch.actor.held_rows_price"):
        seen = _prepare(monkeypatch, patcher, noise_shape, conds)
    assert (seen["minimum"], seen["memory_required"]) == (stock["minimum"], stock["memory_required"])
    assert seen["shadowed"] == []
    lines = [(record.levelname, record.getMessage()) for record in caplog.records]
    assert [level for level, _ in lines] == ["INFO"], lines
    assert f"holds {held_rows} latent cells and ComfyUI's estimate counts {stock_cells}" in lines[0][1]
    assert "ComfyUI's own estimate stands" in lines[0][1]
    assert price.held_cells(patcher.model, noise_shape, conds) is None


@pytest.mark.parametrize("family", ["flux2", "flux"])
def test_a_mask_that_keeps_every_token_is_stocks_own_figure(monkeypatch, comfy_flux, family):
    patcher, noise_shape, positive, *_ = _render(comfy_flux, family, "all")
    masks = positive[0]["token_masks"]
    masks["target"] = torch.ones_like(masks["target"])
    masks["refs"] = [None if mask is None else torch.ones_like(mask) for mask in masks["refs"]]
    assert price.held_cells(patcher.model, noise_shape, {"positive": positive}) is None
    price.install(patcher, family)
    seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    plain = _patcher(patcher.model)
    assert seen["minimum"] == _prepare(monkeypatch, plain, noise_shape, {"positive": positive})["minimum"]
    assert seen["shadowed"] == []


@pytest.mark.parametrize("case", ["view_grid", "grid_refs"])
def test_a_render_with_no_masks_is_untouched(monkeypatch, comfy_flux, case):
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", case)
    assert price.held_cells(patcher.model, noise_shape, {"positive": positive}) is None
    price.install(patcher, "flux2")
    seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    stock = _stock(monkeypatch, comfy_flux, "flux2", case)
    assert (seen["minimum"], seen["memory_required"]) == (stock["minimum"], stock["memory_required"])
    assert seen["shadowed"] == []


@pytest.mark.parametrize("family", ["flux2", "longcat"])
@pytest.mark.parametrize("cut", [0, 1, 2, 3])
def test_the_price_never_exceeds_stocks(monkeypatch, comfy_flux, family, cut):
    patcher, noise_shape, positive, *_ = _render(comfy_flux, family, "all", cut=cut)
    price.install(patcher, family)
    seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    stock = _stock(monkeypatch, comfy_flux, family, "all", cut=cut)
    assert seen["minimum"] <= stock["minimum"] and seen["memory_required"] <= stock["memory_required"]


@pytest.mark.parametrize("family", ["flux2", "flux"])
@pytest.mark.parametrize(("cut", "batch"), [(0, 1), (3, 1), (3, 2)])
def test_a_bare_negative_is_priced_as_its_own_call(monkeypatch, comfy_flux, family, cut, batch):
    """The hardware graphs' form: the negative is ConditioningZeroOut of the bare prompt, so
    it carries no masks and no reference. ComfyUI hands it to prepare_sampling at CFG 1 too.
    Its call holds the whole latent and nothing else; the positive's holds its kept rows. The
    minimum is the larger of the two calls, not stock's mix of one cond's latent with the
    other's references. ``cut`` puts each call on top in turn. (Little dropped at two samples
    holds more than stock counts: the fallback test above.)"""
    patcher, noise_shape, positive, extras, keep, patch = _render(comfy_flux, family, "all", batch, cut)
    conds = {"positive": positive, "negative": _bare_negative(positive)}
    stock = _prepare(monkeypatch, _patcher(patcher.model), noise_shape, conds)
    price.install(patcher, family)
    seen = _prepare(monkeypatch, patcher, noise_shape, conds)

    base, channels = patcher.model, noise_shape[1]
    target, refs = _kept_counts(extras, keep, patch)
    latent = noise_shape[0] * noise_shape[2] * noise_shape[3]
    # The masked call holds its kept target and its kept references once a sample.
    kept_call = base.memory_required(
        [batch, channels, target], cond_shapes={"ref_latents": [[1, 16, refs * batch]]})
    bare_call = base.memory_required(list(noise_shape))
    assert seen["minimum"] == max(kept_call, bare_call) < stock["minimum"]
    assert (kept_call > bare_call) == (batch * (target + refs) > latent), "the larger call must win"
    # Cond and uncond in one call: the whole latent twice, and the kept references.
    assert seen["memory_required"] == base.memory_required(
        [2 * noise_shape[0], *noise_shape[1:]], cond_shapes={"ref_latents": [[1, 16, refs]]})
    assert seen["memory_required"] < stock["memory_required"]
    held = price.held_cells(base, noise_shape, conds)
    assert held.target == held.stock[0] == latent
    assert sum(held.largest) == max(batch * (target + refs), latent)


def test_a_negative_that_holds_references_and_no_masks_keeps_its_own_counts(monkeypatch, comfy_flux):
    """A cond with references and no masks holds every row of both: its call is stock's."""
    import comfy.sampler_helpers as helpers

    patcher, noise_shape, positive, extras, keep, patch = _render(comfy_flux, "flux2", "all")
    bare = {key: value for key, value in extras.items()
            if key not in ("token_masks", "view_grid", "reference_views", "background_latent")}
    negative = helpers.convert_cond([[torch.randn(1, 7, TEXT_DIM), bare]])
    conds = {"positive": positive, "negative": negative}
    held = price.held_cells(patcher.model, noise_shape, conds)
    assert held is not None and held.largest == held.stock
    price.install(patcher, "flux2")
    seen = _prepare(monkeypatch, patcher, noise_shape, conds)
    full_refs = sum(math.prod(latent.shape[2:]) for latent in extras["reference_latents"])
    # The minimum is the negative's own call, which is stock's figure.
    assert seen["minimum"] == patcher.model.memory_required(
        list(noise_shape), cond_shapes={"ref_latents": [[1, 16, full_refs]]})
    _target, kept_refs = _kept_counts(extras, keep, patch)
    assert seen["memory_required"] == patcher.model.memory_required(
        [2 * noise_shape[0], *noise_shape[1:]],
        cond_shapes={"ref_latents": [[1, 16, kept_refs], [1, 16, full_refs]]})


def test_a_negative_with_the_same_masks_is_priced_at_the_kept_rows(monkeypatch, comfy_flux):
    """ConditioningZeroOut of the UltraTex conditioning keeps its extras: both conds drop alike."""
    import comfy.sampler_helpers as helpers

    patcher, noise_shape, positive, extras, keep, patch = _render(comfy_flux, "flux2", "all")
    negative = helpers.convert_cond([[torch.zeros(1, 7, TEXT_DIM), dict(extras)]])
    price.install(patcher, "flux2")
    seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive, "negative": negative})
    target, refs = _kept_counts(extras, keep, patch)
    base = patcher.model
    assert seen["minimum"] == base.memory_required(
        [1, noise_shape[1], target], cond_shapes={"ref_latents": [[1, 16, refs]]})
    assert seen["memory_required"] == base.memory_required(
        [2, noise_shape[1], target], cond_shapes={"ref_latents": [[1, 16, refs], [1, 16, refs]]})


def test_the_fsdp_wrapped_model_is_priced_as_its_plain_twin(monkeypatch, comfy_flux, tmp_path):
    """uly2+fsdp: the wrap of adapters/fsdp.py (tests/test_flux_views_comfy_exactness.py) turns
    every block parameter into a DTensor. The price reads the model's patch size and stock's
    shapes only, no parameter, so the wrapped model gets the plain model's figure."""
    import torch.distributed.fsdp
    from torch.distributed.device_mesh import init_device_mesh

    from dgx_monarch.adapters.fsdp import _BLOCK_LIST_ATTRS
    from dgx_monarch.adapters.fsdp_islands import resolve_module_path

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    twin, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    price.install(twin, "flux2")
    want = _prepare(monkeypatch, twin, noise_shape, {"positive": positive})
    wrapped, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        dm = wrapped.model.diffusion_model
        for parameter in dm.parameters():
            parameter.requires_grad = False
        for attr in _BLOCK_LIST_ATTRS:
            for block in resolve_module_path(dm, attr) or ():
                torch.distributed.fsdp.fully_shard(block, mesh=mesh)
        torch.distributed.fsdp.fully_shard(dm, mesh=mesh, reshard_after_forward=True)
        assert type(dm.img_in.weight).__name__ == "DTensor"
        price.install(wrapped, "flux2")
        seen = _prepare(monkeypatch, wrapped, noise_shape, {"positive": positive})
    finally:
        torch.distributed.destroy_process_group()
    assert seen["shadowed"] == list(price._SHADOWED)
    assert (seen["minimum"], seen["memory_required"]) == (want["minimum"], want["memory_required"])


def test_an_error_in_the_count_leaves_stocks_estimate(monkeypatch, comfy_flux, caplog):
    stock = _stock(monkeypatch, comfy_flux, "flux2", "all")
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    price.install(patcher, "flux2")

    def broken(mask, h, w):
        raise RuntimeError("mask rule moved")

    monkeypatch.setattr(comfy_flux.model, "foreground_tokens", broken)
    with caplog.at_level("WARNING", logger="dgx_monarch.actor.held_rows_price"):
        seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    assert seen["minimum"] == stock["minimum"] and seen["shadowed"] == []
    assert any("left at ComfyUI's own estimate" in record.getMessage() for record in caplog.records)


def test_a_price_above_stocks_is_dropped(monkeypatch, comfy_flux, caplog):
    """The second read of the estimate is the backstop for a count this module got wrong."""
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    price.install(patcher, "flux2")
    real = price.held_cells

    def inflated(base_model, shape, conds):
        held = real(base_model, shape, conds)
        return price.HeldCells(batch=held.batch, stock=held.stock, refs=held.refs,
                               largest=(held.stock[0] * 3, held.largest[1]), target=held.target)

    monkeypatch.setattr(price, "held_cells", inflated)
    with caplog.at_level("WARNING", logger="dgx_monarch.actor.held_rows_price"):
        seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    stock = _stock(monkeypatch, comfy_flux, "flux2", "all")
    assert seen["minimum"] == stock["minimum"] and seen["shadowed"] == []
    assert any("above stock's" in record.getMessage() for record in caplog.records)


def test_a_failed_load_still_restores_the_estimate(monkeypatch, comfy_flux):
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    price.install(patcher, "flux2")

    def fail(_model, _minimum):
        raise MemoryError("load failed")

    with pytest.raises(MemoryError):
        _prepare(monkeypatch, patcher, noise_shape, {"positive": positive}, on_load=fail)
    assert not any(name in patcher.model.__dict__ for name in price._SHADOWED)


def test_the_journal_names_both_figures(monkeypatch, comfy_flux, caplog):
    patcher, noise_shape, positive, extras, keep, patch = _render(comfy_flux, "flux2", "all")
    price.install(patcher, "flux2")
    with caplog.at_level("INFO", logger="dgx_monarch.actor.held_rows_price"):
        _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    target, refs = _kept_counts(extras, keep, patch)
    line = next(record.getMessage() for record in caplog.records)
    stock_cells = noise_shape[0] * noise_shape[2] * noise_shape[3] + sum(
        math.prod(latent.shape[2:]) for latent in extras["reference_latents"])
    assert f"counts {stock_cells} latent cells" in line
    assert f"the largest call any cond makes holds {target + refs} ({target} target, {refs} reference)" in line
    assert "GiB in place of" in line and "token drop" not in line


def test_the_journal_line_for_a_bare_negative_claims_no_drop(monkeypatch, comfy_flux, caplog):
    """The largest call there is the negative's whole latent: nothing was dropped from it."""
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all", cut=3)
    price.install(patcher, "flux2")
    with caplog.at_level("INFO", logger="dgx_monarch.actor.held_rows_price"):
        _prepare(monkeypatch, patcher, noise_shape,
                 {"positive": positive, "negative": _bare_negative(positive)})
    latent = price._cells(noise_shape)
    line = next(record.getMessage() for record in caplog.records)
    assert f"the largest call any cond makes holds {latent} ({latent} target, 0 reference)" in line


def _comfy_rule(free: float, reserve: float):
    """ComfyUI's load decision as ``load_models_gpu`` takes it for a model not yet on
    the device: the weights stay whole only if they fit in what free memory less the
    estimate and the reserve leaves. It writes the flag the partial-load guard reads."""

    def decide(patcher, minimum_memory_required):
        budget = max(0.0, free - (minimum_memory_required + reserve))
        patcher.model.model_lowvram = budget < patcher.model_size()

    return decide


def _pair_with_a_whole_peer(monkeypatch) -> None:
    """World 2 whose other rank is whole: the MIN over the pair is this rank's own flag."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)
    monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None, group=None, async_op=False: None)


def _figures(monkeypatch, comfy):
    """Stock's and the kept-row minimum for one render, and its model size."""
    stock = _stock(monkeypatch, comfy, "flux2", "all")["minimum"]
    patcher, noise_shape, positive, *_ = _render(comfy, "flux2", "all")
    price.install(patcher, "flux2")
    kept = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})["minimum"]
    return stock, kept, patcher.model_size()


def test_a_rank_that_fits_the_kept_rows_loads_whole_and_the_guard_passes(monkeypatch, comfy_flux):
    """The hardware case in small: room for the kept rows, not for stock's full count."""
    stock, kept, size = _figures(monkeypatch, comfy_flux)
    free = size + (stock + kept) / 2  # between the two prices, reserve 0
    _pair_with_a_whole_peer(monkeypatch)

    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    guard.install(patcher, 2, {"ulysses": 2})
    price.install(patcher, "flux2")
    _prepare(monkeypatch, patcher, noise_shape, {"positive": positive}, on_load=_comfy_rule(free, 0))
    assert guard.local_full_load(patcher.model)

    # The same box without the correction: ComfyUI offloads, and the guard refuses.
    unpriced, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    guard.install(unpriced, 2, {"ulysses": 2})
    with pytest.raises(guard.PartialLoadDivergenceError):
        _prepare(monkeypatch, unpriced, noise_shape, {"positive": positive},
                 on_load=_comfy_rule(free, 0))


def test_a_rank_that_cannot_hold_the_model_still_refuses_class_c(monkeypatch, comfy_flux):
    _stock_figure, kept, size = _figures(monkeypatch, comfy_flux)
    free = size + kept / 2  # short of the kept-row price too
    _pair_with_a_whole_peer(monkeypatch)
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    guard.install(patcher, 2, {"ulysses": 2})
    price.install(patcher, "flux2")
    with pytest.raises(guard.PartialLoadDivergenceError) as refused:
        _prepare(monkeypatch, patcher, noise_shape, {"positive": positive},
                 on_load=_comfy_rule(free, 0))
    assert "[dgxm:C guard=partial_load_divergence waivable=0]" in str(refused.value)
    assert not any(name in patcher.model.__dict__ for name in price._SHADOWED)


def test_the_cond_batch_rule_reads_stocks_figure_after_the_load(monkeypatch, comfy_flux):
    """``_calc_cond_batch`` prices a call with the same method during sampling, cond shapes given."""
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    shapes = {"ref_latents": [[1, 16, 7 * SIZE * SIZE]]}
    before = patcher.model.memory_required(list(noise_shape), cond_shapes=shapes)
    price.install(patcher, "flux2")
    _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    assert patcher.model.memory_required(list(noise_shape), cond_shapes=shapes) == before


# The guards around the two bound methods, each on the input that needs it.

def _shadowed(comfy, family: str = "flux2"):
    """A model with the correction bound, as it is inside prepare_sampling: the model,
    the latent shape, what was counted, stock's own two methods, and the restore."""
    patcher, noise_shape, positive, *_ = _render(comfy, family, "all")
    base = patcher.model
    held = price.held_cells(base, noise_shape, {"positive": positive})
    stock = (base.extra_conds_shapes, base.memory_required)
    saved = price._shadow(base, noise_shape, held)
    return base, noise_shape, positive, held, stock, lambda: price._restore(base, saved)


def test_kept_target_cells_never_pass_the_latents_own_cells(comfy_flux):
    """An odd latent side under patch 2: three token rows of two cells cover six, the latent
    has five. Stock counts the latent's own cells, so the kept count stops there."""
    rule = comfy_flux.model.foreground_tokens
    cond = {"token_masks": {"target": torch.ones(1, 80, 80)}, "view_grid": [(0, 0)]}
    assert price._kept_cells(rule, cond["token_masks"]["target"], 5, 5, 2) == 36
    assert price._target_cells(rule, cond, (1, 16, 5, 5), 2) == 25
    assert price._target_cells(rule, cond, (2, 16, 5, 5), 2) == 50, "two samples of one view"


def test_a_reference_total_that_is_not_stocks_is_never_replaced(comfy_flux):
    """If ComfyUI counted references another way than this module reads, the module's kept
    count has no place in stock's shape: the cond keeps what stock reported."""
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    base = patcher.model
    own = price.held_cells(base, noise_shape, {"positive": positive})
    assert own.refs, "with stock's own total the masked references are replaced"
    real = base.extra_conds_shapes(**positive[0])["ref_latents"]
    other = [real[0], real[1], real[2] + 7]
    stub = types.SimpleNamespace(diffusion_model=base.diffusion_model,
                                 extra_conds_shapes=lambda **cond: {"ref_latents": other})
    held = price._call_cells(stub, noise_shape, {"positive": positive})
    assert held.refs == {} and held.largest[1] == real[2] + 7 and held.stock[1] == real[2] + 7


def test_the_minimum_keeps_stocks_figure_for_a_reference_layout_it_did_not_read(comfy_flux):
    """The minimum rewrites one ``[1, channels, cells]`` shape. Two shapes, or another
    layout, are not what it read, so the call goes to stock as it came."""
    base, noise_shape, _positive, held, (_shapes, stock_required), restore = _shadowed(comfy_flux)
    try:
        assert held.largest[1] > 0
        for odd in ([[1, 16, 10], [1, 16, 20]], [[1, 16, 5, 5]]):
            given = {"ref_latents": odd}
            assert base.memory_required(list(noise_shape), cond_shapes=given) == stock_required(
                list(noise_shape), cond_shapes=given), odd
    finally:
        restore()


def test_a_reference_shape_in_another_layout_is_left_as_stock_gave_it(comfy_flux):
    """The bound ``extra_conds_shapes`` replaces only ``[1, channels, stock's total]``."""
    _base_model, noise_shape, positive, held, _stock_methods, restore = _shadowed(comfy_flux)
    restore()
    masks = positive[0]["token_masks"]
    _kept, full = held.refs[id(masks)]
    for odd in ([2, 16, full], [1, 16, full + 3], [1, 16, 4, full // 4]):
        stub = types.SimpleNamespace(extra_conds_shapes=lambda odd=odd, **cond: {"ref_latents": odd},
                                     memory_required=lambda shape, cond_shapes=None: 0.0)
        saved = price._shadow(stub, noise_shape, held)
        assert stub.extra_conds_shapes(token_masks=masks) == {"ref_latents": odd}, odd
        price._restore(stub, saved)
    # And the layout it did read is replaced, so the loop above tests the guard, not a no-op.
    stub = types.SimpleNamespace(extra_conds_shapes=lambda **cond: {"ref_latents": [1, 16, full]},
                                 memory_required=lambda shape, cond_shapes=None: 0.0)
    price._shadow(stub, noise_shape, held)
    assert stub.extra_conds_shapes(token_masks=masks) == {"ref_latents": [1, 16, held.refs[id(masks)][0]]}


def test_a_batch_the_latent_does_not_divide_is_priced_by_stock(comfy_flux):
    """Stock passes this render's latent at its batch or twice it. Any other batch is not a
    call this module counted, so stock prices it."""
    base, noise_shape, _positive, held, (_shapes, stock_required), restore = _shadowed(comfy_flux)
    try:
        shapes = {"ref_latents": [[1, 16, held.stock[1]]]}
        odd = [held.batch + held.batch // 2, *noise_shape[1:]]
        assert odd[0] % held.batch
        assert base.memory_required(odd, cond_shapes=shapes) == stock_required(odd, cond_shapes=shapes)
        # The latent's own batch is counted, so the guard above is what kept the odd one stock's.
        assert base.memory_required(list(noise_shape), cond_shapes=shapes) < stock_required(
            list(noise_shape), cond_shapes=shapes)
    finally:
        restore()


def test_a_masked_target_beside_a_bare_negative_corrects_with_no_masked_reference(
        monkeypatch, comfy_flux):
    """Only the positive's target is masked: its references keep every token, and the bare
    negative holds the whole latent. No reference count moves and stock's target is some
    cond's own, yet the largest call is still under stock's mix of the two."""
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all", cut=3)
    positive[0]["token_masks"]["refs"] = [None for _ in positive[0]["token_masks"]["refs"]]
    conds = {"positive": positive, "negative": _bare_negative(positive)}
    held = price.held_cells(patcher.model, noise_shape, conds)
    assert held is not None and held.refs == {} and held.target == held.stock[0]
    assert sum(held.largest) < sum(held.stock)
    stock = _prepare(monkeypatch, _patcher(patcher.model), noise_shape, conds)
    price.install(patcher, "flux2")
    seen = _prepare(monkeypatch, patcher, noise_shape, conds)
    assert seen["minimum"] < stock["minimum"] and seen["memory_required"] == stock["memory_required"]


# The mask rule itself: its name, and what happens the day ComfyUI moves it.

def test_comfyui_still_names_the_mask_rule_this_price_counts_with(comfy_flux):
    """The canary's check: the rule ``preprocess_token_masks`` applies is the function this
    module calls, under the name and arguments it calls it with."""
    import inspect

    rule = comfy_flux.model.foreground_tokens
    assert list(inspect.signature(rule).parameters) == ["mask", "h", "w"]
    assert "foreground_tokens(" in inspect.getsource(comfy_flux.model.Flux.preprocess_token_masks)


def test_a_comfyui_that_moved_the_mask_rule_is_told_at_install_and_at_the_render(
        monkeypatch, comfy_flux, caplog):
    """Renamed upstream, the rule would leave UltraTex on stock's whole-token estimate and a
    pair refusing at 2048 again. Never in silence: one warning when the wrapper is installed,
    one a render that carries masks, and stock's figures."""
    stock = _stock(monkeypatch, comfy_flux, "flux2", "all")
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    monkeypatch.delattr(comfy_flux.model, "foreground_tokens")
    with caplog.at_level("INFO", logger="dgx_monarch.actor.held_rows_price"):
        price.install(patcher, "flux2")
        seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    assert (seen["minimum"], seen["memory_required"]) == (stock["minimum"], stock["memory_required"])
    assert seen["shadowed"] == []
    lines = [(record.levelname, record.getMessage()) for record in caplog.records]
    assert [level for level, _ in lines] == ["WARNING", "WARNING"], lines
    assert all("has no comfy.ldm.flux.model.foreground_tokens" in text for _, text in lines)
    assert "keeps ComfyUI's own estimate" in lines[0][1] and "partial_load_divergence" in lines[0][1]
    assert "left at ComfyUI's own estimate" in lines[1][1]


def test_a_comfyui_with_the_rule_or_with_no_token_drop_installs_quietly(monkeypatch, comfy_flux, caplog):
    patcher, *_ = _render(comfy_flux, "flux2", "all")
    with caplog.at_level("INFO", logger="dgx_monarch.actor.held_rows_price"):
        price.install(patcher, "flux2")
        # A ComfyUI from before UltraTex: no token drop in the forward, so no rule to miss.
        older = _patcher(patcher.model)
        monkeypatch.delattr(comfy_flux.model, "foreground_tokens")
        monkeypatch.delattr(comfy_flux.model.Flux, "preprocess_token_masks")
        price.install(older, "flux2")
    assert caplog.records == []


def test_install_takes_the_families_whose_forward_drops_tokens_once(comfy_flux):
    import comfy.patcher_extension as pe

    class _Patcher:
        def __init__(self):
            self.model_options: dict = {}

    def installed(patcher):
        return pe.get_wrappers_with_key(
            pe.WrappersMP.PREPARE_SAMPLING, price.WRAPPER_KEY,
            patcher.model_options, is_model_options=True)

    for family in ("chroma", "wan", "unknown", None):
        other = _Patcher()
        price.install(other, family)
        assert installed(other) == [], family
    for family in ("flux", "flux2", "longcat"):
        patcher = _Patcher()
        price.install(patcher, family)
        price.install(patcher, family)
        assert len(installed(patcher)) == 1, family


def test_the_worker_installs_it_at_every_world() -> None:
    """One GPU too: a one-GPU Worker that offloads is not the reference a whole pair matches."""
    tree = ast.parse(WORKER.read_text())
    inject = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_inject_for_topology")
    calls = [statement for statement in inject.body
             if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
             and ast.unparse(statement.value.func) == "held_rows_price.install"]
    assert len(calls) == 1, "held_rows_price.install must sit at the function's top level, under no world gate"
