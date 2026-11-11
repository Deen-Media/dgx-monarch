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


@pytest.mark.parametrize("family", ["flux2", "flux", "longcat"])
@pytest.mark.parametrize("case", ["all", "four_views", "token_keep", "view_refs"])
@pytest.mark.parametrize("batch", [1, 2])
def test_the_load_is_priced_at_the_cells_stocks_forward_keeps(monkeypatch, comfy_flux, family, case, batch):
    patcher, noise_shape, positive, extras, keep, patch = _render(comfy_flux, family, case, batch)
    price.install(patcher, family)
    seen = _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})

    target, refs = _kept_counts(extras, keep, patch)
    base = patcher.model
    shapes = {"ref_latents": [[1, 16, refs]]}
    # Stock's own formula on the kept counts, at its batch and twice it.
    assert seen["minimum"] == base.memory_required([batch, noise_shape[1], target], cond_shapes=shapes)
    assert seen["memory_required"] == base.memory_required(
        [2 * batch, noise_shape[1], target], cond_shapes=shapes)
    stock = _stock(monkeypatch, comfy_flux, family, case, batch)
    assert seen["minimum"] < stock["minimum"] and seen["memory_required"] < stock["memory_required"]
    assert seen["shadowed"] == list(price._SHADOWED) and not seen["force_full_load"]
    assert not any(name in base.__dict__ for name in price._SHADOWED), "the correction outlived the call"


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
@pytest.mark.parametrize("cut", [0, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_a_bare_negative_is_priced_as_its_own_call(monkeypatch, comfy_flux, family, cut, batch):
    """The hardware graphs' form: the negative is ConditioningZeroOut of the bare prompt, so
    it carries no masks and no reference. ComfyUI hands it to prepare_sampling at CFG 1 too.
    Its call holds the whole latent and nothing else; the positive's holds its kept rows. The
    minimum is the larger of the two calls, not stock's mix of one cond's latent with the
    other's references. ``cut`` puts each call on top in turn."""
    import comfy.sampler_helpers as helpers

    patcher, noise_shape, positive, extras, keep, patch = _render(comfy_flux, family, "all", batch, cut)
    text_dim = positive[0]["cross_attn"].shape[-1]
    negative = helpers.convert_cond([[torch.zeros(1, 7, text_dim), {}]])
    conds = {"positive": positive, "negative": negative}
    stock = _prepare(monkeypatch, _patcher(patcher.model), noise_shape, conds)
    price.install(patcher, family)
    seen = _prepare(monkeypatch, patcher, noise_shape, conds)

    base, channels = patcher.model, noise_shape[1]
    target, refs = _kept_counts(extras, keep, patch)
    latent = noise_shape[0] * noise_shape[2] * noise_shape[3]
    kept_call = base.memory_required([batch, channels, target], cond_shapes={"ref_latents": [[1, 16, refs]]})
    bare_call = base.memory_required(list(noise_shape))
    assert seen["minimum"] == max(kept_call, bare_call) < stock["minimum"]
    assert (kept_call > bare_call) == (batch * target + refs > latent), "the larger call must win"
    # Cond and uncond in one call: the whole latent twice, and the kept references.
    assert seen["memory_required"] == base.memory_required(
        [2 * noise_shape[0], *noise_shape[1:]], cond_shapes={"ref_latents": [[1, 16, refs]]})
    assert seen["memory_required"] < stock["memory_required"]
    held = price.held_cells(base, noise_shape, conds)
    assert held.target == held.stock[0] == latent and sum(held.largest) == max(batch * target + refs, latent)


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
    assert f"the largest call a cond makes holds {target + refs} ({target} target, {refs} reference)" in line
    assert "GiB in place of" in line


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
    """``_calc_cond_batch`` prices a call with the same method during sampling."""
    patcher, noise_shape, positive, *_ = _render(comfy_flux, "flux2", "all")
    before = patcher.model.memory_required(list(noise_shape))
    price.install(patcher, "flux2")
    _prepare(monkeypatch, patcher, noise_shape, {"positive": positive})
    assert patcher.model.memory_required(list(noise_shape)) == before


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
