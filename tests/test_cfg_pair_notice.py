"""cfg2 on a prompt pair of equal text rows says so, once a render, and refuses nothing.

One GPU folds an equal pair into one call of two entries and cfg2 makes two
calls of one; those equal stock's own separate calls and read apart from its
batched call (docs/VALIDATION.md, the equal-pair records). A hand-chosen cfg2
tells the user in the browser and the log; a cfg2 that `auto` kept (Krea2 FP8,
or an MXFP8 or NVFP4 file, whose only other route is approximate) logs it at
INFO. tests/test_cfg_pair_kind.py holds the kind the notice reads.
"""
from __future__ import annotations

import inspect
import re
import sys
import types

import pytest
import torch

from dgx_monarch import cfg_pair, first_render
from dgx_monarch.nodes import cfg_pair_notice
from dgx_monarch.topology_cfg_evidence import BLOCK_SCALED_FP8, CAST_FP8


def _pair(pos_rows, neg_rows):
    return ([[torch.zeros(1, pos_rows, 4), {}]], [[torch.zeros(1, neg_rows, 4), {}]])


def test_an_equal_pair_is_read_only_from_two_plain_entries():
    assert cfg_pair.text_rows_equal(*_pair(104, 104))
    assert not cfg_pair.text_rows_equal(*_pair(104, 5))
    positive, negative = _pair(7, 7)
    assert not cfg_pair.text_rows_equal(positive + positive, negative)   # unread: combined
    assert not cfg_pair.text_rows_equal([[torch.zeros(1, 7, 4), {"area": (1, 1, 0, 0)}]],
                                        negative)


def _model(preset):
    return types.SimpleNamespace(mesh=types.SimpleNamespace(topology_preset=preset),
                                 unet_name="model.safetensors")


def _topology(cfg, ulysses=1):
    return types.SimpleNamespace(cfg=cfg, sequence_parallel=ulysses)


@pytest.fixture
def recorded(monkeypatch):
    """Browser notices, by first_render.notice, which writes the log line too."""
    import logging

    seen: list = []

    def notice(*args, **kwargs):
        seen.append((args, kwargs))
        logging.getLogger(first_render.__name__).info("%s", args[1])
        return True

    monkeypatch.setattr(first_render, "notice", notice)
    monkeypatch.setattr(cfg_pair_notice, "_family", lambda _model: ("krea2", "fp8"))
    return seen


def test_a_hand_chosen_cfg2_on_an_equal_pair_tells_the_user(recorded):
    assert cfg_pair_notice.announce(_model("cfg2"), _topology(2), True)
    (phase, message, _note), kwargs = recorded[0]
    assert phase == "cfg2-equal-pair" and kwargs["severity"] == "warn"
    assert "equal stock ComfyUI's own separate calls" in message
    assert "0.0104 to 0.040" in message
    assert "uly2 on the Torch Flash kernel renders this file bytes equal" in message


def test_a_cfg2_auto_kept_logs_at_info_only(recorded, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        assert cfg_pair_notice.announce(_model("auto"), _topology(2), True)
    assert recorded == []
    assert any("cfg2 is running a prompt and a negative of equal text rows" in record.message
               for record in caplog.records)


@pytest.mark.parametrize(("equal", "topology"), [(False, _topology(2)), (True, _topology(1, 2)),
                                                 (True, _topology(2, 2))])
def test_nothing_is_said_off_cfg2_or_for_a_pair_that_differs(recorded, equal, topology):
    assert not cfg_pair_notice.announce(_model("cfg2"), topology, equal)
    assert recorded == []


def test_a_file_with_no_bytes_equal_layout_says_so():
    message = cfg_pair_notice.text("chroma", BLOCK_SCALED_FP8)
    assert "NRMS 0.034" in message
    assert "No layout is recorded bytes equal" in message


@pytest.mark.parametrize("family", ["lens", "qwen_image", "longcat"])
def test_an_unmeasured_family_may_differ(family):
    """Row 68's own family and LongCat (row 28, which keeps equal pairs on cfg2)
    have no equal-pair reading: the notice says the call may differ."""
    message = cfg_pair_notice.text(family, "bf16")
    assert "may differ from its batched call (not measured for this family)" in message
    assert " and differ from" not in message


def test_every_bytes_equal_claim_cites_a_dated_reading_against_one_gpu():
    """An entry needs a dated uly2 reading of bytes equal against one GPU: a
    one-GPU render, or row 0 of a same-build dp2 render, which is the one-GPU
    reference (each dp2 rank renders its latent alone). Each entry names which."""
    table = cfg_pair_notice._BYTES_EQUAL_ULYSSES
    assert set(table) == {("krea2", "bf16"), ("krea2", "fp8"), ("krea2", "int8"),
                          ("qwen_image", "fp8"), ("qwen_image21", "bf16"),
                          ("chroma", "bf16"), ("chroma", "fp8"), ("chroma", "int8"),
                          ("boogu", "fp8")}
    assert cfg_pair_notice.DP2_ROW_0 == "row 0 of a same-build dp2 render, the one-GPU reference"
    for date, reference in table.values():
        assert re.fullmatch(r"20\d\d-\d\d-\d\d( and 20\d\d-\d\d-\d\d)?", date)
        assert reference in (cfg_pair_notice.ONE_GPU, cfg_pair_notice.DP2_ROW_0)
    assert {key for key, (_date, reference) in table.items()
            if reference == cfg_pair_notice.DP2_ROW_0} == {
        ("boogu", "fp8"), ("chroma", "fp8"), ("chroma", "int8")}
    assert "(read 2026-11-24 against one GPU)" in cfg_pair_notice.text("qwen_image21", "bf16")
    assert "(read 2026-10-02 against one GPU)" in cfg_pair_notice.text("chroma", "bf16")
    assert ("(read 2026-10-06 against row 0 of a same-build dp2 render, the one-GPU "
            "reference)") in cfg_pair_notice.text("boogu", "fp8")
    for family in ("boogu", "chroma", "krea2", "qwen_image"):
        for kind in (BLOCK_SCALED_FP8, CAST_FP8):
            assert "No layout is recorded" in cfg_pair_notice.text(family, kind)


def test_a_sweep_shows_one_browser_notice_and_logs_every_render(monkeypatch, caplog):
    """One queued prompt, three renders: one browser notice, three log lines; the
    next prompt gets its own notice."""
    import logging

    sent: list = []
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
        send_sync=lambda event, payload: sent.append(payload)))
    monkeypatch.setitem(sys.modules, "server", server)
    monkeypatch.setattr(cfg_pair_notice, "_family", lambda _model: ("krea2", "bf16"))
    monkeypatch.setattr(cfg_pair_notice, "_shown_for_prompt", None)
    prompt = {"id": "prompt-1"}
    monkeypatch.setattr(cfg_pair_notice, "_prompt_id", lambda: prompt["id"])
    with caplog.at_level(logging.INFO):
        for _seed in range(3):
            assert cfg_pair_notice.announce(_model("cfg2"), _topology(2), True)
    lines = [r for r in caplog.records if "equal text rows" in r.getMessage()]
    assert len(sent) == 1 and len(lines) == 3
    prompt["id"] = "prompt-2"
    assert cfg_pair_notice.announce(_model("cfg2"), _topology(2), True)
    assert len(sent) == 2


def test_the_notice_never_fails_a_render(monkeypatch):
    def boom(_model):
        raise RuntimeError("no header")

    monkeypatch.setattr(cfg_pair_notice, "_family", boom)
    assert cfg_pair_notice.announce(_model("cfg2"), _topology(2), True) is False


def test_the_samplers_read_the_pair_and_the_submission_keeps_it_off_the_wire():
    from dgx_monarch.nodes import render_submit, samplers

    sampler_source = inspect.getsource(samplers)
    assert sampler_source.count(
        "cfg_pair_notice.REQUEST_KEY: text_rows_equal(positive, negative)") == 3
    # Sampler Custom carries what its guider read.
    assert ("cfg_pair_notice.REQUEST_KEY: bool(guider.get(cfg_pair_notice.REQUEST_KEY))"
            in inspect.getsource(samplers.DGXMonarchSamplerCustom.sample))
    guarded = inspect.getsource(render_submit._submit_render_guarded)
    assert (guarded.index("request.pop(cfg_pair_notice.REQUEST_KEY, False)")
            < guarded.index("model = model_for_request(model, request)"))
    assert "cfg_pair_notice.announce(model, topo, equal_pair)" in guarded


def test_a_healed_retry_still_reads_the_pair(monkeypatch):
    """The busy heal runs a second attempt on the same request; each attempt's
    body pops the reading, so the submission holds it between them."""
    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import render_submit

    seen: list = []

    def guarded(_model, request, *_args, **_kwargs):
        seen.append(request.pop(cfg_pair_notice.REQUEST_KEY, None))
        if len(seen) == 1:
            raise mesh_setup.LifecycleBusyError("busy")
        return "pending"

    monkeypatch.setattr(render_submit, "_submit_render_guarded", guarded)
    monkeypatch.setattr(render_submit, "run_guarded_submit", lambda _guard, body, _fail: body())
    monkeypatch.setattr(render_submit, "_heal_abandoned_wedge_once", lambda *_args: True)
    request = {"kind": "ksampler", cfg_pair_notice.REQUEST_KEY: True}
    assert render_submit.submit_render(_model("cfg2"), request, {}, 4.0, 1,
                                       _claimed_adoption_context=None) == "pending"
    assert seen == [True, True]
    assert cfg_pair_notice.REQUEST_KEY not in request


@pytest.fixture
def custom_render(monkeypatch):
    """Sampler Custom through the real submission's take-off of the reading:
    the guarded half pops it and announces, as ``_submit_render_guarded`` does."""
    from dgx_monarch.nodes import render_submit, samplers

    monkeypatch.setattr(samplers.requeue_price, "price_cached_loader", lambda *a, **k: None)
    monkeypatch.setattr(samplers.expander_prompt, "refuse_dead_expander", lambda *a: None)
    monkeypatch.setattr(render_submit, "run_guarded_submit", lambda _guard, body, _fail: body())
    state = {"topology": _topology(2), "requests": []}

    def guarded(model, request, *_args, **_kwargs):
        equal_pair = bool(request.pop(cfg_pair_notice.REQUEST_KEY, False))
        state["requests"].append(dict(request))
        cfg_pair_notice.announce(model, state["topology"], equal_pair)
        return "pending"

    def run_render(model, request, latent, cfg_value=None, steps_hint=1):
        render_submit.submit_render(model, request, latent, cfg_value, steps_hint,
                                    _claimed_adoption_context=None)
        return {"samples": torch.zeros(1, 4, 8, 8), "_dgxm_denoised": torch.ones(1, 4, 8, 8)}

    monkeypatch.setattr(render_submit, "_submit_render_guarded", guarded)
    monkeypatch.setattr(samplers, "run_render", run_render)

    def sample(guider):
        samplers.DGXMonarchSamplerCustom().sample(
            types.SimpleNamespace(seed=1), guider, object(), torch.tensor([1.0, 0.0]),
            {"samples": torch.zeros(1, 4, 8, 8)})

    state["sample"] = sample
    return state


@pytest.mark.parametrize(("preset", "neg_rows", "browser", "logged"), [
    ("cfg2", 7, 1, 1),      # a hand-chosen cfg2 on an equal pair
    ("auto", 7, 0, 1),      # auto kept Krea2 FP8 on cfg2: the INFO line only
    ("cfg2", 5, 0, 0),      # a pair of different rows
])
def test_cfg_guider_and_sampler_custom_tell_an_equal_pair(recorded, custom_render, caplog,
                                                          preset, neg_rows, browser, logged):
    import logging

    from dgx_monarch.nodes import guiders

    positive, negative = _pair(7, neg_rows)
    (guider,) = guiders.DGXMonarchCFGGuider().get_guider(_model(preset), positive, negative, 4.0)
    with caplog.at_level(logging.INFO):
        custom_render["sample"](guider)
    lines = [r for r in caplog.records if "equal text rows" in r.getMessage()]
    assert len(recorded) == browser and len(lines) == logged
    assert cfg_pair_notice.REQUEST_KEY not in custom_render["requests"][0]


def test_the_dual_guider_tells_only_its_one_model_form(recorded, custom_render):
    """With no second model it builds the plain cfg spec, whose equal pair one
    GPU folds; with two models one GPU makes two calls too, so it says nothing."""
    from dgx_monarch.nodes import guiders

    positive, negative = _pair(7, 7)
    model = _model("cfg2")
    (one,) = guiders.DGXMonarchDualModelGuider().get_guider(model, positive, 4.0,
                                                            negative=negative)
    custom_render["sample"](one)
    assert len(recorded) == 1
    uncond = types.SimpleNamespace(slot="uncond", request_dict=lambda: {"unet_name": "u"})
    (two,) = guiders.DGXMonarchDualModelGuider().get_guider(model, positive, 4.0,
                                                            model_negative=uncond,
                                                            negative=negative)
    custom_render["sample"](two)
    assert len(recorded) == 1


@pytest.mark.parametrize(("neg_rows", "equal"), [(7, True), (5, False)])
def test_the_cfg_guiders_read_the_pair_for_sampler_custom(neg_rows, equal):
    from dgx_monarch.nodes import guiders

    positive, negative = _pair(7, neg_rows)
    model = types.SimpleNamespace(slot="cond")
    (basic,) = guiders.DGXMonarchCFGGuider().get_guider(model, positive, negative, 4.0)
    assert basic[cfg_pair_notice.REQUEST_KEY] is equal
    (dual,) = guiders.DGXMonarchDualModelGuider().get_guider(model, positive, 4.0,
                                                             negative=negative)
    assert dual[cfg_pair_notice.REQUEST_KEY] is equal
    (unwired,) = guiders.DGXMonarchDualModelGuider().get_guider(model, positive, 4.0)
    assert unwired[cfg_pair_notice.REQUEST_KEY] is False
