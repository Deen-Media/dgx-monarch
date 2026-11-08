"""The prompt-pair tag that rows 47 (Anima) and 79 (Boogu) need for cfg2 (issue #52).

On 2026-11-07 both rows read bit-identical to one GPU on the templates' pairs,
whose text rows differ, and off one GPU on two pairs of equal rows, which one
GPU folds into one batch-2 call. The sampler nodes tag a pair whose rows differ
on the latent; every resolving site reads the tag, and anything the driver
cannot read stays untagged, which keeps the Ulysses row.

No comfy, no GPU, no mesh: the render and resolution seams are stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.cfg_pair import text_rows_differ
from dgx_monarch.nodes import common, guiders, render_preflight, render_result, samplers

PAIR = render_preflight.CFG_PAIR_DIFFERS_METADATA_KEY


def _cond(rows: int, **options) -> list:
    return [[torch.zeros(1, rows, 8), dict(options)]]


def test_only_one_plain_entry_a_side_with_different_rows_reads_as_differing() -> None:
    # The Boogu template: an empty negative takes the drop template's 66 rows
    # against 56 to 60 for a prompt; the Anima template differs too.
    assert text_rows_differ(_cond(57), _cond(66))
    assert text_rows_differ(_cond(102, t5xxl_ids=torch.zeros(36)), _cond(31))
    # Equal rows may fold on one GPU: the eqsame and eqdiff pairs of check 10,
    # and a Conditioning Zero Out negative with the positive's shape.
    assert not text_rows_differ(_cond(31), _cond(31))
    assert not text_rows_differ(_cond(66, attention_mask=torch.ones(1, 66)), _cond(66))


@pytest.mark.parametrize("key", ["area", "mask", "timestep_start", "timestep_end",
                                 "gligen", "default"])
def test_a_key_that_changes_comfys_grouping_leaves_the_pair_unread(key) -> None:
    assert not text_rows_differ(_cond(57, **{key: (1, 1, 0, 0)}), _cond(66))
    assert not text_rows_differ(_cond(57), _cond(66, **{key: None}))


def test_hooks_and_control_count_by_value() -> None:
    assert text_rows_differ(_cond(57, hooks=None, control=None), _cond(66))
    assert not text_rows_differ(_cond(57, hooks=object()), _cond(66))
    assert not text_rows_differ(_cond(57), _cond(66, control=object()))


@pytest.mark.parametrize("side", [
    None, [], [[torch.zeros(1, 66, 8), {}], [torch.zeros(1, 66, 8), {}]],
    [[None, {}]], [[torch.zeros(66, 8), {}]], [[torch.zeros(1, 66, 8)]],
    [[torch.zeros(1, 66, 8), "options"]], "text",
])
def test_anything_else_reads_as_not_differing(side) -> None:
    # A combined conditioning, the dual-model guider's empty negative and
    # malformed lists answer false: the answer that costs speed, never fidelity.
    assert not text_rows_differ(_cond(57), side)
    assert not text_rows_differ(side, _cond(57))


def _capture_run_render(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    def run_render(model, request, latent, cfg_value, steps_hint):
        seen.append(latent)
        return {"samples": torch.zeros(1, 4, 8, 8)}

    monkeypatch.setattr(samplers, "run_render", run_render)
    monkeypatch.setattr(samplers, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(samplers, "krea2_ref_preflight_summary", lambda *_args: None)
    return seen


@pytest.mark.parametrize(("negative_rows", "tagged"), [(66, True), (57, False)])
def test_the_ksampler_and_advanced_sampler_tag_the_pair(monkeypatch, negative_rows, tagged):
    seen = _capture_run_render(monkeypatch)
    source = {"samples": torch.zeros(1, 4, 8, 8), PAIR: True}
    samplers.DGXMonarchKSampler().sample(
        "model", 0, 8, 4.0, "euler", "simple", _cond(57), _cond(negative_rows), source)
    samplers.DGXMonarchKSamplerAdvanced().sample(
        "model", "enable", 0, 8, 4.0, "euler", "simple", _cond(57), _cond(negative_rows),
        {"samples": torch.zeros(1, 4, 8, 8)}, 0, 10000, "disable")
    assert [latent.get(PAIR, False) for latent in seen] == [tagged, tagged]
    assert source[PAIR] is True  # the node's input is never written


def test_the_pipeline_hands_every_render_the_pair_tag(monkeypatch):
    seen = _capture_run_render(monkeypatch)
    pushed: list[dict] = []

    class Pipeline:
        def __init__(self, depth, on_step=None):
            self.depth = depth

        def push(self, model, request, latent, cfg_value, steps_hint):
            pushed.append(latent)

        def drain(self):
            return [{"samples": torch.zeros(1, 4, 8, 8)} for _latent in pushed]

    monkeypatch.setattr(samplers, "auto_gate_required", lambda *_args: len(seen) == 0)
    monkeypatch.setattr(samplers, "RenderPipeline", Pipeline)
    model = SimpleNamespace(mesh=SimpleNamespace(pipeline_depth=2))
    samplers.DGXMonarchKSamplerPipeline().sample(
        model, "1, 2, 3", 8, 4.0, "euler", "simple", _cond(57), _cond(66),
        {"samples": torch.zeros(1, 4, 8, 8)})
    assert seen and pushed
    assert all(latent.get(PAIR) is True for latent in (*seen, *pushed))


@pytest.mark.parametrize(("negative", "tagged"), [(_cond(66), True), (_cond(57), False)])
def test_the_cfg_guiders_carry_the_answer_to_sampler_custom(monkeypatch, negative, tagged):
    monkeypatch.setattr(guiders, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(guiders, "krea2_ref_preflight_summary", lambda *_args: None)
    (cfg_guider,) = guiders.DGXMonarchCFGGuider().get_guider("model", _cond(57), negative, 4.0)
    (plain,) = guiders.DGXMonarchDualModelGuider().get_guider(
        "model", _cond(57), 4.0, negative=negative)
    assert cfg_guider[PAIR] is tagged and plain[PAIR] is tagged
    (no_negative,) = guiders.DGXMonarchDualModelGuider().get_guider("model", _cond(57), 4.0)
    assert no_negative[PAIR] is False

    seen = _capture_run_render(monkeypatch)
    monkeypatch.setattr(samplers, "_canonical_noise_for_wire", lambda _noise: None)
    for guider in (cfg_guider, {"model": "model", "spec": {}, "cfg": 4.0}):
        samplers.DGXMonarchSamplerCustom().sample(
            SimpleNamespace(seed=0), guider, object(), torch.tensor([1.0, 0.0]),
            {"samples": torch.zeros(1, 4, 8, 8), PAIR: True})
    assert [latent.get(PAIR, False) for latent in seen] == [tagged, False]


def test_the_gate_node_proves_the_render_with_its_pair_tag(monkeypatch):
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes.gate_node import DGXMonarchIdentityGate

    proved: list[dict] = []

    def ceremony(model, request, latent, *_args, **_kwargs):
        proved.append(latent)
        return {"latent": latent, "verdict": "PASS"}

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    monkeypatch.setattr(gate_mod, "conditioning_for_wire", lambda value: value)
    model = SimpleNamespace(mesh=SimpleNamespace(attention="TORCH_FLASH"))
    DGXMonarchIdentityGate().gate(model, _cond(57), _cond(66),
                                  {"samples": torch.zeros(1, 4, 8, 8)},
                                  0, 8, 4.0, "euler", "simple")
    (latent,) = proved
    assert latent.get(PAIR) is True


def test_the_pair_tag_never_reaches_the_wire_or_the_output(monkeypatch):
    latent = {"samples": torch.zeros(1, 4, 8, 8), PAIR: True, "noise_mask": "kept"}
    assert PAIR not in render_preflight.latent_without_topology_metadata(latent)
    monkeypatch.setattr(render_result, "verify_cross_rank_signatures", lambda *_args: None)
    monkeypatch.setattr(render_result, "read_latent_result", lambda value: value)
    out = render_result._finish_render(
        [{"rank": 0, "dp_rank": 0, "latent": torch.ones(1, 4, 8, 8)}],
        SimpleNamespace(dp=1), latent)
    assert PAIR not in out and out["noise_mask"] == "kept"


def test_the_resolver_forwards_the_pair_tag_only_when_set(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(common, "resolve_topology", lambda *_args, **kwargs: calls.append(kwargs))
    samples = object()
    for latent in ({"samples": samples, PAIR: True}, {"samples": samples},
                   {"samples": samples, PAIR: False}):
        common._resolve_topology_for_latent("model", latent, 4.0, 2)
    assert calls == [{"pair_differs": True}, {}, {}]


def test_run_matrix_labels_its_auto_cell_with_the_pair_tag_the_ksampler_node_sets():
    """benchmark/run_matrix.py records the auto topology before the render; it
    must read the pair as the KSampler node does, or an Anima, Boogu or Chroma
    auto cell would record uly2 while the render resolves cfg2."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "benchmark" / "run_matrix.py"
    calls = [node for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "resolve_topology"]
    assert calls
    for call in calls:
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        assert "keeps_uncond" in keywords and "pair_differs" in keywords
        assert "text_rows_differ(positive, negative)" in keywords["pair_differs"]
