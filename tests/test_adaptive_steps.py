"""An adaptive-step sampler under data parallelism with FSDP refuses on the driver.

Each data-parallel group samples its own slice, and ``dpm_adaptive`` sizes its
steps from that slice, so the groups can make different numbers of model
calls; FSDP gathers every call's weights from every rank, so a group with
more calls would wait for ranks that have finished until the sample deadline.
``nodes/adaptive_steps.py`` refuses the combination, class P: at an explicit
preset's UNET loader, before its setup and load, where the graph names the
sampler, at the sampler and Identity Gate nodes before the first-use check,
and at the render submit before setup. The real
ComfyUI readings (which samplers' call counts follow the latent, and the
SAMPLER objects ComfyUI builds) are the comfy seam canary's
"adaptive-step samplers"; here a stand-in ``comfy`` builds its SAMPLER the
way ``comfy.samplers.ksampler`` does.
"""
from __future__ import annotations

import functools
import inspect
import sys
import types

import pytest
import torch

from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import adaptive_steps, common, h3_ring_pad, render_preflight
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from dgx_monarch.topology import Topology, topology_from_preset


@pytest.fixture
def stand_in_comfy(monkeypatch):
    """comfy.samplers.ksampler as ComfyUI writes it: a new closure around the
    k-diffusion solver for dpm_adaptive, the k-diffusion function otherwise."""
    sampling = types.ModuleType("comfy.k_diffusion.sampling")

    def sample_dpm_adaptive(model, x, sigma_min, sigma_max, **options):
        return x

    def sample_euler(model, x, sigmas, **options):
        return x

    sampling.sample_dpm_adaptive, sampling.sample_euler = sample_dpm_adaptive, sample_euler
    samplers = types.ModuleType("comfy.samplers")

    class KSAMPLER:
        def __init__(self, sampler_function, extra_options=None, inpaint_options=None):
            self.sampler_function = sampler_function
            self.extra_options = extra_options or {}

    def ksampler(name, extra_options=None, inpaint_options=None):
        if name == "dpm_adaptive":
            def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable,
                                      **options):
                return sampling.sample_dpm_adaptive(model, noise, sigmas[-1], sigmas[0])
            function = dpm_adaptive_function
        else:
            function = getattr(sampling, f"sample_{name}")
        return KSAMPLER(function, extra_options, inpaint_options)

    samplers.KSAMPLER, samplers.ksampler = KSAMPLER, ksampler
    comfy = types.ModuleType("comfy")
    k_diffusion = types.ModuleType("comfy.k_diffusion")
    comfy.samplers, comfy.k_diffusion, k_diffusion.sampling = samplers, k_diffusion, sampling
    for name, module in (("comfy", comfy), ("comfy.samplers", samplers),
                         ("comfy.k_diffusion", k_diffusion),
                         ("comfy.k_diffusion.sampling", sampling)):
        monkeypatch.setitem(sys.modules, name, module)
    return samplers


def _refusal(exc_info):
    tag = parse_refusal_tag(str(exc_info.value))
    assert tag is not None
    return tag


# --- which samplers ----------------------------------------------------------


def test_the_named_sampler_is_read_and_no_other(stand_in_comfy):
    assert adaptive_steps.adaptive_sampler({"sampler_name": "dpm_adaptive"}) == "sampler dpm_adaptive"
    for name in ("euler", "dpm_fast", "dpmpp_2m", "uni_pc", None, 3):
        assert adaptive_steps.adaptive_sampler({"sampler_name": name}) is None


def test_a_sampler_custom_object_is_read_by_its_solver(stand_in_comfy):
    """SamplerDPMAdaptative's object: a fresh ksampler closure with options,
    matched by its code; a bare KSAMPLER of the solver, and one under a partial."""
    built = stand_in_comfy.ksampler("dpm_adaptive", {"rtol": 0.1})
    bare = stand_in_comfy.KSAMPLER(sys.modules["comfy.k_diffusion.sampling"].sample_dpm_adaptive)
    wrapped = stand_in_comfy.KSAMPLER(functools.partial(built.sampler_function, order=2))
    for sampler in (built, bare, wrapped):
        assert adaptive_steps.adaptive_sampler({"sampler_object": sampler}), sampler
    for sampler in (stand_in_comfy.ksampler("euler"), types.SimpleNamespace(), object()):
        assert adaptive_steps.adaptive_sampler({"sampler_object": sampler}) is None


def test_without_comfy_an_object_reads_as_no_adaptive_sampler(monkeypatch):
    monkeypatch.setattr(adaptive_steps, "_solver_functions", lambda: [])
    sampler = types.SimpleNamespace(sampler_function=lambda *args: None)
    assert adaptive_steps.adaptive_sampler({"sampler_object": sampler}) is None


# --- which layouts -----------------------------------------------------------


@pytest.mark.parametrize(("preset", "world"), [
    ("dp2+fsdp", 2), ("dp2+fsdp", 4), ("uly2+fsdp", 4), ("cfg2+fsdp", 4), ("uly2+fsdp", 8),
])
def test_dp_with_fsdp_refuses_class_p_and_names_the_remedies(stand_in_comfy, preset, world):
    topo = topology_from_preset(preset, world)
    with pytest.raises(UnsupportedModelError) as exc_info:
        adaptive_steps.refuse({"sampler_name": "dpm_adaptive"}, topo)
    tag = _refusal(exc_info)
    assert tag.refusal_class is RefusalClass.PHYSICS and tag.guard is None
    text = str(exc_info.value)
    assert f"under {topo.describe()} on {world} ranks" in text
    assert f"{topo.dp} data-parallel groups" in text
    assert "any but dpm_adaptive" in text and "without +fsdp" in text
    assert "docs/TROUBLESHOOTING.md #156" in text


def test_the_pair_is_offered_the_presets_without_data_parallelism():
    with pytest.raises(UnsupportedModelError) as exc_info:
        adaptive_steps.refuse({"sampler_name": "dpm_adaptive"}, topology_from_preset("dp2+fsdp", 2))
    # ring2+fsdp is no preset at all: ring cannot combine with FSDP.
    assert "(uly2, ring2, cfg2, uly2+fsdp, cfg2+fsdp)" in str(exc_info.value)


def test_an_odd_world_with_no_such_preset_leaves_that_remedy_out():
    with pytest.raises(UnsupportedModelError) as exc_info:
        adaptive_steps.refuse({"sampler_name": "dpm_adaptive"}, topology_from_preset("dp2+fsdp", 3))
    assert "without data parallelism" not in str(exc_info.value)


@pytest.mark.parametrize(("preset", "world"), [
    ("dp2", 2),          # pure data parallelism: no collective in a call
    ("uly2", 4),         # uly2+dp2: each group shares no collective with the other
    ("cfg2", 4),
    ("uly2+fsdp", 2),    # FSDP with one data-parallel group
    ("cfg2+fsdp", 2),
    ("single", 1),
])
def test_every_other_layout_keeps_rendering(preset, world):
    adaptive_steps.refuse({"sampler_name": "dpm_adaptive"}, topology_from_preset(preset, world))


def test_a_fixed_step_sampler_renders_under_dp_with_fsdp(stand_in_comfy):
    topo = topology_from_preset("dp2+fsdp", 2)
    adaptive_steps.refuse({"sampler_name": "euler"}, topo)
    adaptive_steps.refuse({"sampler_object": stand_in_comfy.ksampler("euler")}, topo)


def test_auto_under_capacity_fsdp_at_four_refuses():
    """auto at world 4 for an even batch, where the model does not fit resident."""
    with pytest.raises(UnsupportedModelError):
        adaptive_steps.refuse({"sampler_name": "dpm_adaptive"},
                              Topology(ulysses=2, dp=2, fsdp=True, world=4))


# --- where it runs -----------------------------------------------------------


def _model(world, preset):
    mesh = common.MeshSpec(handle=types.SimpleNamespace(world=world), topology_preset=preset,
                           attention="TORCH_FLASH", sync_ulysses=False, worker_args={})
    return common.ModelSpec(mesh=mesh, unet_name="model.safetensors")


def _latent():
    return {"samples": torch.zeros(2, 4, 8, 8)}


def test_a_fixed_step_sampler_never_resolves_a_topology():
    def resolve(*_args, **_kwargs):
        pytest.fail("a fixed-step sampler resolved the topology")

    adaptive_steps.preflight_for_render(_model(2, "dp2+fsdp"), {"sampler_name": "euler"},
                                        _latent(), 1.0, resolve_topology=resolve)


def test_preflight_for_render_reads_the_mesh_world():
    seen = []

    def resolve(model, latent, cfg_value, world):
        seen.append(world)
        return topology_from_preset("dp2+fsdp", world), False, ""

    with pytest.raises(UnsupportedModelError):
        adaptive_steps.preflight_for_render(_model(4, "dp2+fsdp"), {"sampler_name": "dpm_adaptive"},
                                            _latent(), 1.0, resolve_topology=resolve)
    assert seen == [4]


def test_a_resolution_that_raises_is_left_to_the_render():
    def resolve(*_args, **_kwargs):
        raise FileNotFoundError("the render's own resolution raises this")

    adaptive_steps.preflight_for_render(_model(2, "auto"), {"sampler_name": "dpm_adaptive"},
                                        _latent(), 1.0, resolve_topology=resolve)


def _forbid(monkeypatch, module, *names):
    def forbidden(name):
        def call(*_args, **_kwargs):
            pytest.fail(f"{name} ran before the adaptive-step refusal")
        return call

    for name in names:
        monkeypatch.setattr(module, name, forbidden(name))


def _run_render_until_the_gate(monkeypatch):
    monkeypatch.setattr(render_preflight, "activation_footprint_preflight_for_request",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(common, "_bind_packed_render_model",
                        lambda model, _latent, _cfg, **_kw: (model, None))
    monkeypatch.setattr(h3_ring_pad, "_family_and_path", lambda _model: None)


def test_run_render_refuses_before_the_gate_setup_and_the_load(monkeypatch):
    """The explicit dp2+fsdp preset on the pair: the refusal lands at the
    gate's entry, before it reads its context or runs a proof render, and
    before submit_render, which owns setup and the dispatch that loads."""
    _run_render_until_the_gate(monkeypatch)
    _forbid(monkeypatch, common, "_auto_gate_context", "_quarantine_unproven_paths",
            "submit_render")
    with pytest.raises(UnsupportedModelError) as exc_info:
        common.run_render(_model(2, "dp2+fsdp"),
                          {"kind": "ksampler", "sampler_name": "dpm_adaptive", "cfg": 4.0},
                          _latent(), 4.0, 8)
    assert _refusal(exc_info).refusal_class is RefusalClass.PHYSICS


@pytest.mark.parametrize(("preset", "world", "sampler"), [
    ("uly2", 4, "dpm_adaptive"),     # the product layout the agreement fix lets render
    ("dp2+fsdp", 2, "euler"),
])
def test_run_render_lets_the_others_through(monkeypatch, preset, world, sampler):
    _run_render_until_the_gate(monkeypatch)
    reached = []

    class SubmitReached(Exception):
        pass

    def gate_context(*_args, **_kwargs):
        reached.append("gate")
        return None  # no ceremony to run

    def submit(*_args, **_kwargs):
        reached.append("submit")
        raise SubmitReached

    monkeypatch.setattr(common, "_auto_gate_context", gate_context)
    monkeypatch.setattr(common, "_quarantine_unproven_paths", lambda *_args, **_kw: None)
    monkeypatch.setattr(common, "submit_render", submit)
    with pytest.raises(SubmitReached):
        common.run_render(_model(world, preset),
                          {"kind": "ksampler", "sampler_name": sampler, "cfg": 4.0},
                          _latent(), 4.0, 8)
    assert reached == ["gate", "submit"]


def test_every_entry_checks_before_its_proof_render_and_setup():
    """The gate's entry (every run_render passes it), the pipeline before its
    gate, the Identity Gate node before its ceremony, which renders the
    node's own sampler, and the render submit before setup."""
    from dgx_monarch.nodes import auto_gate, gate_node, pipeline, render_submit

    entry = inspect.getsource(auto_gate.maybe_auto_gate)
    assert (entry.index("adaptive_steps.preflight_for_render(")
            < entry.index("= _auto_gate_context("))
    pushed = inspect.getsource(pipeline.RenderPipeline._push_bound)
    assert pushed.index("adaptive_steps.preflight_for_render(") < pushed.index("_maybe_auto_gate(")
    gate = inspect.getsource(gate_node.DGXMonarchIdentityGate.gate)
    assert (gate.index("adaptive_steps.preflight_for_render(")
            < gate.index("run_identity_ceremony("))
    submit = inspect.getsource(render_submit._submit_render_guarded)
    assert submit.index("adaptive_steps.refuse(request, topo)") < submit.index("ensure_request_setup(")


# --- at the loader of an explicit preset, which sets up and loads at once -----


def _mesh(preset, world):
    return types.SimpleNamespace(topology_preset=preset, handle=types.SimpleNamespace(world=world))


def _graph(**sampler_inputs):
    """A loader (1), and the KSampler (3) that takes its model."""
    return {
        "1": {"class_type": "DGXMonarchUNETLoader", "inputs": {"unet_name": "model.safetensors"}},
        "3": {"class_type": "DGXMonarchKSampler",
              "inputs": {"model": ["1", 0], "sampler_name": "dpm_adaptive", **sampler_inputs}},
    }


def _custom_graph(sampler_node):
    """Sampler Custom (3), its guider (4) on the loader's model, and a SAMPLER node (5)."""
    return {
        "1": {"class_type": "DGXMonarchUNETLoader", "inputs": {"unet_name": "model.safetensors"}},
        "4": {"class_type": "DGXMonarchCFGGuider", "inputs": {"model": ["1", 0]}},
        "5": sampler_node,
        "3": {"class_type": "DGXMonarchSamplerCustom",
              "inputs": {"guider": ["4", 0], "sampler": ["5", 0]}},
    }


@pytest.mark.parametrize(("graph", "label"), [
    (_graph(), "sampler dpm_adaptive"),
    (_custom_graph({"class_type": "KSamplerSelect", "inputs": {"sampler_name": "dpm_adaptive"}}),
     "sampler dpm_adaptive"),
    (_custom_graph({"class_type": "SamplerDPMAdaptative", "inputs": {"order": 3}}),
     "Sampler Custom's SAMPLER"),
], ids=["ksampler", "ksampler-select", "sampler-dpm-adaptative"])
def test_the_loader_refuses_before_its_setup_and_load(graph, label):
    with pytest.raises(UnsupportedModelError) as exc_info:
        adaptive_steps.preflight_loader_graph(_mesh("dp2+fsdp", 2), graph, "1")
    assert _refusal(exc_info).refusal_class is RefusalClass.PHYSICS
    assert str(exc_info.value).startswith(f"[dgxm:P] {label}")


def _switched_off():
    """The dpm_adaptive sampler (3) feeds only the branch a literal switch leaves out."""
    graph = _graph()
    graph["5"] = {"class_type": "DGXMonarchKSampler",
                  "inputs": {"model": ["1", 0], "sampler_name": "euler"}}
    graph["9"] = {"class_type": "ComfySwitchNode",
                  "inputs": {"switch": True, "on_true": ["5", 0], "on_false": ["3", 0]}}
    return graph


@pytest.mark.parametrize(("mesh", "graph", "loader"), [
    (_mesh("auto", 2), _graph(), "1"),                       # auto loads at the first render
    (_mesh("uly2+fsdp", 2), _graph(), "1"),                  # one data-parallel group
    (_mesh("uly2", 4), _graph(), "1"),                       # no FSDP
    (_mesh("dp2+fsdp", 2), _graph(sampler_name="euler"), "1"),
    (_mesh("dp2+fsdp", 2), _graph(sampler_name=["7", 0]), "1"),  # a linked name: the sampler node reads it
    (_mesh("dp2+fsdp", 2), _graph(), "2"),                   # another loader's sampler
    (_mesh("dp2+fsdp", 2), _switched_off(), "1"),            # a branch ComfyUI will not run
    (_mesh("dp2+fsdp", 2), _graph(), None),
    (_mesh("dp2+fsdp", 2), None, "1"),
], ids=["auto", "uly2+fsdp", "uly2-dp2", "euler", "linked-name", "other-loader", "switched-off",
        "no-loader-id", "no-graph"])
def test_the_loader_makes_no_claim_it_cannot_read(mesh, graph, loader):
    adaptive_steps.preflight_loader_graph(mesh, graph, loader)


def test_the_fleet_sampler_never_counts():
    graph = _graph()
    graph["3"]["class_type"] = "DGXMonarchFleetKSampler"
    adaptive_steps.preflight_loader_graph(_mesh("dp2+fsdp", 2), graph, "1")


def test_the_loader_checks_before_ensure_live_setup_and_load():
    from dgx_monarch.nodes import loaders

    source = inspect.getsource(loaders.DGXMonarchUNETLoader.load)
    check = source.index("adaptive_steps.preflight_loader_graph(mesh, prompt, unique_id)")
    assert check < source.index("handle = ensure_live(")
    assert check < source.index("ensure_request_setup(")
    assert check < source.index("handle.call_all(")
