"""Every rank reads the same shared NVFP4 scale switch, and the ranks prove they agree.

Each Linear the shared scale wraps issues one max all-reduce per call
(adapters/quant_activation_scale.py), so a rank that wraps where its peer does
not waits in a collective the peer never issues. Before 1.5.0 a Worker kept its
own ``DGXM_SHARED_ACT_SCALE`` when the worker args left ``shared_act_scale``
out, and a variable exported on one box stalled the render.

Now a Worker writes its variable from the worker args it was sent, an absent
key reading on, and every rank is sent the same args. The payload stays the
published policy: a residency grant names the published policy and the Worker
compares it with the one it recorded (mesh_residency.py), so a payload that
differed from it would refuse every vouched render. Behind that, the post-load
readiness exchange crosses each rank's count of wrapped Linears and refuses,
class P, on every rank when they differ.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch.actor import partial_load_guard, rank_readiness
from dgx_monarch.adapters import quant_activation_scale as qas
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from test_worker_drop_report import AUTO, STOCK, worker  # noqa: F401  (the Worker fixture)


def test_setup_sends_every_rank_the_published_policy(monkeypatch):
    """The setup dispatch's own payload, one per rank, with the key left absent."""
    from dgx_monarch import mesh_safety, mesh_setup, native_return_latch

    sent: list = []

    class _Worker:
        class setup:  # the actor endpoint's name
            @staticmethod
            def call_one(env):
                sent.append(env)

    class _Workers:
        extent = types.SimpleNamespace(labels=["hosts", "gpus"])

        @staticmethod
        def slice(**_coords):
            return _Worker()

    monkeypatch.setattr(native_return_latch, "native_enabled", lambda _flag: False)
    monkeypatch.setattr(mesh_safety, "worker_comfy_dir", lambda *_args: "/comfy")
    handle = types.SimpleNamespace(
        config=types.SimpleNamespace(rdma_latent_return=False, rdma_min_bytes=0),
        workers=_Workers(), n_hosts=2, gpus_per_host=1, world=2, setup_generation=1,
        comfy_dir="/comfy")
    topology = types.SimpleNamespace(ulysses=2, ring=1, cfg=1, dp=1, fsdp=False)
    mesh_setup.dispatch_setup(handle, topology, "TORCH_FLASH", True, {"slab_weights": False},
                              {}, "127.0.0.1", 29500, False)
    assert [env["worker_args"] for env in sent] == [{"slab_weights": False}] * 2


@pytest.fixture
def published_and_ranks(worker):  # noqa: F811  (the fixture imported below)
    """The real driver apply into two real Worker applies: the policy the driver
    published, and the two Workers as that apply left them."""
    from dgx_monarch.actor import worker_env
    from topology_transition_helpers import _Future, _handle, _ValueMesh, _Workers

    def run(new: dict):
        ranks = [worker(AUTO, STOCK), worker(AUTO, STOCK)]

        class _RealWorkers:
            def call(self, effective):
                return _Future(_ValueMesh([worker_env.apply_worker_args_impl(rig, dict(effective))
                                           for rig in ranks]))

        workers = _Workers([])
        workers.apply_worker_args = _RealWorkers()
        handle, _old, _new = _handle(workers)
        handle.active_worker_args = {}
        handle.apply_worker_args(new, merge_config=False)
        return dict(handle.active_worker_args), ranks

    return run


SWITCHES = [pytest.param({}, id="switch left absent"),
            pytest.param({qas.ENABLE_ARG: True}, id="switch on"),
            pytest.param({qas.ENABLE_ARG: False}, id="switch off")]


@pytest.mark.parametrize("new", SWITCHES)
def test_a_gate_grant_bound_to_the_published_policy_holds_on_every_worker(
        published_and_ranks, new):
    """A residency grant's context names the driver's published policy, and each
    Worker compares it with the policy it recorded (mesh_residency.py)."""
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, gate_verdict_token
    from dgx_monarch.mesh_residency import (
        NORMAL_RENDER_RESIDENCY_CAPABILITY,
        assert_normal_render_residency_grant,
        request_combo_key,
    )

    published, ranks = published_and_ranks(new)
    model = {"unet_name": "m.safetensors", "options": {"weight_dtype": "default"}, "loras": []}
    identity = {"digest": "primary", "comfy": "commit", "artifacts": []}
    context = {"worker_args": published, "world": 2}
    combo = request_combo_key(model)
    request = {"model": model, "_dgxm_normal_residency_mode": "required",
               "_dgxm_normal_residency_grant": {
                   "capability": NORMAL_RENDER_RESIDENCY_CAPABILITY,
                   "gate_protocol": GATE_PROTOCOL_VERSION, "dgx_monarch": __version__,
                   "capability_context": context, "combo_key": combo,
                   "model_request": dict(model), "uncond_model_request": None,
                   "artifact_sets": [identity], "comfy": "commit", "artifact_digest": "primary",
                   "gate_token": list(gate_verdict_token(combo, "primary", "commit", context))}}
    for rig in ranks:
        assert assert_normal_render_residency_grant(
            request, [identity], effective_worker_args=dict(rig._active_worker_args),
            rank_world=2)


@pytest.mark.parametrize("new", SWITCHES)
def test_a_fleet_grant_bound_to_the_published_policy_holds_on_every_worker(
        published_and_ranks, new):
    from dgx_monarch.mesh_residency import (
        FLEET_RESIDENCY_CAPABILITY,
        assert_fleet_residency_grant,
    )
    from test_model_sampling_binding import _fleet_gate_grant

    published, ranks = published_and_ranks(new)
    model = {"unet_name": "m.safetensors", "options": {"weight_dtype": "default"}, "loras": []}
    artifacts = [{"digest": "primary", "comfy": "commit", "artifacts": []}]
    context = {"capability": FLEET_RESIDENCY_CAPABILITY, "rank_world": 1, "worker_args": published}
    request = {"model": model, "_dgxm_fleet_job": True,
               "_dgxm_fleet_residency_grant": _fleet_gate_grant(model, artifacts, context)}
    for rig in ranks:
        assert assert_fleet_residency_grant(
            request, artifacts, effective_worker_args=dict(rig._active_worker_args), rank_world=1)


@pytest.mark.parametrize("switch", [pytest.param({}, id="switch left absent"),
                                    pytest.param({qas.ENABLE_ARG: True}, id="switch on")])
def test_a_capacity_consent_bound_to_the_published_policy_holds_on_every_worker(
        published_and_ranks, switch):
    from dgx_monarch import __version__
    from dgx_monarch.mesh_capacity_consent import (
        CAPACITY_CONSENT_CONTRACT,
        RESIDENCY_MODE_CAPACITY_CONSENT,
    )
    from dgx_monarch.mesh_residency import (
        CAPACITY_RESCUE_CONSENT_CAPABILITY,
        assert_normal_render_residency_grant,
        request_combo_key,
    )

    # A consent projects slab residency into the render's policy.
    published, ranks = published_and_ranks({"slab_weights": True, "lora_low_rss": True, **switch})
    model = {"unet_name": "m.safetensors", "options": {"weight_dtype": "default"}, "loras": []}
    identity = {"digest": "primary", "comfy": "commit", "artifacts": []}
    request = {"model": model, "_dgxm_normal_residency_mode": RESIDENCY_MODE_CAPACITY_CONSENT,
               "_dgxm_normal_residency_grant": {
                   "capability": CAPACITY_RESCUE_CONSENT_CAPABILITY, "dgx_monarch": __version__,
                   "consent_contract": CAPACITY_CONSENT_CONTRACT,
                   "consent": {"kind": "rescue-slab", "id": "c" * 32},
                   "capability_context": {"worker_args": published, "world": 2},
                   "combo_key": request_combo_key(model), "model_request": dict(model),
                   "artifact_sets": [identity], "comfy": "commit", "artifact_digest": "primary"}}
    for rig in ranks:
        assert assert_normal_render_residency_grant(
            request, [identity], effective_worker_args=dict(rig._active_worker_args),
            rank_world=2)


_SWITCH = {"absent": {}, "on": {qas.ENABLE_ARG: True}, "off": {qas.ENABLE_ARG: False}}


@pytest.mark.parametrize("before", list(_SWITCH))
@pytest.mark.parametrize("after", list(_SWITCH))
def test_the_drivers_drop_inference_reads_the_switch_as_the_worker_applies_it(
        worker, before, after):  # noqa: F811
    """The driver marks a drop (mesh_setup_transition.drops_residents) exactly
    where the real Worker apply drops its resident for the switch alone."""
    from dgx_monarch.mesh_setup_transition import drops_residents
    from test_worker_drop_report import _apply

    rig = worker(_SWITCH[before], STOCK)
    dropped = _apply(rig, dict(_SWITCH[after])) == ["cond"]
    assert dropped is (before != after and "off" in (before, after))
    assert drops_residents(dict(_SWITCH[before]), dict(_SWITCH[after])) is dropped


def test_a_worker_never_reads_its_own_variable_for_the_switch(monkeypatch):
    monkeypatch.setenv(qas.ENABLE_ENV, "0")      # exported by hand on this box
    qas.env_from_worker_args({})                 # the published policy left it absent
    assert qas.enabled_now() is True
    monkeypatch.setenv(qas.ENABLE_ENV, "1")
    qas.env_from_worker_args({qas.ENABLE_ARG: False})
    assert qas.enabled_now() is False


def test_the_readiness_exchange_refuses_ranks_that_wrap_differently(monkeypatch):
    """Every rank reads the same MIN of (count, -count), so all refuse together."""
    monkeypatch.setattr(partial_load_guard, "all_ranks_true", lambda flag: flag)
    peers = {"wrapped": 0}
    monkeypatch.setattr(partial_load_guard, "all_ranks_min",
                        lambda values: [min(values[0], peers["wrapped"]),
                                        min(values[1], -peers["wrapped"])])
    peers["wrapped"] = 198
    rank_readiness.note_shared_scale(198)
    assert rank_readiness.ready_or_raise() is None
    rank_readiness.note_shared_scale(0)
    with pytest.raises(rank_readiness.SharedScaleDisagreementError) as caught:
        rank_readiness.ready_or_raise()
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "0 on one rank, 198 on another" in str(caught.value)


def test_a_rank_that_failed_its_load_crosses_no_scale_exchange(monkeypatch):
    """The second exchange runs only where every rank read ready, so a failed
    rank and its peers issue the same collectives."""
    monkeypatch.setattr(partial_load_guard, "all_ranks_true", lambda flag: False)
    crossed: list = []
    monkeypatch.setattr(partial_load_guard, "all_ranks_min",
                        lambda values: crossed.append(values) or values)
    rank_readiness.note_shared_scale(7)
    with pytest.raises(rank_readiness.PeerLoadFailureError):
        rank_readiness.ready_or_raise()
    with pytest.raises(ValueError):
        rank_readiness.not_ready(ValueError("this box"))
    assert crossed == []


def test_the_sample_path_notes_its_count_before_the_exchange():
    """Read from the source: the count is noted right before the readiness call."""
    import inspect

    from dgx_monarch.actor import sample_protocol

    source = inspect.getsource(sample_protocol)
    noted = source.index("rank_readiness.note_shared_scale(")
    assert noted < source.index("readiness.ready_or_raise()")
    assert "quant_activation_scale.wrapped_linears(patcher, uncond_patcher)" in source


def test_wrapped_linears_reads_each_models_coverage():
    def patcher(wrapped):
        model = types.SimpleNamespace()
        setattr(model, qas.COVERAGE_ATTR, types.SimpleNamespace(wrapped=wrapped))
        return types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=model))

    assert qas.wrapped_linears(patcher(198), None, patcher(2)) == 200
    assert qas.wrapped_linears(types.SimpleNamespace(model=None)) == 0
