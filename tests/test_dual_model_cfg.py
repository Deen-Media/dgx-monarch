"""Ideogram4 dual-model cfg-parallel (dm-cfg2): split guider, selection, residency.

The split guider runs one model per cfg rank and exchanges the two predictions
with a single all_gather over the cfg group; per-rank residency keeps one
checkpoint per rank (cond on rank 0, uncond on rank 1). The guider math runs
against fake comfy.samplers and xfuser modules that carry comfy's shapes.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch.actor import dual_model_cfg
from dgx_monarch.adapters import cfg_parallel
from dgx_monarch.adapters.base import UnsupportedModelError


class _FakeCFGGuider:
    """comfy.samplers.CFGGuider's shape: patcher slot, cfg, single conds dict."""

    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        self.model_options: dict = {}
        self.conds: dict = {}
        self.original_conds: dict = {}
        self.cfg = 1.0
        self.inner_model = None

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        self.conds = dict(conds)
        self.original_conds = dict(conds)


def _install_fake_comfy(monkeypatch):
    """Install fake comfy.samplers + xfuser and reset the guider class cache."""
    records: dict = {"calc": [], "cfg_fn": [], "gathered_input": [], "contiguous": []}

    def calc_cond_batch(model, conds, x, timestep, model_options):
        records["calc"].append((model, len(conds)))
        # A batch-1 non-contiguous local prediction, so .contiguous() has real work.
        base = torch.arange(2, dtype=torch.float32).reshape(2, 1)
        return [base.t()]  # (1, 2) transpose view: non-contiguous, one forward

    def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep,
                     model_options={}, cond=None, uncond=None):  # noqa: B006  mirrors comfy's signature
        records["cfg_fn"].append((model, cond_scale))
        return uncond_pred + (cond_pred - uncond_pred) * cond_scale

    samplers = types.ModuleType("comfy.samplers")
    samplers.CFGGuider = _FakeCFGGuider
    samplers.calc_cond_batch = calc_cond_batch
    samplers.cfg_function = cfg_function
    comfy = types.ModuleType("comfy")
    comfy.samplers = samplers
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)

    cond_pred = torch.full((1, 2), 3.0)
    uncond_pred = torch.full((1, 2), 1.0)

    class _Group:
        def all_gather(self, tensor, dim=0):
            records["gathered_input"].append(tensor)
            records["contiguous"].append(bool(tensor.is_contiguous()))
            return torch.cat([cond_pred, uncond_pred], dim=dim)

    dist = types.ModuleType("xfuser.core.distributed")
    dist.get_cfg_group = lambda: _Group()
    core = types.ModuleType("xfuser.core")
    core.distributed = dist
    xfuser = types.ModuleType("xfuser")
    xfuser.core = core
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", dist)

    monkeypatch.setattr(dual_model_cfg, "_SPLIT_GUIDER_CLASS", None)
    records["cond_pred"] = cond_pred
    records["uncond_pred"] = uncond_pred
    return records


def _patcher(name):
    return types.SimpleNamespace(model=types.SimpleNamespace(name=name), name=name)


_SPEC = {"kind": "dual_model", "positive": [[torch.zeros(1, 2, 3), {}]],
         "negative": [[torch.ones(1, 2, 3), {}]], "cfg": 7.0}


def test_split_guider_math_ordering_and_single_forward(monkeypatch):
    records = _install_fake_comfy(monkeypatch)
    monkeypatch.setattr(dual_model_cfg, "cfg_world", lambda: 2)
    monkeypatch.setattr(dual_model_cfg, "cfg_rank", lambda: 0)

    cond, uncond = _patcher("cond"), _patcher("uncond")
    guider = dual_model_cfg.build_dual_model_guider(cond, _SPEC, uncond)
    assert guider.model_patcher is cond
    assert guider._local_key == "positive"

    guider.inner_model = types.SimpleNamespace(name="cond-inner")
    out = guider.predict_noise(torch.zeros(1, 2), torch.zeros(1))

    # calc_cond_batch runs exactly once (one model forward on this rank).
    assert len(records["calc"]) == 1 and records["calc"][0][1] == 1
    # The local slice reaches all_gather contiguous.
    assert records["contiguous"] == [True]
    # gathered[:n] is the cond prediction, gathered[n:2n] the uncond one, so the
    # fp32 combine is uncond + (cond - uncond) * cfg = 1 + (3 - 1) * 7 = 15.
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.full((1, 2), 15.0))
    # The combine ran through cfg_function against this rank's own inner model.
    assert records["cfg_fn"] == [(guider.inner_model, 7.0)]


def test_rank1_stage1_uses_uncond_patcher_and_negative_key(monkeypatch):
    _install_fake_comfy(monkeypatch)
    monkeypatch.setattr(dual_model_cfg, "cfg_world", lambda: 2)
    monkeypatch.setattr(dual_model_cfg, "cfg_rank", lambda: 1)
    cond, uncond = _patcher("cond"), _patcher("uncond")
    guider = dual_model_cfg.build_dual_model_guider(cond, _SPEC, uncond)
    assert guider.model_patcher is uncond  # both-resident: the uncond patcher
    assert guider._local_key == "negative"


def test_rank1_stage2_uses_the_resident_primary_when_uncond_patcher_none(monkeypatch):
    _install_fake_comfy(monkeypatch)
    monkeypatch.setattr(dual_model_cfg, "cfg_world", lambda: 2)
    monkeypatch.setattr(dual_model_cfg, "cfg_rank", lambda: 1)
    # Per-rank residency: rank 1 holds only the uncond checkpoint, passed as the
    # primary model_patcher with uncond_patcher None.
    resident_uncond = _patcher("uncond-primary")
    guider = dual_model_cfg.build_dual_model_guider(resident_uncond, _SPEC, None)
    assert guider.model_patcher is resident_uncond
    assert guider._local_key == "negative"


def test_cfg_close_to_one_refuses_at_build(monkeypatch):
    _install_fake_comfy(monkeypatch)
    monkeypatch.setattr(dual_model_cfg, "cfg_world", lambda: 2)
    monkeypatch.setattr(dual_model_cfg, "cfg_rank", lambda: 0)
    spec = dict(_SPEC, cfg=1.0)
    with pytest.raises(UnsupportedModelError, match="cfg above 1"):
        dual_model_cfg.build_dual_model_guider(_patcher("c"), spec, _patcher("u"))


def test_world1_delegates_to_both_resident(monkeypatch):
    from dgx_monarch.actor import dual_model_guider

    seen = {}

    class _Both:
        def set_conds(self, positive, negative):
            seen["conds"] = (positive, negative)

        def set_cfg(self, cfg):
            seen["cfg"] = cfg

    def fake_make(model_patcher, uncond_patcher):
        seen["patchers"] = (model_patcher, uncond_patcher)
        return _Both()

    monkeypatch.setattr(dual_model_guider, "make_both_resident_guider", fake_make)
    monkeypatch.setattr(dual_model_cfg, "cfg_world", lambda: 1)
    cond, uncond = _patcher("cond"), _patcher("uncond")
    guider = dual_model_cfg.build_dual_model_guider(cond, _SPEC, uncond)
    assert isinstance(guider, _Both)
    assert seen["patchers"] == (cond, uncond)
    assert seen["cfg"] == 7.0


def test_uninitialized_xfuser_probe_falls_to_both_resident(monkeypatch):
    from dgx_monarch.actor import dual_model_guider

    def raising_world():
        raise RuntimeError("xfuser not initialized")

    made = {}
    monkeypatch.setattr(dual_model_guider, "make_both_resident_guider",
                        lambda m, u: made.setdefault("g", types.SimpleNamespace(
                            set_conds=lambda *a: None, set_cfg=lambda *a: None)))
    monkeypatch.setattr(dual_model_cfg, "cfg_world", raising_world)
    dual_model_cfg.build_dual_model_guider(_patcher("c"), _SPEC, _patcher("u"))
    assert "g" in made


def test_world1_without_uncond_patcher_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(dual_model_cfg, "cfg_world", lambda: 1)
    with pytest.raises(RuntimeError, match="unconditional model resident"):
        dual_model_cfg.build_dual_model_guider(_patcher("c"), _SPEC, None)


def test_assert_cfg_parallel_grant_and_default_refusal():
    class _NoBatch:
        family = "ideogram4"
        cfg_parallel_supported = False

    adapter = _NoBatch()
    # The default (dual_model_cfg=False) refuses.
    with pytest.raises(UnsupportedModelError, match="separate calls"):
        cfg_parallel.assert_cfg_parallel_supported(adapter, 2)
    # The dm-cfg2 grant early-returns before the refusal.
    cfg_parallel.assert_cfg_parallel_supported(adapter, 2, dual_model_cfg=True)


def test_pixeldit_declares_the_dual_model_cfg_grant():
    from dgx_monarch.adapters import family_supports_dual_model_cfg
    from dgx_monarch.adapters.pixeldit import Ideogram4Adapter

    assert Ideogram4Adapter.dual_model_cfg_supported is True
    assert Ideogram4Adapter.cfg_parallel_supported is False  # the batched shape is refused
    assert family_supports_dual_model_cfg("ideogram4") is True
    assert family_supports_dual_model_cfg("krea2") is False


def _dm_worker():
    return types.SimpleNamespace(
        topology={"cfg": 2, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": False},
        world=2, store=None)


def _select(monkeypatch, worker, request, families, rank=0):
    from dgx_monarch.actor import sample_protocol

    monkeypatch.setattr(
        sample_protocol, "_sniff_request_family",
        lambda w, spec: families.get(spec.get("unet_name")) if spec else None)
    import dgx_monarch.adapters.base as base

    monkeypatch.setattr(base, "cfg_rank", lambda: rank)
    return sample_protocol.dual_model_cfg2_slot(worker, request)


def test_residency_selection_returns_owned_slot_per_rank(monkeypatch):
    worker = _dm_worker()
    request = {"model": {"unet_name": "cond.sft"}, "uncond_model": {"unet_name": "unc.sft"}}
    fam = {"cond.sft": "ideogram4", "unc.sft": "ideogram4"}
    assert _select(monkeypatch, worker, request, fam, rank=0) == "cond"
    assert _select(monkeypatch, worker, request, fam, rank=1) == "uncond"


def test_residency_selection_none_off_the_cfg2_shape(monkeypatch):
    worker = _dm_worker()
    worker.topology = {"cfg": 1, "ulysses": 2, "ring": 1, "dp": 1, "fsdp": False}
    request = {"model": {"unet_name": "cond.sft"}, "uncond_model": {"unet_name": "unc.sft"}}
    assert _select(monkeypatch, worker, request, {"cond.sft": "ideogram4"}) is None


def test_residency_selection_refuses_missing_uncond(monkeypatch):
    worker = _dm_worker()
    request = {"model": {"unet_name": "cond.sft"}, "uncond_model": None}
    with pytest.raises(UnsupportedModelError, match="needs the unconditional model"):
        _select(monkeypatch, worker, request, {"cond.sft": "ideogram4"})


def test_residency_selection_refuses_family_mismatch(monkeypatch):
    worker = _dm_worker()
    request = {"model": {"unet_name": "cond.sft"}, "uncond_model": {"unet_name": "unc.sft"}}
    fam = {"cond.sft": "ideogram4", "unc.sft": "krea2"}
    with pytest.raises(UnsupportedModelError, match="same family"):
        _select(monkeypatch, worker, request, fam)


def test_residency_selection_refuses_non_dm_family_with_uncond(monkeypatch):
    worker = _dm_worker()
    request = {"model": {"unet_name": "k.sft"}, "uncond_model": {"unet_name": "k2.sft"}}
    fam = {"k.sft": "krea2", "k2.sft": "krea2"}
    with pytest.raises(UnsupportedModelError, match="does not run dual-model"):
        _select(monkeypatch, worker, request, fam)


def test_residency_selection_none_for_batched_cfg_family(monkeypatch):
    # krea2 plain CFG under cfg2 (no uncond model) is the ordinary batched path.
    worker = _dm_worker()
    request = {"model": {"unet_name": "k.sft"}, "uncond_model": None}
    assert _select(monkeypatch, worker, request, {"k.sft": "krea2"}) is None


def _inject(monkeypatch, adapter, *, fake_pe=None):
    import dgx_monarch.adapters as adapters_pkg
    from dgx_monarch.actor import partial_load_guard, store_fsdp, worker
    from dgx_monarch.actor import worker as worker_mod

    calls: dict = {"assert": [], "wrappers": []}

    real_assert = cfg_parallel.assert_cfg_parallel_supported

    def record_assert(a, degree, dual_model_cfg=False):
        calls["assert"].append(dual_model_cfg)
        return real_assert(a, degree, dual_model_cfg=dual_model_cfg)

    monkeypatch.setattr(cfg_parallel, "assert_cfg_parallel_supported", record_assert)
    monkeypatch.setattr(store_fsdp, "validate_injection", lambda *a, **k: None)
    monkeypatch.setattr(partial_load_guard, "install", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "_maybe_compile_dit", lambda *a, **k: None)
    monkeypatch.setattr(adapters_pkg, "adapter_for", lambda *a, **k: adapter)
    if fake_pe is not None:
        monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_pe)
        comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
        comfy.patcher_extension = fake_pe
        monkeypatch.setitem(sys.modules, "comfy", comfy)

    fake_self = types.SimpleNamespace(
        topology={"cfg": 2, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": False},
        world=2, store=types.SimpleNamespace(family_override=None))
    base_patcher = types.SimpleNamespace(
        model=types.SimpleNamespace(diffusion_model=object()), model_options={})
    worker.GPUWorker._inject_for_topology(fake_self, base_patcher, "fp8", [], None)
    return calls, base_patcher


def test_worker_inject_grants_and_skips_wrapper_for_ig4(monkeypatch):
    from dgx_monarch.constants import CFG_WRAPPER_KEY

    adapter = types.SimpleNamespace(
        family="ideogram4", dual_model_cfg_supported=True,
        cfg_parallel_supported=False)
    calls, base_patcher = _inject(monkeypatch, adapter)
    # Granted with the dm flag, and the batched cond/uncond wrapper is not installed.
    assert calls["assert"] == [True]
    assert CFG_WRAPPER_KEY not in base_patcher.model_options


def test_worker_inject_installs_wrapper_for_batched_family(monkeypatch):
    from dgx_monarch.constants import CFG_DISPATCH_WRAPPER_KEY, CFG_WRAPPER_KEY

    installed: list = []

    class _WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"
        # Every cfg2 inject outside dm-cfg2 installs the per-cond dispatch on
        # this seam too, whatever split seam the family declares.
        CALC_COND_BATCH = "calc_cond_batch"

    fake_pe = types.ModuleType("comfy.patcher_extension")
    fake_pe.WrappersMP = _WrappersMP
    fake_pe.get_wrappers_with_key = lambda *a, **k: []

    def add_wrapper_with_key(kind, key, wrapper, options, is_model_options=False):
        installed.append(key)
        options[key] = wrapper

    fake_pe.add_wrapper_with_key = add_wrapper_with_key

    adapter = types.SimpleNamespace(
        family="krea2", cfg_parallel_supported=True,
        inject_cfg_pad_forward=lambda dm: None)
    monkeypatch.setattr(
        cfg_parallel, "make_cfg_parallel_wrapper", lambda a: "wrapper-sentinel")
    calls, base_patcher = _inject(monkeypatch, adapter, fake_pe=fake_pe)
    # A batched cfg2 family is checked without the grant and gets both wrappers.
    assert calls["assert"] == [False]
    assert CFG_WRAPPER_KEY in base_patcher.model_options
    assert CFG_DISPATCH_WRAPPER_KEY in base_patcher.model_options


def _run_sample_stage2(monkeypatch, slot, rank):
    """Drive run_sample under a stubbed dm-cfg2 slot, capturing the wiring."""
    import dgx_monarch.adapters as adapters_pkg
    from dgx_monarch.actor import sample_protocol, sampling, worker_env

    rec: dict = {"ensure": [], "identity": [], "clone": [], "custom": []}

    def fake_ensure(worker, unet_name, options, loras, slot="cond", **kw):
        rec["ensure"].append(slot)
        return (types.SimpleNamespace(model=types.SimpleNamespace(name=slot)), slot)

    def fake_clone(patcher, sampling_value):
        rec["clone"].append((patcher, sampling_value))
        return types.SimpleNamespace(model=patcher.model, cloned_from=patcher)

    def fake_run_custom(render_patcher, request, uncond_patcher=None, progress_port=None,
                        cancel_event=None, dp_cond_exempt_keys=frozenset()):
        rec["custom"].append({"render": render_patcher, "uncond": uncond_patcher})
        return (torch.zeros(1, 2), None)

    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda w, r: slot)
    monkeypatch.setattr(sample_protocol.store_fsdp, "ensure", fake_ensure)
    monkeypatch.setattr(worker_env, "verify_sample_artifact_authorization",
                        lambda w, r, fn: [{"id": "cond"}, {"id": "uncond"}])
    monkeypatch.setattr(worker_env, "activate_accuracy_waivers", lambda r: None)
    monkeypatch.setattr(worker_env, "assert_resident_artifact_identity",
                        lambda w, s, e: rec["identity"].append((s, e)))
    monkeypatch.setattr(worker_env, "sample_rescue_consent", lambda r: {})
    monkeypatch.setattr(worker_env, "accuracy_waiver_stamps", lambda: [])
    monkeypatch.setattr(adapters_pkg, "adapter_for",
                        lambda model, override=None: types.SimpleNamespace(cfg_cond_padding="none"))
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))

    worker = types.SimpleNamespace(
        _setup_key=object(), topology={"cfg": 2, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": False},
        world=2, rank=rank, store=types.SimpleNamespace(family_override=None),
        _attn=None, _inject_for_topology=object(), _check_uma_reserve=lambda: None)
    request = {
        "kind": "custom",
        "model": {"unet_name": "cond.sft", "model_sampling": {"kind": "sd3", "shift": 3.0}},
        "uncond_model": {"unet_name": "unc.sft", "model_sampling": {"kind": "sd3", "shift": 5.0}},
        "guider": {"kind": "dual_model", "positive": [[None, {}]], "negative": None, "cfg": 7.0},
        "latent": {"samples": torch.zeros(1, 4, 8, 8)},
    }
    sample_protocol.run_sample(
        worker, request, None, None,
        equalize_cond_lengths=lambda *a: (None, None),
        model_sampling_render_clone=fake_clone,
        request_artifact_identity=lambda name, loras: {"id": name},
        run_ksampler=lambda *a, **k: (torch.zeros(1, 2), None),
        run_custom=fake_run_custom,
        latent_signature=lambda s: {"sig": 1})
    return rec


def test_stage2_rank0_ensures_cond_only_and_runs_it_as_primary(monkeypatch):
    rec = _run_sample_stage2(monkeypatch, slot="cond", rank=0)
    assert rec["ensure"] == ["cond"]  # the uncond checkpoint never loads here
    assert rec["identity"] == [("cond", {"id": "cond"})]
    assert rec["custom"][0]["uncond"] is None  # no second model on this rank
    assert rec["custom"][0]["render"].cloned_from.model.name == "cond"
    assert rec["clone"][0][1] == {"kind": "sd3", "shift": 3.0}  # cond's model_sampling


def test_stage2_rank1_ensures_uncond_only_and_runs_it_as_primary(monkeypatch):
    rec = _run_sample_stage2(monkeypatch, slot="uncond", rank=1)
    assert rec["ensure"] == ["uncond"]  # the cond checkpoint never loads here
    assert rec["identity"] == [("uncond", {"id": "uncond"})]
    assert rec["custom"][0]["uncond"] is None
    assert rec["custom"][0]["render"].cloned_from.model.name == "uncond"
    assert rec["clone"][0][1] == {"kind": "sd3", "shift": 5.0}  # uncond's model_sampling


def _defer_result(monkeypatch, loader_cls, *, topo, family="ideogram4",
                  path_none=False, sniff_raises=False):
    """Run a loader's _defers_for_dual_model_cfg2 under stubbed sniff seams."""
    from dgx_monarch import family_select
    from dgx_monarch.adapters import detect

    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda folder, name: (  # type: ignore[attr-defined]
        None if path_none else f"/models/{name}")
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    def fake_sniff(path):
        if sniff_raises:
            raise detect.CheckpointSniffError("unreadable header")
        return (family, {})

    monkeypatch.setattr(detect, "sniff_checkpoint", fake_sniff)
    monkeypatch.setattr(family_select, "effective_family", lambda fam, override: fam)
    monkeypatch.setattr(family_select, "override_from_worker_args", lambda wa: None)
    mesh = types.SimpleNamespace(worker_args={})
    return loader_cls()._defers_for_dual_model_cfg2(mesh, "m.sft", topo)


def _dm2_topology():
    from dgx_monarch.topology import Topology

    return Topology(ulysses=1, ring=1, cfg=2, dp=1, world=2)


@pytest.mark.parametrize("loader_name", ("DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader"))
def test_defer_trigger_true_for_cfg2_world2_dm_family(monkeypatch, loader_name):
    from dgx_monarch.nodes import loaders

    loader_cls = getattr(loaders, loader_name)
    # The uncond loader inherits the method unchanged, so both slots defer and
    # each rank loads one checkpoint at sample time.
    assert _defer_result(monkeypatch, loader_cls, topo=_dm2_topology()) is True


@pytest.mark.parametrize("loader_name", ("DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader"))
def test_defer_trigger_false_off_the_dm2_shape(monkeypatch, loader_name):
    from dgx_monarch.nodes import loaders
    from dgx_monarch.topology import Topology

    loader_cls = getattr(loaders, loader_name)
    # Wrong topology (uly2, not cfg2): short-circuits before any sniff.
    wrong_topo = Topology(ulysses=2, ring=1, cfg=1, dp=1, world=2)
    assert _defer_result(monkeypatch, loader_cls, topo=wrong_topo) is False
    # Right topology, non-dm family: the eager both-resident load stands.
    assert _defer_result(
        monkeypatch, loader_cls, topo=_dm2_topology(), family="krea2") is False
    # Right topology, unreadable header: fall to the normal eager load.
    assert _defer_result(
        monkeypatch, loader_cls, topo=_dm2_topology(), sniff_raises=True) is False
    # Right topology, missing file: no defer, the load path surfaces it.
    assert _defer_result(
        monkeypatch, loader_cls, topo=_dm2_topology(), path_none=True) is False


def test_dm_cfg2_ready_or_raise_proceeds_when_all_ranks_ready(monkeypatch):
    from dgx_monarch.actor import dm_cfg2_residency as dm2r
    from dgx_monarch.actor import partial_load_guard as p

    monkeypatch.setattr(p, "all_ranks_true", lambda flag: True)
    assert dm2r.ready_or_raise() is None


def test_dm_cfg2_ready_or_raise_refuses_when_a_peer_failed(monkeypatch):
    from dgx_monarch.actor import dm_cfg2_residency as dm2r
    from dgx_monarch.actor import partial_load_guard as p

    monkeypatch.setattr(p, "all_ranks_true", lambda flag: False)
    with pytest.raises(dm2r.DualModelLoadDivergenceError, match="peer rank"):
        dm2r.ready_or_raise()


def test_dm_cfg2_not_ready_joins_the_exchange_then_reraises_local(monkeypatch):
    from dgx_monarch.actor import dm_cfg2_residency as dm2r
    from dgx_monarch.actor import partial_load_guard as p

    seen: list[bool] = []
    monkeypatch.setattr(p, "all_ranks_true", lambda flag: seen.append(flag) or False)
    with pytest.raises(RuntimeError, match="truncated on this box"):
        dm2r.not_ready(RuntimeError("header truncated on this box"))
    assert seen == [False]  # the failed rank still joined the exchange


def test_all_ranks_true_off_group_returns_local_without_reduce(monkeypatch):
    from dgx_monarch.actor import partial_load_guard as p

    called: list[bool] = []
    monkeypatch.setattr(p, "_distributed_world", lambda: 1)
    monkeypatch.setattr(p, "_all_ranks_full", lambda flag: called.append(flag) or True)
    assert p.all_ranks_true(True) is True
    assert p.all_ranks_true(False) is False
    assert called == []  # world 1: the local flag stands, no collective


def _base_run_sample_env(monkeypatch):
    import dgx_monarch.adapters as adapters_pkg
    from dgx_monarch.actor import sample_protocol, sampling, worker_env

    monkeypatch.setattr(worker_env, "verify_sample_artifact_authorization",
                        lambda w, r, fn: [{"id": "cond"}, {"id": "uncond"}])
    monkeypatch.setattr(worker_env, "activate_accuracy_waivers", lambda r: None)
    monkeypatch.setattr(worker_env, "assert_resident_artifact_identity",
                        lambda w, s, e: None)
    monkeypatch.setattr(worker_env, "sample_rescue_consent", lambda r: {})
    monkeypatch.setattr(worker_env, "accuracy_waiver_stamps", lambda: [])
    monkeypatch.setattr(adapters_pkg, "adapter_for",
                        lambda model, override=None: types.SimpleNamespace(cfg_cond_padding="none"))
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    return sample_protocol


def _dm2_worker(rank=0, topology=None):
    return types.SimpleNamespace(
        _setup_key=object(),
        topology=topology or {"cfg": 2, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": False},
        world=2, rank=rank, store=types.SimpleNamespace(family_override=None),
        _attn=None, _inject_for_topology=object(), _check_uma_reserve=lambda: None)


def _dm2_request():
    return {
        "kind": "custom",
        "model": {"unet_name": "cond.sft", "model_sampling": {"kind": "sd3", "shift": 3.0}},
        "uncond_model": {"unet_name": "unc.sft", "model_sampling": {"kind": "sd3", "shift": 5.0}},
        "guider": {"kind": "dual_model", "positive": [[None, {}]], "negative": None, "cfg": 7.0},
        "latent": {"samples": torch.zeros(1, 4, 8, 8)},
    }


def _run(sample_protocol, worker, request):
    return sample_protocol.run_sample(
        worker, request, None, None,
        equalize_cond_lengths=lambda *a: (None, None),
        model_sampling_render_clone=lambda pat, s: types.SimpleNamespace(
            model=pat.model, cloned_from=pat),
        request_artifact_identity=lambda name, loras: {"id": name},
        run_ksampler=lambda *a, **k: (torch.zeros(1, 2), None),
        run_custom=lambda *a, **k: (torch.zeros(1, 2), None),
        latent_signature=lambda s: {"sig": 1})


def test_run_sample_healthy_rank_refuses_when_a_peer_fails_its_load(monkeypatch):
    sample_protocol = _base_run_sample_env(monkeypatch)
    from dgx_monarch.actor import dm_cfg2_residency as dm2r
    from dgx_monarch.actor import partial_load_guard as p

    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda w, r: "cond")
    monkeypatch.setattr(sample_protocol.store_fsdp, "ensure",
                        lambda *a, **k: (types.SimpleNamespace(model=object()), "hot"))
    monkeypatch.setattr(p, "all_ranks_true", lambda flag: False)  # a peer was short
    with pytest.raises(dm2r.DualModelLoadDivergenceError, match="peer rank"):
        _run(sample_protocol, _dm2_worker(rank=0), _dm2_request())


def test_run_sample_failed_rank_joins_exchange_then_reraises(monkeypatch):
    sample_protocol = _base_run_sample_env(monkeypatch)
    from dgx_monarch.actor import partial_load_guard as p

    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda w, r: "uncond")

    def boom(*a, **k):
        raise RuntimeError("uncond header truncated on this box")

    monkeypatch.setattr(sample_protocol.store_fsdp, "ensure", boom)
    joined: list[bool] = []
    monkeypatch.setattr(p, "all_ranks_true", lambda flag: joined.append(flag) or False)
    with pytest.raises(RuntimeError, match="uncond header truncated"):
        _run(sample_protocol, _dm2_worker(rank=1), _dm2_request())
    assert joined == [False]


def test_run_sample_slot_refusal_routes_through_the_exchange(monkeypatch):
    # A per-rank sniff divergence raises in the slot decision; it must still
    # join the exchange so a peer is released, and its message survives.
    sample_protocol = _base_run_sample_env(monkeypatch)
    from dgx_monarch.actor import partial_load_guard as p

    def refuse(w, r):
        raise UnsupportedModelError("needs the unconditional model")

    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", refuse)
    joined: list[bool] = []
    monkeypatch.setattr(p, "all_ranks_true", lambda flag: joined.append(flag) or False)
    with pytest.raises(UnsupportedModelError, match="unconditional model"):
        _run(sample_protocol, _dm2_worker(rank=0), _dm2_request())
    assert joined == [False]


def test_run_sample_non_dm2_load_failure_skips_the_exchange(monkeypatch):
    sample_protocol = _base_run_sample_env(monkeypatch)
    from dgx_monarch.actor import dm_cfg2_residency as dm2r

    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda w, r: None)

    def boom(*a, **k):
        raise RuntimeError("cond load failed")

    monkeypatch.setattr(sample_protocol.store_fsdp, "ensure", boom)
    calls: list[str] = []
    monkeypatch.setattr(dm2r, "ready_or_raise", lambda: calls.append("ready"))
    monkeypatch.setattr(dm2r, "not_ready", lambda e: calls.append("not_ready"))
    topo = {"cfg": 1, "ulysses": 2, "ring": 1, "dp": 1, "fsdp": False}
    with pytest.raises(RuntimeError, match="cond load failed"):
        _run(sample_protocol, _dm2_worker(rank=0, topology=topo), _dm2_request())
    assert calls == []  # uly2 takes rank_readiness's exchange, never dm-cfg2's


def test_run_sample_cancellation_bypasses_the_exchange(monkeypatch):
    sample_protocol = _base_run_sample_env(monkeypatch)
    from dgx_monarch.actor import dm_cfg2_residency as dm2r
    from dgx_monarch.actor.sampling import RenderCancelledError

    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda w, r: "cond")

    def cancel(*a, **k):
        raise RenderCancelledError("cancelled mid-load")

    monkeypatch.setattr(sample_protocol.store_fsdp, "ensure", cancel)
    calls: list[str] = []
    monkeypatch.setattr(dm2r, "ready_or_raise", lambda: calls.append("ready"))
    monkeypatch.setattr(dm2r, "not_ready", lambda e: calls.append("not_ready"))
    with pytest.raises(RenderCancelledError, match="cancelled mid-load"):
        _run(sample_protocol, _dm2_worker(rank=0), _dm2_request())
    assert calls == []  # a cancel never routes through the ready flag
