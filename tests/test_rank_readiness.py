"""Post-load readiness exchange for multi-rank, non-FSDP topologies.

Memory can change between fleet agreement and allocation. If one rank's load
fails, a successful peer would otherwise wait alone in partial_load_guard.check
until the process-group timeout.

CPU-local tests cover the predicate, refusals and run_sample call site. Two-rank
Gloo tests compare the exchange with a negative control: the healthy rank must
refuse before the 5 s group timeout when the exchange runs, and wait for that
timeout when it is disabled.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch.actor import (
    dm_cfg2_residency,
    partial_load_guard,
    rank_readiness,
    sample_protocol,
)
from dgx_monarch.refusal import parse_refusal_tag

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _worker(*, world=2, ulysses=1, ring=1, cfg=1, dp=1, fsdp=False, rank=0):
    return types.SimpleNamespace(
        _setup_key=object(),
        topology={"ulysses": ulysses, "ring": ring, "cfg": cfg, "dp": dp, "fsdp": fsdp},
        world=world, rank=rank, store=types.SimpleNamespace(family_override=None),
        _attn=None, _inject_for_topology=object(), _check_uma_reserve=lambda: None)


def test_every_multi_rank_topology_that_holds_the_whole_model_owes_an_exchange():
    assert rank_readiness.whole_model_topology(_worker(ulysses=2)) is True
    assert rank_readiness.whole_model_topology(_worker(ring=2)) is True
    assert rank_readiness.whole_model_topology(_worker(dp=2)) is True
    assert rank_readiness.whole_model_topology(_worker(cfg=2)) is True


def test_a_single_rank_and_an_fsdp_fleet_owe_no_exchange():
    # One rank cannot disagree with itself and has no peer to release.
    assert rank_readiness.whole_model_topology(_worker(world=1)) is False
    # The rank_readiness module docstring says why an FSDP fleet owes none.
    assert rank_readiness.whole_model_topology(_worker(ulysses=2, fsdp=True)) is False
    assert rank_readiness.whole_model_topology(_worker(dp=2, fsdp=True)) is False


def test_the_predicate_never_raises_on_a_worker_that_answers_nothing():
    # A gate that can raise on one rank and not the other would strand the
    # fleet at the exchange it guards.
    assert rank_readiness.whole_model_topology(types.SimpleNamespace()) is False
    assert rank_readiness.whole_model_topology(
        types.SimpleNamespace(topology=None, world=None)) is False


def test_ready_or_raise_proceeds_when_every_rank_loaded(monkeypatch):
    monkeypatch.setattr(partial_load_guard, "all_ranks_true", lambda flag: True)
    assert rank_readiness.ready_or_raise() is None


def test_ready_or_raise_refuses_class_p_and_names_the_single_box(monkeypatch):
    monkeypatch.setattr(partial_load_guard, "all_ranks_true", lambda flag: False)
    with pytest.raises(rank_readiness.PeerLoadFailureError) as caught:
        rank_readiness.ready_or_raise()
    message = str(caught.value)
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "P"
    # Class P must point somewhere reachable, and must not name a card: MIN
    # carries agreement, not identity, so this rank cannot know the cause.
    assert "on a single box instead" in message
    assert tag.guard is None
    assert "Nothing was sampled" in message


def test_not_ready_joins_the_exchange_before_it_reraises(monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(
        partial_load_guard, "all_ranks_true", lambda flag: seen.append(flag) or False)
    with pytest.raises(RuntimeError, match="could not place the weights"):
        rank_readiness.not_ready(RuntimeError("could not place the weights"))
    assert seen == [False]  # the failed rank sent its flag before re-raising


class _Boom(Exception):
    """What a broken exchange raises: not a refusal, and not the local cause."""


class _Recorder:
    def __init__(self):
        self.warnings = []

    def info(self, *a):
        pass

    def warning(self, message, *args):
        self.warnings.append(message % args)


@pytest.mark.parametrize(
    "module,cause",
    [
        (rank_readiness, "could not place the weights"),
        # dm-cfg2 shares the rule, so a drift between the two shapes fails one
        # of these two cases.
        (dm_cfg2_residency, "header truncated on this box"),
    ],
)
def test_a_broken_exchange_never_replaces_the_local_cause(monkeypatch, module, cause):
    """An exchange that fails on the box that ran short must not replace its
    typed local cause with a CUDA or NCCL error (the comment in
    ``rank_readiness.not_ready`` says why)."""
    def broken(flag):
        raise _Boom("no memory left for the exchange buffer")

    recorder = _Recorder()
    monkeypatch.setattr(partial_load_guard, "all_ranks_true", broken)
    monkeypatch.setattr(module, "log", recorder)
    with pytest.raises(RuntimeError, match=cause):
        module.not_ready(RuntimeError(cause))
    # The journal says the peer is on its own, so a group timeout on the other
    # box reads as a consequence of this fault, not a second one.
    assert any("readiness flag not crossed" in line and "group timeout" in line
               for line in recorder.warnings), recorder.warnings


def test_both_shapes_issue_one_and_the_same_exchange(monkeypatch):
    """dm-cfg2 and the generic shape must not drift into two collectives.

    A second exchange primitive would be a second collective, and a fleet whose
    ranks issued different ones would deadlock where this releases it.
    """
    seen: list[bool] = []
    monkeypatch.setattr(
        partial_load_guard, "all_ranks_true", lambda flag: seen.append(flag) or True)
    rank_readiness.ready_or_raise()
    dm_cfg2_residency.ready_or_raise()
    assert seen == [True, True]


def test_the_dm_cfg2_message_stays_its_own(monkeypatch):
    """The two texts answer different questions and must not be merged.

    dm-cfg2's names the two-checkpoint split and offers uly2; the generic one
    cannot offer uly2, because uly2 is one of the topologies it fires on.
    """
    monkeypatch.setattr(partial_load_guard, "all_ranks_true", lambda flag: False)
    with pytest.raises(dm_cfg2_residency.DualModelLoadDivergenceError) as dm2:
        dm_cfg2_residency.ready_or_raise()
    with pytest.raises(rank_readiness.PeerLoadFailureError) as generic:
        rank_readiness.ready_or_raise()
    assert "dual-model cfg2" in str(dm2.value)
    assert "topology uly2" in str(dm2.value)
    assert "topology uly2" not in str(generic.value)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"cfg": 2}, dm_cfg2_residency),          # dm-cfg2 keeps its own message
        ({"ulysses": 2}, rank_readiness),
        ({"ring": 2}, rank_readiness),
        ({"dp": 2}, rank_readiness),
        ({"cfg": 2, "fsdp": True}, None),         # cfg-parallel under FSDP: shards
        ({"ulysses": 2, "fsdp": True}, None),
        ({"world": 1}, None),
    ],
)
def test_the_selector_picks_one_shape_per_topology(kwargs, expected):
    assert sample_protocol._post_load_readiness(_worker(**kwargs)) is expected


def test_the_selector_never_raises():
    assert sample_protocol._post_load_readiness(types.SimpleNamespace()) is None


def _run_env(monkeypatch):
    import dgx_monarch.adapters as adapters_pkg
    from dgx_monarch import adoption_evidence
    from dgx_monarch.actor import sampling, worker_env

    monkeypatch.setattr(worker_env, "verify_sample_artifact_authorization",
                        lambda w, r, fn: [{"id": "cond"}, {"id": "uncond"}])
    monkeypatch.setattr(worker_env, "activate_accuracy_waivers", lambda r: None)
    monkeypatch.setattr(worker_env, "assert_resident_artifact_identity",
                        lambda w, s, e: None)
    monkeypatch.setattr(worker_env, "sample_rescue_consent", lambda r: {})
    monkeypatch.setattr(worker_env, "accuracy_waiver_stamps", lambda: [])
    monkeypatch.setattr(adapters_pkg, "adapter_for",
                        lambda model, override=None: types.SimpleNamespace(
                            cfg_cond_padding="none", dp_cond_exempt_keys=frozenset()))
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(adoption_evidence, "build_worker_evidence",
                        lambda *a, **k: None)
    return sample_protocol


def _request():
    return {
        "kind": "ksampler",
        "model": {"unet_name": "flux2-dev.safetensors",
                  "model_sampling": {"kind": "sd3", "shift": 3.0}},
        "positive": None,
        "negative": None,
        "latent": {"samples": torch.zeros(2, 4, 8, 8)},
    }


def _run(protocol, worker, request, run_ksampler=None):
    return protocol.run_sample(
        worker, request, None, None,
        equalize_cond_lengths=lambda *a: (None, None),
        model_sampling_render_clone=lambda pat, s: types.SimpleNamespace(
            model=pat.model, cloned_from=pat),
        request_artifact_identity=lambda name, loras: {"id": name},
        run_ksampler=run_ksampler or (lambda *a, **k: (torch.zeros(1, 2), None)),
        run_custom=lambda *a, **k: (torch.zeros(1, 2), None),
        latent_signature=lambda s: {"sig": 1})


def _ensure_ok(*a, **k):
    # The sample path reads model.diffusion_model on a sharded topology.
    return (types.SimpleNamespace(model=types.SimpleNamespace(
        diffusion_model=types.SimpleNamespace())), "hot")


def _ensure_boom(*a, **k):
    raise RuntimeError("this box could not place the weights")


@pytest.mark.parametrize("kwargs", [{"ulysses": 2}, {"dp": 2}])
def test_the_healthy_rank_refuses_instead_of_entering_the_collective(monkeypatch, kwargs):
    """Without the exchange, a healthy uly2 or dp2 rank samples on alone.

    It reaches partial_load_guard.check with a peer that already refused and
    waits there until the process group times out.
    """
    protocol = _run_env(monkeypatch)
    monkeypatch.setattr(protocol.store_fsdp, "ensure", _ensure_ok)
    monkeypatch.setattr(partial_load_guard, "all_ranks_true", lambda flag: False)

    def never(*a, **k):
        raise AssertionError("the sampler must not run when a peer failed")

    with pytest.raises(rank_readiness.PeerLoadFailureError, match="a peer rank"):
        _run(protocol, _worker(**kwargs), _request(), run_ksampler=never)


@pytest.mark.parametrize("kwargs", [{"ulysses": 2}, {"dp": 2}])
def test_the_failed_rank_joins_the_exchange_then_reraises_its_own_cause(
        monkeypatch, kwargs):
    protocol = _run_env(monkeypatch)
    monkeypatch.setattr(protocol.store_fsdp, "ensure", _ensure_boom)
    joined: list[bool] = []
    monkeypatch.setattr(
        partial_load_guard, "all_ranks_true", lambda flag: joined.append(flag) or False)
    with pytest.raises(RuntimeError, match="could not place the weights"):
        _run(protocol, _worker(**kwargs), _request())
    # It re-raised its own cause, not the peer's refusal, after sending its flag.
    assert joined == [False]


def test_a_whole_model_fleet_that_loaded_everywhere_samples(monkeypatch):
    protocol = _run_env(monkeypatch)
    monkeypatch.setattr(protocol.store_fsdp, "ensure", _ensure_ok)
    crossed: list[bool] = []
    monkeypatch.setattr(
        partial_load_guard, "all_ranks_true", lambda flag: crossed.append(flag) or True)
    result = _run(protocol, _worker(ulysses=2), _request())
    assert crossed == [True]  # exactly one exchange, and the render went on
    assert result["rank"] == 0


def test_ring_native_preparation_finishes_before_existing_readiness(monkeypatch):
    protocol = _run_env(monkeypatch)
    events = []

    def loaded(*args, **kwargs):
        events.append("loaded")
        return _ensure_ok(*args, **kwargs)

    worker = _worker(ring=2)
    worker._attn = types.SimpleNamespace(
        prepare_native_attention=lambda: events.append("native-prepared"))
    monkeypatch.setattr(protocol.store_fsdp, "ensure", loaded)
    monkeypatch.setattr(partial_load_guard, "all_ranks_true",
                        lambda flag: events.append(("ready", flag)) or True)
    _run(protocol, worker, _request())
    assert events == ["loaded", "native-prepared", ("ready", True)]


def test_ring_native_failure_votes_not_ready_before_reraising(monkeypatch):
    protocol = _run_env(monkeypatch)
    monkeypatch.setattr(protocol.store_fsdp, "ensure", _ensure_ok)
    crossed = []
    monkeypatch.setattr(partial_load_guard, "all_ranks_true",
                        lambda flag: crossed.append(flag) or False)

    def unavailable():
        raise RuntimeError("native cuDNN normalization unavailable")

    worker = _worker(ring=2)
    worker._attn = types.SimpleNamespace(prepare_native_attention=unavailable)
    with pytest.raises(RuntimeError, match="normalization unavailable"):
        _run(protocol, worker, _request())
    assert crossed == [False]


def test_ring_peer_native_failure_prevents_sampling(monkeypatch):
    protocol = _run_env(monkeypatch)
    monkeypatch.setattr(protocol.store_fsdp, "ensure", _ensure_ok)
    monkeypatch.setattr(partial_load_guard, "all_ranks_true", lambda flag: False)
    worker = _worker(ring=2)
    worker._attn = types.SimpleNamespace(prepare_native_attention=lambda: None)

    def never(*args, **kwargs):
        raise AssertionError("peer native failure must prevent sampling")

    with pytest.raises(rank_readiness.PeerLoadFailureError):
        _run(protocol, worker, _request(), run_ksampler=never)


def test_an_fsdp_fleet_issues_no_readiness_exchange(monkeypatch):
    protocol = _run_env(monkeypatch)
    monkeypatch.setattr(protocol.store_fsdp, "ensure", _ensure_boom)
    crossed: list[bool] = []
    monkeypatch.setattr(
        partial_load_guard, "all_ranks_true", lambda flag: crossed.append(flag) or True)
    with pytest.raises(RuntimeError, match="could not place the weights"):
        _run(protocol, _worker(ulysses=2, fsdp=True), _request())
    assert crossed == []


def test_a_single_rank_render_issues_no_readiness_exchange(monkeypatch):
    protocol = _run_env(monkeypatch)
    monkeypatch.setattr(protocol.store_fsdp, "ensure", _ensure_boom)
    crossed: list[bool] = []
    monkeypatch.setattr(
        partial_load_guard, "all_ranks_true", lambda flag: crossed.append(flag) or True)
    with pytest.raises(RuntimeError, match="could not place the weights"):
        _run(protocol, _worker(world=1), _request())
    assert crossed == []


def test_a_cancelled_render_never_routes_through_the_ready_flag(monkeypatch):
    """A cancel is not a refusal: the driver cancels every rank and recycles."""
    protocol = _run_env(monkeypatch)
    from dgx_monarch.actor.sampling import RenderCancelledError

    def cancel(*a, **k):
        raise RenderCancelledError("cancelled mid-load")

    monkeypatch.setattr(protocol.store_fsdp, "ensure", cancel)
    crossed: list[bool] = []
    monkeypatch.setattr(
        partial_load_guard, "all_ranks_true", lambda flag: crossed.append(flag) or True)
    with pytest.raises(RenderCancelledError, match="cancelled mid-load"):
        _run(protocol, _worker(ulysses=2), _request())
    assert crossed == []


def test_the_exchange_is_the_first_collective_of_the_render():
    """Nothing collective may run between the load and the exchange.

    A rank already inside another collective when its peer reaches this one
    would deadlock at the exchange. On the topologies this covers,
    ``partial_load_guard`` owns the only ``dist`` collective in the package
    outside the FSDP and ring-attention paths, and the sampler reaches it after
    this exchange.
    """
    import ast

    collectives = {"all_reduce", "all_gather", "all_gather_into_tensor", "barrier",
                   "broadcast", "reduce", "reduce_scatter", "gather", "scatter",
                   "send", "recv", "monitored_barrier"}
    # Exempt by design: the guard the exchange runs ahead of; ring attention,
    # which runs inside the sampler after the exchange; and every FSDP file,
    # since an FSDP fleet takes no exchange.
    exempt = {"actor/partial_load_guard.py", "adapters/wan_ring_attention.py"}

    def _is_distributed(func: ast.Attribute) -> bool:
        target = func.value
        if isinstance(target, ast.Name):
            return target.id in ("dist", "torch")
        return isinstance(target, ast.Attribute) and target.attr == "distributed"

    offenders = []
    for path in sorted((SRC / "dgx_monarch").rglob("*.py")):
        name = path.relative_to(SRC / "dgx_monarch").as_posix()
        if name in exempt or "fsdp" in name:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in collectives and _is_distributed(node.func)):
                offenders.append(f"{name}:{node.lineno}")
    assert offenders == []
    # Not vacuous: the guard's own all_reduce is found when it is not exempt.
    guard = ast.parse((SRC / "dgx_monarch/actor/partial_load_guard.py").read_text())
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr in collectives and _is_distributed(node.func)
               for node in ast.walk(guard))


_CHILD = '''
import datetime, json, pathlib, sys, time, types

sys.path.insert(0, sys.argv[6])
rank, store, mode, out, hold_s = (
    int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))

import torch
import torch.distributed as dist

dist.init_process_group(
    "gloo", rank=rank, world_size=2, init_method="file://" + store,
    timeout=datetime.timedelta(seconds=5))

import dgx_monarch.adapters as adapters_pkg
from dgx_monarch import adoption_evidence
from dgx_monarch.actor import (
    partial_load_guard, rank_readiness, sample_protocol, sampling, worker_env)

worker_env.verify_sample_artifact_authorization = (
    lambda w, r, fn: [{"id": "cond"}, {"id": "uncond"}])
worker_env.activate_accuracy_waivers = lambda r: None
worker_env.assert_resident_artifact_identity = lambda w, s, e: None
worker_env.sample_rescue_consent = lambda r: {}
worker_env.accuracy_waiver_stamps = lambda: []
adapters_pkg.adapter_for = lambda model, override=None: types.SimpleNamespace(
    cfg_cond_padding="none", dp_cond_exempt_keys=frozenset())
sampling._dp_info = lambda: (0, 1)
adoption_evidence.build_worker_evidence = lambda *a, **k: None

if mode == "no_exchange":
    # The shape before this change: only dm-cfg2 carried a post-load exchange.
    rank_readiness.whole_model_topology = lambda worker: False


class _Model:
    model_lowvram = False
    diffusion_model = types.SimpleNamespace()


def ensure(worker, unet_name, options, loras, slot="cond", **kw):
    if rank == 1:
        raise RuntimeError("the second box could not place the weights")
    return (types.SimpleNamespace(model=_Model()), "hot")


sample_protocol.store_fsdp = types.SimpleNamespace(ensure=ensure)


def run_ksampler(patcher, request, progress_port, cancel_event=None,
                 dp_cond_exempt_keys=frozenset()):
    # The render's first collective: where a stranded peer waits today.
    partial_load_guard.check(patcher.model)
    return (torch.zeros(1, 2), None)


worker = types.SimpleNamespace(
    _setup_key=object(),
    topology={"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False},
    world=2, rank=rank, store=types.SimpleNamespace(family_override=None),
    _attn=None, _inject_for_topology=object(), _check_uma_reserve=lambda: None)
request = {
    "kind": "ksampler",
    "model": {"unet_name": "flux2-dev.safetensors",
              "model_sampling": {"kind": "sd3", "shift": 3.0}},
    "positive": None, "negative": None,
    "latent": {"samples": torch.zeros(1, 4, 8, 8)},
}

# Both ranks meet here first, so the window measured below is the exchange and
# not one rank's import time. It is the harness lining the ranks up, not a
# collective the render issues.
dist.barrier()

record = {"rank": rank, "mode": mode}
t0 = time.perf_counter()
try:
    sample_protocol.run_sample(
        worker, request, None, None,
        equalize_cond_lengths=lambda *a: (None, None),
        model_sampling_render_clone=lambda pat, s: types.SimpleNamespace(
            model=pat.model, cloned_from=pat),
        request_artifact_identity=lambda name, loras: {"id": name},
        run_ksampler=run_ksampler,
        run_custom=lambda *a, **k: (torch.zeros(1, 2), None),
        latent_signature=lambda s: {"sig": 1})
    record["outcome"] = "rendered"
except BaseException as exc:
    record["outcome"] = type(exc).__name__
    record["message"] = str(exc)[:600]
record["elapsed_s"] = round(time.perf_counter() - t0, 2)
pathlib.Path(out).write_text(json.dumps(record))
# A refused worker stays alive and keeps its group, so the peer's wait is the
# group's timeout rather than a dropped connection. It holds until the peer
# has written its own record, capped at hold_s, instead of sleeping the whole
# cap: the peer no longer needs the group once it has answered.
peer_record = pathlib.Path(out).with_name(f"rank{1 - rank}.json")
deadline = time.monotonic() + hold_s
while time.monotonic() < deadline and not peer_record.exists():
    time.sleep(0.05)
'''


def _two_ranks(tmp_path, mode, hold_s, wait_s):
    """Run one render on two real gloo ranks; rank 1 fails its load."""
    child = tmp_path / "rank_child.py"
    child.write_text(_CHILD)
    store = tmp_path / "gloo_store"
    outs = [tmp_path / f"rank{r}.json" for r in (0, 1)]
    # A pipe nobody drains fills and blocks the child, so the logs land in files.
    logs = [tmp_path / f"rank{r}.log" for r in (0, 1)]
    handles = [log.open("w") for log in logs]
    procs = [
        subprocess.Popen(
            [sys.executable, str(child), str(r), str(store), mode, str(outs[r]),
             str(hold_s), str(SRC)],
            stdout=handles[r], stderr=subprocess.STDOUT, text=True)
        for r in (0, 1)
    ]
    try:
        for proc in procs:
            proc.wait(timeout=wait_s)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        for handle in handles:
            handle.close()
    missing = [out for out in outs if not out.exists()]
    if missing:
        tails = "\n".join(log.read_text()[-2000:] for log in logs)
        pytest.fail(f"a rank wrote no record ({missing}):\n{tails}")
    return [json.loads(out.read_text()) for out in outs]


def test_two_real_ranks_one_refuses_late_and_the_peer_answers_at_once(tmp_path):
    """Over a real process group the healthy rank answers a typed refusal, not a strand.

    Rank 1's load fails after the fleet agreed. Rank 0 loaded cleanly and knows
    nothing of it. With the exchange, rank 0 crosses one MIN all_reduce, learns
    the fleet is not ready, and refuses well inside the 5 s group timeout.
    """
    head, peer = _two_ranks(tmp_path, mode="exchange", hold_s=0.0, wait_s=90)
    assert peer["outcome"] == "RuntimeError"
    assert "could not place the weights" in peer["message"]
    assert head["outcome"] == "PeerLoadFailureError"
    assert "a peer rank could not load" in head["message"]
    assert parse_refusal_tag(head["message"]).refusal_class.value == "P"
    assert head["elapsed_s"] < 4.0, "the healthy rank must not wait on the timeout"


def test_without_the_exchange_the_same_two_ranks_strand_the_healthy_one(tmp_path):
    """The negative control: the same failure with the exchange disabled.

    Rank 0 walks past the load into partial_load_guard.check, all-reduces alone,
    and answers only when the 5 s group timeout expires.
    """
    head, peer = _two_ranks(tmp_path, mode="no_exchange", hold_s=8.0, wait_s=120)
    assert peer["outcome"] == "RuntimeError"  # rank 1 refuses just the same
    assert head["outcome"] != "PeerLoadFailureError"
    assert head["outcome"] != "rendered"
    assert "timed out" in head["message"].lower(), head["message"]
    assert head["elapsed_s"] >= 4.0, (
        "the healthy rank waited on the group timeout, which is the strand")
