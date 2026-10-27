"""Every Worker rank scans the whole request for the MOSS-Image text attention bias (#135).

The driver refuses an effective bias under sequence parallelism before
dispatch only when it can read the checkpoint header
(nodes/model_patch_preflight.py). Without the header the render reaches the
Workers, where the sharded forward refuses only the bias its own call
receives. On uly2+cfg2 the two cfg groups run different prompts, so a bias on
one prompt would make one group refuse in its forward while the other waited
at the cfg gather. Each Worker rank therefore reads both prompts of the whole
request, the request pair or the guider pair, with the driver's own scan
(adapters/moss_image_seams.py), right after the readiness exchange and before
anything reads the conditioning, and refuses class P with the driver's text
when the loaded model is MOSS-Image and its own topology is sequence parallel.

The CPU tests drive run_sample on the stub harness of
tests/test_attention_pins_wiring.py, over the driver test's own tables
(tests/test_moss_image_attention_bias_preflight.py). The two-rank test runs
run_sample on two real gloo processes that stand for the two cfg groups of
uly2+cfg2, with the bias on the negative alone: both refuse at once with the
same class P error. Its control, without the scan, shows the strand the scan
removes: the group whose prompt holds no bias waits at the cfg gather until
the group timeout.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch.actor import partial_load_guard, sample_protocol
from dgx_monarch.adapters import moss_image_seams
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.moss_image_seams import CONTEXT_ATTENTION_BIAS_REFUSAL
from dgx_monarch.nodes import model_patch_preflight
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.topology import topology_from_preset
from test_moss_image_attention_bias_preflight import (
    EFFECTIVE,
    GUIDERS,
    INERT,
    MALFORMED,
    VALUES,
    _biased,
    _prompt,
    _strength_bias,
    _UnreadableBias,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
UNET = "moss_image_sft_bf16.safetensors"

# (preset, world): every preset that binds the sharded forward, and every
# preset that runs stock's forward with the bias.
SHARDED = {"uly2": 2, "ring2": 2, "uly2+fsdp": 2, "uly2+cfg2": 4, "uly4": 4, "ring4": 4}
STOCK_FORWARD = {"single": 1, "cfg2": 2, "dp2": 2, "cfg2+fsdp": 2, "dp2+fsdp": 2}


def _worker(preset: str = "uly2", *, family: str | None = "moss_image"):
    """A Worker after setup on ``preset``, holding a checkpoint of ``family``.

    Its topology is the dict mesh_setup pushes; the store records the loaded
    family as store_load does, the forced one included.
    """
    world = {**SHARDED, **STOCK_FORWARD}[preset]
    topo = topology_from_preset(preset, world)
    return types.SimpleNamespace(
        _setup_key=object(), world=world, rank=0,
        topology={"ulysses": topo.ulysses, "ring": topo.ring, "cfg": topo.cfg,
                  "dp": topo.dp, "fsdp": topo.fsdp},
        store=types.SimpleNamespace(family_override=None,
                                    current=types.SimpleNamespace(family=family)),
        _attn=types.SimpleNamespace(prepare_native_attention=lambda: None),
        _inject_for_topology=object(), _check_uma_reserve=lambda: None)


def _ksampler(positive=None, negative=None) -> dict:
    """A KSampler request as a Worker receives it, both prompts packed for the wire."""
    return {"kind": "ksampler", "model": {"unet_name": UNET},
            "positive": positive if positive is not None else _prompt(),
            "negative": negative if negative is not None else _prompt(),
            "latent": {"samples": torch.zeros(1, 4, 8, 8)}}


def _custom(guider) -> dict:
    """A SamplerCustom request, its prompts inside the guider spec."""
    return {"kind": "custom", "model": {"unet_name": UNET}, "guider": guider,
            "latent": {"samples": torch.zeros(1, 4, 8, 8)}}


def _cfg_guider(positive=None, negative=None) -> dict:
    return {"kind": "cfg", "positive": positive if positive is not None else _prompt(),
            "negative": negative if negative is not None else _prompt(), "cfg": 3.0}


PLACES = {
    "positive": lambda bias: _ksampler(positive=_biased(bias)),
    "negative": lambda bias: _ksampler(negative=_biased(bias)),
    "guider-positive": lambda bias: _custom(_cfg_guider(positive=_biased(bias))),
    "guider-negative": lambda bias: _custom(_cfg_guider(negative=_biased(bias))),
    "both": lambda bias: _ksampler(positive=_biased(bias), negative=_biased(bias)),
}


class _Harness:
    """run_sample on the stub harness of tests/test_attention_pins_wiring.py.

    It records each readiness flag this rank crosses, each render clone, and
    each request the sampler receives.
    """

    def __init__(self, monkeypatch) -> None:
        import dgx_monarch.adapters as adapters_pkg
        from dgx_monarch import adoption_evidence
        from dgx_monarch.actor import sampling, worker_env

        self.crossed: list[bool] = []
        self.clones: list[object] = []
        self.sampled: list[dict] = []
        monkeypatch.setattr(worker_env, "verify_sample_artifact_authorization",
                            lambda w, r, fn: [{"id": "cond"}, {"id": "uncond"}])
        monkeypatch.setattr(worker_env, "activate_accuracy_waivers", lambda r: None)
        monkeypatch.setattr(worker_env, "assert_resident_artifact_identity", lambda w, s, e: None)
        monkeypatch.setattr(worker_env, "sample_rescue_consent", lambda r: {})
        monkeypatch.setattr(adapters_pkg, "adapter_for",
                            lambda model, override=None: types.SimpleNamespace(
                                cfg_cond_padding="none", dp_cond_exempt_keys=frozenset()))
        monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
        monkeypatch.setattr(adoption_evidence, "build_worker_evidence", lambda *a, **k: None)
        # The request names a checkpoint no header read finds, so cfg2 is not dm-cfg2.
        monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda w, r: None)
        patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(diffusion_model=torch.nn.Module()), object_patches={})
        monkeypatch.setattr(sample_protocol.store_fsdp, "ensure", lambda *a, **k: (patcher, "hot"))
        monkeypatch.setattr(partial_load_guard, "all_ranks_true",
                            lambda flag: self.crossed.append(flag) or flag)

    def _clone(self, patcher, sampling):
        self.clones.append(patcher)
        return patcher

    def _sample(self, patcher, request, *args, **kwargs):
        self.sampled.append(request)
        return torch.zeros(1, 2), None

    def run(self, worker, request: dict) -> dict:
        return sample_protocol.run_sample(
            worker, request, None, None,
            equalize_cond_lengths=lambda adapter, positive, negative, latent: (positive, negative),
            model_sampling_render_clone=self._clone,
            request_artifact_identity=lambda name, loras: {"id": name},
            run_ksampler=self._sample,
            run_custom=lambda patcher, request, *a, **k: self._sample(patcher, request),
            latent_signature=lambda samples: {"sig": 1})


@pytest.fixture
def harness(monkeypatch) -> _Harness:
    return _Harness(monkeypatch)


def _assert_refusal(caught) -> None:
    text = str(caught.value)
    assert text == CONTEXT_ATTENTION_BIAS_REFUSAL
    assert parse_leading_refusal_tag(text).refusal_class is RefusalClass.PHYSICS
    assert text.endswith("See docs/TROUBLESHOOTING.md #135.")


def _exchanges(preset: str) -> list[bool]:
    """The readiness flags a rank crosses before the scan: one, except where no exchange runs."""
    worker = _worker(preset)
    return [] if worker.world < 2 or worker.topology["fsdp"] else [True]


# --- One scan for the driver and the Worker ------------------------------------------


@pytest.mark.parametrize("name", sorted(VALUES))
def test_the_driver_and_the_worker_read_one_scan_on_one_predicate(name):
    """The driver's name delegates to the seams scan the Worker calls; the predicate decides both."""
    value, effective = VALUES[name]
    request = _ksampler(positive=_biased(value))
    assert moss_image_seams.carries_context_attention_bias(request) is effective
    assert model_patch_preflight.carries_context_attention_bias(request) is effective
    if effective:
        with pytest.raises(UnsupportedModelError) as caught:
            sample_protocol.refuse_moss_image_attention_bias(_worker(), request, "moss_image")
        _assert_refusal(caught)
    else:
        sample_protocol.refuse_moss_image_attention_bias(_worker(), request, "moss_image")


def test_both_sides_call_the_scan_in_the_seams_module(monkeypatch):
    """Neither side keeps a second copy: both read the seams module's scan at call time."""
    seen: list = []
    monkeypatch.setattr(moss_image_seams, "carries_context_attention_bias",
                        lambda request: seen.append(request) or True)
    request = _ksampler()
    assert model_patch_preflight.carries_context_attention_bias(request) is True
    with pytest.raises(UnsupportedModelError):
        sample_protocol.refuse_moss_image_attention_bias(_worker(), request, "moss_image")
    assert seen == [request, request]


# --- A sequence-parallel MOSS-Image Worker refuses before sampling --------------------


@pytest.mark.parametrize("place", sorted(PLACES))
@pytest.mark.parametrize("preset", sorted(SHARDED))
def test_a_sequence_parallel_worker_refuses_a_bias_in_any_prompt_before_sampling(harness, preset, place):
    """After the load and the readiness exchange, before the render clone or the sampler."""
    with pytest.raises(UnsupportedModelError) as caught:
        harness.run(_worker(preset), PLACES[place](_strength_bias()))
    _assert_refusal(caught)
    assert harness.clones == [] and harness.sampled == []
    assert harness.crossed == _exchanges(preset)


@pytest.mark.parametrize("name", EFFECTIVE)
def test_every_effective_value_refuses_on_uly2(harness, name):
    with pytest.raises(UnsupportedModelError) as caught:
        harness.run(_worker("uly2"), _ksampler(negative=_biased(VALUES[name][0])))
    _assert_refusal(caught)
    assert harness.sampled == []


@pytest.mark.parametrize("name", sorted(GUIDERS))
def test_every_guider_kind_counts(harness, name):
    """basic, cfg on either prompt, and dual-model: the guider spec holds the prompts."""
    with pytest.raises(UnsupportedModelError) as caught:
        harness.run(_worker("uly2+cfg2"), _custom(GUIDERS[name]))
    _assert_refusal(caught)
    assert harness.sampled == []


# --- Everything else passes and samples with the request untouched --------------------


def _assert_sampled_untouched(harness, request: dict) -> None:
    assert len(harness.sampled) == 1 and harness.sampled[0] is request


@pytest.mark.parametrize("kind", ["ksampler", "custom"])
@pytest.mark.parametrize("preset", sorted(STOCK_FORWARD))
def test_a_stock_forward_worker_samples_with_the_bias(harness, preset, kind):
    """single, cfg2 and dp2, with or without FSDP, bind no sharded forward: stock's applies the bias."""
    positive, negative = _biased(_strength_bias()), _biased(_strength_bias())
    held = [prompt[0][1]["context_attention_bias"] for prompt in (positive, negative)]
    request = (_ksampler(positive, negative) if kind == "ksampler"
               else _custom(_cfg_guider(positive, negative)))
    harness.run(_worker(preset), request)
    _assert_sampled_untouched(harness, request)
    assert harness.crossed == _exchanges(preset)
    pair = request if kind == "ksampler" else request["guider"]
    after = [pair[key][0][1]["context_attention_bias"] for key in ("positive", "negative")]
    assert all(now is before for now, before in zip(after, held, strict=True))


@pytest.mark.parametrize("name", INERT)
@pytest.mark.parametrize("preset", ["uly2", "uly2+cfg2"])
def test_an_all_zero_bias_passes_on_a_sequence_parallel_worker(harness, preset, name):
    """It adds nothing in stock, and the sharded forward runs without it."""
    request = PLACES["both"](VALUES[name][0])
    harness.run(_worker(preset), request)
    _assert_sampled_untouched(harness, request)


@pytest.mark.parametrize("family", ["krea2", "flux3", "qwen_image", None])
def test_another_family_passes_on_uly2(harness, family):
    """Only MOSSImage.extra_conds reads the key; a store with no family record reads as none."""
    request = PLACES["both"](_strength_bias())
    harness.run(_worker("uly2", family=family), request)
    _assert_sampled_untouched(harness, request)


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_conditioning_passes_as_on_the_driver(harness, name):
    """A request that is not a mapping passes the scan; every other row samples on uly2+cfg2."""
    malformed = MALFORMED[name]
    if not isinstance(malformed, dict):
        sample_protocol.refuse_moss_image_attention_bias(_worker("uly2"), malformed, "moss_image")
        return
    request = {**_ksampler(), **malformed}
    harness.run(_worker("uly2+cfg2"), request)
    _assert_sampled_untouched(harness, request)


def test_a_bias_the_predicate_cannot_read_passes_the_scan_as_on_the_driver(harness):
    """The scan fails open where it cannot read; the forward's own guard then meets the bias."""
    unreadable = torch.zeros(1, 3).as_subclass(_UnreadableBias)
    request = _ksampler(positive=[[torch.zeros(1, 3, 8), {"context_attention_bias": unreadable}]])
    harness.run(_worker("uly2"), request)
    _assert_sampled_untouched(harness, request)
    with pytest.raises(RuntimeError, match="cannot be read"):
        moss_image_seams.refuse_context_attention_bias(unreadable)


# --- Where the scan runs ---------------------------------------------------------------


def test_run_sample_scans_once_after_the_readiness_exchange_and_before_any_conditioning_read():
    """Top level of run_sample: after the exchange every rank crosses alike, before the
    adoption evidence, the cfg equalization, the render clone and both samplers."""
    tree = ast.parse((SRC / "dgx_monarch/actor/sample_protocol.py").read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_sample")
    body = [ast.unparse(statement) for statement in run.body]
    scan = "refuse_moss_image_attention_bias(worker, request, getattr(resident, 'family', None))"
    assert body.count(scan) == 1
    assert sum(ast.unparse(node).startswith("refuse_moss_image_attention_bias(")
               for node in ast.walk(run) if isinstance(node, ast.Call)) == 1
    at = body.index(scan)
    exchange = next(i for i, line in enumerate(body) if "readiness.ready_or_raise()" in line)
    assert exchange < at
    for later in ("adoption_evidence.build_worker_evidence(", "equalize_cond_lengths(",
                  "model_sampling_render_clone(", "run_ksampler(", "run_custom("):
        assert next(i for i, line in enumerate(body) if later in line) > at, later
    # ``resident`` is the load guard's own read of the loaded slot.
    guard = next(node for node in run.body if isinstance(node, ast.Try))
    assert any(ast.unparse(node).startswith("resident = getattr(worker.store,")
               for node in ast.walk(guard) if isinstance(node, ast.Assign))


# --- Two real ranks: one prompt's bias, one cfg group each -------------------------------


_CHILD = '''
import datetime, json, math, os, pathlib, sys, time, types

sys.path.insert(0, sys.argv[5])
rank, store, mode, out, hold_s = (
    int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[6]))
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

import torch
import torch.distributed as dist

dist.init_process_group(
    "gloo", rank=rank, world_size=2, init_method="file://" + store,
    timeout=datetime.timedelta(seconds=5))

import dgx_monarch.adapters as adapters_pkg
from dgx_monarch import adoption_evidence
from dgx_monarch.actor import sample_protocol, sampling, worker_env
from dgx_monarch.adapters import moss_image_seams

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
if mode == "no_scan":
    # The shape before the scan: only the forward's own guard reads the bias.
    sample_protocol.refuse_moss_image_attention_bias = lambda *a, **k: None

patcher = types.SimpleNamespace(
    model=types.SimpleNamespace(diffusion_model=torch.nn.Module()), object_patches={})
sample_protocol.store_fsdp = types.SimpleNamespace(ensure=lambda *a, **k: (patcher, "hot"))
reached = []


def run_ksampler(patcher, request, progress_port, cancel_event=None,
                 dp_cond_exempt_keys=frozenset()):
    # The cfg split: this process stands for cfg group 0 (the positive) or 1 (the negative).
    reached.append(1)
    prompt = request["positive"] if rank == 0 else request["negative"]
    # The sharded forward's first guard reads only the bias its own call receives.
    moss_image_seams.refuse_context_attention_bias(prompt[0][1].get("context_attention_bias"))
    # The cfg gather that joins the two groups' outputs.
    parts = [torch.zeros(1, 2) for _ in range(2)]
    dist.all_gather(parts, torch.full((1, 2), float(rank)))
    return (torch.cat(parts), None)


bias = torch.zeros(1, 6)
bias[:, 1:4] = math.log(0.5)
request = {
    "kind": "ksampler",
    "model": {"unet_name": "moss_image_sft_bf16.safetensors"},
    "positive": [[torch.zeros(1, 6, 8), {}]],
    "negative": [[torch.zeros(1, 6, 8), {"context_attention_bias": bias}]],
    "latent": {"samples": torch.zeros(1, 4, 8, 8)},
}
worker = types.SimpleNamespace(
    _setup_key=object(),
    topology={"ulysses": 2, "ring": 1, "cfg": 2, "dp": 1, "fsdp": False},
    world=2, rank=rank,
    store=types.SimpleNamespace(family_override=None,
                                current=types.SimpleNamespace(family="moss_image")),
    _attn=None, _inject_for_topology=object(), _check_uma_reserve=lambda: None)

# The harness lines the ranks up, so the window measured below is the render's.
dist.barrier()

record = {"rank": rank, "mode": mode, "module": sample_protocol.__file__}
t0 = time.perf_counter()
try:
    sample_protocol.run_sample(
        worker, request, None, None,
        equalize_cond_lengths=lambda adapter, positive, negative, latent: (positive, negative),
        model_sampling_render_clone=lambda patcher, sampling: patcher,
        request_artifact_identity=lambda name, loras: {"id": name},
        run_ksampler=run_ksampler,
        run_custom=lambda *a, **k: (torch.zeros(1, 2), None),
        latent_signature=lambda samples: {"sig": 1})
    record["outcome"] = "rendered"
except BaseException as exc:
    record["outcome"] = type(exc).__name__
    record["message"] = str(exc)
record["elapsed_s"] = round(time.perf_counter() - t0, 2)
record["sampled"] = bool(reached)
pathlib.Path(out).write_text(json.dumps(record))
# A refused rank keeps its group until the peer has written its record, capped
# at hold_s, so the peer's wait is the group timeout and not a dropped connection.
peer_record = pathlib.Path(out).with_name(f"rank{1 - rank}.json")
deadline = time.monotonic() + hold_s
while time.monotonic() < deadline and not peer_record.exists():
    time.sleep(0.05)
'''


def _two_ranks(tmp_path, mode: str, hold_s: float, wait_s: float) -> list[dict]:
    """Run one uly2+cfg2 render on two real gloo ranks, the bias on the negative alone."""
    child = tmp_path / "rank_child.py"
    child.write_text(_CHILD)
    store = tmp_path / "gloo_store"
    outs = [tmp_path / f"rank{rank}.json" for rank in (0, 1)]
    # A pipe nobody drains fills and blocks the child, so the logs land in files.
    logs = [tmp_path / f"rank{rank}.log" for rank in (0, 1)]
    handles = [log.open("w") for log in logs]
    procs = [
        subprocess.Popen(
            [sys.executable, str(child), str(rank), str(store), mode, str(outs[rank]), str(SRC),
             str(hold_s)],
            stdout=handles[rank], stderr=subprocess.STDOUT, text=True)
        for rank in (0, 1)
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
    records = [json.loads(out.read_text()) for out in outs]
    assert all(record["module"].startswith(str(SRC)) for record in records), records
    return records


def test_two_real_ranks_refuse_a_bias_on_one_prompt_together_before_sampling(tmp_path):
    """Each rank holds the whole request, so the group whose prompt has no bias refuses too,
    with the same class P text and well inside the 5 s group timeout."""
    head, peer = _two_ranks(tmp_path, mode="scan", hold_s=0.0, wait_s=90)
    for record in (head, peer):
        assert record["outcome"] == "UnsupportedModelError", record
        assert record["message"] == CONTEXT_ATTENTION_BIAS_REFUSAL
        assert parse_leading_refusal_tag(record["message"]).refusal_class is RefusalClass.PHYSICS
        assert record["sampled"] is False
        assert record["elapsed_s"] < 4.0, "no rank may wait on the group timeout"


def test_without_the_scan_the_group_without_the_bias_waits_at_the_cfg_gather(tmp_path):
    """The control: the forward's guard refuses in the negative's group alone, and the
    positive's group answers only when the 5 s group timeout ends its gather."""
    head, peer = _two_ranks(tmp_path, mode="no_scan", hold_s=8.0, wait_s=120)
    assert peer["outcome"] == "UnsupportedModelError" and peer["sampled"] is True
    assert peer["message"] == CONTEXT_ATTENTION_BIAS_REFUSAL
    assert head["outcome"] != "UnsupportedModelError" and head["sampled"] is True
    assert "timed out" in head["message"].lower(), head["message"]
    assert head["elapsed_s"] >= 4.0, "the positive's group waited on the group timeout"
