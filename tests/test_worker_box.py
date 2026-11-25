"""A Worker's own walls name no host, and the driver names the box ahead of them.

A Worker cannot know the plain name of its own box (no rank speaks of a peer),
so its sentences say "here" (actor/store_residency.py, actor/store_slab_admit.py,
actor/comfy_bridge.py, actor/sampling_validation.py). In a fleet of more than
one rank it marks its rank on the error (actor/failure.mark_rank), because
Monarch's wrapper carries none. The driver knows each rank's plain name
(box_names.py), so where it receives the error (worker_box.py) it puts
"On the sibling, " ahead of the sentence and takes the mark off: no user reads a
host name, an address or a rank number. No test here resolves a name: each
pair injects which entry is the head.
"""
from __future__ import annotations

import ast
import pickle
import re
import socket
import sys
import types
from pathlib import Path

import pytest

from dgx_monarch import (
    box_names,
    capacity_fit,
    driver_footprint,
    driver_unload,
    mesh_helpers,
    mesh_lease,
    mesh_rpc,
    mesh_safety,
    mesh_setup,
    worker_box,
)
from dgx_monarch.actor import failure, partial_load_guard, store_identity, store_residency, store_slab_admit
from dgx_monarch.mesh_safety import StockLoadCapacityError
from dgx_monarch.nodes import fleet_cleanup, refusal_lead
from dgx_monarch.refusal import (
    RefusalClass,
    parse_leading_refusal_tag,
    parse_refusal_tag,
    read_rank,
    refusal,
)
from dgx_monarch.studio import job_observations

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
GIB = 2 ** 30
HOST = "worker-host-name"   # what socket.gethostname() answers on the Worker
NAME = "p-h3-image_bf16.safetensors"
TAG = "[dgxm:C guard=stock_load_preflight waivable=0] "
WALL = f"stock residency cannot load {NAME} here: the load needs"


def _named_host(monkeypatch) -> None:
    """The Worker's own host name, which must never reach a sentence."""
    monkeypatch.setattr(socket, "gethostname", lambda: HOST)
    monkeypatch.setattr(store_residency, "_host", lambda: HOST)


def _no_host(text: str) -> None:
    assert HOST not in text, text
    assert not re.search(r"\bon host\b|on worker host|\(host ", text), text


def _stock_wall() -> str:
    fit = capacity_fit.StockFit(False, True, int(37.7 * GIB), int(82.1 * GIB))
    blocker = ("slab_weights is off in the effective worker args",
               "use a pruned or quantized artifact, or free unified memory on this host")
    return refusal(RefusalClass.CAPACITY, store_residency._no_rescue_text(NAME, fit, blocker),
                   guard=store_residency.RESCUE_GUARD, waivable=False,
                   troubleshooting=store_residency._TROUBLESHOOTING)


def _worker(rank, world):
    return types.SimpleNamespace(rank=rank, world=world)


def _pair(monkeypatch, *, world=2, head="box-a", hosts=("box-a", "box-b")):
    monkeypatch.setattr(driver_footprint, "host_is_local", lambda name: name == head)
    return types.SimpleNamespace(world=world, gpus_per_host=1, owns_hosts=True,
                                 config=types.SimpleNamespace(hosts=tuple(
                                     types.SimpleNamespace(name=name, address=f"tcp://{name}:26600")
                                     for name in hosts)))


def _local(world=1):
    """A this_host() mesh: every rank on the driver's box."""
    return types.SimpleNamespace(world=world, gpus_per_host=1, owns_hosts=False, config=None)


def _remote(exc: BaseException, method: str = "sample"):
    """A Worker's raise as the driver receives it: Monarch's real ``ActorError``
    after a pickle round trip, the way #139's tests build one."""
    from monarch.actor import ActorError

    try:
        raise exc
    except type(exc) as raised:
        wrapper = ActorError(raised, f"Actor call dgxm_worker.{method} failed.")
    return pickle.loads(pickle.dumps(wrapper))  # noqa: S301 - trusted round-trip


def _marked(text: str, rank: int, world: int = 2) -> StockLoadCapacityError:
    exc = StockLoadCapacityError(text)
    failure.mark_rank(_worker(rank, world), exc)
    return exc


def _unmarked(text: str) -> None:
    assert "[dgxm:rank" not in text, text


# --- The Worker's own sentences ---------------------------------------------

def test_the_stock_wall_says_here_and_names_no_host(monkeypatch):
    _named_host(monkeypatch)
    text = _stock_wall()
    assert text.startswith(TAG + WALL), text
    _no_host(text)


def test_the_rescue_offer_says_here_and_names_no_host(monkeypatch):
    _named_host(monkeypatch)
    fit = capacity_fit.StockFit(False, True, int(35 * GIB), int(40 * GIB))
    text = store_residency._rescue_offer_text(types.SimpleNamespace(unet_name="flux.safetensors"), fit)
    assert text.startswith("capacity rescue available for flux.safetensors here: stock residency needs")
    _no_host(text)


def test_the_comfy_managed_wall_says_here_and_names_no_host(monkeypatch, tmp_path):
    _named_host(monkeypatch)
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"\0" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 1)
    decision = store_residency.ResidencyDecision(False, store_residency.RUNG_COMFY_MANAGED, "reason", False)
    with pytest.raises(StockLoadCapacityError) as raised:
        store_residency.preload_capacity_check(decision, str(path), "m.safetensors", {},
                                               preflight=lambda *_a: None, pinned_staging=False)
    text = str(raised.value)
    assert "comfy-managed residency cannot load m.safetensors here: the checkpoint is" in text
    _no_host(text)


def test_the_slab_card_says_here_and_its_helpers_take_no_host(monkeypatch):
    _named_host(monkeypatch)
    fit = capacity_fit.SlabFit(False, True, 60 * GIB, 40 * GIB, 5 * GIB, "", 0)
    text = store_slab_admit.card(fit, "flux2.safetensors")
    assert text.startswith("slab residency cannot load flux2.safetensors here: the zero-copy slab")
    _no_host(text)
    with pytest.raises(StockLoadCapacityError) as raised:
        store_slab_admit.admit(fit, "flux2.safetensors")
    assert "slab residency cannot load flux2.safetensors here:" in str(raised.value)
    for helper in (store_slab_admit.card, store_slab_admit.admit, store_slab_admit.admit_bake):
        assert "host" not in helper.__code__.co_varnames[:helper.__code__.co_argcount], helper


def test_the_lora_bake_backstop_says_here(monkeypatch):
    import torch

    _named_host(monkeypatch)

    class _Model:
        def __getattr__(self, _name):
            return torch.zeros(1 << 20, dtype=torch.bfloat16)  # 2 MiB a key

    active = types.SimpleNamespace(model=_Model(), patches={f"w{i}": [] for i in range(40)})
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 5 << 30)
    with pytest.raises(StockLoadCapacityError) as raised:
        store_slab_admit.admit_bake(active, "m.safetensors")
    text = str(raised.value)
    assert "the LoRA bake for m.safetensors cannot proceed here: comfy's merge-and-free" in text
    _no_host(text)


def test_a_missing_model_says_it_was_not_found_here(monkeypatch):
    from dgx_monarch.actor import comfy_bridge

    _named_host(monkeypatch)
    monkeypatch.setitem(sys.modules, "folder_paths",
                        types.SimpleNamespace(get_full_path=lambda _kind, _name: None))
    with pytest.raises(FileNotFoundError) as raised:
        comfy_bridge.resolve_model_path("diffusion_models", "x.safetensors")
    text = str(raised.value)
    assert text.startswith("model 'x.safetensors' (kind 'diffusion_models') was not found here. "
                           "Cluster renders need the model present on every worker host")
    _no_host(text)


@pytest.mark.parametrize("sampler, scheduler, opening", [
    ("res_2s", "simple", "sampler 'res_2s' is not registered with ComfyUI here; "),
    ("euler", "bong_tangent", "scheduler 'bong_tangent' is not registered with ComfyUI here; "),
])
def test_an_unregistered_sampler_or_scheduler_says_here(monkeypatch, sampler, scheduler, opening):
    from dgx_monarch.actor.sampling_validation import require_known_sampler

    _named_host(monkeypatch)
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSampler = type("KSampler", (), {"SAMPLERS": ["euler"], "SCHEDULERS": ["simple"]})
    comfy = types.ModuleType("comfy")
    comfy.samplers = samplers
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    with pytest.raises(ValueError) as raised:
        require_known_sampler(sampler, scheduler)
    assert str(raised.value).startswith(opening)
    _no_host(str(raised.value))


def test_no_worker_wall_interpolates_a_host():
    banned = re.compile(r"gethostname|_host\(\)|host_name\(\)|\{host\}")
    for name in ("actor/store_residency.py", "actor/store_slab_admit.py", "actor/store_load.py",
                 "actor/comfy_bridge.py", "actor/sampling_validation.py"):
        for node in ast.walk(ast.parse((SRC / name).read_text())):
            if isinstance(node, ast.JoinedStr):
                assert not banned.search(ast.unparse(node)), (name, ast.unparse(node))


# --- The Worker says which rank raised ------------------------------------

def test_a_worker_of_several_ranks_marks_its_rank_and_the_text_keeps_one_tag():
    wall = _stock_wall()
    text = str(_marked(wall, 1))
    assert text == wall + "\n[dgxm:rank 1]"
    assert parse_leading_refusal_tag(text) is not None
    assert parse_refusal_tag(text) is not None, "the mark is never read as a second tag"
    assert len(job_observations._TAG.findall(text)) == 1
    assert read_rank(text) == (wall, 1)


def test_no_mark_where_the_driver_knows_the_box_or_the_text_speaks_for_the_fleet():
    wall = _stock_wall()
    # A Fleet actor is rank 0 of its own world-1 mesh, and a Worker not set up has no world.
    assert str(_marked(wall, 0, world=1)) == wall
    assert str(_marked(wall, None, world=None)) == wall
    divergence = partial_load_guard.PartialLoadDivergenceError(partial_load_guard.REFUSAL_TEXT)
    failure.mark_rank(_worker(1, 2), divergence)
    assert str(divergence) == partial_load_guard.REFUSAL_TEXT

    class Shaped(RuntimeError):
        def __init__(self, expected, actual):
            super().__init__(expected, actual)

    shaped = Shaped(3, 4)
    failure.mark_rank(_worker(1, 2), shaped)
    assert shaped.args == (3, 4), "a constructor of other arguments keeps them, so it still unpickles"
    stop = KeyboardInterrupt("stop")
    failure.mark_rank(_worker(1, 2), stop)
    assert stop.args == ("stop",)
    twice = _marked(wall, 1)
    failure.mark_rank(_worker(1, 2), twice)
    assert str(twice).count("[dgxm:rank") == 1


def test_every_load_and_sample_leaves_the_worker_with_its_mark(monkeypatch):
    mm = types.ModuleType("comfy.model_management")
    mm.soft_empty_cache = lambda *a, **k: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(failure, "_cuda_context_dead", lambda: False)
    wall = _stock_wall()

    class Worker:
        rank, world, store = 1, 2, None

        @failure.cleanup_on_failure
        def load(self):
            raise StockLoadCapacityError(wall)

    with pytest.raises(StockLoadCapacityError) as raised:
        Worker().load()
    assert str(raised.value) == wall + "\n[dgxm:rank 1]"
    poisoned = Worker()
    poisoned._cuda_context_poisoned = True
    with pytest.raises(failure.CudaContextPoisonedError) as raised:
        poisoned.load()
    assert str(raised.value).endswith("\n[dgxm:rank 1]")


def test_the_parity_proof_says_which_rank_lacks_the_file(monkeypatch):
    def missing(_name, _loras):
        raise FileNotFoundError("model 'x.safetensors' (kind 'diffusion_models') was not found here.")

    with pytest.raises(FileNotFoundError) as raised:
        store_identity.identity_reply(_worker(1, 2), [{"unet_name": "x.safetensors"}], missing)
    assert str(raised.value).endswith("\n[dgxm:rank 1]")
    with pytest.raises(FileNotFoundError) as raised:
        store_identity.identity_reply(_worker(0, 1), [{"unet_name": "x.safetensors"}], missing)
    _unmarked(str(raised.value))
    monkeypatch.setattr(socket, "gethostname", lambda: HOST)
    reply = store_identity.identity_reply(_worker(1, 2), [{"unet_name": "x", "loras": []}],
                                          lambda name, loras: {"id": name, "loras": loras})
    assert reply == {"host": HOST, "rank": 1, "artifact_sets": [{"id": "x", "loras": []}]}


def test_the_mark_survives_the_wire():
    wrapper = _remote(_marked(_stock_wall(), 1))
    assert str(wrapper.exception).endswith("\n[dgxm:rank 1]")


# --- The driver names the box ------------------------------------------------

@pytest.mark.parametrize("rank, words", [(0, "On the head, "), (1, "On the sibling, ")])
def test_a_wall_from_either_box_of_the_pair_opens_with_that_box(monkeypatch, rank, words):
    wrapper = _remote(_marked(_stock_wall(), rank))
    worker_box.name(wrapper, _pair(monkeypatch))
    inner = str(wrapper.exception)
    assert inner.startswith(TAG + words + WALL), inner
    assert parse_leading_refusal_tag(inner) is not None
    for text in (inner, str(wrapper)):
        _unmarked(text)
        assert "box-a" not in text and "box-b" not in text and "26600" not in text
    assert str(wrapper).startswith(f"{words}actor call dgxm_worker.sample failed.")


def test_a_local_fleet_reads_here_with_no_prefix(monkeypatch):
    """Cell 9: one rank on the driver's box. "here" is the box the card calls this box."""
    wall = _stock_wall()
    wrapper = _remote(StockLoadCapacityError(wall))
    worker_box.name(wrapper, _local())
    assert str(wrapper.exception) == wall
    assert str(wrapper).startswith("Actor call dgxm_worker.sample failed.")


def test_one_rank_on_the_sibling_alone_names_the_sibling(monkeypatch):
    wrapper = _remote(StockLoadCapacityError(_stock_wall()))
    worker_box.name(wrapper, _pair(monkeypatch, world=1, head="elsewhere", hosts=("box-b",)))
    assert str(wrapper.exception).startswith(TAG + "On the sibling, " + WALL)


def test_a_fleet_job_takes_the_box_of_the_actor_it_went_to(monkeypatch):
    # A Fleet actor is world 1 and marks nothing; the driver's index decides.
    wrapper = _remote(StockLoadCapacityError(_stock_wall()))
    worker_box.name(wrapper, _pair(monkeypatch), 1)
    assert str(wrapper.exception).startswith(TAG + "On the sibling, " + WALL)


def test_an_unmarked_wall_at_world_two_is_left_as_its_worker_wrote_it(monkeypatch):
    """A Worker one release behind marks nothing and still names its own host."""
    old = refusal(RefusalClass.CAPACITY, f"stock residency cannot load {NAME} on {HOST}: the load needs 9 GiB",
                  guard=store_residency.RESCUE_GUARD, waivable=False)
    wrapper = _remote(StockLoadCapacityError(old))
    worker_box.name(wrapper, _pair(monkeypatch))
    assert str(wrapper.exception) == old


def test_a_file_missing_on_the_sibling_names_it_on_the_wrapper_too(monkeypatch):
    """Untagged: the prompt error stays Monarch's wrapper (#145), so its first line names the box."""
    missing = FileNotFoundError("model 'x.safetensors' (kind 'diffusion_models') was not found here. "
                                "Cluster renders need the model present on every worker host under the same name.")
    failure.mark_rank(_worker(1, 2), missing)
    wrapper = _remote(missing, "artifact_identity")
    worker_box.name(wrapper, _pair(monkeypatch))
    first, _newline, rest = str(wrapper).partition("\n")
    assert first == "On the sibling, actor call dgxm_worker.artifact_identity failed."
    assert "was not found here." in rest
    _unmarked(str(wrapper))
    assert str(wrapper.exception).startswith(
        "On the sibling, model 'x.safetensors' (kind 'diffusion_models') was not found here.")


def test_a_crash_keeps_its_acronym_and_a_driver_error_is_left_alone(monkeypatch):
    crash = RuntimeError("CUDA error: an illegal memory access was encountered")
    failure.mark_rank(_worker(0, 2), crash)
    wrapper = _remote(crash)
    worker_box.name(wrapper, _pair(monkeypatch))
    assert str(wrapper.exception) == "On the head, CUDA error: an illegal memory access was encountered"
    driver = RuntimeError("Actor mesh is not set up")
    worker_box.name(driver, _pair(monkeypatch, world=1))
    assert str(driver) == "Actor mesh is not set up"


def test_naming_twice_names_once(monkeypatch):
    wrapper = _remote(StockLoadCapacityError(_stock_wall()))
    handle = _pair(monkeypatch)
    worker_box.name(wrapper, handle, 1)
    worker_box.name(wrapper, handle, 1)
    assert str(wrapper.exception).count("On the sibling, ") == 1
    assert str(wrapper).count("On the sibling, ") == 1


def test_the_fleet_wide_divergence_refusal_names_no_box(monkeypatch):
    divergence = partial_load_guard.PartialLoadDivergenceError(partial_load_guard.REFUSAL_TEXT)
    failure.mark_rank(_worker(1, 2), divergence)
    wrapper = _remote(divergence)
    worker_box.name(wrapper, _pair(monkeypatch))
    assert str(wrapper.exception) == partial_load_guard.REFUSAL_TEXT


def test_the_words_come_from_the_one_rule():
    assert box_names.on_box(box_names.HEAD) == "On the head, "
    assert box_names.on_box(box_names.SIBLING) == "On the sibling, "
    assert box_names.on_box(box_names.LOCAL) == ""
    assert box_names.on_box(box_names.label(box_names.HEAD, 1)) == "On GPU 1 on the head, "
    assert worker_box.lead(TAG + "stock residency x", "On the sibling, ") == TAG + "On the sibling, stock residency x"
    assert worker_box.lead("Actor call a failed.", "On the head, ") == "On the head, actor call a failed."
    source = (SRC / "worker_box.py").read_text()
    assert '"the head"' not in source and '"the sibling"' not in source


# --- Every driver surface reads the named text --------------------------------

def test_the_prompt_error_leads_with_the_tag_then_the_box(monkeypatch):
    wrapper = _remote(_marked(_stock_wall(), 1))
    worker_box.name(wrapper, _pair(monkeypatch))

    @refusal_lead.lead_with_refusal
    def sample():
        raise wrapper

    with pytest.raises(StockLoadCapacityError) as raised:
        sample()
    assert str(raised.value).startswith(TAG + "On the sibling, " + WALL)
    assert raised.value.__cause__ is wrapper


def test_a_mark_no_receive_point_took_off_never_reaches_the_prompt_error():
    wrapper = _remote(_marked(_stock_wall(), 1))

    @refusal_lead.lead_with_refusal
    def sample():
        raise wrapper

    with pytest.raises(StockLoadCapacityError) as raised:
        sample()
    _unmarked(str(raised.value))
    assert str(raised.value).startswith(TAG + WALL)


def test_the_driver_unload_quotes_the_named_wall(monkeypatch):
    wrapper = _remote(_marked(_stock_wall(), 1))
    worker_box.name(wrapper, _pair(monkeypatch))
    amended = driver_unload._amended(wrapper, "The driver tried to unload first.")
    first = str(amended).partition("\n")[0]
    assert first.startswith(TAG + "On the sibling, " + WALL)
    assert first.endswith("The driver tried to unload first.")
    _unmarked(str(amended))


def test_an_eager_load_or_gate_leg_is_named_where_call_all_receives_it(monkeypatch):
    wrapper = _remote(_marked(_stock_wall(), 1), "load_model")
    handle = _pair(monkeypatch)
    handle.active_worker_args = {}
    handle.workers = types.SimpleNamespace(load_model=types.SimpleNamespace(call=lambda *_a, **_k: object()))

    def await_or_evict(_future, _timeout):
        raise wrapper

    handle._await_or_evict = await_or_evict
    monkeypatch.setattr(mesh_setup, "dispatch_endpoint",
                        lambda _handle, _name, send, setup_token=None: send())
    monkeypatch.setattr(mesh_setup, "AMBIGUOUS_MUTATION_ENDPOINTS", frozenset())
    with pytest.raises(type(wrapper)):
        mesh_rpc.call_all_bound(handle, "load_model", NAME, {}, [])
    assert str(wrapper.exception).startswith(TAG + "On the sibling, " + WALL)
    _unmarked(str(wrapper))


def test_a_collective_sample_is_named_where_it_is_collected(monkeypatch):
    wrapper = _remote(_marked(_stock_wall(), 0))

    def collect(*_args):
        raise wrapper

    monkeypatch.setattr(mesh_lease, "collect_with_liveness", collect)
    monkeypatch.setattr(mesh_helpers, "mark_defunct_preserving_primary", lambda *_a, **_k: None)
    with pytest.raises(type(wrapper)):
        mesh_lease.collect_sample(_pair(monkeypatch), object(), 5.0)
    assert str(wrapper.exception).startswith(TAG + "On the head, " + WALL)


def test_a_fleet_job_is_named_where_the_drain_collects_it(monkeypatch):
    wrapper = _remote(StockLoadCapacityError(_stock_wall()))
    handle = _pair(monkeypatch)

    def collect_one(_future, timeout_s):
        raise wrapper

    handle.collect_one = collect_one
    item = fleet_cleanup._FleetPending(job_index=0, future=object(), progress=None, rank=1)
    failures = list(fleet_cleanup._drain_fleet_collections(
        handle, [item], 5.0, lambda _i, result: result, lambda *_a: None,
        lambda *_a: None, lambda *_a: None))
    assert [exc for _item, exc, _label in failures] == [wrapper]
    assert str(wrapper.exception).startswith(TAG + "On the sibling, " + WALL)


def test_a_fleet_parity_proof_names_the_box_that_lacks_the_file(monkeypatch):
    from dgx_monarch.actor import model_store

    missing = FileNotFoundError("model 'x.safetensors' (kind 'diffusion_models') was not found here.")
    wrapper = _remote(missing, "artifact_identity")
    handle = _pair(monkeypatch)
    handle.setup_generation = 3
    picked = []

    def pick(**coords):
        picked.append(coords)
        return types.SimpleNamespace(artifact_identity=types.SimpleNamespace(call_one=lambda _specs: object()))

    handle.workers = types.SimpleNamespace(extent=types.SimpleNamespace(labels=("hosts", "gpus")), slice=pick)

    def await_or_evict(_future, timeout_s):
        raise wrapper

    handle._await_or_evict = await_or_evict
    monkeypatch.setattr(model_store, "request_artifact_identity", lambda *_a: {})
    monkeypatch.setattr(mesh_safety, "assert_request_artifact_binding", lambda *_a: None)
    monkeypatch.setattr(mesh_safety, "assert_normal_render_residency_mode", lambda *_a: None)
    with pytest.raises(type(wrapper)):
        mesh_rpc.verify_request_artifacts(handle, {"model": {"unet_name": "x.safetensors"}}, 1)
    assert picked == [{"gpus": 0, "hosts": 1}]
    assert str(wrapper).startswith("On the sibling, actor call dgxm_worker.artifact_identity failed.")
