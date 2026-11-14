"""The render site frees the driver's models before a model load still to come that will not fit.

On a pair, p-h3-image's first use staged its text encoder and prompt upsampler
on the driver (about 25 GiB) before its 37.7 GiB file loaded, and the head
priced the stock load at 84.1 GiB against 74.0 available: the ranks' load
agreement offered the slab card and the first-use check could not complete
(2026-11-13). nodes/pending_load_relief.py prices that load at the Workers'
own price once the render-site checks pass and before the gate, confirms a
short reading with the Workers' own quote, runs driver_unload's rescue where
it does not fit, and refuses nothing. This file holds:

- which loads are still to come: auto, an explicit preset's deferral, the
  gate's capacity check, and never a placed or rendered model;
- a pending load that fits pays no unload and no Worker call, and one the
  Workers already hold, or would swap with their own credit, pays the quote
  alone;
- what the Workers' quote counts as a stop;
- the rescue's order (unload, then trim) and that it never raises, freed or not;
- the boxes: a local or one-box fleet, and a driver on a box with no Worker;
- the cases it leaves alone: a dual-model request, comfy-managed residency,
  an FSDP preset, a busy driver, a fault;
- the render-site call order, and a render the checks refuse keeping today's
  words;
- one Spark end to end, where the Worker's own load check is the judge: staged
  text models freed and the load fits, a fit that frees nothing, and the same
  graph without the step meeting the Worker's refusal;
- end to end on the first-use rig: the no-render path, a LoRA ceremony, an
  explicit preset's eager placement, and a warm re-queue; and the no-render
  path and a LoRA ceremony on a model an earlier render left loaded, whose
  unload makes the load still to come again.
"""
from __future__ import annotations

import types

import pytest
import torch

from dgx_monarch import driver_unload, mesh_safety, swap_credit
from dgx_monarch.capacity_fit import SlabFit
from dgx_monarch.nodes import pending_load_relief, render_preflight
from dgx_monarch.telemetry import render_progress
from first_use_gate_helpers import _forget_classifications  # noqa: F401  # autouse fixture import.
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
)

_GIB = 2 ** 30
UNET = "p-h3-image_bf16.safetensors"
FILE_GIB = 37.7
# The Driver rig stands this read in; the fail-open test puts it back.
_REAL_WORKERS_STOP = pending_load_relief.workers_stop


@pytest.fixture(autouse=True)
def _rig(monkeypatch):
    pending_load_relief.reset()
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", None)
    monkeypatch.setattr(render_preflight, "_LAST_SUBMITTED_UNET", None)
    monkeypatch.delenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, raising=False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    # The render-site checks' swap credit asks a Worker; these cases replace no resident.
    monkeypatch.setattr(driver_unload, "_swap_credit", lambda _swap: swap_credit.NONE)
    yield
    pending_load_relief.reset()


@pytest.fixture
def checkpoint(tmp_path):
    """A sparse file of the BF16 DiT's size: the stock price reads only its length."""
    path = tmp_path / UNET
    with open(path, "wb") as handle:
        handle.truncate(int(FILE_GIB * _GIB))
    return str(path)


class Driver:
    """The driver's box: ComfyUI's held models, MemAvailable, the actors' trim,
    and the Workers' quote.

    ``steps`` records, in order, every unload, trim and quote, and every Worker
    call the handle makes. ``workers_admit`` is the quote's answer whenever it
    is asked: a reuse, or a swap the Worker's own credit covers, admits; by
    default the Worker prices as the local price does, which asks only when
    that price reads short, so it stops the load.
    """

    def __init__(self, monkeypatch, *, reading_gib=74.0, held=("Qwen35TEModel_", "MiniMaxH3TEModel_"),
                 freed_gib=25.0, trimmable=True, workers_admit=False):
        self.steps: list[str] = []
        self.held = list(held)
        self.reading = reading_gib * _GIB
        self.freed = freed_gib * _GIB
        self.workers_admit = workers_admit
        driver = self

        def quote(*_args, **_kwargs):
            driver.steps.append("quote")
            return not driver.workers_admit

        monkeypatch.setattr(pending_load_relief, "workers_stop", quote)

        class MM:
            @property
            def current_loaded_models(self):
                return [types.SimpleNamespace(model=types.SimpleNamespace(
                    model=type(name, (), {})())) for name in driver.held]

            def unload_all_models(self):
                driver.steps.append("comfy-unload")
                driver.held = []
                driver.reading += driver.freed

            def soft_empty_cache(self):
                pass

        monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: MM())
        monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(driver.reading))
        monkeypatch.setattr(driver_unload, "trimmable", lambda _handle: trimmable)

        def trim(_handle, level=driver_unload.TRIM_LEVEL):
            driver.steps.append("actor-trim")
            before = int(driver.reading)
            driver.reading += 2 * _GIB
            return driver_unload.Trimmed(2, before, int(driver.reading))

        monkeypatch.setattr(driver_unload, "trim_actors", trim)


class Handle:
    """A fleet on the driver's own box (a local mesh), recording every Worker call."""

    def __init__(self, driver: Driver, *, owns_hosts=False, world=2):
        self.owns_hosts = owns_hosts
        self.world = world
        self.config = types.SimpleNamespace(hosts=(), worker_args={})
        self.defunct = False
        self.teardown_complete = False
        self.setup_cleanup_state = None
        self.driver = driver

    def call_all(self, method, *_args, **_kwargs):
        self.driver.steps.append(method)
        return []


def _model(handle, *, preset="auto", worker_args=None, unet=UNET):
    mesh = types.SimpleNamespace(handle=handle, topology_preset=preset,
                                 worker_args=dict(worker_args or {}))
    return types.SimpleNamespace(mesh=mesh, unet_name=unet, options={}, loras=())


def _relieve(model, checkpoint, family="minimax_h3", request=None, weights_loaded=None):
    pending_load_relief.relieve(model, request or {"kind": "custom"}, family=family,
                                path=checkpoint, weights_loaded=weights_loaded)


# --- which loads are still to come ----------------------------------------------------


def test_which_loads_are_still_to_come(monkeypatch):
    auto = _model(None)
    explicit = _model(None, preset="uly2")
    assert pending_load_relief.load_pending(auto, None)
    assert not pending_load_relief.load_pending(explicit, None)  # the loader placed it
    pending_load_relief.note_deferred(UNET)
    assert pending_load_relief.load_pending(explicit, None)      # the loader deferred it
    pending_load_relief.note_placed(UNET)
    assert not pending_load_relief.load_pending(explicit, None)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", UNET)
    assert not pending_load_relief.load_pending(auto, None)      # a render placed it
    # The gate's capacity check says itself whether the load happened.
    assert pending_load_relief.load_pending(auto, False)
    assert not pending_load_relief.load_pending(_model(None), True)


def test_every_unload_of_every_rank_marks_the_load_still_to_come():
    """Each driver site that unloads every rank marks the next load as still to
    come, so a placement it undid is never read as standing: the ceremonies, the
    no-render path and its fall-through, the slab proof, the cross-mode
    reference, the FSDP cleanup, a rescue's dropped slot, and a hard Clear VRAM."""
    import re
    from pathlib import Path

    nodes = Path(pending_load_relief.__file__).parent
    sites = []
    for source in sorted(nodes.glob("*.py")):
        if source.name == "pending_load_relief.py":
            continue  # its docstring names the sites
        lines = source.read_text().splitlines()
        for index, line in enumerate(lines):
            if re.search(r"call_all\(\s*\"unload\"", line):
                sites.append(source.name)
                after = "\n".join(lines[index:index + 5])
                assert "pending_load_relief.note_unloaded()" in after, f"{source.name}:{index + 1}"
    assert len(sites) >= 7
    ops = (nodes / "ops.py").read_text()
    workers = ops.index('handle.call_all("clear_vram", level')
    assert "pending_load_relief.note_unloaded()" in ops[workers:workers + 300]


def test_a_stop_of_the_workers_makes_an_eager_placement_pending_again():
    """Every door that stops the Workers' processes drains the memos
    (``recycle_drain.drop_residency_memos``); an explicit preset's eager
    placement went with them, so the next render's load is still to come."""
    from dgx_monarch.nodes import recycle_drain

    explicit = _model(None, preset="uly2")
    pending_load_relief.note_placed(UNET)
    assert not pending_load_relief.load_pending(explicit, None)
    recycle_drain.drop_residency_memos()
    assert pending_load_relief.load_pending(explicit, None)
    pending_load_relief.note_placed(UNET)  # the loader places it again
    assert not pending_load_relief.load_pending(explicit, None)


def test_an_unload_makes_the_load_pending_over_the_memos_until_a_load_succeeds(
        monkeypatch, checkpoint):
    """The memos still name a model the Workers no longer hold. The mark stays
    through the next render's check, which places nothing, and through a load
    that fails; only a load that succeeds clears it: a successful render or a
    loader's placement. The memo drain at a reset is not a load."""
    driver = Driver(monkeypatch, reading_gib=100.1)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", UNET)
    auto, explicit = _model(Handle(driver)), _model(Handle(driver), preset="uly2")
    pending_load_relief.note_unloaded()
    assert pending_load_relief.load_pending(auto, None)
    assert pending_load_relief.load_pending(explicit, None)
    assert not pending_load_relief.load_pending(auto, True)  # the capacity check's own word
    _relieve(explicit, checkpoint)                           # the next render's check
    assert pending_load_relief.load_pending(explicit, None)  # its load may still fail
    render_preflight.note_successful_render(None)            # a memo drain places nothing
    assert pending_load_relief.load_pending(explicit, None)
    render_preflight.note_successful_render(UNET)            # a render that succeeded
    assert not pending_load_relief.load_pending(explicit, None)
    pending_load_relief.note_unloaded()
    pending_load_relief.note_placed(UNET)                    # a loader's placement
    assert not pending_load_relief.load_pending(explicit, None)
    assert driver.steps == []  # it fitted: no quote, no unload


def test_a_failed_load_after_a_reset_leaves_the_rescue_for_the_next_queue(monkeypatch, checkpoint):
    """An explicit preset placed eagerly, then a reset (ComfyUI's cached loader
    does not place it again). The first check's quote times out, so nothing is
    freed (fail open), and the Worker then refuses the load for capacity. The
    next queue still reads the load as pending and runs the rescue."""
    from dgx_monarch.nodes import old_model_clear, recycle_drain

    driver = Driver(monkeypatch)
    monkeypatch.setattr(pending_load_relief, "workers_stop", _REAL_WORKERS_STOP)
    handle = Handle(driver)
    handle.setup_key = ("set up",)
    explicit = _model(handle, preset="uly2")
    pending_load_relief.note_placed(UNET)
    recycle_drain.drop_residency_memos()
    answers: list[object] = [TimeoutError("the Workers did not answer in time")]

    def quote(*_args, **_kwargs):
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(old_model_clear, "driver_host_answers", quote)
    _relieve(explicit, checkpoint)
    assert driver.steps == [] and driver.held  # the read failed: nothing freed
    # The Worker refused the load: no render succeeded, nothing was placed.
    answers.append(_answers([("cond", "refuse")]))
    _relieve(explicit, checkpoint)
    assert driver.steps == ["comfy-unload"]


# --- never worse ---------------------------------------------------------------------


def test_a_pending_load_that_fits_pays_no_unload_and_no_worker_call(monkeypatch, checkpoint):
    driver = Driver(monkeypatch, reading_gib=100.1)
    _relieve(_model(Handle(driver)), checkpoint)
    assert driver.steps == [] and driver.held


def test_a_placed_model_never_triggers_however_short_the_box(monkeypatch, checkpoint):
    driver = Driver(monkeypatch, reading_gib=40.0)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", UNET)
    _relieve(_model(Handle(driver)), checkpoint)
    _relieve(_model(Handle(driver), preset="uly2"), checkpoint)
    assert driver.steps == [] and driver.held


LORA_BAKE_GIB = 20.0


def _slab_price(monkeypatch, driver):
    """The slab price on the sparse file: the file and the floor, no host copy,
    and a stack's bake where the stack and a resolver are passed, as
    ``capacity_fit.slab_load_fit`` prices it. Records each call's LoRA inputs."""
    priced: list[tuple] = []

    def slab_fit(path, options, *, reserve_bytes=None, lora_stack=None, resolve_lora_path=None,
                 **_kwargs):
        priced.append((path, lora_stack, resolve_lora_path is not None))
        avail = int(driver.reading)
        size = int(FILE_GIB * _GIB)
        lora = int(LORA_BAKE_GIB * _GIB) if lora_stack and resolve_lora_path else 0
        return SlabFit(size + 5 * _GIB + lora <= avail, True, size, avail, 5 * _GIB, lora_bytes=lora)

    monkeypatch.setattr("dgx_monarch.capacity_fit.slab_load_fit", slab_fit)
    return priced


SLAB_ON = {"slab_weights": True}
STACK = [{"name": "style.safetensors", "strength": 1.0}]


def test_a_slab_load_is_priced_at_the_slab_price(monkeypatch, checkpoint):
    """slab_weights on outright: the file and the floor, no host copy."""
    driver = Driver(monkeypatch, reading_gib=50.0)
    priced = _slab_price(monkeypatch, driver)
    _relieve(_model(Handle(driver), worker_args=SLAB_ON), checkpoint, family="krea2")
    assert priced == [(checkpoint, None, False)] and driver.steps == []


def test_a_vouched_family_under_auto_is_priced_stock_and_asks_where_that_reads_short(
        monkeypatch, checkpoint):
    """Under auto the Worker prices a vouched family stock on the first load of a
    file (a cold family memo this driver cannot read), so the local test takes
    the stock price, the larger one: short at 50 GiB though slab would fit, it
    asks, and the Worker's answer decides."""
    driver = Driver(monkeypatch, reading_gib=50.0, workers_admit=True)
    priced = _slab_price(monkeypatch, driver)
    _relieve(_model(Handle(driver)), checkpoint, family="krea2")
    assert driver.steps == ["quote"] and driver.held
    # Slab is read only as the floor a no-quote answer would need, not as the fit.
    assert priced == [(checkpoint, None, False)]


@pytest.mark.parametrize("low_rss", [None, True])  # lora_low_rss is on unless set off
def test_a_slab_loras_bake_is_priced_as_the_worker_prices_it(monkeypatch, checkpoint, low_rss):
    """The file and floor fit at 50 GiB; the stack's bake under lora_low_rss does
    not, so the load reaches the quote and the rescue."""
    driver = Driver(monkeypatch, reading_gib=50.0)
    priced = _slab_price(monkeypatch, driver)
    worker_args = dict(SLAB_ON, **({} if low_rss is None else {"lora_low_rss": low_rss}))
    request = {"kind": "custom", "model": {"unet_name": UNET, "loras": STACK}}
    _relieve(_model(Handle(driver), worker_args=worker_args), checkpoint, family="krea2",
             request=request)
    assert driver.steps == ["quote", "comfy-unload"]
    assert priced[0] == (checkpoint, STACK, True)


def test_slab_asked_for_with_lora_low_rss_off_is_priced_stock(monkeypatch, checkpoint):
    """The Worker turns slab off without lora_low_rss (actor/comfy_bridge.py), so
    the load is priced stock, not at the slab price, and short at 90 GiB less
    the stack's classic bake, which only the Worker prices: it is asked."""
    driver = Driver(monkeypatch, reading_gib=100.1, workers_admit=True)
    priced = _slab_price(monkeypatch, driver)
    request = {"kind": "custom", "model": {"unet_name": UNET, "loras": STACK}}
    _relieve(_model(Handle(driver), worker_args=dict(SLAB_ON, lora_low_rss=False)), checkpoint,
             family="krea2", request=request)
    assert priced == [] and driver.steps == ["quote"] and driver.held


def test_a_lora_this_box_cannot_find_is_left_to_the_bakes_own_fallback(monkeypatch):
    """The resolver raises for a LoRA ComfyUI cannot find, as the Worker's does;
    ``capacity_lora_bake.lora_bake_bytes`` then prices the whole checkpoint,
    which only raises the local price, so the load is asked, never read as a fit."""
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda kind, name: (  # type: ignore[attr-defined]
        f"/loras/{name}" if (kind, name) == ("loras", "found.safetensors") else None)
    monkeypatch.setitem(__import__("sys").modules, "folder_paths", fake)
    assert pending_load_relief._resolve_lora("found.safetensors") == "/loras/found.safetensors"
    with pytest.raises(FileNotFoundError):
        pending_load_relief._resolve_lora("missing.safetensors")


def test_a_stock_loads_classic_bake_asks_the_workers_however_the_local_price_reads(
        monkeypatch, checkpoint):
    """With lora_low_rss off the Worker prices a stock load's LoRA stack (its
    classic bake) from its own store, which this driver cannot read: a load that
    fits locally is asked, and the Worker's answer decides. With no stack the
    same load asks nothing."""
    driver = Driver(monkeypatch, reading_gib=100.1)
    # The Worker's stock price with the classic bake: 84.2 GiB and about 6 more.
    monkeypatch.setattr(pending_load_relief, "workers_stop", lambda *_args, **_kwargs: (
        driver.steps.append("quote"), driver.reading < 90.0 * _GIB)[1])
    classic = {"lora_low_rss": False}
    request = {"kind": "custom", "model": {"unet_name": UNET, "loras": STACK}}
    _relieve(_model(Handle(driver), worker_args=classic), checkpoint, request=request)
    assert driver.steps == ["quote"] and driver.held
    driver.reading = 80.0 * _GIB  # the bake does not fit: the rescue runs
    _relieve(_model(Handle(driver), worker_args=classic), checkpoint, request=request)
    assert driver.steps == ["quote", "quote", "comfy-unload", "quote"]
    driver.steps.clear()
    _relieve(_model(Handle(driver), worker_args=classic), checkpoint)  # no stack: as before
    assert driver.steps == []


def test_a_stock_loads_lora_stack_under_lora_low_rss_prices_as_the_file(monkeypatch, checkpoint):
    """Under lora_low_rss, the UMA default, the Worker's stock price takes no
    LoRA term (in_place_fallback.route returns None), so a load that fits
    locally asks nothing, stack or no stack."""
    driver = Driver(monkeypatch, reading_gib=100.1)
    request = {"kind": "custom", "model": {"unet_name": UNET, "loras": STACK}}
    _relieve(_model(Handle(driver)), checkpoint, request=request)
    assert driver.steps == [] and driver.held


def test_a_loader_cast_is_priced_at_the_copy_it_builds(monkeypatch, checkpoint):
    """The options go through the Worker's own mapping, so stock_load_fit sees
    the cast's dtype; an option the mapping cannot read is asked."""
    import torch as torch_mod

    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import store_detect

    driver = Driver(monkeypatch, reading_gib=100.1, workers_admit=True)
    seen: list[dict] = []

    stock = capacity_fit.stock_load_fit

    def recording(path, options):
        seen.append(dict(options))
        return stock(path, options)

    monkeypatch.setattr(capacity_fit, "stock_load_fit", recording)
    model = _model(Handle(driver))
    model.options = {"weight_dtype": "fp8_e4m3fn"}
    _relieve(model, checkpoint)
    assert seen and seen[0].get("dtype") is torch_mod.float8_e4m3fn

    def unreadable(_options):
        raise ValueError("an option this mapping does not know")

    monkeypatch.setattr(store_detect, "_to_comfy_model_options", unreadable)
    _relieve(model, checkpoint)
    assert driver.steps == ["quote"] and driver.held


def test_a_granted_rescue_consent_is_priced_at_the_slab_load_it_grants(monkeypatch, checkpoint):
    """The loader projects a live capacity-rescue consent as ``slab_weights=True``
    (nodes/consent_projection.py), so the load is priced at slab and fits."""
    driver = Driver(monkeypatch)
    priced = _slab_price(monkeypatch, driver)
    _relieve(_model(Handle(driver), worker_args={"slab_weights": True}), checkpoint)
    assert priced == [(checkpoint, None, False)] and driver.steps == [] and driver.held


@pytest.mark.parametrize("family", ["minimax_h3", "krea2"])
def test_a_model_the_workers_already_hold_pays_the_quote_alone(monkeypatch, checkpoint, family, caplog):
    """A model the Workers hold that no memo names (a render that loaded it and
    then failed, say), stock or slab: MemAvailable already carries it, so the
    local price reads short, and the Worker answers the reuse as fitting."""
    driver = Driver(monkeypatch, reading_gib=40.0, workers_admit=True)
    _slab_price(monkeypatch, driver)
    with caplog.at_level("INFO"):
        _relieve(_model(Handle(driver)), checkpoint, family=family)
    assert driver.steps == ["quote"] and driver.held
    assert ("but no Worker on this box would stop it, so the driver frees nothing more"
            in caplog.text)


def test_a_requeue_after_an_interrupted_render_frees_nothing_on_a_reuse(monkeypatch, checkpoint):
    """A re-queue after an interrupted or failed render: the rendered memo is
    set only on success, so the load reads as still to come and the local
    price reads short with the model resident. The Worker answers the reuse of
    the model it kept as fitting, so the driver frees nothing and the re-queue
    pays the one quote."""
    driver = Driver(monkeypatch, reading_gib=40.0, workers_admit=True)
    monkeypatch.setattr(render_preflight, "_LAST_DRIVER_CHARGED_UNET", UNET)
    _relieve(_model(Handle(driver)), checkpoint)
    assert driver.steps == ["quote"] and driver.held


def test_models_taking_turns_pay_the_quote_alone(monkeypatch, checkpoint):
    """Two models rendered in turn under auto: each load is a swap the Worker's
    own credit covers, so neither unloads anything."""
    driver = Driver(monkeypatch, workers_admit=True)
    other = "other_bf16.safetensors"
    for name in (UNET, other, UNET):
        monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", other if name == UNET else UNET)
        _relieve(_model(Handle(driver), unet=name), checkpoint)
    assert driver.steps == ["quote"] * 3 and driver.held


# --- the Workers' quote --------------------------------------------------------------


def _answers(*rows):
    from dgx_monarch.nodes.old_model_clear import HostAnswers

    return HostAnswers(tuple(types.SimpleNamespace(quotes=tuple(
        types.SimpleNamespace(slot=slot, verdict=verdict) for slot, verdict in answer))
        for answer in rows), None, 1, True)


@pytest.mark.parametrize(("rows", "stops"), [
    ([[("cond", "refuse")]], True),
    ([[("cond", "rescue")]], True),             # the slab card is a stop for the stock load
    ([[("cond", "fits")], [("cond", "refuse")]], True),
    ([[("cond", "fits")]], False),              # a reuse, or a swap the credit covers
    ([[("cond", "no_claim")]], False),
    ([[("uncond", "refuse"), ("cond", "fits")]], False),
    (None, False),                              # a fleet that cannot be asked
])
def test_what_the_workers_quote_counts_as_a_stop(monkeypatch, rows, stops):
    from dgx_monarch.nodes import old_model_clear

    asked: list[tuple] = []

    def answers(mesh, unet_name, options, slot, loras=None):
        asked.append((unet_name, options, slot, loras))
        return None if rows is None else _answers(*rows)

    monkeypatch.setattr(old_model_clear, "driver_host_answers", answers)
    request = {"model": {"unet_name": UNET, "options": {"weight_dtype": "default"},
                         "loras": [["a.safetensors", 1.0]]}}
    mesh = types.SimpleNamespace(handle=types.SimpleNamespace(setup_key=("set up",)))
    assert pending_load_relief.workers_stop(mesh, request, UNET, {}) is stops
    # The request's own spec, LoRA stack included, as the agreement asks it.
    assert asked == [(UNET, {"weight_dtype": "default"}, "cond", [["a.safetensors", 1.0]])]


class Fleet:
    """A live fleet for the real ``workers_stop`` and ``driver_host_answers``:
    a pair (rank 0 on this box), or a local mesh. ``status`` is each rank's
    store as the Worker reports it."""

    def __init__(self, *, local=False, world=2, set_up=True, status=None):
        self.owns_hosts = not local
        self.world = world
        self.gpus_per_host = 1
        hosts = () if local else tuple(types.SimpleNamespace(name=f"box-{n}.invalid") for n in range(world))
        self.config = types.SimpleNamespace(hosts=hosts, worker_args={})
        self.setup_key = ("set up",) if set_up else None
        self.workers = object()
        self.defunct = False
        self.teardown_complete = False
        self.status = status or []
        self.calls: list[str] = []

    def call_all(self, method, *_args, **_kwargs):
        self.calls.append(method)
        return list(self.status) if method == "status" else []


def _rank(rank, verdict):
    from dgx_monarch.capacity_agreement import RankAnswer

    return RankAnswer(rank, quotes=(types.SimpleNamespace(slot="cond", verdict=verdict),))


@pytest.mark.parametrize(("local", "world", "verdicts", "stops"), [
    (False, 2, ["refuse", "fits"], True),    # the head's own price stops the load
    (False, 2, ["fits", "refuse"], False),   # only the sibling: the driver cannot help it
    (True, 2, ["fits", "rescue"], True),     # a local mesh: every rank is on this box
    (True, 1, ["rescue"], True),             # one box
    (True, 1, ["fits"], False),
])
def test_the_quote_reads_only_the_ranks_on_this_box(monkeypatch, local, world, verdicts, stops):
    from dgx_monarch import capacity_agreement, driver_footprint

    fleet = Fleet(local=local, world=world)
    monkeypatch.setattr(driver_footprint, "rank0_co_resident", lambda _mesh: True)
    monkeypatch.setattr(capacity_agreement, "collect", lambda handle, request, world, **_kw: [
        _rank(rank, verdict) for rank, verdict in enumerate(verdicts)])
    mesh = types.SimpleNamespace(handle=fleet)
    assert pending_load_relief.workers_stop(mesh, {"kind": "custom"}, UNET, {}) is stops


def _store(cond=None, retained=0):
    return {"cond": cond, "uncond": None, "retained_failed_load_slabs": retained}


@pytest.mark.parametrize(("local", "status", "stops"), [
    (False, [{"rank": 0, "models": _store()}, {"rank": 1, "models": _store({"family": "x"})}], True),
    (False, [{"rank": 0, "models": _store({"family": "minimax_h3"})}], False),  # a warm Worker
    (False, [{"rank": 0, "models": _store(retained=1)}], False),
    (True, [{"rank": 0, "models": _store()}, {"rank": 1, "models": _store({"family": "x"})}], False),
    (False, [{"rank": 0}], False),                                              # unreadable
    (False, [], False),
])
def test_before_setup_a_status_read_stands_in_for_the_quote(local, status, stops):
    """``auto`` sets the fleet up at the first render, and before that the quote
    cannot be asked. A rank that holds no model prices as the local price does."""
    fleet = Fleet(local=local, set_up=False, status=status)
    mesh = types.SimpleNamespace(handle=fleet)
    assert pending_load_relief.workers_stop(mesh, {"kind": "custom"}, UNET, {}) is stops
    assert fleet.calls == ["status"]


def test_a_stopped_fleet_with_none_in_its_place_holds_nothing(monkeypatch):
    """After a reset, the next fleet attaches with nothing loaded; an unconfirmed
    stop may still hold its model."""
    monkeypatch.setattr("dgx_monarch.mesh._MESHES", {})  # no fleet attached in its place
    stopped = Fleet()
    stopped.teardown_complete = True
    assert pending_load_relief.workers_stop(
        types.SimpleNamespace(handle=stopped), {"kind": "custom"}, UNET, {}) is True
    unconfirmed = Fleet()
    unconfirmed.defunct = True
    assert pending_load_relief.workers_stop(
        types.SimpleNamespace(handle=unconfirmed), {"kind": "custom"}, UNET, {}) is False
    assert stopped.calls == unconfirmed.calls == []


# --- where no quote can be asked -----------------------------------------------------


def _no_quote(monkeypatch, driver, where):
    """A handle with no quote to ask, read by the real ``workers_stop``: before
    setup, whose status read shows every rank holding nothing, or a stopped
    fleet with none attached in its place."""
    monkeypatch.setattr(pending_load_relief, "workers_stop", _REAL_WORKERS_STOP)
    handle = Handle(driver)
    empty = [{"rank": rank, "models": _store()} for rank in range(2)]

    def call_all(method, *_args, **_kwargs):
        driver.steps.append(method)
        return empty if method == "status" else []

    handle.call_all = call_all  # type: ignore[method-assign]
    if where == "stopped":
        monkeypatch.setattr("dgx_monarch.mesh._MESHES", {})
        handle.teardown_complete = True
    return handle


@pytest.mark.parametrize("where", ["before setup", "stopped"])
@pytest.mark.parametrize("unread", ["classic bake", "unmappable option"])
def test_with_no_quote_an_unread_price_frees_nothing_on_a_box_with_room(
        monkeypatch, checkpoint, where, unread):
    """With no quote, an empty rank prices as the local price only where that
    price is complete. A price it could not read (a stock LoRA stack with
    lora_low_rss off, or an option it cannot map) on a box with room must not
    be read as short: nothing is read and nothing is freed."""
    from dgx_monarch.actor import store_detect

    driver = Driver(monkeypatch, reading_gib=110.0)
    handle = _no_quote(monkeypatch, driver, where)
    request = {"kind": "custom"}
    worker_args: dict = {}
    if unread == "classic bake":
        request["model"] = {"unet_name": UNET, "loras": STACK}
        worker_args["lora_low_rss"] = False
    else:
        monkeypatch.setattr(store_detect, "_to_comfy_model_options",
                            lambda _options: (_ for _ in ()).throw(ValueError("unknown")))
    _relieve(_model(handle, worker_args=worker_args), checkpoint, request=request)
    assert driver.steps == [] and driver.held


# The rows of ``no_quote_floor``'s table: (worker args, family, floor). With
# the sparse file the stock price is 84.2 GiB and the slab price 42.7.
FLOOR_ROWS = {
    "lora_low_rss off": ({"lora_low_rss": False, "slab_weights": True}, "krea2", "stock"),
    "slab_weights off": ({"slab_weights": False}, "krea2", "stock"),
    "slab on, compile off": ({"slab_weights": True}, "minimax_h3", "slab"),
    "slab on, compile on": ({"slab_weights": True, "compile_dit": True}, "flux2", "slab"),
    "auto, vouched family": ({}, "krea2", "slab"),
    "auto, other known family": ({"slab_weights": "auto"}, "minimax_h3", "stock"),
    "unknown spelling": ({"slab_weights": "sometimes"}, "krea2", None),
    "auto, unknown family": ({}, "unknown", None),
}
UNLOADS = ["status", "comfy-unload"]


@pytest.mark.parametrize("row", list(FLOOR_ROWS))
def test_the_no_quote_floor_is_the_smallest_price_a_fresh_worker_could_choose(row):
    worker_args, family, floor = FLOOR_ROWS[row]
    assert pending_load_relief.no_quote_floor(worker_args, family) == floor


@pytest.mark.parametrize(("row", "reading", "steps"), [
    # A stock floor: short at 80, so an empty rank stops the load.
    ("lora_low_rss off", 80.0, UNLOADS),
    ("slab_weights off", 80.0, UNLOADS),
    ("auto, other known family", 80.0, UNLOADS),
    # A slab floor: it fits at 80 (stock short), so nothing is read or freed;
    # short at 40 too, and an empty rank stops it.
    ("slab on, compile off", 80.0, []),
    ("slab on, compile off", 40.0, UNLOADS),
    ("slab on, compile on", 80.0, []),
    ("slab on, compile on", 40.0, UNLOADS),
    ("auto, vouched family", 80.0, []),
    ("auto, vouched family", 40.0, UNLOADS),
    # No floor: nothing is read or freed however short the box reads.
    ("unknown spelling", 40.0, []),
    ("auto, unknown family", 40.0, []),
])
def test_with_no_quote_each_floor_row_frees_only_where_its_price_is_short(
        monkeypatch, checkpoint, row, reading, steps):
    """Before setup, every rank empty, read by the real ``workers_stop``."""
    worker_args, family, _floor = FLOOR_ROWS[row]
    driver = Driver(monkeypatch, reading_gib=reading)
    _slab_price(monkeypatch, driver)
    handle = _no_quote(monkeypatch, driver, "before setup")
    _relieve(_model(handle, worker_args=worker_args), checkpoint, family=family)
    assert driver.steps == steps


def test_explicit_slab_with_compile_on_a_noop_family_before_setup(monkeypatch, checkpoint):
    """flux2 with slab asked for and compile_dit on is priced stock here, but a
    fresh Worker keeps slab for a family compile leaves eager, reading its
    persisted memo, and admits: slab fits at 50 GiB, so nothing is freed."""
    driver = Driver(monkeypatch, reading_gib=50.0)
    _slab_price(monkeypatch, driver)
    handle = _no_quote(monkeypatch, driver, "before setup")
    _relieve(_model(handle, worker_args={"slab_weights": True, "compile_dit": True}),
             checkpoint, family="flux2")
    assert driver.steps == [] and driver.held


# --- the rescue ----------------------------------------------------------------------


def test_a_short_pending_load_fits_after_the_driver_unloads(monkeypatch, checkpoint, caplog):
    driver = Driver(monkeypatch)
    with caplog.at_level("INFO"):
        _relieve(_model(Handle(driver)), checkpoint)
    assert driver.steps == ["quote", "comfy-unload"]
    assert ("driver unload: the pending load of p-h3-image_bf16.safetensors fits after "
            "unloading the 2 models (Qwen35TEModel_, MiniMaxH3TEModel_)") in caplog.text


def test_the_order_is_unload_then_trim(monkeypatch, checkpoint, caplog):
    driver = Driver(monkeypatch, freed_gib=9.0)  # 83.0 after the unload, 85.0 after the trim
    with caplog.at_level("INFO"):
        _relieve(_model(Handle(driver)), checkpoint)
    assert driver.steps == ["quote", "comfy-unload", "quote", "actor-trim"]
    assert "actor trim: the pending load of p-h3-image_bf16.safetensors fits" in caplog.text


def test_a_rescue_that_falls_short_refuses_nothing(monkeypatch, checkpoint, caplog):
    driver = Driver(monkeypatch, freed_gib=1.0)
    with caplog.at_level("INFO"):
        _relieve(_model(Handle(driver)), checkpoint)  # no raise
    assert driver.steps == ["quote", "comfy-unload", "quote", "actor-trim", "quote"]
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("pending load:")]
    assert line and "the Workers' load agreement decides it" in line[0]
    assert "Before refusing, the driver unloaded the 2 models" in line[0]


def test_an_explicit_presets_deferred_load_is_rescued(monkeypatch, checkpoint):
    driver = Driver(monkeypatch)
    pending_load_relief.note_deferred(UNET)
    _relieve(_model(Handle(driver), preset="uly2"), checkpoint)
    assert driver.steps == ["quote", "comfy-unload"]


# --- boxes ---------------------------------------------------------------------------


def test_a_one_box_fleet_is_covered(monkeypatch, checkpoint):
    """World 1 has no fleet agreement; the render site covers it as it covers a pair."""
    driver = Driver(monkeypatch)
    _relieve(_model(Handle(driver, world=1)), checkpoint)
    assert driver.steps == ["quote", "comfy-unload"]


def test_a_driver_on_a_box_with_no_worker_frees_nothing(monkeypatch, checkpoint):
    driver = Driver(monkeypatch)
    handle = Handle(driver, owns_hosts=True)
    handle.config = types.SimpleNamespace(hosts=(types.SimpleNamespace(name="sparks-far-away.invalid"),),
                                          worker_args={})
    _relieve(_model(handle), checkpoint)
    assert driver.steps == [] and driver.held


# --- left alone ----------------------------------------------------------------------


@pytest.mark.parametrize("case", ["dual model", "comfy-managed", "fsdp preset"])
def test_cases_priced_elsewhere_are_left_alone(monkeypatch, checkpoint, case):
    driver = Driver(monkeypatch)
    request = {"kind": "custom"}
    model = _model(Handle(driver))
    if case == "dual model":
        request["uncond_model"] = {"unet_name": "u.safetensors"}
    elif case == "comfy-managed":
        model.mesh.worker_args = {"comfy_managed": True}
    else:
        model.mesh.topology_preset = "uly2+fsdp"
        pending_load_relief.note_deferred(UNET)
    _relieve(model, checkpoint, request=request)
    assert driver.steps == [] and driver.held


@pytest.fixture
def renders():
    started: list[object] = []

    def start(count: int) -> None:
        for _ in range(count):
            token = object()
            render_progress.start(1, {}, token=token)
            started.append(token)

    yield start
    for token in started:
        render_progress.finish(token=token)


def test_a_busy_driver_keeps_its_models_and_asks_nothing(monkeypatch, checkpoint, renders):
    """The rescue keeps every model while a render runs, so no quote is worth asking."""
    driver = Driver(monkeypatch)
    renders(1)
    _relieve(_model(Handle(driver)), checkpoint)
    assert driver.steps == [] and driver.held


@pytest.mark.parametrize("read", ["quote", "status"])
def test_a_read_that_fails_or_times_out_frees_nothing(monkeypatch, checkpoint, caplog, read):
    """Fail open: a quote or a status read that raises answers no, so the
    driver keeps its models and the render goes on; the read is logged at
    DEBUG with its milliseconds."""
    from dgx_monarch.nodes import old_model_clear

    driver = Driver(monkeypatch)
    monkeypatch.setattr(pending_load_relief, "workers_stop", _REAL_WORKERS_STOP)

    def silent(*_args, **_kwargs):
        raise TimeoutError("the Workers did not answer in time")

    handle = Handle(driver)
    if read == "quote":
        handle.setup_key = ("set up",)
        monkeypatch.setattr(old_model_clear, "driver_host_answers", silent)
    else:
        handle.call_all = silent  # type: ignore[method-assign]
    with caplog.at_level("DEBUG", logger=pending_load_relief.log.name):
        _relieve(_model(handle), checkpoint)  # no raise
    assert driver.steps == [] and driver.held
    line = [r.getMessage() for r in caplog.records if " ms and would not stop" in r.getMessage()]
    assert line and "took" in line[0]


def test_the_off_switch_asks_nothing(monkeypatch, checkpoint):
    """DGXM_DISABLE_DRIVER_UNLOAD=1 turns the step off, quote included."""
    driver = Driver(monkeypatch)
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")
    _relieve(_model(Handle(driver)), checkpoint)
    assert driver.steps == [] and driver.held


def test_a_fault_leaves_the_render_to_the_agreement(monkeypatch, checkpoint):
    driver = Driver(monkeypatch)

    def broken(*_args, **_kwargs):
        raise OSError("the header could not be read")

    monkeypatch.setattr("dgx_monarch.capacity_fit.stock_load_fit", broken)
    _relieve(_model(Handle(driver)), checkpoint)
    assert driver.steps == []


def _render_site(monkeypatch, checkpoint):
    """The render site's own resolution, faked down to the checkpoint on disk."""
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda _kind, _name: checkpoint  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "folder_paths", fake)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint", lambda _path: ("minimax_h3", "bf16"))
    monkeypatch.setattr(render_preflight.family_select, "warn_auto_admits_checkpoint",
                        lambda *_args: None)
    monkeypatch.setattr(render_preflight.latent_scale, "rendered_latent_shape", lambda *_args: None)


def test_the_render_site_runs_the_relief_once_its_checks_pass(monkeypatch, checkpoint):
    order: list[str] = []
    monkeypatch.setattr(pending_load_relief, "relieve", lambda *a, **k: order.append("relief"))

    def checks(steps, **_kwargs):
        order.append("checks")

    monkeypatch.setattr(driver_unload, "check_with_unload", checks)
    _render_site(monkeypatch, checkpoint)
    render_preflight.activation_footprint_preflight_for_request(
        _model(None), {"kind": "custom"}, {"samples": torch.zeros(1, 16, 1, 8, 8)})
    assert order == ["checks", "relief"]


def test_a_render_the_checks_refuse_never_reaches_the_relief_and_keeps_its_words(
        monkeypatch, checkpoint):
    """The relief comes after the checks that refuse, so their own rescue finds
    ComfyUI's models still held and its refusal says it unloaded them, as today."""
    from dgx_monarch import driver_footprint, h3_activation, render_memory_price

    driver = Driver(monkeypatch, trimmable=False)
    reached: list[str] = []
    monkeypatch.setattr(pending_load_relief, "relieve", lambda *a, **k: reached.append("relief"))
    monkeypatch.setattr(render_memory_price, "preflight_render_memory", lambda *_args: None)
    monkeypatch.setattr(h3_activation, "preflight_h3_activation_for_request", lambda *a, **k: None)

    def footprint(*_args):
        raise driver_footprint.DriverFootprintCapacityError("the driver host is short.")

    monkeypatch.setattr(render_preflight, "_driver_footprint_preflight_for_request", footprint)
    _render_site(monkeypatch, checkpoint)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as refused:
        render_preflight.activation_footprint_preflight_for_request(
            _model(Handle(driver)), {"kind": "custom"}, {"samples": torch.zeros(1, 16, 1, 8, 8)})
    assert reached == [] and driver.steps == ["comfy-unload"]
    assert ("Before refusing, the driver unloaded the 2 models (Qwen35TEModel_, "
            "MiniMaxH3TEModel_) ComfyUI held on this host") in str(refused.value)


# --- end to end on the first-use rig -----------------------------------------------


def _first_use(monkeypatch, tmp_path, checkpoint, *, widgets=None, lora=False, world=2):
    """The first-use rig on the driver's own box, with the render site's relief live."""
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes import gate_nothing_to_gate
    from dgx_monarch.topology import Topology
    from first_use_gate_helpers import (
        TEMPLATES,
        doc,
        graph_model,
        mesh_rig,
        no_material_rows,
        swapped_rows,
    )

    latent = torch.zeros(1, 4, 8, 8)
    # A LoRA stack gives the ceremony something to compare, so it runs its legs.
    handle, _topology, dispatched = mesh_rig(
        monkeypatch, tmp_path, swapped_rows() if lora else no_material_rows(),
        [latent.clone() for _ in range(4)],
        topology=None if world == 2 else Topology(world=world, dp=world))
    handle.world = world
    monkeypatch.delenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, raising=False)
    driver = Driver(monkeypatch)
    handle.owns_hosts = False  # rank 0 on the driver's box
    inner = handle.call_all

    def call_all(method, *args, **kwargs):
        driver.steps.append(method)
        return inner(method, *args, **kwargs)

    handle.call_all = call_all
    # The rig stands the render-site checks down; this keeps the relief alone live.
    monkeypatch.setattr(
        render_preflight, "activation_footprint_preflight_for_request",
        lambda model, request, latent, *, weights_loaded=None: pending_load_relief.relieve(
            model, request, family="minimax_h3", path=checkpoint, weights_loaded=weights_loaded))
    monkeypatch.setattr(gate_nothing_to_gate, "_family", lambda _model: ("minimax_h3", False))
    monkeypatch.setattr(gate_nothing_to_gate, "_rescue_consent_live", lambda _model: False)
    # The rig's ceremony legs skip common.run_render; each passes the render
    # site first here, as a real leg does, and is recorded as "render".
    fake_render = gate_mod.run_render

    def render(render_model, request, latent_in, **kwargs):
        render_preflight.activation_footprint_preflight_for_request(render_model, request, latent_in)
        driver.steps.append("render")
        return fake_render(render_model, request, latent_in, **kwargs)

    monkeypatch.setattr(gate_mod, "run_render", render)
    graph = doc(TEMPLATES / "dgx-monarch-p-h3-image.json")
    model = graph_model(graph, widgets=widgets)
    if lora:
        model = model.with_lora("p-h3-image_lora_bf16.safetensors", 1.0)
    return handle, dispatched, driver, graph, model


def test_the_no_render_path_loads_after_the_render_site_frees_the_driver(
        monkeypatch, tmp_path, checkpoint):
    from first_use_gate_helpers import queue

    _handle, dispatched, driver, graph, model = _first_use(monkeypatch, tmp_path, checkpoint)
    queue(graph, model)
    assert driver.steps.count("comfy-unload") == 1
    assert driver.steps.index("comfy-unload") < driver.steps.index("load_model")
    assert "render" not in driver.steps and len(dispatched) == 1


def test_a_warm_requeue_frees_nothing(monkeypatch, tmp_path, checkpoint):
    from first_use_gate_helpers import queue

    _handle, dispatched, driver, graph, model = _first_use(monkeypatch, tmp_path, checkpoint)
    queue(graph, model)
    driver.held = ["Qwen35TEModel_", "MiniMaxH3TEModel_"]  # the next encode staged them again
    driver.reading = 74.0 * _GIB
    queue(graph, model)
    assert driver.steps.count("comfy-unload") == 1 and len(dispatched) == 2


def test_a_lora_ceremony_frees_the_driver_once_before_its_first_proof_render(
        monkeypatch, tmp_path, checkpoint):
    from first_use_gate_helpers import queue

    _handle, _dispatched, driver, graph, model = _first_use(
        monkeypatch, tmp_path, checkpoint, lora=True)
    queue(graph, model)
    assert driver.steps.count("comfy-unload") == 1
    assert driver.steps.index("comfy-unload") < driver.steps.index("render")


def test_a_lora_ceremony_on_a_warm_model_frees_the_driver_before_its_clean_reload(
        monkeypatch, tmp_path, checkpoint):
    """A model an earlier render left warm passes the user's render site as
    placed. The ceremony then unloads every rank and reloads it from disk for
    its first proof render, whose own render site frees the driver first
    (gate_ceremony.py marks the unload)."""
    from first_use_gate_helpers import queue

    _handle, _dispatched, driver, graph, model = _first_use(
        monkeypatch, tmp_path, checkpoint, lora=True)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", UNET)
    queue(graph, model)
    steps = driver.steps
    assert steps.count("comfy-unload") == 1
    assert steps.index("unload") < steps.index("comfy-unload") < steps.index("render")


def test_the_no_render_path_on_a_warm_model_frees_the_driver_before_its_load(
        monkeypatch, tmp_path, checkpoint):
    """The same on the no-render path: it unloads every rank and loads with no
    render, so it asks for the driver's room itself, before the agreement."""
    from dgx_monarch.nodes import gate_nothing_to_gate
    from first_use_gate_helpers import queue

    _handle, dispatched, driver, graph, model = _first_use(monkeypatch, tmp_path, checkpoint)
    monkeypatch.setattr(gate_nothing_to_gate, "_checkpoint_path", lambda _model: checkpoint)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", UNET)
    queue(graph, model)
    steps = driver.steps
    assert steps.count("comfy-unload") == 1
    assert steps.index("unload") < steps.index("comfy-unload") < steps.index("load_model")
    assert "render" not in steps and len(dispatched) == 1


def _relief_steps(driver):
    """The steps this file asserts on, without the rig's other Worker calls."""
    return [step for step in driver.steps
            if step in ("quote", "comfy-unload", "actor-trim", "worker-load")]


def _one_spark(monkeypatch, tmp_path, checkpoint, reading_gib):
    """One Spark under ``auto`` with no first-use check: the model's first load
    comes at the render, where the Worker's own load check is the judge (there
    is no agreement at world 1). The Worker here prices as the local price
    does, and its load refuses below that price, as its ladder would."""
    from dgx_monarch import capacity_fit
    from dgx_monarch import mesh_safety as safety
    from dgx_monarch.nodes import common

    _handle, dispatched, driver, graph, model = _first_use(
        monkeypatch, tmp_path, checkpoint, widgets={"auto_gate": "off"}, world=1)
    driver.reading = reading_gib * _GIB
    monkeypatch.setattr(pending_load_relief, "workers_stop", lambda *_args, **_kwargs: (
        driver.steps.append("quote"),
        not capacity_fit.stock_load_fit(checkpoint, {}).fits)[1])
    dispatch = common.submit_render

    def worker_load(model, request, latent, *args, **kwargs):
        fit = capacity_fit.stock_load_fit(checkpoint, {})
        if not fit.fits:
            raise safety.StockLoadCapacityError(
                f"stock residency cannot load {UNET} here: needs "
                f"{fit.required_bytes / _GIB:.1f} GiB, {fit.avail_bytes / _GIB:.1f} GiB available")
        driver.steps.append("worker-load")
        return dispatch(model, request, latent, *args, **kwargs)

    monkeypatch.setattr(common, "submit_render", worker_load)
    return dispatched, driver, graph, model


def test_one_spark_frees_the_staged_text_models_and_the_workers_load_fits(
        monkeypatch, tmp_path, checkpoint, caplog):
    from first_use_gate_helpers import queue

    dispatched, driver, graph, model = _one_spark(monkeypatch, tmp_path, checkpoint, 74.0)
    with caplog.at_level("INFO"):
        queue(graph, model)
    assert _relief_steps(driver) == ["quote", "comfy-unload", "worker-load"]
    assert len(dispatched) == 1 and not driver.held
    assert "driver unload: the pending load of p-h3-image_bf16.safetensors fits" in caplog.text


def test_one_spark_that_fits_frees_nothing(monkeypatch, tmp_path, checkpoint):
    from first_use_gate_helpers import queue

    dispatched, driver, graph, model = _one_spark(monkeypatch, tmp_path, checkpoint, 100.1)
    queue(graph, model)
    assert _relief_steps(driver) == ["worker-load"] and driver.held and len(dispatched) == 1


def test_one_spark_without_the_render_sites_step_meets_the_workers_refusal(
        monkeypatch, tmp_path, checkpoint):
    """The control for the first test: with the step off, the same staged text
    models leave the Worker's own load short, as before 1.5."""
    from first_use_gate_helpers import queue

    dispatched, driver, graph, model = _one_spark(monkeypatch, tmp_path, checkpoint, 74.0)
    monkeypatch.setattr(pending_load_relief, "relieve", lambda *_args, **_kwargs: None)
    with pytest.raises(Exception, match="stock residency cannot load"):
        queue(graph, model)
    assert "comfy-unload" not in driver.steps and not dispatched


def test_an_explicit_presets_eager_placement_frees_nothing(monkeypatch, tmp_path, checkpoint):
    """The loader placed the model at its node and no gate reloads it: nothing is
    still to come at the render site, however short the box reads."""
    from first_use_gate_helpers import queue

    _handle, _dispatched, driver, graph, model = _first_use(
        monkeypatch, tmp_path, checkpoint, widgets={"topology": "uly2", "auto_gate": "off"})
    pending_load_relief.note_placed(UNET)
    driver.reading = 40.0 * _GIB  # the placed DiT is inside this reading
    try:
        queue(graph, model)
    except Exception:
        pass
    assert "comfy-unload" not in driver.steps


def test_an_explicit_presets_deferred_placement_is_freed_before_the_gate(
        monkeypatch, tmp_path, checkpoint):
    from first_use_gate_helpers import queue

    _handle, dispatched, driver, graph, model = _first_use(
        monkeypatch, tmp_path, checkpoint, widgets={"topology": "uly2"})
    pending_load_relief.note_deferred(UNET)
    queue(graph, model)
    assert driver.steps.count("comfy-unload") == 1
    assert driver.steps.index("comfy-unload") < driver.steps.index("load_model")
    assert len(dispatched) == 1


def test_an_explicit_preset_requeued_after_the_no_render_paths_load_asks_nothing(
        monkeypatch, tmp_path, checkpoint):
    """Under an explicit preset the loader deferred the placement, and the
    no-render path then loaded the model. A re-queue after a render that failed
    before its rendered memo was set must read the model as placed (the
    no-render path marks it), not as a deferral still to come."""
    from first_use_gate_helpers import queue

    _handle, _dispatched, driver, graph, model = _first_use(
        monkeypatch, tmp_path, checkpoint, widgets={"topology": "uly2"})
    pending_load_relief.note_deferred(UNET)
    queue(graph, model)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", None)  # the render failed
    monkeypatch.setattr(render_preflight, "_LAST_SUBMITTED_UNET", None)
    assert not pending_load_relief.load_pending(model, None)
    before = list(driver.steps)
    driver.held = ["Qwen35TEModel_", "MiniMaxH3TEModel_"]
    driver.reading = 74.0 * _GIB
    _relieve(model, checkpoint)
    assert driver.steps == before and driver.held
