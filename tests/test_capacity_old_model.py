"""The "Old model in the way" card (old_model.py, nodes/old_model_clear.py).

On 2026-11-04, 19 of 20 template errors were capacity refusals caused by a
model a Worker still held from an earlier template, and no refusal named the
one action that clears it. These tests hold the card's condition at both
sites from synthetic price rows, the Worker facts it reads, the memo flag and
the env var, the consent accept, and the Always flow against a fake mesh.

CPU only: no torch, no CUDA, no ComfyUI beyond a stub folder_paths.
"""
from __future__ import annotations

import ast
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from dgx_monarch import (
    capacity_agreement,
    consent_descriptor,
    consent_pending,
    consent_store,
    mesh_safety,
    old_model,
)
from dgx_monarch.actor import actor_footprint, capacity_quote, worker_capacity
from dgx_monarch.actor.capacity_quote import RankPrice
from dgx_monarch.capacity_agreement import FleetCapacityError, RankAnswer, decide
from dgx_monarch.capacity_fit import StockFit
from dgx_monarch.consent_descriptor import SlabResidencyRescueOffer
from dgx_monarch.nodes import consent_routes, old_model_clear, recycle_drain
from dgx_monarch.refusal import parse_leading_refusal_tag

REPO = Path(__file__).resolve().parents[1]
_GIB = 2 ** 30
NEW = "qwen_image_2512_bf16.safetensors"
OLD = "flux1-dev.safetensors"


def _stock(verdict: str, *, size_gib: float, avail_gib: float, slot: str = "cond",
           resident: str | None = OLD, word: str | None = "stock",
           resident_gib: float = 22.2, arena_gib: float = 0.0,
           descriptor: object = None) -> RankPrice:
    """A stock-priced row as the Worker sends it, with its resident facts."""
    fit = StockFit(False, True, int(size_gib * _GIB), int(avail_gib * _GIB))
    return RankPrice(
        verdict=verdict, rung="stock_fits", priced=True, measured=dict(fit.measured()),
        descriptor=descriptor, would_load="swap" if resident else "fresh",
        checkpoint_bytes=int(size_gib * _GIB), slot=slot, shmem_bytes=0,
        holds_resident={"cond": word if resident else None, "uncond": None},
        resident_names={"cond": resident, "uncond": None},
        resident_bytes={"cond": int(resident_gib * _GIB)} if resident else {},
        arena_bytes=int(arena_gib * _GIB), reason="short")


def _fits(rank: int, box: str) -> RankAnswer:
    """A rank with room. It states no file size, so no byte check compares it."""
    fit = StockFit(True, True, 19 * _GIB, 200 * _GIB)
    row = RankPrice(verdict="fits", rung="stock_fits", priced=True,
                    measured=dict(fit.measured()), would_load="fresh",
                    checkpoint_bytes=0, slot="cond", reason="fits")
    return RankAnswer(rank, f"host{rank}", (row,), configured_host=box)


def _answer(rank: int, *rows: RankPrice, box: str = "") -> RankAnswer:
    return RankAnswer(rank, f"host{rank}", tuple(rows), elapsed_s=0.01,
                      configured_host=box or ("head" if rank == 0 else "sibling"))


# Captured before the rig stubs it, so the real wait is tested once.
_WAIT_SETTLED = old_model_clear.wait_settled


def _shortfall_gib(row: RankPrice) -> float:
    return (row.measured["required_bytes"] - row.measured["mem_available_bytes"]) / _GIB


@pytest.fixture(autouse=True)
def _rig(monkeypatch, tmp_path):
    consent_pending.clear_all()
    actor_footprint.forget()
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "memo.json"))
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    models = tmp_path / "diffusion_models"
    models.mkdir()
    for name in (NEW, OLD):
        (models / name).write_bytes(b"x")
    output = tmp_path / "output"
    output.mkdir()
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda folder, name: (  # type: ignore[attr-defined]
        str(tmp_path / folder / name) if (tmp_path / folder / name).exists() else None)
    fake.get_output_directory = lambda: str(output)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    monkeypatch.setattr(old_model_clear, "wait_settled", lambda *a, **k: None)
    from dgx_monarch.nodes import routes

    # The reset route's single-flight gate is process-wide; each test starts open.
    monkeypatch.setattr(routes, "_RECYCLE_RATE", {"active": False,
                                                 "last_finished": float("-inf")})
    yield
    consent_pending.clear_all()
    actor_footprint.forget()


# The card condition at the cross-rank agreement, from price rows.

def test_a_clear_that_makes_room_raises_the_old_model_refusal():
    """qwen-image-t2i on 2026-11-04: 45.0 GiB against 27.4 with flux1-dev held."""
    short = _stock("refuse", size_gib=19.0, avail_gib=27.4)
    assert 0 < _shortfall_gib(short) < 22.2
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=NEW)
    text = str(raised.value)
    assert ("The old model (flux1-dev) is still loaded on head and qwen_image_2512_bf16 "
            "does not fit beside it.") in text
    assert '"Clear and retry"' in text and '"Old model in the way"' in text
    assert "Reset attached mesh, then queue the render again" in text
    assert "DGXM_ALLOW_CLEAR_RETRY=1" in text
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.guard == "cross_rank_capacity" and not tag.waivable
    assert mesh_safety.is_stock_load_capacity_error(raised.value)
    offer = old_model.offer_of(raised.value)
    assert offer is not None and offer.helps and offer.hosts == 2


def test_a_cast_load_short_beside_an_old_model_still_offers_the_card():
    """A loader dtype cast is priced (2026-11-05), so a cast load an old model
    blocks meets the same card: 13.3 GiB of fp8 cast to a 26.6 GiB bf16 copy
    needs 46.2 GiB against 27.4 with flux1-dev's 22.2 GiB held."""
    fit = StockFit(False, True, int(13.3 * _GIB), int(27.4 * _GIB), cast_bytes=int(26.6 * _GIB))
    short = replace(_stock("refuse", size_gib=13.3, avail_gib=27.4),
                    measured=dict(fit.measured()),
                    blocker=("the loader casts this checkpoint's weight dtype", "free memory"))
    assert 0 < _shortfall_gib(short) < 22.2 - 2
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=NEW)
    text = str(raised.value)
    assert "The old model (flux1-dev) is still loaded on head" in text
    assert "a 13.3 GiB file cast to a 26.6 GiB copy, plus the host copy" in text
    assert '"Clear and retry"' in text


def _credited_swap(avail_gib: float, arena_gib: float) -> RankPrice:
    """qwen-image over flux1-dev in one slot, priced after the store drops flux1-dev."""
    credit = int(22.2 * _GIB)
    fit = StockFit(False, True, int(19.0 * _GIB), int(avail_gib * _GIB), credit_bytes=credit)
    fit = replace(fit, fits=fit.required_bytes <= fit.avail_bytes)
    return replace(_stock("refuse", size_gib=19.0, avail_gib=avail_gib, arena_gib=arena_gib),
                   measured=dict(fit.measured()))


def test_a_credited_swap_weighs_a_clear_against_the_uncredited_price():
    """A swap over the slot's own stock resident is credited that copy (2026-11-05);
    a cleared box holds no copy to drop, so the card compares the whole price."""
    short = _credited_swap(avail_gib=10.0, arena_gib=30.0)
    assert short.measured["required_gib"] == 22.7          # 44.9 less flux1-dev's 22.2
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=NEW)
    text = str(raised.value)
    assert ("priced a stock load of qwen_image_2512_bf16.safetensors after the Worker drops "
            "flux1-dev, the 22.2 GiB copy its slot holds. Beyond that copy the load needs "
            "22.7 GiB") in text
    assert "does not model the drop" not in text
    offer = old_model.offer_of(raised.value)
    assert offer is not None and old_model.card_numbers(offer) == "needs 44.9 GiB, 10.0 GiB available on head"


def test_a_credited_swap_with_no_arena_to_return_says_a_clear_would_not_make_room():
    short = _credited_swap(avail_gib=10.0, arena_gib=0.0)
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=NEW)
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    # 10.0 available plus flux1-dev's 22.2 back, less the 2 GiB margin, against 44.9.
    assert ("A clear would not make room: qwen_image_2512_bf16 needs 44.9 GiB on head, and a "
            "cleared box would have about 30.2 GiB.") in str(raised.value)


def test_a_clear_that_would_not_make_room_keeps_the_ordinary_refusal_in_plain_words():
    """wan-animate2 BF16: 69.1 GiB against 25.8 is short more than flux1-dev holds."""
    short = _stock("refuse", size_gib=30.5, avail_gib=25.8)
    assert _shortfall_gib(short) > 22.2
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=NEW)
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    text = str(raised.value)
    assert "The old model (flux1-dev) is still loaded on head" in text
    assert "A clear would not make room" in text
    # 25.8 available plus 22.2 back, less the 2 GiB a fresh Worker takes back.
    assert "a cleared box would have about 46.0 GiB" in text
    assert "Choose a smaller or quantized artifact" in text
    assert "Clear and retry" not in text


def test_the_arena_an_actor_kept_counts_toward_what_a_clear_returns():
    """The head actor held 45.5 GiB RSS beside a 22.2 GiB flux1-dev at 05:08."""
    short = _stock("refuse", size_gib=30.5, avail_gib=25.8, arena_gib=30.0)
    with pytest.raises(old_model.OldModelRefusal):
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=NEW)


def test_a_slab_resident_is_credited_its_arena_only():
    """Dropping a slab resident barely moves MemAvailable (DESIGN 5.9)."""
    slab = _stock("refuse", size_gib=9.5, avail_gib=20.6, word="slab", resident_gib=9.5)
    assert 0 < _shortfall_gib(slab) < 9.5
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, slab), _fits(1, "sibling")], 2, unet_name=NEW)
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    assert old_model.rank_gain(slab) == 0
    with pytest.raises(old_model.OldModelRefusal):
        decide([_answer(0, replace(slab, arena_bytes=40 * _GIB)), _fits(1, "sibling")], 2)


def test_every_short_box_must_gain_room_and_every_old_model_is_named():
    head = _stock("refuse", size_gib=19.0, avail_gib=27.4)
    sibling = _stock("refuse", size_gib=19.0, avail_gib=2.0,
                     resident="ltx-2.5-22b.safetensors")
    assert _shortfall_gib(sibling) > 22.2
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, head), _answer(1, sibling)], 2, unet_name=NEW)
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    text = str(raised.value)
    assert ("The old model (flux1-dev) is still loaded on head, the old model "
            "(ltx-2.5-22b) on sibling, and qwen_image_2512_bf16 does not fit beside them.") in text
    assert "on sibling, and a cleared box would have about" in text
    with pytest.raises(old_model.OldModelRefusal) as both:
        decide([_answer(0, head), _answer(1, replace(head, resident_names={
            "cond": OLD, "uncond": None}))], 2, unet_name=NEW)
    assert "The old model (flux1-dev) is still loaded on head and sibling, and" in str(both.value)


def test_a_rescue_offer_with_an_old_model_in_the_way_raises_the_clear_card_instead(monkeypatch):
    """Ten run-2 rescue cards were priced over flux1-dev. A clear keeps stock residency."""
    rescue = _stock("rescue", size_gib=13.3, avail_gib=19.7,
                    descriptor={"kind": "rescue-slab"})
    handle = types.SimpleNamespace(world=2, workers=object(), active_worker_args={})
    monkeypatch.setattr(capacity_agreement, "collect", lambda *_a, **_k: [
        _answer(0, rescue), _fits(1, "sibling")])
    with pytest.raises(old_model.OldModelRefusal) as raised:
        capacity_agreement.agree_or_refuse(handle, {}, unet_name=NEW)
    assert not isinstance(raised.value, SlabResidencyRescueOffer)
    cards = consent_pending.pending_cards()
    assert [card["kind"] for card in cards] == ["clear-retry"]
    card = cards[0]
    assert card["title"] == "Old model in the way"
    assert card["primary"] == {"label": "Clear and retry", "action": "accept"}
    assert card["always"] == "Always clear and retry" and card["runs"] == "recycle"
    assert card["detail"] == (
        "The old model (flux1-dev) is still loaded on head and qwen_image_2512_bf16 does "
        "not fit beside it. Clear the Sparks and run again?")
    assert "on head" in card["numbers"]


def test_a_row_from_an_older_worker_keeps_the_held_wording_and_offers_no_card():
    old = replace(_stock("refuse", size_gib=19.0, avail_gib=27.4),
                  resident_names={"cond": None, "uncond": None}, resident_bytes={})
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, old), _fits(1, "sibling")], 2, unet_name=NEW)
    text = str(raised.value)
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    assert "rank 0 (host host0) holds a stock resident in the cond slot" in text
    assert "does not model the drop" in text


def test_a_short_row_with_no_figures_never_claims_room():
    row = replace(_stock("refuse", size_gib=19.0, avail_gib=27.4), measured=None)
    assert old_model.fleet_offer([(_answer(0, row), row)], [_answer(0, row)], NEW) is None


def test_two_slots_on_one_rank_count_that_rank_once():
    cond = _stock("refuse", size_gib=10.0, avail_gib=20.0, arena_gib=1.0)
    uncond = replace(cond, slot="uncond")
    answer = _answer(0, cond, uncond)
    offer = old_model.fleet_offer([(answer, cond), (answer, uncond)], [answer], NEW)
    assert offer is not None
    assert [box.gain_bytes for box in offer.boxes] == [old_model.rank_gain(cond)]


def test_a_local_fleet_says_this_box_and_the_log_line_says_what_was_cleared():
    short = _stock("refuse", size_gib=19.0, avail_gib=27.4)
    local = _answer(0, short, box=capacity_agreement.LOCAL_HOST_NAME)
    offer = old_model.fleet_offer([(local, short)], [local, replace(local, rank=1)], NEW)
    assert offer is not None and offer.local and offer.helps
    assert old_model.card_body(offer).endswith("on this box and qwen_image_2512_bf16 does "
                                               "not fit beside it. Clear this box and run again?")
    assert old_model.log_line(offer) == (
        "cleared this box to make room for qwen_image_2512_bf16; flux1-dev was loaded "
        "earlier")
    pair = old_model.fleet_offer([(_answer(0, short), short)],
                                 [_answer(0, short), _fits(1, "sibling")], NEW)
    assert pair is not None
    assert old_model.log_line(pair).startswith("cleared both Sparks to make room for")


def test_the_names_the_card_uses_are_the_ones_the_driver_and_the_loader_use():
    assert old_model.LOCAL_BOX == capacity_agreement.LOCAL_HOST_NAME
    from dgx_monarch.nodes import loader_preflight

    assert old_model.LOADER_GUARD == loader_preflight.LOADER_GUARD
    spec = consent_pending.KIND_SPECS[consent_descriptor.KIND_CLEAR_RETRY]
    assert spec.env_var == old_model.CLEAR_ENV == consent_descriptor.ENV_CLEAR_RETRY
    assert spec.primary_label == old_model.CLEAR_LABEL
    assert spec.always_label == old_model.ALWAYS_LABEL
    assert spec.title == old_model.CARD_TITLE
    # No worker raises it, so no refusal text may carry it.
    assert not consent_descriptor.valid_kind(consent_descriptor.KIND_CLEAR_RETRY)
    assert old_model.display_name("video/wan2.1_t2v_14B_fp16.safetensors") == (
        "wan2.1_t2v_14B_fp16")


# The Worker's facts.

class _Stored:
    def __init__(self, name):
        self.base_key = (name, ())


def test_the_worker_reports_each_residents_name_size_and_the_arena(monkeypatch):
    store = types.SimpleNamespace(
        slab_weights="auto", lora_low_rss=True, slab_blocked_reason="", family_override=None,
        current=_Stored(OLD), uncond=None,
        snapshot=lambda: {"cond": {"residency": "stock", "request_key": "('x',)"},
                          "uncond": None, "cleanup_failed_slots": []})
    worker = types.SimpleNamespace(rank=0, world=2, topology={}, _setup_key=("ready",),
                                   store=store)
    monkeypatch.setattr(capacity_quote, "price",
                        lambda **kw: _stock("fits", size_gib=1.0, avail_gib=200.0))
    monkeypatch.setattr(worker_capacity, "_pinned_staging", lambda: True)
    monkeypatch.setattr(worker_capacity, "_spec_bytes",
                        lambda spec: 7 * _GIB if spec["unet_name"] == OLD else 0)
    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda: 3 * _GIB)
    from dgx_monarch.actor import model_store

    monkeypatch.setattr(model_store, "resolve_model_path", lambda _kind, name: f"/m/{name}")
    monkeypatch.setattr(model_store, "_slab_capable_path", lambda _path: True)
    monkeypatch.setattr(model_store, "_to_comfy_model_options", lambda o: dict(o or {}))
    envelope = worker_capacity.quote(worker, {"specs": [
        {"unet_name": NEW, "options": {}, "loras": [], "slot": "cond"}]})
    (row,) = envelope["quotes"]
    assert row["resident_names"] == {"cond": OLD, "uncond": None}
    assert row["resident_bytes"] == {"cond": 7 * _GIB, "uncond": 0}
    assert row["arena_bytes"] == 3 * _GIB


def test_the_arena_is_measured_from_the_first_load_and_never_moves(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nRssAnon:\t 4194304 kB\n")
    assert actor_footprint.arena_bytes(str(status)) == 0      # no floor yet
    actor_footprint.note_first_load(str(status))
    status.write_text("RssAnon:\t 39845888 kB\n")             # 38 GiB now
    actor_footprint.note_first_load(str(status))              # a later load: no move
    assert actor_footprint.arena_bytes(str(status)) == 34 * _GIB
    status.write_text("no such line\n")
    assert actor_footprint.arena_bytes(str(status)) == 0
    actor_footprint.forget()
    actor_footprint.note_first_load(str(tmp_path / "missing"))
    status.write_text("RssAnon:\t 39845888 kB\n")
    assert actor_footprint.arena_bytes(str(status)) == 0      # an unread floor stays unread


def test_the_floor_is_taken_before_the_first_load_drops_or_places_anything():
    tree = ast.parse((REPO / "src/dgx_monarch/actor/store_load.py").read_text())
    fresh = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "fresh_load")
    calls = [ast.unparse(node.func) for node in ast.walk(fresh) if isinstance(node, ast.Call)]
    assert calls.index("actor_footprint.note_first_load") < calls.index("store._drop")
    assert calls.index("actor_footprint.note_first_load") < calls.index(
        "store_residency.resolve")


# The loader-site condition: the head's Worker shares the driver host.

class _LiveHandle:
    def __init__(self, *, local=True, defunct=False, hosts=()):
        self.owns_hosts = not local
        self.defunct = defunct
        self.setup_key = ("ready",)
        self.workers = object()
        self.world = 2 if hosts else 1
        self.gpus_per_host = 1
        self.config = types.SimpleNamespace(hosts=hosts, worker_args={})
        self.recycles = 0

    def effective_worker_args(self, worker_args=None):
        return dict(worker_args or {})

    def recycle_detailed(self):
        from dgx_monarch.mesh_recycle import RecycleOutcome, RecycleStatus

        self.recycles += 1
        self.defunct = True
        return RecycleOutcome(RecycleStatus.RECYCLED, "Attached mesh reset")


def _window(projected_gib: float, available_gib: float, reserve_gib: float):
    """One loader window: what it places, MemAvailable, and the reserve it keeps."""
    from dgx_monarch import driver_footprint

    return driver_footprint.DriverFootprintEstimate(
        weight_bytes=int(projected_gib * _GIB), mem_available=int(available_gib * _GIB),
        reserve=int(reserve_gib * _GIB))


def test_the_loader_asks_only_a_live_fleet_on_its_own_host(monkeypatch):
    held = _stock("fits", size_gib=1.0, avail_gib=200.0, resident="longcat_image_bf16.safetensors",
                  resident_gib=12.0)
    asked: list[float] = []

    def collect(handle, request, world, *, deadline_s):
        asked.append(deadline_s)
        return [_answer(0, held, box="head"), _answer(1, held, box="sibling")]

    monkeypatch.setattr(capacity_agreement, "collect", collect)
    from dgx_monarch import driver_footprint

    monkeypatch.setattr(driver_footprint, "host_is_local", lambda name: name == "head")
    hosts = (types.SimpleNamespace(name="head"), types.SimpleNamespace(name="sibling"))
    mesh = types.SimpleNamespace(handle=_LiveHandle(local=False, hosts=hosts))
    offer = old_model_clear.loader_offer(mesh, NEW, {}, windows=(_window(45.9, 47.3, 5.0),))
    assert asked == [old_model_clear.LOADER_QUOTE_DEADLINE_S]
    assert offer is not None and offer.helps and not offer.local and offer.hosts == 2
    (box,) = offer.boxes
    assert box.box == "head" and box.old_models == ("longcat_image_bf16",)
    assert box.gain_bytes == 12 * _GIB            # the sibling's resident is not on this host
    mesh.handle.defunct = True
    assert old_model_clear.loader_offer(mesh, NEW, {}, windows=(_window(45.9, 47.3, 5.0),)) is None

    def broken(*_a, **_k):
        raise RuntimeError("no answer")

    monkeypatch.setattr(capacity_agreement, "collect", broken)
    mesh.handle.defunct = False
    assert old_model_clear.loader_offer(mesh, NEW, {}, windows=(_window(45.9, 47.3, 5.0),)) is None


def _loader_rig(monkeypatch, tmp_path, held_gib):
    """A local fleet whose head Worker holds a stock old model, and a short box."""
    from dgx_monarch import driver_footprint
    from dgx_monarch.adapters import detect
    from dgx_monarch.nodes import loader_preflight

    loader_preflight.reset_memos()
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 40 * _GIB)
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda path: 20 * _GIB)
    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: ("minimax_h3", "bf16"))
    held = _stock("fits", size_gib=1.0, avail_gib=200.0, resident_gib=held_gib,
                  resident="longcat_image_bf16.safetensors")
    monkeypatch.setattr(capacity_agreement, "collect",
                        lambda *_a, **_k: [_answer(0, held, box="this box")])
    return types.SimpleNamespace(handle=_LiveHandle(), topology_preset="uly2", world=1,
                                 worker_args={})


def test_the_loader_raises_the_card_when_a_clear_makes_room(monkeypatch, tmp_path):
    """ltx-i2v-av on 2026-11-04 refused at the loader with longcat still held."""
    from dgx_monarch.nodes import loader_preflight

    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    with pytest.raises(old_model.LoaderOldModelError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None)
    text = str(raised.value)
    assert "driver-side footprint preflight at the loader node" in text
    assert "The old model (longcat_image_bf16) is still loaded on this box" in text
    assert '"Clear and retry"' in text and "Load with slab residency" not in text
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.guard == "loader_footprint_preflight" and not tag.waivable
    assert [card["kind"] for card in consent_pending.pending_cards()] == ["clear-retry"]
    assert consent_pending.pending_cards()[0]["detail"].endswith(
        "Clear this box and run again?")


def test_the_loader_keeps_its_own_refusal_when_a_clear_would_not_make_room(
        monkeypatch, tmp_path):
    from dgx_monarch import driver_footprint
    from dgx_monarch.nodes import loader_preflight

    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=1.0)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None)
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    text = str(raised.value)
    assert "The old model (longcat_image_bf16) is still loaded on this box" in text
    assert "A clear would not make room" in text
    assert "clear-retry" not in [card["kind"] for card in consent_pending.pending_cards()]


# The memo flag and the env var.

def test_the_always_flag_sits_beside_auto_rescue_and_an_older_memo_still_reads(tmp_path):
    assert consent_store.standing("auto_clear") is False
    consent_store.set_standing("auto_clear", True)
    assert consent_store.standing("auto_clear") is True
    assert consent_store.auto_rescue() is False
    consent_store.set_auto_rescue(True)
    assert consent_store.read()["auto_clear"] is True and consent_store.auto_rescue()
    # A schema-2 memo from before the flag: its consents and auto_rescue stay.
    Path(consent_store.MEMO_PATH).write_text(
        '{"schema": 2, "auto_rescue": true, "consents": {"k": {"id": "x"}}}')
    memo = consent_store.read()
    assert memo["auto_clear"] is False and memo["auto_rescue"] is True
    assert memo["consents"] == {"k": {"id": "x"}}
    assert consent_store.SCHEMA == 2


def test_the_env_var_turns_always_on_headless_and_a_store_fault_reads_off(monkeypatch):
    assert old_model_clear.always_on() is False
    monkeypatch.setenv("DGXM_ALLOW_CLEAR_RETRY", "1")
    assert old_model_clear.always_on() is True and old_model_clear.always_env()
    monkeypatch.delenv("DGXM_ALLOW_CLEAR_RETRY")

    def broken(_flag):
        raise OSError("disk gone")

    monkeypatch.setattr(consent_store, "standing", broken)
    assert old_model_clear.always_on() is False


# The consent route.

def _card(fleet: object = None) -> dict:
    """A pending card, bound to ``fleet``: an unbound card resets nothing."""
    offer = old_model.ClearOffer(NEW, (old_model.ShortBox(
        "head", ("flux1-dev",), 45 * _GIB, 27 * _GIB, 22 * _GIB),), 2, False)
    card = old_model_clear.register_card(offer, old_model.FLEET_GUARD, fleet)
    assert card is not None
    return card


@pytest.fixture
def attached(monkeypatch):
    """A fake attached mesh, reset through the recycle route's own teardown."""
    from dgx_monarch import mesh
    from dgx_monarch.nodes import routes

    handle = _LiveHandle()
    monkeypatch.setattr(mesh, "_MESHES", {"key": handle})
    drained: list[int] = []
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: drained.append(1))
    monkeypatch.setattr(routes, "_RECYCLE_RATE", {"active": False,
                                                 "last_finished": float("-inf")})
    handle.drained = drained
    return handle


def test_the_state_lists_the_card_and_the_toggle_without_a_quarantine_read(monkeypatch):
    from dgx_monarch.nodes import consent_quarantine

    def no_scan(queries):
        assert not queries, "an action card asks no quarantine question"
        return {}

    monkeypatch.setattr(consent_quarantine, "quarantine_decisions", no_scan)
    card = _card()
    state = consent_routes.consent_state()
    assert [row["id"] for row in state["pending"]] == [card["id"]]
    assert state["auto_clear"] is False and state["auto_clear_env"] is False
    assert state["kinds"]["clear-retry"]["class"] == "C"


def test_accept_clears_the_attached_mesh_and_saves_nothing(attached):
    card = _card(attached)
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["ok"] and payload["cleared"]
    assert payload["detail"] == "Cleared both Sparks; queue the render again."
    assert attached.recycles == 1 and attached.drained == [1]
    assert consent_pending.pending_cards() == []
    memo = consent_store.read()
    assert memo["consents"] == {} and memo["auto_clear"] is False
    output = Path(sys.modules["folder_paths"].get_output_directory())
    assert list(output.iterdir()) == []                      # no ledger row


def test_accept_with_always_saves_the_toggle_too(attached):
    card = _card(attached)
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"], "always": True})
    assert status == 200 and payload["always"] is True
    assert consent_store.standing("auto_clear") is True and attached.recycles == 1


def test_a_refused_reset_keeps_the_card_open(attached):
    card = _card(attached)
    attached.recycle_detailed = lambda: types.SimpleNamespace(
        ok=False, retryable=True, as_dict=lambda: {
            "ok": False, "status": "active_work", "retryable": True, "detail": "busy"})
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 409 and not payload["ok"] and "busy" in payload["detail"]
    assert [c["id"] for c in consent_pending.pending_cards()] == [card["id"]]


def test_no_attached_mesh_means_nothing_to_clear(monkeypatch):
    from dgx_monarch import mesh
    from dgx_monarch.nodes import routes

    monkeypatch.setattr(mesh, "_MESHES", {})
    monkeypatch.setattr(routes, "_RECYCLE_RATE", {"active": False,
                                                 "last_finished": float("-inf")})
    card = _card(_LiveHandle())
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["cleared"] is False
    assert payload["detail"] == old_model_clear.NOTHING_TO_CLEAR


def test_a_reset_already_running_answers_429(attached, monkeypatch):
    from dgx_monarch.nodes import routes

    monkeypatch.setattr(routes, "_RECYCLE_RATE", {"active": True,
                                                 "last_finished": float("-inf")})
    card = _card(attached)
    status, _payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 429 and attached.recycles == 0


def test_the_toggle_action_answers_open_cards_and_wants_a_boolean():
    _card()
    assert consent_routes.handle_action({"action": "auto_clear", "value": "yes"})[0] == 400
    status, payload = consent_routes.handle_action({"action": "auto_clear", "value": True})
    assert status == 200 and payload["cleared"] == 1 and consent_pending.pending_cards() == []
    assert consent_store.standing("auto_clear") is True


def test_the_post_route_gives_a_clear_the_recycle_deadline(monkeypatch):
    """A recycle is bounded at 180 s; the consent action's 20 s would cut it off."""
    import asyncio

    from dgx_monarch.nodes import routes

    seen: list[float] = []

    async def bounded(fn, timeout_s):
        seen.append(timeout_s)
        return 200, {"ok": True}

    monkeypatch.setattr(routes, "_await_bounded", bounded)
    monkeypatch.setattr(routes, "_action_request_allowed", lambda *_a: (True, ""))
    handlers = {}

    class _Routes:
        def get(self, path):
            return lambda fn: handlers.setdefault(path, fn)

        def post(self, path):
            return lambda fn: handlers.setdefault(path, fn)

    web = types.SimpleNamespace(json_response=lambda payload, status=200: (status, payload),
                                HTTPForbidden=RuntimeError)
    consent_routes.register(types.SimpleNamespace(routes=_Routes()), web)
    card = _card()

    def request(body):
        import json

        raw = json.dumps(body).encode()

        class _Content:
            async def read(self, _n):
                return raw

        return types.SimpleNamespace(headers={}, scheme="http", content_length=len(raw),
                                     content=_Content())

    post = handlers["/dgxm/consent"]
    asyncio.run(post(request({"action": "accept", "key": card["key"], "id": card["id"]})))
    asyncio.run(post(request({"action": "dismiss", "key": card["key"], "id": card["id"]})))
    assert seen == [routes._RECYCLE_TIMEOUT_S, consent_routes._ACTION_TIMEOUT_S]


# The Always flow against a fake mesh.

def _refusal(offer_helps: bool = True) -> BaseException:
    gain = 22 if offer_helps else 1
    offer = old_model.ClearOffer(NEW, (old_model.ShortBox(
        "head", ("flux1-dev",), 45 * _GIB, 27 * _GIB, gain * _GIB),), 2, False)
    try:
        old_model.raise_fleet_refusal(offer, 1, 2, "priced.")
    except old_model.OldModelRefusal as exc:
        return exc
    raise AssertionError("unreachable")


def test_always_clears_once_and_says_so(monkeypatch, caplog):
    handle = _LiveHandle()
    assert old_model_clear.clear_for_retry(handle, _refusal()) is False   # off
    consent_store.set_standing("auto_clear", True)
    _card(handle)
    with caplog.at_level("INFO"):
        assert old_model_clear.clear_for_retry(handle, _refusal()) is True
    assert handle.recycles == 1
    assert consent_pending.pending_cards() == []         # the reset's drain answered it
    assert ("cleared both Sparks to make room for qwen_image_2512_bf16; flux1-dev was "
            "loaded earlier") in caplog.text


def test_always_never_clears_a_pinned_session_a_refusal_it_cannot_answer_or_a_busy_mesh(
        monkeypatch):
    monkeypatch.setenv("DGXM_ALLOW_CLEAR_RETRY", "1")
    handle = _LiveHandle()
    assert not old_model_clear.clear_for_retry(handle, _refusal(), inherited=True)
    assert not old_model_clear.clear_for_retry(handle, _refusal(offer_helps=False))
    assert not old_model_clear.clear_for_retry(handle, FleetCapacityError("short"))
    assert handle.recycles == 0
    handle.recycle_detailed = lambda: types.SimpleNamespace(
        ok=False, status=types.SimpleNamespace(value="active_work"), detail="busy")
    assert not old_model_clear.clear_for_retry(handle, _refusal())


def test_settled_is_two_close_readings_or_the_deadline():
    readings = iter([10 * _GIB, 60 * _GIB, 90 * _GIB, 90 * _GIB + 1])
    sleeps: list[float] = []
    value = _WAIT_SETTLED(read=lambda: next(readings), sleep=sleeps.append,
                          clock=lambda: 0.0)
    assert value == 90 * _GIB + 1
    assert sleeps == [old_model_clear.SETTLE_POLL_S] * 3
    # Still climbing at the deadline: the last reading, and no hang.
    climbing = iter(range(1, 100))
    clock = iter([0.0, 1.0, 2.0, 16.0])
    value = _WAIT_SETTLED(read=lambda: next(climbing) * _GIB, sleep=lambda _s: None,
                          clock=lambda: next(clock))
    assert value == 3 * _GIB
    assert _WAIT_SETTLED(read=lambda: None, sleep=lambda _s: None, clock=lambda: 0.0) is None


def test_the_sample_path_clears_once_then_retries_once(monkeypatch):
    from dgx_monarch.nodes import render_submit

    consent_store.set_standing("auto_clear", True)
    handle = _LiveHandle()
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    monkeypatch.setattr(render_submit, "_claim_adoption_context", lambda: None)
    monkeypatch.setattr(render_submit, "_consume_adoption_context_claim", lambda claim: None)
    outcomes = [_refusal(), "pending"]

    def guarded(*_args, **kwargs):
        args = _args
        args[9].handle = handle                 # the guard publishes the fleet
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(render_submit, "_submit_render_guarded", guarded)
    monkeypatch.setattr(render_submit, "run_guarded_submit",
                        lambda guard, body, on_failure: body())
    assert render_submit.submit_render(object(), {}, {}, 1.0, 8) == "pending"
    assert handle.recycles == 1

    # Still short on the fresh fleet: the plain refusal, and no second clear.
    from dgx_monarch.nodes import routes

    monkeypatch.setattr(routes, "_RECYCLE_RATE", {"active": False,
                                                 "last_finished": float("-inf")})
    handle = _LiveHandle()
    outcomes[:] = [_refusal(), FleetCapacityError("still short on a fresh fleet")]
    with pytest.raises(FleetCapacityError, match="still short"):
        render_submit.submit_render(object(), {}, {}, 1.0, 8)
    assert handle.recycles == 1


def test_the_loader_clears_once_then_prices_again(monkeypatch):
    import contextlib

    from dgx_monarch.nodes import loader_preflight, loaders, render_preflight, render_session
    from dgx_monarch.nodes.common import MeshSpec

    consent_store.set_standing("auto_clear", True)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    old = _LiveHandle()
    fresh = _LiveHandle()
    loaded: list[str] = []
    fresh.call_all = lambda endpoint, *_a, **_k: (
        loaded.append(endpoint) or [{"host": "h", "transition": "load"}])
    fresh.abandoned_sample_leases = {}
    fresh.sample_leases = {}
    fresh.world = 2
    prices = [_loader_refusal(), object()]

    def price(*_a, **_k):
        outcome = prices.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", price)
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *_a, **_k: None)
    monkeypatch.setattr(loaders, "ensure_live",
                        lambda handle, **_k: fresh if old.defunct else old)
    monkeypatch.setattr(capacity_agreement, "agree_load", lambda *_a, **_k: None)
    from dgx_monarch import mesh_setup

    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda *_a, **_k: None)
    monkeypatch.setattr(mesh_setup, "token_kwargs", lambda _token: {})
    monkeypatch.setattr(render_preflight, "preflight_sol_sequence_parallel", lambda *_a: None)
    monkeypatch.setattr(render_session, "mutation_render_session",
                        lambda _handle: contextlib.nullcontext())
    mesh = MeshSpec(handle=old, topology_preset="uly2", attention="TORCH_FLASH",
                    sync_ulysses=True)
    loaders.DGXMonarchUNETLoader().load(mesh, NEW)
    assert old.recycles == 1 and prices == [] and loaded == ["load_model"]


def _loader_refusal() -> BaseException:
    offer = old_model.ClearOffer(NEW, (old_model.ShortBox(
        "this box", ("flux1-dev",), 46 * _GIB, 42 * _GIB, 12 * _GIB),), 1, True)
    try:
        old_model.raise_loader_refusal(offer, "driver-side footprint preflight refused.")
    except old_model.LoaderOldModelError as exc:
        return exc
    raise AssertionError("unreachable")


# Review round 1 (2026-11-04): one block per finding.

def test_the_loader_card_needs_room_in_both_windows_against_unclamped_budgets(monkeypatch):
    """Finding 1: the worst failing window decides, with the reserve unclamped."""
    def offer_for(held_gib, *windows):
        held = _stock("fits", size_gib=1.0, avail_gib=200.0, resident_gib=held_gib)
        monkeypatch.setattr(capacity_agreement, "collect",
                            lambda *_a, **_k: [_answer(0, held, box="this box")])
        return old_model_clear.loader_offer(types.SimpleNamespace(handle=_LiveHandle()),
                                            NEW, {}, windows=windows)

    transient = _window(75.9, 45.0, 5.0)          # short by 35.9 GiB
    settled = _window(46.6, 45.0, 15.0)           # short by 16.6 GiB
    both = offer_for(27.0, transient, settled)
    assert both is not None and not both.helps
    assert both.boxes[0].shortfall_bytes == transient.shortfall
    assert offer_for(27.0, _window(30.0, 45.0, 5.0), settled).helps   # settled alone short
    clamped = _window(14.0, 12.5, 15.0)           # MemAvailable under the reserve
    assert clamped.usable == 0 and clamped.shortfall == int(16.5 * _GIB)
    offer = offer_for(17.0, clamped)
    assert offer is not None and offer.boxes[0].shortfall_bytes == clamped.shortfall
    assert not offer.helps                        # 16.5 short plus the margin is over 17
    assert offer_for(27.0, _window(10.0, 45.0, 5.0)) is None             # nothing short


def test_a_slot_the_request_reuses_is_neither_old_nor_credited():
    """Finding 2: the cond model a two-slot request reuses is not in the way."""
    cond_name = "ideogram4_fp8_scaled.safetensors"
    cond = replace(_stock("fits", size_gib=20.0, avail_gib=200.0, resident=cond_name,
                          resident_gib=20.0), would_load="reuse", checkpoint_bytes=0)
    uncond = replace(_stock("refuse", size_gib=10.0, avail_gib=10.0, resident=cond_name,
                            resident_gib=20.0), slot="uncond", would_load="fresh")
    assert 0 < _shortfall_gib(uncond) < 18.0      # would read as room without the fix
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, cond, uncond), _fits(1, "sibling")], 2,
               unet_name="ideogram4_unconditional_fp8_scaled.safetensors")
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    assert "ideogram4_fp8_scaled) is still loaded" not in str(raised.value)


def test_a_model_this_graph_loads_is_never_the_old_model(monkeypatch):
    """Finding 2: names the request or the graph loads, at both agreement doors."""
    short = _stock("refuse", size_gib=19.0, avail_gib=27.4)
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=NEW,
               keep=frozenset({OLD}))
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    handle = types.SimpleNamespace(world=2, workers=object(), active_worker_args={})
    monkeypatch.setattr(capacity_agreement, "collect", lambda *_a, **_k: [
        _answer(0, short), _fits(1, "sibling")])
    with pytest.raises(FleetCapacityError) as eager:          # the uncond loader
        capacity_agreement.agree_load(handle, NEW, {}, [], "uncond", keep={OLD, NEW})
    assert not isinstance(eager.value, old_model.OldModelRefusal)
    with old_model.graph_loads({OLD}), pytest.raises(FleetCapacityError) as named:
        capacity_agreement.agree_load(handle, NEW, {}, [], "uncond")   # as loaders.py calls it
    assert not isinstance(named.value, old_model.OldModelRefusal)
    assert old_model.graph_names() == frozenset()
    request = capacity_agreement.quote_request([
        {"unet_name": OLD, "slot": "cond"}, {"unet_name": NEW, "slot": "uncond"}])
    with pytest.raises(FleetCapacityError) as sample:         # the sample path
        capacity_agreement.agree_or_refuse(handle, request, unet_name=NEW)
    assert not isinstance(sample.value, old_model.OldModelRefusal)
    held = _stock("fits", size_gib=1.0, avail_gib=200.0)
    monkeypatch.setattr(capacity_agreement, "collect",
                        lambda *_a, **_k: [_answer(0, held, box="this box")])
    assert old_model_clear.loader_offer(types.SimpleNamespace(handle=_LiveHandle()), NEW, {},
                                        windows=(_window(45.9, 47.3, 5.0),),
                                        keep={OLD}) is None


def test_every_reset_answers_the_open_cards(monkeypatch):
    """Finding 3: a manual reset, a heal or an eviction all drain through here."""
    card_key = _card(_LiveHandle())["key"]
    assert consent_pending.pending_cards()
    recycle_drain.drop_residency_memos()
    assert consent_pending.pending_cards() == []
    assert old_model_clear.bound_fleet(card_key) is None


def test_a_card_never_resets_a_fleet_that_came_later(monkeypatch):
    """Finding 3: the card resets the fleet that refused, or nothing."""
    from dgx_monarch import mesh

    first, later = _LiveHandle(), _LiveHandle()
    offer = old_model.ClearOffer(NEW, (old_model.ShortBox(
        "head", ("flux1-dev",), 45 * _GIB, 27 * _GIB, 22 * _GIB),), 2, False)
    card = old_model_clear.register_card(offer, old_model.FLEET_GUARD, first)
    assert card is not None
    monkeypatch.setattr(mesh, "_MESHES", {"key": later})
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["cleared"] is False
    assert payload["detail"] == old_model_clear.GONE
    assert later.recycles == 0 and consent_pending.pending_cards() == []
    from dgx_monarch.nodes import routes

    monkeypatch.setattr(routes, "_RECYCLE_RATE", {"active": False,
                                                 "last_finished": float("-inf")})
    card = old_model_clear.register_card(offer, old_model.FLEET_GUARD, later)
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["cleared"] is True and later.recycles == 1


def test_the_agreement_binds_its_card_to_the_fleet_that_refused(monkeypatch):
    handle = types.SimpleNamespace(world=2, workers=object(), active_worker_args={})
    short = _stock("refuse", size_gib=19.0, avail_gib=27.4)
    monkeypatch.setattr(capacity_agreement, "collect", lambda *_a, **_k: [
        _answer(0, short), _fits(1, "sibling")])
    with pytest.raises(old_model.OldModelRefusal) as raised:
        capacity_agreement.agree_or_refuse(handle, {}, unet_name=NEW)
    assert raised.value.fleet is handle
    (card,) = consent_pending.pending_cards()
    assert old_model_clear.bound_fleet(card["key"]) is handle


def test_a_shard_is_credited_its_arena_only_and_a_cast_the_copy_it_built(monkeypatch):
    """Finding 4: an FSDP shard does not hold its whole file, and a load cast to
    another weight dtype holds the copy its cast built, the figure its stock
    price charged (capacity_fit.cast_resident_bytes), not the file."""
    monkeypatch.setattr(worker_capacity, "_spec_bytes", lambda spec: 20 * _GIB)
    casts = []
    monkeypatch.setattr(worker_capacity, "_cast_bytes",
                        lambda spec: casts.append(spec) or 10 * _GIB)
    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda: 0)

    def facts(stored, topology=None):
        store = types.SimpleNamespace(current=stored, uncond=None)
        worker = types.SimpleNamespace(topology=topology or {})
        return worker_capacity._resident_facts(worker, store)["resident_bytes"]["cond"]

    plain = types.SimpleNamespace(base_key=(OLD, ()), fsdp_checkpoint_pin=None)
    assert facts(plain) == 20 * _GIB
    assert facts(plain, {"fsdp": True}) == 0
    pinned = types.SimpleNamespace(base_key=(OLD, ()), fsdp_checkpoint_pin=object())
    assert facts(pinned) == 0
    cast = types.SimpleNamespace(base_key=(OLD, (("weight_dtype", "fp8_e4m3fn"),)),
                                 fsdp_checkpoint_pin=None)
    assert facts(cast) == 10 * _GIB
    assert casts == [{"unet_name": OLD, "options": {"weight_dtype": "fp8_e4m3fn"}}]
    assert facts(cast, {"fsdp": True}) == 0


def test_a_clear_must_beat_the_shortfall_by_a_margin():
    """Finding 4: a fresh Worker takes some memory back on its first load."""
    assert old_model.CLEAR_MARGIN_BYTES == actor_footprint.RESET_MARGIN_BYTES == 2 * _GIB
    tight = old_model.ShortBox("head", ("old",), 10 * _GIB, 0, 11 * _GIB)
    roomy = old_model.ShortBox("head", ("old",), 10 * _GIB, 0, 13 * _GIB)
    assert not tight.clear_makes_room and roomy.clear_makes_room
    assert tight.cleared_bytes == 9 * _GIB


def test_the_worker_wall_names_the_arena_an_earlier_render_left(monkeypatch):
    """Finding 5: the one-GPU and Fleet wall prices after its own drop."""
    from dgx_monarch.actor import store_residency

    fit = StockFit(False, True, 10 * _GIB, 20 * _GIB)
    assert 0 < fit.required_bytes - fit.avail_bytes < 20 * _GIB
    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda: 30 * _GIB)
    expected = ("This Worker still holds about 30.0 GiB that a model loaded earlier left "
                "behind, and a reset gives it back: click Reset attached mesh in the DGX "
                "Monarch panel, then queue the render again.")
    assert expected in store_residency._no_rescue_text(OLD, fit, ("why", "a smaller file"))
    offer = types.SimpleNamespace(unet_name=OLD)
    assert expected in store_residency._rescue_offer_text(offer, fit)
    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda: 1 * _GIB)
    assert "Reset attached mesh" not in store_residency._no_rescue_text(OLD, fit, ("w", "o"))
    assert actor_footprint.ladder_clause(1, None) == ""


def _loader_seams(monkeypatch, old, fresh, *, agree=None, prompt_seen=None):
    """The eager loader with inert preflights, a cache and a recording agreement."""
    import contextlib

    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import loader_preflight, loaders, render_preflight, render_session

    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    for handle in (old, fresh):
        handle.call_all = lambda endpoint, *_a, _h=handle, **_k: (
            _h.__dict__.setdefault("loaded", []).append(endpoint)
            or [{"host": "h", "transition": "load"}])
    fresh.world = 2
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *_a, **_k: None)
    monkeypatch.setattr(loaders, "ensure_live",
                        lambda handle, **_k: fresh if old.defunct else old)
    monkeypatch.setattr(capacity_agreement, "agree_load", agree or (lambda *_a, **_k: None))
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda *_a, **_k: None)
    monkeypatch.setattr(mesh_setup, "token_kwargs", lambda _token: {})
    sols: list[object] = []
    monkeypatch.setattr(render_preflight, "preflight_sol_sequence_parallel",
                        lambda topo, _attention: sols.append(topo))
    monkeypatch.setattr(render_session, "mutation_render_session",
                        lambda _handle: contextlib.nullcontext())
    return sols


def test_the_eager_load_clears_once_and_runs_its_preflights_on_the_fresh_fleet(monkeypatch):
    """Findings 7 and 2: the agreement's refusal at the eager site, and the graph's names."""
    from dgx_monarch.nodes import loader_graph, loader_preflight, loaders
    from dgx_monarch.nodes.common import MeshSpec

    consent_store.set_standing("auto_clear", True)
    old, fresh = _LiveHandle(), _LiveHandle()
    old.world = 2
    keeps: list[object] = []

    def agree(handle, *_args, **_kwargs):
        keeps.append(old_model.graph_names())     # what the loader names around its agreement
        if handle is old:
            raise _refusal()

    sols = _loader_seams(monkeypatch, old, fresh, agree=agree)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        lambda *_a, **_k: object())
    prompt = {"1": {"class_type": "DGXMonarchUNETLoader", "inputs": {"unet_name": NEW}},
              "2": {"class_type": "DGXMonarchUncondUNETLoader", "inputs": {"unet_name": OLD}}}
    mesh = MeshSpec(handle=old, topology_preset="uly2", attention="TORCH_FLASH",
                    sync_ulysses=True)
    loaders.DGXMonarchUNETLoader().load(mesh, NEW, prompt=prompt)
    assert old.recycles == 1 and fresh.loaded == ["load_model"]
    assert not getattr(old, "loaded", [])
    assert len(sols) == 2                       # the preflight ran on the fresh topology too
    assert keeps == [loader_graph.unet_names(prompt)] * 2 and OLD in keeps[0]


def test_the_loader_never_clears_under_a_pinned_session(monkeypatch):
    """Finding 6: a Pipeline or Gate session owns its mesh, at both loader sites."""
    from dgx_monarch.nodes import loader_preflight, loaders, render_session
    from dgx_monarch.nodes.common import MeshSpec

    consent_store.set_standing("auto_clear", True)
    monkeypatch.setattr(render_session, "active_render_session", lambda: object())
    old, fresh = _LiveHandle(), _LiveHandle()
    old.world = 2
    _loader_seams(monkeypatch, old, fresh)

    def refuse(*_a, **_k):
        raise _loader_refusal()

    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", refuse)
    mesh = MeshSpec(handle=old, topology_preset="uly2", attention="TORCH_FLASH",
                    sync_ulysses=True)
    with pytest.raises(old_model.LoaderOldModelError):
        loaders.DGXMonarchUNETLoader().load(mesh, NEW)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        lambda *_a, **_k: object())

    def agree(*_args, **_kwargs):
        raise _refusal()

    monkeypatch.setattr(capacity_agreement, "agree_load", agree)
    with pytest.raises(old_model.OldModelRefusal):
        loaders.DGXMonarchUNETLoader().load(mesh, NEW)
    assert old.recycles == 0


def test_always_takes_the_reset_gate_and_its_drain_answers_the_cards(monkeypatch):
    """Finding 8 and N1: the reset holds the gate, and its own drain takes the cards."""
    from dgx_monarch.nodes import routes

    consent_store.set_standing("auto_clear", True)
    handle = _LiveHandle()
    _card(handle)
    seen: list[int] = []
    real = handle.recycle_detailed

    def recycle():
        seen.append(len(consent_pending.pending_cards()))
        assert routes._RECYCLE_RATE["active"] is True
        return real()

    handle.recycle_detailed = recycle
    assert old_model_clear.clear_for_retry(handle, _refusal()) is True
    assert seen == [1]                    # open until the reset runs, then drained
    assert consent_pending.pending_cards() == [] and routes._RECYCLE_RATE["active"] is False


def test_a_gated_or_refused_always_clear_leaves_the_card_open(monkeypatch):
    """N1: the refusal that is re-raised still points at the card, so it stays."""
    from dgx_monarch.nodes import routes

    consent_store.set_standing("auto_clear", True)
    gated = _LiveHandle()
    card = _card(gated)
    monkeypatch.setattr(routes, "_RECYCLE_RATE", {"active": True,
                                                 "last_finished": float("-inf")})
    assert old_model_clear.clear_for_retry(gated, _refusal()) is False
    assert gated.recycles == 0
    assert [c["id"] for c in consent_pending.pending_cards()] == [card["id"]]
    assert old_model_clear.bound_fleet(card["key"]) is gated     # binding kept too
    monkeypatch.setattr(routes, "_RECYCLE_RATE", {"active": False,
                                                 "last_finished": float("-inf")})
    gated.recycle_detailed = lambda: types.SimpleNamespace(
        ok=False, status=types.SimpleNamespace(value="lifecycle_busy"), detail="busy")
    assert old_model_clear.clear_for_retry(gated, _refusal()) is False
    assert [c["id"] for c in consent_pending.pending_cards()] == [card["id"]]


def test_a_card_with_no_binding_resets_nothing(attached):
    """Residual 3a: no known fleet means the old model is not known to be there."""
    card = _card()
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["cleared"] is False
    assert payload["detail"] == old_model_clear.GONE and attached.recycles == 0
    assert consent_pending.pending_cards() == []


def test_a_card_answered_after_its_binding_was_read_resets_nothing(attached, monkeypatch):
    """Bot finding B2: the card and its binding are read again under the reset gate."""
    from dgx_monarch.nodes import routes

    card = _card(attached)
    real_begin = routes._begin_recycle_request

    def begin_after_a_toggle():
        # Always clear and retry turned on between the first read and the gate.
        consent_routes.handle_action({"action": "auto_clear", "value": True})
        return real_begin()

    monkeypatch.setattr(routes, "_begin_recycle_request", begin_after_a_toggle)
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["cleared"] is False
    assert payload["detail"] == old_model_clear.GONE and attached.recycles == 0


def test_a_removed_card_keeps_no_hold_on_its_fleet(monkeypatch):
    """Bot finding B3: dismissal and expiry drop the binding with the card."""
    import gc
    import weakref

    fleet = _LiveHandle()
    alive = weakref.ref(fleet)
    card = _card(fleet)
    assert old_model_clear.bound_fleet(card["key"]) is fleet
    status, _payload = consent_routes.handle_action(
        {"action": "dismiss", "key": card["key"], "id": card["id"]})
    assert status == 200 and old_model_clear.bound_fleet(card["key"]) is None
    del fleet
    gc.collect()
    assert alive() is None                      # nothing else held the fleet
    expiring = _LiveHandle()
    alive = weakref.ref(expiring)
    card = _card(expiring)
    del expiring
    monkeypatch.setattr(consent_pending, "PENDING_TTL_S", -1.0)
    assert consent_pending.pending_cards() == []           # the read expires it
    gc.collect()
    assert alive() is None and old_model_clear.bound_fleet(card["key"]) is None


def test_after_a_clear_a_later_loader_refusal_binds_the_fresh_fleet(monkeypatch, tmp_path):
    """Bot finding B1: the cached MeshSpec holds the retired fleet; the fresh one answers."""
    from dgx_monarch import mesh
    from dgx_monarch.nodes import loader_preflight

    cached = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    retired = cached.handle
    retired.defunct = True                      # what a Clear and retry leaves behind
    fresh = _LiveHandle()
    asked: list[object] = []
    held = _stock("fits", size_gib=1.0, avail_gib=200.0, resident_gib=12.0,
                  resident="longcat_image_bf16.safetensors")

    def collect(handle, *_args, **_kwargs):
        asked.append(handle)
        return [_answer(0, held, box="this box")]

    monkeypatch.setattr(capacity_agreement, "collect", collect)
    monkeypatch.setattr(mesh, "_MESHES", {"key": fresh})
    with pytest.raises(old_model.LoaderOldModelError) as raised:
        loader_preflight.preflight_loader_footprint(cached, NEW, {}, None)
    assert asked == [fresh] and raised.value.fleet is fresh
    (card,) = consent_pending.pending_cards()
    assert old_model_clear.bound_fleet(card["key"]) is fresh
    consent_store.set_standing("auto_clear", True)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    assert old_model_clear.clear_for_retry(cached.handle, raised.value) is True
    assert fresh.recycles == 1 and retired.recycles == 0
    monkeypatch.setattr(mesh, "_MESHES", {})            # nothing attached: no question asked
    assert old_model_clear.loader_offer(cached, NEW, {},
                                        windows=(_window(45.9, 47.3, 5.0),)) is None


def test_a_card_for_the_second_attached_mesh_resets_that_mesh(monkeypatch):
    """Residual 3b: the bound fleet is found among every attached mesh."""
    from dgx_monarch import mesh

    first, second = _LiveHandle(), _LiveHandle()
    monkeypatch.setattr(mesh, "_MESHES", {"a": first, "b": second})
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    card = _card(second)
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["cleared"] is True
    assert second.recycles == 1 and first.recycles == 0


def test_the_worker_walls_for_slab_and_shards_name_the_arena_too(monkeypatch):
    """Residual 5: the ladder's slab refusal and its FSDP refusal carry the sentence."""
    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import store_residency

    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda: 40 * _GIB)
    slab = capacity_fit.SlabFit(False, True, 10 * _GIB, 5 * _GIB, 4 * _GIB)
    shard = capacity_fit.preflight_wall_measured(10 * _GIB, 5 * _GIB, 20 * _GIB, False)
    rows = [
        RankPrice(verdict="refuse", rung="explicit", use_slab=True, priced=True,
                  slab_measured=dict(slab.measured()), checkpoint_bytes=10 * _GIB,
                  reason="slab_weights=on was requested"),
        RankPrice(verdict="refuse", rung="stock_fits", priced=True, measured=shard,
                  checkpoint_bytes=10 * _GIB, reason="the shard build does not fit here"),
    ]
    for row in rows:
        monkeypatch.setattr(capacity_quote, "price", lambda _row=row, **_kw: _row)
        with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
            store_residency.resolve(
                path="/m/x.safetensors", unet_name=OLD, model_options={},
                slab_weights=True, slab_capable_path=True, lora_low_rss=False,
                fsdp_launch=False, blocked_reason="", authoritative_slab_retry=False,
                memoized_family=lambda _path: None, vouched_families=frozenset())
        assert ("This Worker still holds about 40.0 GiB that a model loaded earlier left "
                "behind") in str(raised.value)
        assert "Reset attached mesh in the DGX Monarch panel" in str(raised.value)
    assert actor_footprint.block_clause(None) == ""


def test_the_refusals_read_plainly():
    """Finding 9: whole sentences, one name per box, and a next step every time."""
    short = _stock("refuse", size_gib=19.0, avail_gib=27.4)
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=NEW)
    text = str(raised.value)
    assert "1 of 2 cannot hold it beside the model loaded there earlier" in text
    assert "earlier render" not in text
    assert "rank 0 on head priced a stock load" in text and "(host host0)" not in text
    with pytest.raises(FleetCapacityError) as plain:
        decide([_answer(0, replace(short, arena_bytes=0, resident_bytes={})),
                _fits(1, "sibling")], 2, unet_name=NEW)
    assert "Choose a smaller or quantized artifact" in str(plain.value)


def test_no_card_means_the_remedy_names_only_the_reset(monkeypatch):
    """Finding 9: a driver that cannot resolve the checkpoint raises no card."""
    missing = "not_on_this_driver.safetensors"
    short = _stock("refuse", size_gib=19.0, avail_gib=27.4)
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide([_answer(0, short), _fits(1, "sibling")], 2, unet_name=missing)
    text = str(raised.value)
    assert "Open the DGX Monarch panel and click Reset attached mesh, then queue" in text
    assert '"Clear and retry"' not in text and "DGXM_ALLOW_CLEAR_RETRY=1" in text
    offer = old_model.ClearOffer(missing, (old_model.ShortBox(
        "this box", ("flux1-dev",), 46 * _GIB, 42 * _GIB, 12 * _GIB),), 1, True)
    with pytest.raises(old_model.LoaderOldModelError) as loader:
        old_model_clear.refuse_at_loader(offer, "refused.")
    assert '"Clear and retry"' not in str(loader.value)
    assert consent_pending.pending_cards() == []


def test_the_card_copy_and_its_answers_are_plain():
    """Finding 9: no project terms on the card or in its answer."""
    spec = consent_pending.KIND_SPECS[consent_descriptor.KIND_CLEAR_RETRY]
    assert spec.risk == ("A model loaded earlier is still in memory, so this one does not "
                         "fit beside it.")
    assert spec.evidence == ("Clear and retry gives back all the memory the Sparks hold "
                             "for models, the same as Reset attached mesh, and changes no "
                             "setting.")
    local = old_model.ClearOffer(NEW, (), 1, True)
    pair = old_model.ClearOffer(NEW, (), 2, False)
    assert old_model_clear.cleared(local) == "Cleared this box; queue the render again."
    assert old_model_clear.cleared(pair) == "Cleared both Sparks; queue the render again."
