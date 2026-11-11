"""A LoRA change on a held base is credited ComfyUI's host copy of the patched weights.

After a base-plus-LoRA load on the BF16 Wan-Animate 2 templates each actor keeps
about 34 GiB of anonymous memory: ComfyUI's backup of every weight the stack
patched, as it was, on the host. On the 2026-11-05 re-check (E3, one Spark) a
strength change from that state swapped in place, grew the actor by 0.8 GiB,
took 1.7 s of gpu_load and rendered a fresh load's latent; the agreement charged
it 38.6 GiB, which on the pair's 24 GiB head raised Old model in the way (E1).
The Worker now reports the backup's host bytes, a lower bound, and the agreement
credits them to a change whose LoRA files are all in the held stack. CPU only.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

from dgx_monarch import box_names, capacity_agreement, capacity_fit, capacity_lora_bake, mesh_safety, old_model
from dgx_monarch.actor import actor_footprint, worker_capacity
from dgx_monarch.actor import capacity_quote as cq
from dgx_monarch.actor import model_store as ms
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.actor.stored_model import StoredModel
from dgx_monarch.capacity_agreement import RankAnswer, decide
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.

_GIB = 2 ** 30
BASE = "wan_animate_2_bf16.safetensors"
LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
NARROW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
SCAIL = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
BF16_BYTES = 32_789_894_104                     # the 30.5 GiB file
# The backup, from the safetensors headers on disk: the distill LoRA patches 1261 of
# the base's 1303 tensors, and those hold 32,785,804,928 bytes. The 42 it leaves out
# hold 3.8 MiB. The Worker reports the same sum from the tensors it holds.
BACKUP_BYTES = 32_785_804_928
# The LoRA files on disk, which the Worker reads again on every in-place change.
LORA_FILES = {LORA: 738_005_744, NARROW: 1_226_977_424, SCAIL: 1_226_936_552}
HEAD_AVAILABLE = int(24.3 * _GIB)               # B3's head after the first queue (E1: 24.5)
SIBLING_AVAILABLE = int(42.2 * _GIB)
HEAD_ARENA = int(33.0 * _GIB)                   # E1's card: about 33.0 GiB held beyond the copy
PATCHED = 1261                                  # the distill LoRA's patched weights
BACKUP_SENTENCE = (
    "the change needs {needed} GiB beside the 30.5 GiB copy of this file and the 30.5 GiB "
    "host copy of its patched weights (ComfyUI's backup) the Worker already holds, room for "
    "the 0.7 GiB of LoRA files the new stack reads, the weights that backup does not hold and "
    "the floor.")


class _HostTensor:
    def __init__(self, nbytes: int, device: str = "cpu"):
        self.nbytes, self.device = nbytes, types.SimpleNamespace(type=device)

    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1


def _backup(total: int = BACKUP_BYTES, keys: int = PATCHED, device: str = "cpu") -> dict:
    share, rest = divmod(total, keys)
    return {f"w{i}": types.SimpleNamespace(weight=_HostTensor(share + (rest if i == 0 else 0), device))
            for i in range(keys)}


def _b3_store(stack=((LORA, 1.0),), **changes) -> ms.ModelStore:
    """B3's Worker state: base plus the distill LoRA held, placed whole, the backup on the host."""
    store = ms.ModelStore()
    store.lora_low_rss = False
    store.slab_weights = False
    model = types.SimpleNamespace(model_loaded_weight_memory=BF16_BYTES, model_lowvram=False)
    base = types.SimpleNamespace(model=model, backup=_backup())
    active = types.SimpleNamespace(patches={f"w{i}": [("patch",)] for i in range(PATCHED)})
    store.current = StoredModel(
        base_key=(BASE, ()), request_key=(BASE, (), tuple(stack), "bf16"), base_patcher=base,
        active_patcher=active, quant_kind="bf16", precision_evidence=LivePrecisionEvidence("bf16", "bf16"),
        family="wan_animate2", artifact_identity={}, lora_sig=tuple(stack),
        residency_rung=cq.RUNG_EXPLICIT_STOCK)
    for key, value in changes.items():
        target, attr = key.split("__")
        setattr({"store": store, "stored": store.current, "base": base, "model": model,
                 "active": active}[target], attr, value)
    return store


@pytest.fixture
def quote(monkeypatch, tmp_path):
    """worker_capacity.quote on one rank for a stack, against the head's B3 reading."""
    checkpoint = tmp_path / BASE
    with open(checkpoint, "wb") as handle:
        handle.truncate(BF16_BYTES)
    monkeypatch.setattr(ms, "resolve_model_path", lambda _kind, _name: str(checkpoint))
    monkeypatch.setattr(ms, "_slab_capable_path", lambda _path: False)
    monkeypatch.setattr(ms, "_to_comfy_model_options", lambda options: dict(options or {}))
    monkeypatch.setattr(worker_capacity, "_pinned_staging", lambda: False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda *_a: HEAD_ARENA)
    (tmp_path / "loras").mkdir()
    for name, size in LORA_FILES.items():
        with open(tmp_path / "loras" / name, "wb") as handle:
            handle.truncate(size)
    monkeypatch.setattr(cq, "default_resolve_lora_path", lambda name: str(tmp_path / "loras" / name))

    def run(store, stack, available=HEAD_AVAILABLE) -> cq.RankPrice:
        monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: available)
        worker = types.SimpleNamespace(rank=0, world=2, topology={}, _setup_key=("ready",),
                                       store=store, uma_reserve_gb=0.0)
        envelope = worker_capacity.quote(worker, {"specs": [
            {"unet_name": BASE, "options": {}, "loras": [{"name": n, "strength": s} for n, s in stack],
             "slot": "cond"}], "rescue_consent": None})
        return cq.RankPrice.from_row(envelope["quotes"][0])

    return run


def _sibling(verdict: str = "fits") -> cq.RankPrice:
    fit = capacity_fit.StockFit(True, True, BF16_BYTES, SIBLING_AVAILABLE, credit_bytes=BF16_BYTES)
    return cq.RankPrice(
        verdict=verdict, rung=cq.RUNG_EXPLICIT_STOCK, priced=True, measured=dict(fit.measured()),
        would_load=cq.WOULD_LOAD_SWAP, checkpoint_bytes=BF16_BYTES, slot="cond",
        holds_resident={"cond": "stock", "uncond": None}, resident_names={"cond": BASE, "uncond": None},
        resident_bytes={"cond": BF16_BYTES}, arena_bytes=HEAD_ARENA, reason="fits")


def _fleet(head: cq.RankPrice) -> list[RankAnswer]:
    return [RankAnswer(0, "host0", (head,), box=box_names.HEAD),
            RankAnswer(1, "host1", (_sibling(),), box=box_names.SIBLING)]


@pytest.fixture(autouse=True)
def _no_card_store(monkeypatch):
    from dgx_monarch.nodes import old_model_clear

    monkeypatch.setattr(old_model_clear, "card_possible", lambda _name: True)


# B3's strength change: admitted, no card.

def test_b3_strength_change_is_admitted_with_the_backup_credited(quote, caplog):
    head = quote(_b3_store(), [(LORA, 0.8)])
    assert (head.would_load, head.verdict) == ("swap", "fits")
    assert head.measured["backup_bytes"] == BACKUP_BYTES
    assert head.measured["stack_bytes"] == LORA_FILES[LORA]
    # The credit is the file's copy plus the backup, net of the LoRA file the swap reads.
    assert head.measured["credit_bytes"] == BF16_BYTES + BACKUP_BYTES - LORA_FILES[LORA]
    # 69.1 GiB less the file's copy and the backup, plus the 0.7 GiB LoRA file: 8.7 GiB.
    assert head.measured["required_gib"] == 8.7 and head.measured["headroom_gib"] == 15.6
    with caplog.at_level(logging.INFO, logger=capacity_agreement.log.name):
        decide(_fleet(head), 2, unet_name=BASE, keep=frozenset({BASE}))   # raises nothing
    assert "cross-rank capacity admitted: the head=cond:fits/explicit_stock@15.6GiB" in caplog.text


@pytest.mark.parametrize(("stack", "why"), [
    ([(LORA, 0.8)], "a strength change"),
    ([(LORA, 1.0), (LORA, 0.5)], "the same file twice"),
    ([], "the held LoRA removed"),
])
def test_a_stack_of_the_held_files_is_credited_the_backup(quote, stack, why):
    head = quote(_b3_store(), stack)
    assert head.measured.get("backup_bytes") == BACKUP_BYTES, why


def test_a_reorder_is_credited_the_backup(quote):
    store = _b3_store(stack=((LORA, 1.0), (NARROW, 0.5)))
    head = quote(store, [(NARROW, 0.5), (LORA, 1.0)])
    assert head.measured.get("backup_bytes") == BACKUP_BYTES and head.verdict == "fits"


def test_the_credited_sentence_names_the_backup_when_the_box_is_still_short(quote):
    head = quote(_b3_store(), [(LORA, 0.8)], available=int(6.0 * _GIB))
    assert head.verdict == "refuse"
    with pytest.raises(capacity_agreement.FleetCapacityError) as raised:
        decide(_fleet(head), 2, unet_name=BASE, keep=frozenset({BASE}))
    text = str(raised.value)
    assert ("The head priced a LoRA change on wan_animate_2_bf16.safetensors: "
            + BACKUP_SENTENCE.format(needed=8.7)) in text
    assert "Only 6.0 GiB of unified memory is available, 2.7 GiB short." in text
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY


def test_a_clear_is_weighed_against_the_price_a_cleared_box_pays(quote):
    """credit_bytes carries both credits, so the card's need is still 69.1 GiB, the
    fresh load a cleared box makes, and the new stack's files are not counted twice."""
    head = quote(_b3_store(), [(LORA, 0.8)], available=int(8.0 * _GIB))
    assert head.verdict == "refuse"
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide(_fleet(head), 2, unet_name=BASE, keep=frozenset({BASE}))
    offer = old_model.offer_of(raised.value)
    assert offer is not None and offer.same_file
    (box,) = offer.boxes
    assert round(box.required_bytes / _GIB, 1) == 69.1
    # Beyond the copy and the backup the price credited, net of the files it charged.
    assert box.left_bytes == HEAD_ARENA - BACKUP_BYTES + LORA_FILES[LORA]
    assert old_model.card_numbers(offer) == "needs 69.1 GiB, 8.0 GiB available on the head"


def test_the_new_stacks_own_files_stay_charged_beside_the_backup(quote):
    """Review finding 2: every in-place change reads the new stack's LoRA files into
    anonymous memory (the pread backend), so a stack of three is charged all three."""
    stack = ((LORA, 1.0), (NARROW, 1.0), (SCAIL, 1.0))
    files = sum(LORA_FILES.values())                    # 2.97 GiB
    head = quote(_b3_store(stack=stack), [(LORA, 0.5), (NARROW, 0.5), (SCAIL, 0.5)])
    assert head.measured["stack_bytes"] == files
    assert head.measured["required_bytes"] == (
        capacity_fit.stock_required_bytes(BF16_BYTES) - BF16_BYTES - BACKUP_BYTES + files)
    assert head.measured["required_gib"] == 11.0
    # A removal to the bare base reads no file and is charged none.
    bare = quote(_b3_store(), [])
    assert bare.measured["stack_bytes"] == 0 and bare.measured["required_gib"] == 8.1


def test_a_stack_whose_files_cannot_be_sized_gets_no_backup_credit(quote, monkeypatch):
    monkeypatch.setattr(cq, "default_resolve_lora_path", lambda name: "/nonexistent/" + name)
    head = quote(_b3_store(), [(LORA, 0.8)])
    assert "backup_bytes" not in head.measured and head.measured["required_gib"] == 38.6


# A LoRA the held stack lacks keeps #146's price and path.

def test_a_lora_the_held_stack_lacks_keeps_146s_price_and_card(quote):
    head = quote(_b3_store(), [(NARROW, 1.0)])
    assert "backup_bytes" not in head.measured
    assert head.measured["credit_bytes"] == BF16_BYTES and head.measured["required_gib"] == 38.6
    assert head.measured["headroom_gib"] == -14.3
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide(_fleet(head), 2, unet_name=BASE, keep=frozenset({BASE}))
    assert old_model.offer_of(raised.value) is not None
    assert ("the change needs 38.6 GiB beside the 30.5 GiB copy of this file the Worker already "
            "holds, room for a host copy of the weights its LoRA stack changes and the floor."
            ) in str(raised.value)


def test_a_lora_the_backup_does_not_cover_still_reloads(rig, monkeypatch):
    """#146's path: the store drops the copy and loads base and the new stack fresh."""
    store, _calls = rig
    store.lora_low_rss = False

    def build(self, base, stack):
        keys = ["w1", "w2", "w3"] if stack and stack[0]["name"] == NARROW else ["w1", "w2"]
        return types.SimpleNamespace(model=base.model, patches={key: [("patch",)] for key in keys})

    monkeypatch.setattr(ms.ModelStore, "_build_active", build)
    store.ensure(BASE, {}, [{"name": LORA, "strength": 1.0}])
    held = store.current
    held.base_patcher.model.model_loaded_weight_memory = BF16_BYTES
    held.base_patcher.backup = {"w1": object(), "w2": object()}
    _patcher, transition = store.ensure(BASE, {}, [{"name": LORA, "strength": 0.8}])
    assert transition == "hot-swap" and store.current is held           # covered: in place
    _patcher, transition = store.ensure(BASE, {}, [{"name": NARROW, "strength": 1.0}])
    assert transition == "load" and store.current is not held           # uncovered: reload


# What the Worker reports, and only for the measured state.

@pytest.mark.parametrize(("changes", "why"), [
    ({"store__lora_low_rss": True}, "the low-RSS bake frees the backup"),
    ({"store__uncond": object()}, "the other slot holds a model"),
    ({"stored__slab": object()}, "a slab resident"),
    ({"stored__residency_rung": cq.RUNG_COMFY_MANAGED}, "comfy-managed"),
    ({"base___dgxm_fsdp": True}, "an FSDP shard"),
    ({"stored__base_key": (BASE, (("weight_dtype", "bfloat16"),))}, "a loader cast is unmeasured"),
    ({"stored__quant_kind": "int8"}, "a quantized swap unloads every model first"),
    ({"model__model_loaded_weight_memory": 0}, "not placed on the GPU"),
    ({"model__model_lowvram": True}, "placed in part"),
    ({"stored__lora_sig": ()}, "a bare base holds no stack"),
    ({"active__patches": {"w0": [], "not_backed": []}}, "a patched weight the backup lacks"),
    ({"base__backup": None}, "no backup dict"),
    ({"base__backup": _backup(device="cuda")}, "a backup on the GPU is no host copy"),
])
def test_the_worker_reports_the_backup_only_for_the_measured_state(changes, why):
    store = _b3_store(**changes)
    assert worker_capacity._backup_bytes(None, store, store.current, "cond") == 0, why


def test_the_worker_reports_the_host_bytes_of_b3s_backup():
    """The state every gate case above changes one thing of; an FSDP launch reports none."""
    store = _b3_store()
    worker = types.SimpleNamespace(topology={})
    assert worker_capacity._backup_bytes(worker, store, store.current, "cond") == BACKUP_BYTES
    worker.topology = {"fsdp": True}
    assert worker_capacity._backup_bytes(worker, store, store.current, "cond") == 0


def test_a_backup_read_for_another_resident_is_dropped():
    """Bound to the request key the facts were read from, as #142 binds the GPU copy."""
    keys: dict = {}
    backups: dict = {}
    store = _b3_store()
    facts = worker_capacity._resident_facts(None, store, keys, backups)
    snapshot = {"cond": {"request_key": "another key", "residency": "stock"}, "uncond": None}
    out = worker_capacity._with_resident_bytes(snapshot, facts, keys, backups)
    assert "backup_bytes" not in out["cond"]


# A resident ComfyUI holds only partly: its drop copies its device weights to the host
# before it frees them, so the box dips first and the swap credit must not count it.

@pytest.mark.parametrize(("lowvram", "credited"), [(True, False), (False, True)])
def test_a_partly_loaded_resident_carries_no_swap_credit(lowvram, credited):
    from dataclasses import replace

    from dgx_monarch import swap_credit
    from test_capacity_old_model import _answer, _stock

    store = _b3_store(model__model_lowvram=lowvram)
    keys: dict = {}
    backups: dict = {}
    partials: dict = {}
    facts = worker_capacity._resident_facts(None, store, keys, backups, partials)
    assert partials == {"cond": lowvram, "uncond": False}
    assert "resident_partial" not in facts, "facts are RankPrice fields; the flag rides the snapshot"
    snapshot = {"cond": {"request_key": keys["cond"], "residency": "stock"}, "uncond": None}
    out = worker_capacity._with_resident_bytes(snapshot, facts, keys, backups, partials)
    assert out["cond"]["resident_partial"] is lowvram
    held = {**out["cond"], "resident_bytes": int(16.9 * _GIB), "resident_name": "flux-2-klein.safetensors"}
    fit = capacity_lora_bake.credit_stock_swap(
        capacity_fit.StockFit(False, True, 1 * _GIB, 200 * _GIB), held, {"would_load": "swap"})
    assert bool(getattr(fit, "credit_bytes", 0)) is credited
    assert ("dropped" in fit.measured()) is credited
    row = _stock("fits", size_gib=1.0, avail_gib=200.0, resident="flux-2-klein.safetensors",
                 word="stock", resident_gib=16.9, slot="cond")
    row = replace(row, measured=dict(fit.measured()), would_load="swap")
    assert bool(swap_credit.from_answers([_answer(0, row)], "cond")) is credited


@pytest.mark.parametrize(("lowvram", "credited"), [(True, False), (False, True)])
def test_the_workers_own_quote_gives_a_partly_loaded_resident_no_credit(quote, lowvram, credited):
    """End to end through ``worker_capacity.quote``: another file in the slot, so the load is a swap."""
    other = "flux-2-klein.safetensors"
    store = _b3_store(stack=(), model__model_lowvram=lowvram, base__backup={},
                      stored__base_key=(other, ()), stored__request_key=(other, (), (), "bf16"))
    head = quote(store, [], available=int(100 * _GIB))
    assert head.would_load == "swap" and head.resident_names["cond"] == other
    assert (head.measured.get("credit_bytes", 0) > 0) is credited
    assert (head.measured.get("dropped") == other) is credited


def test_one_question_with_both_loads_gives_each_rank_the_rows_of_the_slots_it_holds(quote):
    """What the driver's dual-model credit leans on (``swap_credit.growth``): the real quote
    answers a two-spec question with a row a slot, each with its load and its file's size, and
    under dm-cfg2 a rank sends only the row of the one slot it holds."""
    from dgx_monarch import swap_credit

    other = "flux-2-klein.safetensors"
    store = _b3_store(stack=(), base__backup={}, stored__base_key=(other, ()),
                      stored__request_key=(other, (), (), "bf16"))
    quote(store, [], available=int(100 * _GIB))     # the fixture's box reading
    specs = [{"unet_name": BASE, "options": {}, "loras": [], "slot": slot} for slot in ("cond", "uncond")]

    def ask(topology: dict, rank: int = 0) -> list[RankAnswer]:
        worker = types.SimpleNamespace(rank=rank, world=2, topology=topology, _setup_key=("ready",),
                                       store=store, uma_reserve_gb=0.0)
        rows = worker_capacity.quote(worker, {"specs": specs, "rescue_consent": None})["quotes"]
        return [RankAnswer(rank, "host0", tuple(cq.RankPrice.from_row(row) for row in rows),
                           box=box_names.HEAD)]

    every = ask({"ulysses": 2})
    assert [(row.slot, row.would_load, row.checkpoint_bytes) for row in every[0].quotes] == [
        ("cond", "swap", BF16_BYTES), ("uncond", "fresh", BF16_BYTES)]
    assert swap_credit.from_answers(every, "cond").dropped == other
    assert swap_credit.growth(every, "uncond") == BF16_BYTES, "a fresh load into the empty slot"
    one = ask({"cfg": 2})
    assert [row.slot for row in one[0].quotes] == ["cond"], "rank 0 holds the cond model only"
    assert swap_credit.from_answers(one, "cond").dropped == other
    assert swap_credit.growth(one, "uncond") == 0, "the second DiT loads on the other rank"
    assert [row.slot for row in ask({"cfg": 2}, rank=1)[0].quotes] == ["uncond"]


# The credit itself.

def _held(backup: object = BACKUP_BYTES, stack=((LORA, 1.0),), residency: str = "stock") -> dict:
    return {"residency": residency, "request_key": repr((BASE, (), tuple(stack), "bf16")),
            "backup_bytes": backup, "swaps_in_place": True}


@pytest.mark.parametrize(("held", "asked", "expected", "why"), [
    (_held(), ((LORA, 0.8),), BACKUP_BYTES, "a strength change"),
    (_held(), (), BACKUP_BYTES, "the stack removed"),
    (_held(), ((NARROW, 1.0),), 0, "a file the held stack lacks"),
    (_held(), ((LORA, 1.0), (NARROW, 1.0)), 0, "a file added to the held stack"),
    (_held(stack=()), ((LORA, 1.0),), 0, "a bare base holds no backup to reuse"),
    (_held(backup=True), ((LORA, 0.8),), 0, "a flag is not a byte count"),
    (_held(backup=None), ((LORA, 0.8),), 0, "an older Worker reports none"),
    (_held(residency="slab"), ((LORA, 0.8),), 0, "a slab resident"),
    (_held(backup=40 * _GIB), ((LORA, 0.8),), int(capacity_fit.STOCK_PLACEMENT_RATIO * BF16_BYTES),
     "never more than the host copy the stock price charges"),
])
def test_the_backup_credit_rule(held, asked, expected, why):
    key = repr((BASE, (), tuple(asked), "bf16"))
    assert capacity_lora_bake.held_backup_credit(held, key, BF16_BYTES) == expected, why


def test_a_cast_resident_gets_no_backup_credit():
    fit = capacity_fit.StockFit(False, True, BF16_BYTES, HEAD_AVAILABLE, cast_bytes=BF16_BYTES // 2)
    held = {**_held(), "resident_bytes": BF16_BYTES // 2}
    key = repr((BASE, (), ((LORA, 0.8),), "bf16"))
    credited = capacity_lora_bake.credit_hot_swap(fit, held, key, {"would_load": "swap"}, False)
    assert credited.backup_bytes == 0 and credited.credit_bytes == BF16_BYTES // 2


def test_the_rescue_card_words_name_the_backup():
    credit = BF16_BYTES + BACKUP_BYTES - LORA_FILES[LORA]
    assert capacity_fit.stock_terms(30.5, credit, BACKUP_BYTES, LORA_FILES[LORA]) == (
        "beside the 30.5 GiB copy of this file and the 30.5 GiB host copy of its patched "
        "weights (ComfyUI's backup) the Worker already holds, room for the 0.7 GiB of LoRA "
        "files the new stack reads, the weights that backup does not hold and the floor,")


# The docs cite what the code computes (review of #148, findings 1 and 6).

ROOT = Path(__file__).resolve().parents[1]


def _doc(name: str) -> str:
    return " ".join((ROOT / name).read_text().split())


def test_the_docs_cite_the_backup_the_headers_prove_and_the_price_it_gives(quote):
    """Finding 1: the figures come from the header sum, not a rounded reading."""
    head = quote(_b3_store(), [(LORA, 0.8)])
    price = f"about {head.measured['required_gib']} GiB"
    design, trouble, changelog = (_doc(name) for name in
                                  ("docs/DESIGN.md", "docs/TROUBLESHOOTING.md", "CHANGELOG.md"))
    assert f"{price} there" in design and "32,785,804,928 bytes" in design
    assert f"{price} for the BF16 Wan-Animate 2 strength change" in trouble
    assert f"needs {price} where it needed 38.6 GiB" in changelog
    for text in (design, trouble, changelog):
        assert "29.7 GiB" not in text


@pytest.mark.parametrize(("doc", "present", "absent", "why"), [
    ("docs/DESIGN.md", "An in-place swap runs unpriced on the Worker, so for such a change the "
     "agreement's price is the only guard", "", "(a) why the credit must hold with no wall"),
    ("docs/TROUBLESHOOTING.md", "the price cannot tell which weights a new LoRA patches",
     "which a new LoRA cannot reuse", "(b) E3 q3 swapped a new LoRA in place"),
    ("docs/TROUBLESHOOTING.md", "which costs more than the change. A new strength",
     "strength change that needed 38.6. Then the card reads", "(c) the card follows its own case"),
    ("docs/CONCEPTS.md", "or one of them removed, reuses that host copy", "", "(d) removal too"),
])
def test_the_backup_credit_docs_agree_with_section_5_9(doc, present, absent, why):
    """Finding 6, (a) to (d)."""
    text = _doc(doc)
    assert present in text, why
    assert not absent or absent not in text, why
