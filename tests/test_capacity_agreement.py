"""The pre-load cross-rank capacity agreement and the loader's batch check.

Every rank prices the load before any rank allocates, the driver collects the
rows and raises, and the leanest answer governs. CPU only: no CUDA, no ComfyUI
beyond tiny stubs, and no fleet; torch imports through the actor package and
runs nothing. The worker endpoint's body is driven directly, the driver's
collection through a fake handle that records the order it sends and awaits,
and the two call sites through their callers: seam A is ``agree_load`` in the
loader node, seam B is ``agree_sample`` in ``mesh_setup.dispatch_collective_sample``.

Four rules these tests hold:

* a rank that did not answer must never read as a stock-load capacity refusal,
  or the gate's cross-residency leg writes a capacity verdict for a liveness
  fault;
* two ranks can never produce equal descriptors, so the rescue arm does not
  compare them: it carries the leanest rank's descriptor whole, and the consent
  id, built from rank-invariant inputs, matches on every rank;
* a refusal or a timeout evicts nothing and latches nothing;
* the loader's batch check answers class P above the footprint card, so a graph
  that is both over budget and batch indivisible is never offered a rescue.
"""
from __future__ import annotations

import json
import struct
import sys
import types
from dataclasses import replace

import pytest

from dgx_monarch import capacity_agreement, consent_pending, consent_store, mesh_safety
from dgx_monarch.actor import capacity_quote, worker_capacity
from dgx_monarch.actor.capacity_quote import RankPrice
from dgx_monarch.capacity_agreement import (
    FleetCapacityError,
    FleetQuoteUnavailableError,
    RankAnswer,
    decide,
)
from dgx_monarch.capacity_fit import SlabFit, StockFit
from dgx_monarch.nodes import render_validation
from dgx_monarch.refusal import parse_leading_refusal_tag

_GIB = 2 ** 30
CHECKPOINT_BYTES = 60 * _GIB


def _fit(avail_gib: float, size_bytes: int = CHECKPOINT_BYTES) -> StockFit:
    avail = int(avail_gib * _GIB)
    fit = StockFit(True, True, size_bytes, avail)
    return StockFit(fit.required_bytes <= avail, True, size_bytes, avail)


def _price(verdict: str, *, rung: str = "stock_fits", avail_gib: float = 200.0,
           slot: str = "cond", size_bytes: int = CHECKPOINT_BYTES,
           blocker: tuple[str, str] | None = None,
           descriptor: object = None) -> RankPrice:
    fit = _fit(avail_gib, size_bytes)
    return RankPrice(
        verdict=verdict, rung=rung, priced=True, measured=dict(fit.measured()),
        blocker=blocker, descriptor=descriptor, would_load="fresh",
        checkpoint_bytes=size_bytes, slot=slot, shmem_bytes=0,
        reason=f"{verdict} on the {rung} rung")


def _slab(avail_gib: float, size_bytes: int = CHECKPOINT_BYTES,
          floor_bytes: int = 4 * _GIB) -> SlabFit:
    avail = int(avail_gib * _GIB)
    return SlabFit(size_bytes + floor_bytes <= avail, True, size_bytes, avail,
                   floor_bytes)


def _slab_price(verdict: str, *, avail_gib: float = 53.8, rung: str = "explicit",
                slot: str = "cond") -> RankPrice:
    """A slab rung's row: ``slab_measured`` and no stock block at all."""
    fit = _slab(avail_gib)
    return RankPrice(
        verdict=verdict, rung=rung, use_slab=True, priced=True,
        slab_measured=dict(fit.measured()), would_load="fresh",
        checkpoint_bytes=fit.size_bytes, slot=slot, shmem_bytes=0,
        reason="slab_weights=on was requested")


def _answer(rank: int, *prices: RankPrice, host: str = "") -> RankAnswer:
    return RankAnswer(rank, host or f"box{rank}", tuple(prices), elapsed_s=0.03)


def _fits(rank: int, **kwargs) -> RankAnswer:
    return _answer(rank, _price("fits", **kwargs))


def _refuses(rank: int, avail_gib: float = 75.2, **kwargs) -> RankAnswer:
    return _answer(rank, _price(
        "refuse", avail_gib=avail_gib,
        blocker=("lora_low_rss is off", "turn lora_low_rss on, or free memory"),
        **kwargs))


def test_a_fleet_that_all_fits_is_admitted_and_recorded(caplog):
    with caplog.at_level("INFO"):
        decide([_fits(0), _fits(1)], 2)
    assert "cross-rank capacity admitted" in caplog.text
    assert "rank 1=cond:fits/stock_fits" in caplog.text


def test_world_one_is_a_no_op():
    decide([_refuses(0)], 1)


def test_an_unpriced_fleet_admits_and_a_no_claim_never_vetoes(caplog):
    unpriced = _answer(0, RankPrice(verdict="unpriced", rung="vouched_auto"))
    no_claim = _answer(1, RankPrice(verdict="no_claim", rung="",
                                    reason="this rank has not completed setup"))
    with caplog.at_level("INFO"):
        decide([unpriced, no_claim], 2)
    assert "unpriced" in caplog.text and "no_claim" in caplog.text


def test_one_short_rank_refuses_the_whole_fleet_and_names_every_rank():
    with pytest.raises(FleetCapacityError) as raised:
        decide([_fits(0), _refuses(1)], 2, unet_name="flux2.safetensors")
    text = str(raised.value)
    assert "1 of 2 cannot" in text
    assert "rank 1 (host box1) priced a stock load of flux2.safetensors" in text
    assert "rank 0 (host box0) took the stock_fits rung" in text
    assert "nothing was loaded on any rank" in text
    assert "What would fit: turn lora_low_rss on" in text


def test_two_short_ranks_are_reported_worst_headroom_first():
    with pytest.raises(FleetCapacityError) as raised:
        decide([_refuses(0, avail_gib=90.0), _refuses(1, avail_gib=40.0)], 2)
    text = str(raised.value)
    assert "2 of 2 cannot" in text
    assert text.index("rank 1 (host box1)") < text.index("rank 0 (host box0)")


def test_a_silent_rank_refuses_and_says_what_the_others_did():
    silent = RankAnswer(1, "box1", silent="did not answer in time")
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide([_fits(0), silent], 2)
    text = str(raised.value)
    assert "rank 1 (host box1) did not answer in time" in text
    assert "rank 0 answered in 0.03 s" in text
    assert "the fleet was not torn down" in text


def test_a_short_rank_inventory_refuses():
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide([_fits(0)], 2)
    assert "returned 1 answers from ranks [0]" in str(raised.value)


def test_two_rows_with_the_same_rank_id_refuse():
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide([_fits(0), _fits(0)], 2)
    assert "short or duplicated" in str(raised.value)


def test_ranks_that_disagree_about_the_bytes_refuse_with_the_parity_sentence():
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide([_fits(0), _fits(1, size_bytes=CHECKPOINT_BYTES + 1)], 2)
    text = str(raised.value)
    assert "the worker hosts hold different files under the same checkpoint name" in text
    assert "Stage the same artifact on every host" in text


def test_a_refusal_outranks_a_rescue_on_another_rank():
    rescue = _answer(0, _price("rescue", avail_gib=80.0))
    with pytest.raises(FleetCapacityError):
        decide([rescue, _refuses(1)], 2)


@pytest.mark.parametrize("quiet", [
    RankPrice(verdict="no_claim", rung="", reason="this rank has not completed setup"),
    RankPrice(verdict="unpriced", rung="vouched_auto"),
])
def test_a_rank_that_stated_no_size_is_not_a_byte_disagreement(quiet):
    """Zero is the absence of a content-derived value, not a smaller artifact.

    Comparing it would veto on rung asymmetry alone: one rank that made no
    claim or answered unpriced, beside one rank priced stock and fitting.
    """
    decide([_fits(0), _answer(1, quiet)], 2)


def test_a_rank_that_made_no_claim_is_named_rather_than_given_an_empty_rung():
    quiet = RankPrice(verdict="no_claim", rung="", reason="no probe ran here")
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, quiet), _refuses(1)], 2)
    assert "rank 0 (host box0) made no claim" in str(raised.value)


def test_a_wall_priced_refusal_carries_its_numbers_and_sorts_worst_first():
    """The managed and FSDP rungs answer in their own wall's block.

    A block with byte keys alone falls through to the reason fallback, so the
    card prints no numbers, and the sort reads it as zero headroom, which puts
    the leanest rank anywhere but first.
    """
    from dgx_monarch.capacity_fit import preflight_wall_measured

    def wall(rung, avail_gib, required_gib):
        size, avail = 40 * _GIB, int(avail_gib * _GIB)
        required = int(required_gib * _GIB)
        return RankPrice(
            verdict="refuse", rung=rung, priced=True, would_load="fresh",
            measured=dict(preflight_wall_measured(size, avail, required, False)),
            checkpoint_bytes=size, slot="cond",
            reason=f"{rung} cannot hold this checkpoint")

    answers = [RankAnswer(0, "box0", (wall("comfy_managed", 30.0, 44.0),)),
               RankAnswer(1, "box1", (wall("stock_fits", 12.0, 44.0),))]
    with pytest.raises(FleetCapacityError) as raised:
        decide(answers, 2, unet_name="flux2.safetensors")
    text = str(raised.value)
    assert "44.0 GiB" in text and "32.0 GiB short" in text
    # Worst headroom first: rank 1 is 32 GiB short and rank 0 is 14 GiB short.
    assert text.index("rank 1") < text.index("rank 0")
    assert "plus the host copy it is placed from" not in text


def test_a_slab_shortfall_carries_its_own_numbers_and_a_remedy():
    """A slab rung answers in ``slab_measured`` and carries no blocker pair.

    A card reading only ``measured`` prints a slab refusal with no required
    figure, no available figure and no remedy: an unpriced answer under a
    priced verdict.
    """
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(1, _slab_price("refuse")), _fits(0)], 2,
               unet_name="flux2.safetensors")
    text = str(raised.value)
    assert "rank 1 (host box1) priced a slab load of flux2.safetensors" in text
    assert "the load needs 64.0 GiB" in text
    assert "a 60.0 GiB file plus a 4.0 GiB floor for the rest of the box" in text
    assert "53.8 GiB of unified memory is available, 10.2 GiB short" in text
    assert "What would fit: 10.2 GiB more free memory" in text
    assert "the host copy it is placed from" not in text


def test_slab_rows_are_ordered_by_the_headroom_they_measured():
    """A headroom of zero for every slab row leaves the ordering undone."""
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, _slab_price("refuse", avail_gib=60.0)),
                _answer(1, _slab_price("refuse", avail_gib=40.0))], 2)
    text = str(raised.value)
    assert text.index("rank 1 (host box1)") < text.index("rank 0 (host box0)")


def test_a_price_taken_over_a_resident_says_so_on_the_card():
    """The quote prices each slot as held, so the card must say so.

    A slot holding one checkpoint while another is requested prices the second
    against memory the first still occupies, so an operator reading a shortfall
    needs to know the figure was taken with that resident in place.
    """
    held = replace(_price("refuse", avail_gib=75.2), would_load="swap",
                   holds_resident={"cond": "slab", "uncond": None})
    with pytest.raises(FleetCapacityError) as raised:
        decide([_fits(0), _answer(1, held)], 2)
    text = str(raised.value)
    assert "rank 1 (host box1) holds a slab resident in the cond slot" in text
    assert "this load reads as a swap" in text
    assert "does not model the drop" in text


def test_a_fresh_price_holds_nothing_and_says_nothing():
    with pytest.raises(FleetCapacityError) as raised:
        decide([_fits(0), _refuses(1)], 2)
    assert "does not model the drop" not in str(raised.value)


def test_a_byte_disagreement_outranks_a_shortfall():
    """An identity fault leaves the price unknown, so it answers first."""
    with pytest.raises(FleetQuoteUnavailableError):
        decide([_refuses(0), _fits(1, size_bytes=CHECKPOINT_BYTES + 1)], 2)


def test_silence_is_never_a_stock_load_capacity_error():
    """A rank that did not answer has not said it cannot load anything.

    ``mesh_safety.is_stock_load_capacity_error`` matches the class name or the
    failure text, and the gate's cross-residency leg (``nodes/gate_cross_mode.py``)
    turns any member of that family into a CAPACITY verdict, so a liveness
    fault inside it would write a capacity verdict for a box that never spoke.
    """
    arms = {
        "silent": [_fits(0), RankAnswer(1, "box1", silent="did not answer in time")],
        "short": [_fits(0)],
        "duplicate": [_fits(0), _fits(0)],
        "bytes": [_fits(0), _fits(1, size_bytes=CHECKPOINT_BYTES + 1)],
    }
    for name, answers in arms.items():
        with pytest.raises(FleetQuoteUnavailableError) as raised:
            decide(answers, 2)
        assert not mesh_safety.is_stock_load_capacity_error(raised.value), name
        assert "StockLoadCapacityError" not in str(raised.value), name
        tag = parse_leading_refusal_tag(str(raised.value))
        assert tag is not None and tag.refusal_class.value == "C", name
        assert tag.guard == "cross_rank_capacity" and not tag.waivable, name

    with pytest.raises(FleetCapacityError) as shortfall:
        decide([_fits(0), _refuses(1)], 2)
    assert mesh_safety.is_stock_load_capacity_error(shortfall.value)
    tag = parse_leading_refusal_tag(str(shortfall.value))
    assert tag is not None and tag.guard == "cross_rank_capacity"


@pytest.fixture
def checkpoint(tmp_path):
    """A real, minimal safetensors file: the ladder reads its header."""
    path = tmp_path / "diffusion_models" / "flux2_bf16.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"blocks.0.w": {
        "dtype": "BF16", "shape": [8], "data_offsets": [0, 16]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 16)
    return path


@pytest.fixture(autouse=True)
def _rig(monkeypatch, tmp_path):
    consent_pending.clear_all()
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "memo.json"))
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda folder, name: (  # type: ignore[attr-defined]
        str(tmp_path / folder / name) if (tmp_path / folder / name).exists() else None)
    fake.get_output_directory = lambda: str(output)  # type: ignore[attr-defined]
    fake.get_filename_list = lambda _kind: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    yield
    consent_pending.clear_all()


def _rescue_answer(rank: int, checkpoint, avail_gib: float) -> RankAnswer:
    """A rescue row built from this rank's own StockFit, as the worker builds it."""
    from dgx_monarch.actor import store_residency

    fit = _fit(avail_gib)
    descriptor = store_residency._descriptor(
        unet_name="flux2_bf16.safetensors", path=str(checkpoint), fit=fit,
        model_options={}, file_identity=lambda _path: "the same bytes on both boxes",
        family_hint="flux2", node_options={}, lora_stack=[])
    row = RankPrice(
        verdict="rescue", rung="stock_fits", priced=True,
        measured=dict(fit.measured()), descriptor=descriptor.public(),
        would_load="fresh", checkpoint_bytes=fit.size_bytes, slot="cond")
    return RankAnswer(rank, f"box{rank}", (RankPrice.from_row(row.to_row()),),
                      elapsed_s=0.02)


def test_rescue_descriptors_agree_across_ranks(checkpoint):
    """Real per-rank values, not synthetic identical rows.

    ``_descriptor`` stamps the host into ``measured`` and embeds this box's
    availability in ``human_reason``, so two ranks can never produce equal
    descriptors. Agreeing on the whole descriptor would refuse every world-2
    rescue.
    """
    head = _rescue_answer(0, checkpoint, 75.2)
    second = _rescue_answer(1, checkpoint, 107.0)
    assert head.quotes[0].descriptor != second.quotes[0].descriptor
    assert (head.quotes[0].descriptor["consent_id"]
            == second.quotes[0].descriptor["consent_id"])

    from dgx_monarch.consent_descriptor import SlabResidencyRescueOffer

    with pytest.raises(SlabResidencyRescueOffer) as raised:
        decide([head, second], 2)
    text = str(raised.value)
    assert "capacity rescue available for flux2_bf16.safetensors" in text
    assert "rank 0 (host box0) is the leanest" in text
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.guard == "stock_load_preflight" and tag.waivable


def test_the_driver_raised_rescue_reaches_the_panel(checkpoint, monkeypatch):
    handle = _FakeHandle(2, {})
    monkeypatch.setattr(
        capacity_agreement, "collect",
        lambda *_a, **_k: [_rescue_answer(0, checkpoint, 75.2),
                           _rescue_answer(1, checkpoint, 107.0)])
    from dgx_monarch.consent_descriptor import SlabResidencyRescueOffer

    with pytest.raises(SlabResidencyRescueOffer):
        capacity_agreement.agree_or_refuse(handle, {})
    assert consent_pending.pending_count() == 1


def test_a_failed_registration_never_swallows_the_refusal(checkpoint, monkeypatch):
    from dgx_monarch import consent_observe
    from dgx_monarch.consent_descriptor import SlabResidencyRescueOffer

    handle = _FakeHandle(2, {})
    monkeypatch.setattr(
        capacity_agreement, "collect",
        lambda *_a, **_k: [_rescue_answer(0, checkpoint, 75.2), _fits(1)])
    monkeypatch.setattr(consent_observe, "observe_refusal", _explode)
    with pytest.raises(SlabResidencyRescueOffer):
        capacity_agreement.agree_or_refuse(handle, {})


def _explode(*_args, **_kwargs):
    raise RuntimeError("the card registry is down")


class _FakeHandle:
    """A fleet that records the order it is sent to and awaited on."""

    def __init__(self, world, replies, *, gpus_per_host=1, labels=("hosts", "gpus"),
                 config=None):
        self.world = world
        self.gpus_per_host = gpus_per_host
        self.config = config
        self.defunct = False
        self.setup_cleanup_state = None
        self.setup_key = ("ready",)
        self.replies = replies
        self.events: list[tuple[str, int]] = []
        self.timeouts: list[float] = []
        self.workers = types.SimpleNamespace(
            extent=types.SimpleNamespace(labels=list(labels)), slice=self._slice)

    def _slice(self, **coords):
        index = coords.get("hosts", 0) * self.gpus_per_host + coords["gpus"]
        return types.SimpleNamespace(capacity_quote=types.SimpleNamespace(
            call_one=lambda request, rank=index: self._send(rank, request)))

    def _send(self, rank, request):
        self.events.append(("send", rank))
        self.requests = getattr(self, "requests", [])
        self.requests.append(request)
        return rank

    def _await_or_evict(self, future, timeout_s):
        self.events.append(("await", future))
        self.timeouts.append(timeout_s)
        reply = self.replies[future]
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _reply(rank, *rows, host=None):
    return {"host": host or f"box{rank}", "rank": rank, "world": 2, "setup": True,
            "quotes": [row.to_row() for row in rows]}


def test_collect_checks_require_endpoint_before_any_future_is_sent():
    handle = _FakeHandle(2, {})
    handle.defunct = True
    with pytest.raises(RuntimeError, match="worker fleet is defunct"):
        capacity_agreement.collect(handle, {}, 2)
    assert handle.events == []


def test_collect_sends_every_future_before_it_awaits_any():
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")), 1: _reply(1, _price("fits"))})
    answers = capacity_agreement.collect(handle, {}, 2)
    assert [kind for kind, _rank in handle.events] == ["send", "send", "await", "await"]
    assert [answer.rank for answer in answers] == [0, 1]
    assert [answer.host for answer in answers] == ["box0", "box1"]


def test_the_whole_fleet_shares_one_deadline():
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")), 1: _reply(1, _price("fits"))})
    capacity_agreement.collect(handle, {}, 2, deadline_s=5.0)
    assert handle.timeouts[0] <= 5.0
    assert handle.timeouts[1] <= handle.timeouts[0]
    assert sum(handle.timeouts) < 2 * 5.0


@pytest.mark.parametrize("failure", [TimeoutError("no answer"), RuntimeError("boom")])
def test_a_rank_that_cannot_answer_refuses_without_evicting_the_fleet(failure):
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")), 1: failure})
    answers = capacity_agreement.collect(handle, {}, 2)
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide(answers, 2)
    assert "rank 1" in str(raised.value)
    assert handle.defunct is False
    assert handle.setup_cleanup_state is None


def test_an_answer_with_no_rows_reads_as_silence():
    """``decide`` counts answers, not rows.

    A rank that replied with an empty inventory priced nothing, so reading it
    as live admits the whole fleet on the other rank's word.
    """
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")), 1: _reply(1)})
    answers = capacity_agreement.collect(handle, {}, 2)
    assert answers[1].silent
    with pytest.raises(FleetQuoteUnavailableError):
        decide(answers, 2)


def test_a_reply_cannot_answer_for_a_rank_the_driver_did_not_ask():
    """The driver knows which rank it sent to; the reply does not get to say.

    A row that renames itself makes one rank's answer stand for another's, and
    the short-inventory rule that would catch it never fires.
    """
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")),
                             1: _reply(0, _price("fits"), host="box1")})
    answers = capacity_agreement.collect(handle, {}, 2)
    assert [answer.rank for answer in answers] == [0, 1]
    assert "rank 0" in answers[1].silent
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide(answers, 2)
    assert "rank 1" in str(raised.value)


def test_each_rank_reports_the_wait_it_cost_and_not_the_fleet_total(monkeypatch):
    """The futures are all sent at once but read in order, so a prompt rank
    read after a slow one would be reported as the slow one."""
    ticks = iter([0.0, 0.0, 5.0, 5.0, 5.1])
    monkeypatch.setattr(capacity_agreement.time, "monotonic", lambda: next(ticks))
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")), 1: _reply(1, _price("fits"))})
    answers = capacity_agreement.collect(handle, {}, 2)
    assert answers[0].elapsed_s == pytest.approx(5.0)
    assert answers[1].elapsed_s == pytest.approx(0.1)


def test_an_unreadable_quote_row_reads_as_silence():
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")),
                             1: {"host": "box1", "rank": 1, "quotes": [{"verdict": "fits"}]}})
    answers = capacity_agreement.collect(handle, {}, 2)
    assert answers[1].silent
    with pytest.raises(FleetQuoteUnavailableError):
        decide(answers, 2)


def test_the_remote_failure_text_never_reaches_the_card():
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")),
                             1: RuntimeError("a whole remote traceback here")})
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide(capacity_agreement.collect(handle, {}, 2), 2)
    assert "remote traceback" not in str(raised.value)


def test_a_world_one_fleet_is_never_asked():
    handle = _FakeHandle(1, {})
    capacity_agreement.agree_or_refuse(handle, {})
    assert handle.events == []


def _cluster(*names):
    """A cluster config carrying nothing but the driver's host list."""
    return types.SimpleNamespace(
        hosts=tuple(types.SimpleNamespace(name=name) for name in names))


def test_a_silent_rank_is_named_from_the_driver_host_list():
    """On the 2026-09-03 silent-rank leg a frozen rank refused as "rank 1 (host
    unnamed)" while the driver held the host list. The rank said nothing, so
    the name is marked as the driver's own rather than the box's."""
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")), 1: TimeoutError("frozen")},
                         config=_cluster("head", "second"))
    answers = capacity_agreement.collect(handle, {}, 2)
    assert answers[1].configured_host == "second"
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide(answers, 2)
    text = str(raised.value)
    assert "rank 1 (host second as configured) did not answer in time" in text
    assert "host unnamed" not in text


def test_a_local_fleet_names_the_driver_box_for_a_silent_rank():
    """Every rank of a local fleet runs in the driver's own process tree."""
    handle = _FakeHandle(2, {0: _reply(0, _price("fits")), 1: TimeoutError("frozen")})
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide(capacity_agreement.collect(handle, {}, 2), 2)
    assert "rank 1 (host this box as configured)" in str(raised.value)
    assert "host unnamed" not in str(raised.value)


def test_the_rank_own_host_name_still_wins_over_the_config():
    """The box speaking about itself outranks the driver's name for it."""
    handle = _FakeHandle(2, {0: _reply(0, _price("fits"), host="head.local"),
                             1: _reply(1, _price("refuse"), host="second.local")},
                         config=_cluster("head", "second"))
    answers = capacity_agreement.collect(handle, {}, 2)
    assert [answer.where for answer in answers] == [
        "rank 0 (host head.local)", "rank 1 (host second.local)"]


def test_ranks_map_to_hosts_in_blocks_of_gpus_per_host():
    """Rank i sits on host i // gpus, the order collect slices the mesh in."""
    handle = _FakeHandle(4, dict.fromkeys(range(4), TimeoutError("frozen")),
                         gpus_per_host=2, config=_cluster("head", "second"))
    answers = capacity_agreement.collect(handle, {}, 4)
    assert [answer.configured_host for answer in answers] == [
        "head", "head", "second", "second"]


def test_a_host_list_shorter_than_the_world_names_no_box_it_cannot():
    handle = _FakeHandle(2, {0: TimeoutError("x"), 1: TimeoutError("x")},
                         config=_cluster("head"))
    answers = capacity_agreement.collect(handle, {}, 2)
    assert answers[1].where == "rank 1 (host unnamed)"


_HELD_KEY = "('m.safetensors', (), (), 'bf16')"


def _resident_row(monkeypatch, *, avail_bytes: int | None,
                  rung: str = "vouched_auto") -> RankPrice:
    """The row a warm slot returns, built by ``capacity_quote.price`` itself.

    A hand-built row could pin a shape the worker does not emit: a reuse
    carries the box's own reading in the half the card reads. With
    ``avail_bytes`` None this builds the one reuse the card must answer for, a
    probe that could not read MemAvailable, and that row states no headroom.
    The path does not exist, so the file is zero bytes; a reuse prices nothing
    either way.
    """
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: avail_bytes)
    return capacity_quote.price(
        path="/models/m.safetensors", unet_name="m.safetensors",
        model_options={}, slab_weights="auto", slab_capable_path=True,
        lora_low_rss=True, fsdp_launch=False, blocked_reason="",
        authoritative_slab_retry=False, memoized_family=lambda _path: None,
        vouched_families=frozenset(), request_key=_HELD_KEY,
        store_snapshot={"cond": {"residency": "slab", "residency_rung": rung,
                                 "request_key": _HELD_KEY},
                        "uncond": None, "cleanup_failed_slots": []})


def test_a_reuse_that_read_nothing_is_not_printed_as_zero_headroom(
        caplog, monkeypatch):
    """The 2026-09-03 silent-rank leg printed "cond:fits/vouched_auto@0.0GiB" on
    a warm fleet, which reads as the leanest box rather than the one already
    holding the file.

    The worker fills that block from the box's own reading. A box with no
    readable MemAvailable fills nothing, and zero is wrong here too: a reuse
    places nothing, so it is the roomiest answer.
    """
    row = _resident_row(monkeypatch, avail_bytes=None)
    assert (row.priced, row.would_load) == (False, "reuse")
    assert (row.measured, row.slab_measured) == (None, None)
    with caplog.at_level("INFO"):
        decide([_answer(0, row), _fits(1)], 2)
    assert "rank 0=cond:fits/vouched_auto@resident" in caplog.text
    assert "@0.0GiB" not in caplog.text


def test_a_reuse_that_read_its_box_prints_that_reading(caplog, monkeypatch):
    """The word never replaces a number the rank sent.

    A warm rank that read its own MemAvailable reports it, and the card prints
    it, so the operator reading the fleet line sees how much room that box has.
    """
    row = _resident_row(monkeypatch, avail_bytes=48 * _GIB)
    assert row.slab_measured["headroom_gib"] == 48.0
    with caplog.at_level("INFO"):
        decide([_answer(0, row), _fits(1)], 2)
    assert "rank 0=cond:fits/vouched_auto@48.0GiB" in caplog.text
    assert "@resident" not in caplog.text


def test_a_priced_row_whose_probe_read_no_availability_says_so(caplog):
    """A slab probe that could not read MemAvailable states no headroom, and
    a zero there would claim a measurement nobody took."""
    blind = RankPrice(
        verdict="fits", rung="explicit", use_slab=True, priced=True,
        slab_measured=dict(SlabFit(True, True, CHECKPOINT_BYTES, None).measured()),
        would_load="fresh", checkpoint_bytes=CHECKPOINT_BYTES, reason="slab")
    with caplog.at_level("INFO"):
        decide([_answer(0, blind), _fits(1)], 2)
    assert "rank 0=cond:fits/explicit@headroom unstated" in caplog.text
    assert "@0.0GiB" not in caplog.text


def test_a_measured_headroom_still_prints_its_own_number(caplog):
    with caplog.at_level("INFO"):
        decide([_fits(0, avail_gib=200.0), _fits(1, avail_gib=200.0)], 2)
    assert "GiB" in caplog.text
    assert "@resident" not in caplog.text
    assert "headroom unstated" not in caplog.text


def test_a_real_zero_headroom_still_prints_zero(caplog):
    """The word only replaces a headroom no row stated, never a measured one."""
    exact = _fit(0.0)
    row = RankPrice(verdict="fits", rung="stock_fits", priced=True,
                    measured={**dict(exact.measured()), "headroom_gib": 0.0},
                    would_load="fresh", checkpoint_bytes=CHECKPOINT_BYTES,
                    reason="exactly fits")
    with caplog.at_level("INFO"):
        decide([_answer(0, row), _fits(1)], 2)
    assert "rank 0=cond:fits/stock_fits@0.0GiB" in caplog.text


class _Store:
    def __init__(self, **overrides):
        self.slab_weights = overrides.get("slab_weights", "auto")
        self.lora_low_rss = overrides.get("lora_low_rss", True)
        self.slab_blocked_reason = ""
        self.family_override = None
        self.current = None
        self.uncond = None
        self._snapshot = overrides.get("snapshot") or {
            "cond": None, "uncond": None, "cleanup_failed_slots": [],
            "failed_load_cleanup_pending": False,
            "retained_failed_load_slabs": 2,
        }

    def snapshot(self):
        return dict(self._snapshot)


def _worker(**overrides):
    spec = {"rank": 0, "world": 2, "topology": {}, "_setup_key": ("ready",),
            "store": _Store()}
    spec.update(overrides)
    return types.SimpleNamespace(**spec)


@pytest.fixture
def priced(monkeypatch):
    """Stand in for the ladder so the test reads the endpoint's own plumbing."""
    seen: list[dict] = []

    def fake_price(**kwargs):
        seen.append(kwargs)
        return _price("fits", slot=kwargs["slot"])

    monkeypatch.setattr(capacity_quote, "price", fake_price)
    monkeypatch.setattr(worker_capacity, "_pinned_staging", lambda: True)
    from dgx_monarch.actor import model_store

    monkeypatch.setattr(model_store, "resolve_model_path",
                        lambda _kind, name: f"/models/{name}")
    monkeypatch.setattr(model_store, "_slab_capable_path", lambda _path: True)
    monkeypatch.setattr(model_store, "_to_comfy_model_options", lambda options: dict(options or {}))
    return seen


def _specs(*slots):
    return {"specs": [{"unet_name": f"{slot}.safetensors", "options": {},
                       "loras": [], "slot": slot} for slot in slots],
            "rescue_consent": None}


def test_the_endpoint_answers_rows_and_never_raises(priced):
    envelope = worker_capacity.quote(_worker(), _specs("cond"))
    assert envelope["rank"] == 0 and envelope["world"] == 2 and envelope["setup"]
    assert envelope["host"]
    assert [row["slot"] for row in envelope["quotes"]] == ["cond"]
    assert priced[0]["pinned_staging"] is True


def test_every_row_carries_shmem_the_residents_and_the_retained_slabs(priced):
    envelope = worker_capacity.quote(_worker(), _specs("cond", "uncond"))
    for row in envelope["quotes"]:
        assert row["shmem_bytes"] is None or isinstance(row["shmem_bytes"], int)
        assert set(row["holds_resident"]) == {"cond", "uncond"}
        assert row["retained_failed_load_slabs"] == 2
        assert row["failed_load_cleanup_pending"] is False


def test_a_rank_prices_its_slots_in_load_order_against_one_running_budget(priced):
    worker_capacity.quote(_worker(), _specs("cond", "uncond"))
    assert [call["slot"] for call in priced] == ["cond", "uncond"]
    assert priced[0]["budget_bytes_already_charged"] == 0
    assert priced[1]["budget_bytes_already_charged"] > 0
    assert len({id(call["store_snapshot"]) for call in priced}) == 1


def test_a_rank_that_has_not_completed_setup_makes_no_claim():
    envelope = worker_capacity.quote(_worker(_setup_key=None), _specs("cond"))
    assert envelope["setup"] is False
    assert [row["verdict"] for row in envelope["quotes"]] == ["no_claim"]
    assert "has not completed setup" in envelope["quotes"][0]["reason"]


def test_the_endpoint_drives_the_shipped_ladder_end_to_end(checkpoint, monkeypatch):
    """No stub between the endpoint and the ladder, so the kwargs are proven.

    Every other endpoint test here replaces ``price`` to read the plumbing on
    its own; this one runs the shipped ladder, so a renamed or dropped argument
    fails here before a fleet meets it.
    """
    from dgx_monarch.actor import model_store
    from dgx_monarch.adapters import fsdp as adapters_fsdp

    monkeypatch.setattr(model_store, "resolve_model_path",
                        lambda _kind, name: str(checkpoint.parent / name))
    # The shipped ladder prices unified memory; a CI box with no CUDA device
    # must read as integrated or the stock probe skips and the row answers no_claim.
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(adapters_fsdp, "gpu_is_integrated", lambda: True)
    envelope = worker_capacity.quote(_worker(), {
        "specs": [{"unet_name": checkpoint.name, "options": {}, "loras": [],
                   "slot": "cond"}],
        "rescue_consent": None})

    row = envelope["quotes"][0]
    assert row["verdict"] != "no_claim", row["reason"]
    assert row["verdict"] in capacity_quote.VERDICTS
    assert row["rung"] in capacity_quote.RUNGS
    assert row["slot"] == "cond"
    assert set(row["holds_resident"]) == {"cond", "uncond"}
    assert row["retained_failed_load_slabs"] == 2
    assert RankPrice.from_row(row).verdict == row["verdict"]


def test_a_ladder_that_cannot_price_makes_no_claim_rather_than_raising(monkeypatch):
    monkeypatch.setattr(capacity_quote, "price", _explode)
    envelope = worker_capacity.quote(_worker(), _specs("cond"))
    assert [row["verdict"] for row in envelope["quotes"]] == ["no_claim"]


@pytest.mark.parametrize("rank,held", [(0, "cond"), (1, "uncond")])
def test_under_dm_cfg2_a_rank_prices_only_the_checkpoint_it_holds(priced, rank, held):
    """Charging both slots on both ranks spends a budget no rank spends.

    Under dm-cfg2 cfg rank 0 holds the conditional checkpoint and cfg rank 1
    the unconditional one, so pricing the pair on each rank would refuse a
    shape that runs.
    """
    worker = _worker(rank=rank, topology={"cfg": 2, "ulysses": 1, "ring": 1, "dp": 1})
    envelope = worker_capacity.quote(worker, _specs("cond", "uncond"))
    assert [row["slot"] for row in envelope["quotes"]] == [held]


def test_outside_dm_cfg2_every_rank_prices_every_slot(priced):
    worker = _worker(rank=1, topology={"ulysses": 2, "cfg": 1, "ring": 1, "dp": 1})
    envelope = worker_capacity.quote(worker, _specs("cond", "uncond"))
    assert [row["slot"] for row in envelope["quotes"]] == ["cond", "uncond"]


def test_the_endpoint_keeps_the_shmem_the_ladder_already_read(priced, monkeypatch):
    """One /proc/meminfo parser on this path, and one read per row.

    A second parser in the endpoint would overwrite the figure the ladder took
    through ``capacity_memory`` with a later read, and two parsers would have
    to stay correct.
    """
    monkeypatch.setattr(capacity_quote, "price", lambda **kwargs: replace(
        _price("fits", slot=kwargs["slot"]), shmem_bytes=4321))
    envelope = worker_capacity.quote(_worker(), _specs("cond"))
    assert envelope["quotes"][0]["shmem_bytes"] == 4321


def test_the_operator_reserve_reaches_the_worker_wall(priced):
    """``driver_footprint`` charges it, so a worker wall that skips it admits
    on the rank what the driver refuses on the head."""
    worker = _worker()
    worker.uma_reserve_gb = 10.0
    worker_capacity.quote(worker, _specs("cond"))
    assert priced[0]["reserve_bytes"] == 10 * _GIB


@pytest.mark.parametrize("value", ["junk", None, -1.0])
def test_a_reserve_this_rank_cannot_read_charges_nothing(priced, value):
    worker = _worker()
    worker.uma_reserve_gb = value
    worker_capacity.quote(worker, _specs("cond"))
    assert priced[0]["reserve_bytes"] == 0


def test_a_slab_row_charges_the_next_slot_the_slab_price():
    """The row picks the block, not the block order.

    A consented rescue and the rescue seam both carry two blocks, and a slab
    row charged the stock transient spends the host copy
    (``capacity_fit.STOCK_PLACEMENT_RATIO`` times the file) that nobody places,
    which can refuse a second slot for bytes that never arrive.
    """
    stock, slab = _fit(200.0), _slab(200.0)
    row = RankPrice(
        verdict="fits", rung="consented_rescue", use_slab=True, priced=True,
        measured=dict(stock.measured()), slab_measured=dict(slab.measured()),
        would_load="fresh", checkpoint_bytes=slab.size_bytes)
    assert worker_capacity._charged_bytes(row) == slab.required_bytes
    assert worker_capacity._charged_bytes(
        replace(row, use_slab=False)) == stock.required_bytes
    assert worker_capacity._charged_bytes(replace(row, would_load="reuse")) == 0
    # A slab row whose probe was skipped falls back to the file, never to the
    # stock block beside it.
    assert worker_capacity._charged_bytes(
        replace(row, slab_measured=None)) == slab.size_bytes


def test_an_unpriceable_slot_charges_its_own_file_to_the_next(monkeypatch, tmp_path):
    """A slot this rank could not price still takes room on the box.

    The row makes no claim, which admits, but the next slot must be priced
    against a box holding this file, or a pair no rank can hold is admitted
    because the first of the two was the one that broke.
    """
    from dgx_monarch.actor import model_store

    (tmp_path / "cond.safetensors").write_bytes(b"x" * 3072)
    monkeypatch.setattr(model_store, "resolve_model_path",
                        lambda _kind, name: str(tmp_path / name))
    monkeypatch.setattr(model_store, "_slab_capable_path", lambda _path: True)
    monkeypatch.setattr(model_store, "_to_comfy_model_options",
                        lambda options: dict(options or {}))
    monkeypatch.setattr(worker_capacity, "_pinned_staging", lambda: True)
    seen: list[dict] = []

    def price(**kwargs):
        seen.append(kwargs)
        if kwargs["slot"] == "cond":
            raise RuntimeError("this rank cannot price the conditional slot")
        return _price("fits", slot="uncond")

    monkeypatch.setattr(capacity_quote, "price", price)
    envelope = worker_capacity.quote(_worker(), _specs("cond", "uncond"))
    assert envelope["quotes"][0]["verdict"] == "no_claim"
    assert seen[1]["budget_bytes_already_charged"] == 3072


def test_the_endpoint_hands_the_ladder_the_whole_reuse_key(priced):
    """``ensure`` reuses on the name, the options, the LoRA stack and the quant
    kind, so the quote must ask about the same key or a LoRA change reads as a
    reuse and prices nothing."""
    snapshot = {"cond": {"residency": "stock", "quant": "bf16",
                         "request_key": "('cond.safetensors', (), (), 'bf16')"},
                "uncond": None, "cleanup_failed_slots": [],
                "retained_failed_load_slabs": 0,
                "failed_load_cleanup_pending": False}
    worker = _worker(store=_Store(snapshot=snapshot))
    worker_capacity.quote(worker, _specs("cond"))
    assert priced[0]["request_key"] == "('cond.safetensors', (), (), 'bf16')"


def _mesh(handle, preset="uly2", **overrides):
    spec = {"topology_preset": preset, "handle": handle, "worker_args": {},
            "attention": "TORCH_FLASH", "sync_ulysses": True, "world": 2}
    spec.update(overrides)
    return types.SimpleNamespace(**spec)


@pytest.fixture
def loader_rig(monkeypatch):
    """Stub every answer around the eager load, so the test reads only the order."""
    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import (
        consent_projection,
        loader_preflight,
        loaders,
        render_preflight,
        render_session,
    )

    events: list[str] = []
    monkeypatch.setattr(loader_preflight, "preflight_upstream_gated_artifact",
                        lambda *_a: None)
    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology",
                        lambda *_a: events.append("comfy-managed"))
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        lambda *_a: events.append("footprint"))
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *_a, **_k: None)
    monkeypatch.setattr(consent_projection, "project_for_loader", lambda *_a, **_k: None)
    monkeypatch.setattr(render_preflight, "preflight_sol_sequence_parallel",
                        lambda *_a: None)
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda *_a, **_k: None)

    import contextlib

    monkeypatch.setattr(render_session, "mutation_render_session",
                        lambda _handle: contextlib.nullcontext())

    handle = types.SimpleNamespace(
        world=2, defunct=False,
        call_all=lambda *_a, **_k: events.append("call_all") or [])
    monkeypatch.setattr(loaders, "ensure_live", lambda *_a, **_k: handle)
    monkeypatch.setattr(capacity_agreement, "agree_load",
                        lambda *args, **_k: events.append(f"agree:{args[-1]}"))
    return types.SimpleNamespace(events=events, handle=handle)


def test_seam_a_agrees_the_fleet_before_any_rank_is_asked_to_load(loader_rig):
    from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader

    DGXMonarchUNETLoader().load(_mesh(loader_rig.handle), "flux2.safetensors")
    assert loader_rig.events[-2:] == ["agree:cond", "call_all"]


def test_the_uncond_loader_quotes_the_uncond_slot(loader_rig):
    from dgx_monarch.nodes.loaders import DGXMonarchUncondUNETLoader

    DGXMonarchUncondUNETLoader().load(_mesh(loader_rig.handle), "flux2.safetensors")
    assert "agree:uncond" in loader_rig.events


def test_seam_a_refuses_before_call_all_load_model(loader_rig, monkeypatch):
    monkeypatch.setattr(capacity_agreement, "agree_load", _explode)
    from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader

    with pytest.raises(RuntimeError):
        DGXMonarchUNETLoader().load(_mesh(loader_rig.handle), "flux2.safetensors")
    assert "call_all" not in loader_rig.events


@pytest.mark.parametrize("slab_weights,residency", [(False, "stock"), (True, "slab")])
def test_seam_b_admits_a_resident_checkpoint_no_load_would_replace(
        checkpoint, monkeypatch, slab_weights, residency):
    """A repeat render of the bytes the slot already holds must not refuse.

    ``ensure`` short circuits on a resident it can reuse, so no load runs and
    nothing is placed. Pricing a fresh load against the availability that
    resident already consumed refuses on every rank, and two refusing ranks
    refuse the render before dispatch. The 2026-09-03 shmem record
    (docs/VALIDATION.md) has a settled 60 GiB slab resident rendering at
    11.61 GiB available, so this is the ordinary case.
    """
    from dgx_monarch.actor import model_store

    size = checkpoint.stat().st_size
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: size // 2)
    monkeypatch.setattr(capacity_quote, "comfy_managed_active", lambda: False)
    monkeypatch.setattr(model_store, "resolve_model_path",
                        lambda _kind, name: str(checkpoint.parent / name))
    snapshot = {
        # The key ``ModelStore.make_keys`` builds for this request: the name,
        # the normalized options, the LoRA signature and the resident's quant.
        "cond": {"request_key": repr((checkpoint.name, (), (), None)),
                 "quant": None, "residency": residency},
        "uncond": None, "cleanup_failed_slots": [],
        "failed_load_cleanup_pending": False, "retained_failed_load_slabs": 0,
    }
    store = _Store(slab_weights=slab_weights, snapshot=snapshot)
    replies = {rank: worker_capacity.quote(
        _worker(rank=rank, store=store), {
            "specs": [{"unet_name": checkpoint.name, "options": {}, "loras": [],
                       "slot": "cond"}],
            "rescue_consent": None}) for rank in (0, 1)}
    for rank, envelope in replies.items():
        row = envelope["quotes"][0]
        assert row["would_load"] == "reuse", (rank, row["reason"])
        assert row["holds_resident"]["cond"] == residency
        assert row["verdict"] == "fits", (rank, row["reason"])

    capacity_agreement.agree_sample(_FakeHandle(2, replies), {
        "model": {"unet_name": checkpoint.name, "options": {}, "loras": []}})


def test_seam_b_runs_above_the_artifact_parity_proof(monkeypatch):
    from dgx_monarch import mesh_setup

    events: list[str] = []
    handle = types.SimpleNamespace(
        world=2, defunct=False, setup_cleanup_state=None,
        _verify_request_artifacts=lambda *_a, **_k: events.append("parity"),
        workers=types.SimpleNamespace(sample=types.SimpleNamespace(
            call=lambda *_a, **_k: events.append("sample"))))
    monkeypatch.setattr(capacity_agreement, "agree_sample",
                        lambda *_a, **_k: events.append("agree"))
    monkeypatch.setattr(mesh_setup, "dispatch_sample",
                        lambda *_a, **_k: events.append("dispatch"))

    mesh_setup.dispatch_collective_sample(handle, {"model": {}}, None, None)
    assert events == ["agree", "parity", "dispatch"]


def test_seam_b_refuses_before_the_sample_is_leased_and_evicts_nothing(monkeypatch):
    from dgx_monarch import mesh_setup

    events: list[str] = []
    handle = types.SimpleNamespace(
        world=2, defunct=False, setup_cleanup_state=None,
        _verify_request_artifacts=lambda *_a, **_k: events.append("parity"),
        workers=types.SimpleNamespace(sample=types.SimpleNamespace(
            call=lambda *_a, **_k: events.append("sample"))))
    monkeypatch.setattr(capacity_agreement, "agree_sample", _explode)

    with pytest.raises(RuntimeError):
        mesh_setup.dispatch_collective_sample(handle, {"model": {}}, None, None)
    assert events == []
    assert handle.defunct is False and handle.setup_cleanup_state is None


def test_seam_b_prices_both_model_slots_of_a_dual_model_request(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(capacity_agreement, "agree_or_refuse",
                        lambda _handle, request, **_k: sent.append(request))
    capacity_agreement.agree_sample(types.SimpleNamespace(world=2), {
        "model": {"unet_name": "cond.safetensors", "options": {}, "loras": []},
        "uncond_model": {"unet_name": "uncond.safetensors", "options": {}, "loras": []},
    })
    assert [spec["slot"] for spec in sent[0]["specs"]] == ["cond", "uncond"]
    assert sent[0]["specs"][1]["unet_name"] == "uncond.safetensors"


def test_seam_b_carries_a_granted_rescue_consent_into_the_quote(monkeypatch):
    """Without the grant the driver raises the same rescue card again after
    every click."""
    from dgx_monarch.mesh_residency import CAPACITY_RESCUE_CONSENT_CAPABILITY

    sent: list[dict] = []
    monkeypatch.setattr(capacity_agreement, "agree_or_refuse",
                        lambda _handle, request, **_k: sent.append(request))
    capacity_agreement.agree_sample(types.SimpleNamespace(world=2), {
        "model": {"unet_name": "cond.safetensors", "options": {}, "loras": []},
        "_dgxm_normal_residency_grant": {
            "capability": CAPACITY_RESCUE_CONSENT_CAPABILITY,
            "consent": {"id": "abc"}},
    })
    assert sent[0]["rescue_consent"] == {"id": "abc"}


def _dp2_graph(batch, latent_class="EmptyLatentImage", link=True):
    return {
        "8": {"class_type": latent_class,
              "inputs": {"width": 512, "height": 512, "batch_size": batch}},
        "9": {"class_type": "DGXMonarchKSampler",
              "inputs": {"model": ["2", 0],
                         "latent_image": ["8", 0] if link else 3}},
    }


def _dp2_mesh(preset="dp2", world=2, defunct=False):
    return types.SimpleNamespace(
        topology_preset=preset,
        handle=types.SimpleNamespace(world=world, defunct=defunct))


def test_an_indivisible_literal_batch_refuses_class_p_at_the_loader():
    with pytest.raises(ValueError) as raised:
        render_validation.preflight_graph_batch_divides_dp(_dp2_mesh(), _dp2_graph(1))
    text = str(raised.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class.value == "P"
    assert tag.guard is None and not tag.waivable
    assert "latent batch 1 is not divisible by 2" in text
    assert "Increase batch to a multiple of 2" in text


def test_a_divisible_literal_batch_makes_no_claim():
    render_validation.preflight_graph_batch_divides_dp(_dp2_mesh(), _dp2_graph(2))


@pytest.mark.parametrize("mesh,prompt", [
    (_dp2_mesh(preset="auto"), _dp2_graph(1)),
    (_dp2_mesh(world=0), _dp2_graph(1)),
    (_dp2_mesh(defunct=True), _dp2_graph(1)),
    (_dp2_mesh(preset="uly2"), _dp2_graph(1)),
    (_dp2_mesh(), _dp2_graph(1, latent_class="SomeCustomLatentSource")),
    (_dp2_mesh(), _dp2_graph(1, link=False)),
    (_dp2_mesh(), _dp2_graph("one")),
    (_dp2_mesh(), None),
    (types.SimpleNamespace(topology_preset="dp2", handle=None), _dp2_graph(1)),
])
def test_uncertain_evidence_makes_no_claim_at_the_loader(mesh, prompt):
    """No claim costs nothing here: the render submit still holds the real
    tensor and answers the same question with it."""
    render_validation.preflight_graph_batch_divides_dp(mesh, prompt)


def test_both_indivisible_batch_sites_print_one_message():
    """docs/TROUBLESHOOTING.md #16 promises one message at both places, so the
    loader-node answer and the render-submit answer must be the same string for
    the same shape, not a shared remedy under two lead-ins."""
    from dgx_monarch.topology import Topology

    model = types.SimpleNamespace(mesh=types.SimpleNamespace(topology_preset="dp2"))
    samples = types.SimpleNamespace(shape=(1, 4, 64, 64))
    with pytest.raises(ValueError) as raised:
        render_validation.validate_render_topology(
            model, Topology(dp=2, world=2), samples)
    submit = str(raised.value)
    tag = parse_leading_refusal_tag(submit)
    assert tag is not None and tag.refusal_class.value == "P"
    with pytest.raises(ValueError) as loader_raised:
        render_validation.preflight_graph_batch_divides_dp(_dp2_mesh(), _dp2_graph(1))
    assert str(loader_raised.value) == submit


def test_the_batch_answer_lands_above_the_footprint_card(loader_rig, monkeypatch):
    """A graph that is both over budget and batch indivisible must not be
    offered a rescue card, clicked, loaded, and only then refused by a policy
    no click can clear (docs/TROUBLESHOOTING.md #16)."""
    from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader

    loader_rig.handle.world = 2
    with pytest.raises(ValueError, match="not divisible by 2"):
        DGXMonarchUNETLoader().load(
            _mesh(loader_rig.handle, preset="dp2"), "flux2.safetensors",
            prompt=_dp2_graph(1))
    assert loader_rig.events == []
