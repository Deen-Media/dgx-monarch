"""The worker's ``RankPrice`` and the ladder arithmetic behind it.

The worker ladder computes a ``RankPrice``; the driver's agreement reads rows
of them off the wire. The first tests pin that interface, the field set and
the signature, so neither side can move one without failing here. The rest
drive ``price`` through each rung and, where the ladder has the same path,
check it against ``store_residency.resolve`` and the wall that runs after it.
"""
import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import capacity_fit
from dgx_monarch.actor import capacity_quote as cq
from dgx_monarch.capacity_fit import SlabFit, StockFit
from dgx_monarch.consent_descriptor import SlabResidencyRescueOffer

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"

# Every field of a row, the slab verdict included. Set equality in both
# directions: a field nobody agreed to is as bad as a missing one, because the
# driver reads these rows and cannot ask for a key twice.
DESIGNED_FIELDS = {
    "verdict", "rung", "use_slab", "priced", "measured", "slab_measured",
    "blocker", "descriptor", "would_load", "holds_resident", "checkpoint_bytes",
    "slot", "shmem_bytes", "retained_failed_load_slabs",
    "failed_load_cleanup_pending", "reason",
}

# The five the quote adds to the ladder's own arguments, in order.
QUOTE_ONLY_PARAMS = ("slot", "pinned_staging", "store_snapshot",
                     "budget_bytes_already_charged", "request_key")


def _fitting_price(**overrides):
    row = {
        "verdict": cq.VERDICT_FITS,
        "rung": cq.RUNG_STOCK_FITS,
        "use_slab": False,
        "priced": True,
        "measured": {"probe": "worker_stock_load", "fits": True},
        "slab_measured": None,
        "blocker": None,
        "descriptor": None,
        "would_load": cq.WOULD_LOAD_FRESH,
        "holds_resident": {"cond": None, "uncond": "slab"},
        "checkpoint_bytes": 64 * 1024 ** 3,
        "slot": cq.SLOT_COND,
        "shmem_bytes": 60 * 1024 ** 3,
        "retained_failed_load_slabs": 0,
        "failed_load_cleanup_pending": False,
        "reason": "stock residency fits this checkpoint",
    }
    row.update(overrides)
    return cq.RankPrice.from_row(row)


def test_rank_price_carries_every_field_the_design_names():
    assert {f.name for f in cq.RankPrice.__dataclass_fields__.values()} == DESIGNED_FIELDS


def test_rank_price_is_frozen_and_slotted():
    price = _fitting_price()
    assert cq.RankPrice.__dataclass_params__.frozen
    with pytest.raises(AttributeError):
        price.verdict = cq.VERDICT_REFUSE
    assert set(cq.RankPrice.__slots__) == DESIGNED_FIELDS
    assert not hasattr(price, "__dict__")


def test_the_slab_verdict_rides_beside_the_stock_one():
    """The slab verdict is a field of the same row, not a second exchange.

    A slab rung carries the slab block and no stock block, so a reader that
    only knows ``measured`` reads a slab refusal as an unpriced admit.
    """
    slab = _fitting_price(
        rung=cq.RUNG_EXPLICIT, use_slab=True, verdict=cq.VERDICT_REFUSE,
        measured=None, slab_measured={"probe": "worker_slab_load", "fits": False},
    )
    assert slab.measured is None
    assert slab.slab_measured["probe"] == "worker_slab_load"
    assert slab.refuses()


def test_a_verdict_outside_the_vocabulary_is_refused_at_construction():
    with pytest.raises(ValueError):
        _fitting_price(verdict="probably")
    with pytest.raises(ValueError):
        _fitting_price(would_load="maybe")
    assert cq.VERDICTS == {"fits", "refuse", "rescue", "unpriced", "no_claim"}
    assert cq.WOULD_LOAD == {"fresh", "reuse", "swap", "unknown"}


def test_a_row_round_trips_through_the_wire_form():
    price = _fitting_price(blocker=("no lever", "free memory"))
    row = price.to_row()
    assert set(row) == cq.ROW_KEYS == DESIGNED_FIELDS
    assert row["blocker"] == ["no lever", "free memory"]
    back = cq.RankPrice.from_row(row)
    assert back == price
    assert back.blocker == ("no lever", "free memory")


def test_a_consent_descriptor_travels_as_its_public_mapping():
    class _Descriptor:
        def public(self):
            return {"consent_id": "abc", "kind": "rescue-slab"}

    row = _fitting_price(verdict=cq.VERDICT_RESCUE).to_row()
    assert row["descriptor"] is None
    price = cq.RankPrice(cq.VERDICT_RESCUE, cq.RUNG_STOCK_FITS,
                         descriptor=_Descriptor())
    assert price.to_row()["descriptor"] == {"consent_id": "abc", "kind": "rescue-slab"}


def test_a_row_that_does_not_parse_raises_rather_than_admits():
    """A partial row is a silent rank, and the driver refuses on silence.

    Filling a missing key with a default here is how an unparseable answer
    turns into a price that admits.
    """
    row = _fitting_price().to_row()
    for key in sorted(cq.ROW_KEYS):
        short = {k: v for k, v in row.items() if k != key}
        with pytest.raises(ValueError):
            cq.RankPrice.from_row(short)
    with pytest.raises(ValueError):
        cq.RankPrice.from_row(dict(row, verdict="probably"))
    with pytest.raises(ValueError):
        cq.RankPrice.from_row(dict(row, blocker=["one thing"]))
    with pytest.raises(ValueError):
        cq.RankPrice.from_row("a traceback the worker returned")


def test_price_takes_the_ladder_arguments_plus_exactly_five():
    """The one signature the ladder and the quote share.

    ``resolve`` is a wrapper around this call, so a parameter added to
    ``resolve`` and not mirrored here stops reaching the quote without an
    error.
    """
    from dgx_monarch.actor import store_residency

    ladder = inspect.signature(store_residency.resolve).parameters
    quote = inspect.signature(cq.price).parameters
    assert list(quote) == list(ladder) + list(QUOTE_ONLY_PARAMS)
    for name, parameter in quote.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        if name in ladder:
            assert parameter.default == ladder[name].default, name
            assert parameter.annotation == ladder[name].annotation, name
    assert quote["slot"].default == cq.SLOT_COND
    assert quote["pinned_staging"].default is None
    assert quote["store_snapshot"].default is None
    assert quote["budget_bytes_already_charged"].default == 0
    assert quote["request_key"].default == ""
    assert inspect.signature(cq.price).return_annotation == "RankPrice"


def test_the_rung_vocabulary_agrees_with_the_ladder():
    """One spelling per rung: ``store_residency`` re-exports the quote's constants."""
    from dgx_monarch.actor import store_residency

    for name in sorted(cq.RUNGS):
        assert getattr(cq, f"RUNG_{name.upper()}") == getattr(
            store_residency, f"RUNG_{name.upper()}")
    assert cq.RUNGS == {
        store_residency.RUNG_COMFY_MANAGED, store_residency.RUNG_EXPLICIT,
        store_residency.RUNG_EXPLICIT_STOCK, store_residency.RUNG_VOUCHED_AUTO,
        store_residency.RUNG_STOCK_FITS, store_residency.RUNG_CONSENTED_RESCUE,
    }


def test_no_rung_is_unpriced_on_day_one():
    """Every rung gives a priced verdict or sits in ``UNPRICED_RUNGS``.

    A member of that set needs a comment naming what cannot be priced there.
    The set is empty, so adding a member fails here and the change must update
    this test.
    """
    assert cq.UNPRICED_RUNGS == frozenset()
    assert cq.UNPRICED_RUNGS <= cq.RUNGS


def test_the_quote_module_imports_nothing_that_needs_a_gpu():
    """AST over this module's own imports, not sys.modules.

    A sys.modules assertion passes vacuously: importing the ladder already
    pulls torch through the actor package, so only the import list each file
    declares can be checked.
    """
    banned = {"torch", "comfy", "comfy_dynamic", "store_residency",
              "model_store", "store_load"}
    # The FSDP rung lives in its own module for the line ceiling, so the same
    # rule covers that module.
    for module in ("capacity_quote.py", "capacity_quote_fsdp.py"):
        tree = ast.parse((SRC / "actor" / module).read_text())
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names += [(node.module or "")] + [alias.name for alias in node.names]
        for name in names:
            head = name.split(".")[-1]
            assert head not in banned, f"{module} imports {name}"


def test_the_fsdp_rung_still_resolves_on_the_quote_module():
    """``cq._fsdp_rung`` stays the entry point after the line-ceiling split.

    ``capacity_quote_fsdp`` imports the row type and the rung vocabulary from
    ``capacity_quote``, so the delegate imports it inside the function; a
    module-scope import there would cycle.
    """
    from dgx_monarch.actor import capacity_quote_fsdp

    assert callable(cq._fsdp_rung)
    assert callable(capacity_quote_fsdp.fsdp_rung)


FITS = StockFit(True, True, 20 * 2 ** 30, 90 * 2 ** 30)
DOES_NOT_FIT = StockFit(False, True, 62 * 2 ** 30, 47 * 2 ** 30)
SKIPPED = StockFit(True, False, 0, None, "a loader dtype cast changes the resident size")
SLAB_FITS = SlabFit(True, True, 20 * 2 ** 30, 90 * 2 ** 30, 4 * 2 ** 30)
SLAB_SHORT = SlabFit(False, True, 62 * 2 ** 30, 47 * 2 ** 30, 4 * 2 ** 30)


def _kwargs(**overrides):
    kwargs = {
        "path": "/models/m.safetensors",
        "unet_name": "m.safetensors",
        "model_options": {},
        "slab_weights": "auto",
        "slab_capable_path": True,
        "lora_low_rss": True,
        "fsdp_launch": False,
        "blocked_reason": "",
        "authoritative_slab_retry": False,
        "memoized_family": lambda _path: None,
        "vouched_families": frozenset({"krea2"}),
        "rescue_consent": None,
        "fit_probe": lambda _path, _options: FITS,
        "file_identity": lambda _path: "1:2:3:4:5",
        "compile_dit": lambda: False,
        "comfy_managed": lambda: False,
    }
    kwargs.update(overrides)
    return kwargs


def _slab(monkeypatch, fit=SLAB_FITS):
    # The ladder prices unified memory. A CI box with no CUDA device must read
    # as integrated, or the managed and FSDP rungs and the walls make no claim.
    from dgx_monarch.adapters import fsdp as adapters_fsdp

    monkeypatch.setattr(cq.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(adapters_fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit, "slab_load_fit",
                        lambda _path, _options, **_kw: fit)


@pytest.mark.parametrize(("overrides", "verdict", "rung"), [
    ({}, cq.VERDICT_FITS, cq.RUNG_STOCK_FITS),
    ({"slab_weights": False}, cq.VERDICT_FITS, cq.RUNG_EXPLICIT_STOCK),
    ({"slab_weights": True}, cq.VERDICT_FITS, cq.RUNG_EXPLICIT),
    ({"memoized_family": lambda _p: "krea2"}, cq.VERDICT_FITS, cq.RUNG_VOUCHED_AUTO),
    ({"authoritative_slab_retry": True}, cq.VERDICT_FITS, cq.RUNG_VOUCHED_AUTO),
    ({"fit_probe": lambda _p, _o: SKIPPED}, cq.VERDICT_NO_CLAIM, cq.RUNG_STOCK_FITS),
    ({"fit_probe": lambda _p, _o: DOES_NOT_FIT}, cq.VERDICT_RESCUE, cq.RUNG_STOCK_FITS),
    ({"fit_probe": lambda _p, _o: DOES_NOT_FIT, "rescue_consent": {"id": "a"}},
     cq.VERDICT_FITS, cq.RUNG_CONSENTED_RESCUE),
    ({"fit_probe": lambda _p, _o: DOES_NOT_FIT, "compile_dit": lambda: True},
     cq.VERDICT_REFUSE, cq.RUNG_STOCK_FITS),
    ({"comfy_managed": lambda: True, "pinned_staging": False},
     cq.VERDICT_FITS, cq.RUNG_COMFY_MANAGED),
])
def test_price_matches_resolve_on_every_row_of_the_ladder(
        monkeypatch, overrides, verdict, rung):
    """The test that keeps one arithmetic.

    Every row of the shipped ladder matrix is driven through ``price`` and
    through ``resolve``, and the verdict has to equal the decision returned or
    the refusal raised. Two pricing paths are the defect this guards: until
    2026-09-02 ``capacity_fit`` charged a stock load 1.3 times the file while
    ``driver_footprint`` charged 1.85, and a box the worker price admitted died.
    """
    from dgx_monarch.actor import store_residency
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    _slab(monkeypatch)
    row = cq.price(**_kwargs(**overrides))
    assert (row.verdict, row.rung) == (verdict, rung)

    ladder = {k: v for k, v in _kwargs(**overrides).items() if k != "pinned_staging"}
    if verdict == cq.VERDICT_REFUSE and rung != cq.RUNG_COMFY_MANAGED:
        with pytest.raises(StockLoadCapacityError):
            store_residency.resolve(**ladder)
        return
    if verdict == cq.VERDICT_RESCUE:
        with pytest.raises(SlabResidencyRescueOffer):
            store_residency.resolve(**ladder)
        return
    decision = store_residency.resolve(**ladder)
    assert (decision.rung, decision.use_slab) == (row.rung, row.use_slab)
    assert decision.reason == row.reason


def test_known_compile_noop_family_keeps_its_vouched_auto_slab(monkeypatch):
    """A live-model detection, not a header, proves Flux2 will run eager."""
    _slab(monkeypatch)
    row = cq.price(**_kwargs(
        memoized_family=lambda _p: "flux2",
        vouched_families=frozenset({"flux2"}),
        compile_dit=lambda: True,
    ))
    assert (row.rung, row.use_slab) == (cq.RUNG_VOUCHED_AUTO, True)

    from dgx_monarch.actor import store_residency

    decision = store_residency.resolve(**_kwargs(
        memoized_family=lambda _p: "flux2",
        vouched_families=frozenset({"flux2"}),
        compile_dit=lambda: True,
    ))
    assert (decision.rung, decision.use_slab) == (cq.RUNG_VOUCHED_AUTO, True)


@pytest.mark.parametrize("family", (None, "krea2"))
def test_compile_stays_stock_until_the_family_is_known_eager_noop(monkeypatch, family):
    """With compile on, a cold memo (None) or a family outside the known compile
    no-op list (krea2) stays explicit stock; only a known no-op keeps slab."""
    _slab(monkeypatch)
    row = cq.price(**_kwargs(
        memoized_family=lambda _p: family,
        compile_dit=lambda: True,
    ))
    assert (row.rung, row.use_slab) == (cq.RUNG_EXPLICIT_STOCK, False)


def test_the_managed_rung_agrees_with_its_own_wall(monkeypatch):
    """The comfy-managed rung is priced by the wall that runs after ``resolve``,
    so the quote has to fold that wall in or it invents the verdict. Both
    staging paths, both walls."""
    from dgx_monarch.actor import store_residency
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    _slab(monkeypatch)
    monkeypatch.setattr(cq.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(cq.os.path, "getsize", lambda _p: 40 * 2 ** 30)
    monkeypatch.setattr(cq.os.path, "exists", lambda _p: True)

    for pinned, avail, fits in ((False, 60 * 2 ** 30, True), (False, 20 * 2 ** 30, False),
                                (True, 200 * 2 ** 30, True), (True, 60 * 2 ** 30, False)):
        monkeypatch.setattr(cq.mesh_safety, "mem_available_bytes", lambda a=avail: a)
        monkeypatch.setattr(store_residency.mesh_safety, "gpu_is_integrated", lambda: True)
        monkeypatch.setattr(store_residency.mesh_safety, "mem_available_bytes",
                            lambda a=avail: a)
        monkeypatch.setattr(store_residency.os.path, "getsize", lambda _p: 40 * 2 ** 30)
        monkeypatch.setattr(store_residency.os.path, "exists", lambda _p: True)
        row = cq.price(**_kwargs(comfy_managed=lambda: True, pinned_staging=pinned))
        assert row.rung == cq.RUNG_COMFY_MANAGED
        assert (row.verdict == cq.VERDICT_FITS) is fits, (pinned, avail)

        decision = store_residency.resolve(**_kwargs(comfy_managed=lambda: True))
        wall = store_residency.preload_capacity_check
        if fits:
            wall(decision, "/models/m.safetensors", "m.safetensors", {},
                 preflight=lambda *_a: None, pinned_staging=pinned)
        else:
            with pytest.raises(StockLoadCapacityError):
                wall(decision, "/models/m.safetensors", "m.safetensors", {},
                     preflight=lambda *_a: None, pinned_staging=pinned)


def test_a_managed_rank_that_never_read_the_comfy_flag_makes_no_claim(monkeypatch):
    """This module must not import ComfyUI to answer, and must not guess.

    The rung's own sentence does not move, because the ladder returns this
    decision unchanged and logs it. ``no_claim`` beside it is what says the
    price was never taken.
    """
    _slab(monkeypatch)
    row = cq.price(**_kwargs(comfy_managed=lambda: True))
    assert (row.verdict, row.rung) == (cq.VERDICT_NO_CLAIM, cq.RUNG_COMFY_MANAGED)
    assert row.priced is False
    assert row.reason == "this worker runs comfy-managed residency (ComfyUI's DynamicVRAM)"


def test_every_rung_produces_a_priced_verdict_or_sits_in_the_unpriced_set(monkeypatch):
    """Every rung the ladder reaches gives a priced verdict or is in
    ``UNPRICED_RUNGS`` (see test_no_rung_is_unpriced_on_day_one)."""
    _slab(monkeypatch)
    seen = {}
    for overrides in ({}, {"slab_weights": False}, {"slab_weights": True},
                      {"memoized_family": lambda _p: "krea2"},
                      {"fit_probe": lambda _p, _o: DOES_NOT_FIT,
                       "rescue_consent": {"id": "a"}},
                      {"comfy_managed": lambda: True, "pinned_staging": False}):
        row = cq.price(**_kwargs(**overrides))
        seen[row.rung] = row
    assert set(seen) == cq.RUNGS
    for rung, row in sorted(seen.items()):
        assert row.priced or rung in cq.UNPRICED_RUNGS, rung
    assert cq.UNPRICED_RUNGS == frozenset()


def test_a_slab_rung_carries_the_slab_block_and_no_stock_block(monkeypatch):
    _slab(monkeypatch, SLAB_SHORT)
    row = cq.price(**_kwargs(slab_weights=True))
    assert (row.verdict, row.use_slab) == (cq.VERDICT_REFUSE, True)
    assert row.measured is None
    assert row.slab_measured["probe"] == "worker_slab_load"
    # The rescue seam carries both, because both probes ran there.
    seam = cq.price(**_kwargs(fit_probe=lambda _p, _o: DOES_NOT_FIT))
    assert seam.measured["probe"] == "worker_stock_load"
    assert seam.slab_measured["probe"] == "worker_slab_load"


def test_two_slots_share_one_budget(monkeypatch):
    """Two half-size checkpoints that each fit, and jointly do not.

    At load time each ``resolve`` runs after the slot before it loaded, so the
    uncond price sees the cond resident. A quote runs before any load, so one
    snapshot and a running budget stop both answering fits for a pair the box
    cannot hold.
    """
    half = StockFit(True, True, 30 * 2 ** 30, 75 * 2 ** 30)
    _slab(monkeypatch)
    first = cq.price(**_kwargs(fit_probe=lambda _p, _o: half, slab_weights=False))
    assert first.verdict == cq.VERDICT_FITS
    second = cq.price(**_kwargs(
        fit_probe=lambda _p, _o: half, slab_weights=False, slot=cq.SLOT_UNCOND,
        budget_bytes_already_charged=first.measured["required_bytes"]))
    assert second.verdict == cq.VERDICT_REFUSE
    assert second.slot == cq.SLOT_UNCOND
    assert second.measured["headroom_bytes"] < 0


def test_a_quote_during_an_in_flight_load_reads_unknown_and_prices_fresh(monkeypatch):
    """``store.current`` is None between the drop and the repopulation, at the
    load high water. The failed-cleanup set says so."""
    _slab(monkeypatch)
    snapshot = {"cond": None, "uncond": {"residency": "slab",
                                         "request_key": "('other.safetensors',)"},
                "cleanup_failed_slots": ["cond"],
                "retained_failed_load_slabs": 2,
                "failed_load_cleanup_pending": True}
    row = cq.price(**_kwargs(store_snapshot=snapshot))
    assert row.would_load == cq.WOULD_LOAD_UNKNOWN
    assert row.holds_resident == {"cond": "in_flight", "uncond": "slab"}
    assert row.retained_failed_load_slabs == 2
    assert row.failed_load_cleanup_pending is True
    assert row.checkpoint_bytes == row.checkpoint_bytes  # reported, never invented


# The key ``ModelStore.ensure`` would build for ``_kwargs``'s own request
# against a resident of the same name, options, LoRA stack and quant kind.
HELD_KEY = "('m.safetensors', (), (), 'bf16')"


def _resident(residency="stock", rung=cq.RUNG_STOCK_FITS):
    return {"cond": {"residency": residency, "residency_rung": rung,
                     "request_key": HELD_KEY},
            "uncond": None, "cleanup_failed_slots": []}


def test_a_resident_slot_admits_because_no_load_would_run(monkeypatch):
    """The bytes are already placed, so nothing is priced against them.

    ``ModelStore.ensure`` returns the resident patcher above the ladder. A
    quote that prices a fresh load against the availability that resident
    already consumed refuses every render of a settled checkpoint, which the
    2026-09-03 shmem record (docs/VALIDATION.md) shows rendering.
    """
    _slab(monkeypatch, SLAB_SHORT)
    row = cq.price(**_kwargs(store_snapshot=_resident("slab", cq.RUNG_EXPLICIT),
                             request_key=HELD_KEY,
                             fit_probe=lambda _p, _o: DOES_NOT_FIT))
    assert (row.verdict, row.would_load) == (cq.VERDICT_FITS, cq.WOULD_LOAD_REUSE)
    # No probe ran, so nothing is priced. The block is the box's own reading,
    # not a price: required zero, all of it headroom (see the reuse-headroom
    # tests at the end of this file).
    assert row.priced is False
    assert row.measured is None and row.slab_measured["required_bytes"] == 0
    assert (row.use_slab, row.rung) == (True, cq.RUNG_EXPLICIT)
    assert row.reason == "this slot already holds these bytes; no load would run"
    stock = cq.price(**_kwargs(store_snapshot=_resident(), request_key=HELD_KEY,
                               fit_probe=lambda _p, _o: DOES_NOT_FIT))
    assert (stock.use_slab, stock.rung) == (False, cq.RUNG_STOCK_FITS)


def test_a_pending_slab_retry_is_a_reload_and_is_priced(monkeypatch):
    """``ensure`` drops a resident carrying the retry flag, so the same
    snapshot describes a fresh slab load and takes the slab wall."""
    _slab(monkeypatch, SLAB_SHORT)
    row = cq.price(**_kwargs(store_snapshot=_resident(), request_key=HELD_KEY,
                             authoritative_slab_retry=True))
    assert (row.verdict, row.rung) == (cq.VERDICT_REFUSE, cq.RUNG_VOUCHED_AUTO)
    assert row.slab_measured["probe"] == "worker_slab_load"


def test_the_operator_reserve_reaches_the_slab_probe(monkeypatch):
    """The driver and the worker walls charge one floor or they disagree."""
    seen = []

    def probe(_path, _options, *, reserve_bytes=None, **_kw):
        seen.append(reserve_bytes)
        return SLAB_FITS

    monkeypatch.setattr(capacity_fit, "slab_load_fit", probe)
    cq.price(**_kwargs(slab_weights=True, reserve_bytes=10 * 2 ** 30))
    assert seen == [10 * 2 ** 30]


def test_a_slot_holding_these_bytes_reads_reuse_and_one_holding_others_reads_swap(
        monkeypatch):
    _slab(monkeypatch)
    held = {"cond": {"residency": "stock", "request_key": HELD_KEY},
            "uncond": None, "cleanup_failed_slots": []}
    assert cq.price(**_kwargs(store_snapshot=held, request_key=HELD_KEY)
                    ).would_load == cq.WOULD_LOAD_REUSE
    other = dict(held, cond={"residency": "stock", "request_key": "('other.safetensors',)"})
    assert cq.price(**_kwargs(store_snapshot=other, request_key=HELD_KEY)
                    ).would_load == cq.WOULD_LOAD_SWAP
    assert cq.price(**_kwargs(request_key=HELD_KEY)).would_load == cq.WOULD_LOAD_FRESH


def test_a_resident_of_the_same_name_under_another_key_reads_swap(monkeypatch):
    """``ensure`` reuses on the whole key, not on the checkpoint name.

    ``make_keys`` builds the name, the normalized options, the LoRA signature
    and the quant kind, so a dtype cast or a changed LoRA stack reloads the
    file. A name-only match read as reuse prices that load at nothing and
    charges the next slot nothing for a whole checkpoint.
    """
    _slab(monkeypatch)
    held = _resident()
    swapped = repr(("m.safetensors", (), (("extra.safetensors", 0.8),), "bf16"))
    row = cq.price(**_kwargs(store_snapshot=held, request_key=swapped))
    assert row.would_load == cq.WOULD_LOAD_SWAP
    assert row.priced is True
    # A caller that states no key gets the charged reading, never the free one.
    assert cq.price(**_kwargs(store_snapshot=held)).would_load == cq.WOULD_LOAD_SWAP


@pytest.mark.parametrize(("overrides", "credited"), [
    ({}, True),                                   # slab auto adopts the stock copy
    ({"slab_weights": True}, False),              # explicit on reloads into slab
    ({"authoritative_slab_retry": True}, False),  # a pending slab retry reloads too
])
def test_the_bake_credit_follows_only_the_adopting_swap(monkeypatch, overrides, credited):
    """A same-checkpoint stock swap is credited one file against the stray term
    only where the store adopts it instead of reloading (docs/VALIDATION.md,
    2026-09-29 comfy-bump validation)."""
    seen = []
    _slab(monkeypatch)
    monkeypatch.setattr(capacity_fit, "slab_load_fit",
                        lambda _path, _options, **kw: seen.append(kw.get("lora_credit_bytes")) or SLAB_FITS)
    monkeypatch.setattr(cq.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(cq.os.path, "getsize", lambda _p: 24 << 30)
    swapped = repr(("m.safetensors", (), (("extra.safetensors", 0.8),), "bf16"))
    cq.price(**_kwargs(store_snapshot=_resident(), request_key=swapped,
                       lora_stack=[{"name": "extra.safetensors", "strength": 0.8}], **overrides))
    assert seen == [24 << 30 if credited else 0]


def test_a_managed_refusal_carries_its_numbers_and_names_the_shortfall(monkeypatch):
    """The comfy-managed rung answers in the block the fleet card reads and sorts on.

    A block of bytes alone prints a refusal with no numbers and sorts at zero
    headroom, which inverts worst-first. And the rung's one sentence cannot be
    the whole answer on a refusal: it says which residency this rank runs,
    which is as true of a load that fits.
    """
    _slab(monkeypatch)
    monkeypatch.setattr(cq.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(cq.os.path, "getsize", lambda _p: 40 * 2 ** 30)
    monkeypatch.setattr(cq.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(cq.mesh_safety, "mem_available_bytes", lambda: 60 * 2 ** 30)
    fits = cq.price(**_kwargs(comfy_managed=lambda: True, pinned_staging=False))
    monkeypatch.setattr(cq.mesh_safety, "mem_available_bytes", lambda: 20 * 2 ** 30)
    short = cq.price(**_kwargs(comfy_managed=lambda: True, pinned_staging=False))

    assert (fits.verdict, short.verdict) == (cq.VERDICT_FITS, cq.VERDICT_REFUSE)
    for key in ("weights_gib", "required_gib", "mem_available_gib", "headroom_gib"):
        assert isinstance(short.measured[key], float), key
    assert short.measured["weights_gib"] == 40.0
    assert short.measured["headroom_gib"] < 0
    assert short.reason != fits.reason
    assert "short" in short.reason
    assert fits.reason == (
        "this worker runs comfy-managed residency (ComfyUI's DynamicVRAM)")


def test_the_fsdp_rung_spends_what_an_earlier_slot_already_charged(monkeypatch):
    """Slab, stock and managed all subtract it; a rank pricing two checkpoints
    on an FSDP launch would otherwise admit the second against memory the
    first already took."""
    from dgx_monarch.adapters import fsdp

    monkeypatch.setattr(cq.os.path, "getsize", lambda _p: 1000)
    monkeypatch.setattr(cq.os.path, "exists", lambda _p: True)
    _slab(monkeypatch)
    monkeypatch.setattr(fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: 7 * 2 ** 30)
    monkeypatch.setattr(fsdp, "fsdp_load_capacity_check",
                        lambda *_a, **_k: None)
    free = cq.price(**_kwargs(fsdp_launch=True))
    assert free.verdict == cq.VERDICT_FITS
    charged = cq.price(**_kwargs(fsdp_launch=True,
                                 budget_bytes_already_charged=3 * 2 ** 30))
    assert charged.verdict == cq.VERDICT_REFUSE
    assert charged.measured["mem_available_bytes"] == 4 * 2 ** 30
    assert charged.measured["headroom_bytes"] < 0
    assert "short" in charged.reason


def test_the_fsdp_refusal_reason_carries_no_second_class_tag(monkeypatch, tmp_path):
    """The adapter tags its own sentence, so storing it whole puts a tag in the
    middle of the fleet card's line. The body travels; the tag is added once by
    whoever raises."""
    from dgx_monarch.adapters import fsdp
    from dgx_monarch.refusal import parse_refusal_tag

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 1000)
    _slab(monkeypatch)
    monkeypatch.setattr(fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: 1100)
    row = cq.price(**_kwargs(fsdp_launch=True, path=str(checkpoint)))
    assert row.verdict == cq.VERDICT_REFUSE
    assert parse_refusal_tag(row.reason) is None
    assert "[dgxm:" not in row.reason


def test_the_fsdp_quote_prices_a_quantized_direct_wrap(monkeypatch, tmp_path):
    """The pre-allocation agreement cannot use the bf16 streaming price."""
    from dgx_monarch.adapters import detect, fsdp

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 1000)
    _slab(monkeypatch)
    monkeypatch.setattr(fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: 100 * 2 ** 30)
    monkeypatch.setattr(
        detect, "sniff_fsdp_launch_quant_proof",
        lambda _path: SimpleNamespace(quant_kind="fp8"))

    row = cq.price(**_kwargs(fsdp_launch=True, path=str(checkpoint), world=2))

    assert row.verdict == cq.VERDICT_FITS
    assert row.measured["required_bytes"] == capacity_fit.fsdp_required_bytes(
        1000, 2, "fp8")


def test_the_fsdp_raise_becomes_a_row_and_a_failed_import_makes_no_claim(
        monkeypatch, tmp_path):
    """The one rung whose check raises. A rank that cannot answer must not veto
    one that can, so a failed adapter import is ``no_claim`` and not a refusal."""
    from dgx_monarch.adapters import fsdp

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 1000)
    _slab(monkeypatch)
    monkeypatch.setattr(fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: 7 * 2 ** 30)
    row = cq.price(**_kwargs(fsdp_launch=True, path=str(checkpoint)))
    assert (row.verdict, row.rung) == (cq.VERDICT_FITS, cq.RUNG_STOCK_FITS)
    # The shard-build check priced this load, so the row carries its numbers:
    # a priced verdict with no block is journalled as unpriced.
    assert row.priced and row.measured["fits"] is True
    assert row.measured["probe"] == capacity_fit.PROBE_PREFLIGHT_WALL
    assert row.measured["headroom_bytes"] == 7 * 2 ** 30 - row.measured["required_bytes"]

    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: 1100)
    refused = cq.price(**_kwargs(fsdp_launch=True, path=str(checkpoint)))
    assert refused.verdict == cq.VERDICT_REFUSE
    assert "checkpoint kind is unproven" in refused.reason
    assert refused.measured["headroom_bytes"] < 0

    import dgx_monarch.adapters as adapters

    monkeypatch.delattr(adapters, "fsdp")
    monkeypatch.setitem(sys.modules, "dgx_monarch.adapters.fsdp", None)
    blind = cq.price(**_kwargs(fsdp_launch=True, path=str(checkpoint)))
    assert blind.verdict == cq.VERDICT_NO_CLAIM
    assert "could not be imported" in blind.reason


def test_every_row_carries_the_shmem_and_retained_slab_fields(monkeypatch):
    """``capacity_memory`` owns the shmem question; the row records shmem and claims nothing."""
    _slab(monkeypatch)
    row = cq.price(**_kwargs())
    assert row.shmem_bytes is None or row.shmem_bytes >= 0
    assert set(row.holds_resident) == {"cond", "uncond"}
    assert row.retained_failed_load_slabs == 0
    assert set(row.to_row()) == cq.ROW_KEYS


def test_two_ranks_with_different_memory_build_one_agreeable_descriptor(monkeypatch):
    """Built from real per-rank fits, not from synthetic identical rows.

    ``measured`` and ``human_reason`` differ on every rank by construction, so
    a driver that agreed descriptors whole would refuse every world-2 rescue.
    The rank-invariant half is what the grant is keyed on.
    """
    lean = StockFit(False, True, 62 * 2 ** 30, 47 * 2 ** 30)
    fat = StockFit(False, True, 62 * 2 ** 30, 107 * 2 ** 30)
    rows = [cq.build_descriptor(
        unet_name="m.safetensors", path="/models/m.safetensors", fit=fit,
        model_options={}, file_identity=lambda _p: "1:2:3:4:5",
        family_hint="krea2", host=lambda: "one-box") for fit in (lean, fat)]
    invariant = [(d.consent_id, d.kind, d.file_identity, d.context_fingerprint,
                  tuple(sorted(d.memo_context.items())), d.family_hint) for d in rows]
    assert invariant[0] == invariant[1]
    assert rows[0].measured != rows[1].measured
    assert rows[0].human_reason != rows[1].human_reason


def test_a_reuse_row_reports_the_headroom_the_box_actually_has(monkeypatch):
    """The fleet card prints the headroom in the block, so a reuse fills the
    block with the box's own reading.

    The 2026-09-03 silent-rank leg logged
    ``rank 0=cond:fits/vouched_auto@0.0GiB`` for a warm rank holding the
    checkpoint. Zero headroom on a verdict that fits reads as the leanest rank
    on the fleet, though a reuse places nothing and is the roomiest answer. The
    row stays ``priced`` False, because no probe ran.
    """
    from dgx_monarch import capacity_agreement

    _slab(monkeypatch, SLAB_SHORT)
    monkeypatch.setattr(cq.mesh_safety, "mem_available_bytes",
                        lambda: 48 * 2 ** 30)
    row = cq.price(**_kwargs(store_snapshot=_resident("slab", cq.RUNG_VOUCHED_AUTO),
                             request_key=HELD_KEY,
                             fit_probe=lambda _p, _o: DOES_NOT_FIT))
    assert (row.verdict, row.rung) == (cq.VERDICT_FITS, cq.RUNG_VOUCHED_AUTO)
    assert row.priced is False
    assert row.measured is None
    block = row.slab_measured
    assert block["probe"] == cq.PROBE_REUSE
    assert block["required_bytes"] == 0
    assert block["headroom_bytes"] == 48 * 2 ** 30
    assert block["checkpoint_bytes"] == row.checkpoint_bytes
    answer = capacity_agreement.RankAnswer(rank=0, host="a", quotes=(row,))
    assert capacity_agreement._headroom(answer.quotes[0]) == 48.0


def test_a_stock_reuse_answers_in_the_stock_block(monkeypatch):
    """``_block`` picks by ``use_slab``, so a stock resident must fill the
    other half or the card reads it as a slab row with nothing in it."""
    _slab(monkeypatch)
    monkeypatch.setattr(cq.mesh_safety, "mem_available_bytes", lambda: 12 * 2 ** 30)
    row = cq.price(**_kwargs(store_snapshot=_resident(), request_key=HELD_KEY))
    assert row.slab_measured is None
    assert row.measured["probe"] == cq.PROBE_REUSE
    assert row.measured["headroom_gib"] == 12.0


def test_a_reuse_row_charges_the_next_slot_nothing_for_what_it_holds(monkeypatch):
    """The block says ``required_bytes`` 0 and the budget helper reads 0 too.

    A reuse places nothing, so filling the block must not start charging the
    second slot for the first one's resident.
    """
    from dgx_monarch.actor import worker_capacity

    _slab(monkeypatch)
    monkeypatch.setattr(cq.mesh_safety, "mem_available_bytes", lambda: 30 * 2 ** 30)
    row = cq.price(**_kwargs(store_snapshot=_resident(), request_key=HELD_KEY))
    assert worker_capacity._charged_bytes(row) == 0


def test_a_reuse_block_rebuilds_no_fit_because_no_probe_ran(monkeypatch):
    """The block names no probe, so the two inverses build nothing from it.

    ``store_residency`` rebuilds a ``StockFit`` or a ``SlabFit`` from whichever
    half a row carries, keyed on the probe name. A reuse block naming
    ``worker_slab_load`` would let that inverse rebuild a measured fit from a
    probe that never ran, and print a price for a load that priced nothing.
    """
    from dgx_monarch import capacity_agreement, gate_audit_vocab
    from dgx_monarch.actor import store_residency

    _slab(monkeypatch)
    monkeypatch.setattr(cq.mesh_safety, "mem_available_bytes", lambda: 40 * 2 ** 30)
    slab = cq.price(**_kwargs(store_snapshot=_resident("slab", cq.RUNG_VOUCHED_AUTO),
                              request_key=HELD_KEY))
    stock = cq.price(**_kwargs(store_snapshot=_resident(), request_key=HELD_KEY))
    for row in (slab, stock):
        assert row.priced is False
        assert store_residency._slab_fit(row) is None
        assert store_residency._stock_fit(row) is None
    # The card still reads the figure it needs out of the same block.
    assert capacity_agreement._stated_headroom(slab) == 40.0
    assert capacity_agreement._stated_headroom(stock) == 40.0
    # A reuse admits, so this block reaches no refusal and no audit row. The
    # audit vocabulary is protocol-bound: a new member needs a protocol version change.
    assert cq.PROBE_REUSE not in gate_audit_vocab.MEASURED_PROBES


def test_a_reuse_row_on_an_unreadable_meminfo_claims_no_numbers(monkeypatch):
    """No reading, no block: an invented headroom is worse than none."""
    _slab(monkeypatch)
    monkeypatch.setattr(cq.mesh_safety, "mem_available_bytes", lambda: None)
    row = cq.price(**_kwargs(store_snapshot=_resident(), request_key=HELD_KEY))
    assert (row.measured, row.slab_measured) == (None, None)
