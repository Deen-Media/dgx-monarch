"""The cross-rank partial-load guard: exchange first, then every rank refuses.

The hazard, measured on hardware 2026-08-12 (leg L9c of out/ltx25_ladder): a
bf16 DiT with a baked LoRA stack under forced-stock residency put one box of
the pair into ComfyUI's partial-load regime while the other full-loaded, and
the two ranks then computed different weights. The cross-rank latent canary
saw 2.27e-03 relative drift; on a harder leg the pressured rank stalled six
minutes and died. docs/TROUBLESHOOTING.md #79 is the operator's entry.

The guard must refuse rather than quietly switch the render's residency
(fail-closed by the maintainer's choice), and it must refuse on every rank:
the deciding fact crosses the process group before any rank raises, so no
rank is left alone inside a collective.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dgx_monarch.actor import partial_load_guard as guard  # noqa: E402
from dgx_monarch.refusal import (  # noqa: E402
    GUARDS,
    RefusalClass,
    parse_leading_refusal_tag,
)


class _FakeBaseModel:
    """Only the attribute ComfyUI writes beside its own load journal line."""

    def __init__(self, lowvram: bool) -> None:
        self.model_lowvram = lowvram


class _FakeFleet:
    """Every rank of one world, MIN-reducing one shared set of flags.

    ``all_reduce`` here does what NCCL does: every rank leaves with the same
    value. Each rank then runs the real ``check``, so the test sees what each
    one does with that value.
    """

    def __init__(self, per_rank_full: list[bool]) -> None:
        self.per_rank_full = per_rank_full
        self.reduced = 0

    def install(self, monkeypatch) -> None:
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: len(self.per_rank_full))
        monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")

        def all_reduce(tensor, op=None, group=None, async_op=False):
            self.reduced += 1
            tensor.fill_(1 if all(self.per_rank_full) else 0)
            return None

        monkeypatch.setattr(dist, "all_reduce", all_reduce)


def _run_every_rank(fleet: _FakeFleet) -> list[BaseException | None]:
    outcomes: list[BaseException | None] = []
    for full in fleet.per_rank_full:
        try:
            guard.check(_FakeBaseModel(lowvram=not full))
            outcomes.append(None)
        except BaseException as exc:  # the test asserts on the outcome
            outcomes.append(exc)
    return outcomes


def test_every_rank_full_loaded_passes_through(monkeypatch):
    fleet = _FakeFleet([True, True])
    fleet.install(monkeypatch)
    assert _run_every_rank(fleet) == [None, None]
    assert fleet.reduced == 2, "both ranks must reach the exchange, not just the deciding one"


def test_one_partial_rank_refuses_on_every_rank(monkeypatch):
    """The full-loaded rank refuses too.

    A raise on one rank mid-collective wedges the fleet: the healthy rank
    would wait in the next all_gather until the NCCL timeout. Refusing on the
    healthy rank keeps the mesh usable for the next render.
    """
    fleet = _FakeFleet([True, False])
    fleet.install(monkeypatch)
    outcomes = _run_every_rank(fleet)

    assert all(isinstance(outcome, guard.PartialLoadDivergenceError) for outcome in outcomes), (
        f"a partial rank must refuse on EVERY rank, got {outcomes}")
    messages = {str(outcome) for outcome in outcomes}
    assert len(messages) == 1, (
        "the ranks raised different text for one fact; a MIN exchange carries "
        f"agreement, not identity: {messages}")
    assert fleet.reduced == 2, "the exchange must complete on every rank before any raise"


def test_three_ranks_refuse_when_a_single_middle_rank_is_partial(monkeypatch):
    fleet = _FakeFleet([True, False, True])
    fleet.install(monkeypatch)
    outcomes = _run_every_rank(fleet)
    assert all(isinstance(outcome, guard.PartialLoadDivergenceError) for outcome in outcomes)
    assert fleet.reduced == 3


def test_every_rank_partial_at_different_splits_is_the_same_refusal(monkeypatch):
    """Observed on hardware 2026-08-12: both ranks partial, at different splits.

    A 60 GiB checkpoint on two boxes left one rank holding 6201 MB and the
    other 43713 MB. The predicate is a MIN over "this rank is whole", so it
    refuses here, but the message must not promise another rank that "holds
    all of them": no such rank exists, and the reader would hunt for it.
    """
    fleet = _FakeFleet([False, False])
    fleet.install(monkeypatch)
    outcomes = _run_every_rank(fleet)
    assert all(isinstance(outcome, guard.PartialLoadDivergenceError)
               for outcome in outcomes)
    message = str(outcomes[0])
    assert "while another holds all of them" not in message
    assert "Several ranks" in message and "different splits" in message
    assert "every rank to hold the whole model" in message


def test_world_one_never_refuses_and_never_reduces(monkeypatch):
    """A single rank cannot disagree with itself, and comfy's partial load
    there is its ordinary, correct behavior."""
    import torch.distributed as dist

    fleet = _FakeFleet([False])
    fleet.install(monkeypatch)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 1)
    guard.check(_FakeBaseModel(lowvram=True))
    assert fleet.reduced == 0


def test_no_process_group_never_refuses(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    guard.check(_FakeBaseModel(lowvram=True))


def test_the_exchange_reduces_one_int32_element_with_MIN(monkeypatch):
    """Shape, dtype and op are the contract: MIN over one element is what makes
    'any rank partial' readable by every rank in one collective."""
    import torch.distributed as dist

    seen: dict[str, object] = {}

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)
    monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")

    def all_reduce(tensor, op=None, group=None, async_op=False):
        seen["shape"] = tuple(tensor.shape)
        seen["dtype"] = tensor.dtype
        seen["value"] = int(tensor.item())
        seen["op"] = op
        seen["group"] = group
        tensor.fill_(0)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    with pytest.raises(guard.PartialLoadDivergenceError):
        guard.check(_FakeBaseModel(lowvram=False))
    assert seen["shape"] == (1,)
    assert seen["dtype"] is torch.int32
    assert seen["value"] == 1, "a full-loaded rank must contribute 1, not its own verdict"
    assert seen["op"] is dist.ReduceOp.MIN
    assert seen["group"] is None, "the default group is the one every rank is already in"


def test_local_full_load_reads_comfys_own_flag():
    assert guard.local_full_load(_FakeBaseModel(lowvram=False)) is True
    assert guard.local_full_load(_FakeBaseModel(lowvram=True)) is False
    # comfy sets the attribute at patcher construction; a model that never saw
    # a load manager is not evidence of a partial one.
    assert guard.local_full_load(object()) is True


def _raised_message(monkeypatch) -> str:
    """The message an actual refusal carries, not a rebuilt copy of it."""
    fleet = _FakeFleet([True, False])
    fleet.install(monkeypatch)
    with pytest.raises(guard.PartialLoadDivergenceError) as raised:
        guard.check(_FakeBaseModel(lowvram=False))
    return str(raised.value)


def test_the_refusal_is_built_at_the_raise_and_never_at_import():
    """The driver imports this module to unpickle the exception a worker raised.

    A module-level ``refusal(...)`` runs there, against whatever refusal.py
    that long-lived driver process loaded at startup, and a worker-only deploy
    then replaces the clean refusal with ``unknown refusal guard``. Seen on
    hardware on 2026-08-12.
    """
    source = (REPO / "src" / "dgx_monarch" / "actor" / "partial_load_guard.py").read_text()
    module_level = [
        line for line in source.splitlines()
        if line.startswith("REFUSAL") and "refusal(" in line
    ]
    assert not module_level, (
        f"the refusal must be built at the raise, not at import: {module_level}")


def test_the_refusal_is_class_C_non_waivable_and_names_its_guard(monkeypatch):
    message = _raised_message(monkeypatch)
    tag = parse_leading_refusal_tag(message)
    assert tag is not None, "the tag must lead the message or the driver cannot own the failure"
    assert tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == guard.GUARD
    assert tag.waivable is False, (
        "a waiver here would authorize ranks that knowingly compute different weights")
    assert GUARDS[guard.GUARD].waivable_now is False
    assert "DGX Monarch panel" not in message, "a non-waivable refusal must not name a card"


def test_the_refusal_names_the_alternatives_that_actually_fit(monkeypatch):
    message = _raised_message(monkeypatch)
    for alternative in ("int8", "fp8", "lora_low_rss", "single box"):
        assert alternative in message, f"the refusal does not name {alternative}"
    assert "TROUBLESHOOTING.md #79" in message


def test_a_tagged_refusal_retires_its_lease_as_consumed(monkeypatch):
    """Lease semantics follow the tag, not the exception class.

    ``nodes.pending.typed_worker_refusal`` reads the leading tag off the
    preserved inner exception, so this class-C raise retires CONSUMED and the
    operator can act on the refusal and queue again without a fleet recycle.
    """
    from dgx_monarch.nodes import pending

    monarch_actor = pytest.importorskip("monarch.actor")
    wrapped = monarch_actor.ActorError(RuntimeError(_raised_message(monkeypatch)))
    assert pending.typed_worker_refusal(wrapped) is True


def test_the_guard_is_not_swept_into_the_stock_load_capacity_family(monkeypatch):
    """The weights loaded. Routing this to the residency rescue ladder, or to
    the ceremony's "cannot load under stock" verdict, would both be false."""
    from dgx_monarch import mesh_safety

    error = guard.PartialLoadDivergenceError(_raised_message(monkeypatch))
    assert not isinstance(error, mesh_safety.StockLoadCapacityError)
    assert not mesh_safety.is_stock_load_capacity_error(error)


class _FakeExecutor:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.result


def test_the_wrapper_checks_after_the_load_and_returns_the_result_untouched(monkeypatch):
    fleet = _FakeFleet([True, True])
    fleet.install(monkeypatch)
    model = _FakeBaseModel(lowvram=False)
    result = (model, {"positive": []}, [])
    executor = _FakeExecutor(result)

    wrapper = guard.make_prepare_sampling_guard()
    assert wrapper(executor, object(), (1, 2, 3)) is result
    assert executor.calls == 1
    assert fleet.reduced == 1, "the check must run after comfy has decided, not before"


def test_the_wrapper_refuses_after_a_partial_load(monkeypatch):
    fleet = _FakeFleet([True, False])
    fleet.install(monkeypatch)
    executor = _FakeExecutor((_FakeBaseModel(lowvram=True), {}, []))
    wrapper = guard.make_prepare_sampling_guard()
    with pytest.raises(guard.PartialLoadDivergenceError):
        wrapper(executor, object())
    assert executor.calls == 1


def test_install_is_world_gated_and_idempotent(monkeypatch):
    pe = pytest.importorskip(
        "comfy.patcher_extension", reason="no ComfyUI on sys.path (canary job covers it)")

    class _Patcher:
        def __init__(self):
            self.model_options: dict = {}

    def installed(patcher):
        return pe.get_wrappers_with_key(
            pe.WrappersMP.PREPARE_SAMPLING, guard.WRAPPER_KEY,
            patcher.model_options, is_model_options=True)

    solo = _Patcher()
    guard.install(solo, 1)
    guard.install(solo, None)
    assert installed(solo) == [], "world 1 has nothing to disagree about"

    pair = _Patcher()
    guard.install(pair, 2)
    assert len(installed(pair)) == 1
    # comfy appends under one key, so without the check a re-injection across a
    # residency respelling would add a second exchange to every sample.
    guard.install(pair, 2)
    assert len(installed(pair)) == 1
