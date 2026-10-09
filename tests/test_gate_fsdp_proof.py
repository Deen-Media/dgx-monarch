"""Driver-side validation for the FSDP clean-reload proof."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import fsdp_reload_price as price_mod
from dgx_monarch import mesh_safety as mesh_safety_mod
from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION
from dgx_monarch.mesh_safety import StockLoadCapacityError
from dgx_monarch.nodes import gate as gate_mod
from dgx_monarch.nodes import gate_abort
from dgx_monarch.nodes.gate_fsdp import (
    FsdpGateProofError,
    cleanup_aborted_fsdp_proof,
    establish,
    require_confirmed_fsdp_unload,
)
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
    _cross_mode_rig,
    _run,
)


def _handle(*, world: int = 2, setup_generation: int = 7):
    return SimpleNamespace(world=world, setup_generation=setup_generation)


def _row(rank: int, **overrides):
    row = {
        "rank": rank,
        "setup_generation": 7,
        "conclusive": True,
        "proof": "fsdp_clean_reload",
        "baseline_verified": True,
        "baseline_artifact_identity_verified": True,
        "baseline_family": "wan",
        "baseline_quant": "bf16",
        "baseline_live_dtype_profile": "all_bf16",
        "baseline_auxiliary_parameter_count": 0,
        "baseline_auxiliary_parameter_bytes": 0,
        "baseline_checkpoint_precision": "bf16",
        "baseline_slab_active": False,
        "baseline_fsdp_ready": True,
        "transitions": {"reload": "load"},
        "family": "wan",
        "quant": "bf16",
        "live_dtype_profile": "all_bf16",
        "auxiliary_parameter_count": 0,
        "auxiliary_parameter_bytes": 0,
        "checkpoint_precision": "bf16",
        "slab_active": False,
        "fsdp_ready": True,
        "artifact_identity_verified": True,
    }
    row.update(overrides)
    return row


def test_fsdp_reload_proof_accepts_complete_exact_rank_evidence():
    proof = establish([_row(0), _row(1)], _handle())

    assert proof.conclusive is True
    assert proof.reasons == []


@pytest.mark.parametrize(
    "cycle",
    [
        pytest.param([_row(0), None], id="malformed-row"),
        pytest.param([_row(0), _row(1), None], id="extra-malformed-row"),
        pytest.param([_row(0)], id="missing-rank"),
        pytest.param([_row(0), _row(0)], id="duplicate-rank"),
        pytest.param([_row(0), _row(True)], id="boolean-rank"),
    ],
)
def test_fsdp_reload_proof_rejects_malformed_duplicate_or_missing_rank_evidence(
    cycle,
):
    proof = establish(cycle, _handle())

    assert proof.conclusive is False
    assert any("rank" in reason for reason in proof.reasons)


def test_fsdp_reload_proof_requires_exact_builtin_rpc_containers():
    class ListSubclass(list):
        pass

    class DictSubclass(dict):
        pass

    for cycle in (
        ListSubclass([_row(0), _row(1)]),
        [_row(0), DictSubclass(_row(1))],
    ):
        proof = establish(cycle, _handle())
        assert proof.conclusive is False
        assert "FSDP reload cycle response is malformed" in proof.reasons


def test_fsdp_reload_proof_rejects_integer_subclasses_at_rpc_boundary():
    class IntSubclass(int):
        pass

    rank = establish([_row(0), _row(IntSubclass(1))], _handle())
    world = establish(
        [_row(0), _row(1)],
        _handle(world=IntSubclass(2)),
    )

    assert rank.conclusive is False
    assert world.conclusive is False


def test_fsdp_reload_proof_rejects_nested_spoof_values():
    class SpoofStr(str):
        def __eq__(self, _other):
            return True

    class DictSubclass(dict):
        pass

    class ExplosiveInt(int):
        def __eq__(self, _other):
            raise AssertionError("malformed scalar equality was evaluated")

    overrides = (
        {"proof": SpoofStr("wrong")},
        {"family": SpoofStr("wan"), "baseline_family": SpoofStr("wan")},
        {
            "live_dtype_profile": SpoofStr("all_bf16"),
            "baseline_live_dtype_profile": SpoofStr("all_bf16"),
        },
        {"quant": SpoofStr("bf16"), "baseline_quant": SpoofStr("bf16")},
        {
            "checkpoint_precision": SpoofStr("bf16"),
            "baseline_checkpoint_precision": SpoofStr("bf16"),
        },
        {"transitions": {"reload": SpoofStr("reuse")}},
        {"transitions": DictSubclass({"reload": "load"})},
    )

    for override in overrides:
        proof = establish([_row(0, **override), _row(1, **override)], _handle())
        assert proof.conclusive is False

    explosive = establish(
        [_row(0), _row(1, auxiliary_parameter_count=ExplosiveInt(0))],
        _handle(),
    )
    assert explosive.conclusive is False


def test_fsdp_reload_proof_rejects_non_list_cycle_and_boolean_world():
    malformed = establish(None, _handle())
    boolean_world = establish([_row(0)], _handle(world=True))

    assert malformed.conclusive is False
    assert "FSDP reload cycle response is malformed" in malformed.reasons
    assert boolean_world.conclusive is False


def test_fsdp_reload_proof_rejects_setup_generation_drift():
    proof = establish(
        [_row(0), _row(1, setup_generation=8)],
        _handle(),
    )

    assert proof.conclusive is False
    assert "FSDP reload cycle setup generation is invalid or drifted" in proof.reasons


@pytest.mark.parametrize(
    "generation",
    [
        pytest.param(7.0, id="equal-float"),
        pytest.param(True, id="boolean"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
    ],
)
def test_fsdp_reload_proof_requires_exact_positive_integer_generation(generation):
    proof = establish(
        [_row(0, setup_generation=generation), _row(1)],
        _handle(),
    )

    assert proof.conclusive is False
    assert "FSDP reload cycle setup generation is invalid or drifted" in proof.reasons


def test_fsdp_reload_proof_rejects_non_integer_driver_generation():
    proof = establish(
        [_row(0), _row(1)],
        _handle(setup_generation=7.0),
    )

    assert proof.conclusive is False
    assert "FSDP reload cycle setup generation is invalid or drifted" in proof.reasons


def test_interrupted_cleanup_publication_latches_dirty_without_second_unload():
    class StopNow(BaseException):
        pass

    cancellation = StopNow("interrupted after cleanup claim")

    class State(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == "cleanup_attempted" and value is True:
                raise cancellation

    class Handle:
        world = 2

        def __init__(self):
            self.unloads = 0
            self.latches = []

        def call_all(self, *_args, **_kwargs):
            self.unloads += 1
            return [{"unloaded": True}, {"unloaded": True}]

        def _latch_ambiguous_mutation(self, *args):
            self.latches.append(args)

    state = State()
    handle = Handle()
    logger = SimpleNamespace(error=lambda *_args: None)

    with pytest.raises(StopNow) as raised:
        cleanup_aborted_fsdp_proof(
            handle, cancellation, logger=logger, state=state)
    assert raised.value is cancellation

    assert cleanup_aborted_fsdp_proof(
        handle, cancellation, logger=logger, state=state) is False
    assert handle.unloads == 0
    assert len(handle.latches) == 1
    assert state == {"cleanup_attempted": True, "cleanup_confirmed": False}


def test_fsdp_unload_requires_exact_builtin_rpc_containers():
    class ListSubclass(list):
        pass

    class DictSubclass(dict):
        pass

    class Handle:
        world = 2

        def __init__(self, responses):
            self.responses = responses
            self.latches = []

        def call_all(self, *_args, **_kwargs):
            return self.responses

        def _latch_ambiguous_mutation(self, *args):
            self.latches.append(args)

    malformed_responses = (
        ListSubclass([{"unloaded": True}, {"unloaded": True}]),
        [{"unloaded": True}, DictSubclass({"unloaded": True})],
    )
    logger = SimpleNamespace(error=lambda *_args: None)

    for responses in malformed_responses:
        handle = Handle(responses)
        with pytest.raises(FsdpGateProofError):
            require_confirmed_fsdp_unload(
                responses, handle, True, phase="baseline")

        primary = RuntimeError("abort")
        state = {}
        assert cleanup_aborted_fsdp_proof(
            handle, primary, logger=logger, state=state) is False
        assert len(handle.latches) == 1
        assert state == {
            "cleanup_attempted": True,
            "cleanup_confirmed": False,
        }


@pytest.mark.parametrize(
    "transition",
    [
        pytest.param({"reload": "reuse"}, id="reuse"),
        pytest.param({"reload": "hot_swap"}, id="hot-swap"),
        pytest.param({}, id="missing-transition"),
    ],
)
def test_fsdp_reload_proof_requires_an_exact_fresh_load_transition(transition):
    proof = establish(
        [_row(0), _row(1, transitions=transition)],
        _handle(),
    )

    assert proof.conclusive is False
    assert "FSDP reload cycle lacks a freshly loaded, ready shard set of an admitted checkpoint kind" in proof.reasons


def test_fsdp_reload_proof_rejects_cross_rank_precision_profile_mismatch():
    proof = establish(
        [
            _row(0),
            _row(
                1,
                live_dtype_profile="wan_fp32_patch_embedding_v1",
                auxiliary_parameter_count=2,
                auxiliary_parameter_bytes=8,
                baseline_live_dtype_profile="wan_fp32_patch_embedding_v1",
                baseline_auxiliary_parameter_count=2,
                baseline_auxiliary_parameter_bytes=8,
            ),
        ],
        _handle(),
    )

    assert proof.conclusive is False
    assert (
        "FSDP reload cycle live precision profile is missing or drifted across ranks"
        in proof.reasons
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("auxiliary_parameter_count", 3),
        ("auxiliary_parameter_bytes", 16),
    ],
)
def test_fsdp_reload_proof_rejects_cross_rank_wan_auxiliary_drift(field, value):
    wan = {
        "live_dtype_profile": "wan_fp32_patch_embedding_v1",
        "auxiliary_parameter_count": 2,
        "auxiliary_parameter_bytes": 8,
        "baseline_live_dtype_profile": "wan_fp32_patch_embedding_v1",
        "baseline_auxiliary_parameter_count": 2,
        "baseline_auxiliary_parameter_bytes": 8,
    }
    drifted = dict(wan)
    drifted[field] = value
    drifted[f"baseline_{field}"] = value

    proof = establish([_row(0, **wan), _row(1, **drifted)], _handle())

    assert proof.conclusive is False
    assert "FSDP reload cycle precision evidence is incomplete or drifted across ranks" in proof.reasons


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {
                "live_dtype_profile": "all_bf16",
                "auxiliary_parameter_count": 2,
                "auxiliary_parameter_bytes": 8,
            },
            id="all-bf16-with-auxiliary-parameters",
        ),
        pytest.param(
            {
                "auxiliary_parameter_count": False,
                "auxiliary_parameter_bytes": 0,
                "baseline_auxiliary_parameter_count": False,
            },
            id="all-bf16-boolean-count",
        ),
        pytest.param(
            {
                "auxiliary_parameter_count": 0,
                "auxiliary_parameter_bytes": 0.0,
                "baseline_auxiliary_parameter_bytes": 0.0,
            },
            id="all-bf16-float-bytes",
        ),
        pytest.param(
            {
                "live_dtype_profile": "wan_fp32_patch_embedding_v1",
                "auxiliary_parameter_count": 1,
                "auxiliary_parameter_bytes": 8,
            },
            id="wan-profile-wrong-count",
        ),
        pytest.param(
            {"live_dtype_profile": "unreviewed_mixed_precision"},
            id="unknown-profile",
        ),
    ],
)
def test_fsdp_reload_proof_rejects_profile_evidence_mismatch(overrides):
    proof = establish([_row(0), _row(1, **overrides)], _handle())

    assert proof.conclusive is False
    assert any("precision" in reason or "BF16" in reason for reason in proof.reasons)


def test_fsdp_reload_proof_rejects_baseline_to_reload_precision_drift():
    proof = establish(
        [_row(0), _row(1, baseline_auxiliary_parameter_count=2)],
        _handle(),
    )

    assert proof.conclusive is False
    assert "FSDP baseline/reload precision evidence is incomplete or drifted" in proof.reasons


GIB = 1 << 30


def _fsdp_cycle_rows() -> list[dict]:
    """The shared rig's handle reports setup generation 4."""
    return [_row(0, setup_generation=4), _row(1, setup_generation=4)]


def _ceremony(monkeypatch, tmp_path, *, renders=None, cycle_error=None,
              status_rows=None):
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)] if renders is None else renders,
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=_fsdp_cycle_rows(),
        cycle_error=cycle_error,
    )
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True
    if status_rows is not None:
        inner = handle.call_all

        def call_all(method, *args, **kwargs):
            if method == "status":
                handle.calls.append((method, args))
                return status_rows
            return inner(method, *args, **kwargs)

        handle.call_all = call_all
    published: list[tuple] = []
    monkeypatch.setattr(
        gate_mod, "_publish_process_gate_verdicts",
        lambda tokens, verdict, ceremony=None: published.append((tokens, verdict)))
    return handle, ledger, model, published


def _shortfall(monkeypatch, *, checkpoint_gib=60, available_gib=30.2):
    """Stub an integrated GPU and the checkpoint size, and return rank status
    rows; the defaults refuse."""
    monkeypatch.setattr(mesh_safety_mod, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(
        price_mod, "checkpoint_bytes", lambda _name: int(checkpoint_gib * GIB))
    return [
        {"rank": rank, "host": {"mem_gib": {"MemAvailable": available_gib}}}
        for rank in range(2)
    ]


def test_a_priced_capacity_stop_mutates_nothing_and_refuses_typed(
    monkeypatch, tmp_path,
):
    """A refusing price unloads nothing, latches nothing, opens no RETESTING
    row and publishes no process verdict."""
    rows = _shortfall(monkeypatch)
    handle, ledger, model, published = _ceremony(
        monkeypatch, tmp_path, status_rows=rows)

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert raised.value.verdict == "INCONCLUSIVE"
    assert "auto_gate=off" in str(raised.value)
    assert handle.unload_calls == 0          # the operator's shards survive
    assert handle.latch_attempts == 0        # nothing was left ambiguous
    assert [method for method, _args in handle.calls] == ["status"]
    assert ledger.retests == []              # no RETESTING row
    assert published == []                   # no process verdict of any kind


def test_the_priced_stop_writes_one_inconclusive_capacity_row(monkeypatch, tmp_path):
    rows = _shortfall(monkeypatch)
    _handle, ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, status_rows=rows)

    with pytest.raises(FsdpGateProofError):
        _run(model)

    assert len(ledger.records) == 1
    _key, _artifacts, _commit, verdict, detail, _context = ledger.records[0]
    assert verdict == "INCONCLUSIVE"
    assert detail["cross_mode"] == "CAPACITY"
    measured = detail["measured"]
    assert measured["probe"] == "stock_load_preflight"
    assert measured["checkpoint_bytes"] == 60 * GIB
    assert measured["mem_available_bytes"] == int(30.2 * GIB)
    assert measured["required_bytes"] > measured["mem_available_bytes"]
    assert measured["headroom_bytes"] < 0
    # No certificate row: no rank byte-verified anything here.
    assert all(record[3] != "CAPACITY_CERTIFIED" for record in ledger.records)
    # The row uses existing vocabulary, so it needs no protocol bump; the
    # comment on gate_ledger.GATE_PROTOCOL_VERSION says what does.
    assert GATE_PROTOCOL_VERSION == 13


@pytest.mark.parametrize(
    "cycle_error",
    [
        pytest.param(
            StockLoadCapacityError("the FSDP launch cannot load flux2"),
            id="direct"),
        pytest.param(
            RuntimeError(
                "ActorError: StockLoadCapacityError('the FSDP launch cannot "
                "load flux2')"),
            id="actor-wrapped"),
    ],
)
def test_an_in_flight_capacity_refusal_is_recorded_and_named(
    monkeypatch, tmp_path, cycle_error,
):
    retracted = _capture_retractions(monkeypatch)
    _handle, ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, renders=[torch.zeros(1)], cycle_error=cycle_error)

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert raised.value.verdict == "INCONCLUSIVE"
    assert "capacity" in str(raised.value)
    assert "aborted before a terminal verdict" not in str(raised.value)
    assert len(ledger.records) == 1
    assert ledger.records[0][3] == "INCONCLUSIVE"
    detail = ledger.records[0][4]
    assert detail["cross_mode"] == "CAPACITY"
    # The worker refusal carries no measured tag, so the row has no numbers. It
    # names what refused in a bounded classification, never the refusal text
    # (fsdp_reload_price.capacity_classification says why).
    assert detail["measured"] is None
    assert detail["capacity_detail"] == f"{type(cycle_error).__name__}/untagged"
    assert "flux2" not in detail["capacity_detail"]
    # The operator still reads the refusal itself, in the sentence.
    assert "the FSDP launch cannot load flux2" in str(raised.value)
    # The capacity arm answered, so it also closes the transaction: one row,
    # and the process-local prejudgement is withdrawn.
    assert len(ledger.records) == 1
    assert retracted


def test_an_abort_before_the_retest_transaction_writes_no_terminal_row(
    monkeypatch, tmp_path,
):
    """An abort inside the price, before the retest transaction opens, must
    write no terminal row, or it revokes an inherited PASS for a proof that
    never started (the retest_opened comment in gather_ceremony_evidence)."""
    handle, ledger, model, published = _ceremony(monkeypatch, tmp_path)

    def explode(*_args, **_kwargs):
        raise RuntimeError("the fleet status read died mid price")

    monkeypatch.setattr(price_mod, "price_fsdp_clean_reload", explode)

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert raised.value.verdict == "INCONCLUSIVE"
    assert ledger.retests == []
    assert ledger.records == []
    assert published == []
    # Unlike the priced stop (gather_ceremony_evidence), an untyped abort here
    # still runs the cleanup unload: it costs a reload and leaves nothing
    # ambiguous.
    assert handle.latch_attempts == 0


def test_an_abort_after_the_retest_transaction_writes_its_terminal_row(
    monkeypatch, tmp_path,
):
    """The same untyped abort after the transaction opens must close it."""
    _handle, ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, renders=[torch.zeros(1)],
        cycle_error=RuntimeError("the reload cycle died"))

    with pytest.raises(FsdpGateProofError):
        _run(model)

    assert len(ledger.retests) == 1
    assert len(ledger.records) == 1
    assert ledger.records[0][3] == "INCONCLUSIVE"


def test_a_fitting_price_runs_the_whole_ceremony_unchanged(monkeypatch, tmp_path):
    rows = _shortfall(monkeypatch, checkpoint_gib=4, available_gib=96.0)
    handle, ledger, model, published = _ceremony(
        monkeypatch, tmp_path, status_rows=rows)

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert result["proof_kind"] == "fsdp_clean_reload"
    assert handle.unload_calls == 1
    assert len(ledger.retests) == 1
    assert published and published[-1][1] == "PASS"


def test_a_no_claim_price_runs_the_whole_ceremony_unchanged(monkeypatch, tmp_path):
    """A discrete or unmeasurable host makes no price claim; the ceremony runs
    as usual."""
    monkeypatch.setattr(mesh_safety_mod, "gpu_is_integrated", lambda: False)
    handle, _ledger, model, _published = _ceremony(monkeypatch, tmp_path)

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert handle.unload_calls == 1
    assert all(method != "status" for method, _args in handle.calls)


def test_cleanup_skips_the_unload_when_the_fleet_is_already_evicted():
    """An evicted fleet's residency is released when its procs stop, so a
    cleanup unload could only latch DIRTY."""
    calls: list = []
    handle = SimpleNamespace(
        world=2,
        defunct=True,
        call_all=lambda *args, **kwargs: calls.append(("unload", args)),
        _latch_ambiguous_mutation=lambda *args: calls.append(("latch", args)),
    )
    state: dict[str, bool] = {}
    logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None)

    confirmed = cleanup_aborted_fsdp_proof(
        handle, RuntimeError("proof aborted"), logger=logger, state=state)

    assert confirmed is False
    assert state == {"cleanup_attempted": True, "cleanup_confirmed": False}
    assert calls == []
    assert getattr(handle, "setup_cleanup_state", None) is None


def test_cleanup_still_unloads_a_live_fleet():
    calls: list = []
    handle = SimpleNamespace(
        world=2,
        defunct=False,
        call_all=lambda method, **kwargs: (
            calls.append(method) or [{"unloaded": True}, {"unloaded": True}]),
    )
    state: dict[str, bool] = {}

    confirmed = cleanup_aborted_fsdp_proof(
        handle, RuntimeError("proof aborted"),
        logger=SimpleNamespace(error=lambda *a, **k: None,
                               info=lambda *a, **k: None),
        state=state)

    assert confirmed is True
    assert calls == ["unload"]


def test_the_reload_is_repriced_after_the_baseline_render(monkeypatch):
    """The first price sees the host before the driver's text encoder lands;
    the second runs right before the reload cycle and stops in flight when the
    memory is gone (hardware, 2026-08-26; figures in
    gate_ceremony.gather_ceremony_evidence)."""
    def _host(rank, gib):
        return {"rank": rank, "host": {"mem_gib": {"MemAvailable": gib}}}

    statuses = [
        # First price: fits (39.8 GiB needed).
        [_host(rank, 48.4) for rank in (0, 1)],
        # After the baseline render: the text encoder is in memory.
        [_host(rank, 20.0) for rank in (0, 1)],
    ]

    calls: list[str] = []

    def call_all(method, **kwargs):
        calls.append(method)
        if method == "status":
            return statuses.pop(0) if statuses else [_host(0, 20.0), _host(1, 20.0)]
        if method == "gate_fsdp_reload_cycle":
            raise AssertionError("the reload cycle must not run after a failed reprice")
        return [{"rank": 0}, {"rank": 1}]

    handle = SimpleNamespace(world=2, setup_generation=1, call_all=call_all)
    monkeypatch.setattr(mesh_safety_mod, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(price_mod, "checkpoint_bytes", lambda _name: 60 * GIB)
    monkeypatch.setattr(price_mod, "checkpoint_kind", lambda _name: "bf16")

    first = price_mod.price_fsdp_clean_reload(handle, {"unet_name": "flux2-dev.safetensors"})
    assert first.applies and first.fits
    second = price_mod.price_fsdp_clean_reload(handle, {"unet_name": "flux2-dev.safetensors"})
    assert second.applies and not second.fits
    assert calls == ["status", "status"]


def test_gate_ceremony_reprices_before_the_reload_cycle():
    """The ceremony source runs the price twice: once before any mutation and
    once between the baseline render and the reload cycle."""
    import inspect

    from dgx_monarch.nodes import gate_ceremony

    source = inspect.getsource(gate_ceremony.gather_ceremony_evidence)
    first = source.index('price_fsdp_clean_reload')
    second = source.index('price_fsdp_clean_reload', first + 1)
    assert 'raw_cycle = handle.call_all' in source[second:]
    assert source.index('transaction_render', 0, second) < second
    # The in-flight stop names its stage, so its sentence does not claim that
    # nothing was unloaded (fsdp_reload_price.operator_sentence, 2026-08-26).
    assert 'stage="reprice"' in source[second:source.index('raw_cycle')]


def test_mid_build_device_oom_is_a_typed_capacity_refusal():
    from dgx_monarch.adapters.fsdp_shard_build import _shard_capacity_refusal
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    exc = _shard_capacity_refusal("blocks.0.lin.weight", RuntimeError("NV_ERR_NO_MEMORY"))
    assert isinstance(exc, StockLoadCapacityError)
    text = str(exc)
    assert "[dgxm:C" in text and "priced" in text and "Clear" in text


# A settled refusal inside the proof answers the combination.


def _capture_retractions(monkeypatch) -> list:
    """Record every process-verdict retraction the ceremony asks for."""
    retracted: list = []
    monkeypatch.setattr(
        gate_mod, "_retract_process_gate_verdicts",
        lambda tokens: retracted.append(list(tokens)))
    return retracted


def _class_p_card() -> str:
    from dgx_monarch.refusal import RefusalClass, refusal

    return refusal(
        RefusalClass.PHYSICS,
        "cfg-parallel got a model call of batch=1 on 2 ranks, and a batch of 1 "
        "does not split into 2 equal slices. Use a topology without cfg2.",
    )


def _ring_pad_card(*, waivable: bool) -> str:
    from dgx_monarch import accuracy_waiver
    from dgx_monarch.refusal import RefusalClass, refusal

    if waivable:
        return refusal(
            RefusalClass.KNOWN_WRONG,
            "this sequence length needs divisibility padding, and the exact pad "
            "exclusion is ulysses-only.",
            guard="ring_pad", waivable=True,
            panel_action=accuracy_waiver.panel_action("ring_pad"))
    return refusal(
        RefusalClass.KNOWN_WRONG,
        "this sequence length needs divisibility padding and there is no "
        "waiver for this refusal on this build.",
        guard="ring_pad", waivable=False)


def _sol_attn_card() -> str:
    """A waivable class K card whose remedy is a kernel, not a token count."""
    from dgx_monarch import accuracy_waiver
    from dgx_monarch.refusal import RefusalClass, refusal

    return refusal(
        RefusalClass.KNOWN_WRONG,
        "the sol-attn kernel is approximate by construction and is NOT "
        "identity preserving.",
        guard="sol_attn:minimax_h3", waivable=True,
        panel_action=accuracy_waiver.panel_action("sol_attn:minimax_h3"),
        troubleshooting=84)


def _worker_refusal(text: str):
    from monarch.actor import ActorError

    return ActorError(RuntimeError(text))


def _tag_of(exc: BaseException):
    from dgx_monarch.consent_observe import refusal_text
    from dgx_monarch.refusal import parse_leading_refusal_tag

    # Never str() the wrapper: its __str__ expands the whole remote traceback.
    return parse_leading_refusal_tag(refusal_text(exc))


def test_a_settled_class_p_refusal_stays_on_the_wire(monkeypatch, tmp_path):
    """A class P refusal is an answer, not an abort, so its class reaches the
    caller."""
    from monarch.actor import ActorError

    _capture_retractions(monkeypatch)
    _handle, _ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, renders=[_worker_refusal(_class_p_card())])

    with pytest.raises(ActorError) as caught:
        _run(model)

    assert "aborted before a terminal verdict" not in str(caught.value)
    tag = _tag_of(caught.value)
    assert tag is not None and tag.refusal_class.value == "P"


def test_a_settled_refusal_closes_its_retest_with_one_terminal_row(
    monkeypatch, tmp_path,
):
    from monarch.actor import ActorError

    _capture_retractions(monkeypatch)
    _handle, ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, renders=[_worker_refusal(_class_p_card())])

    with pytest.raises(ActorError):
        _run(model)

    assert len(ledger.retests) == 1
    assert len(ledger.records) == 1
    _key, _artifacts, _commit, verdict, detail, _context = ledger.records[0]
    assert verdict == "INCONCLUSIVE"
    assert detail["refusal_class"] == "P"
    assert "cross_mode" not in detail
    assert "inconclusive_kind" not in detail
    assert detail["measured"] is None
    # The row names what refused without carrying the refusal prose.
    assert "cfg-parallel" not in detail["refusal_type"]


def test_a_settled_refusal_retracts_exactly_what_it_published(
    monkeypatch, tmp_path,
):
    """The next queue must meet the guard again, not a cached prejudgement."""
    from monarch.actor import ActorError

    retracted = _capture_retractions(monkeypatch)
    _handle, _ledger, model, published = _ceremony(
        monkeypatch, tmp_path, renders=[_worker_refusal(_class_p_card())])

    with pytest.raises(ActorError):
        _run(model)

    assert retracted == [published[0][0]]


def test_a_waivable_class_k_refusal_is_restated_at_the_ceremony_boundary(
    monkeypatch, tmp_path,
):
    """The card's waiver offer cannot be spent inside a ceremony, so the
    boundary restates the refusal as unwaivable, with the guard's remedy taken
    from the boundary's own table, not from the card
    (gate_abort.ceremony_waiver_boundary_error says why)."""
    retracted = _capture_retractions(monkeypatch)
    _handle, ledger, model, _published = _ceremony(
        monkeypatch, tmp_path,
        renders=[_worker_refusal(_ring_pad_card(waivable=True))])

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    text = str(raised.value)
    assert text.startswith("[dgxm:K")
    assert "cannot be waived" in text
    assert "ring_pad" in text
    assert "Open the DGX Monarch panel" not in text
    assert raised.value.verdict == "INCONCLUSIVE"
    # The guard's own remedy and its own entries survive the restatement.
    assert "divisibility padding" in text
    assert "#21" in text and "#49" in text
    assert f"#{gate_abort.TROUBLESHOOTING}" in text
    assert len(ledger.records) == 1
    detail = ledger.records[0][4]
    assert detail["refusal_class"] == "K"
    assert detail["refusal_guard"] == "ring_pad"
    assert retracted


def test_the_restated_boundary_names_its_own_operator_entry():
    """The entry number is in shipped text, so it is pinned, as entry 91 is in
    tests/test_upstream_gated_artifact.py."""
    assert gate_abort.TROUBLESHOOTING == 92


def test_the_restated_boundary_keeps_the_sol_attn_remedy(monkeypatch, tmp_path):
    """A sol-attn refusal is answered by an exact kernel, never by padding."""
    _capture_retractions(monkeypatch)
    _handle, ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, renders=[_worker_refusal(_sol_attn_card())])

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    text = str(raised.value)
    assert text.startswith("[dgxm:K")
    assert "cannot be waived" in text
    assert "sol_attn:minimax_h3" in text
    assert "SAGE_" in text and "TORCH_FLASH" in text
    assert "divisibility padding" not in text
    assert "#84" in text and "#21" not in text
    assert f"#{gate_abort.TROUBLESHOOTING}" in text
    assert ledger.records[0][4]["refusal_guard"] == "sol_attn:minimax_h3"


def test_a_class_k_refusal_that_no_waiver_clears_passes_through(
    monkeypatch, tmp_path,
):
    """The boundary wrap is keyed on the waiver, not on the class."""
    from monarch.actor import ActorError

    _capture_retractions(monkeypatch)
    _handle, _ledger, model, _published = _ceremony(
        monkeypatch, tmp_path,
        renders=[_worker_refusal(_ring_pad_card(waivable=False))])

    with pytest.raises(ActorError) as caught:
        _run(model)

    tag = _tag_of(caught.value)
    assert tag is not None and tag.refusal_class.value == "K"
    assert tag.waivable is False


def test_an_untyped_abort_carries_its_cause_and_closes_its_row(
    monkeypatch, tmp_path,
):
    retracted = _capture_retractions(monkeypatch)
    _handle, ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, renders=[torch.zeros(1)],
        cycle_error=RuntimeError("the reload cycle exploded"))

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    text = str(raised.value)
    assert text.startswith(
        "FSDP clean-reload proof aborted before a terminal verdict: ")
    assert "the reload cycle exploded" in text
    assert raised.value.verdict == "INCONCLUSIVE"
    assert len(ledger.records) == 1
    detail = ledger.records[0][4]
    assert ledger.records[0][3] == "INCONCLUSIVE"
    assert detail["inconclusive_reasons"] == [
        "the FSDP clean-reload proof aborted before a terminal verdict"]
    assert "refusal_class" not in detail
    assert len(ledger.retests) == 1
    # Nothing answered, so the process-local denial stands.
    assert retracted == []


def test_an_incomplete_proof_closes_its_row_and_retracts_nothing(
    monkeypatch, tmp_path,
):
    retracted = _capture_retractions(monkeypatch)
    broken = [_row(0, setup_generation=4), _row(1, setup_generation=4)]
    broken[1]["fsdp_ready"] = False
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=broken)
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True
    monkeypatch.setattr(
        gate_mod, "_publish_process_gate_verdicts",
        lambda tokens, verdict, ceremony=None: None)

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert "FSDP clean-reload proof is incomplete" in str(raised.value)
    assert len(ledger.records) == 1
    assert ledger.records[0][3] == "INCONCLUSIVE"
    assert "refusal_class" not in ledger.records[0][4]
    assert retracted == []


def test_an_unproven_baseline_unload_closes_its_row(monkeypatch, tmp_path):
    retracted = _capture_retractions(monkeypatch)
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=_fsdp_cycle_rows(),
        initial_unload_response=[{"unloaded": True}])
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True
    monkeypatch.setattr(
        gate_mod, "_publish_process_gate_verdicts",
        lambda tokens, verdict, ceremony=None: None)

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert "baseline all-rank unload" in str(raised.value)
    assert len(ledger.records) == 1
    assert ledger.records[0][3] == "INCONCLUSIVE"
    assert retracted == []


def _record_gate_log(monkeypatch) -> list:
    """Capture what the abort arm tells the operator, formatted."""
    lines: list = []

    def error(message, *args):
        lines.append(message % args if args else message)

    recorder = SimpleNamespace(error=error, warning=error, info=error)
    monkeypatch.setattr(gate_mod, "log", recorder)
    # The failed-withdrawal warning is the abort module's own, not the gate's.
    monkeypatch.setattr(gate_abort, "log", recorder)
    return lines


def test_a_settled_stop_before_the_prejudgement_claims_no_retraction(
    monkeypatch, tmp_path,
):
    """The priced stop's class C refusal comes before the ceremony publishes a
    prejudgement, so the log must not claim one was withdrawn
    (gate_abort.close_aborted_proof and log_aborted_proof say why)."""
    lines = _record_gate_log(monkeypatch)
    _capture_retractions(monkeypatch)
    rows = _shortfall(monkeypatch)
    _handle, _ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, status_rows=rows)

    with pytest.raises(FsdpGateProofError):
        _run(model)

    settled = [line for line in lines if "settled class C refusal" in line]
    assert settled and "no process-local denial was retracted" in settled[0]


def test_a_failed_retraction_reports_the_denial_that_stands(
    monkeypatch, tmp_path,
):
    """A withdrawal that raised leaves the denial in force, and says so."""
    from monarch.actor import ActorError

    lines = _record_gate_log(monkeypatch)

    def refuse(_tokens):
        raise RuntimeError("the verdict maps are wedged")

    monkeypatch.setattr(gate_mod, "_retract_process_gate_verdicts", refuse)
    _handle, _ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, renders=[_worker_refusal(_class_p_card())])

    with pytest.raises(ActorError):
        _run(model)

    settled = [line for line in lines if "settled class P refusal" in line]
    assert settled and "no process-local denial was retracted" in settled[0]
    assert any("was not withdrawn" in line for line in lines)


def test_a_settled_refusal_that_retracted_says_so(monkeypatch, tmp_path):
    from monarch.actor import ActorError

    lines = _record_gate_log(monkeypatch)
    _capture_retractions(monkeypatch)
    _handle, _ledger, model, _published = _ceremony(
        monkeypatch, tmp_path, renders=[_worker_refusal(_class_p_card())])

    with pytest.raises(ActorError):
        _run(model)

    assert any("the process-local denial was retracted" in line
               for line in lines)


def test_settled_refusal_tag_is_total_for_a_hostile_diagnostic():
    """settled_refusal_tag never raises: the abort arm still has residency to
    clean up."""
    from dgx_monarch.nodes.gate_fsdp import settled_refusal_tag

    class HostileFailure(RuntimeError):
        def __str__(self):
            raise ValueError("unreadable")

        @property
        def args(self):
            raise ValueError("unreadable")

    assert settled_refusal_tag(HostileFailure()) is None
