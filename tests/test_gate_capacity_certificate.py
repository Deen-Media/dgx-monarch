"""Capacity certificate audit rows from a capacity-skipped cross leg."""
import torch

from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
    _cross_mode_rig,
    _run,
)

_V9_CERTIFICATE = {
    "algorithm": "memcmp-tensor-sampled-v1",
    "digest": "e4" + "a" * 62,
    "complete": True,
    "tensors_verified": 1042,
    "tensors_total": 1042,
    "bytes_verified": 66266529792,
    "checkpoint_bytes": 66266529792,
    "artifact_signature": "9d1c0b47ee2a6f3184c5b70a",
    "file_identity": "66306:12583041:66266529792:1754300000000000000:1754300000000000000",
}
_V9_MEASURED = {
    "probe": "stock_load_preflight",
    "checkpoint_bytes": 66266529792,
    "mem_available_bytes": 41234567168,
    "driver_projection_bytes": None,
    "worker_projection_bytes": 66266529792,
    "headroom_bytes": -25031962624,
}


def _certified_cycle(**overrides):
    row = {"conclusive": True, "transitions": ["lazy"], "family": "krea2",
           "slab_active": True, "certificate": dict(_V9_CERTIFICATE)}
    row.update(overrides)
    return [row]


def _capacity_refusal(with_numbers: bool = True):
    from dgx_monarch.gate_audit import measured_tag
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    text = "stock residency cannot load model.safetensors"
    if with_numbers:
        text = f"{text} {measured_tag(_V9_MEASURED)}"
    return StockLoadCapacityError(text)


def test_ceremony_capacity_skip_records_a_certificate_row(monkeypatch, tmp_path):
    """The stock leg could not load, so slab exactness stays unproven and the
    verdict stays INCONCLUSIVE. What was proven, that the slab holds the
    checkpoint's own bytes, lands in an additional v9 audit row under its
    own key namespace, where it can never displace a verdict."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), _capacity_refusal()], wa,
        cycle_response=_certified_cycle())
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "CAPACITY"
    assert result["cross_mode"]["certificate"]["ranks_certified"] == 1
    assert result["cross_mode"]["measured"] == _V9_MEASURED
    assert [record[3] for record in ledger.records] == [
        "INCONCLUSIVE", "INCONCLUSIVE", "CAPACITY_CERTIFIED"]
    key, artifacts, _commit, _verdict, detail, context = ledger.records[2]
    assert key == "audit:capacity:" + ledger.records[0][0]
    assert artifacts is ledger.records[0][1]
    assert (detail["slab_resident"], detail["stock_available"]) == (True, False)
    assert detail["certificate_origin"] == "gate_cross_leg"
    assert "consent_id" not in detail  # nothing consented; nothing invented
    assert detail["measured"]["checkpoint_bytes"] == 66266529792
    assert detail["unet_name"] == "model.safetensors"
    assert detail["run_id"] == "x"
    assert context["record"] == "capacity_certificate"


def test_ceremony_cross_leg_pass_records_no_certificate_row(monkeypatch, tmp_path):
    """Certificates present, but nothing was capacity-limited: the stock leg
    ran, so there is no capacity claim to record."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa,
        cycle_response=_certified_cycle())
    result = _run(model)
    assert result["cross_mode"]["verdict"] == "PASS"
    assert [record[3] for record in ledger.records] == ["PASS", "PASS"]


def test_ceremony_capacity_skip_without_measured_numbers_records_nothing(
    monkeypatch, tmp_path,
):
    """A capacity claim with no numbers is not evidence. Fail closed: no row
    rather than an unnumbered one."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), _capacity_refusal(with_numbers=False)], wa,
        cycle_response=_certified_cycle())
    result = _run(model)
    assert result["cross_mode"]["verdict"] == "CAPACITY"
    assert result["cross_mode"]["measured"] is None
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE", "INCONCLUSIVE"]


def test_ceremony_capacity_skip_without_a_certificate_records_nothing(
    monkeypatch, tmp_path,
):
    """A worker too old to certify reports no certificate, and no certificate
    means no capacity row, never an uncertified claim."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), _capacity_refusal()], wa,
        cycle_response=_certified_cycle(certificate=None))
    result = _run(model)
    assert result["cross_mode"]["certificate"] is None
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE", "INCONCLUSIVE"]


_V9_SLAB_MEASURED = {
    "probe": "worker_slab_load",
    "checkpoint_bytes": 66266529792,
    "mem_available_bytes": 41234567168,
    "driver_projection_bytes": None,
    "worker_projection_bytes": None,
    "headroom_bytes": -29323638272,
}


def test_ceremony_meeting_the_slab_wall_stays_inconclusive(monkeypatch, tmp_path):
    """The vouching ceremony cannot vouch for a load no rank could place.

    The slab rungs have a capacity wall, so a ceremony whose slab leg refuses
    on capacity ends where a stock-leg refusal does: a capacity reason, an
    INCONCLUSIVE verdict and no vouching verdict written. The refusal carries
    the slab probe's own block, which the ledger vocabulary admits.
    """
    from dgx_monarch.gate_audit import measured_tag
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    refusal = StockLoadCapacityError(
        "slab residency cannot load model.safetensors on this host "
        + measured_tag(_V9_SLAB_MEASURED))
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), refusal], wa,
        cycle_response=_certified_cycle())
    result = _run(model)

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "CAPACITY"
    assert result["cross_mode"]["measured"] == _V9_SLAB_MEASURED
    assert "PASS" not in [record[3] for record in ledger.records]
