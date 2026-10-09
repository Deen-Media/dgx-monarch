"""Check the opt-in debug fault between load agreement and checkpoint reads.

Cross-rank agreement rejects missing or truncated files before any rank loads,
so those faults cannot exercise the post-load readiness exchange. This knob
provides a repeatable fault in that gap (docs/VALIDATION.md, debug-knob record,
2026-09-06).

It must be absent by default, require the acceptance marker, target exactly
one rank, log its activation, and raise after agreement but before reading
weights.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dgx_monarch.actor import load_fault
from dgx_monarch.refusal import parse_refusal_tag

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch" / "actor"


@pytest.fixture(autouse=True)
def _disarmed(monkeypatch):
    """No test inherits another's armed flag or another's environment."""
    monkeypatch.delenv(load_fault.RANK_ENV, raising=False)
    monkeypatch.delenv(load_fault.ACCEPTANCE_ENV, raising=False)
    monkeypatch.setattr(load_fault, "_ARMED", False)
    yield
    monkeypatch.setattr(load_fault, "_ARMED", False)


def test_the_fault_is_off_when_neither_variable_is_set(caplog):
    with caplog.at_level(logging.WARNING):
        assert load_fault.arm(0) is False
        assert load_fault.arm(1) is False
    assert load_fault.check() is None
    assert caplog.text == ""


def test_a_worker_that_never_armed_the_knob_still_loads():
    """A load path that runs without a setup, as every store test does."""
    assert load_fault.check() is None


def test_an_unreadable_rank_stays_off_and_says_so(caplog, monkeypatch):
    monkeypatch.setenv(load_fault.RANK_ENV, "the second box")
    monkeypatch.setenv(load_fault.ACCEPTANCE_ENV, "1")
    with caplog.at_level(logging.WARNING):
        assert load_fault.arm(1) is False
    assert load_fault.check() is None
    assert "does not name a rank" in caplog.text


def test_the_knob_is_refused_without_the_acceptance_marker(caplog, monkeypatch):
    monkeypatch.setenv(load_fault.RANK_ENV, "1")
    with caplog.at_level(logging.WARNING):
        assert load_fault.arm(1) is False
    assert load_fault.check() is None
    assert load_fault.ACCEPTANCE_ENV in caplog.text
    assert "stays off" in caplog.text


@pytest.mark.parametrize("marker", ["0", "", "true", "yes", "2"])
def test_only_the_exact_marker_admits_the_knob(marker, monkeypatch):
    monkeypatch.setenv(load_fault.RANK_ENV, "1")
    monkeypatch.setenv(load_fault.ACCEPTANCE_ENV, marker)
    assert load_fault.arm(1) is False
    assert load_fault.check() is None


def test_the_named_rank_fails_and_the_other_rank_does_not(monkeypatch):
    monkeypatch.setenv(load_fault.RANK_ENV, "1")
    monkeypatch.setenv(load_fault.ACCEPTANCE_ENV, "1")

    assert load_fault.arm(0) is False
    assert load_fault.check() is None

    assert load_fault.arm(1) is True
    with pytest.raises(load_fault.InjectedLoadFaultError) as caught:
        load_fault.check()
    assert "acceptance fault injected on this rank" in str(caught.value)


def test_the_refusal_is_class_p_with_no_guard_and_names_the_way_out(monkeypatch):
    """No consent can clear a box that was told to fail, so there is no card."""
    monkeypatch.setenv(load_fault.RANK_ENV, "0")
    monkeypatch.setenv(load_fault.ACCEPTANCE_ENV, "1")
    load_fault.arm(0)
    with pytest.raises(load_fault.InjectedLoadFaultError) as caught:
        load_fault.check()
    message = str(caught.value)
    tag = parse_refusal_tag(message)
    assert tag is not None
    assert tag.refusal_class.value == "P"
    assert tag.guard is None and tag.waivable is False
    assert "instead" in message
    assert f"#{load_fault.TROUBLESHOOTING}" in message


def test_the_fault_repeats_until_the_variable_goes_away(monkeypatch):
    """A leg must be repeatable, which a timed permission change is not."""
    monkeypatch.setenv(load_fault.RANK_ENV, "1")
    monkeypatch.setenv(load_fault.ACCEPTANCE_ENV, "1")
    load_fault.arm(1)
    for _ in range(3):
        with pytest.raises(load_fault.InjectedLoadFaultError):
            load_fault.check()
    monkeypatch.delenv(load_fault.RANK_ENV)
    assert load_fault.arm(1) is False
    assert load_fault.check() is None


def test_arming_is_loud_and_names_both_variables_and_both_ranks(caplog, monkeypatch):
    monkeypatch.setenv(load_fault.RANK_ENV, "1")
    monkeypatch.setenv(load_fault.ACCEPTANCE_ENV, "1")
    with caplog.at_level(logging.WARNING):
        load_fault.arm(0)
        load_fault.arm(1)
    armed = [record for record in caplog.records
             if record.levelno == logging.WARNING
             and "DEBUG LOAD FAULT ARMED" in record.getMessage()]
    assert len(armed) == 2
    assert "loads as usual" in armed[0].getMessage()
    assert "fails" in armed[1].getMessage()
    assert load_fault.RANK_ENV in armed[0].getMessage()
    assert load_fault.ACCEPTANCE_ENV in armed[0].getMessage()


def test_the_knob_is_read_at_setup_and_nowhere_else():
    """Read once when the worker comes up, so nothing can arm a live render."""
    env = (SRC / "worker_env.py").read_text()
    assert env.count("load_fault.arm(rank)") == 1
    setup = env.index("def setup_impl(")
    assert env.index("load_fault.arm(rank)") > setup
    fault = (SRC / "load_fault.py").read_text()
    arm_body = fault[fault.index("def arm("):fault.index("def check(")]
    assert fault.count("os.environ") == arm_body.count("os.environ") == 2
    for module in sorted(SRC.glob("*.py")):
        if module.name in ("load_fault.py", "worker_env.py"):
            continue
        assert load_fault.RANK_ENV not in module.read_text()


def test_both_variables_survive_the_worker_loop_environment_filter():
    """A drop-in that never reaches the actor would make the leg unrunnable.

    The worker loop deletes every name outside its reviewed list before Monarch
    is imported, and actor launchers inherit what is left, so a knob set on the
    unit is only real if both names are on that list.
    """
    from dgx_monarch.cli import worker_process_env

    environment = {
        "PATH": "/usr/bin",
        load_fault.RANK_ENV: "1",
        load_fault.ACCEPTANCE_ENV: "1",
    }
    assert worker_process_env.filter_worker_environment(environment) == 0
    assert environment[load_fault.RANK_ENV] == "1"
    assert environment[load_fault.ACCEPTANCE_ENV] == "1"


def test_the_fault_fires_after_the_price_and_before_the_weights_are_read():
    """The window the readiness exchange covers, and no earlier moment.

    Earlier is what the cross-rank agreement already refuses, which is why no
    injectable fault reached this seam. Later has read the file.
    """
    load = (SRC / "store_load.py").read_text()
    assert load.count("load_fault.check()") == 1
    fault = load.index("load_fault.check()")
    assert fault > load.index("store_residency.preload_capacity_check(")
    assert fault < load.index("comfy_bridge.load_diffusion_model_slab(")
    assert fault < load.index("comfy_sd.load_diffusion_model(load_path")


def test_the_failing_load_sits_inside_the_readiness_guard():
    """So the peer refuses in seconds instead of waiting out the group timeout."""
    sample = (SRC / "sample_protocol.py").read_text()
    guarded = sample.index("store_fsdp.ensure(")
    assert guarded < sample.index("readiness.not_ready(exc)")
    assert guarded < sample.index("readiness.ready_or_raise()")
