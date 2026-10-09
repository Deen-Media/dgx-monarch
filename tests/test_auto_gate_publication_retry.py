"""Auto-gate cleanup and verdict publication retry regressions."""
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
from dgx_monarch.nodes import gate_process_state
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
    _cross_mode_rig,
    _run,
)


class _PublicationInterrupt(BaseException):
    def __bool__(self):
        return False


_CLEANUP_RETRY_CASES = [
    pytest.param((RuntimeError, None), id="ordinary-then-success"),
    pytest.param((_PublicationInterrupt, None), id="cancellation-then-success"),
    pytest.param(
        (RuntimeError, _PublicationInterrupt),
        id="ordinary-then-cancellation",
    ),
    pytest.param(
        (_PublicationInterrupt, RuntimeError),
        id="cancellation-then-ordinary",
    ),
    pytest.param((RuntimeError, ValueError), id="ordinary-exhaustion"),
]


def _cleanup_retry_errors(failure_types, label):
    return [
        failure_type(label) if failure_type is not None else None
        for failure_type in failure_types
    ]


def _expected_cleanup_retry_error(errors):
    cancellation = next(
        (
            error
            for error in errors
            if error is not None and not isinstance(error, Exception)
        ),
        None,
    )
    if cancellation is not None:
        return cancellation
    if errors[-1] is None:
        return None
    return errors[0]


@pytest.mark.parametrize("failure_types", _CLEANUP_RETRY_CASES)
def test_auto_gate_active_restore_retry_semantics(failure_types, monkeypatch):
    errors = _cleanup_retry_errors(failure_types, "active restore failure")

    class Active:
        def __init__(self):
            self.attempts = 0
            self.value = True

        @property
        def on(self):
            return self.value

        @on.setter
        def on(self, value):
            error = errors[self.attempts]
            self.attempts += 1
            if error is not None:
                raise error
            self.value = value

    active = Active()
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_ACTIVE", active)
    result = common._auto_gate_impl.restore_auto_gate_active(False)

    assert result is _expected_cleanup_retry_error(errors)
    assert active.attempts == 2
    assert active.on is (errors[-1] is not None)


@pytest.mark.parametrize("failure_types", _CLEANUP_RETRY_CASES)
def test_auto_gate_claim_release_retry_semantics(failure_types, monkeypatch):
    errors = _cleanup_retry_errors(failure_types, "claim release failure")

    class Condition:
        def __init__(self):
            self.attempts = 0
            self.notifications = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def notify_all(self):
            error = errors[self.attempts]
            self.attempts += 1
            if error is not None:
                raise error
            self.notifications += 1

    token = ("combo", "artifacts", "commit", "context")
    condition = Condition()
    running = {token}
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_CONDITION", condition)
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_RUNNING", running)
    result = common._auto_gate_impl.release_auto_gate_claim(token)

    assert result is _expected_cleanup_retry_error(errors)
    assert condition.attempts == 2
    assert condition.notifications == int(errors[-1] is None)
    assert token not in running


_PUBLICATION_RETRY_CASES = [
    pytest.param((RuntimeError,), id="ordinary-then-success"),
    pytest.param((_PublicationInterrupt,), id="cancellation-then-success"),
    pytest.param(
        (RuntimeError, _PublicationInterrupt),
        id="ordinary-then-cancellation",
    ),
]


@pytest.mark.parametrize(
    "failure_types",
    _PUBLICATION_RETRY_CASES,
)
def test_auto_gate_final_publication_retry_semantics(monkeypatch, failure_types):
    token = ("combo", "artifacts", "commit", "context")
    publish_attempts = []
    real_publish = common._record_process_gate_verdicts
    publication_errors = [
        failure_type("interrupted auto final publication")
        for failure_type in failure_types
    ]
    publication_cancel = next(
        (
            error
            for error in publication_errors
            if not isinstance(error, Exception)
        ),
        None,
    )

    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", token))
    monkeypatch.setattr(
        gate_mod,
        "run_identity_ceremony",
        lambda *_args, **_kwargs: {
            "verdict": "PASS",
            "_gate_token": token,
            "_gate_tokens": [token],
        },
    )

    def fail_once_then_publish(tokens, verdict, ceremony=None):
        publish_attempts.append(verdict)
        real_publish(tokens, verdict, ceremony)
        attempt = len(publish_attempts) - 1
        if attempt < len(publication_errors):
            raise publication_errors[attempt]

    monkeypatch.setattr(
        common, "_record_process_gate_verdicts", fail_once_then_publish)
    common._AUTO_GATE_ACTIVE.on = False

    if publication_cancel is None:
        result = common._maybe_auto_gate(
            SimpleNamespace(), {"kind": "ksampler", "steps": 2}, {}, 1.0, 2)
        assert result == "PASS"
    else:
        with pytest.raises(_PublicationInterrupt) as caught:
            common._maybe_auto_gate(
                SimpleNamespace(),
                {"kind": "ksampler", "steps": 2},
                {},
                1.0,
                2,
            )
        assert caught.value is publication_cancel

    assert publish_attempts == ["PASS", "PASS"]
    assert common._process_gate_verdict(token) == "PASS"
    assert token not in common._AUTO_GATE_RUNNING


@pytest.mark.parametrize(
    "failure_types",
    _PUBLICATION_RETRY_CASES,
)
def test_retesting_publication_retry_semantics(
    monkeypatch, tmp_path, failure_types,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)],
        worker_args,
    )
    publish_attempts = []
    published_tokens = []
    real_publish = common._record_process_gate_verdicts
    publication_errors = [
        failure_type("interrupted RETESTING publication")
        for failure_type in failure_types
    ]
    publication_cancel = next(
        (
            error
            for error in publication_errors
            if not isinstance(error, Exception)
        ),
        None,
    )
    retest_attempts = 0

    def fail_first_retesting_publication(tokens, verdict, ceremony=None):
        nonlocal retest_attempts
        publish_attempts.append(verdict)
        published_tokens.append(tuple(tokens))
        real_publish(tokens, verdict, ceremony)
        if verdict == "INCONCLUSIVE":
            attempt = retest_attempts
            retest_attempts += 1
            if attempt < len(publication_errors):
                raise publication_errors[attempt]

    monkeypatch.setattr(
        common, "_record_process_gate_verdicts", fail_first_retesting_publication)

    if publication_cancel is None:
        result = _run(model)
        assert result["verdict"] == "PASS"
        assert publish_attempts == ["INCONCLUSIVE", "INCONCLUSIVE", "PASS"]
        assert handle.calls[0][0] == "provenance_baseline"
        assert handle.calls[1][0] == "unload"
        terminal_verdict = "PASS"
        terminal_tokens = published_tokens[-1]
    else:
        with pytest.raises(_PublicationInterrupt) as caught:
            _run(model)
        assert caught.value is publication_cancel
        assert publish_attempts == ["INCONCLUSIVE", "INCONCLUSIVE"]
        terminal_verdict = "INCONCLUSIVE"
        terminal_tokens = published_tokens[0]

    assert published_tokens
    assert all(
        common._process_gate_verdict(token) == terminal_verdict
        for token in terminal_tokens
    )


@pytest.mark.parametrize(
    "failure_types",
    _PUBLICATION_RETRY_CASES,
)
def test_explicit_final_publication_retry_semantics(
    monkeypatch, tmp_path, failure_types,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    _handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)],
        worker_args,
    )
    publish_attempts = []
    published_tokens = []
    final_attempts = 0
    real_publish = common._record_process_gate_verdicts
    publication_errors = [
        failure_type("interrupted explicit final publication")
        for failure_type in failure_types
    ]
    publication_cancel = next(
        (
            error
            for error in publication_errors
            if not isinstance(error, Exception)
        ),
        None,
    )

    def fail_first_final_publication(tokens, verdict, ceremony=None):
        nonlocal final_attempts
        publish_attempts.append(verdict)
        published_tokens.append(tuple(tokens))
        if verdict == "PASS":
            final_attempts += 1
        real_publish(tokens, verdict, ceremony)
        if verdict == "PASS" and final_attempts <= len(publication_errors):
            raise publication_errors[final_attempts - 1]

    monkeypatch.setattr(
        common, "_record_process_gate_verdicts", fail_first_final_publication)

    if publication_cancel is None:
        result = _run(model)
        assert result["verdict"] == "PASS"
        assert result["_gate_tokens"]
    else:
        with pytest.raises(_PublicationInterrupt) as caught:
            _run(model)
        assert caught.value is publication_cancel

    assert publish_attempts == ["INCONCLUSIVE", "PASS", "PASS"]
    final_tokens = published_tokens[-1]
    assert final_tokens
    assert all(
        common._process_gate_verdict(tuple(token)) == "PASS"
        for token in final_tokens
    )
