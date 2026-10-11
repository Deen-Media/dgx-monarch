from __future__ import annotations

from types import SimpleNamespace

import pytest

from dgx_monarch.cli.update_payload_ownership import PayloadOwnershipError
from dgx_monarch.cli.update_pin_diagnostic import PinDiagnostic, inspect_match, pin_refusal_message


@pytest.mark.parametrize("value", ["/private/token", "../secret", "x\nspoof", "x\x1b[31m", "x\u202eevil", "x" * 241])
def test_unsafe_context_is_not_printed(value):
    diagnostic = PinDiagnostic("record_mismatch", value, value)
    assert diagnostic.distribution is None
    assert diagnostic.relative_path is None
    assert value not in diagnostic.terminal_message()


def test_unknown_exception_is_not_disclosed_and_remains_false():
    def probe():
        raise OSError("/private/credential?token=secret")

    matched, diagnostic = inspect_match(probe)
    assert not matched
    assert diagnostic == PinDiagnostic("inspection_failed")
    assert "secret" not in pin_refusal_message(SimpleNamespace(last_diagnostic=diagnostic))


def test_known_ownership_error_is_preserved_without_raw_exception():
    def probe():
        raise PayloadOwnershipError(
            "sensitive raw message", reason="pinned_payload_claim",
            distribution="ultralytics", relative_path="tests/__init__.py",
        )

    matched, diagnostic = inspect_match(probe)
    assert not matched
    assert diagnostic == PinDiagnostic("pinned_payload_claim", "ultralytics", "tests/__init__.py")
    message = pin_refusal_message(SimpleNamespace(last_diagnostic=diagnostic))
    assert "ultralytics" in message and "tests/__init__.py" in message
    assert "sensitive" not in message


def test_success_clears_diagnostic_and_false_preserves_refusal():
    assert inspect_match(lambda: True) == (True, None)
    assert inspect_match(lambda: False) == (False, PinDiagnostic("payload_mismatch"))


def test_cancellation_is_not_converted_to_false():
    def probe():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        inspect_match(probe)
