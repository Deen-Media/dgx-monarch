from __future__ import annotations

from types import SimpleNamespace

import pytest

from dgx_monarch.cli import update_pin_diagnostic
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


def test_a_record_outside_site_packages_names_its_cause_and_how_to_see_it():
    """Issue 158's first ask: the refusal says why and what to do. pip's prefix
    bytecode no longer reaches this reason, so a reinstall without the prefix
    would write the same rows again: the remedy shows the rows instead."""
    diagnostic = PinDiagnostic("record_outside_site", "xfuser", "xfuser-0.7.0+dgxm.npuimport1.dist-info/RECORD")
    message = pin_refusal_message(SimpleNamespace(last_diagnostic=diagnostic))
    assert "lists files outside site-packages that are not bytecode compiled for its own sources" in message
    assert "distribution=xfuser" in message and "xfuser-0.7.0+dgxm.npuimport1.dist-info/RECORD" in message
    assert "grep '^\\.\\./' <site-packages>/<relative path>" in message
    assert "a reinstall writes the same rows again" in message
    assert "PYTHONPYCACHEPREFIX" not in message and "--force-reinstall" not in message
    assert "docs/TROUBLESHOOTING.md #155" in message
    assert message.endswith("no package repair was attempted.")
    assert "Inspect the selected environment" not in message


@pytest.mark.parametrize(("reason", "says"), [
    ("record_malformed", "RECORD is malformed"),
    ("distribution_duplicated", "more than one metadata directory"),
    ("shared_payload_missing", "Reinstall torchmonarch"),
    ("pinned_payload_claim", "Inspect the selected environment's package ownership"),
])
def test_each_ownership_reason_carries_its_own_remedy(reason, says):
    message = pin_refusal_message(SimpleNamespace(last_diagnostic=PinDiagnostic(reason, "other")))
    assert says in message and message.endswith("no package repair was attempted.")


def test_a_worker_refusal_line_round_trips_and_stays_bounded():
    parse_attest_line = update_pin_diagnostic.parse_attest_line
    diagnostic = PinDiagnostic("record_outside_site", "xfuser", "xfuser-1.dist-info/RECORD")
    assert parse_attest_line("noise\n" + diagnostic.attest_line() + "\n") == diagnostic
    hostile = "DGXM_ATTEST_REFUSAL=record_outside_site distribution=x\x1b[31m path=../../etc/passwd"
    parsed = parse_attest_line(hostile)
    assert parsed is None or (parsed.distribution is None and parsed.relative_path is None)
    forged = parse_attest_line("DGXM_ATTEST_REFUSAL=made_up distribution=x path=y")
    assert forged is not None and forged.reason == "inspection_failed"
    assert parse_attest_line("RELEASE_MATCH") is None


def test_prior_drift_names_each_host_on_the_terminal_and_keeps_the_receipt_text_bounded(capsys):
    prior_drift_message = update_pin_diagnostic.prior_drift_message
    releases = SimpleNamespace(prior_refusals=[
        ("worker-b", PinDiagnostic("record_outside_site", "xfuser", "xfuser-1.dist-info/RECORD"))])
    message = prior_drift_message(releases)
    assert message.startswith("worker hosts do not exactly match the current driver release")
    assert "worker-b" not in message and "xfuser" not in message
    printed = capsys.readouterr().err
    assert "worker-b: the Worker's torchmonarch check refused" in printed
    assert "distribution=xfuser" in printed and "grep '^\\.\\./'" in printed
    assert prior_drift_message(SimpleNamespace(prior_refusals=[])) == (
        "worker hosts do not exactly match the current driver release")


def test_prior_drift_names_a_host_whose_release_differs_with_no_package_check_line(capsys):
    """A Worker whose payload digest, source or version differs prints no
    ownership line; its host is still named, and the receipt text stays fixed."""
    prior_drift_message = update_pin_diagnostic.prior_drift_message
    message = prior_drift_message(SimpleNamespace(prior_refusals=[("worker-b", None)]))
    assert message == ("worker hosts do not exactly match the current driver release; "
                       "the host lines above name each host that differs")
    printed = capsys.readouterr().err
    assert printed.startswith("  worker-b: the Worker does not run the driver's release exactly")
    assert "torchmonarch payload differs, or its installation or the check could not be read" in printed

