"""A refused benchmark case records its refusal class, never the refusal's text.

Three expected refusals on the FLUX.3 ladder of 2026-10-15 read only
"N/A ActorError future.py:256": the runner kept the exception type and raise
site, so two extra API legs were needed to learn each class. The failure row
now carries the class, the guard and the consent kind, read from the tag and
the consent card, and keeps every free-text field out as before.
"""
from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import pytest

from dgx_monarch import consent_descriptor
from dgx_monarch.refusal import PanelAction, RefusalClass, refusal

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dgxm_benchmark_run_matrix_refusal_row", REPO / "benchmark" / "run_matrix.py"
)
assert SPEC and SPEC.loader
run_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_matrix)

PRIVATE_TEXT = "private-run-detail"
PRIVATE_UNET = "private-unet-name.safetensors"
WAIVE = PanelAction(label="Waive for this model", env="DGXM_WAIVE_KNOWN_WRONG")


class UnsupportedModelError(ValueError):
    """The Worker-side exception a refusal rides."""


class ActorError(Exception):
    """Monarch's wrapper shape: the inner exception and its remote traceback text."""

    def __init__(self, exception: BaseException) -> None:
        self.exception = exception
        self.exception_formatted = "".join(
            line.replace(
                "Traceback (most recent call last):",
                "Traceback of where the remote call failed (most recent call last):",
            )
            for line in traceback.TracebackException.from_exception(exception).format()
        )

    def __str__(self) -> str:
        return f"A remote actor call has failed.\n {self.exception_formatted}"


def _raised(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as caught:
        return caught


def _row(error: BaseException) -> dict:
    return run_matrix._not_available_result("cell", {"name": "cell"}, {}, error)


def _card(kind: str) -> str:
    descriptor = consent_descriptor.ConsentDescriptor(
        version=consent_descriptor.CONSENT_DESCRIPTOR_VERSION,
        kind=kind,
        consent_id=consent_descriptor.consent_id(kind, "file-identity", "context"),
        unet_name=PRIVATE_UNET,
        file_identity="file-identity",
        context_fingerprint="context",
        family_hint="flux3",
        measured={"probe": "ring_pad", "measured": PRIVATE_TEXT},
        human_reason=f"{PRIVATE_TEXT} pad rows reach attention",
        evidence="measured on hardware",
        panel_action="Waive for this model",
        env_fallback="DGXM_WAIVE_KNOWN_WRONG",
    )
    return consent_descriptor.encode(descriptor)


def _assert_no_free_text(row: dict) -> None:
    serialized = json.dumps(row)
    assert PRIVATE_TEXT not in serialized
    assert PRIVATE_UNET not in serialized


def test_refused_row_records_class_k_and_guard_from_an_actor_error():
    message = refusal(
        RefusalClass.KNOWN_WRONG,
        f"a checkpoint pins an approximate attention kernel ({PRIVATE_TEXT})",
        guard="sol_attn:checkpoint_pin",
        waivable=True,
        panel_action=WAIVE,
    )
    error = _raised(ActorError(_raised(UnsupportedModelError(message))))

    row = _row(error)

    assert row["status"] == "N/A"
    assert row["reason"] == "refused"
    assert row["error_type"] == "ActorError"
    assert row["refusal_class"] == "K"
    assert row["refusal_guard"] == "sol_attn:checkpoint_pin"
    assert "consent_kind" not in row
    assert list(row)[-1] == "settings"
    _assert_no_free_text(row)
    sanitized = run_matrix.sanitize_report({"results": [row]})["results"][0]
    assert sanitized["refusal_guard"] == "sol_attn:checkpoint_pin"


@pytest.mark.parametrize("tagged", [True, False], ids=["tag-and-card", "card-alone"])
def test_refused_row_records_the_consent_kind_of_a_card(tagged):
    tag = refusal(
        RefusalClass.KNOWN_WRONG,
        f"ring attention cannot exclude the pad rows ({PRIVATE_TEXT})",
        guard="ring_pad",
        waivable=True,
        panel_action=WAIVE,
    ) if tagged else f"ring attention cannot exclude the pad rows ({PRIVATE_TEXT})"
    message = f"{tag}\n{_card('waive-known-wrong:ring-pad')}"
    error = _raised(ActorError(_raised(UnsupportedModelError(message))))

    row = _row(error)

    assert row["reason"] == "refused"
    assert row["refusal_class"] == "K"
    assert row["consent_kind"] == "waive-known-wrong:ring-pad"
    if tagged:
        assert row["refusal_guard"] == "ring_pad"
    else:
        assert "refusal_guard" not in row
    _assert_no_free_text(row)


def test_refused_row_records_class_p_without_a_guard():
    message = refusal(
        RefusalClass.PHYSICS,
        "FSDP capacity mode supports bf16, fp16, fp8 and int8 checkpoints at "
        f"launch ({PRIVATE_TEXT})",
    )

    row = _row(_raised(UnsupportedModelError(message)))

    assert row["reason"] == "refused"
    assert row["error_type"] == "UnsupportedModelError"
    assert row["refusal_class"] == "P"
    assert "refusal_guard" not in row and "consent_kind" not in row
    _assert_no_free_text(row)


def test_refused_row_reads_the_remote_traceback_when_the_inner_message_is_untagged():
    # Pickling drops __cause__, so a Worker that wraps its refusal leaves the
    # tag only in the formatted remote traceback.
    message = refusal(RefusalClass.PHYSICS, f"no shard layout ({PRIVATE_TEXT})")
    try:
        try:
            raise UnsupportedModelError(message)
        except UnsupportedModelError as refused:
            raise RuntimeError(f"worker setup failed ({PRIVATE_TEXT})") from refused
    except RuntimeError as wrapped:
        error = _raised(ActorError(wrapped))

    row = _row(error)

    assert row["refusal_class"] == "P"
    _assert_no_free_text(row)


def test_refused_row_keeps_only_listed_guards_and_kinds():
    shaped = f"[dgxm:K guard={PRIVATE_TEXT} waivable=0] {PRIVATE_TEXT}"

    row = _row(_raised(UnsupportedModelError(shaped)))

    assert row["refusal_class"] == "K"
    assert "refusal_guard" not in row
    _assert_no_free_text(row)

    unlisted = _row(_raised(UnsupportedModelError(_card("waive-known-wrong:unlisted"))))

    assert unlisted["reason"] == "case execution failed"
    assert "refusal_class" not in unlisted and "consent_kind" not in unlisted


def test_untagged_failure_row_is_unchanged():
    error = _raised(ActorError(_raised(RuntimeError(f"worker crashed ({PRIVATE_TEXT})"))))

    row = _row(error)

    assert row == {
        "name": "cell",
        "status": "N/A",
        "reason": "case execution failed",
        "error_type": "ActorError",
        "error_site": row["error_site"],
        "settings": {"name": "cell"},
    }
    assert row["error_site"].startswith("test_run_matrix_refusal_row.py:")
