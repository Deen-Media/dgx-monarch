"""Bounded local pin diagnostics; receipts retain only their stable refusal code."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_REASONS = {
    "inventory_ambiguous": "foreign distribution inventory is ambiguous",
    "metadata_mismatch": "foreign distribution metadata does not match its record",
    "pinned_payload_claim": "foreign distribution claims pinned payload",
    "protected_package_overlap": "foreign payload overlaps a protected package",
    "record_mismatch": "foreign payload does not match its record",
    "ownership_unverified": "shared payload ownership could not be established",
    "inspection_failed": "payload inspection failed",
    "payload_mismatch": "installed payload or metadata differs from the exact pin",
}


def _label(value: str | None, *, path: bool = False) -> str | None:
    if not isinstance(value, str) or len(value) > (240 if path else 128):
        return None
    pattern = r"[A-Za-z0-9_.+/-]+" if path else r"[A-Za-z0-9][A-Za-z0-9_.+-]*"
    if re.fullmatch(pattern, value) is None:
        return None
    if path and any(part in ("", ".", "..") for part in value.split("/")):
        return None
    return value


@dataclass(frozen=True)
class PinDiagnostic:
    reason: str
    distribution: str | None = None
    relative_path: str | None = None

    def __post_init__(self) -> None:
        if self.reason not in _REASONS:
            object.__setattr__(self, "reason", "inspection_failed")
        object.__setattr__(self, "distribution", _label(self.distribution))
        object.__setattr__(self, "relative_path", _label(self.relative_path, path=True))

    def terminal_message(self) -> str:
        context = "; ".join(
            value for value in (
                f"distribution={self.distribution}" if self.distribution else "",
                f"relative path={self.relative_path}" if self.relative_path else "",
            ) if value
        )
        return _REASONS[self.reason] + (f" ({context})" if context else "")


def inspect_match(probe: Callable[[], bool]) -> tuple[bool, PinDiagnostic | None]:
    # Imported here because ownership errors themselves carry this diagnostic.
    from .update_payload_ownership import PayloadOwnershipError

    try:
        matched = probe()
        return matched, None if matched else PinDiagnostic("payload_mismatch")
    except PayloadOwnershipError as exc:
        return False, exc.diagnostic
    except Exception:
        return False, PinDiagnostic("inspection_failed")


def pin_refusal_message(pins: object) -> str:
    diagnostic = getattr(pins, "last_diagnostic", None)
    message = "the driver's torchmonarch does not match the current checkout's exact pin"
    if isinstance(diagnostic, PinDiagnostic):
        message += ": " + diagnostic.terminal_message()
    return message + ". Inspect the selected environment's package ownership before retrying; no package repair was attempted."
