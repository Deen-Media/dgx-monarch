"""Bounded local pin diagnostics; receipts retain only their stable refusal code."""
from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass

_REASONS = {
    "record_outside_site": (
        "foreign distribution RECORD lists files outside site-packages that are not bytecode "
        "compiled for its own sources"),
    "record_malformed": "foreign distribution RECORD is malformed",
    "distribution_duplicated": "foreign distribution is installed more than once",
    "shared_payload_missing": "a file torchmonarch shares with another distribution is missing",
    "metadata_mismatch": "foreign distribution metadata does not match its record",
    "pinned_payload_claim": "foreign distribution claims pinned payload",
    "protected_package_overlap": "foreign payload overlaps a protected package",
    "record_mismatch": "foreign payload does not match its record",
    "ownership_unverified": "shared payload ownership could not be established",
    "inspection_failed": "payload inspection failed",
    "payload_mismatch": "installed payload or metadata differs from the exact pin",
}
_TROUBLESHOOTING = "docs/TROUBLESHOOTING.md #155"
# The cause is known, and so is what clears it.
_REMEDIES = {
    "record_outside_site": (
        "That RECORD lists a file outside site-packages the check cannot vouch for: one with a "
        "hash (a data file, header or script), a script's compiled cache, or bytecode for "
        "another Python, and a reinstall writes the same rows again. See them with `grep "
        "'^\\.\\./' <site-packages>/<relative path>`; uninstall that distribution from this "
        f"environment, or install it in another one, then run the update again ({_TROUBLESHOOTING})"),
    "record_malformed": (
        "Reinstall that distribution (`<python> -m pip install --no-deps --force-reinstall "
        f"<name>==<version>`), then run the update again ({_TROUBLESHOOTING})"),
    "distribution_duplicated": (
        "That distribution has more than one metadata directory in this site-packages, so no "
        "single record says which files it owns. List them (`ls -d <site-packages>/*.dist-info`), "
        "remove every copy (`<python> -m pip uninstall <name>`, again while `<python> -m pip show "
        "<name>` still finds one), install the version you use once, then run the update again "
        f"({_TROUBLESHOOTING})"),
    "shared_payload_missing": (
        "Reinstall torchmonarch (`<python> -m pip install --no-deps --force-reinstall "
        "torchmonarch==<the installed version>`): removing or upgrading a distribution that ships "
        f"the same file took torchmonarch's copy with it; then run the update again ({_TROUBLESHOOTING})"),
}
_INSPECT = "Inspect the selected environment's package ownership before retrying"
# The one line a Worker's attestation prints when it refuses on package
# ownership, so the controller can name the host, the cause and the remedy.
ATTEST_PREFIX = "DGXM_ATTEST_REFUSAL="
_ATTEST_LINE = re.compile(
    re.escape(ATTEST_PREFIX) + r"(?P<reason>[a-z_]{1,40}) distribution=(?P<distribution>\S{1,128}) "
    r"path=(?P<path>\S{1,240})")


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

    def remedy(self) -> str:
        """What to do, ending without a full stop; the generic advice when no repair is known."""
        return _REMEDIES.get(self.reason, _INSPECT)

    def attest_line(self) -> str:
        """The bounded line a Worker attestation prints before it refuses."""
        return (f"{ATTEST_PREFIX}{self.reason} distribution={self.distribution or '-'} "
                f"path={self.relative_path or '-'}")


def parse_attest_line(text: str) -> PinDiagnostic | None:
    """The refusal a Worker attestation printed, re-bounded; None when it printed none."""
    for line in text.splitlines():
        match = _ATTEST_LINE.fullmatch(line.strip())
        if match is not None:
            return PinDiagnostic(
                match["reason"],
                None if match["distribution"] == "-" else match["distribution"],
                None if match["path"] == "-" else match["path"],
            )
    return None


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
        return f"{message}: {diagnostic.terminal_message()}. {diagnostic.remedy()}; no package repair was attempted."
    return f"{message}. {_INSPECT}; no package repair was attempted."


def worker_refusal_message(host_name: str, diagnostic: PinDiagnostic) -> str:
    """One host's line for a Worker attestation that refused on package ownership."""
    return (f"  {host_name}: the Worker's torchmonarch check refused: "
            f"{diagnostic.terminal_message()}. {diagnostic.remedy()}, on that host.")


def worker_mismatch_message(host_name: str) -> str:
    """One host's line for a Worker whose release differs with no package check refusing."""
    return (f"  {host_name}: the Worker does not run the driver's release exactly: its source, "
            "version or torchmonarch payload differs, or its installation or the check "
            "could not be read there.")


def prior_drift_message(releases: object) -> str:
    """The prior_worker_drift refusal; prints a line for each host that differs first.

    The receipt keeps this message, so host names and package names go only
    to the terminal, as the dependency diagnostics do.
    """
    named = remedied = False
    for host_name, diagnostic in getattr(releases, "prior_refusals", None) or ():
        if not isinstance(host_name, str):
            continue
        if isinstance(diagnostic, PinDiagnostic):
            print(worker_refusal_message(host_name, diagnostic), file=sys.stderr)
            remedied = True
        else:
            print(worker_mismatch_message(host_name), file=sys.stderr)
        named = True
    message = "worker hosts do not exactly match the current driver release"
    if remedied:
        return message + ("; the host lines above name each host that differs, and the package "
                          "check that refused and its remedy")
    if named:
        return message + "; the host lines above name each host that differs"
    return message
