"""Guided setup's verification: doctor, then the cluster smoke; the result names no host or path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..config import ClusterConfig

_UNKNOWN_SMOKE_CODES = frozenset(
    {
        "cleanup_attempt_repeated",
        "cleanup_not_confirmed",
        "cleanup_outcome_untyped",
        "cleanup_recycle_failed",
    }
)


def _unknown_smoke_code(code: object) -> bool:
    """Return whether smoke failed without reaching a definite result.

    ``cluster_smoke`` assigns ``{phase}_unknown`` to faults without a typed verdict.
    Checking the suffix also covers new phases.
    """
    return (
        code in _UNKNOWN_SMOKE_CODES
        or (isinstance(code, str) and code.endswith("_unknown"))
    )


def smoke_succeeded(result: object) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, Mapping):
        value = result.get("result")
        return type(value) is str and value == "PASS"
    as_dict = getattr(result, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if not isinstance(payload, Mapping):
            return False
        value = payload.get("result")
        return type(value) is str and value == "PASS"
    return False


@dataclass(frozen=True)
class SetupVerification:
    doctor_ok: bool | None
    smoke_attempted: bool
    smoke_ok: bool | None
    world: int
    status_rank_coverage: int
    source_rank_coverage: int
    source_matched: bool
    teardown_confirmed: bool

    @property
    def unknown(self) -> bool:
        return self.doctor_ok is None or self.smoke_ok is None

    def smoke_counts(self) -> dict[str, int]:
        return {
            "attempted": int(self.smoke_attempted),
            "passed": int(self.smoke_ok is True),
            "source_matched": int(self.source_matched),
            "source_rank_coverage": self.source_rank_coverage,
            "status_rank_coverage": self.status_rank_coverage,
            "teardown_confirmed": int(self.teardown_confirmed),
            "world": self.world,
        }


@dataclass
class SetupVerificationProgress:
    doctor_attempted: bool = False
    doctor_completed: bool = False
    doctor_ok: bool | None = None
    smoke_attempted: bool = False
    smoke_completed: bool = False


def verify_setup(
    config: ClusterConfig,
    config_path: Path,
    comfy_dir: str,
    expected_source: str,
    *,
    doctor: Callable[[ClusterConfig], bool],
    smoke: Callable[[Path, str], object],
    progress: SetupVerificationProgress | None = None,
) -> SetupVerification:
    """Run passive doctor first; attach smoke only after it passes."""
    state = progress or SetupVerificationProgress()
    state.doctor_attempted = True
    try:
        doctor_result = doctor(config)
    except Exception:
        return SetupVerification(None, False, None, 0, 0, 0, False, False)
    if type(doctor_result) is not bool:
        return SetupVerification(None, False, None, 0, 0, 0, False, False)
    doctor_ok = doctor_result
    state.doctor_completed = True
    state.doctor_ok = doctor_ok
    if not doctor_ok:
        return SetupVerification(False, False, False, 0, 0, 0, False, False)
    state.smoke_attempted = True
    try:
        result = smoke(config_path, comfy_dir)
        payload: object = result if isinstance(result, Mapping) else getattr(result, "as_dict", lambda: {})()
        if not isinstance(payload, Mapping):
            raise ValueError("invalid smoke evidence")
        verdict = payload.get("result")
        if type(verdict) is not str or verdict not in {"PASS", "FAIL"}:
            raise ValueError("invalid smoke verdict")
        if verdict == "FAIL":
            if _unknown_smoke_code(payload.get("error")):
                raise ValueError("smoke cleanup outcome is unknown")
            state.smoke_completed = True
            return SetupVerification(True, True, False, 0, 0, 0, False, False)
        world = _evidence_count(payload.get("world"))
        status_ranks = _evidence_count(payload.get("status_rank_coverage"))
        source_ranks = _evidence_count(payload.get("source_rank_coverage"))
        source = payload.get("source_manifest_sha256")
        teardown_value = payload.get("teardown_confirmed")
        if (
            world is None
            or status_ranks is None
            or source_ranks is None
            or type(source) is not str
            or type(teardown_value) is not bool
        ):
            raise ValueError("invalid smoke readback")
        source_matched = source == expected_source
        teardown = teardown_value
        if not teardown:
            raise ValueError("smoke teardown was not confirmed")
        passed = (
            verdict == "PASS"
            and world == config.world_size
            and status_ranks == world
            and source_ranks == world
            and source_matched
            and teardown
        )
        verification = SetupVerification(
            True,
            True,
            passed,
            world,
            status_ranks,
            source_ranks,
            source_matched,
            teardown,
        )
    except Exception:
        return SetupVerification(True, True, None, 0, 0, 0, False, False)
    state.smoke_completed = True
    return verification


def _evidence_count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value < 2**31 else None


def run_setup_smoke(config_path: Path, comfy_dir: str) -> object:
    from .cluster_smoke import ClusterSmokeError, run_cluster_smoke

    try:
        return run_cluster_smoke(config_path, comfy_dir=comfy_dir or None)
    except ClusterSmokeError as exc:
        # verify_setup reads the typed error's code: unknown codes become UNKNOWN,
        # the rest a refusal. Any other Exception propagates and also lands UNKNOWN.
        return exc.as_dict()
