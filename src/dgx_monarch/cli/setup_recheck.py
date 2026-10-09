"""Recheck a reviewed guided-setup plan on the live hosts before apply changes anything."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..config import ClusterConfig
from .setup_config_io import setup_readiness
from .setup_probe import ArtifactComparison, HostProbe, compare_artifacts, probe_hosts, validate_artifact_paths
from .setup_source import SetupSource


@dataclass(frozen=True)
class SetupRecheck:
    probes: tuple[HostProbe, ...]
    artifacts: tuple[ArtifactComparison, ...]
    source: SetupSource | None
    blockers: tuple[str, ...]
    matching_hosts: int
    source_matched: bool
    matched: bool

    def receipt_counts(self) -> dict[str, int]:
        return {
            "artifacts": len(self.artifacts),
            "blockers": len(self.blockers),
            "hosts": len(self.probes),
            "matching_hosts": self.matching_hosts,
            "source_matched": int(self.source_matched),
        }


def recheck_setup(
    config: ClusterConfig,
    *,
    reviewed_probes: Sequence[HostProbe],
    reviewed_artifacts: Sequence[ArtifactComparison],
    reviewed_source: SetupSource,
    expected_gpus: Sequence[int],
    artifact_paths: Sequence[str],
    fabric_profile: str,
    install_service: bool,
    start_workers: bool,
    verify: bool,
    transport_security: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    source_snapshot: Callable[[], SetupSource],
) -> SetupRecheck:
    """Probe the plan's hosts again; ``matched`` requires no blocker and the reviewed probes, artifacts and source."""
    selected_artifacts = validate_artifact_paths(artifact_paths)
    probes = probe_hosts(config, artifacts=selected_artifacts, runner=runner)
    artifacts = compare_artifacts(probes, len(selected_artifacts))
    blockers, _warnings = setup_readiness(
        probes=probes,
        expected_gpus=expected_gpus,
        artifacts=artifacts,
        fabric_profile=fabric_profile,
        install_service=install_service,
        start_workers=start_workers,
        verify=verify,
        transport_security=transport_security,
    )
    try:
        source: SetupSource | None = source_snapshot()
    except Exception:
        source = None
        blockers.append("source_state_unavailable")
    expected = tuple(reviewed_probes)
    matching = sum(current == prior for current, prior in zip(probes, expected, strict=False))
    exact = (
        len(probes) == len(expected)
        and matching == len(expected)
        and artifacts == tuple(reviewed_artifacts)
        and source == reviewed_source
    )
    return SetupRecheck(
        probes,
        artifacts,
        source,
        tuple(blockers),
        matching,
        source == reviewed_source,
        exact and not blockers,
    )
