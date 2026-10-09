"""Validation and reporting for torchmonarch surface snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .monarch_surface import SNAPSHOT_SCHEMA, TOUCHPOINTS, _manifest_digest


@dataclass(frozen=True)
class SnapshotComparison:
    baseline_version: str
    candidate_version: str
    changes: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "compatible": self.compatible,
            "changes": list(self.changes),
            "blockers": list(self.blockers),
        }


def snapshot_blockers(snapshot: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if snapshot.get("distribution") != "torchmonarch":
        blockers.append(
            "snapshot distribution is "
            f"{snapshot.get('distribution')!r}, expected 'torchmonarch'"
        )
    version = snapshot.get("version")
    if (
        not isinstance(version, str)
        or not version.strip()
        or version.strip().lower().startswith("unavailable (")
    ):
        blockers.append(f"snapshot has no trustworthy torchmonarch version: {version!r}")
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        blockers.append(
            f"snapshot schema is {snapshot.get('schema')!r}, expected {SNAPSHOT_SCHEMA}"
        )
    if snapshot.get("manifest_digest") != _manifest_digest():
        blockers.append("snapshot was produced by a different Monarch manifest")
    touchpoints = snapshot.get("touchpoints")
    if not isinstance(touchpoints, Mapping):
        blockers.append("snapshot has no touchpoint mapping")
    else:
        for touchpoint in TOUCHPOINTS:
            record = touchpoints.get(touchpoint.path)
            if not isinstance(record, Mapping):
                blockers.append(f"{touchpoint.path}: absent from snapshot")
            elif record.get("status") != "ok":
                blockers.append(
                    f"{touchpoint.path}: {record.get('status')} ({record.get('detail')})"
                )
    semantics = snapshot.get("semantics")
    if not isinstance(semantics, Mapping):
        blockers.append("snapshot has no semantic observations")
    else:
        for name in (
            "raw_python_task_await_in_asyncio",
            "public_future_get_in_asyncio",
            "unhandled_fault_hook_assignment",
            "actor_queue_dispatch_default",
        ):
            observation = semantics.get(name)
            if not isinstance(observation, Mapping):
                blockers.append(f"semantic probe {name}: absent")
            elif observation.get("outcome") == "probe_error":
                blockers.append(
                    f"semantic probe {name}: {observation.get('exception_type')}: "
                    f"{observation.get('message')}"
                )
    if isinstance(semantics, Mapping):
        hook = semantics.get("unhandled_fault_hook_assignment")
        if isinstance(hook, Mapping) and hook.get("outcome") != "writable":
            blockers.append("monarch.actor.unhandled_fault_hook is not writable")
    return blockers


def _version(snapshot: Mapping[str, Any]) -> str:
    value = snapshot.get("version")
    return value if isinstance(value, str) else repr(value)


def compare_snapshots(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> SnapshotComparison:
    changes: list[str] = []
    blockers = [f"baseline: {item}" for item in snapshot_blockers(baseline)]
    blockers.extend(f"candidate: {item}" for item in snapshot_blockers(candidate))

    baseline_touchpoints = baseline.get("touchpoints")
    candidate_touchpoints = candidate.get("touchpoints")
    if isinstance(baseline_touchpoints, Mapping) and isinstance(
        candidate_touchpoints, Mapping
    ):
        baseline_paths = set(baseline_touchpoints)
        candidate_paths = set(candidate_touchpoints)
        for path in sorted(baseline_paths - candidate_paths):
            blockers.append(f"surface removed from candidate snapshot: {path}")
        for path in sorted(candidate_paths - baseline_paths):
            changes.append(f"surface added to candidate snapshot: {path}")
        for path in sorted(baseline_paths & candidate_paths):
            before = baseline_touchpoints[path]
            after = candidate_touchpoints[path]
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                continue
            if before.get("kind") != after.get("kind"):
                changes.append(
                    f"{path}: kind {before.get('kind')!r} -> {after.get('kind')!r}"
                )
            if before.get("signature") != after.get("signature"):
                changes.append(
                    f"{path}: signature {before.get('signature')!r} -> "
                    f"{after.get('signature')!r}"
                )

    baseline_semantics = baseline.get("semantics")
    candidate_semantics = candidate.get("semantics")
    if isinstance(baseline_semantics, Mapping) and isinstance(
        candidate_semantics, Mapping
    ):
        for name in sorted(set(baseline_semantics) | set(candidate_semantics)):
            before = baseline_semantics.get(name)
            after = candidate_semantics.get(name)
            if before != after:
                blockers.append(
                    f"semantic drift in {name}: "
                    f"{json.dumps(before, sort_keys=True)} -> "
                    f"{json.dumps(after, sort_keys=True)}"
                )

    baseline_version = _version(baseline)
    candidate_version = _version(candidate)
    if baseline_version != candidate_version:
        changes.insert(0, f"version {baseline_version} -> {candidate_version}")
    return SnapshotComparison(
        baseline_version,
        candidate_version,
        tuple(dict.fromkeys(changes)),
        tuple(dict.fromkeys(blockers)),
    )


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_comparison_markdown(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    comparison: SnapshotComparison,
) -> str:
    verdict = "PASS" if comparison.compatible else "REVIEW REQUIRED"
    lines = [
        "# torchmonarch surface comparison",
        "",
        f"- Verdict: **{verdict}**",
        f"- Exact pin snapshot: `{comparison.baseline_version}`",
        f"- Latest stable snapshot: `{comparison.candidate_version}`",
        "- This report is advisory only; the workflow never changes the repository pin.",
        "",
        "## Compatibility blockers",
        "",
    ]
    if comparison.blockers:
        lines.extend(f"- {_markdown_cell(blocker)}" for blocker in comparison.blockers)
    else:
        lines.append("- None.")
    lines.extend(["", "## Surface changes", ""])
    if comparison.changes:
        lines.extend(f"- {_markdown_cell(change)}" for change in comparison.changes)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Event-loop and hook semantics",
            "",
            "| Probe | Exact pin | Latest stable |",
            "|---|---|---|",
        ]
    )
    baseline_semantics = baseline.get("semantics", {})
    candidate_semantics = candidate.get("semantics", {})
    if isinstance(baseline_semantics, Mapping) and isinstance(
        candidate_semantics, Mapping
    ):
        for name in sorted(set(baseline_semantics) | set(candidate_semantics)):
            before = json.dumps(baseline_semantics.get(name), sort_keys=True)
            after = json.dumps(candidate_semantics.get(name), sort_keys=True)
            lines.append(
                f"| `{_markdown_cell(name)}` | `{_markdown_cell(before)}` | "
                f"`{_markdown_cell(after)}` |"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "SnapshotComparison",
    "compare_snapshots",
    "render_comparison_markdown",
    "snapshot_blockers",
]
