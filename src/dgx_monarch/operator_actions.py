"""Inert, allowlisted remediation descriptions for operator readiness."""
from __future__ import annotations

from collections.abc import Sequence

_ORDER = (
    "recover-attached-mesh",
    "tighten-config-permissions",
    "inspect-worker-service",
    "restore-telemetry",
    "inspect-attached-mesh",
    "inspect-render-session",
    "resolve-doctor-failures",
    "rerun-doctor",
    "review-doctor-warnings",
    "wait-for-active-work",
)


def action(
    action_id: str,
    title: str,
    detail: str,
    safety: str,
    *,
    repairable: bool = False,
    reversible: bool = True,
    requires_confirmation: bool = False,
    command: str | None = None,
    docs_ref: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "id": action_id,
        "title": title,
        "detail": detail,
        "safety": safety,
        "repairable": repairable,
        "reversible": reversible,
        "requires_confirmation": requires_confirmation,
    }
    if command is not None:
        item["command"] = command
    if docs_ref is not None:
        item["docs_ref"] = docs_ref
    return item


def recover_mesh() -> dict[str, object]:
    return action(
        "recover-attached-mesh",
        "Recover the attached mesh",
        "The next attach cannot use the attached mesh. Confirm that no render session is active, "
        "then follow the remedy the mesh row names. A reset cannot clear a poisoned transport, "
        "which needs a ComfyUI restart, or a fleet that was never published.",
        "manual",
        reversible=False,
        requires_confirmation=True,
        docs_ref="docs/TROUBLESHOOTING.md#65",
    )


def inspect_workers() -> dict[str, object]:
    return action(
        "inspect-worker-service",
        "Inspect the worker service",
        "One or more worker services are stopped, unhealthy, or could not be identified.",
        "manual",
        command="dgxm doctor",
        docs_ref="docs/TROUBLESHOOTING.md",
    )


def unknown(action_id: str, title: str, detail: str) -> dict[str, object]:
    return action(
        action_id,
        title,
        detail,
        "manual",
        command="dgxm doctor",
        docs_ref="docs/TROUBLESHOOTING.md",
    )


def wait() -> dict[str, object]:
    return action(
        "wait-for-active-work",
        "Wait for active work",
        "The cluster is busy, not broken. Let the current render or mesh operation finish.",
        "automatic",
    )


def tighten_config_permissions() -> dict[str, object]:
    return action(
        "tighten-config-permissions",
        "Tighten config permissions",
        "The loaded cluster config is not owner-only, or doctor could not check its mode.",
        "automatic",
        repairable=True,
        reversible=True,
        requires_confirmation=True,
        command="dgxm doctor --repair",
        docs_ref="docs/CLUSTER.md",
    )


def ordered(items: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {str(item["id"]): item for item in items}
    rank = {action_id: index for index, action_id in enumerate(_ORDER)}
    return [by_id[key] for key in sorted(by_id, key=lambda key: (rank.get(key, len(rank)), key))]
