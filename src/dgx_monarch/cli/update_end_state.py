"""Name where a verified update that stopped short left the hosts, from its own steps.

Every end state after the Worker stop says which release each host is on,
whether its Worker runs, and the repair: one command where one exists, else the
case in TROUBLESHOOTING entry 118 for the update's code. The steps are the ones
the receipt records, so the sentence and the receipt cannot disagree. Each
sentence, with its prefix, fits one receipt note (240 characters).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .release_slot import TROUBLESHOOTING as _SLOT_ENTRY

_ENTRY = "TROUBLESHOOTING entry " + _SLOT_ENTRY.rsplit("#", 1)[1]
_FORWARD = f"use the forward repair in {_ENTRY}"
_BY_CODE = f"repair it as {_ENTRY} says for this code"
END_STATES: Mapping[str, str] = {
    "old_some_stopped": (
        "Nothing was activated: every host still runs the old release and some Workers may be stopped; "
        "`dgxm up` starts them, then rerun the update."),
    "old_running": "Every host runs the old release again with its Worker; clear the cause and rerun the update.",
    "old_not_running": (
        "Every host is back on the old release, but not every Worker runs (named above); `dgxm up` starts them."),
    "split": (
        "The hosts may run different releases, every Worker stopped: when every host's link names the old "
        f"release, `dgxm up` starts them; otherwise {_FORWARD}."),
    "split_switched": (
        "At least one host runs the new release, one could not be read back, the rest run the old one, every "
        f"Worker stopped; {_FORWARD}."),
    "new_stopped": (
        "Every host's link names the new release, its Worker stopped, and the driver was not confirmed on it; "
        f"{_BY_CODE}."),
    "new_unconfirmed": (
        "Every host's link names the new release, its Worker may still run, and the driver was not confirmed "
        f"on it; {_BY_CODE}."),
    "new_running": (
        "The driver and every Worker run the new release; only the cleanup of the previous release or of the "
        "staging area was not confirmed."),
}
_RESTORE_AFTER = frozenset({"activation", "worker_start", "doctor", "cluster_smoke", "exact_readback", "driver_commit"})


class _Step(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def counts(self) -> Mapping[str, int] | None: ...


def end_sentence(steps: Sequence[_Step]) -> str:
    """The sentence for ``end_state``, saying first when a restore of the old release fell short."""
    key = end_state(steps)
    restoring = any(step.name in _RESTORE_AFTER and step.status == "failed" for step in steps)
    prefix = "The restore was not confirmed. " if restoring and key != "old_running" else ""
    return prefix + END_STATES[key]


def end_state(steps: Sequence[_Step]) -> str:
    """The key of ``END_STATES`` for an update that started its Worker stop and stopped short."""
    seen: dict[str, list[_Step]] = {}
    for step in steps:
        seen.setdefault(step.name, []).append(step)

    def last(name: str) -> str | None:
        return seen[name][-1].status if name in seen else None

    if last("worker_stop") != "succeeded":
        return "old_some_stopped"
    if last("driver_commit") == "succeeded":
        return "new_running"
    if "prior_release_start" in seen:
        return "old_running" if last("prior_release_start") == "succeeded" else "old_not_running"
    compensation = last("compensation") or last("post_activation_compensation")
    if compensation is not None:
        return "old_not_running" if compensation == "succeeded" else "split"
    if last("activation") != "succeeded":
        counts = seen["activation"][-1].counts if last("activation") == "unknown" else None
        return "split_switched" if counts and counts.get("hosts_switched", 0) > 0 else "split"
    if "fail_closed_stop" in seen:
        # The first stop after target start decides; a later one follows a restore.
        return "new_stopped" if seen["fail_closed_stop"][0].status == "succeeded" else "new_unconfirmed"
    return "new_unconfirmed"
