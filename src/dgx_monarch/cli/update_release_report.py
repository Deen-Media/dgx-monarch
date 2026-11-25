"""Per-host lines a verified update prints on stderr when a release change stops short.

Host names and paths stay here, out of the receipt, whose notes redact them.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from ..config import HostConfig
from .release_slot import TROUBLESHOOTING as _SLOT_ENTRY
from .update_types import Certainty

_BASE = ".local/share/dgx-monarch"
_LINK = "~/" + _BASE + "/src"
_FORWARD = f"use the forward repair in {_SLOT_ENTRY}"


def _names(hosts: Sequence[HostConfig]) -> str:
    return ", ".join(host.name for host in hosts)


def _say(line: str) -> None:
    print(line, file=sys.stderr)


def _prior_source(snapshot: dict[str, Any]) -> tuple[str, str]:
    """What the host's source was before the update: a description, and the condition to check."""
    live_value, home_value = snapshot.get("live"), snapshot.get("home")
    live: dict[str, Any] = live_value if isinstance(live_value, dict) else {}
    home = home_value if isinstance(home_value, str) else "~"
    raw_target = live.get("target")
    if live.get("kind") == "symlink" and isinstance(raw_target, str):
        target = os.path.normpath(os.path.join(home, _BASE, raw_target))
        # Plain readlink prints the link's own text, the form the snapshot recorded; -f would resolve it.
        return f"a link to {target}", f"`readlink {_LINK}` prints {raw_target}"
    if live.get("kind") == "directory":
        return "a plain source directory", f"{_LINK} is still a plain directory"
    return "no source link", f"there is still no {_LINK}"


def report_activation(hosts: Sequence[HostConfig], slot: Any, certainty: Certainty) -> None:
    """Name each host's release after a switch that stopped short, with the repair that fits."""
    stopped_at = slot.activation_attempted - 1
    host = hosts[stopped_at]
    switched, untouched = hosts[:stopped_at], hosts[stopped_at + 1:]
    if certainty == Certainty.UNKNOWN:
        snapshot = slot.installations.get(host.name, {})
        home_value = snapshot.get("home")
        home = home_value if isinstance(home_value, str) else "~"
        prior, check = _prior_source(snapshot)
        new = f"{home}/{_BASE}/releases/{slot.token}/site"
        _say(f"activation on {host.name} is unknown: its source is either the old release's ({prior}) "
             f"or a link to the new release, {new}")
    else:
        _say(f"activation on {host.name} failed and left a state the update can restore")
    if switched:
        _say(f"switched to the new release before it: {_names(switched)}")
    if untouched:
        _say(f"never sent an activation, still on the old release: {_names(untouched)}")
    if certainty != Certainty.UNKNOWN:
        return
    if switched:
        _say(f"the hosts run different releases, every Worker stopped: {_FORWARD}")
    else:
        _say(f"every Worker is stopped: on {host.name}, if {check}, `dgxm up` starts every Worker on "
             f"the old release; otherwise {_FORWARD}")


def report_compensation(results: Sequence[tuple[HostConfig, Certainty]], prior_matches: bool) -> None:
    """After a compensation that was not confirmed everywhere, name the hosts it restored and the rest."""
    restored = [host for host, certainty in results if certainty == Certainty.SUCCEEDED]
    unconfirmed = [host for host, certainty in results if certainty != Certainty.SUCCEEDED]
    if unconfirmed:
        _say(f"compensation restored the old release on: {_names(restored) or 'no host'}")
        _say(f"compensation could not confirm the old release on: {_names(unconfirmed)}")
    elif not prior_matches:
        _say("compensation finished on every host it reached, but the old release did not read back on every host")


def report_prior_start(results: Sequence[tuple[HostConfig, Certainty]]) -> None:
    """Name the hosts whose old-release Worker did not start; the rest run it."""
    missing = [host for host, certainty in results if certainty != Certainty.SUCCEEDED]
    if missing:
        _say(f"the old release's Worker did not start on: {_names(missing)}; `dgxm up` starts it")
