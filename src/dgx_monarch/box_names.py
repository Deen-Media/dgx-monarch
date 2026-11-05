"""The one rule for naming a box in text a user reads.

The box the driver runs on is "the head". With one other box, that box is "the
sibling"; with more than one, each is "the sibling at <host name>", so no two
read alike. A local fleet, every rank in the driver's own process tree, is "this
box": it has no pair to tell apart. A box that holds more than one rank names
the GPU too ("GPU 1 on the head"). The capacity refusals, the Old model in the
way card and the agreement's log line take their words from here, so none names
a box by its rank number or by the address cluster.toml gives it.

The head is the ``[[hosts]]`` entry that ``driver_footprint.host_is_local``
resolves to this machine, the check ``rank0_co_resident`` makes; no rule assumes
the first entry is the head. Machine fields (a quote row's ``host``, a ledger
row's ``rank``) keep the host name and the rank number for operators and parsers.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

HEAD = "the head"
SIBLING = "the sibling"
LOCAL = "this box"
# A rank past the configured hosts: a cluster.toml that lists fewer GPUs than the fleet.
UNLISTED = "a box the cluster config does not list"


def host_words(hosts: Sequence[str], is_head: Callable[[str], bool]) -> list[str]:
    """The plain name of each configured host, in config order.

    When no entry is the head (a driver box with no Worker), every box is
    another box, so two of them are named by host.
    """
    heads = [bool(name) and is_head(name) for name in hosts]
    others = len(hosts) - sum(heads)
    return [HEAD if head else SIBLING if others == 1 or not name else f"{SIBLING} at {name}"
            for name, head in zip(hosts, heads, strict=True)]


def rank_boxes(hosts: Sequence[str], world: int, gpus: int,
               is_head: Callable[[str], bool]) -> list[tuple[str, int | None]]:
    """Each rank's box and, on a box that holds more than one rank, its GPU.

    Ranks fill hosts in blocks of ``gpus``, the order the agreement slices the
    worker mesh in, so rank ``i`` sits on host ``i // gpus``. No host list is a
    local fleet, whose GPUs are its ranks.
    """
    if not hosts:
        return [(LOCAL, rank if world > 1 else None) for rank in range(world)]
    names = host_words(hosts, is_head)
    gpus = max(int(gpus), 1)
    return [(names[rank // gpus] if rank // gpus < len(names) else UNLISTED,
             rank % gpus if gpus > 1 else None) for rank in range(world)]


def label(box: str, gpu: int | None = None) -> str:
    """One rank in plain words: its box, or 'GPU <n> on <box>'."""
    return box if gpu is None else f"GPU {gpu} on {box}"


def opening(words: str) -> str:
    """A name that opens a sentence: 'The head priced ...'."""
    return words[:1].upper() + words[1:]


def fleet_boxes(handle: Any, world: int) -> list[tuple[str, int | None]]:
    """``rank_boxes`` for an attached fleet, read off its handle.

    A ``this_host()`` mesh (``owns_hosts`` False) is local whatever the config
    lists, the way ``driver_footprint.rank0_co_resident`` reads it.
    """
    from . import driver_footprint

    hosts: tuple[Any, ...] = ()
    if getattr(handle, "owns_hosts", True) is not False:
        hosts = tuple(getattr(getattr(handle, "config", None), "hosts", None) or ())
    gpus = int(getattr(handle, "gpus_per_host", 1) or 1)
    return rank_boxes([str(getattr(host, "name", "") or "") for host in hosts],
                      int(world), gpus, lambda name: driver_footprint.host_is_local(name))


def fleet_labels(handle: Any, world: int) -> list[str]:
    """Each rank of an attached fleet in plain words, in rank order."""
    return [label(box, gpu) for box, gpu in fleet_boxes(handle, world)]
