"""The one rule for naming a box in text a user reads.

The box the driver runs on is "the head". With one other box, that box is "the
sibling"; with more than one, each is "the sibling at <host name>", so no two
read alike. A fleet on the driver's box alone (a local fleet, or a cluster.toml
that lists only that box) is "this box": it has no pair to tell apart. A box
that holds more than one rank names the GPU too ("GPU 1 on the head"). The
capacity refusals (the agreement's, the FSDP clean-reload price's and the
undeliverable-actor refusals of ``restart_reattach``), the Old model in the way
card, the agreement's log line and the Fleet timeline take their words from
here, so none names a box by its rank number or by the address cluster.toml
gives it. A Worker's own refusal says "here" and the driver puts its box ahead
of it ("On the sibling, ...", ``on_box`` and ``worker_box.py``).

The head is the ``[[hosts]]`` entry that ``driver_footprint.host_is_local``
resolves to this machine, the check ``rank0_co_resident`` makes; no rule assumes
the first entry is the head. When that check cannot resolve the driver's entry,
every box is another box and reads "the sibling at <host name>", the same miss
that leaves ``rank0_co_resident`` false. Machine fields (a quote row's ``host``,
a ledger row's ``rank``, the attach trace) keep the host name, the address and
the rank number for operators and parsers.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

HEAD = "the head"
SIBLING = "the sibling"
LOCAL = "this box"
# A rank past the configured hosts: a cluster.toml that lists fewer GPUs than the fleet.
UNLISTED = "a box the cluster config does not list"


def host_words(hosts: Sequence[str], is_head: Callable[[str], bool]) -> list[str]:
    """The plain name of each configured host, in config order.

    When no entry is the head (a driver box with no Worker), every box is
    another box, so two of them are named by host. When every entry is the
    head, the fleet runs on the driver's box alone, which is "this box".
    """
    heads = [bool(name) and is_head(name) for name in hosts]
    if heads and all(heads):
        return [LOCAL] * len(hosts)
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


def join(names: Iterable[str]) -> str:
    """'the head', 'the head and the sibling', 'a, b and c'."""
    items = list(names)
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


def label(box: str, gpu: int | None = None) -> str:
    """One rank in plain words: its box, or 'GPU <n> on <box>'."""
    return box if gpu is None else f"GPU {gpu} on {box}"


def opening(words: str) -> str:
    """A name that opens a sentence: 'The head priced ...'."""
    return words[:1].upper() + words[1:]


def on_box(box: str) -> str:
    """The words that put a Worker's own sentence on its box: 'On the sibling, '.

    None for a fleet on the driver's box alone, where the Worker's "here" is
    already the box the user reads about (worker_box.py).
    """
    return "" if box == LOCAL else f"{opening('on ' + box)}, "


def fleet_boxes(handle: Any, world: int) -> list[tuple[str, int | None]]:
    """``rank_boxes`` for an attached fleet, read off its handle.

    A ``this_host()`` mesh (``owns_hosts`` False) is local whatever the config
    lists, the way ``driver_footprint.rank0_co_resident`` reads it.
    """
    hosts: tuple[Any, ...] = ()
    if getattr(handle, "owns_hosts", True) is not False:
        hosts = _hosts(getattr(handle, "config", None))
    gpus = int(getattr(handle, "gpus_per_host", 1) or 1)
    return rank_boxes([_name(host) for host in hosts], int(world), gpus, _is_head)


def address_words(config: Any) -> dict[str, str]:
    """Each configured host's address mapped to its plain name.

    For texts that start from a Worker loop's address rather than a rank: an
    undeliverable actor, a loop's proc id, a liveness answer.
    """
    hosts = _hosts(config)
    words = host_words([_name(host) for host in hosts], _is_head)
    return {str(getattr(host, "address", "") or ""): word
            for host, word in zip(hosts, words, strict=True)}


def _hosts(config: Any) -> tuple[Any, ...]:
    return tuple(getattr(config, "hosts", None) or ())


def _name(host: Any) -> str:
    return str(getattr(host, "name", "") or "")


def _is_head(name: str) -> bool:
    from . import driver_footprint

    return driver_footprint.host_is_local(name)


def fleet_labels(handle: Any, world: int) -> list[str]:
    """Each rank of an attached fleet in plain words, in rank order."""
    return [label(box, gpu) for box, gpu in fleet_boxes(handle, world)]
