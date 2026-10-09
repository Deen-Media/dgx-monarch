"""Resolve the address advertised by the Monarch client transport.

Hostname lookup can select loopback; a non-loopback ``cluster.client_bind``
is the authoritative fabric address for transport and doctor.
"""
from __future__ import annotations

from ..config import ClusterConfig, tcp_endpoint, unmapped_bind_ip


def _is_loopback(address: str) -> bool:
    """True only for an IP literal that parses and is loopback.

    Classified on the unwrapped address, so ``::ffff:127.0.0.1`` reads as the
    loopback it is rather than as a routable IPv6 address.
    """
    try:
        return unmapped_bind_ip(address)[0].is_loopback
    except ValueError:
        return False


def pinned_fabric_ip(config: ClusterConfig | None) -> str:
    """The unicast IP a configured client_bind pins, else "".

    Local mode, missing binds, loopback binds and every non-unicast form use
    hostname resolution, so each returns empty. A mapped literal returns empty
    too: the transport advertises the string as written, while every other
    check classifies the unwrapped IPv4 address.
    """
    if config is None or not config.client_bind:
        return ""
    try:
        address, _ = tcp_endpoint(config.client_bind)
        ip, mapped = unmapped_bind_ip(address)
    except ValueError:
        # Config load refuses these; a hand-built ClusterConfig may carry one.
        return ""
    unusable = mapped or ip.is_unspecified or ip.is_multicast or ip.is_loopback
    return "" if unusable else address


def _verdict(lead: str, config: ClusterConfig | None) -> tuple[bool, str]:
    """Return (ok, detail) for an unsuitable client transport lookup result."""
    pinned = pinned_fabric_ip(config)
    if pinned:
        return True, (
            f"{lead}, but cluster.client_bind pins {pinned}: the client "
            "transport advertises that IP and never resolves the hostname")
    return False, (
        f"{lead}. Multi-host needs cluster.client_bind on the fabric IP; "
        "an /etc/hosts entry is not enough")


def hostname_row(
    hostname: str, resolved: str, config: ClusterConfig | None
) -> tuple[bool, str]:
    """(ok, detail) for doctor's `hostname resolution` row."""
    if not _is_loopback(resolved):
        return True, f"{hostname} -> {resolved}"
    return _verdict(f"{hostname} -> {resolved} (loopback)", config)


def unresolved_row(config: ClusterConfig | None, exc: OSError) -> tuple[bool, str]:
    """(ok, detail) for the same row when the lookup itself failed.

    A pinned fabric address makes missing hostname DNS harmless, so the same
    configured evidence clears this row.
    """
    return _verdict(f"hostname does not resolve ({exc!r})", config)
