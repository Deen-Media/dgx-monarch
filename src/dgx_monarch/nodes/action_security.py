"""Shared same-origin boundary for browser-triggered driver actions."""
from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit


def _action_request_allowed(
    headers: Mapping[str, str], request_scheme: str, action: str,
) -> tuple[bool, str]:
    """Validate an action POST's CSRF boundary without aiohttp dependencies."""
    normalized = {str(key).lower(): str(value).strip() for key, value in headers.items()}
    if normalized.get("x-dgxm-action") != action:
        return False, f"missing X-DGXM-Action: {action} (use the DGX Monarch panel)"

    fetch_site = normalized.get("sec-fetch-site", "").lower()
    if fetch_site == "cross-site":
        return False, f"cross-site {action} requests are forbidden"

    host = normalized.get("host", "")
    origin = normalized.get("origin", "")
    # An SSL-terminating reverse proxy hands aiohttp an http socket while the
    # browser's Origin says https. The client-facing hop still has to match
    # Host below, so this reconciliation does not weaken the origin boundary.
    forwarded = normalized.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    scheme = forwarded or str(request_scheme or "").lower()
    if not host or not origin or scheme not in {"http", "https"}:
        return False, f"{action} requires same-origin Host and Origin headers"

    try:
        host_parts = urlsplit(f"{scheme}://{host}")
        origin_parts = urlsplit(origin)
        if (
            not host_parts.hostname
            or host_parts.username is not None
            or host_parts.password is not None
            or host_parts.path
            or host_parts.query
            or host_parts.fragment
            or origin_parts.scheme.lower() not in {"http", "https"}
            or not origin_parts.hostname
            or origin_parts.username is not None
            or origin_parts.password is not None
            or origin_parts.path
            or origin_parts.query
            or origin_parts.fragment
        ):
            return False, f"malformed Host or Origin on {action} request"

        def endpoint(parts, endpoint_scheme: str) -> tuple[str, int]:
            default_port = 443 if endpoint_scheme == "https" else 80
            return parts.hostname.lower().rstrip("."), parts.port or default_port

        host_endpoint = endpoint(host_parts, scheme)
        origin_endpoint = endpoint(origin_parts, origin_parts.scheme.lower())
    except ValueError:
        return False, f"malformed Host or Origin on {action} request"

    if origin_parts.scheme.lower() != scheme or origin_endpoint != host_endpoint:
        return False, f"{action} Origin does not match the request Host"
    return True, ""


def _recycle_request_allowed(
    headers: Mapping[str, str], request_scheme: str,
) -> tuple[bool, str]:
    """The attached-mesh reset POST's CSRF boundary."""
    return _action_request_allowed(headers, request_scheme, "recycle")
