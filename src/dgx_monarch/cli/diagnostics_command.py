"""Read-only bounded capture of the driver's existing telemetry endpoint."""
from __future__ import annotations

import http.client
import json
import sys
import time
from urllib.parse import urlsplit

from .. import diagnostic_session

_MAX_RESPONSE_BYTES = 2_097_152


def fetch_telemetry(host: str, timeout_s: float) -> object:
    """Fetch fixed-path JSON without redirects, with a byte and time budget."""
    parsed = urlsplit(f"http://{host}")
    if (not parsed.hostname or parsed.username or parsed.password or parsed.path
            or parsed.query or parsed.fragment):
        raise ValueError("diagnostics host must be HOST:PORT without credentials or a path")
    deadline = time.monotonic() + timeout_s
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=timeout_s)
    try:
        connection.request("GET", "/dgxm/telemetry")
        socket = connection.sock
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("telemetry returned an unsuccessful status")
        chunks = []
        size = 0
        while True:
            if response.isclosed():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("diagnostics fetch exceeded its deadline")
            if socket is not None:
                socket.settimeout(remaining)
            chunk = response.read1(min(16_384, _MAX_RESPONSE_BYTES - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_RESPONSE_BYTES:
                raise ValueError("telemetry response exceeds capture input limit")
            chunks.append(chunk)
        return json.loads(b"".join(chunks))
    finally:
        connection.close()


def run(args, *, resolve_host) -> int:
    if args.capabilities:
        print(json.dumps(diagnostic_session.capabilities(), indent=2))
        return 0
    if not args.output:
        print("error: diagnostics capture requires --output to a new private artifact", file=sys.stderr)
        return 2
    try:
        host = resolve_host(args.host)
        capture = diagnostic_session.collect(
            lambda timeout: fetch_telemetry(host, timeout), duration_s=args.duration_s,
            max_events=args.max_events, max_bytes=args.max_bytes)
        output = capture.export(args.output)
    except (OSError, ValueError) as error:
        print(f"error: diagnostics capture could not complete ({type(error).__name__})", file=sys.stderr)
        return 2
    report = capture.report()
    partial = report["capture"]["partial"]
    print(f"{'Partial capture' if partial else 'Capture'} saved: {output}")
    print("Read-only telemetry observations; not ownership, cleanup, or fidelity proof.")
    return 3 if partial else 0
