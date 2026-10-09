"""Discover local ComfyUI processes and candidate HTTP ports.

Instances can use custom ports or run alongside one another. Diagnostics share
this discovery path so each row probes the same candidates.
"""
from __future__ import annotations

from collections.abc import Iterable

_INIT_NODE = "DGXMonarchInit"
_INIT_CATEGORY = "DGX Monarch"
_INIT_OUTPUT = "DGXM_MESH"
_CONFIRM_TIMEOUT_S = 2.0
_TELEMETRY_TIMEOUT_S = 5.0


def _has_valid_port_flag(args: list[str]) -> bool:
    """Accept a valid Comfy ``--port N`` or ``--port=N`` argument."""
    for index, arg in enumerate(args):
        if arg == "--port":
            value = args[index + 1] if index + 1 < len(args) else ""
        elif arg.startswith("--port="):
            value = arg[len("--port="):]
        else:
            continue
        try:
            port = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= port <= 65535:
            return True
    return False


def looks_like_comfy(cmdline: list) -> bool:
    """Return whether a process is worth a Comfy-specific endpoint probe.

    Most launchers expose ``ComfyUI`` in the checkout path.  The shipped
    ``scripts/comfy-driver.sh`` first changes into that checkout, however, so
    psutil sees only ``python main.py ... --port N``.  Admit that exact bare
    script plus a valid port flag as a *candidate*, not a confirmation: callers
    still require the pack's exact Init-node schema from Comfy's
    ``/object_info/DGXMonarchInit`` endpoint.
    """
    args = [str(a) for a in cmdline or []]
    main_args = [arg for arg in args if arg.endswith("main.py")]
    return bool(main_args) and (
        any("comfy" in arg.lower() for arg in args)
        or ("main.py" in main_args and _has_valid_port_flag(args))
    )


def comfy_candidate_count() -> int | None:
    """Count Comfy-shaped processes, or return unknown on an incomplete scan."""
    try:
        import psutil

        denied = object()
        count = 0
        for process in psutil.process_iter(["cmdline"], ad_value=denied):
            command = process.info.get("cmdline", denied)
            if command is denied:
                return None
            if isinstance(command, (list, tuple)) and looks_like_comfy(list(command)):
                count += 1
    except Exception:
        return None
    return count


def find_comfy_ports() -> dict[int, int | None]:
    """Discover candidate local ComfyUI listen ports, not confirmations.

    HTTP consumers confirm a Comfy-specific endpoint before acting; this
    process pass only makes custom ports available to those probes.

    Returns port -> pid so the @latest resolver can use the process's cwd (a
    relative argv[0] must never resolve against doctor's cwd). 8188 and 8191
    are always listed (pid None unless found), even without psutil."""
    ports: dict[int, int | None] = {8188: None, 8191: None}
    try:
        import psutil

        for proc in psutil.process_iter(["cmdline", "pid"]):
            try:
                if not looks_like_comfy(proc.info.get("cmdline")):
                    continue
                for conn in proc.net_connections(kind="tcp"):
                    if conn.status == psutil.CONN_LISTEN and conn.laddr:
                        ports[conn.laddr.port] = proc.info.get("pid")
            except (psutil.Error, OSError):
                continue
    except Exception:
        pass  # psutil unavailable: the fixed pair still covers the defaults
    return dict(sorted(ports.items()))


def driver_candidates() -> tuple[str, ...]:
    """Candidate ``host:port`` values for a confirming loopback HTTP probe."""
    return tuple(f"127.0.0.1:{port}" for port in find_comfy_ports())


def _init_schema(payload: object) -> bool:
    """Validate Comfy's stock one-node object_info response for this pack."""
    if not isinstance(payload, dict) or set(payload) != {_INIT_NODE}:
        return False
    info = payload.get(_INIT_NODE)
    return bool(
        isinstance(info, dict)
        and info.get("name") == _INIT_NODE
        and info.get("category") == _INIT_CATEGORY
        and info.get("output") == [_INIT_OUTPUT]
    )


def _confirmed_driver(candidate: str) -> bool | None:
    """Confirm the pack's Init-node schema through Comfy's object-info endpoint.

    Return ``None`` on a timeout or transport error: neither proves absence.
    Connection refusal, HTTP errors, and schema mismatches return ``False``.
    Callers publishing activity or lease state must preserve this distinction.
    """
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
                f"http://{candidate}/object_info/{_INIT_NODE}",
                timeout=_CONFIRM_TIMEOUT_S) as response:
            payload = json.load(response)
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError as exc:
        return False if isinstance(exc.reason, ConnectionRefusedError) else None
    except (TimeoutError, OSError):
        return None
    except Exception:
        return False  # an answer this pack cannot read is not this pack
    return _init_schema(payload)


def _live_worker_count(candidate: str) -> int:
    """Optional telemetry ranking; failure leaves confirmation intact."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
                f"http://{candidate}/dgxm/telemetry",
                timeout=_TELEMETRY_TIMEOUT_S) as response:
            telemetry = json.load(response)
    except Exception:
        return 0
    workers = telemetry.get("workers") if isinstance(telemetry, dict) else None
    if not isinstance(workers, list):
        return 0
    return sum(isinstance(worker, dict) and "status_error" not in worker
               for worker in workers)


def probe_driver(candidate: str) -> int | None:
    """Return live workers for a schema-confirmed pack driver, else ``None``.

    Telemetry is ranking-only: its cold path may time out or answer 503 without
    making an already-confirmed driver disappear.
    """
    if not _confirmed_driver(candidate):
        return None
    return _live_worker_count(candidate)


def resolve_driver_host(host: str | None, ports: Iterable[int]) -> str:
    """Honor an explicit host; otherwise prefer a confirmed live mesh."""
    if host:
        return host
    responders = []
    for port in ports:
        candidate = f"127.0.0.1:{port}"
        workers = probe_driver(candidate)
        if workers is not None:
            responders.append((candidate, workers))
    return next((candidate for candidate, workers in responders if workers > 0),
                responders[0][0] if responders else "127.0.0.1:8188")


def driver_probe(host: str | None, ports: Iterable[int]) -> tuple[str | None, bool]:
    """Return a confirmed driver and whether discovery reached a definite answer.

    The second value is True if a driver confirmed or every candidate answered.
    A timeout leaves absence unproven; mutation callers must not treat that as an
    idle host.
    """
    candidates = [host] if host else []
    candidates.extend(f"127.0.0.1:{port}" for port in ports)
    observed = True
    for candidate in dict.fromkeys(candidates):
        confirmed = _confirmed_driver(candidate)
        if confirmed:
            return candidate, True
        observed = observed and confirmed is False
    return None, observed
