"""User-facing adapter for the plan-first guided setup backend."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from .. import TORCHMONARCH_PIN
from ..config import unmapped_bind_ip
from ..config_schema import MAX_CLUSTER_HOSTS, MAX_GPUS_PER_HOST, MAX_WORLD_SIZE
from ..operator_profiles import PROFILE_NAMES, ProfileRefusal
from . import setup_command
from .setup_models import CandidateHost, SetupOutcome, SetupPlan, SetupRequest

_DEFAULT_CONFIG = Path("~/.config/dgx-monarch/cluster.toml")
_INPUT_MESSAGE = "setup input is invalid; review the explicit host and setup options"
_IO_MESSAGE = "setup stopped before it finished: a local read or write failed, or an unexpected error occurred"
_PROFILE_MESSAGE = "the requested profile is unavailable for the observed hardware; use --profile balanced"
_CANCEL_MESSAGE = "setup was cancelled before completion"
_JSON_APPLY_MESSAGE = "JSON apply requires --yes so stdout remains a single machine-readable object"
_BLOCKER_ACTIONS = {
    "python_unavailable": (
        "Make the configured interpreter executable on this host, or rerun setup with "
        "--python-bin pointing to Python 3.11 or newer."
    ),
    "python_version_unsupported": (
        "Install Python 3.11 or newer, then rerun setup with --python-bin pointing to that interpreter."
    ),
    "cuda_torch_unavailable": (
        "Install a CUDA-enabled PyTorch build in the configured interpreter, confirm CUDA is available, "
        "then rerun setup."
    ),
    "torchmonarch_pin_mismatch": (
        f"Install torchmonarch=={TORCHMONARCH_PIN} in the configured interpreter, then rerun setup."
    ),
    "comfy_missing": (
        "Install ComfyUI separately, then set --comfy-dir to its checkout and rerun setup; "
        "see docs/INSTALL.md."
    ),
    "comfy_runtime_marker_missing": (
        "Point --comfy-dir at a ComfyUI checkout containing comfy/sd.py, then rerun setup; "
        "see docs/INSTALL.md."
    ),
}


class SetupCliInputError(ValueError):
    """Invalid setup input, reported with one of two fixed messages and never the input itself."""

    def __init__(self, message: str = _INPUT_MESSAGE) -> None:
        self.public_message = message if message in {_INPUT_MESSAGE, _JSON_APPLY_MESSAGE} else _INPUT_MESSAGE
        super().__init__(self.public_message)


def _prompt(label: str) -> str:
    """Prompt on stderr so an interactive JSON run keeps stdout parseable."""
    print(label, end="", file=sys.stderr, flush=True)
    return input()


def _clean(value: object, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise SetupCliInputError(_INPUT_MESSAGE)
    text = value.strip()
    if required and not text:
        raise SetupCliInputError(_INPUT_MESSAGE)
    if len(text) > 4096 or any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise SetupCliInputError(_INPUT_MESSAGE)
    return text


def _positive_int(value: object, maximum: int) -> int:
    if isinstance(value, bool):
        raise SetupCliInputError(_INPUT_MESSAGE)
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        number = int(value.strip())
    else:
        raise SetupCliInputError(_INPUT_MESSAGE)
    if not 1 <= number <= maximum:
        raise SetupCliInputError(_INPUT_MESSAGE)
    return number


def _literal_ip(value: object) -> str:
    text = _clean(value, required=True)
    try:
        address, mapped = unmapped_bind_ip(text)
    except ValueError as exc:
        raise SetupCliInputError(_INPUT_MESSAGE) from exc
    # The cluster.toml loader refuses an IPv4-mapped literal, so setup refuses
    # it too instead of writing the unwrapped address in its place.
    if mapped or address.is_unspecified or address.is_multicast:
        raise SetupCliInputError(_INPUT_MESSAGE)
    return address.compressed


def parse_host_spec(spec: object) -> CandidateHost:
    """Parse ``NAME,FABRIC_IP[,GPUS[,SSH_USER[,COMFY_DIR]]]`` strictly."""
    text = _clean(spec, required=True)
    fields = text.split(",")
    if not 2 <= len(fields) <= 5:
        raise SetupCliInputError(_INPUT_MESSAGE)
    fields.extend([""] * (5 - len(fields)))
    name = _clean(fields[0], required=True)
    if name.startswith("-") or any(char.isspace() for char in name):
        raise SetupCliInputError(_INPUT_MESSAGE)
    fabric_ip = _literal_ip(fields[1])
    gpus = 1 if not fields[2].strip() else _positive_int(fields[2], MAX_GPUS_PER_HOST)
    ssh_user = _clean(fields[3])
    if ssh_user and (ssh_user.startswith("-") or any(char.isspace() for char in ssh_user)):
        raise SetupCliInputError(_INPUT_MESSAGE)
    comfy_dir = _clean(fields[4])
    return CandidateHost(name, fabric_ip, gpus, ssh_user, comfy_dir)


def _string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise SetupCliInputError(_INPUT_MESSAGE)
    if not all(isinstance(item, str) for item in value):
        raise SetupCliInputError(_INPUT_MESSAGE)
    return tuple(value)


def _interactive_hosts(ask: Callable[[str], str]) -> tuple[CandidateHost, ...]:
    count = _positive_int(ask(f"Host count [1..{MAX_CLUSTER_HOSTS}]: "), MAX_CLUSTER_HOSTS)
    hosts = []
    for ordinal in range(1, count + 1):
        name = _clean(ask(f"Host {ordinal} SSH name: "), required=True)
        address = _literal_ip(ask(f"Host {ordinal} fabric IP: "))
        raw_gpus = ask(f"Host {ordinal} GPU count [1]: ").strip()
        gpus = 1 if not raw_gpus else _positive_int(raw_gpus, MAX_GPUS_PER_HOST)
        ssh_user = _clean(ask(f"Host {ordinal} SSH user [current user]: "))
        comfy_dir = _clean(ask(f"Host {ordinal} ComfyUI directory [cluster default]: "))
        hosts.append(CandidateHost(name, address, gpus, ssh_user, comfy_dir))
    return tuple(hosts)


def collect_hosts(specs: object, *, ask: Callable[[str], str] | None = None) -> tuple[CandidateHost, ...]:
    """Use explicit specs, or ask for an explicit inventory without scanning."""
    raw = _string_list(specs)
    hosts = tuple(parse_host_spec(spec) for spec in raw) if raw else _interactive_hosts(ask or _prompt)
    if len(hosts) > MAX_CLUSTER_HOSTS or sum(host.gpus for host in hosts) > MAX_WORLD_SIZE:
        raise SetupCliInputError(_INPUT_MESSAGE)
    if len({host.name for host in hosts}) != len(hosts):
        raise SetupCliInputError(_INPUT_MESSAGE)
    return hosts


def _value(args: Namespace, name: str, default: object = None) -> object:
    return getattr(args, name, default)


def _flag(args: Namespace, name: str) -> bool:
    value = _value(args, name, False)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise SetupCliInputError(_INPUT_MESSAGE)
    return value


def _optional_text(args: Namespace, name: str, default: str = "") -> str:
    value = _value(args, name, default)
    if value is None:
        return default
    return _clean(value)


def _path(value: object, *, default: Path | None = None, absolute: bool = False) -> Path | None:
    if value is None or (isinstance(value, str) and value == ""):
        result = default
    elif isinstance(value, (str, Path)):
        result = Path(value)
    else:
        raise SetupCliInputError(_INPUT_MESSAGE)
    if result is not None and absolute and not result.expanduser().is_absolute():
        raise SetupCliInputError(_INPUT_MESSAGE)
    return result


def request_from_args(args: Namespace, *, ask: Callable[[str], str] | None = None) -> SetupRequest:
    """Validate a parser namespace and construct the backend request."""
    ask = ask or _prompt
    raw_hosts = _value(args, "host", _value(args, "hosts"))
    hosts = collect_hosts(raw_hosts, ask=ask)
    client_value = _value(args, "client_ip")
    if client_value in (None, "") and not _string_list(raw_hosts):
        client_value = ask("Driver/client fabric IP: ")
    client_ip = _literal_ip(client_value)
    profile = _optional_text(args, "profile", "balanced") or "balanced"
    if profile not in PROFILE_NAMES:
        raise SetupCliInputError(_INPUT_MESSAGE)
    fabric = _optional_text(args, "fabric_profile") or None
    configured = _value(args, "setup_config_path")
    if configured is None or (isinstance(configured, str) and not configured):
        configured = _value(args, "config")
    config_path = _path(configured, default=_DEFAULT_CONFIG)
    if config_path is None:  # the default above makes this defensive only
        raise SetupCliInputError(_INPUT_MESSAGE)
    receipt = _path(_value(args, "receipt"), absolute=True)
    artifacts = tuple(_clean(item, required=True) for item in _string_list(_value(args, "artifact")))
    return SetupRequest(
        hosts=hosts,
        client_ip=client_ip,
        config_path=config_path,
        profile=profile,
        fabric_profile=fabric,
        python_bin=_optional_text(args, "python_bin", "python3") or "python3",
        ssh_key=_optional_text(args, "ssh_key"),
        comfy_dir=_optional_text(args, "comfy_dir"),
        transport_security=("trusted_fabric" if _flag(args, "acknowledge_trusted_fabric") else ""),
        artifacts=artifacts,
        apply=_flag(args, "apply"),
        assume_yes=_flag(args, "yes"),
        install_service=_flag(args, "install_service"),
        start_workers=_flag(args, "start_worker_service"),
        verify=_flag(args, "verify"),
        privileged_process_inspection=_flag(args, "privileged_process_inspection"),
        receipt_path=receipt,
    )


def _blocker_kind(blocker: str) -> str:
    fields = blocker.split("_", 2)
    if len(fields) == 3 and fields[0] == "host" and fields[1].isascii() and fields[1].isdigit():
        return fields[2]
    return blocker


def _plan_payload(plan: SetupPlan) -> dict[str, object]:
    payload = plan.as_dict()
    raw_blockers = payload.get("blockers")
    blockers = raw_blockers if isinstance(raw_blockers, list) else []
    payload["remediations"] = [
        {"blocker": blocker, "action": _BLOCKER_ACTIONS[kind]}
        for blocker in blockers
        if isinstance(blocker, str) and (kind := _blocker_kind(blocker)) in _BLOCKER_ACTIONS
    ]
    return payload


def print_plan(plan: SetupPlan, *, stream: TextIO | None = None) -> None:
    """Print sanitized facts followed by the exact, terminal-only config diff."""
    if stream is None:
        stream = sys.stdout
    print("setup plan (sanitized):", file=stream)
    print(json.dumps(_plan_payload(plan), indent=2, sort_keys=True), file=stream)
    print("config diff (exact; terminal only):", file=stream)
    if plan.diff:
        stream.write(plan.diff)
        if not plan.diff.endswith("\n"):
            stream.write("\n")
    else:
        print("(no config changes)", file=stream)


def _result_payload(request: SetupRequest, outcome: SetupOutcome) -> dict[str, object]:
    """Build JSON only from backend disclosure-safe views, never the plan diff."""
    payload = {"operation": "setup", **outcome.as_dict()}
    payload["plan"] = _plan_payload(outcome.plan)
    if request.receipt_path is not None and not outcome.receipt_written:
        payload["status"] = "failed" if outcome.status == "planned" else "partial"
        if not payload.get("failure"):
            payload["failure"] = "receipt_write_failed"
    return payload


def _result_code(request: SetupRequest, outcome: SetupOutcome) -> int:
    if request.receipt_path is not None and not outcome.receipt_written:
        return 1
    if outcome.failure in {"confirmation_declined", "readiness_blocked"}:
        return 2
    if outcome.status in {"failed", "partial"}:
        return 1
    return 0


def _emit_error(code: str, message: str, *, as_json: bool, stream: TextIO | None = None) -> None:
    target = stream if stream is not None else (sys.stdout if as_json else sys.stderr)
    if as_json:
        print(
            json.dumps(
                {
                    "operation": "setup",
                    "status": "refused" if code != "setup_failed" else "failed",
                    "error": {"code": code, "message": message},
                },
                sort_keys=True,
            ),
            file=target,
        )
    else:
        print(f"setup: {message}", file=target)


def run(args: Namespace) -> int:
    """Build one frozen plan, present it, and apply that same plan if requested."""
    json_value = _value(args, "json", False)
    json_mode = json_value is True
    try:
        if json_value is not None and not isinstance(json_value, bool):
            raise SetupCliInputError(_INPUT_MESSAGE)
        request = request_from_args(args)
        if json_mode and request.apply and not request.assume_yes:
            raise SetupCliInputError(_JSON_APPLY_MESSAGE)
        capture = contextlib.redirect_stdout(io.StringIO()) if json_mode else contextlib.nullcontext()
        with capture:
            plan = setup_command.build_setup_plan(request)
        if not json_mode:
            print_plan(plan)
        with capture:
            outcome = setup_command.run_setup(request, plan=plan)
    except ProfileRefusal:
        _emit_error("profile_refused", _PROFILE_MESSAGE, as_json=json_mode)
        return 2
    except SetupCliInputError as exc:
        _emit_error("input_invalid", exc.public_message, as_json=json_mode)
        return 2
    except ValueError:
        _emit_error("input_invalid", _INPUT_MESSAGE, as_json=json_mode)
        return 2
    except EOFError:
        _emit_error("cancelled", _CANCEL_MESSAGE, as_json=json_mode)
        return 2
    except KeyboardInterrupt:
        _emit_error("cancelled", _CANCEL_MESSAGE, as_json=json_mode)
        raise
    except OSError:
        _emit_error("setup_failed", _IO_MESSAGE, as_json=json_mode)
        return 1
    except Exception:
        _emit_error("setup_failed", _IO_MESSAGE, as_json=json_mode)
        return 1

    if json_mode:
        print(json.dumps(_result_payload(request, outcome), indent=2, sort_keys=True))
    else:
        display_status = (
            "failed"
            if request.receipt_path is not None and not outcome.receipt_written and outcome.status == "planned"
            else outcome.status
        )
        print(f"setup: {display_status}")
        if outcome.failure:
            print(f"  result: {outcome.failure}")
        if request.receipt_path is not None and not outcome.receipt_written:
            print("  receipt: FAILED (requested receipt was not written)")
        elif outcome.receipt_written:
            print("  receipt: written")
    return _result_code(request, outcome)
