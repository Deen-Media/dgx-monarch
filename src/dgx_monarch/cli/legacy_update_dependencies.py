"""Candidate dependency admission for the non-transactional plain updater."""
from __future__ import annotations

import inspect
import json
import re
import shlex
import subprocess
import sys
from typing import Any

from ..config import ClusterConfig
from . import lifecycle
from .update_dependency_contract import check_contract, dependency_contract, parse_wheel_metadata
from .update_dependency_diagnostics import report_result


def candidate_errors(pin: str, roots: list[str]) -> list[str]:
    """Download only this interpreter's candidate wheel; never install or import it."""
    import hashlib
    import importlib.metadata as metadata
    import tempfile
    import zipfile
    from pathlib import Path, PurePosixPath

    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", pin):
            return ["candidate wheel could not be verified"]
        try:
            installed = metadata.distribution("torchmonarch")
        except metadata.PackageNotFoundError:
            installed = None
        # No pin change needs no download. Keep already-installed plain updates
        # usable offline while checking that runtime's declared prerequisites.
        if installed is not None and installed.version == pin:
            current = installed.read_text("METADATA")
            if current is None or len(current) > 1 << 20:
                return ["candidate dependency inspection failed"]
            payload = current.encode("utf-8")
            name, version = parse_wheel_metadata(payload, ValueError)
            if re.sub(r"[-_.]+", "-", name).lower() != "torchmonarch" or version != pin:
                return ["candidate dependency inspection failed"]
            return check_contract(dependency_contract(payload, pin, ValueError), roots)
        with tempfile.TemporaryDirectory(prefix="dgxm-dependency-check-") as scratch:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "download", "--no-deps", "--no-cache-dir",
                 "--disable-pip-version-check", "--only-binary=:all:", "--dest", scratch,
                 f"torchmonarch=={pin}"],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode:
                return ["candidate wheel download failed"]
            wheels = list(Path(scratch).iterdir())
            if len(wheels) != 1 or wheels[0].suffix != ".whl" or wheels[0].is_symlink():
                return ["candidate wheel could not be verified"]
            wheel = wheels[0]
            with wheel.open("rb") as stream:
                before = hashlib.file_digest(stream, "sha256").digest()
            with zipfile.ZipFile(wheel) as archive:
                entries = archive.infolist()
                metadata_entries = [entry for entry in entries if
                    len(PurePosixPath(entry.filename).parts) == 2
                    and PurePosixPath(entry.filename).parts[0].endswith(".dist-info")
                    and PurePosixPath(entry.filename).name == "METADATA"]
                if len(entries) > 100_000 or len(metadata_entries) != 1 or metadata_entries[0].file_size > 1 << 20:
                    return ["candidate wheel could not be verified"]
                payload = archive.read(metadata_entries[0])
            with wheel.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").digest() != before:
                    return ["candidate wheel could not be verified"]
            name, version = parse_wheel_metadata(payload, ValueError)
            if re.sub(r"[-_.]+", "-", name).lower() != "torchmonarch" or version != pin:
                return ["candidate wheel could not be verified"]
            contract = dependency_contract(payload, pin, ValueError)
            return check_contract(contract, roots)
    except Exception:
        # pip and metadata exceptions can contain credentials, URLs or host paths.
        return ["candidate dependency inspection failed"]


def probe_source() -> str:
    """Run the shared closure checker in each interpreter, without a source install."""
    return "\n".join([
        "import json,re,subprocess,sys",
        "from email.parser import BytesParser",
        "from email.policy import default as email_policy",
        inspect.getsource(parse_wheel_metadata),
        inspect.getsource(dependency_contract),
        inspect.getsource(check_contract),
        inspect.getsource(candidate_errors),
        "errors = candidate_errors(sys.argv[1], json.loads(sys.argv[2]))",
        "print(json.dumps(errors))",
        "raise SystemExit(1 if errors else 0)",
    ])




def preflight(pin: str, roots: list[str], config: ClusterConfig | None) -> bool:
    """Admit every interpreter before the plain updater changes any pin or service."""
    source, payload = probe_source(), json.dumps(roots)

    def inspect_result(label: str, invoke: Any) -> bool:
        try:
            return report_result(invoke(), label)
        except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
            print(f"  {label}: candidate dependency inspection failed", file=sys.stderr)
            return False

    passed = inspect_result("driver", lambda: subprocess.run(
        [sys.executable, "-c", source, pin, payload],
        capture_output=True, text=True, timeout=210))
    if config is not None:
        script = (
            f"PYBIN={shlex.quote(config.python_bin)}\n"
            'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac\n'
            f'"$PYBIN" -c {shlex.quote(source)} {shlex.quote(pin)} {shlex.quote(payload)}\n')
        for ordinal, host in enumerate(config.hosts, 1):
            result = inspect_result(f"host_{ordinal}", lambda host=host: lifecycle.run_on_host(
                config, host, script, timeout=210))
            passed = passed and result
    return passed
