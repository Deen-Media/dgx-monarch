"""Compare loaded NCCL versions and library hashes across cluster hosts.

Torch reports its build-time NCCL version, which may differ from a replacement
library loaded at runtime. The probe imports torch, queries the loaded library
and hashes its mapped file. Classification uses the returned fields and needs
no SSH or CUDA.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .probe_certainty import FAIL, OK, WARN

NAME = "NCCL library"

# Doctor selects this probe's line from the per-host script output by this prefix.
LINE_PREFIX = "nccl_lib="

# Run under the host's `config.python_bin` inside doctor's per-host heredoc.
# The version and the hash fail apart, so a library that answers but cannot be
# read still reports its version.
PROBE_SOURCE = """_nlib, _nsha = "unknown", "unknown"
try:
    import ctypes, hashlib, torch
    _ncode = ctypes.c_int()
    ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(_ncode))
    _nlib = str(_ncode.value)
    _npath = next(ln.split(None, 5)[5].strip() for ln in open("/proc/self/maps") if "/libnccl.so" in ln)
    _nsha = hashlib.sha256(open(_npath, "rb").read()).hexdigest()[:12]
except Exception:
    pass
print(f"nccl_lib={_nlib} nccl_sha={_nsha}")"""


def normalize_proto(value: object) -> str:
    """Canonicalize NCCL's comma-separated protocol selector for checks."""
    return ",".join(
        item.strip().upper() for item in str(value or "").split(",") if item.strip())


def uses_ll(value: object) -> bool:
    return "LL" in normalize_proto(value).split(",")


def fields(lines: Sequence[str]) -> dict[str, str]:
    """The probe's `key=value` fields out of a host's output, empty when it printed none."""
    line = next((ln for ln in lines if ln.startswith(LINE_PREFIX)), "")
    return dict(token.split("=", 1) for token in line.split() if "=" in token)


def _version(code: str) -> str:
    """`23007` as `2.30.7`, or empty when the host did not report a version."""
    if not code.isdigit() or int(code) <= 0:
        return ""
    number = int(code)
    return f"{number // 10000}.{number // 100 % 100}.{number % 100}"


def _digest(value: str) -> str:
    return value if value and value != "unknown" else ""


def doctor_row(per_host: Mapping[str, Mapping[str, str]], names: Sequence[str]) -> tuple[str, str, str]:
    """`(status, name, detail)`: one row over every configured host.

    A host in `names` with no entry in `per_host` never answered. Two hosts that
    report different versions FAIL. Equal versions with different file hashes,
    or a host that did not report a version or a hash to compare, WARN. A
    single host with a version is ok.
    """
    seen = {name: (_version(per_host.get(name, {}).get("nccl_lib", "")),
                   _digest(per_host.get(name, {}).get("nccl_sha", ""))) for name in names}
    shown = "; ".join(f"{name} {version or 'no version'} sha256 {digest or 'unknown'}"
                      for name, (version, digest) in seen.items())
    silent = [name for name, (version, _) in seen.items() if not version]
    if len({version for version, _ in seen.values() if version}) > 1:
        return (FAIL, NAME, f"hosts load different NCCL versions: {shown}. "
                            "Rebuild or relink the library so every host loads the same version")
    if len({digest for _, digest in seen.values() if digest}) > 1:
        return (WARN, NAME, f"hosts load the same NCCL version from different files: {shown}"
                            + (f"; no report from {', '.join(silent)}" if silent else ""))
    unhashed = [name for name, (version, digest) in seen.items() if version and not digest]
    if silent or (unhashed and len(seen) > 1):
        missing = ", ".join(silent + unhashed)
        return (WARN, NAME, f"unobserved: no complete report from {missing}, so it is unknown "
                            f"whether every host loads the same library ({shown})")
    return (OK, NAME, shown or "no worker hosts")
