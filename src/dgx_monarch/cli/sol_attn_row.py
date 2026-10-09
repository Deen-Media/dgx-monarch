"""Report sol-attn dependencies across configured worker interpreters.

Rendering requires sol-attn and apache-tvm-ffi. Missing nvidia-cutlass-dsl does
not refuse dispatch: a (12, 1) host can silently use the portable reference
instead of CuTe (docs/TROUBLESHOOTING.md #84).

This row reports installation, not the resolved backend. Doctor never attaches
a mesh; the actor environment selects the backend and its kernel-build log
identifies it. The classifier accepts parsed probe fields and needs no SSH,
CUDA or installed kernel.
"""
from __future__ import annotations

from collections.abc import Mapping

from .probe_certainty import OK, WARN

NAME = "sol-attn install"
ENTRY = "docs/TROUBLESHOOTING.md #84"

# Must match PROBE_SOURCE's output prefix so doctor can select its line.
# tests/test_doctor_sol_attn_row.py checks the contract.
LINE_PREFIX = "sol="

# Reuse the CuTe capability and compiler install command from the backend;
# tests/test_doctor_sol_attn_row.py checks that diagnostics stay in sync.
CUTE_CAPABILITY = "12.1"
DSL_INSTALL = "pip install nvidia-cutlass-dsl==4.7.0"

# What each host runs under `config.python_bin`, inside the per-host probe
# doctor already sends. It calls `find_spec`, not import: doctor is passive and
# runs beside a live render, and the dispatch shim tests the same predicate. It
# reads the capability from nvidia-smi because torch would build a CUDA context
# on a box that may be rendering.
PROBE_SOURCE = """import importlib.metadata as _md, importlib.util as _iu, subprocess as _sp
def _sol_spec(name):
    try:
        return _iu.find_spec(name) is not None
    except Exception:
        return False
def _sol_dist(name):
    try:
        return _md.version(name)
    except Exception:
        return "MISSING"
try:
    _smi = _sp.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                   capture_output=True, text=True, timeout=15)
    _cap = (_smi.stdout.split() or ["unknown"])[0] if _smi.returncode == 0 else "unknown"
except Exception:
    _cap = "unknown"
print(f"sol={_sol_spec('sol_attn')} tvm={_sol_spec('tvm_ffi')} "
      f"cutlass={_sol_spec('cutlass')} solver={_sol_dist('sol-attn')} "
      f"soldsl={_sol_dist('nvidia-cutlass-dsl')} cap={_cap}")"""


def _flag(fields: Mapping[str, str], key: str) -> bool:
    """One probe boolean, read in the probe's own spelling."""
    return fields.get(key) == "True"


def _hosts(names: list[str]) -> str:
    return ", ".join(sorted(names))


def _distinct(per_host: Mapping[str, Mapping[str, str]], key: str) -> str:
    """One version to print, or every version when the hosts disagree."""
    values = sorted({fields.get(key, "MISSING") for fields in per_host.values()})
    return "/".join(values) if values else "MISSING"


def _capability(raw: str) -> str:
    """A reported capability as the pair the dispatch table is keyed on."""
    parts = raw.split(".")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return f"({parts[0]}, {parts[1]})"
    return "unknown"


def _asymmetric(present: list[str], absent: list[str], package: str) -> str:
    """Detail for a package the kernel needs that some box lacks.

    apache-tvm-ffi can be missing everywhere while sol-attn is installed
    everywhere, so the sentence has a form that names no present host.
    """
    where = (f"{package} is missing on {_hosts(absent)}" if not present else
             f"{package} is installed on {_hosts(present)} and missing on "
             f"{_hosts(absent)}")
    return (f"{where}. A render that selects the sol-attn kernel refuses and "
            "names the box that lacks it, because the driver checks every rank "
            "before it dispatches. Install it there or leave the kernel off: "
            f"{ENTRY}")


def _off_capability(caps: Mapping[str, str], hosts: list[str]) -> str:
    """Explain why these hosts cannot use the configured CuTe dispatch entry."""
    if all(_capability(caps[host]) == "unknown" for host in hosts):
        return (f"device capability is unknown on {_hosts(hosts)} because "
                "nvidia-smi did not answer, so this row cannot say whether the "
                "CuTe dispatch entry would apply there")
    reported = "/".join(sorted({_capability(caps[host]) for host in hosts}))
    return (f"device capability {reported} on {_hosts(hosts)} is not "
            f"{_capability(CUTE_CAPABILITY)}, so upstream's own dispatch decides "
            "the backend there")


def doctor_row(per_host: Mapping[str, Mapping[str, str]]) -> tuple[str, str, str]:
    """Return one (status, name, detail) row for all configured hosts.

    Include the row even if no host answered, keeping missing observations visible
    in the report and check count. It is never FAIL or critical: this optional
    kernel is off by default, and its absence must not fail a stock install.
    """
    blind = [host for host, fields in per_host.items() if not fields]
    if blind:
        return (WARN, NAME, f"unobserved: the sol-attn install probe did not answer "
                            f"on {_hosts(blind)}, so it is unknown whether those boxes "
                            "can run the kernel")
    installed = [host for host, fields in per_host.items() if _flag(fields, "sol")]
    if not installed:
        return (OK, NAME, "not installed on any host. The sol-attn kernel is optional "
                          "and off by default, and a render that selects it refuses "
                          "and names each box without it. To use it, install it on "
                          f"every host: {ENTRY}")
    absent = sorted(set(per_host) - set(installed))
    if absent:
        return (WARN, NAME, _asymmetric(installed, absent, "sol-attn"))
    no_tvm = sorted(host for host, fields in per_host.items() if not _flag(fields, "tvm"))
    if no_tvm:
        present = sorted(set(per_host) - set(no_tvm))
        return (WARN, NAME, _asymmetric(present, no_tvm, "apache-tvm-ffi"))
    version = _distinct(per_host, "solver")
    caps = {host: fields.get("cap", "unknown") for host, fields in per_host.items()}
    # Check the compiler before the capability, on each host that reports the
    # capability the dispatch entry needs. A box at (12, 1) without the DSL runs
    # Triton with no refusal and no warning; only the worker's INFO lines at
    # kernel build name the backend. No other host's answer may hide such a box:
    # nvidia-smi can fail on one box while another runs Triton.
    no_dsl = sorted(host for host, cap in caps.items()
                    if cap == CUTE_CAPABILITY and not _flag(per_host[host], "cutlass"))
    if no_dsl:
        return (WARN, NAME, f"sol-attn {version} is installed on every host, but "
                            f"nvidia-cutlass-dsl is missing on {_hosts(no_dsl)}. "
                            "Nothing refuses: that box runs the portable Triton path "
                            "with no warning, and the selection still reads as the fast "
                            f"kernel. Fix: run `{DSL_INSTALL}` there, then read the worker "
                            "log line `sol-attn kernel ... backend=... cute=...` again "
                            f"({ENTRY})")
    off_cap = sorted(host for host, cap in caps.items() if cap != CUTE_CAPABILITY)
    if off_cap:
        return (OK, NAME, f"sol-attn {version} and apache-tvm-ffi on every host; "
                          f"{_off_capability(caps, off_cap)}. The worker log line "
                          "`sol-attn kernel ... backend=...` names the backend that ran "
                          f"({ENTRY})")
    return (OK, NAME, f"sol-attn {version}, apache-tvm-ffi and nvidia-cutlass-dsl "
                      f"{_distinct(per_host, 'soldsl')} on every host, device "
                      f"capability {_capability(CUTE_CAPABILITY)}: the CuTe dispatch "
                      "entry would apply. This row reports install state only, never "
                      "the backend that ran: read that in the worker log line "
                      "`sol-attn kernel ... backend=... cute=...` written at kernel "
                      f"build ({ENTRY})")
