"""Report the INT8 GEMM kernel option and its prerequisites, per host and here.

The option (kernels/int8_gemm.py) runs comfy-kitchen's INT8 ConvRot GEMM as a
Triton kernel in a process that has a validated comfy-kitchen release, Triton
and Python.h for its interpreter. A process without them runs stock CUTLASS
and logs one WARNING, so these rows warn and never fail.

The rows report configuration and installed files. Doctor attaches no mesh and
cannot read a running process's environment, so it never sees which kernel a
running actor or ComfyUI uses; the ``INT8 GEMM:`` line in the Worker journal
and in the driver log names that. Triton is found, not imported, because
doctor runs beside live renders. The classifiers take parsed probe fields and
need no SSH, CUDA or Triton.
"""
from __future__ import annotations

import inspect
import os
from collections.abc import Mapping

from ..config import ClusterConfig
from ..kernels.int8_gemm import (
    CUTLASS,
    DEFAULT,
    ENV,
    TRITON,
    VALIDATED_KITCHEN,
    WORKER_ARG,
    compiler_include_dirs,
    parse,
)
from .probe_certainty import OK, WARN

ENTRY = "docs/TROUBLESHOOTING.md #138"
# Must match the probe's output prefix so doctor can select its line;
# tests/test_doctor_int8_gemm_row.py checks the contract.
LINE_PREFIX = "int8gemm="
LOCAL_NAME = "INT8 GEMM here"
UNSEEN = ("Doctor attaches no mesh and reads no running process's environment, so it cannot "
          "see the kernel that runs; the `INT8 GEMM:` line in the Worker journal and in the "
          "driver log names it")


def int8_kitchen_state() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"
    except Exception:
        return "unknown"


def int8_triton_state() -> str:
    import importlib.util

    try:
        return "installed" if importlib.util.find_spec("triton") is not None else "MISSING"
    except Exception:
        return "unknown"


def int8_header_state() -> str:
    """Python.h where Triton's launcher build looks for this interpreter."""
    import os
    import sysconfig

    try:
        scheme = sysconfig.get_default_scheme()
        if scheme == "posix_local":
            scheme = "posix_prefix"
        path = os.path.join(sysconfig.get_paths(scheme=scheme)["include"], "Python.h")
        return "present" if os.path.isfile(path) else "missing"
    except Exception:
        return "unknown"


# What each host runs under `config.python_bin`, inside the per-host probe
# doctor already sends; self-contained, with names no other probe uses.
PROBE_SOURCE = "\n\n".join(inspect.getsource(fn) for fn in (
    int8_kitchen_state, int8_triton_state, int8_header_state,
)) + ("\nprint(f\"int8gemm=probe kitchen={int8_kitchen_state()} "
      "triton={int8_triton_state()} pyheaders={int8_header_state()}\")")


def local_fields() -> dict[str, str]:
    """The probe's fields for this interpreter, read in this process."""
    return {"kitchen": int8_kitchen_state(), "triton": int8_triton_state(),
            "pyheaders": int8_header_state()}


def _problems(fields: Mapping[str, str]) -> list[str]:
    """What keeps the Triton kernel off in a process with these prerequisites."""
    problems = []
    kitchen = fields.get("kitchen", "unknown")
    if kitchen == "MISSING":
        problems.append("comfy-kitchen is not installed, so there is no INT8 GEMM to replace")
    elif kitchen == "unknown":
        problems.append("the comfy-kitchen version could not be read")
    elif kitchen not in VALIDATED_KITCHEN:
        problems.append(f"comfy-kitchen {kitchen} is not in the validated set "
                        f"({', '.join(VALIDATED_KITCHEN)})")
    triton = fields.get("triton", "unknown")
    if triton == "MISSING":
        problems.append("Triton is not installed")
    elif triton != "installed":
        problems.append("whether Triton is installed could not be read")
    headers = fields.get("pyheaders", "unknown")
    if headers == "missing":
        problems.append("Python.h is missing")
    elif headers != "present":
        problems.append("whether Python.h is present could not be read")
    return problems


def _kernel(mode: str) -> str:
    return "the Triton kernel" if mode == TRITON else "stock CUTLASS"


def _configured(worker_args: Mapping[str, object] | None) -> tuple[str, str]:
    """The actors' mode and where it comes from, as cluster.toml states it."""
    if worker_args and WORKER_ARG in worker_args:
        return str(worker_args[WORKER_ARG]), f"[worker_args] {WORKER_ARG}"
    return DEFAULT, "the default"


def host_row(host: str, fields: Mapping[str, str],
             worker_args: Mapping[str, object] | None) -> tuple[str, str, str]:
    """One (status, name, detail) row for the actors on one configured host."""
    name = f"{host} INT8 GEMM"
    mode, source = _configured(worker_args)
    if mode == CUTLASS:
        return (OK, name, f"cutlass from {source}: the actors there run stock CUTLASS. {UNSEEN}")
    if not fields:
        return (WARN, name, f"unobserved: the INT8 GEMM probe did not answer on {host}, so it is "
                            "unknown whether the actors there can run the Triton kernel, which "
                            f"{source} selects. {UNSEEN}")
    problems = _problems(fields)
    if problems:
        return (WARN, name, f"{source} selects the Triton kernel, but in the Worker venv "
                            f"{'; '.join(problems)}. The actors there run stock CUTLASS and log "
                            f"one WARNING; renders still run, without the speedup. Fix: {ENTRY}. "
                            f"{UNSEEN}")
    return (OK, name, f"{source} selects the Triton kernel, and the Worker venv has comfy-kitchen "
                      f"{fields['kitchen']} (validated), Triton and Python.h, so the actors there "
                      f"can run it. {UNSEEN}")


def _headers_on_compiler_path(environ: Mapping[str, str]) -> bool:
    """Python.h in a CPATH or C_INCLUDE_PATH directory, as the runtime's search reads them."""
    return any(os.path.isfile(os.path.join(path, "Python.h")) for path in compiler_include_dirs(environ))


def local_row(worker_args: Mapping[str, object] | None, *, local_mode: bool,
              environ: Mapping[str, str] | None = None,
              fields: Mapping[str, str] | None = None) -> tuple[str, str, str]:
    """One row for this box: a ComfyUI started from this shell, and local-mode actors.

    The driver reads DGXM_INT8_GEMM from its own environment; doctor reads the
    variable from its own, which is the one a ComfyUI started from this shell
    inherits. That ComfyUI inherits CPATH and C_INCLUDE_PATH too, so its
    Python.h search is the runtime's (``int8_gemm.python_headers``): the
    interpreter's include directory, then those directories. In local mode the
    actors run on this box and read ``[worker_args] int8_gemm`` as Worker
    actors do, and like them they never inherit either variable, so for them
    only the interpreter's include directory counts, as in the per-host probe.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV, "").strip()
    driver_mode = parse(raw or None)
    shown = f"{ENV}={raw!r}" if raw else f"{ENV} unset"
    base = dict(local_fields() if fields is None else fields)
    # Only a readable include directory without Python.h can be rescued: Triton
    # reads that directory before it compiles, so an unreadable one stops it.
    on_path = base.get("pyheaders") == "missing" and _headers_on_compiler_path(env)
    driver = {**base, "pyheaders": "present"} if on_path else base
    status = OK
    sentences = []
    parties: list[tuple[str, tuple[str, ...]]] = []
    if driver_mode is None:
        status = WARN
        sentences.append(f"{shown} in this shell names neither \"triton\" nor \"cutlass\", so a "
                         "ComfyUI started from it runs stock CUTLASS and logs one WARNING")
    else:
        sentences.append(f"A ComfyUI started from this shell selects {_kernel(driver_mode)} "
                         f"({shown})")
        if driver_mode == TRITON:
            parties.append(("that ComfyUI", tuple(_problems(driver))))
    if local_mode:
        mode, source = _configured(worker_args)
        sentences.append(f"The local actors select {_kernel(mode)} ({source})")
        if mode == TRITON:
            parties.append(("the local actors", tuple(_problems(base))))
    groups: dict[tuple[str, ...], list[str]] = {}
    for who, problems in parties:
        groups.setdefault(problems, []).append(who)
    for problems, whos in groups.items():
        if not problems:
            sentences.append("That ComfyUI finds Python.h through CPATH or C_INCLUDE_PATH, and "
                             "this interpreter has a validated comfy-kitchen and Triton" if on_path
                             else "This interpreter has a validated comfy-kitchen, Triton and Python.h")
            continue
        status = WARN
        verbs = ("runs stock CUTLASS and logs" if whos == ["that ComfyUI"]
                 else "run stock CUTLASS and log")
        sentence = (f"In this interpreter {'; '.join(problems)}, so {' and '.join(whos)} {verbs} "
                    "one WARNING")
        if on_path and "the local actors" in whos:
            sentence += ". CPATH and C_INCLUDE_PATH hold Python.h here, but no actor inherits them"
        sentences.append(f"{sentence}. Fix: {ENTRY}")
    return (status, LOCAL_NAME, ". ".join(sentences) + f". {UNSEEN}")


def select(lines: list[str]) -> dict[str, str]:
    """This probe's fields from one host's probe output; empty when it printed none."""
    line = next((line for line in lines if line.startswith(LINE_PREFIX)), "")
    return dict(token.split("=", 1) for token in line.split() if "=" in token)


def host_rows(config: ClusterConfig, per_host: Mapping[str, Mapping[str, str]]) -> list[tuple[str, str, str]]:
    """One row per configured host, in host order; a host that never answered reads unobserved."""
    return [host_row(host.name, per_host.get(host.name, {}), config.worker_args) for host in config.hosts]


def local_row_for(config: ClusterConfig | None) -> tuple[str, str, str]:
    """The row for this box from doctor's config; never raises."""
    try:
        return local_row(config.worker_args if config is not None else None,
                         local_mode=config is None or not config.hosts)
    except Exception as exc:
        return (WARN, LOCAL_NAME, f"probe unavailable ({exc!r})")
