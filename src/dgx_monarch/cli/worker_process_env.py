"""Least-privilege environment boundary for the long-lived worker loop.

Actor launchers inherit this environment, so it is filtered before Monarch is
imported. Both ``nohup`` and systemd execute
:mod:`dgx_monarch.cli.worker_loop`; systemd ``PassEnvironment=`` entries remain
subject to the same filter.
"""
from __future__ import annotations

import os
import stat
from collections.abc import MutableMapping
from pathlib import Path

# Allow only runtime identity, locale, interpreter lookup, caches, and private
# scratch. Cloud, SSH, registry, and proxy settings must not reach actor launch
# options through service inheritance.
_SAFE_EXACT = frozenset({
    "HOME",
    "USER",
    "LOGNAME",
    "PATH",
    "LANG",
    "LANGUAGE",
    "TZ",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
    # CUDA, PyTorch, thread-count and xfuser controls the project or its
    # dependencies read. Device assignment and executable or library selectors
    # are actor-owned and must stay absent.
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_MAXSIZE",
    "CUDA_CACHE_PATH",
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "XDIT_LOGGING_LEVEL",
    # Project controls consumed by worker or bootstrap code. Do not allow
    # the whole DGXM_ namespace: DGXM_SSH_KEY is a launcher-only credential.
    "DGXM_RDMA_QP_SPLIT",
    "DGXM_RDMA_MIN_BYTES",
    "DGXM_NUM_THREADS",
    "DGXM_NO_CUSTOM_NODES",
    "DGXM_SKIP_NODE_PACKS",
    "DGXM_SLAB_CERTIFY",
    # Fault injection requires both names and can only refuse a rank's load.
    # Neither carries credentials or enables extra operations. An incomplete pair logs a
    # warning and leaves loading unchanged (actor/load_fault.py).
    "DGXM_FAULT_LOAD_RANK",
    "DGXM_ACCEPTANCE",
})

# Locale is the only inherited namespace. Fabric settings come from validated
# cluster configuration and are installed inside each actor rather than
# inherited from the worker loop's ambient environment.
_SAFE_PREFIXES = ("LC_",)

_SECRET_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "CREDENTIAL",
    "CREDENTIALS",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "ACCESSKEY",
    "PRIVATE_KEY",
    "PRIVATEKEY",
    "PRESHARED_KEY",
    "PSK",
    "LICENSE_KEY",
    "TLS_KEY",
    "SIGNING_KEY",
    "ENCRYPTION_KEY",
    "SSH_KEY",
    "AUTH",
    "AUTHORIZATION",
    "COOKIE",
    "SESSION",
    "BEARER",
)

_DANGEROUS_EXACT = frozenset({
    "LD_PRELOAD",
    "LD_AUDIT",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "PYTHONBREAKPOINT",
    # NCCL settings owned by project invariants or the topology: they must stay
    # unset until the validated actor setup decides whether to install them.
    "NCCL_PROTO",
    "NCCL_P2P_DISABLE",
    "NCCL_LAUNCH_ORDER_IMPLICIT",
})
# Last-resort base when neither systemd nor a login shell supplied a usable
# runtime or home directory. The child path is UID-specific, atomically created,
# ownership-checked, symlink-rejected, and forced to 0700 before use.
_FALLBACK_RUNTIME_ROOT = Path("/tmp")  # noqa: S108


def _has_marker(name: str, marker: str) -> bool:
    padded = f"_{name}_"
    # Providers use both delimited (NGC_API_KEY) and compact (NGC_APIKEY)
    # suffixes. Suffix matching covers the latter without treating unrelated
    # words containing e.g. "AUTH" or "SESSION" as credentials.
    return f"_{marker}_" in padded or name.endswith(marker)


def worker_environment_key_allowed(name: str) -> bool:
    """Return whether *name* may be inherited by worker actor processes.

    Decisions use names only. Values of rejected variables are never read,
    copied, rendered, or logged.
    """
    if not name or name != name.upper() or name in _DANGEROUS_EXACT:
        return False
    if any(_has_marker(name, marker) for marker in _SECRET_MARKERS):
        return False
    return name in _SAFE_EXACT or name.startswith(_SAFE_PREFIXES)


def filter_worker_environment(environ: MutableMapping[str, str]) -> int:
    """Delete disallowed keys in place and return only the removal count."""
    rejected = tuple(name for name in environ if not worker_environment_key_allowed(name))
    for name in rejected:
        del environ[name]
    return len(rejected)


def _private_directory(path: Path, *, create_parents: bool) -> Path:
    if create_parents:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass

    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise RuntimeError(f"worker runtime path is not an owned directory: {path}")
    path.chmod(0o700, follow_symlinks=False)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise RuntimeError(f"worker runtime directory is not private: {path}")
    return path


def _trusted_runtime_base(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        return None
    try:
        info = path.lstat()
    except OSError:
        return None
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        return None
    return path


def secure_worker_runtime(environ: MutableMapping[str, str]) -> Path:
    """Create a private runtime directory and route child scratch into it."""
    xdg_base = _trusted_runtime_base(environ.get("XDG_RUNTIME_DIR"))
    if xdg_base is not None:
        runtime = _private_directory(xdg_base / "dgx-monarch", create_parents=False)
    else:
        home = environ.get("HOME")
        if home and Path(home).is_absolute():
            candidate = Path(home) / ".local" / "state" / "dgx-monarch" / "runtime"
        else:
            candidate = _FALLBACK_RUNTIME_ROOT / f"dgx-monarch-{os.geteuid()}"
        try:
            runtime = _private_directory(candidate, create_parents=True)
        except OSError:
            runtime = _private_directory(
                _FALLBACK_RUNTIME_ROOT / f"dgx-monarch-{os.geteuid()}",
                create_parents=False,
            )

    rendered = str(runtime)
    environ["XDG_RUNTIME_DIR"] = rendered
    environ["TMPDIR"] = rendered
    environ["TMP"] = rendered
    environ["TEMP"] = rendered
    return runtime


def prepare_worker_process_environment(
    managed_pythonpath: str | os.PathLike[str],
    environ: MutableMapping[str, str] | None = None,
) -> tuple[int, Path]:
    """Install the worker boundary before importing Monarch.

    Never restore the caller's umask: 077 must hold for the worker loop and
    every actor process it launches.
    """
    os.umask(0o077)
    managed = Path(managed_pythonpath)
    if not managed.is_absolute():
        raise ValueError("managed worker Python path must be absolute")
    target = os.environ if environ is None else environ
    removed = filter_worker_environment(target)
    # Never retain ambient import paths. The worker entrypoint supplies the
    # resolved package root, and remote actor bootstrap consumes the same root
    # through DGXM_PYTHONPATH.
    target["PYTHONPATH"] = str(managed)
    target["DGXM_PYTHONPATH"] = str(managed)
    from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX

    target["PYTHONDONTWRITEBYTECODE"] = "1"
    target["PYTHONPYCACHEPREFIX"] = SOURCE_ONLY_PYCACHE_PREFIX
    return removed, secure_worker_runtime(target)
