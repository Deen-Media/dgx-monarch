"""Passive, bounded Sol compiler observations in the configured interpreter.

The remote payload is self-contained. It never imports a kernel or changes a
cache. Login-shell cache settings do not attest the running Worker environment.
"""
from __future__ import annotations

import inspect


def dependency_state() -> str:
    """Check active installed requirements, including transitive protobuf bounds.

    Only default requirements apply: installing optional extras is not implied.
    Missing metadata/parser or an oversized graph is unknown, never compatible.
    Return distribution names only, not arbitrary requirement/error text.
    """
    import importlib.metadata as metadata
    import re

    try:
        from packaging.requirements import Requirement
    except ImportError:
        return "unknown"
    pending = ["sol-attn", "apache-tvm-ffi", "nvidia-cutlass-dsl"]
    seen: set[str] = set()
    problems: set[str] = set()
    unknown = False
    count = 0
    while pending and len(seen) < 64:
        name = pending.pop()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in seen:
            continue
        seen.add(name)
        try:
            requirements = metadata.requires(name)
            if requirements is None:
                requirements = []
            for raw in requirements:
                count += 1
                if count > 512:
                    return "unknown"
                req = Requirement(raw)
                if req.marker and not req.marker.evaluate({"extra": ""}):
                    continue
                dependency = re.sub(r"[-_.]+", "-", req.name).lower()
                if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", dependency):
                    unknown = True
                    continue
                try:
                    version = metadata.version(req.name)
                except metadata.PackageNotFoundError:
                    problems.add(dependency)
                    continue
                if req.url or req.extras:
                    unknown = True  # URL pins and requested extras need additional evidence.
                if req.specifier and not req.specifier.contains(version, prereleases=True):
                    problems.add(dependency)
                pending.append(dependency)
        except Exception:
            unknown = True
    if pending:
        unknown = True
    if problems:
        return "conflict:" + ",".join(sorted(problems)[:12])
    return "unknown" if unknown else "observed"


def header_state() -> str:
    import os
    import sysconfig

    try:
        path = os.path.join(sysconfig.get_paths()["include"], "Python.h")
        return "present" if os.path.isfile(path) and os.access(path, os.R_OK) else "missing"
    except Exception:
        return "unknown"


def default_cache_state() -> str:
    """Inspect only the account-home Triton candidate, never claim actor access.

    TRITON_CACHE_DIR is filtered by worker_process_env; HOME may differ between
    a login shell and service. The actual running actor's cache remains unknown.
    Existing child entries are not traversed, so parent access proves no writes.
    """
    import os
    import pwd
    import stat
    from pathlib import Path

    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        candidate = home / ".triton" / "cache"
        for path in (home, home / ".triton", candidate):
            try:
                info = path.lstat()
            except FileNotFoundError:
                return "absent"
            if stat.S_ISLNK(info.st_mode):
                return "unknown"
            if not stat.S_ISDIR(info.st_mode):
                return "blocked"
            if info.st_uid != os.getuid() or not os.access(path, os.W_OK | os.X_OK):
                return "blocked"
        return "parent-access"
    except Exception:
        return "unknown"


PROBE_SOURCE = "\n\n".join(inspect.getsource(fn) for fn in (
    dependency_state, header_state, default_cache_state,
))
