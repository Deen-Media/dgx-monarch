"""Data-only validation of Studio files bound by the Python source inventory."""

from __future__ import annotations

import inspect


def validate_studio_resources(root: str, require_all: bool = True) -> dict[str, str]:
    """Read literal hashes, without importing or executing candidate Python code.

    The caller must first bind the candidate Python source digest to its reviewed
    expected source digest. Empty legacy packages have no Studio resource table.
    """
    import ast
    import grp
    import hashlib
    import os
    import pathlib
    import pwd
    import re
    import stat

    package = pathlib.Path(root)
    manifest = package / "studio/resource_hashes.py"
    resource_roots = [package / "studio/data", package / "studio/static"]

    def read(path: pathlib.Path) -> bytes:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or before.st_size > 32 * 1024 * 1024:
                raise RuntimeError("unsafe Studio resource")
            if before.st_mode & 0o002:
                raise RuntimeError("world-writable Studio resource")
            if before.st_mode & 0o020:
                try:
                    owner = pwd.getpwuid(os.geteuid()).pw_name
                    group = grp.getgrgid(before.st_gid)
                    members = {entry.pw_name for entry in pwd.getpwall() if entry.pw_gid == before.st_gid} | set(group.gr_mem)
                except KeyError as exc:
                    raise RuntimeError("unknown Studio resource group ownership") from exc
                if members - {owner}:
                    raise RuntimeError("Studio resource is writable by another owner")
            chunks = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final = path.lstat()
        identities = [(item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
                      for item in (before, after, final)]
        if identities[0] != identities[1] or identities[1] != identities[2]:
            raise RuntimeError("Studio resource changed during validation")
        return b"".join(chunks)

    found: dict[str, pathlib.Path] = {}
    for directory in resource_roots:
        if not os.path.lexists(directory):
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError("unsafe Studio resource directory")
        for current, directories, files in os.walk(directory, followlinks=False):
            base = pathlib.Path(current)
            if any((base / name).is_symlink() for name in directories):
                raise RuntimeError("symlinked Studio resource directory")
            for name in files:
                path = base / name
                found[path.relative_to(package).as_posix()] = path
    if not os.path.lexists(manifest):
        if found:
            raise RuntimeError("Studio resources lack their source-bound hash manifest")
        return {}
    try:
        tree = ast.parse(read(manifest).decode("utf-8"), filename="resource_hashes.py")
    except (SyntaxError, UnicodeError) as exc:
        raise RuntimeError("Studio resource manifest is not valid literal Python") from exc
    statements = [node for node in tree.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                                                    and isinstance(node.value.value, str))]
    if len(statements) != 1 or not isinstance(statements[0], ast.Assign):
        raise RuntimeError("Studio resource manifest must contain one literal assignment")
    assignment = statements[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "RESOURCE_SHA256":
        raise RuntimeError("Studio resource manifest has an unsupported declaration")
    try:
        table = ast.literal_eval(assignment.value)
    except (ValueError, TypeError, RecursionError) as exc:
        raise RuntimeError("Studio resource manifest contains nonliteral data") from exc
    exact = {"studio/data/parity.json", "studio/data/presentation.json", "studio/data/workflows.json", "studio/static/index.html"}
    if not isinstance(table, dict) or len(table) > 2048 or not exact <= set(table):
        raise RuntimeError("Studio resource manifest has an incomplete inventory")
    for relative, expected in table.items():
        if (not isinstance(relative, str) or not isinstance(expected, str)
                or re.fullmatch(r"[a-f0-9]{64}", expected) is None
                or (relative not in exact and re.fullmatch(r"studio/static/assets/[A-Za-z0-9_.-]+-[A-Za-z0-9_-]{8,}\.(?:js|css)", relative) is None)):
            raise RuntimeError("Studio resource manifest contains an unsupported path or hash")
    if set(found) - set(table) or (require_all and set(found) != set(table)):
        raise RuntimeError("Studio resource inventory differs from its source-bound manifest")
    for relative, path in found.items():
        if hashlib.sha256(read(path)).hexdigest() != table[relative]:
            raise RuntimeError("Studio resource content differs from its source-bound manifest")
    return table


def studio_resource_validation_source() -> str:
    """Embed this fixed validator in isolated scripts, never candidate imports."""
    return inspect.getsource(validate_studio_resources)
