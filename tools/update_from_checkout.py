"""Run a reviewed fixed updater against an unchanged original installation.

Invoke with the original environment's Python, -I and -B, from a separate
clean checkout of the reviewed target commit. Normal update confirmation,
Worker ownership, pin verification, rollback and receipts remain in force.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def reviewed_source(controller: Path, requested: str | None) -> str:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
             "-c", "core.untrackedCache=false", "-C", str(controller), *arguments],
            capture_output=True, text=True, timeout=30,
            env={"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_NO_REPLACE_OBJECTS": "1",
                 "PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
        )
        if result.returncode:
            raise ValueError("recovery controller Git inspection failed")
        return result.stdout

    head = git("rev-parse", "HEAD").strip()
    target = head if requested is None else requested
    if re.fullmatch(r"[0-9a-f]{40}", target) is None or target != head:
        raise ValueError("controller HEAD must equal the reviewed full target commit")
    if git("status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("recovery controller checkout must be clean")
    expected = {}
    for row in git("ls-tree", "-r", "-z", target, "--", "src/dgx_monarch").split("\0"):
        if not row:
            continue
        metadata, relative = row.split("\t", 1)
        mode, kind, digest = metadata.split()
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError("recovery controller contains unsupported source entries")
        expected[relative] = digest
    if not expected:
        raise ValueError("recovery controller source inventory is empty")
    package = controller / "src/dgx_monarch"
    if package.is_symlink() or package.parent.is_symlink():
        raise ValueError("recovery controller source uses a symlink")
    actual = {}
    for directory, children, files in os.walk(package, followlinks=False):
        for name in children:
            if (Path(directory) / name).is_symlink():
                raise ValueError("recovery controller source uses a symlink")
        children[:] = [name for name in children if name != "__pycache__"]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("recovery controller source is not regular")
            data = path.read_bytes()
            actual[path.relative_to(controller).as_posix()] = hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False
            ).hexdigest()
    if actual != expected:
        raise ValueError("recovery controller source differs from its reviewed commit")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="original installed Git checkout")
    parser.add_argument("--config", type=Path, required=True, help="existing cluster configuration")
    parser.add_argument("--target", help="reviewed full commit; defaults to this controller checkout's HEAD")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        parser.error("run this tool with the original environment's Python using -I -B")
    if any(name == "dgx_monarch" or name.startswith("dgx_monarch.") for name in sys.modules):
        parser.error("dgx_monarch was imported before the recovery controller")
    controller = Path(__file__).resolve().parents[1]
    try:
        target = reviewed_source(controller, args.target)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    # -B stops writes but does not prevent reading unchecked cached bytecode.
    with tempfile.TemporaryDirectory(prefix="dgxm-recovery-cache-") as cache:
        sys.pycache_prefix = cache
        package_path = controller / "src/dgx_monarch"
        spec = importlib.util.spec_from_file_location(
            "dgx_monarch", package_path / "__init__.py", submodule_search_locations=[str(package_path)],
        )
        if spec is None or spec.loader is None:
            parser.error("reviewed controller package could not be loaded")
        package = importlib.util.module_from_spec(spec)
        sys.modules["dgx_monarch"] = package
        spec.loader.exec_module(package)
        from dgx_monarch.cli.update_bootstrap import launch_verified_update
        from dgx_monarch.config import load_cluster_config

        return launch_verified_update(
            load_cluster_config(args.config.expanduser().resolve(strict=True)), repo=args.repo,
            target_ref=target, assume_yes=False, driver_host=None, receipt_path=args.receipt,
            controller_repository=controller,
        )


if __name__ == "__main__":
    raise SystemExit(main())
