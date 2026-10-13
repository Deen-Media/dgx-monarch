"""Compatibility implementation for the original in-place ``dgxm update``."""
from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ..config import ClusterConfig
from . import lifecycle, probe_certainty
from .legacy_update_dependencies import preflight as dependency_preflight
from .update_lock import UpdateLock, UpdateLockUnavailable

DriverProbe = Callable[[str | None], tuple[str | None, bool]]


class _Args(Protocol):
    host: str | None


def run(
    args: _Args,
    *,
    load_config: Callable[[object], ClusterConfig | None],
    driver_probe: DriverProbe,
    repo: Path | None = None,
    lock_factory: Callable[[Path], UpdateLock] | None = None,
) -> int:
    """Run the old command under the update lock `dgxm update --verify` also takes."""
    checkout = repo or Path(__file__).resolve().parents[3]
    if not (checkout / "pyproject.toml").is_file():
        print(
            f"{checkout} is not a dgx-monarch checkout (wheel install?). The project publishes "
            "no package, so `pip install -U dgx-monarch --no-deps` cannot update it; reinstall "
            "from an updated source checkout instead (docs/INSTALL.md).",
            file=sys.stderr,
        )
        return 1
    try:
        with (lock_factory or UpdateLock)(checkout):
            return _run_locked(
                args,
                repo=checkout,
                load_config=load_config,
                driver_probe=driver_probe,
            )
    except (OSError, UpdateLockUnavailable):
        print("another dgxm update is already active", file=sys.stderr)
        return 2


def _fetch_candidate(repo: Path) -> str | None:
    """Fetch as `git pull --ff-only` would and return the commit it would reach, unmerged.

    Dependency admission reads this commit's pyproject.toml, so a refusal leaves
    the checkout where it was; the merge then names this exact commit.
    """
    git = ["git", "-C", str(repo)]
    print(f"git fetch in {repo}")
    if subprocess.run([*git, "fetch"], text=True).returncode:
        print("git fetch failed; resolve it by hand, then rerun `dgxm update`", file=sys.stderr)
        return None
    try:
        path = subprocess.run([*git, "rev-parse", "--git-path", "FETCH_HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        lines = (repo / path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError):
        lines = []
    # `git pull` merges only entries not marked not-for-merge: the branch's upstream.
    heads = [fields[0] for fields in (line.split("\t") for line in lines)
             if len(fields) >= 3 and fields[1] == ""]
    if len(heads) != 1 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", heads[0]):
        print("the current branch has no single upstream to fast-forward to (git pull "
              "--ff-only refuses too); set its upstream, then rerun `dgxm update`",
              file=sys.stderr)
        return None
    if subprocess.run([*git, "merge-base", "--is-ancestor", "HEAD", heads[0]]).returncode:
        print("the checkout cannot fast-forward to its fetched upstream; resolve it by hand, "
              "then rerun `dgxm update`", file=sys.stderr)
        return None
    return heads[0]


def _run_locked(
    args: _Args,
    *,
    repo: Path,
    load_config: Callable[[object], ClusterConfig | None],
    driver_probe: DriverProbe,
) -> int:
    """Run the legacy update while holding the checkout lock.

    Proceed only when the driver probe confirms no driver is running. A live
    driver or unanswered probe blocks pulling, pin changes and worker restarts.
    """
    active, observed = driver_probe(getattr(args, "host", None))
    if active is not None:
        print(
            f"refusing to update while ComfyUI is running at {active}: the driver "
            "would keep old Python modules while restarted workers load new ones. "
            "Stop ComfyUI, rerun `dgxm update`, then start ComfyUI again.",
            file=sys.stderr,
        )
        return 2
    if not observed:
        print(
            "refusing to update: a driver probe never answered, so driver activity "
            "is unknown (activity_unknown). A driver this run could not reach would "
            "keep old Python modules while restarted workers load new ones. Rerun "
            "`dgxm update` once every probed ComfyUI port answers (the local ports "
            "and any --host), or stop ComfyUI first.",
            file=sys.stderr,
        )
        return probe_certainty.UNKNOWN_EXIT
    candidate = None
    if (repo / ".git").is_dir():
        candidate = _fetch_candidate(repo)
        if candidate is None:
            return 1
    else:
        print(f"{repo} is not a plain git checkout (no .git directory); skipping the fetch "
              "and checking the working tree's pyproject.toml")

    try:
        if candidate is None:
            project = (repo / "pyproject.toml").read_bytes()
        else:  # admit the fetched commit's contract before the checkout moves
            project = subprocess.run(
                ["git", "-C", str(repo), "show", f"{candidate}:pyproject.toml"],
                capture_output=True, check=True).stdout
        dependencies = tomllib.loads(project.decode("utf-8"))["project"]["dependencies"]
        spec = next(item for item in dependencies if item.startswith("torchmonarch=="))
        pin = spec.partition("==")[2]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", pin):
            raise ValueError(f"unsafe pin {pin!r}")
    except (KeyError, OSError, StopIteration, subprocess.CalledProcessError,
            tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError) as exc:
        print(f"cannot determine the exact torchmonarch pin: {exc}", file=sys.stderr)
        return 1

    config = load_config(args)
    if not dependency_preflight(pin, dependencies, config):
        unmoved = (" The checkout stays at its current commit; the fetch moved only "
                   "remote-tracking refs." if candidate is not None else "")
        print("update refused before pin installation or Worker changes: candidate dependencies "
              "are missing, incompatible, or could not be inspected. Review the bounded details "
              "above and install prerequisites separately using docs/INSTALL.md; preserve the "
              f"working Torch build.{unmoved}", file=sys.stderr)
        return 1
    if candidate is not None:
        print(f"git merge --ff-only {candidate[:12]} in {repo}")
        if subprocess.run(["git", "-C", str(repo), "merge", "--ff-only", candidate],
                          text=True).returncode:
            print("git merge --ff-only failed after dependency admission, before any pin or "
                  "Worker change; resolve it by hand, then rerun `dgxm update`", file=sys.stderr)
            return 1

    try:
        installed_pin = importlib.metadata.version("torchmonarch")
    except importlib.metadata.PackageNotFoundError:
        installed_pin = ""
    if installed_pin != pin:
        print(f"installing exact torchmonarch pin {pin} in the driver interpreter (--no-deps)")
        if subprocess.run(
            [sys.executable, "-m", "pip", "install", f"torchmonarch=={pin}", "--no-deps", "-q"],
            text=True,
        ).returncode:
            print("driver torchmonarch pin install failed; workers were not restarted", file=sys.stderr)
            return 1

    print("pip install -e (package only, --no-deps: torch and NCCL stay as installed)")
    if subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(repo), "--no-deps", "-q"],
        text=True,
    ).returncode:
        return 1
    if config is not None and config.hosts:
        if not lifecycle.ensure_torchmonarch_pin(config, pin):
            print("worker torchmonarch pin install failed; workers were not restarted", file=sys.stderr)
            return 1
        config_args = ["--config", str(config.source)] if config.source else []
        if subprocess.run(
            [sys.executable, "-m", "dgx_monarch", *config_args, "restart"], text=True
        ).returncode:
            return 1
        doctor = subprocess.run(
            [sys.executable, "-m", "dgx_monarch", *config_args, "doctor"], text=True
        )
        return 0 if doctor.returncode == 0 else 1
    return 0
