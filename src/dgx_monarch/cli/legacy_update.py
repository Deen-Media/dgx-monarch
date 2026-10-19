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
from .release_slot import TROUBLESHOOTING as SLOT_TROUBLESHOOTING
from .release_slot import release_slot_hosts
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


def _linked_worktree(repo: Path) -> bool:
    """Whether ``repo`` is a linked worktree, as `git worktree add` makes one.

    Its .git is a file naming a private git directory that holds a
    ``commondir`` file. A submodule or a separate git directory also has a
    .git file, but no ``commondir``, so it keeps the plain path's old behavior.
    """
    try:
        text = (repo / ".git").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if not text.startswith("gitdir: "):
        return False
    # The join keeps an absolute gitdir as it is and reads a relative one from ``repo``.
    return (repo / text.removeprefix("gitdir: ").strip() / "commondir").is_file()


# The file admission reads, and the one `pip install -e` builds metadata from.
_CONTRACT_FILES = ("pyproject.toml",)


def _contract_edited(repo: Path) -> bool:
    """Refuse a contract file whose working copy may differ from its committed blob.

    `git merge --ff-only` keeps a local edit when the fetched commit leaves the
    file alone, so admission would read the committed file while `pip install
    -e` builds from the edited one. Staged, unstaged and index-hidden edits
    (assume-unchanged or skip-worktree) all refuse; other tracked files are the
    editable tree itself, live before and after the update, and never checked.
    """
    git = ["git", "-C", str(repo)]
    for name in _CONTRACT_FILES:
        status = subprocess.run([*git, "status", "--porcelain=v1", "--untracked-files=no",
                                 "--", name], capture_output=True, text=True)
        flags = subprocess.run([*git, "ls-files", "-v", "--", name],
                               capture_output=True, text=True)
        if status.returncode or flags.returncode:
            print(f"cannot read the local state of {name}; resolve it by hand, then rerun "
                  "`dgxm update`", file=sys.stderr)
            return True
        if status.stdout.strip() or flags.stdout.split(" ", 1)[0] != "H":
            print(f"{name} has local changes (staged, unstaged, or hidden by an assume-unchanged "
                  "or skip-worktree flag) that the update would install without checking them. "
                  "Commit or discard them, then rerun `dgxm update`.", file=sys.stderr)
            return True
    return False


def _fetch_candidate(repo: Path) -> tuple[str, bool] | None:
    """Fetch as `git pull --ff-only` would; return the commit to install and whether to merge.

    Dependency admission reads this commit's pyproject.toml, so a refusal leaves
    the checkout where it was. When HEAD already contains the fetched upstream,
    as with local commits, HEAD is the commit to install and nothing merges;
    otherwise the merge names this exact commit.
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
        head = subprocess.run([*git, "rev-parse", "--verify", "HEAD^{commit}"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError):
        lines, head = [], ""
    # `git pull` merges only entries not marked not-for-merge: the branch's upstream.
    heads = [fields[0] for fields in (line.split("\t") for line in lines)
             if len(fields) >= 3 and fields[1] == ""]
    if len(heads) != 1 or not all(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha)
                                  for sha in (heads[0], head)):
        print("the current branch has no single upstream to fast-forward to (git pull "
              "--ff-only refuses too); set its upstream, then rerun `dgxm update`",
              file=sys.stderr)
        return None
    if not subprocess.run([*git, "merge-base", "--is-ancestor", heads[0], head]).returncode:
        print(f"HEAD {head[:12]} already contains its fetched upstream; checking HEAD and "
              "leaving the checkout where it is")
        return head, False
    if subprocess.run([*git, "merge-base", "--is-ancestor", head, heads[0]]).returncode:
        print("the checkout has diverged from its fetched upstream and cannot fast-forward "
              "(git pull --ff-only refuses too); resolve it by hand, then rerun `dgxm update`",
              file=sys.stderr)
        return None
    return heads[0], True


def _release_slot_refusal(config: ClusterConfig) -> int | None:
    """Refuse before the fetch when any Worker runs from a release slot, or when one went unread.

    Plain update pulls the driver forward and leaves every slot where it was,
    so it would split the cluster. A source link that resolves outside
    releases/, as guided setup's link to a colocated checkout does, follows
    the pull and does not refuse.
    """
    slots, unobserved = release_slot_hosts(config)
    for host, token in slots:
        print(
            f"refusing to update: {host.name} runs its Worker from release slot {token} "
            "(release_slot_worker). Its ~/.local/share/dgx-monarch/src links into releases/, and "
            "plain `dgxm update` cannot refresh a release slot: the driver would move while that "
            "Worker stays on its old release. Run `dgxm update --verify --target <ref>` instead "
            f"({SLOT_TROUBLESHOOTING}). Nothing was fetched or changed.",
            file=sys.stderr,
        )
    if slots:
        return 2
    for host in unobserved:
        print(
            f"refusing to update: the source link on {host.name} could not be read "
            "(release_slot_unknown), so whether its Worker runs from a release slot, which plain "
            "`dgxm update` cannot refresh, is unknown. Rerun once that host answers "
            f"({SLOT_TROUBLESHOOTING}). Nothing was fetched or changed.",
            file=sys.stderr,
        )
    return probe_certainty.UNKNOWN_EXIT if unobserved else None


def _run_locked(
    args: _Args,
    *,
    repo: Path,
    load_config: Callable[[object], ClusterConfig | None],
    driver_probe: DriverProbe,
    expected_candidate: tuple[str, bool] | None = None,
    expected_source_manifest: str | None = None,
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
    config = load_config(args)
    if config is not None and config.hosts:
        refused = _release_slot_refusal(config)
        if refused is not None:
            return refused
    candidate: str | None = None
    merge = False
    if (repo / ".git").is_dir():
        if _contract_edited(repo):
            return 1
        fetched = _fetch_candidate(repo)
        if fetched is None:
            return 1
        candidate, merge = fetched
    elif _linked_worktree(repo):
        print(
            f"{repo} is a linked git worktree: its .git is a file, not a directory. Plain "
            "`dgxm update` updates the checkout this dgxm was installed from, and it fetches "
            "and fast-forwards only a main checkout, so it refuses here rather than skip the "
            "fetch. Run `dgxm update --verify`, which fetches origin itself and fast-forwards "
            "this checkout to a verified release, or reinstall dgx-monarch from the main "
            "checkout and run plain `dgxm update` from that install (docs/INSTALL.md).",
            file=sys.stderr,
        )
        return 1
    else:
        print(f"{repo} is not a plain git checkout (no .git directory); skipping the fetch "
              "and checking the working tree's pyproject.toml")

    if expected_candidate is not None and (candidate, merge) != expected_candidate:
        print("reviewed legacy update target changed; nothing was installed or restarted", file=sys.stderr)
        return 2

    try:
        if candidate is None:
            project = (repo / "pyproject.toml").read_bytes()
        else:  # admit the commit to install before the checkout moves
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

    if not dependency_preflight(pin, dependencies, config):
        unmoved = (" The checkout stays at its current commit; the fetch moved only "
                   "remote-tracking refs." if candidate is not None else "")
        print("update refused before pin installation or Worker changes: candidate dependencies "
              "are missing, incompatible, or could not be inspected. Review the bounded details "
              "above and install prerequisites separately using docs/INSTALL.md; preserve the "
              f"working Torch build.{unmoved}", file=sys.stderr)
        return 1
    if candidate is not None and merge:
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
    if expected_source_manifest is not None and subprocess.run(
        _installed_command(repo, [], expected_source_manifest), text=True,
    ).returncode:
        print("installed source differs from review; workers were not changed", file=sys.stderr)
        return 1
    if config is not None and config.hosts:
        if not lifecycle.ensure_torchmonarch_pin(config, pin):
            print("worker torchmonarch pin install failed; workers were not restarted", file=sys.stderr)
            return 1
        config_args = ["--config", str(config.source)] if config.source else []
        if subprocess.run(
            _installed_command(repo, [*config_args, "restart"], expected_source_manifest), text=True
        ).returncode:
            return 1
        doctor = subprocess.run(
            _installed_command(repo, [*config_args, "doctor"], expected_source_manifest), text=True
        )
        return 0 if doctor.returncode == 0 else 1
    return 0


# Isolated children must load the reviewed, newly installed source, not the
# private controller package in their inherited working directory.
_INSTALLED_CLI = r"""
import importlib.util
import sys
from pathlib import Path
source, expected = Path(sys.argv[1]), sys.argv[2]
spec = importlib.util.find_spec("dgx_monarch")
if spec is None or spec.origin is None or Path(spec.origin).resolve() != source / "__init__.py":
    raise SystemExit("installed package path differs from reviewed source")
from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256
if dgx_source_manifest_sha256(source) != expected:
    raise SystemExit("installed package source differs from reviewed hash")
if sys.argv[3:]:
    from dgx_monarch.cli.main import main
    raise SystemExit(main(sys.argv[3:]))
"""


def _installed_command(repo: Path, args: list[str], manifest: str | None) -> list[str]:
    if manifest is None:
        return [sys.executable, "-m", "dgx_monarch", *args]
    return [sys.executable, "-I", "-c", _INSTALLED_CLI, str(repo / "src/dgx_monarch"), manifest, *args]
