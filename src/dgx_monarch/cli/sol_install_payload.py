"""Self-contained host implementation for reviewed optional Sol installation.

Executed under the shared lifecycle lock. Staging has disk/network effects;
only apply mutates packages. No service, cache or system package repair.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import re
import stat
import subprocess
import sys
import sysconfig
import zipfile
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict, cast, overload

from .sol_install_guard import require_idle
from .sol_install_inventory import distribution_inventory
from .sol_install_requirements import validate_wheel_requirements

SOL_SHA = "71350faed59a6865173ddfd5a7d4471da4bde328"
ROOT_NAMES = {"sol-attn", "nvidia-cutlass-dsl", "apache-tvm-ffi"}
REQUESTS = [f"sol-attn @ git+https://github.com/NVlabs/Sana@{SOL_SHA}#subdirectory=techniques/sparse_backends",
            "nvidia-cutlass-dsl==4.7.0", "apache-tvm-ffi==0.1.14.post1"]



class PackageChange(TypedDict):
    name: str
    old: str | None
    new: str
    url: str
    archive_sha256: str | None


class Artifact(TypedDict):
    file: str
    sha256: str


class InstallPlan(TypedDict):
    schema: int
    token: str
    python: str
    allowed: list[str]
    before: dict[str, str]
    protected: str
    changes: list[PackageChange]
    artifacts: list[Artifact]
    baseline_violations: list[str]
    digest: str


class SolInstallError(RuntimeError):
    """Only installer-authored bounded errors may be presented."""

def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def protected(name: str) -> bool:
    name = canonical(name)
    if name in {"cuda-python", "cuda-bindings", "cuda-pathfinder", "nvidia-cuda-nvdisasm"}:
        return False
    return (name in {"torch", "torchvision", "torchaudio", "torchmonarch", "triton", "xfuser", "yunchang", "dgx-monarch"}
            or (name.startswith(("torch-", "cuda-", "nccl", "nvidia-"))
            and not name.startswith("nvidia-cutlass-dsl")))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def installed() -> dict[str, str]:
    result = {}
    for distribution in distribution_inventory().values():
        name = canonical(distribution.metadata["Name"])
        if name in result:
            raise SolInstallError("duplicate installed distribution metadata")
        result[name] = distribution.version
    return result


def scheme_roots() -> set[Path]:
    prefix = Path(sys.prefix)
    if sys.prefix == sys.base_prefix or prefix.is_symlink() or prefix.resolve() != prefix:
        raise SolInstallError("supported owned virtual environment required")
    roots = {Path(sysconfig.get_path(key)) for key in ("purelib", "platlib", "scripts")}
    # Existing RECORDs may own venv-local headers/manpages. This admits only
    # these existing inventory roots, never the whole prefix as a data target.
    roots.update(prefix / name for name in ("include", "share") if os.path.lexists(prefix / name))
    for root in roots:
        if not root.is_relative_to(prefix):
            raise SolInstallError("installation scheme escapes virtual environment")
        info = prefix.lstat()
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise SolInstallError("virtual environment not owned by current user")
        path = prefix
        for part in root.relative_to(prefix).parts:
            path = path / part
            info = path.lstat()
            if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
                raise SolInstallError("unsafe virtual environment scheme directory")
    return roots


def inventory_path(distribution: md.Distribution, item: str | md.PackagePath) -> Path:
    path = Path(os.path.abspath(str(distribution.locate_file(item))))
    roots = scheme_roots()
    root = next((root for root in roots if path.is_relative_to(root)), None)
    if root is None:
        raise SolInstallError("installed inventory escapes supported interpreter scheme")
    parent = root
    for part in path.relative_to(root).parts:
        parent = parent / part
        if parent.is_symlink():
            raise SolInstallError("installed inventory crosses a symlink")
    return path


def protected_state() -> tuple[str, set[str]]:
    rows = []
    for distribution in distribution_inventory().values():
        name = canonical(distribution.metadata["Name"])
        if not protected(name):
            continue
        files = distribution.files
        if files is None:
            raise SolInstallError("protected distribution inventory unavailable")
        for item in files:
            path = inventory_path(distribution, item)
            if str(path).endswith(".pyc") or "__pycache__" in path.parts:
                continue
            if path.is_symlink() or not path.is_file():
                raise SolInstallError("protected inventory not regular")
            rows.append((str(path), file_hash(path)))
    if "torch" not in installed():
        raise SolInstallError("working Torch installation required")
    return digest(sorted(rows)), {path for path, _ in rows}


def private_plan(token: str, create: bool = False) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", token):
        raise SolInstallError("invalid plan token")
    root = Path.home()
    for part in (".local", "state", "dgx-monarch", "sol-install", token):
        root = root / part
        if create:
            try:
                root.mkdir(mode=0o700)
            except FileExistsError:
                pass
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or root.is_symlink() or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise SolInstallError("unsafe staging directory")
    if root.stat().st_mode & 0o077:
        raise SolInstallError("plan staging is not private")
    return root


def run_pip(args: Sequence[str], timeout: int = 900) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
    # Ignore install redirection and indexes from ambient pip configuration.
    result = subprocess.run([sys.executable, "-I", "-m", "pip", "--isolated", "--require-virtualenv", *args], capture_output=True, text=True,
                            timeout=timeout, env=env)
    if result.returncode:
        raise SolInstallError("pip operation failed; inspect package compatibility separately")


def validate_plan(report: Mapping[str, Any], before: Mapping[str, str], allowed: Collection[str]) -> list[PackageChange]:
    changes: list[PackageChange] = []
    for item in report.get("install", []):
        meta = item["metadata"]
        name, version = canonical(meta["name"]), meta["version"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name):
            raise SolInstallError("invalid resolver distribution name")
        if protected(name):
            raise SolInstallError(f"protected package {name} cannot change")
        old = before.get(name)
        if old is not None and name not in allowed and name not in ROOT_NAMES:
            raise SolInstallError(f"{name} requires --allow-change")
        url = item["download_info"]["url"]
        vcs = item["download_info"].get("vcs_info")
        if vcs:
            if name != "sol-attn" or vcs.get("commit_id") != SOL_SHA or vcs.get("vcs") != "git":
                raise SolInstallError("unreviewed source revision")
            url = REQUESTS[0].split(" @ ", 1)[1]
        elif not url.startswith("https://") or not url.split("?", 1)[0].endswith(".whl"):
            raise SolInstallError("dependency must resolve to an HTTPS wheel")
        elif not re.fullmatch(r"[a-f0-9]{64}", item["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256", "")):
            raise SolInstallError("resolver wheel SHA256 unavailable")
        changes.append({"name": name, "old": old, "new": version, "url": url, "archive_sha256": item["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")})
    if len({row["name"] for row in changes}) != len(changes):
        raise SolInstallError("duplicate resolver result")
    return changes


@overload
def preserved_paths(changing: Collection[str], *, include_replaced: Literal[False] = False) -> set[str]: ...


@overload
def preserved_paths(changing: Collection[str], *, include_replaced: Literal[True]) -> tuple[set[str], set[str]]: ...


def preserved_paths(changing: Collection[str], *, include_replaced: bool = False) -> set[str] | tuple[set[str], set[str]]:
    """Refuse ownership overlaps before pip can uninstall an allowed replacement."""
    preserved: set[str] = set()
    replaced: set[str] = set()
    for distribution in distribution_inventory().values():
        target = replaced if canonical(distribution.metadata["Name"]) in changing else preserved
        files = distribution.files
        if files is None:
            raise SolInstallError("installed file ownership inventory unavailable")
        target.update(str(inventory_path(distribution, item)) for item in files
                      if not str(item).endswith(".pyc"))
    if preserved & replaced:
        raise SolInstallError("allowed replacement overlaps another installed distribution")
    return (preserved, replaced) if include_replaced else preserved


def check_destination(target: str, protected_paths: Collection[str], replaceable_paths: Collection[str],
                      directory: bool = False) -> None:
    path = Path(target)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise SolInstallError("wheel destination crosses a symlink")
    if target in protected_paths or (not directory and os.path.lexists(target) and target not in replaceable_paths):
        raise SolInstallError("wheel overlaps protected package files")


def replacement_paths(name: str) -> set[str]:
    distribution = distribution_inventory().get(canonical(name))
    if distribution is None:
        return set()
    if distribution.files is None:
        raise SolInstallError("replacement ownership inventory unavailable")
    return {str(inventory_path(distribution, item)) for item in distribution.files}


def wheel_identity(path: Path, protected_paths: Collection[str], replaceable_paths: Collection[str] = (),
                   staged_destinations: set[str] | None = None) -> tuple[str, str]:
    import configparser
    from email.parser import BytesParser

    from pip._vendor.packaging.utils import parse_wheel_filename

    name, version, _, _ = parse_wheel_filename(path.name)
    wheel_destinations: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        metadata_files = [item for item in archive.namelist() if item.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise SolInstallError("wheel metadata ambiguous")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        if (canonical(metadata["Name"]), metadata["Version"]) != (canonical(name), str(version)):
            raise SolInstallError("wheel filename and metadata differ")
        for entry_file in (item for item in archive.namelist() if item.endswith(".dist-info/entry_points.txt")):
            class ExactConfigParser(configparser.ConfigParser):
                def optionxform(self, optionstr: str) -> str:
                    return optionstr
            parser = ExactConfigParser(interpolation=None)
            parser.read_string(archive.read(entry_file).decode("utf-8"))
            for group in ("console_scripts", "gui_scripts"):
                for script in parser[group] if group in parser else ():
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+", script):
                        raise SolInstallError("unsafe wheel entry point")
                    target = str(Path(sysconfig.get_path("scripts")) / script)
                    check_destination(target, protected_paths, replaceable_paths)
                    wheel_destinations.add(target)
        for item in archive.namelist():
            relative = Path(item)
            if relative.is_absolute() or ".." in relative.parts:
                raise SolInstallError("unsafe wheel member")
            destinations = [(key, relative) for key in ("purelib", "platlib")]
            if relative.parts and relative.parts[0].endswith(".data"):
                if len(relative.parts) < 3 or relative.parts[1] not in {"purelib", "platlib", "scripts"}:
                    if item.endswith("/"):
                        continue
                    raise SolInstallError("wheel contains unsupported data destinations")
                destinations = [(relative.parts[1], Path(*relative.parts[2:]))]
            for key, member in destinations:
                target = str(Path(sysconfig.get_path(key)) / member)
                check_destination(target, protected_paths, replaceable_paths, item.endswith("/"))
                if not item.endswith("/"):
                    wheel_destinations.add(target)
    if staged_destinations is not None:
        if wheel_destinations & staged_destinations:
            raise SolInstallError("staged distributions overlap file ownership")
        staged_destinations.update(wheel_destinations)
    return canonical(name), str(version)


def requirement_violations(versions: Mapping[str, str], changing: Collection[str] = ()) -> list[str]:
    from pip._vendor.packaging.requirements import Requirement

    violations = set()
    for distribution in distribution_inventory().values():
        owner = canonical(distribution.metadata["Name"])
        if owner in changing:
            continue
        for raw in distribution.requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = canonical(requirement.name)
            if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name) for name in (owner, dependency)):
                raise SolInstallError("invalid installed distribution name")
            value = versions.get(dependency)
            if value is None or (requirement.specifier and value not in requirement.specifier):
                violations.add(owner + ":" + dependency)
    return sorted(violations)


def plan_host(token: str, allowed: Collection[str]) -> InstallPlan:
    root = private_plan(token, True)
    if (root / "plan.json").exists():
        raise SolInstallError("plan already exists")
    scheme_roots()
    before = installed()
    protected_hash, paths = protected_state()
    constraints = root / "constraints.txt"
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(before.items())
                                   if protected(name)))
    report_path = root / "resolve.json"
    run_pip(["install", "--dry-run", "--report", str(report_path), "--constraint", str(constraints), *REQUESTS])
    changes = validate_plan(json.loads(report_path.read_text()), before, allowed)
    changing = {row["name"] for row in changes}
    preserved, _replaced = preserved_paths(changing, include_replaced=True)
    paths |= preserved
    projected = dict(before)
    projected.update({row["name"]: row["new"] for row in changes})
    baseline_violations = requirement_violations(before)
    new_violations = sorted(set(requirement_violations(projected, changing)) - set(baseline_violations))
    if new_violations:
        owner, dependency = new_violations[0].split(":")
        raise SolInstallError(f"plan breaks {owner} requirement for {dependency}")
    wheels = root / "wheels"
    wheels.mkdir(mode=0o700)
    artifacts: list[Artifact] = []
    staged_destinations: set[str] = set()
    for change in changes:
        previous = set(wheels.iterdir())
        run_pip(["wheel", "--no-deps", "--wheel-dir", str(wheels), change["url"]])
        added = set(wheels.iterdir()) - previous
        if len(added) != 1:
            raise SolInstallError("wheel staging ambiguous")
        wheel = added.pop()
        if change["archive_sha256"] is not None and file_hash(wheel) != change["archive_sha256"]:
            raise SolInstallError("downloaded wheel differs from resolver SHA256")
        if wheel_identity(wheel, paths, replacement_paths(change["name"]), staged_destinations) != (change["name"], change["new"]):
            raise SolInstallError("staged wheel metadata differs from plan")
        validate_wheel_requirements(wheel, projected)
        artifacts.append({"file": wheel.name, "sha256": file_hash(wheel)})
    if installed() != before or protected_state()[0] != protected_hash:
        raise SolInstallError("environment changed during planning")
    plan: dict[str, Any] = {"schema": 1, "token": token, "python": sys.executable,
            "allowed": sorted(allowed), "before": before, "protected": protected_hash,
            "changes": changes, "artifacts": artifacts, "baseline_violations": baseline_violations}
    plan["digest"] = digest(plan)
    with (root / "plan.json").open("x") as stream:
        os.chmod(stream.fileno(), 0o600)
        json.dump(plan, stream, sort_keys=True)
    return cast(InstallPlan, plan)


def read_plan(token: str) -> InstallPlan:
    root = private_plan(token)
    path = root / "plan.json"
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise SolInstallError("unsafe plan file")
    plan = json.loads(path.read_text())
    claimed = plan.pop("digest")
    if digest(plan) != claimed or plan["token"] != token:
        raise SolInstallError("plan changed")
    plan["digest"] = claimed
    return cast(InstallPlan, plan)


def verify_requirements(names: Collection[str]) -> None:
    from pip._vendor.packaging.requirements import Requirement

    inventory = distribution_inventory()
    for name in names:
        for raw in inventory[canonical(name)].requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            version = inventory[canonical(requirement.name)].version
            if requirement.specifier and version not in requirement.specifier:
                raise SolInstallError("installed dependency requirements unsatisfied")


def apply_host(token: str, expected: str, allowed: Collection[str]) -> InstallPlan:
    plan = read_plan(token)
    if plan["digest"] != expected or plan["allowed"] != sorted(allowed) or plan["python"] != sys.executable:
        raise SolInstallError("reviewed plan identity changed")
    if installed() != plan["before"] or protected_state()[0] != plan["protected"]:
        raise SolInstallError("environment changed after planning")
    root = private_plan(token)
    wheels: list[str] = []
    staged_destinations: set[str] = set()
    projected = dict(plan["before"])
    projected.update({row["name"]: row["new"] for row in plan["changes"]})
    preserved, _replaced = preserved_paths({row["name"] for row in plan["changes"]}, include_replaced=True)
    protected_paths = protected_state()[1] | preserved
    for change, artifact in zip(plan["changes"], plan["artifacts"], strict=True):
        name = artifact["file"]
        if Path(name).name != name:
            raise SolInstallError("invalid artifact name")
        path = root / "wheels" / name
        if path.is_symlink() or file_hash(path) != artifact["sha256"]:
            raise SolInstallError("staged artifact changed")
        if wheel_identity(path, protected_paths, replacement_paths(change["name"]), staged_destinations) != (change["name"], change["new"]):
            raise SolInstallError("artifact identity changed")
        validate_wheel_requirements(path, projected)
        wheels.append(str(path))
    require_idle()  # Embedded by the command immediately before package mutation.
    failure = None
    try:
        if wheels:
            run_pip(["install", "--no-index", "--no-deps", *wheels])
    except BaseException as exc:
        failure = exc
    # Read back even a failed pip invocation: it may have partly changed packages.
    expected_after = dict(plan["before"])
    expected_after.update({row["name"]: row["new"] for row in plan["changes"]})
    readback: dict[str, bool | None] = {"protected_unchanged": None, "packages_match": None,
                "requirements_match": None, "pip_completed": failure is None}
    try:
        after = installed()
        readback["packages_match"] = after == expected_after
        readback["requirements_match"] = not (
            set(requirement_violations(after)) - set(plan["baseline_violations"]))
    except Exception:
        pass
    try:
        readback["protected_unchanged"] = protected_state()[0] == plan["protected"]
    except Exception:
        pass
    result_path = root / "readback.json"
    if not result_path.exists():
        with result_path.open("x") as stream:
            os.chmod(stream.fileno(), 0o600)
            json.dump(readback, stream, sort_keys=True)
    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        raise failure
    if failure is not None:
        raise SolInstallError("pip failed; post-attempt readback retained; partial maintenance") from None
    if not all(readback.values()):
        raise SolInstallError("package readback or protected content differs; partial maintenance")
    verify_requirements({"sol-attn", "nvidia-cutlass-dsl", "apache-tvm-ffi"}
                        | {row["name"] for row in plan["changes"]})
    return cast(InstallPlan, plan)
