"""Exact torchmonarch pin change for the driver interpreter.

The prior and target wheels are downloaded and each installed into its own
private site before the live interpreter is touched; the live install then
runs offline from those wheels. Metadata and payload checks never import torchmonarch.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Protocol

from .update_cancellation import reraise_after_cleanup
from .update_payload_ownership import foreign_namespace_files
from .update_transaction import Certainty, OperationResult

_PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}")
_DIST_INFO = re.compile(r".+\.dist-info", re.IGNORECASE)
_MAX_WHEEL_MEMBERS = 100_000
_MAX_WHEEL_BYTES = 8 * 1024**3
_PROBE = """
import importlib.metadata as metadata
import json
try:
    distribution = metadata.distribution("torchmonarch")
    print(json.dumps({"root": str(distribution.locate_file("")), "version": distribution.version}))
except Exception:
    raise SystemExit(3)
"""


class DriverPinError(RuntimeError):
    """A pin artifact could not be staged or verified safely."""


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


def _default_runner(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=cwd, capture_output=capture_output, text=text,
        timeout=timeout, env=None if env is None else dict(env),
    )


@dataclass(frozen=True)
class _Artifact:
    pin: str
    wheel: Path
    wheel_sha256: str
    site: Path
    files: Mapping[str, str]
    manifest_sha256: str


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _validate_pin(pin: str) -> str:
    if not isinstance(pin, str) or _PIN.fullmatch(pin) is None:
        raise ValueError("torchmonarch pin is not a safe exact version")
    return pin


def _safe_archive_path(name: str) -> tuple[str, ...]:
    raw = name[:-1] if name.endswith("/") else name
    if (
        not raw or name.startswith("/") or "\\" in name or ":" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise DriverPinError("wheel contains an unsafe path")
    parts = tuple(raw.split("/"))
    path = PurePosixPath(raw)
    if any(part in ("", ".", "..") for part in parts) or path.as_posix() != raw:
        raise DriverPinError("wheel contains an unsafe path")
    return parts


def _installed_path(parts: tuple[str, ...]) -> tuple[str, ...] | None:
    first = parts[0]
    if _DIST_INFO.fullmatch(first):
        return None
    if first.lower().endswith(".data"):
        if len(parts) < 3:
            raise DriverPinError("wheel data path is incomplete")
        scheme = parts[1].lower()
        if scheme == "scripts":
            return None
        if scheme not in ("purelib", "platlib"):
            raise DriverPinError("wheel uses an unsupported install scheme")
        parts = parts[2:]
        if _DIST_INFO.fullmatch(parts[0]):
            return None
    if parts[-1].endswith(".pyc"):
        return None
    return parts


def _hash_reader(reader: object) -> str:
    digest = hashlib.sha256()
    while True:
        block = reader.read(1024 * 1024)  # type: ignore[attr-defined]
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _hash_file(path: Path) -> str:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise DriverPinError("installed payload contains a non-regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DriverPinError("installed payload contains a non-regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return _hash_reader(handle)
    finally:
        os.close(descriptor)


def _manifest_digest(files: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, content_hash in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(content_hash))
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_metadata(payload: bytes) -> tuple[str, str]:
    try:
        message = BytesParser(policy=email_policy).parsebytes(payload, headersonly=True)
        name = str(message["Name"])
        version = str(message["Version"])
    except Exception as exc:
        raise DriverPinError("wheel METADATA is unreadable") from exc
    if not name or not version or name == "None" or version == "None":
        raise DriverPinError("wheel METADATA lacks distribution identity")
    return name, version


def _inspect_wheel(wheel: Path, pin: str) -> tuple[str, Mapping[str, str]]:
    try:
        wheel_hash = _hash_file(wheel)
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            if len(members) > _MAX_WHEEL_MEMBERS:
                raise DriverPinError("wheel has too many members")
            if sum(member.file_size for member in members) > _MAX_WHEEL_BYTES:
                raise DriverPinError("wheel payload exceeds the size limit")
            seen: set[str] = set()
            metadata: list[bytes] = []
            files: dict[str, str] = {}
            for member in members:
                parts = _safe_archive_path(member.filename)
                key = "/".join(parts)
                if key in seen:
                    raise DriverPinError("wheel contains duplicate paths")
                seen.add(key)
                mode = (member.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise DriverPinError("wheel contains a non-regular member")
                if member.flag_bits & 1:
                    raise DriverPinError("wheel contains an encrypted member")
                if member.is_dir():
                    continue
                with archive.open(member) as source:
                    if len(parts) == 2 and _DIST_INFO.fullmatch(parts[0]) and parts[1] == "METADATA":
                        metadata.append(source.read())
                        continue
                    installed = _installed_path(parts)
                    if installed is None:
                        continue
                    installed_name = "/".join(installed)
                    if installed_name in files:
                        raise DriverPinError("wheel paths collide after installation")
                    files[installed_name] = _hash_reader(source)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise DriverPinError("wheel artifact is unreadable") from exc
    if len(metadata) != 1:
        raise DriverPinError("wheel must contain exactly one METADATA file")
    name, version = _parse_metadata(metadata[0])
    if _canonical_name(name) != "torchmonarch" or version != pin:
        raise DriverPinError("wheel METADATA does not match the exact pin")
    if not files:
        raise DriverPinError("wheel contains no verifiable package payload")
    names = set(files)
    if any(any(parent.as_posix() in names for parent in PurePosixPath(name).parents) for name in names):
        raise DriverPinError("wheel payload has a file/directory collision")
    return wheel_hash, files


def _scan_tree(site: Path, expected: Mapping[str, str]) -> Mapping[str, str]:
    anchors = {PurePosixPath(name).parts[0] for name in expected}
    found: dict[str, str] = {}
    for anchor in anchors:
        base = site / anchor
        if not base.exists() and not base.is_symlink():
            continue
        pending = [base]
        while pending:
            current = pending.pop()
            relative = current.relative_to(site)
            if current.is_symlink():
                raise DriverPinError("installed payload contains a symlink")
            metadata = current.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                if _DIST_INFO.fullmatch(current.name):
                    continue
                with os.scandir(current) as entries:
                    pending.extend(Path(entry.path) for entry in entries)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise DriverPinError("installed payload contains a non-regular file")
            if current.name.endswith(".pyc"):
                continue
            found[relative.as_posix()] = _hash_file(current)
    for name in foreign_namespace_files(site, expected, found):
        del found[name]
    return found


def _site_metadata_matches(site: Path, pin: str) -> bool:
    matches = 0
    try:
        with os.scandir(site) as entries:
            candidates = [Path(entry.path) for entry in entries if entry.name.lower().endswith(".dist-info")]
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            metadata = candidate / "METADATA"
            if not metadata.is_file() or metadata.is_symlink():
                continue
            name, version = _parse_metadata(metadata.read_bytes())
            if _canonical_name(name) == "torchmonarch":
                matches += version == pin
                if version != pin:
                    return False
    except (DriverPinError, OSError):
        return False
    return matches == 1


class DriverPinTransition:
    """Prepare and execute one exact, recoverable driver pin transition."""

    def __init__(
        self,
        *,
        python_bin: str = sys.executable,
        command_runner: CommandRunner = _default_runner,
        timeout: float = 300,
    ) -> None:
        if not python_bin or "\0" in python_bin:
            raise ValueError("driver interpreter is invalid")
        self._python = python_bin
        self._run = command_runner
        self._timeout = timeout
        self._root: Path | None = None
        self._prior: _Artifact | None = None
        self._target: _Artifact | None = None

    @property
    def prepared(self) -> bool:
        return self._prior is not None and self._target is not None

    @property
    def target_site(self) -> Path | None:
        return None if self._target is None else self._target.site

    @property
    def prior_payload_digest(self) -> str | None:
        return None if self._prior is None else self._prior.manifest_sha256

    def prepare(self, root: Path, prior_pin: str, target_pin: str) -> None:
        if self.prepared or self._root is not None:
            raise DriverPinError("driver pin transition was already prepared")
        prior_pin = _validate_pin(prior_pin)
        target_pin = _validate_pin(target_pin)
        release = self._owned_root(Path(root))
        stage = release / "driver-pin"
        stage.mkdir(mode=0o700)
        self._root = stage
        try:
            prior = self._prepare_artifact(stage / "prior", prior_pin)
            target = self._prepare_artifact(stage / "target", target_pin)
            self._prior, self._target = prior, target
        except BaseException as error:
            self._root = None
            reraise_after_cleanup(error, lambda: self._remove_stage(stage, release))

    def _owned_root(self, root: Path) -> Path:
        root = root.expanduser().absolute()
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise DriverPinError("driver release root is not a real directory")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise DriverPinError("driver release root has the wrong owner")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise DriverPinError("driver release root is not private")
        return root

    def _prepare_artifact(self, root: Path, pin: str) -> _Artifact:
        artifacts, site = root / "artifacts", root / "site"
        artifacts.mkdir(parents=True, mode=0o700)
        site.mkdir(mode=0o700)
        downloaded = self._pip(
            "download", "--no-deps", "--only-binary=:all:", "--dest", str(artifacts),
            f"torchmonarch=={pin}",
        )
        if downloaded.returncode != 0:
            raise DriverPinError("exact torchmonarch wheel download failed")
        entries = list(artifacts.iterdir())
        if len(entries) != 1 or entries[0].suffix.lower() != ".whl" or entries[0].is_symlink():
            raise DriverPinError("wheel download did not produce one regular artifact")
        wheel = entries[0]
        wheel_hash, files = _inspect_wheel(wheel, pin)
        installed = self._pip(
            "install", "--no-index", "--no-deps", "--no-compile", "--target", str(site), str(wheel),
        )
        if installed.returncode != 0:
            raise DriverPinError("offline private wheel installation failed")
        if _scan_tree(site, files) != files or not _site_metadata_matches(site, pin):
            raise DriverPinError("private wheel installation does not match its artifact")
        return _Artifact(pin, wheel, wheel_hash, site, files, _manifest_digest(files))

    def _pip(self, action: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return self._run(
            [self._python, "-m", "pip", "--isolated", "--disable-pip-version-check", action, *arguments],
            capture_output=True, text=True, timeout=self._timeout,
        )

    def matches_prior(self) -> bool:
        return self._matches(self._prior)

    def matches_target(self) -> bool:
        return self._matches(self._target)

    def target_matches_site(self, site: Path) -> bool:
        target = self._target
        try:
            return bool(
                target is not None and self._artifact_ready(target)
                and _scan_tree(site, target.files) == target.files
                and _site_metadata_matches(site, target.pin)
            )
        except Exception:
            return False
    def _matches(self, artifact: _Artifact | None) -> bool:
        try:
            return bool(
                artifact is not None and self._artifact_ready(artifact)
                and self._live_matches(artifact)
            )
        except Exception:
            return False

    def _artifact_ready(self, artifact: _Artifact) -> bool:
        wheel_hash, files = _inspect_wheel(artifact.wheel, artifact.pin)
        return bool(
            wheel_hash == artifact.wheel_sha256 and files == artifact.files
            and _manifest_digest(files) == artifact.manifest_sha256
            and _scan_tree(artifact.site, files) == files
            and _site_metadata_matches(artifact.site, artifact.pin)
        )

    def _live_matches(self, artifact: _Artifact) -> bool:
        environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH")}
        result = self._run(
            [self._python, "-c", _PROBE], capture_output=True, text=True, timeout=60,
            cwd=self._root, env=environment,
        )
        if result.returncode != 0:
            return False
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict) or payload.get("version") != artifact.pin:
            return False
        raw_root = payload.get("root")
        if not isinstance(raw_root, str) or not raw_root or "\0" in raw_root:
            return False
        site = Path(raw_root).resolve(strict=True)
        return bool(
            site.is_dir() and _scan_tree(site, artifact.files) == artifact.files
            and _site_metadata_matches(site, artifact.pin)
        )

    def promote(self) -> OperationResult:
        prior, target = self._prior, self._target
        try:
            if prior is None or target is None or not self._artifact_ready_pair() or not self._live_matches(prior):
                return OperationResult(Certainty.UNKNOWN)
            result = self._install_live(target.wheel)
            target_matches = self._live_matches(target)
            if result.returncode == 0 and target_matches:
                return OperationResult(Certainty.SUCCEEDED)
            if result.returncode != 0 and not target_matches and self._live_matches(prior):
                return OperationResult(Certainty.FAILED)
        except Exception:
            pass
        return OperationResult(Certainty.UNKNOWN)

    def restore_prior(self) -> OperationResult:
        prior, target = self._prior, self._target
        try:
            if prior is None or target is None or not self._artifact_ready_pair():
                return OperationResult(Certainty.UNKNOWN)
            if self._live_matches(prior):
                return OperationResult(Certainty.SUCCEEDED)
            if not self._live_matches(target):
                return OperationResult(Certainty.UNKNOWN)
            self._install_live(prior.wheel)
            if self._live_matches(prior):
                return OperationResult(Certainty.SUCCEEDED)
        except Exception:
            pass
        return OperationResult(Certainty.UNKNOWN)

    def _artifact_ready_pair(self) -> bool:
        return bool(
            self._prior is not None and self._target is not None
            and self._artifact_ready(self._prior) and self._artifact_ready(self._target)
        )

    def _install_live(self, wheel: Path) -> subprocess.CompletedProcess[str]:
        return self._pip("install", "--no-index", "--no-deps", "--no-compile", "--force-reinstall", str(wheel))

    @staticmethod
    def _remove_stage(stage: Path, release: Path) -> None:
        if stage.parent != release or stage.name != "driver-pin":
            return
        try:
            if stage.is_symlink():
                stage.unlink()
            elif stage.is_dir():
                shutil.rmtree(stage)
        except OSError:
            pass


def site_payload_digest(site: Path, pin: str) -> str:
    """Digest pinned payload, excluding verified foreign namespace files."""
    if not _site_metadata_matches(site, _validate_pin(pin)):
        raise DriverPinError("site metadata does not match the exact pin")
    distributions = [
        item for item in importlib.metadata.distributions(path=[str(site)])
        if _canonical_name(item.metadata["Name"]) == "torchmonarch"
    ]
    if len(distributions) != 1 or distributions[0].files is None:
        raise DriverPinError("site distribution inventory is ambiguous")
    anchors: dict[str, str] = {}
    for item in distributions[0].files or ():
        parts = tuple(PurePosixPath(str(item)).parts)
        if not parts or any(part in ("", ".", "..") for part in parts):
            continue
        installed = _installed_path(parts)
        if installed is not None:
            anchors["/".join(installed)] = ""
    if not anchors:
        raise DriverPinError("site distribution has no payload inventory")
    return _manifest_digest(_scan_tree(site, anchors))
