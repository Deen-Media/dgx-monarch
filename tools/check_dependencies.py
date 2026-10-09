"""Check dependencies, verifying the one known cuSPARSELt wheel tag defect."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from email.parser import Parser
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath
from typing import IO

PACKAGE = "nvidia-cusparselt-cu13"
KNOWN_FAILURE = f"{PACKAGE} 0.8.1 is not supported on this platform"
WHEEL_SHA256 = "4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f"
WHEEL_URL = (
    "https://pypi.nvidia.com/nvidia-cusparselt-cu13/"
    "nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl"
)
ISSUE_URL = "https://github.com/Deen-Media/dgx-monarch/issues/5"
DIST_INFO = "nvidia_cusparselt_cu13-0.8.1.dist-info"
CHUNK_SIZE = 1024 * 1024
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024


def download_wheel(destination: Path) -> None:
    """Fetch the pinned artifact without modifying the environment."""
    deadline = time.monotonic() + 300
    received = 0
    with urllib.request.urlopen(WHEEL_URL, timeout=30) as response, destination.open("wb") as target:
        if not response.geturl().startswith("https://"):
            raise ValueError("Wheel download redirected away from HTTPS")
        while chunk := response.read(CHUNK_SIZE):
            received += len(chunk)
            if received > MAX_DOWNLOAD_BYTES or time.monotonic() > deadline:
                raise ValueError("Wheel download exceeded its size or time limit")
            target.write(chunk)


def stream_digest(stream: IO[bytes]) -> bytes:
    digest = hashlib.sha256()
    while chunk := stream.read(CHUNK_SIZE):
        digest.update(chunk)
    return digest.digest()


def verify_known_wheel() -> None:
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise ValueError("The exception applies only to Linux aarch64")
    dist = distribution(PACKAGE)
    if dist.version != "0.8.1":
        raise ValueError("Installed cuSPARSELt version differs from 0.8.1")
    wheel_metadata = dist.read_text("WHEEL")
    if wheel_metadata is None or Parser().parsestr(wheel_metadata).get_all("Tag") != [
        "py3-none-manylinux2014_sbsa"
    ]:
        raise ValueError("Installed WHEEL tag does not match the known defect")
    with tempfile.TemporaryDirectory(prefix="dgxm-dependency-check-") as directory:
        wheel = Path(directory) / "official.whl"
        download_wheel(wheel)
        with wheel.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != WHEEL_SHA256:
                raise ValueError("Official wheel hash mismatch")
        with zipfile.ZipFile(wheel) as archive:
            seen: set[str] = set()
            for member in archive.infolist():
                name = member.filename
                path = PurePosixPath(name)
                if (
                    name in seen or path.is_absolute() or ".." in path.parts
                    or "\\" in name or not path.parts
                    or path.as_posix() != name.rstrip("/")
                ):
                    raise ValueError(f"Unsafe or duplicate wheel member: {name}")
                seen.add(name)
                if member.is_dir() or name == f"{DIST_INFO}/RECORD":
                    continue
                installed_path = Path(str(dist.locate_file(name)))
                if not installed_path.is_file() or installed_path.is_symlink():
                    raise ValueError(f"Installed file missing or not regular: {name}")
                with archive.open(member) as original, installed_path.open("rb") as installed:
                    if stream_digest(original) != stream_digest(installed):
                        raise ValueError(f"Installed file differs: {name}")
            if not {f"{DIST_INFO}/WHEEL", f"{DIST_INFO}/METADATA", f"{DIST_INFO}/RECORD"} <= seen:
                raise ValueError("Official wheel is missing required metadata")


def main() -> int:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        print(f"Dependency check failed: {exc}", file=sys.stderr)
        return 1
    if result.returncode == 0 and not result.stderr and result.stdout in (
        "", "No broken requirements found.", "No broken requirements found.\n",
    ):
        return 0
    if result.returncode == 1 and not result.stderr and result.stdout in (KNOWN_FAILURE, KNOWN_FAILURE + "\n"):
        try:
            verify_known_wheel()
        except Exception as exc:
            reason = f"Could not verify the known wheel defect: {exc}"
        else:
            print(f"Known upstream platform-tag failure: {PACKAGE} 0.8.1; verified sha256 {WHEEL_SHA256}; {ISSUE_URL}")
            return 0
    else:
        reason = f"Unexpected pip check output or exit status ({result.returncode})"
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    print(f"\nDependency check failed: {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
