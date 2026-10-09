"""Verify remote Worker source and dependency payloads against the expected release."""
from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import ClusterConfig, HostConfig
from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX

_DIGEST = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}")
_SITE = re.compile(
    r"\$HOME/\.local/share/dgx-monarch/"
    r"(?:src|releases/u-[0-9a-f]{12}-[0-9a-f]{16}/site)"
)
_TOOLS = re.compile(
    r"\$HOME/\.local/share/dgx-monarch/"
    r"releases/u-[0-9a-f]{12}-[0-9a-f]{16}/verifier"
)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("attested tree contains a symlink")
        if path.suffix == ".pyc":
            raise ValueError("attested tree contains sourceless bytecode")
        encoded = relative.as_posix().encode("utf-8")
        if path.is_dir():
            digest.update(b"D\0" + encoded + b"\0")
            continue
        if not path.is_file():
            raise ValueError("attested tree contains a special file")
        digest.update(b"F\0" + encoded + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest() + b"\0")
    return digest.hexdigest()


def verifier_tree_digest(root: Path) -> str:
    """Hash the updater-owned verifier file inventory and contents."""
    return _tree_digest(root)


def site_tree_digest(root: Path) -> str:
    """Hash the complete release site, excluding only ``__pycache__`` bytecode."""
    return _tree_digest(root)


HostRunner = Callable[
    [ClusterConfig, HostConfig, str, int], subprocess.CompletedProcess[str]
]


@dataclass(frozen=True)
class WorkerReleaseIdentity:
    version: str
    torchmonarch_pin: str
    source_manifest: str
    pin_payload_digest: str
    site_manifest: str | None = None

    def validate(self) -> None:
        if _VERSION.fullmatch(self.version) is None:
            raise ValueError("worker version is invalid")
        if _VERSION.fullmatch(self.torchmonarch_pin) is None:
            raise ValueError("worker pin is invalid")
        if _DIGEST.fullmatch(self.source_manifest) is None:
            raise ValueError("worker source manifest is invalid")
        if _DIGEST.fullmatch(self.pin_payload_digest) is None:
            raise ValueError("worker pin payload digest is invalid")
        if self.site_manifest is not None and _DIGEST.fullmatch(self.site_manifest) is None:
            raise ValueError("worker site manifest is invalid")


def _python_shell(python_bin: str) -> str:
    return (
        f"PYBIN={shlex.quote(python_bin)}\n"
        'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; '
        '"~") PYBIN="$HOME";; esac'
    )


def remote_release_matches(
    config: ClusterConfig,
    host: HostConfig,
    *,
    site: str,
    tools: str,
    identity: WorkerReleaseIdentity,
    host_runner: HostRunner,
    verifier_digest: str,
    link_target: str | None = None,
    captured_site: bool = False,
) -> bool:
    """Verify source and the full active pinned payload without importing Worker actors."""
    script = release_attestation_script(
        config, site=site, tools=tools, identity=identity,
        verifier_digest=verifier_digest, link_target=link_target,
        captured_site=captured_site,
    )
    try:
        result = host_runner(config, host, script, 90)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "RELEASE_MATCH" in result.stdout


def release_attestation_script(
    config: ClusterConfig, *, site: str, tools: str,
    identity: WorkerReleaseIdentity, verifier_digest: str,
    link_target: str | None = None, captured_site: bool = False,
) -> str:
    """Return an isolated source-and-pin check for an operation holding its lock."""
    identity.validate()
    if _DIGEST.fullmatch(verifier_digest) is None:
        raise ValueError("remote verifier digest is invalid")
    valid_site = _SITE.fullmatch(site) is not None
    if captured_site:
        valid_site = site.startswith("/") and not any(ord(char) < 32 for char in site)
    if not valid_site or _TOOLS.fullmatch(tools) is None:
        raise ValueError("remote worker attestation path is invalid")
    if link_target is not None and _SITE.fullmatch(link_target) is None:
        raise ValueError("remote live-link target is invalid")
    link_check = (
        f'test -L "$SITE"\ntest "$(readlink -f -- "$SITE")" = "{link_target}"'
        if link_target is not None else ":"
    )
    return f"""
set -eu
SITE={shlex.quote(site) if captured_site else chr(34) + site + chr(34)}
TOOLS="{tools}"
test -d "$SITE/dgx_monarch"
test -d "$TOOLS/dgx_monarch"
{link_check}
{_python_shell(config.python_bin)}
"$PYBIN" -I -B -P - "$SITE" "$TOOLS" {shlex.quote(identity.version)} {shlex.quote(identity.torchmonarch_pin)} {shlex.quote(identity.source_manifest)} {shlex.quote(identity.pin_payload_digest)} {shlex.quote(identity.site_manifest or '-')} {shlex.quote(verifier_digest)} <<'PY'
import ast
import hashlib
import importlib.metadata as md
from importlib.machinery import BYTECODE_SUFFIXES, EXTENSION_SUFFIXES, SOURCE_SUFFIXES
import re
import sys
from pathlib import Path
worker_site, tools, want_version, want_pin, want_source, want_digest, want_site, want_verifier = sys.argv[1:]
def tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts: continue
        if path.is_symlink(): raise SystemExit(4)
        if path.suffix == ".pyc": raise SystemExit(4)
        encoded = relative.as_posix().encode("utf-8")
        if path.is_dir():
            digest.update(b"D\\0" + encoded + b"\\0")
            continue
        if not path.is_file(): raise SystemExit(4)
        digest.update(b"F\\0" + encoded + b"\\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest() + b"\\0")
    return digest.hexdigest()
tools_root = Path(tools).resolve(strict=True)
if tree_digest(tools_root / "dgx_monarch") != want_verifier: raise SystemExit(4)
sys.dont_write_bytecode = True
sys.pycache_prefix = {SOURCE_ONLY_PYCACHE_PREFIX!r}
if any(name == "dgx_monarch" or name.startswith("dgx_monarch.") for name in sys.modules): raise SystemExit(4)
sys.path[:] = [str(tools_root), *(item for item in sys.path if item not in ("", tools, str(tools_root)))]
import dgx_monarch as verifier_package
from dgx_monarch.cli import update_driver_pin as pin_verifier
from dgx_monarch import runtime_provenance as provenance
def verify_module_origins():
    for name, module in tuple(sys.modules.items()):
        if name != "dgx_monarch" and not name.startswith("dgx_monarch."): continue
        try: Path(module.__file__).resolve(strict=True).relative_to(tools_root)
        except (AttributeError, OSError, TypeError, ValueError): raise SystemExit(4)
verify_module_origins()
tree = ast.parse((Path(worker_site) / "dgx_monarch" / "__init__.py").read_text("utf-8"))
versions = [node.value.value for node in tree.body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__version__" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
if len(versions) != 1: raise SystemExit(2)
source = provenance.dgx_source_manifest_sha256(Path(worker_site) / "dgx_monarch")
site_digest = tree_digest(Path(worker_site))
sys.path[:] = [worker_site, *(item for item in sys.path if item not in ("", worker_site, tools))]
distribution = md.distribution("torchmonarch")
root = Path(distribution.locate_file("")).resolve(strict=True)
anchors = set()
import_suffixes = sorted((*SOURCE_SUFFIXES, *EXTENSION_SUFFIXES, *BYTECODE_SUFFIXES), key=len, reverse=True)
for item in distribution.files or ():
    raw = str(item)
    if ".." in raw.split("/"):
        if re.fullmatch(r"(?:[.][.]/)+bin/[^/\\\\:]+", raw): continue
        raise SystemExit(3)
    try: parts = pin_verifier._installed_path(pin_verifier._safe_archive_path(raw))
    except pin_verifier.DriverPinError: raise SystemExit(3)
    if parts is None: continue
    if len(parts) > 1:
        anchors.add(parts[0])
    else:
        for suffix in import_suffixes:
            if parts[0].endswith(suffix):
                anchors.add(parts[0][:-len(suffix)])
                break
if not anchors: raise SystemExit(3)
for entry in sys.path:
    try: candidate = Path(entry).resolve(strict=True)
    except OSError: continue
    if candidate == root: break
    for anchor in anchors:
        names = (anchor, *(anchor + suffix for suffix in import_suffixes))
        if any((candidate / name).exists() or (candidate / name).is_symlink() for name in names): raise SystemExit(3)
actual = (versions[0], distribution.version, source, pin_verifier.site_payload_digest(root, want_pin))
verify_module_origins()
if tree_digest(tools_root / "dgx_monarch") != want_verifier: raise SystemExit(4)
site_matches = want_site == "-" or site_digest == want_site
raise SystemExit(0 if actual == (want_version, want_pin, want_source, want_digest) and site_matches else 1)
PY
{link_check}
echo RELEASE_MATCH
"""
