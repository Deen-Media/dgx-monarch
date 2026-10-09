"""Read a selected release without importing the controller's working directory."""
from __future__ import annotations

import sys
from pathlib import Path

from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX
from .update_release import CommandRunner

_PROBE = f'''
import importlib.metadata as metadata
import sys
from pathlib import Path
site = Path(sys.argv[1]).resolve(strict=True)
package = site / "dgx_monarch"
if package.is_symlink() or not package.is_dir(): raise SystemExit(2)
sys.dont_write_bytecode = True
sys.pycache_prefix = {SOURCE_ONLY_PYCACHE_PREFIX!r}
if any(name == "dgx_monarch" or name.startswith("dgx_monarch.") for name in sys.modules): raise SystemExit(2)
sys.path.insert(0, str(site))
from dgx_monarch import __version__
from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256
actual = (__version__, metadata.version("torchmonarch"), dgx_source_manifest_sha256())
for name, module in tuple(sys.modules.items()):
    if name != "dgx_monarch" and not name.startswith("dgx_monarch."): continue
    try: Path(module.__file__).resolve(strict=True).relative_to(package)
    except (AttributeError, OSError, TypeError, ValueError): raise SystemExit(2)
raise SystemExit(0 if actual == tuple(sys.argv[2:]) else 1)
'''


def site_matches(site: Path, version: str, pin: str, source_manifest: str, run: CommandRunner) -> bool:
    result = run(
        [sys.executable, "-I", "-B", "-c", _PROBE, str(site.resolve(strict=True)), version, pin, source_manifest],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0
