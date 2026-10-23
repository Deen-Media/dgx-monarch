"""dgxm studio works without the studio extra: its modules import and its help runs with aiohttp and Pillow hidden.

The modules are hidden the way a missing install hides them (the "absent"
mode of tests/test_studio_optional_deps.py), in a fresh interpreter, so this
holds whatever an earlier test imported.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULES = ("dgx_monarch.cli.studio_command", "dgx_monarch.studio.launch", "dgx_monarch.studio.sign_in",
           "dgx_monarch.studio.unit", "dgx_monarch.studio.unit_paths", "dgx_monarch.studio.unit_state",
           "dgx_monarch.studio.tunnel", "dgx_monarch.studio.driver_hint")
PROBE = """
import importlib, sys
for name in ("aiohttp", "PIL"):
    sys.modules[name] = None
import dgx_monarch.cli.main
early = sorted(name for name in sys.modules if name.startswith("dgx_monarch.studio"))
print("studio modules imported by dgx_monarch.cli.main:", early)
for name in sys.argv[1:]:
    importlib.import_module(name)
dgx_monarch.cli.main.main(["studio", "--help"])
"""


def _probe(*modules: str) -> subprocess.CompletedProcess[str]:
    path = os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))
    return subprocess.run([sys.executable, "-c", PROBE, *modules], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": path, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120)


def test_studio_modules_import_and_help_runs_without_the_studio_extra():
    result = _probe(*MODULES)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage: dgxm studio" in result.stdout
    for action in ("up", "down", "open", "token", "tunnel"):
        assert action in result.stdout


def test_the_dgxm_entry_point_imports_no_studio_module():
    # cli/main.py and cli/arguments.py reach Studio only inside cmd_studio and cmd_uninstall.
    result = _probe()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "studio modules imported by dgx_monarch.cli.main: []" in result.stdout
