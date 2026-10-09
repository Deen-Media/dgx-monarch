"""Safe launcher and optional desktop-entry contracts.

These tests use a fake interpreter or replace only the probe's network calls.
They do not start ComfyUI, a browser, or any dgx-monarch Worker service.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/comfy-driver.sh"
DESKTOP = ROOT / "scripts/dgxm-desktop.sh"
ICON = ROOT / "docs/media/dgx-monarch.svg"


def _checkout(path: Path, *, link_repository: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "comfy").mkdir()
    (path / "custom_nodes").mkdir()
    if link_repository:
        (path / "custom_nodes/dgx-monarch").symlink_to(ROOT, target_is_directory=True)
    (path / "main.py").write_text("raise SystemExit('launcher must not run this fixture')\n")
    (path / "comfy/cli_args.py").write_text("# checkout marker\n")
    return path


def _fake_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#!/usr/bin/env bash
set -eu
if [ "${1:-}" = "-" ]; then
  BODY=$(cat)
  if [[ "$BODY" == *"webbrowser.open"* ]]; then
    printf 'browser:%s\\n' "${2:-}" >> "$DGXM_RECORD"
    exit "${DGXM_FAKE_BROWSER_STATUS:-0}"
  fi
  printf 'probe\\n' >> "$DGXM_RECORD"
  case "${DGXM_FAKE_PROBE_STATUS:-10}" in
    0) exit 0 ;;
    10) exit 10 ;;
    *) exit 11 ;;
  esac
fi
printf 'cwd:%s\\n' "$PWD" >> "$DGXM_RECORD"
for ARG in "$@"; do
  printf 'arg:%s\\n' "$ARG" >> "$DGXM_RECORD"
done
exit "${DGXM_FAKE_MAIN_STATUS:-0}"
"""
    )
    path.chmod(0o755)
    return path


def _run_launcher(
    comfy: Path,
    python: Path,
    record: Path,
    *args: str,
    probe: int = 10,
    main_status: int = 0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "DGXM_RECORD": str(record),
            "DGXM_FAKE_PROBE_STATUS": str(probe),
            "DGXM_FAKE_MAIN_STATUS": str(main_status),
        }
    )
    return subprocess.run(
        [
            str(LAUNCHER),
            "--comfy-dir",
            str(comfy),
            "--python",
            str(python),
            *args,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_accepts_repository_symlink_beneath_custom_nodes(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    python = _fake_python(tmp_path / "venv/bin/python")
    record = tmp_path / "calls"

    result = _run_launcher(comfy, python, record, "--check")

    assert result.returncode == 0, result.stderr
    assert "launcher check: ready" in result.stdout
    repository = comfy / "custom_nodes/dgx-monarch"
    assert repository.is_symlink()
    assert repository.resolve() == ROOT.resolve()
    assert not record.exists()


def test_launcher_accepts_direct_repository_beneath_custom_nodes(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI", link_repository=False)
    repository = comfy / "custom_nodes/dgx-monarch"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "comfy-driver.sh"
    shutil.copy2(LAUNCHER, launcher)
    python = _fake_python(tmp_path / "venv/bin/python")
    record = tmp_path / "calls"

    result = subprocess.run(
        [
            str(launcher),
            "--comfy-dir",
            str(comfy),
            "--python",
            str(python),
            "--check",
        ],
        env={**os.environ, "DGXM_RECORD": str(record)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "launcher check: ready" in result.stdout
    assert repository.is_dir()
    assert not repository.is_symlink()
    assert not record.exists()


def test_launcher_refuses_checkout_without_this_repository_install(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI", link_repository=False)
    python = _fake_python(tmp_path / "venv/bin/python")
    record = tmp_path / "calls"

    result = _run_launcher(comfy, python, record, "--check")

    assert result.returncode == 2
    assert "not discoverable" in result.stderr
    assert "clone it there or add a direct symlink" in result.stderr
    assert str(ROOT.resolve()) in result.stderr
    assert not record.exists()


def test_new_driver_execs_selected_python_and_propagates_its_exit(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "Comfy UI")
    python = _fake_python(tmp_path / "venv with spaces/bin/python")
    record = tmp_path / "calls"

    result = _run_launcher(
        comfy,
        python,
        record,
        "--no-browser",
        "--",
        "--preview-method",
        "auto",
        "--output-directory",
        str(tmp_path / "Comfy Output"),
        main_status=7,
    )

    assert result.returncode == 7
    calls = record.read_text().splitlines()
    assert calls[0] == "probe"
    assert f"cwd:{comfy}" in calls
    assert "arg:main.py" in calls
    assert "arg:--disable-auto-launch" in calls
    assert "arg:--auto-launch" not in calls
    assert f"arg:{tmp_path / 'Comfy Output'}" in calls
    assert "Worker services are not changed" in result.stdout


def test_new_driver_delegates_browser_open_to_comfy_readiness(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    python = _fake_python(tmp_path / "venv/bin/python")
    record = tmp_path / "calls"

    result = _run_launcher(comfy, python, record)

    assert result.returncode == 0
    calls = record.read_text().splitlines()
    assert "arg:--auto-launch" in calls
    assert "arg:--disable-auto-launch" not in calls
    assert not any(line.startswith("browser:") for line in calls)


def test_confirmed_existing_driver_is_reused_without_starting_another(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    python = _fake_python(tmp_path / "venv/bin/python")
    record = tmp_path / "calls"

    result = _run_launcher(comfy, python, record, "--no-browser", probe=0)

    assert result.returncode == 0
    assert record.read_text().splitlines() == ["probe"]
    assert "not starting a duplicate" in result.stdout


def test_confirmed_existing_driver_uses_explicit_browser_helper(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    python = _fake_python(tmp_path / "venv/bin/python")
    record = tmp_path / "calls"

    result = _run_launcher(comfy, python, record, probe=0)

    assert result.returncode == 0
    assert record.read_text().splitlines() == ["probe", "browser:http://127.0.0.1:8188"]


def test_unconfirmed_listener_is_refused_without_starting_driver(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    python = _fake_python(tmp_path / "venv/bin/python")
    record = tmp_path / "calls"

    result = _run_launcher(comfy, python, record, probe=11)

    assert result.returncode == 2
    assert record.read_text().splitlines() == ["probe"]
    assert "not a confirmed DGX Monarch driver" in result.stderr


def _probe_environment(tmp_path: Path, payload: dict[str, object]) -> dict[str, str]:
    """Replace only the probe interpreter's network calls through sitecustomize."""
    site = tmp_path / "probe-site"
    site.mkdir(parents=True)
    (site / "sitecustomize.py").write_text(
        """import io
import os
import socket
import urllib.request

class _Context:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value if self.value is not None else self

    def __exit__(self, *_args):
        return False


class _Response(io.BytesIO):
    def __init__(self, value, url):
        super().__init__(value)
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return os.environ.get("DGXM_TEST_FINAL_URL", self.url)


class _Opener:
    def open(self, url, **_kwargs):
        return _Response(os.environ["DGXM_TEST_SCHEMA"].encode(), url)


def _build_opener(*handlers):
    assert any(
        isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {}
        for handler in handlers
    )
    assert any(type(handler).__name__ == "_NoRedirect" for handler in handlers)
    return _Opener()

socket.create_connection = lambda *_args, **_kwargs: _Context()
urllib.request.build_opener = _build_opener
"""
    )
    return {
        **os.environ,
        "PYTHONPATH": str(site),
        "DGXM_TEST_SCHEMA": json.dumps(payload),
    }


def test_live_probe_requires_the_exact_dgx_monarch_init_schema(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    valid = {
        "DGXMonarchInit": {
            "name": "DGXMonarchInit",
            "category": "DGX Monarch",
            "output": ["DGXM_MESH"],
        }
    }
    result = subprocess.run(
        [
            str(LAUNCHER),
            "--comfy-dir",
            str(comfy),
            "--python",
            sys.executable,
            "--port",
            "49119",
            "--no-browser",
        ],
        env=_probe_environment(tmp_path, valid),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "not starting a duplicate" in result.stdout

    refused_env = _probe_environment(tmp_path / "second", {"SomeOtherNode": valid["DGXMonarchInit"]})
    refused = subprocess.run(
        [
            str(LAUNCHER),
            "--comfy-dir",
            str(comfy),
            "--python",
            sys.executable,
            "--port",
            "49120",
            "--no-browser",
        ],
        env=refused_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 2
    assert "not a confirmed DGX Monarch driver" in refused.stderr


def test_live_probe_rejects_a_response_from_another_endpoint(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    valid = {
        "DGXMonarchInit": {
            "name": "DGXMonarchInit",
            "category": "DGX Monarch",
            "output": ["DGXM_MESH"],
        }
    }
    env = _probe_environment(tmp_path, valid)
    env["DGXM_TEST_FINAL_URL"] = "http://127.0.0.1:49122/redirected"

    result = subprocess.run(
        [
            str(LAUNCHER),
            "--comfy-dir",
            str(comfy),
            "--python",
            sys.executable,
            "--port",
            "49121",
            "--no-browser",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "not a confirmed DGX Monarch driver" in result.stderr


def test_launcher_requires_absolute_validated_paths_and_owns_network_flags(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    python = _fake_python(tmp_path / "venv/bin/python")
    record = tmp_path / "calls"
    env = {**os.environ, "DGXM_RECORD": str(record)}

    relative = subprocess.run(
        [str(LAUNCHER), "--comfy-dir", "ComfyUI", "--python", str(python), "--check"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert relative.returncode == 2
    assert "absolute path" in relative.stderr
    assert not record.exists()

    conflict = _run_launcher(comfy, python, record, "--", "--port", "9000")
    assert conflict.returncode == 2
    assert "managed by this launcher" in conflict.stderr
    assert not record.exists()

    both = subprocess.run(
        [
            str(LAUNCHER),
            "--comfy-dir",
            str(comfy),
            "--venv",
            str(python.parent.parent),
            "--python",
            str(python),
            "--check",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert both.returncode == 2
    assert "either --venv or --python" in both.stderr


def test_desktop_installer_writes_owned_terminal_entry_and_uninstalls(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "Comfy $UI 100%")
    python = _fake_python(tmp_path / "venv with spaces/bin/python")
    xdg = tmp_path / "xdg data"
    env = {**os.environ, "XDG_DATA_HOME": str(xdg), "HOME": str(tmp_path / "home")}

    installed = subprocess.run(
        [
            str(DESKTOP),
            "install",
            "--comfy-dir",
            str(comfy),
            "--python",
            str(python),
            "--no-browser",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr
    entry = xdg / "applications/dgx-monarch.desktop"
    source = entry.read_text()
    assert "Terminal=true" in source
    assert "X-DGX-Monarch-Managed=true" in source
    assert f'Exec="{LAUNCHER}"' in source
    assert "--no-browser" in source
    assert "PREP" not in source
    assert "sudo" not in source
    assert "dgxm up" not in source
    assert str(ICON.resolve()) in source
    escaped_comfy = str(comfy).replace("$", r"\$").replace("%", "%%")
    assert escaped_comfy in source

    validator = shutil.which("desktop-file-validate")
    if validator:
        checked = subprocess.run([validator, str(entry)], text=True, capture_output=True, check=False)
        assert checked.returncode == 0, checked.stderr

    removed = subprocess.run(
        [str(DESKTOP), "uninstall"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert removed.returncode == 0
    assert not entry.exists()


def test_desktop_installer_refuses_an_unowned_entry(tmp_path: Path) -> None:
    comfy = _checkout(tmp_path / "ComfyUI")
    python = _fake_python(tmp_path / "venv/bin/python")
    xdg = tmp_path / "xdg"
    entry = xdg / "applications/dgx-monarch.desktop"
    entry.parent.mkdir(parents=True)
    entry.write_text("[Desktop Entry]\nType=Application\nName=Someone Else\n")
    env = {**os.environ, "XDG_DATA_HOME": str(xdg), "HOME": str(tmp_path / "home")}

    result = subprocess.run(
        [str(DESKTOP), "install", "--comfy-dir", str(comfy), "--python", str(python)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unmanaged desktop entry" in result.stderr
    assert "Someone Else" in entry.read_text()


def test_launcher_assets_have_no_hidden_lifecycle_or_unsafe_svg_surface() -> None:
    launcher = LAUNCHER.read_text()
    desktop = DESKTOP.read_text()
    prose = (ROOT / "scripts/README.md").read_text()
    icon = ICON.read_text()

    assert 'exec "$PYTHON_BIN" main.py' in launcher
    assert "--auto-launch" in launcher
    for forbidden in ("nohup", "systemctl", "dgxm up", "dgxm down", "dgxm restart", "sudo"):
        assert forbidden not in launcher
    assert "Terminal=true" in desktop
    assert "/home/" not in launcher + desktop + prose + icon
    assert "\u2014" not in launcher + desktop + prose + icon
    assert "\u2013" not in launcher + desktop + prose + icon

    assert '<svg xmlns="http://www.w3.org/2000/svg"' in icon
    assert 'viewBox="0 0 256 256"' in icon
    assert "<script" not in icon.lower()
    assert "href=" not in icon.lower()
