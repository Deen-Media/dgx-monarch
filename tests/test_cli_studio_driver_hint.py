"""The first-run start command: its precedence, both launcher reads, exact quoting, and no process started.

aiohttp-free: driver_hint runs in the companion before its event loop, and
the command it renders is only text the page shows.
"""
from __future__ import annotations

import ast
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from dgx_monarch.studio import driver_hint

REPO = Path(__file__).resolve().parents[1]
ORIGIN = "http://127.0.0.1:8191"
HOSTS = '[[hosts]]\nname = "worker"\naddress = "tcp://192.0.2.12:26600"\n'


def private_config(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o600)
    return path


def comfy_tree(root: Path) -> tuple[Path, Path]:
    """A ComfyUI checkout with the node pack at its documented place and a copy of the real launcher."""
    comfy = root / "ComfyUI"
    (comfy / "comfy").mkdir(parents=True)
    (comfy / "main.py").write_text("")
    (comfy / "comfy" / "cli_args.py").write_text("")
    pack = comfy / "custom_nodes" / "dgx-monarch"
    (pack / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "comfy-driver.sh", pack / "scripts" / "comfy-driver.sh")
    (pack / "src" / "dgx_monarch").mkdir(parents=True)
    (pack / "src" / "dgx_monarch" / "__init__.py").write_text("")
    return comfy, pack


@pytest.fixture
def python(tmp_path, monkeypatch):
    """The companion's interpreter, faked; the package path points away from any checkout."""
    fake = tmp_path / "env" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setattr(driver_hint.sys, "executable", str(fake))
    monkeypatch.setattr(driver_hint, "_package_file", lambda: tmp_path / "site" / "dgx_monarch" / "__init__.py")
    return fake


def words(command: str | None) -> list[str]:
    assert command is not None
    return shlex.split(command)


def test_the_driver_service_comes_first(tmp_path, python):
    comfy, _pack = comfy_tree(tmp_path)
    config = private_config(tmp_path / "cluster.toml", f'[cluster]\ncomfy_dir = "{comfy}"\n\n'
                            '[studio]\ndriver_command = "bash /home/user/start-comfyui.sh"\n\n' + HOSTS)
    assert driver_hint.driver_start(ORIGIN, "comfyui-driver.service", config) == {
        "command": "systemctl --user start comfyui-driver.service", "source": "driver_service", "config": str(config)}
    assert driver_hint.driver_start(ORIGIN, "comfyui-driver.service", None) == {
        "command": "systemctl --user start comfyui-driver.service", "source": "driver_service", "config": None}


def test_the_driver_command_comes_before_the_launcher_verbatim(tmp_path, python):
    comfy, _pack = comfy_tree(tmp_path)
    text = "bash /home/user/start-comfyui.sh --no-browser"
    config = private_config(tmp_path / "cluster.toml",
                            f'[cluster]\ncomfy_dir = "{comfy}"\n\n[studio]\ndriver_command = "{text}"\n\n' + HOSTS)
    assert driver_hint.driver_start(ORIGIN, None, config) == {
        "command": text, "source": "driver_command", "config": str(config)}


def test_a_config_without_hosts_still_gives_its_driver_command(tmp_path, python):
    config = private_config(tmp_path / "cluster.toml", '[studio]\ndriver_command = "bash /home/user/start-comfyui.sh"\n')
    assert driver_hint.driver_start(ORIGIN, None, config)["command"] == "bash /home/user/start-comfyui.sh"


def test_the_install_layout_names_the_launcher(tmp_path, python, monkeypatch):
    comfy, pack = comfy_tree(tmp_path)
    monkeypatch.setattr(driver_hint, "_package_file", lambda: pack / "src" / "dgx_monarch" / "__init__.py")
    result = driver_hint.driver_start(ORIGIN, None, None)
    assert result == {"command": result["command"], "source": "launcher", "config": None}
    assert words(result["command"]) == [
        "bash", str(pack / "scripts" / "comfy-driver.sh"), "--comfy-dir", str(comfy), "--python", str(python),
        "--listen", "127.0.0.1", "--port", "8191", "--no-browser"]


@pytest.mark.parametrize("missing", ["main.py", "scripts/comfy-driver.sh"])
def test_the_install_layout_needs_both_files(tmp_path, python, monkeypatch, missing):
    comfy, pack = comfy_tree(tmp_path)
    (comfy / missing if missing == "main.py" else pack / missing).unlink()
    monkeypatch.setattr(driver_hint, "_package_file", lambda: pack / "src" / "dgx_monarch" / "__init__.py")
    assert driver_hint.driver_start(ORIGIN, None, None) == {"command": None, "source": None, "config": None}


def test_the_configured_comfy_dir_names_the_launcher_and_the_config(tmp_path, python):
    comfy, pack = comfy_tree(tmp_path)
    config = private_config(tmp_path / "cluster.toml", f'[cluster]\ncomfy_dir = "{comfy}"\n\n' + HOSTS)
    result = driver_hint.driver_start(ORIGIN, None, config)
    assert (result["source"], result["config"]) == ("launcher", str(config))
    assert words(result["command"]) == [
        "bash", str(pack / "scripts" / "comfy-driver.sh"), "--comfy-dir", str(comfy), "--python", str(python),
        "--listen", "127.0.0.1", "--port", "8191", "--no-browser", "--cluster-config", str(config)]


def test_the_python_is_the_companions_own_never_the_worker_word(tmp_path, python):
    comfy, _pack = comfy_tree(tmp_path)
    config = private_config(tmp_path / "cluster.toml",
                            f'[cluster]\ncomfy_dir = "{comfy}"\npython = "python3"\n\n' + HOSTS)
    argv = words(driver_hint.driver_start(ORIGIN, None, config)["command"])
    assert argv[argv.index("--python") + 1] == str(python)
    assert "python3" not in argv


def test_the_rendered_launcher_command_passes_the_launcher_check(tmp_path, python):
    # A space in every path proves the quoting survives a real shell.
    comfy, _pack = comfy_tree(tmp_path / "with space")
    config = private_config(tmp_path / "with space" / "cluster.toml", f'[cluster]\ncomfy_dir = "{comfy}"\n\n' + HOSTS)
    command = driver_hint.driver_start(ORIGIN, None, config)["command"]
    assert command is not None and "'" in command
    result = subprocess.run(["bash", "-c", command + " --check"], capture_output=True, text=True, timeout=60,
                            check=False)
    assert result.returncode == 0, result.stderr
    assert "launcher check: ready" in result.stdout
    assert "driver: http://127.0.0.1:8191" in result.stdout


def test_home_is_expanded_before_quoting(tmp_path, python, monkeypatch):
    home = tmp_path / "home"
    comfy, pack = comfy_tree(home / "Comfy UI")
    monkeypatch.setenv("HOME", str(home))
    config = private_config(tmp_path / "cluster.toml", '[cluster]\ncomfy_dir = "~/Comfy UI/ComfyUI"\n\n' + HOSTS)
    command = driver_hint.driver_start(ORIGIN, None, config)["command"]
    assert command is not None and "~" not in command
    assert shlex.quote(str(comfy)) in command and shlex.quote(str(pack / "scripts" / "comfy-driver.sh")) in command


@pytest.mark.parametrize(("origin", "listen", "port"), [
    ("http://localhost:8191", "127.0.0.1", "8191"),
    ("http://[::1]:8191", "::1", "8191"),
    ("http://127.0.0.1", "127.0.0.1", "80"),
])
def test_listen_and_port_follow_the_origin(tmp_path, python, monkeypatch, origin, listen, port):
    _comfy, pack = comfy_tree(tmp_path)
    monkeypatch.setattr(driver_hint, "_package_file", lambda: pack / "src" / "dgx_monarch" / "__init__.py")
    argv = words(driver_hint.driver_start(origin, None, None)["command"])
    assert argv[argv.index("--listen") + 1] == listen and argv[argv.index("--port") + 1] == port


@pytest.mark.parametrize("origin", [
    "http://192.0.2.10:8191", "https://127.0.0.1:8191", "http://127.0.0.2:8191", "http://studio-host:8191",
    "http://127.0.0.1:99999",
])
def test_no_launcher_command_for_another_origin(tmp_path, python, monkeypatch, origin):
    _comfy, pack = comfy_tree(tmp_path)
    monkeypatch.setattr(driver_hint, "_package_file", lambda: pack / "src" / "dgx_monarch" / "__init__.py")
    assert driver_hint.driver_start(origin, None, None) == {"command": None, "source": None, "config": None}


def test_no_launcher_command_without_the_script(tmp_path, python):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").write_text("")
    config = private_config(tmp_path / "cluster.toml", f'[cluster]\ncomfy_dir = "{comfy}"\n\n' + HOSTS)
    assert driver_hint.driver_start(ORIGIN, None, config) == {"command": None, "source": None, "config": str(config)}


@pytest.mark.parametrize("fault", ["missing", "unreadable", "not_private", "invalid_toml", "invalid_table",
                                   "invalid_hosts"])
def test_a_config_studio_cannot_use_gives_no_command(tmp_path, python, monkeypatch, fault):
    if fault == "unreadable" and os.geteuid() == 0:
        pytest.skip("root reads a mode 000 file")
    # A launcher exists both ways, so only the refused config can take the command away.
    comfy, pack = comfy_tree(tmp_path)
    monkeypatch.setattr(driver_hint, "_package_file", lambda: pack / "src" / "dgx_monarch" / "__init__.py")
    config = tmp_path / "cluster.toml"
    good = f'[cluster]\ncomfy_dir = "{comfy}"\n\n[studio]\ndriver_command = "bash /home/user/start-comfyui.sh"\n\n' + HOSTS
    if fault != "missing":
        private_config(config, {"invalid_toml": "[studio\n",
                                "invalid_table": '[studio]\ndriver_command = ""\n\n' + HOSTS,
                                "invalid_hosts": '[studio]\ndriver_command = "bash /home/user/start-comfyui.sh"\n\n'
                                                 '[[hosts]]\nname = "worker"\n'}.get(fault, good))
    if fault == "unreadable":
        config.chmod(0o000)
    if fault == "not_private":
        config.chmod(0o666)
    assert driver_hint.driver_start(ORIGIN, None, config) == {"command": None, "source": None, "config": str(config)}


def test_the_config_field_is_the_absolute_path(tmp_path, python, monkeypatch):
    monkeypatch.chdir(tmp_path)
    private_config(tmp_path / "cluster.toml", '[studio]\ndriver_command = "bash /home/user/start-comfyui.sh"\n')
    assert driver_hint.driver_start(ORIGIN, None, Path("cluster.toml"))["config"] == str(tmp_path / "cluster.toml")


_PROCESS_MODULES = frozenset({"subprocess", "asyncio", "multiprocessing", "pty", "pexpect"})
_PROCESS_CALLS = re.compile(r"^(system|popen|Popen|run|call|check_call|check_output|fork\w*|exec\w*|spawn\w*"
                            r"|posix_spawn\w*|startfile|create_subprocess_\w*)$")


def process_uses(source: str) -> list[str]:
    """Every import of a process module and every call spelled like a process start."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.partition(".")[0] in _PROCESS_MODULES]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.partition(".")[0] in _PROCESS_MODULES or (module == "os" and any(
                    _PROCESS_CALLS.match(alias.name) for alias in node.names)):
                found.append(module)
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            if _PROCESS_CALLS.match(name):
                found.append(name)
    return found


def test_driver_hint_starts_no_process():
    # The scan itself catches each form it guards against.
    for planted in ("import subprocess", "from asyncio import create_subprocess_exec", "from os import system",
                    "os.system('x')", "os.execv('x', [])", "asyncio.create_subprocess_shell('x')",
                    "subprocess.Popen(['x'])", "os.posix_spawn('x', [], {})"):
        assert process_uses(planted), planted
    assert process_uses(Path(driver_hint.__file__).read_text()) == []
