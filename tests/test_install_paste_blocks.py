"""INSTALL.md paste blocks that change the machine must stop where a step refused."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

INSTALL = Path(__file__).resolve().parents[1] / "docs" / "INSTALL.md"


def _section_blocks(heading: str) -> list[str]:
    text = INSTALL.read_text(encoding="utf-8")
    start = text.index(f"\n{heading}\n")
    end = text.find("\n## ", start + 1)
    section = text[start:end if end >= 0 else None]
    return [
        "\n".join(line.removeprefix("   ") for line in body.splitlines())
        for body in re.findall(r"^ *```bash\n(.*?)^ *```$", section, flags=re.MULTILINE | re.DOTALL)
    ]


def _commands(block: str) -> list[str]:
    return [line for line in block.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_the_uninstall_command_stands_alone_in_its_block():
    """Its config prompt would read a pasted next line as the answer."""
    first, _second = _section_blocks("## Uninstalling")
    assert _commands(first) == ["dgxm uninstall"]


def _uninstall_says(monkeypatch, capsys, *, config, studio: str = "absent") -> tuple[int, str]:
    """What ``dgxm uninstall`` prints, with every host effect faked to succeed."""
    from types import SimpleNamespace

    from dgx_monarch.cli import main as cli
    from dgx_monarch.studio import unit as studio_unit

    monkeypatch.setattr(studio_unit, "owned_unit_state", lambda: studio)
    monkeypatch.setattr(cli, "_load_config", lambda _args: config)
    monkeypatch.setattr(cli.lifecycle, "units_are_removable", lambda _config: True)
    monkeypatch.setattr(cli.lifecycle, "down", lambda _config: True)
    monkeypatch.setattr(cli.lifecycle, "uninstall_systemd", lambda _config: True)
    status = cli.cmd_uninstall(SimpleNamespace())
    captured = capsys.readouterr()
    return status, captured.out + captured.err


def test_the_uninstall_step_quotes_only_what_the_command_prints(monkeypatch, capsys):
    """The step goes by the exit status, and each phrase it quotes is real output."""
    from dgx_monarch.config import ClusterConfig, HostConfig

    text = INSTALL.read_text(encoding="utf-8")
    start = text.index("\n## Uninstalling\n")
    step = re.sub(r"```.*?```", "", text[start:text.index("\n2. ", start)], flags=re.DOTALL)
    assert "`echo $?`" in step and "`0` means" in step
    removed = _uninstall_says(
        monkeypatch, capsys, config=ClusterConfig(hosts=(HostConfig("one", "tcp://127.0.0.1:26600"),)))
    nothing = _uninstall_says(monkeypatch, capsys, config=None)
    studio = _uninstall_says(monkeypatch, capsys, config=None, studio="ours")
    assert removed[0] == 0 and nothing[0] == 0 and studio[0] != 0
    assert "worker services and units removed" in removed[1] and "remove the" in removed[1].splitlines()[-1]
    printed = removed[1] + nothing[1] + studio[1]
    quoted = [phrase for phrase in re.findall(r"`([^`]+)`", step) if phrase not in {"echo $?", "0"}]
    assert {"worker services and units removed", "no cluster config was found"} <= set(quoted)
    assert [phrase for phrase in quoted if phrase not in printed] == []


def _run_removal(tmp_path: Path, *, units: tuple[str, ...] = (), comfy_dir: bool = True,
                 link: bool = False, config_home: Path | None = None,
                 ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    _first, removal = _section_blocks("## Uninstalling")
    home = tmp_path / "home"
    unit_dir = (config_home or home / ".config") / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (home / ".config").mkdir(parents=True, exist_ok=True)
    for unit in units:
        (unit_dir / unit).write_text("[Unit]\n", encoding="ascii")
    comfy = tmp_path / "ComfyUI"
    checkout = tmp_path / "checkout" if link else comfy / "custom_nodes" / "dgx-monarch"
    (checkout / "src").mkdir(parents=True)
    (checkout / "src" / "local-work.py").write_text("kept\n", encoding="ascii")
    pack = comfy / "custom_nodes" / "dgx-monarch"
    if link:
        pack.parent.mkdir(parents=True)
        pack.symlink_to(checkout, target_is_directory=True)
    environment = {"HOME": str(home), "PATH": os.defpath}
    if config_home is not None:
        environment["XDG_CONFIG_HOME"] = str(config_home)
    if comfy_dir:
        environment["COMFY_DIR"] = str(comfy)
    result = subprocess.run(["bash", "-c", removal], capture_output=True, text=True, env=environment, timeout=30)
    return result, pack, checkout


@pytest.mark.parametrize("unit", ["dgxm-studio.service", "dgxm-worker.service"])
def test_the_removal_keeps_the_checkout_while_a_service_unit_is_installed(tmp_path, unit):
    result, pack, _checkout = _run_removal(tmp_path, units=(unit,))
    assert result.returncode != 0
    assert (pack / "src" / "local-work.py").read_text(encoding="ascii") == "kept\n"
    assert "finish step 1 first" in result.stdout


def test_the_removal_finds_a_studio_unit_under_a_redirected_config_home(tmp_path):
    """Studio writes its unit under XDG_CONFIG_HOME when that is set."""
    result, pack, _checkout = _run_removal(
        tmp_path, units=("dgxm-studio.service",), config_home=tmp_path / "elsewhere")
    assert result.returncode != 0 and pack.is_dir()
    assert "finish step 1 first" in result.stdout


def test_the_removal_needs_comfy_dir(tmp_path):
    result, pack, _checkout = _run_removal(tmp_path, comfy_dir=False)
    assert result.returncode != 0
    assert pack.is_dir() and "set COMFY_DIR" in result.stderr


def test_the_removal_deletes_the_checkout_once_uninstall_finished(tmp_path):
    result, pack, _checkout = _run_removal(tmp_path)
    assert result.returncode == 0, result.stderr
    assert not pack.exists() and not pack.is_symlink()


def test_the_removal_of_a_linked_pack_keeps_the_folder_it_names(tmp_path):
    result, pack, checkout = _run_removal(tmp_path, link=True)
    assert result.returncode == 0, result.stderr
    assert not pack.is_symlink() and not pack.exists()
    assert (checkout / "src" / "local-work.py").read_text(encoding="ascii") == "kept\n"


def test_no_install_block_removes_a_tree_after_a_command_it_does_not_check():
    """A removal shares its block with no other command, unless chained or guarded."""
    text = INSTALL.read_text(encoding="utf-8")
    for body in re.findall(r"^ *```bash\n(.*?)^ *```$", text, flags=re.MULTILINE | re.DOTALL):
        commands = _commands(body)
        for index, line in enumerate(commands):
            if re.search(r"\brm\s+-[a-z]*r", line):
                assert "${" in line and ":?" in line, line
                earlier = commands[:index]
                assert not earlier or earlier[-1].rstrip().endswith(("&&", "\\", "else")) or any(
                    command.lstrip().startswith("if ") for command in earlier), body
