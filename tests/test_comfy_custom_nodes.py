"""Headless custom-node preloading logs why it loaded nothing."""
from __future__ import annotations

import sys
import types

import pytest

from dgx_monarch.actor import comfy_custom_nodes


def _capture_info(monkeypatch) -> list[str]:
    messages: list[str] = []

    def record(message: str, *args) -> None:
        messages.append(message % args)

    monkeypatch.setattr(comfy_custom_nodes.log, "info", record)
    return messages


def test_explicit_custom_node_opt_out_is_visible(monkeypatch):
    messages = _capture_info(monkeypatch)
    monkeypatch.setenv("DGXM_NO_CUSTOM_NODES", "1")

    comfy_custom_nodes.load_custom_node_modules()

    assert messages == ["custom node preload disabled by DGXM_NO_CUSTOM_NODES"]


@pytest.mark.parametrize("contents", [(), ("ComfyUI-Manager", ".hidden.py")])
def test_empty_or_all_skipped_preload_is_visible(tmp_path, monkeypatch, contents):
    messages = _capture_info(monkeypatch)
    monkeypatch.delenv("DGXM_NO_CUSTOM_NODES", raising=False)
    for name in contents:
        path = tmp_path / name
        if "." in name:
            path.write_text("")
        else:
            path.mkdir()

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_folder_paths = lambda _kind: [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(
        comfy_custom_nodes,
        "ensure_prompt_server_stub",
        lambda: True,
    )

    comfy_custom_nodes.load_custom_node_modules()

    assert messages == [
        "custom node preload found no eligible packs; custom-node directories "
        "were empty or every entry was skipped/disabled"
    ]
