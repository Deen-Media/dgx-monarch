"""The loader's native log level on a comfy-managed Worker.

Left at its default the loader calls Python at every device allocation, under
torch's allocator lock, and a status read that met it stopped the Worker for
good (docs/TROUBLESHOOTING.md #62). The bring-up sets ComfyUI's own level
before the loader's hooks go in.
"""
from __future__ import annotations

import ast
import logging

from comfy_managed_helpers import _func, _tree
from dgx_monarch.actor import comfy_dynamic


class _Control:
    """The loader's control module as the bring-up meets it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def set_log_info(self) -> None:
        self.calls.append("info")

    def set_log_none(self) -> None:
        self.calls.append("none")

    def set_log_debug(self) -> None:
        self.calls.append("debug")


def test_the_bring_up_sets_the_level_comfyui_itself_sets():
    control = _Control()

    assert comfy_dynamic.set_native_log_level(control) is True
    # INFO and nothing else: the per-allocation lines are below it, and the
    # loader's warnings and errors stay in the Worker's log.
    assert control.calls == ["info"]


def test_a_loader_with_no_level_to_set_is_said_and_not_raised(caplog):
    with caplog.at_level(logging.WARNING):
        assert comfy_dynamic.set_native_log_level(object()) is False

    assert "no log level to set" in caplog.text


def test_the_level_is_set_after_the_library_loads_and_before_its_hooks_go_in():
    """``init`` makes the library's handle; ``init_devices`` installs the hooks
    and starts the logging, so the level has to stand between the two."""
    body = ast.unparse(_func(_tree("actor/comfy_dynamic.py"), "stage_b"))

    assert body.index("ctl.init(") < body.index("set_native_log_level(ctl)")
    assert body.index("set_native_log_level(ctl)") < body.index("init_devices")


def test_nothing_in_the_pack_turns_the_loader_back_to_everything():
    """Debug and verbose levels bring the per-allocation callback back."""
    tree = _tree("actor/comfy_dynamic.py")
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    assert not {"set_log_debug", "set_log_verbose", "set_log_vverbose"} & (names | literals)
