"""The loader's native log level on a comfy-managed Worker.

Left at its default the loader calls Python at every device allocation, under
torch's allocator lock, and a status read that met it stopped the Worker for
good (docs/TROUBLESHOOTING.md #62). The bring-up sets ComfyUI's own level
before the loader's hooks go in.
"""
from __future__ import annotations

import ast
import ctypes
import logging
from types import SimpleNamespace

from comfy_managed_helpers import _func, _tree
from dgx_monarch.actor import comfy_dynamic


class _Control:
    """The loader's control module as the bring-up meets it."""

    def __init__(self, lib: object = True) -> None:
        self.lib = lib  # the library handle ``init`` leaves, None when it failed to load
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


def test_a_library_that_failed_to_load_is_left_for_the_typed_refusal():
    """``init`` returns False and leaves ``lib`` at None; the real setter would
    then raise AttributeError out of the bring-up, in the place of the class P
    refusal ``init_devices`` leads to."""
    class _Unloaded(_Control):
        def set_log_info(self) -> None:
            raise AttributeError("'NoneType' object has no attribute 'set_log_level_info'")

    control = _Unloaded(lib=None)

    assert comfy_dynamic.set_native_log_level(control) is False
    assert control.calls == []


def test_the_bring_up_still_asks_the_library_for_its_devices_after_the_level():
    """The refusal in words hangs on ``init_devices`` returning False, so
    nothing between the level and that call may return or raise first."""
    stage_b = _func(_tree("actor/comfy_dynamic.py"), "stage_b")
    statements = [ast.unparse(node) for node in stage_b.body]
    level = next(i for i, text in enumerate(statements) if text == "set_native_log_level(ctl)")
    devices = next(i for i, text in enumerate(statements) if "init_devices" in text)
    refusal = next(i for i, text in enumerate(statements)
                   if text.startswith("if not ok") and "RefusalClass.PHYSICS" in text)

    assert level < devices < refusal
    assert not any(isinstance(node, ast.Return | ast.Raise)
                   for statement in stage_b.body[level:devices]
                   for node in ast.walk(statement))


def test_the_level_is_read_back_from_the_library_and_not_from_what_was_asked():
    # Any exported C int stands in for the loader's ``log_level`` symbol here.
    process = SimpleNamespace(lib=ctypes.CDLL(None))
    expected = ctypes.c_int.in_dll(process.lib, "optind").value

    assert comfy_dynamic.native_log_level(process, symbol="optind") == expected
    assert isinstance(comfy_dynamic.native_log_level(process, symbol="optind"), int)


def test_a_level_that_cannot_be_read_is_none_and_never_raises():
    assert comfy_dynamic.native_log_level(SimpleNamespace(lib=None)) is None
    assert comfy_dynamic.native_log_level(object()) is None
    assert comfy_dynamic.native_log_level(
        SimpleNamespace(lib=ctypes.CDLL(None)), symbol="dgxm_no_such_symbol") is None


def test_the_bring_up_line_says_the_level_it_read_back():
    assert comfy_dynamic.native_log_words(4) == "native log level INFO (read back as 4)"
    assert comfy_dynamic.native_log_words(None) == "native log level not readable"
    assert "NOT the INFO (4)" in comfy_dynamic.native_log_words(0)
    # The read sits after the hooks go in and feeds the line a user can look for.
    body = ast.unparse(_func(_tree("actor/comfy_dynamic.py"), "stage_b"))
    assert body.index("init_devices") < body.index("level = native_log_level(ctl)")
    assert body.index("level = native_log_level(ctl)") < body.index("residency ACTIVE")
    assert "native_log_words(level)" in body
