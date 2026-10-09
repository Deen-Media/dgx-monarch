"""Helpers for the systemd unit lines that lifecycle and guided setup write."""
from __future__ import annotations


def quote(value: str, *, preserve_home_specifier: bool = False) -> str:
    """Quote one systemd argument without routing config through a shell."""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError("systemd arguments must not contain control characters")
    sentinel = "\x00"
    if preserve_home_specifier:
        value = value.replace("%h", sentinel, 1)
    value = value.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    if preserve_home_specifier:
        value = value.replace(sentinel, "%h")
    return f'"{value}"'


def resolve_python(python_bin: str) -> str:
    """Render a configured leading tilde as systemd's user-home specifier."""
    if python_bin == "~":
        return "%h"
    if python_bin.startswith("~/"):
        return f"%h/{python_bin[2:]}"
    return python_bin
