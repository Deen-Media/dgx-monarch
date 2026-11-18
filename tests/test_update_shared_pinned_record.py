"""The one shared-file allowance is the official wheel's own record of that file.

``update_payload_ownership._SHARED_PINNED`` lets one foreign wheel share a file
with torchmonarch when that wheel's record of the file is the official one.
``WHEEL_ROW`` is line 113 of the RECORD inside the official
``xfuser-0.8.0-py3-none-any.whl``, copied byte for byte: the wheel's published
sha256 is 153ecae2af96835e791224e1153646bec1231491c68d07044f081f2e8097cad8 and
the RECORD member's is
6507bc09f0b34f39c8a53939634f34a63c4ffd7ba1bf81bad095ffe47e696841. The first
test reads the allowance out of that row with the production parser; the
second reads it out of an installed xfuser 0.8.0, whose RECORD pip writes from
the bytes it installed, wherever this Python holds one.
"""
from __future__ import annotations

import importlib.metadata

import pytest

from dgx_monarch.cli import update_payload_ownership as ownership

WHEEL_ROW ="tests/conftest.py,sha256=utFLTVnq1Ypbjm9HNO0sKd_8Z4_jZXUgH-Pqg3zLboQ,376\n"


def _allowance() -> dict[str, tuple[str, str]]:
    # A new entry needs its own wheel's row and an installed-copy check beside these.
    assert set(ownership._SHARED_PINNED) == {("xfuser", "0.8.0")}
    table = ownership._SHARED_PINNED[("xfuser", "0.8.0")]
    assert table
    return table


def test_the_allowance_is_the_row_the_official_wheel_records():
    records, invalid = ownership._records(WHEEL_ROW.encode())
    assert invalid is None
    table = _allowance()
    assert table == {path: records[path] for path in table}


def test_an_installed_xfuser_0_8_0_records_the_same_row():
    try:
        installed = importlib.metadata.distribution("xfuser")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("this Python has no xfuser")
    if installed.version != "0.8.0":
        pytest.skip(f"this Python holds xfuser {installed.version}, not 0.8.0")
    records, _invalid = ownership._records((installed.read_text("RECORD") or "").encode())
    table = _allowance()
    assert table == {path: records[path] for path in table}
