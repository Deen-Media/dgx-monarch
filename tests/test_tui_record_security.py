"""Private-file boundary for full-fidelity TUI recordings."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from dgx_monarch.tui.data import RingStore
from dgx_monarch.tui.view_helpers import _append_record


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_snapshot_and_live_append_create_private_replayable_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.jsonl"
    ring = RingStore()
    ring.push({"t": 1.0, "kind": "first"})

    assert ring.save(str(snapshot)) == 1
    assert _mode(snapshot) == 0o600
    assert list(RingStore.load(str(snapshot)).ticks) == [
        {"t": 1.0, "kind": "first"}
    ]

    live = tmp_path / "live.jsonl"
    _append_record(str(live), {"t": 1.0, "kind": "first"})
    _append_record(str(live), {"t": 2.0, "kind": "second"})
    assert _mode(live) == 0o600
    assert list(RingStore.load(str(live)).ticks) == [
        {"t": 1.0, "kind": "first"},
        {"t": 2.0, "kind": "second"},
    ]


@pytest.mark.parametrize("writer", ["save", "append"])
def test_record_writers_refuse_symlinks_without_touching_target(
    tmp_path: Path, writer: str
) -> None:
    outside = tmp_path / "outside"
    outside.write_text("keep\n", encoding="utf-8")
    os.chmod(outside, 0o600)
    link = tmp_path / "record.jsonl"
    link.symlink_to(outside)

    with pytest.raises(PermissionError, match="owner-held regular file"):
        if writer == "save":
            RingStore().save(str(link))
        else:
            _append_record(str(link), {"t": 1.0})

    assert outside.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize("writer", ["save", "append"])
def test_record_writers_refuse_unsafe_mode_without_changing_file(
    tmp_path: Path, writer: str
) -> None:
    target = tmp_path / "record.jsonl"
    target.write_text(json.dumps({"t": 0.0}) + "\n", encoding="utf-8")
    os.chmod(target, 0o640)

    with pytest.raises(PermissionError, match="mode 0600"):
        if writer == "save":
            RingStore().save(str(target))
        else:
            _append_record(str(target), {"t": 1.0})

    assert target.read_text(encoding="utf-8") == '{"t": 0.0}\n'
    assert _mode(target) == 0o640


@pytest.mark.parametrize("writer", ["save", "append"])
def test_record_writers_refuse_non_regular_targets(tmp_path: Path, writer: str) -> None:
    target = tmp_path / "record.jsonl"
    target.mkdir(mode=0o700)

    with pytest.raises(PermissionError, match="regular file"):
        if writer == "save":
            RingStore().save(str(target))
        else:
            _append_record(str(target), {"t": 1.0})
