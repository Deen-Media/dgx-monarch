"""A Worker's diagnostic switches from the one file its user owns (``actor/worker_switches.py``).

A managed unit clears its environment, so the file is the way to set a
Worker-side switch there. These tests hold what makes that safe: six names
and no other, a file only the Worker's user can write, the environment's
precedence, a read at every actor start, and never a silent switch.
"""
from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_setup
from dgx_monarch.actor import (
    backup_agreement,
    context_room,
    load_fault,
    lora_partial_load,
    resident_ledger,
    worker_status,
    worker_switches,
)
from dgx_monarch.cli.worker_process_env import worker_environment_key_allowed
from dgx_monarch.log import get_logger

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"


def _file(tmp_path: Path, text: str, mode: int = 0o600) -> Path:
    path = tmp_path / "worker.env"
    path.write_text(text)
    path.chmod(mode)
    return path


def test_the_names_are_the_six_the_actor_modules_read_and_the_loop_lets_through():
    assert worker_switches.NAMES == (
        "DGXM_DISABLE_CONTEXT_ROOM", "DGXM_DISABLE_COPY_FREE_DROP", "DGXM_DISABLE_BACKUP_FREE",
        "DGXM_FAULT_LOAD_RANK", "DGXM_FAULT_BACKUP_RANK", "DGXM_ACCEPTANCE")
    # Taken from the modules that read them, not written a second time.
    assert worker_switches.NAMES == (
        context_room.DISABLE_ENV, resident_ledger.COPY_FREE_DROP_DISABLE_ENV,
        lora_partial_load.BACKUP_FREE_DISABLE_ENV, load_fault.RANK_ENV,
        backup_agreement.FAULT_RANK_ENV, load_fault.ACCEPTANCE_ENV)
    assert all(worker_environment_key_allowed(name) for name in worker_switches.NAMES)


def test_the_file_is_under_the_home_of_the_workers_user(monkeypatch):
    monkeypatch.setattr(worker_switches, "FILE", Path(".config") / "dgx-monarch" / "worker.env")
    monkeypatch.setenv("HOME", "/home/user")
    assert worker_switches.file_path() == Path("/home/user/.config/dgx-monarch/worker.env")


def test_no_file_sets_nothing_and_says_nothing(tmp_path, caplog):
    environ: dict[str, str] = {"HOME": "/h"}
    with caplog.at_level(logging.INFO):
        assert worker_switches.apply(environ, tmp_path / "worker.env") == ()
    assert environ == {"HOME": "/h"} and caplog.text == ""


def test_the_files_switches_reach_the_actor_and_each_is_named_at_warning(tmp_path, caplog):
    path = _file(tmp_path, "# the pair's proof cell\n\nDGXM_DISABLE_BACKUP_FREE=1\n"
                           "  DGXM_FAULT_BACKUP_RANK = \"1\"\nDGXM_ACCEPTANCE='1'\n")
    environ: dict[str, str] = {}
    with caplog.at_level(logging.WARNING):
        active = worker_switches.apply(environ, path)
    assert environ == {"DGXM_DISABLE_BACKUP_FREE": "1", "DGXM_FAULT_BACKUP_RANK": "1", "DGXM_ACCEPTANCE": "1"}
    assert active == ("DGXM_DISABLE_BACKUP_FREE=1 (worker.env)", "DGXM_FAULT_BACKUP_RANK=1 (worker.env)",
                      "DGXM_ACCEPTANCE=1 (worker.env)")
    assert "Worker switches set on this box: " + ", ".join(active) in caplog.text
    assert all(record.levelno == logging.WARNING for record in caplog.records)


def test_a_name_that_is_not_a_diagnostic_switch_is_ignored_and_named_without_its_value(tmp_path, caplog):
    path = _file(tmp_path, "LD_PRELOAD=/tmp/evil.so\nDGXM_SSH_KEY=very-private\nPYTHONPATH=/elsewhere\n"
                           "CUDA_VISIBLE_DEVICES=1\nnot a line\nDGXM_DISABLE_CONTEXT_ROOM=1\n")
    environ: dict[str, str] = {}
    with caplog.at_level(logging.WARNING):
        active = worker_switches.apply(environ, path)
    assert environ == {"DGXM_DISABLE_CONTEXT_ROOM": "1"}
    assert active == ("DGXM_DISABLE_CONTEXT_ROOM=1 (worker.env)",)
    ignored = [record.getMessage() for record in caplog.records if "ignored" in record.getMessage()]
    assert len(ignored) == 1
    for name in ("LD_PRELOAD", "DGXM_SSH_KEY", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "line 5"):
        assert name in ignored[0]
    assert "very-private" not in caplog.text and "evil.so" not in caplog.text and "/elsewhere" not in caplog.text


@pytest.mark.parametrize("value", ["", "$(reboot)", "a b", "x" * 65, "1;2"])
def test_a_switch_with_no_plain_value_is_not_set_and_is_said(tmp_path, caplog, value):
    path = _file(tmp_path, f"DGXM_FAULT_LOAD_RANK={value}\n")
    environ: dict[str, str] = {}
    with caplog.at_level(logging.WARNING):
        assert worker_switches.apply(environ, path) == ()
    assert environ == {} and "DGXM_FAULT_LOAD_RANK has no plain value" in caplog.text
    assert value not in caplog.text or value == ""


def test_the_environment_wins_over_the_file(tmp_path, caplog):
    """Where a drop-in or the shell does reach the loop, it keeps its say."""
    path = _file(tmp_path, "DGXM_FAULT_BACKUP_RANK=1\nDGXM_ACCEPTANCE=1\nDGXM_DISABLE_BACKUP_FREE=1\n")
    environ = {"DGXM_FAULT_BACKUP_RANK": "0", "DGXM_DISABLE_BACKUP_FREE": "1"}
    with caplog.at_level(logging.WARNING):
        active = worker_switches.apply(environ, path)
    assert environ["DGXM_FAULT_BACKUP_RANK"] == "0" and environ["DGXM_ACCEPTANCE"] == "1"
    assert active == ("DGXM_DISABLE_BACKUP_FREE=1 (environment)", "DGXM_FAULT_BACKUP_RANK=0 (environment)",
                      "DGXM_ACCEPTANCE=1 (worker.env)")
    # Said once, for the name whose two values differ; the equal one needs no word.
    wins = [record.getMessage() for record in caplog.records if "which wins" in record.getMessage()]
    assert len(wins) == 1 and "DGXM_FAULT_BACKUP_RANK" in wins[0]


def test_a_switch_set_only_in_the_environment_is_still_named(tmp_path, caplog):
    """A forgotten switch from a drop-in is as dangerous as one from the file."""
    environ = {"DGXM_DISABLE_COPY_FREE_DROP": "1", "DGXM_FAULT_LOAD_RANK": "a value; with spaces"}
    with caplog.at_level(logging.WARNING):
        active = worker_switches.apply(environ, tmp_path / "absent")
    assert active == ("DGXM_DISABLE_COPY_FREE_DROP=1 (environment)", "DGXM_FAULT_LOAD_RANK=<set> (environment)")
    assert "with spaces" not in caplog.text


def test_a_file_another_user_owns_is_ignored_and_said(tmp_path, monkeypatch, caplog):
    """The unit clears its environment on purpose; this must not be a way round that for another user."""
    path = _file(tmp_path, "DGXM_DISABLE_BACKUP_FREE=1\n")
    monkeypatch.setattr(os, "geteuid", lambda: path.stat().st_uid + 1)
    environ: dict[str, str] = {}
    with caplog.at_level(logging.WARNING):
        assert worker_switches.apply(environ, path) == ()
    assert environ == {} and "it is not owned by the Worker's user" in caplog.text


@pytest.mark.parametrize("mode", [0o620, 0o602, 0o666, 0o660])
def test_a_file_group_or_others_can_write_is_ignored_and_said(tmp_path, caplog, mode):
    path = _file(tmp_path, "DGXM_DISABLE_BACKUP_FREE=1\n", mode)
    environ: dict[str, str] = {}
    with caplog.at_level(logging.WARNING):
        assert worker_switches.apply(environ, path) == ()
    assert environ == {} and "writable by group or others" in caplog.text and f"{mode:04o}" in caplog.text


@pytest.mark.parametrize("mode", [0o600, 0o644, 0o400, 0o640])
def test_a_file_only_its_owner_can_write_is_read(tmp_path, mode):
    path = _file(tmp_path, "DGXM_DISABLE_BACKUP_FREE=1\n", mode)
    environ: dict[str, str] = {}
    assert worker_switches.apply(environ, path) == ("DGXM_DISABLE_BACKUP_FREE=1 (worker.env)",)


def test_a_link_a_directory_and_an_oversized_file_are_ignored_and_said(tmp_path, caplog):
    real = tmp_path / "real"
    real.mkdir()
    target = _file(real, "DGXM_DISABLE_BACKUP_FREE=1\n")
    link = tmp_path / "link.env"
    link.symlink_to(target)
    folder = tmp_path / "folder.env"
    folder.mkdir()
    big = tmp_path / "big.env"
    big.write_text("# pad\n" * 4000 + "DGXM_DISABLE_BACKUP_FREE=1\n")
    big.chmod(0o600)
    for path, why in ((link, "symbolic link"), (folder, "not a regular file"), (big, "larger than")):
        caplog.clear()
        environ: dict[str, str] = {}
        with caplog.at_level(logging.WARNING):
            assert worker_switches.apply(environ, path) == ()
        assert environ == {} and why in caplog.text, path


def test_a_change_to_the_file_is_seen_by_the_next_actor_to_start(tmp_path):
    """Read at every spawn, not once when the loop starts: a recycle applies it with no restart."""
    path = tmp_path / "worker.env"
    first: dict[str, str] = {}
    assert worker_switches.apply(first, path) == ()
    _file(tmp_path, "DGXM_DISABLE_BACKUP_FREE=1\n")
    second: dict[str, str] = {}
    assert worker_switches.apply(second, path) == ("DGXM_DISABLE_BACKUP_FREE=1 (worker.env)",)
    path.unlink()
    third: dict[str, str] = {}
    assert worker_switches.apply(third, path) == () and first == {} and third == {}


def test_every_actor_reads_the_file_as_it_starts_and_keeps_what_it_found():
    """The read sits in the Worker's constructor, which runs in each new actor process, and nowhere
    in the worker loop, which a recycle does not restart."""
    tree = ast.parse((SRC / "actor" / "worker.py").read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPUWorker")
    init = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assert "self._switches = worker_switches.apply()" in [ast.unparse(node) for node in init.body]
    for relative in ("cli/worker_loop.py", "cli/worker_process_env.py"):
        assert "worker_switches" not in (SRC / relative).read_text(), relative


def test_the_status_reply_carries_the_switches_before_and_after_setup(monkeypatch):
    monkeypatch.setattr(worker_status, "source_manifest_sha256", lambda: "source")
    names = ("DGXM_DISABLE_BACKUP_FREE=1 (worker.env)",)
    worker = SimpleNamespace(rank=0, world=1, topology={}, _attn=None, _setup_key=None,
                             store=SimpleNamespace(snapshot=lambda: {}), _switches=names)
    status = worker_status.status_impl(worker)
    assert status["worker_switches"] == list(names) and status["vram"] is None
    del worker._switches
    assert worker_status.status_impl(worker)["worker_switches"] == []


def test_the_setup_reply_carries_the_switches():
    source = ast.unparse(next(node for node in ast.parse((SRC / "actor" / "worker_env.py").read_text()).body
                              if isinstance(node, ast.FunctionDef) and node.name == "setup_info"))
    assert "'switches': list(getattr(worker, '_switches', ()))" in source


def test_the_driver_repeats_a_workers_switches_at_every_setup(caplog):
    handle = SimpleNamespace()
    quiet = {"host": "box-a", "rank": 0, "switches": []}
    loud = {"host": "box-b", "rank": 1, "switches": ["DGXM_DISABLE_BACKUP_FREE=1 (worker.env)"]}
    # The driver makes this logger inside the call, after pytest has hooked the loggers that exist.
    logger = get_logger(mesh_setup.__name__)
    logger.addHandler(caplog.handler)
    try:
        mesh_setup.publish_worker_capabilities(handle, [quiet, loud])
    finally:
        logger.removeHandler(caplog.handler)
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == ["Worker switches set on box-b: DGXM_DISABLE_BACKUP_FREE=1 (worker.env). "
                        "Remove them when the diagnosis is done (docs/CLUSTER.md)"]
    assert handle.worker_capabilities == [quiet, loud]
