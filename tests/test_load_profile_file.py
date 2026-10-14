"""Init's load_profile reports land in a bounded cache directory.

The reports used to go to /tmp as one new file per slow load, with nothing to
remove them. They now go to ~/.cache/dgx-monarch/profiles/, and each write
keeps only the newest PROFILE_KEEP reports there.
"""
from __future__ import annotations

import cProfile
import os
import stat
import time

import pytest

from dgx_monarch.actor import comfy_bridge, load_profile_file


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class _Log:
    """comfy_bridge's logger does not propagate, so read it at the module seam."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def info(self, fmt, *args):
        self.lines.append(("info", str(fmt) % args))

    def debug(self, fmt, *args):
        self.lines.append(("debug", str(fmt) % args))


@pytest.fixture
def bridge_log(monkeypatch):
    recorder = _Log()
    monkeypatch.setattr(comfy_bridge, "log", recorder)
    # The load-time counter is process-wide; give back the value it had.
    before = comfy_bridge._GPU_LOAD_S[0]
    yield recorder
    comfy_bridge._GPU_LOAD_S[0] = before


def _profiler() -> cProfile.Profile:
    profiler = cProfile.Profile()
    profiler.enable()
    sum(range(1000))
    profiler.disable()
    return profiler


def _seed(directory, count: int, start_ns: int = 1_000_000_000) -> list:
    os.makedirs(directory, exist_ok=True)
    paths = []
    for index in range(count):
        path = os.path.join(directory, f"dgxm-load-profile.1.{index}.seed{index:02d}.txt")
        with open(path, "w") as handle:
            handle.write("old report\n")
        stamp = start_ns + index * 1_000_000_000
        os.utime(path, ns=(stamp, stamp))
        paths.append(path)
    return paths


def test_the_report_goes_to_the_cache_dir_private_and_complete(home):
    path = load_profile_file.write_load_profile(_profiler(), 3.25, 41.9)

    directory = home / ".cache" / "dgx-monarch" / "profiles"
    assert os.path.dirname(path) == str(directory)
    name = os.path.basename(path)
    assert name.startswith(f"dgxm-load-profile.{os.getpid()}.41.")
    assert name.endswith(".txt")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    with open(path) as handle:
        text = handle.read()
    assert text.startswith("load_models_gpu took 3.2s\n")
    assert "cumulative" in text


def test_a_write_keeps_the_newest_reports_up_to_the_cap(home):
    directory = load_profile_file.profile_dir()
    seeded = _seed(directory, 25)
    stranger = os.path.join(directory, "notes.txt")
    with open(stranger, "w") as handle:
        handle.write("not a report\n")

    path = load_profile_file.write_load_profile(_profiler(), 2.5, 7.0)

    reports = sorted(
        name for name in os.listdir(directory)
        if name.startswith("dgxm-load-profile.") and name.endswith(".txt"))
    assert len(reports) == load_profile_file.PROFILE_KEEP == 20
    assert os.path.basename(path) in reports
    # The 19 newest seeds survive beside the new report; the six oldest go.
    assert all(not os.path.exists(old) for old in seeded[:6])
    assert all(os.path.exists(kept) for kept in seeded[6:])
    assert os.path.exists(stranger)


def test_the_new_report_survives_a_prune_even_with_an_older_mtime(home):
    directory = load_profile_file.profile_dir()
    _seed(directory, 30, start_ns=time.time_ns() + 3600 * 1_000_000_000)

    path = load_profile_file.write_load_profile(_profiler(), 2.0, 1.0)

    assert os.path.exists(path)
    assert len(os.listdir(directory)) == load_profile_file.PROFILE_KEEP


def test_prune_skips_reports_another_process_already_removed(home, monkeypatch):
    directory = load_profile_file.profile_dir()
    seeded = _seed(directory, 5)
    real_unlink = os.unlink

    def racing_unlink(path):
        real_unlink(path)
        if path == seeded[0]:
            raise FileNotFoundError(path)

    monkeypatch.setattr(load_profile_file.os, "unlink", racing_unlink)
    load_profile_file.prune_load_profiles(directory, keep=2)

    assert sorted(os.listdir(directory)) == sorted(
        os.path.basename(path) for path in seeded[3:])


def test_prune_of_a_missing_directory_is_quiet(tmp_path):
    load_profile_file.prune_load_profiles(str(tmp_path / "absent"), keep=1)


def test_a_slow_load_logs_the_report_path(home, bridge_log):
    t0 = time.perf_counter() - 3.0

    comfy_bridge._record_gpu_load_profile(_profiler(), t0)

    written = [line for level, line in bridge_log.lines
               if level == "info" and "written to" in line]
    assert len(written) == 1
    path = written[0].rsplit(" ", 1)[1]
    assert os.path.dirname(path) == load_profile_file.profile_dir()
    assert os.path.isfile(path)


def test_a_fast_load_or_no_profiler_writes_nothing(home, bridge_log):
    comfy_bridge._record_gpu_load_profile(_profiler(), time.perf_counter())
    comfy_bridge._record_gpu_load_profile(None, time.perf_counter() - 3.0)

    assert not os.path.exists(load_profile_file.profile_dir())
    assert bridge_log.lines == []


def test_a_failed_write_never_fails_the_load(home, monkeypatch, bridge_log):
    def refuse(*_args):
        raise PermissionError("read-only cache")

    monkeypatch.setattr(load_profile_file, "write_load_profile", refuse)

    comfy_bridge._record_gpu_load_profile(_profiler(), time.perf_counter() - 3.0)

    assert [level for level, _line in bridge_log.lines] == ["debug"]
    assert "not written" in bridge_log.lines[0][1]
