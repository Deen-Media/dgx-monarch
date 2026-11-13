"""The host trim: glibc keeps freed heap pages with the process until something trims them.

``actor/host_trim.py`` says why it exists. The second test here is the
mechanism itself, in a fresh interpreter so no earlier allocation of this
test run has moved glibc's threshold: free one mapped block just under 32 MiB,
after which blocks of that size and smaller come from the heap; allocate and
free about 2 GiB of them beneath a small live block; the pages stay; a trim
returns them. glibc raises its threshold only to the size of a freed mapped
block of 32 MiB or less, so the first block is 31 MiB, not larger.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys

import pytest

from dgx_monarch.actor import host_trim

_MECHANISM = r"""
import ctypes, sys

MIB = 2 ** 20

def rss():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return 0

def touched(size):
    block = bytearray(size)
    block[::4096] = b"\1" * len(range(0, size, 4096))
    return block

libc = ctypes.CDLL(None)
libc.gnu_get_libc_version.restype = ctypes.c_char_p
first = touched(31 * MIB)
del first                                   # a mapped block of 32 MiB or less, freed: the threshold moves up to it
held = [touched(24 * MIB) for _ in range(80)]   # 1.9 GiB of blocks under the threshold
top = touched(1 * MIB)                      # a small live block above them: the heap's top is not free
before = rss()
del held
freed = rss()
libc.malloc_trim.argtypes, libc.malloc_trim.restype = [ctypes.c_size_t], ctypes.c_int
libc.malloc_trim(0)
print(before, freed, rss(), len(top))
"""


def _glibc() -> bool:
    try:
        ctypes.CDLL(None).gnu_get_libc_version.restype = ctypes.c_char_p
        return True
    except Exception:
        return False


def test_the_trim_runs_reports_and_never_raises(monkeypatch, caplog):
    ran, seconds, before, after = host_trim.trim()
    assert isinstance(ran, bool) and seconds >= 0.0 and before >= 0 and after >= 0
    if _glibc():
        assert ran is True

    def no_library(name):
        raise OSError("no such library")

    monkeypatch.setattr(host_trim, "_unavailable", None)
    monkeypatch.setattr(host_trim.ctypes, "CDLL", no_library)
    with caplog.at_level("INFO"):
        assert host_trim.trim()[0] is False
        assert host_trim.trim()[0] is False
    assert caplog.text.count("offers no malloc_trim") == 1  # said once for the process


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
def test_freed_blocks_under_the_moved_threshold_come_back_only_at_a_trim():
    if not _glibc():
        pytest.skip("this C library is not glibc: no moving mapping threshold and no malloc_trim")
    import psutil

    if psutil.virtual_memory().available < 6 * 2 ** 30:
        pytest.skip("under 6 GiB of memory available: the test holds about 2 GiB")
    done = subprocess.run([sys.executable, "-I", "-c", _MECHANISM], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    before, freed, trimmed, top = (int(value) for value in done.stdout.split())
    assert top == 2 ** 20 and before > 1800
    if before - freed > 1500:
        pytest.skip("this glibc returned the freed blocks by itself (they were mapped one by one, or its "
                    f"heap trimmed): {before} MiB held, {freed} MiB after the free")
    assert freed - trimmed > 1500, (
        f"the trim returned too little: {before} MiB held, {freed} MiB after the free, {trimmed} MiB after the trim")
