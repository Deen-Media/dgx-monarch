"""cli/driver_memory_row.py: doctor says which free-memory reading the running driver has.

The row classifies ComfyUI's stock ``/system_stats`` answer: the device's free
figure and the host's available memory, read in one process at one moment.
Figures here are a Spark's (the device and the host report one 124543 MiB
pool). No case needs a driver, CUDA or the network.
"""
from __future__ import annotations

import pytest

from dgx_monarch import driver_uma_memory
from dgx_monarch.cli import doctor, driver_memory_row
from dgx_monarch.cli.probe_certainty import OK, WARN

GIB = 2 ** 30
MIB = 2 ** 20
SPARK = 124543 * MIB


def _stats(*, device_free, available, cached=0, total=SPARK, host_total=SPARK, kind="cuda"):
    return {"system": {"ram_total": host_total, "ram_free": available},
            "devices": [{"name": "cuda:0 NVIDIA GB10", "type": kind, "index": 0, "vram_total": total,
                         "vram_free": device_free + cached, "torch_vram_total": cached, "torch_vram_free": cached}]}


def test_a_corrected_driver_reads_the_box_as_it_is():
    floor = driver_uma_memory.floor_bytes(SPARK)
    status, detail = driver_memory_row.classify(
        _stats(device_free=66 * GIB - floor, available=66 * GIB, cached=3 * GIB))
    assert status == OK
    assert detail == ("reads the box as it is: 58.4 GiB free on the device, which is the 66.0 GiB "
                      "available less a 7.6 GiB floor")


def test_a_driver_that_follows_memfree_is_named_with_both_figures():
    status, detail = driver_memory_row.classify(_stats(device_free=5 * GIB, available=66 * GIB, cached=2 * GIB))
    assert status == WARN
    assert "reads 5.0 GiB free on a device that shares 66.0 GiB available with the host" in detail
    assert "follows MemFree" in detail and driver_uma_memory.DISABLE_ENV in detail
    assert "TROUBLESHOOTING.md #12" in detail


def test_an_uncorrected_driver_on_an_empty_box_is_still_named():
    """MemFree near MemAvailable: stock reads more than available less the floor would give."""
    status, detail = driver_memory_row.classify(_stats(device_free=109 * GIB, available=110 * GIB))
    assert status == WARN and "reads 109.0 GiB free" in detail


@pytest.mark.parametrize("stats, expected", [
    (_stats(device_free=20 * GIB, available=40 * GIB, total=24 * GIB, host_total=64 * GIB),
     "the GPU has memory of its own (24 GiB beside 64 GiB of host memory)"),
    (_stats(device_free=20 * GIB, available=40 * GIB, kind="cpu"), "device type 'cpu'"),
], ids=["a discrete GPU", "the CPU"])
def test_a_device_with_nothing_to_correct_is_ok(stats, expected):
    status, detail = driver_memory_row.classify(stats)
    assert status == OK and expected in detail and "nothing to correct" in detail


@pytest.mark.parametrize("payload", [
    {}, {"system": {}, "devices": []}, {"system": {"ram_total": SPARK}, "devices": [{"type": "cuda"}]},
    {"system": {"ram_total": SPARK, "ram_free": True},
     "devices": [{"type": "cuda", "vram_total": SPARK, "vram_free": 1, "torch_vram_free": 0}]},
], ids=["empty", "no device", "no figures", "a boolean for a figure"])
def test_an_answer_without_the_figures_is_unobserved_not_a_guess(payload):
    status, detail = driver_memory_row.classify(payload)
    assert status == WARN and "unobserved" in detail


def test_no_driver_answering_says_what_a_driver_would_do():
    status, name, detail = driver_memory_row.verdict([])
    assert (status, name) == (OK, "driver memory reading")
    assert "no local ComfyUI answered" in detail and driver_uma_memory.DISABLE_ENV in detail


def test_the_worst_driver_leads_the_row():
    floor = driver_uma_memory.floor_bytes(SPARK)
    good = _stats(device_free=66 * GIB - floor, available=66 * GIB)
    bad = _stats(device_free=5 * GIB, available=66 * GIB)
    status, _name, detail = driver_memory_row.verdict([("127.0.0.1:8188", good), ("127.0.0.1:8189", bad)])
    assert status == WARN
    assert detail.startswith("127.0.0.1:8189: reads 5.0 GiB free")
    assert detail.endswith(f"(also probed 127.0.0.1:8188: {OK})")


def test_the_row_never_raises_and_is_in_doctors_imports(monkeypatch):
    monkeypatch.setattr(driver_memory_row, "fetch", lambda: (_ for _ in ()).throw(OSError("no route")))
    status, name, detail = driver_memory_row.doctor_row()
    assert (status, name) == (WARN, "driver memory reading") and "probe unavailable" in detail
    monkeypatch.setattr(driver_memory_row, "fetch", lambda: [])
    assert driver_memory_row.doctor_row()[0] == OK
    assert doctor.driver_memory_row is driver_memory_row


def test_fetch_skips_a_candidate_that_does_not_answer():
    # Port 9 (discard) on loopback: refused at once, so the probe returns nothing and does not raise.
    assert driver_memory_row.fetch(("127.0.0.1:9",)) == []


def test_doctor_lists_the_row():
    import inspect

    assert "driver_memory_row.doctor_row()" in inspect.getsource(doctor._doctor_rows)


def test_a_comfyui_that_does_not_serve_this_pack_is_not_judged(monkeypatch):
    """A stock ComfyUI on a local port reads memory its own way by the user's choice; the row passes over it."""
    import io
    import json

    floor = driver_uma_memory.floor_bytes(SPARK)
    stats = {"stock:8188": _stats(device_free=5 * GIB, available=66 * GIB),
             "ours:8191": _stats(device_free=66 * GIB - floor, available=66 * GIB),
             "unsure:8192": _stats(device_free=5 * GIB, available=66 * GIB)}
    confirmed = {"stock:8188": False, "ours:8191": True, "unsure:8192": None}
    monkeypatch.setattr(driver_memory_row.comfy_ports, "_confirmed_driver", lambda candidate: confirmed[candidate])
    monkeypatch.setattr(driver_memory_row.urllib.request, "urlopen",
                        lambda url, timeout=None: io.BytesIO(json.dumps(stats[url.split("/")[2]]).encode()))
    answers = driver_memory_row.fetch(tuple(stats))
    # A candidate that did not answer the pack's own probe either way is still read: silence proves nothing.
    assert [candidate for candidate, _payload in answers] == ["ours:8191", "unsure:8192"]


def test_under_a_cgroup_limit_one_pool_is_judged_against_the_box():
    """ComfyUI reports a cgroup limit as its host total; the device still shares the box's memory."""
    floor = driver_uma_memory.floor_bytes(SPARK)
    limited = _stats(device_free=20 * GIB - floor, available=20 * GIB, host_total=32 * GIB)
    status, detail = driver_memory_row.classify(limited)
    assert status == OK and "the GPU has memory of its own" in detail  # the misreading, with no box total
    status, detail = driver_memory_row.classify(limited, machine_total=float(SPARK))
    assert status == OK and detail.startswith("reads the box as it is: 12.4 GiB free on the device")
    uncorrected = _stats(device_free=5 * GIB, available=20 * GIB, host_total=32 * GIB)
    status, detail = driver_memory_row.classify(uncorrected, machine_total=float(SPARK))
    assert status == WARN and "run doctor again before acting" in detail
