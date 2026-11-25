"""``dgxm doctor`` says whether the last VAE decode split, and over how many ranks.

The row names the ``[worker_args] worker_vae_decode`` setting and, with
``"auto"``, reads the driver's last decode record over HTTP (never attaching a
mesh): split over N ranks, stayed on the driver and why, or fell back after a
rank failed, which is a WARN that points at the troubleshooting entry.
"""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

from dgx_monarch.cli import vae_decode_row
from dgx_monarch.cli.probe_certainty import OK, WARN

AUTO = SimpleNamespace(worker_args={"worker_vae_decode": "auto"})


def test_off_is_the_default_and_reads_nothing(monkeypatch):
    monkeypatch.setattr(vae_decode_row, "fetch_last", lambda: (_ for _ in ()).throw(AssertionError("no fetch")))
    status, name, detail = vae_decode_row.doctor_row(SimpleNamespace(worker_args={}))
    assert (status, name) == (OK, "worker VAE decode")
    assert detail == "off ([worker_args] worker_vae_decode); every decode stays on the driver"
    assert vae_decode_row.doctor_row(None)[2].startswith("off ")


def test_auto_without_a_driver_or_a_decode():
    assert vae_decode_row.verdict("auto", False, None)[2] == (
        "auto; no local driver answered, so the last decode is unobserved")
    assert vae_decode_row.verdict("auto", True, None)[2] == (
        "auto; the driver has made no stock VAE decode since it started")


def test_auto_reports_the_last_decode():
    split = {"family": "minimax_h3", "split": True, "ranks": 2, "units": 7, "local": 4,
             "remote": {"1": 3}, "seconds": 33.14, "failures": []}
    assert vae_decode_row.verdict("auto", True, split) == (
        OK, "worker VAE decode",
        "auto; the last decode (minimax_h3) split over 2 ranks: 7 units, 4 on the driver, 3 on rank 1, 33.1 s")
    stayed = {"family": "wan21", "split": False, "reason": "use VAE Decode (Tiled) to split it", "failures": []}
    assert vae_decode_row.verdict("auto", True, stayed)[2] == (
        "auto; the last decode (wan21) stayed on the driver: use VAE Decode (Tiled) to split it")
    fell = {"family": "ltx", "split": False, "units": 12, "fallback": 6,
            "failures": ["rank 1: rank 1 on sibling declined: this host's vae/x is not the driver's file"]}
    status, _name, detail = vae_decode_row.verdict("auto", True, fell)
    assert status == WARN and "fell back to the driver for 6 of 12 units" in detail
    assert detail.endswith("(docs/TROUBLESHOOTING.md #140)")


def test_the_newest_record_from_any_local_driver(monkeypatch):
    payloads = {
        "127.0.0.1:8188": {"render": {"vae_decode": {"t": 10.0, "family": "wan21", "split": False,
                                                     "reason": "old", "failures": []}}},
        "127.0.0.1:8189": {"render": {"vae_decode": {"t": 20.0, "family": "minimax_h3", "split": True,
                                                     "ranks": 2, "units": 3, "local": 2, "remote": {"1": 1},
                                                     "seconds": 1.0, "failures": []}}},
    }

    def urlopen(url, timeout):
        for candidate, payload in payloads.items():
            if candidate in url:
                return io.BytesIO(json.dumps(payload).encode())
        raise ConnectionRefusedError

    monkeypatch.setattr(vae_decode_row.urllib.request, "urlopen", urlopen)
    answered, record = vae_decode_row.fetch_last(("127.0.0.1:8188", "127.0.0.1:8189", "127.0.0.1:8190"))
    assert answered and record["family"] == "minimax_h3"
    monkeypatch.setattr(vae_decode_row, "fetch_last", lambda: (answered, record))
    assert vae_decode_row.doctor_row(AUTO)[2].startswith("auto; the last decode (minimax_h3) split over 2 ranks")


def test_the_row_never_raises(monkeypatch):
    monkeypatch.setattr(vae_decode_row, "fetch_last", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    status, _name, detail = vae_decode_row.doctor_row(AUTO)
    assert status == WARN and detail.startswith("probe unavailable")
