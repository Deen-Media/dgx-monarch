"""Classify frontend/backend skew from /system_stats with an injected resolver.

A frontend newer than the backend supports can blank the canvas while API
requests still return 200 (docs/TROUBLESHOOTING.md #19).
"""
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.cli.doctor import _FAIL, _OK, _WARN, classify_frontend_skew


def _system(required="1.45.20", argv=None, installed=None):
    pkgs = ([{"name": "comfyui-frontend-package", "installed": installed}]
            if installed else [])
    return {"required_frontend_version": required,
            "argv": argv or ["main.py"],
            "comfy_package_versions": pkgs}


def test_latest_resolved_ahead_fails_with_pin_fix():
    """The reported case: @latest served 1.48.0 on a 0.27.0 backend requiring
    1.45.20, and the canvas went blank."""
    status, detail = classify_frontend_skew(
        _system(argv=["main.py", "--front-end-version",
                      "Comfy-Org/ComfyUI_frontend@latest"]),
        resolve_latest=lambda repo: "1.48.0")
    assert status == _FAIL
    assert "@1.45.20" in detail            # copy-pasteable pin
    assert "@latest" in detail             # moving-target callout
    assert "blank" in detail


def test_pinned_flag_ahead_fails_without_latest_note():
    status, detail = classify_frontend_skew(
        _system(argv=["main.py", "--front-end-version",
                      "Comfy-Org/ComfyUI_frontend@1.48.0"]))
    assert status == _FAIL
    assert "re-downloads the newest" not in detail


def test_package_match_is_ok():
    status, _detail = classify_frontend_skew(_system(installed="1.45.20"))
    assert status == _OK


def test_package_older_warns_with_upgrade_fix():
    status, detail = classify_frontend_skew(_system(installed="1.40.0"))
    assert status == _WARN
    assert "comfyui-frontend-package==1.45.20" in detail


def test_latest_matching_today_still_warns_moving_target():
    status, detail = classify_frontend_skew(
        _system(argv=["main.py", "--front-end-version",
                      "Comfy-Org/ComfyUI_frontend@latest"]),
        resolve_latest=lambda repo: "1.45.20")
    assert status == _WARN
    assert "re-downloads the newest" in detail


def test_unresolvable_served_version_warns():
    status, detail = classify_frontend_skew(
        _system(argv=["main.py", "--front-end-version",
                      "Comfy-Org/ComfyUI_frontend@latest"]),
        resolve_latest=lambda repo: None)
    assert status == _WARN
    assert "TROUBLESHOOTING" in detail


def test_missing_required_version_warns():
    status, _ = classify_frontend_skew(_system(required=None, installed="1.45.20"))
    assert status == _WARN


@pytest.mark.parametrize("served,required,expected", [
    ("1.45.20", "1.45.20", _OK),
    ("1.48.0", "1.45.20", _FAIL),   # numeric compare, not lexicographic
    ("1.45.2", "1.45.20", _WARN),
])
def test_version_comparison_is_numeric(served, required, expected):
    status, _ = classify_frontend_skew(
        _system(required=required,
                argv=["main.py", "--front-end-version", f"o/r@{served}"]))
    assert status == expected


def test_incomparable_versions_warn_not_fail(monkeypatch):
    """Versions that cannot be compared must never read as FAIL: 'different'
    is not 'newer'."""
    from dgx_monarch.cli import doctor

    def boom(v):
        raise ValueError(v)

    monkeypatch.setattr(doctor, "_parse_version", boom)
    status, detail = classify_frontend_skew(
        _system(required="1.45.20.rc1",
                argv=["main.py", "--front-end-version", "o/r@1.45.20.rc2"]))
    assert status == _WARN
    assert "cannot be compared" in detail
    # equal strings stay OK even when unparseable
    status, _ = classify_frontend_skew(
        _system(required="weird", argv=["main.py", "--front-end-version", "o/r@weird"]))
    assert status == _OK


def test_looks_like_comfy_filter():
    from dgx_monarch.cli.comfy_ports import looks_like_comfy as _looks_like_comfy

    assert _looks_like_comfy(["python", "/home/u/ComfyUI/main.py", "--port", "8195"])
    assert _looks_like_comfy(["python", "main.py", "--front-end-version",
                              "Comfy-Org/ComfyUI_frontend@latest"])
    # The opening of the argv scripts/comfy-driver.sh runs after it cd's into
    # ComfyUI. This is a discovery candidate; HTTP endpoints confirm it in callers.
    assert _looks_like_comfy([
        "python", "main.py", "--listen", "127.0.0.1", "--port", "8195",
        "--disable-pinned-memory", "--disable-async-offload",
    ])
    assert not _looks_like_comfy(["python", "main.py"])
    assert not _looks_like_comfy(["python", "main.py", "--port", "not-a-port"])
    assert not _looks_like_comfy(["python", "src/app/main.py"])   # FastAPI etc.
    assert not _looks_like_comfy(["python", "-m", "comfy_thing"])  # no main.py
    assert not _looks_like_comfy([])


def test_process_discovery_includes_shipped_bare_main_custom_port(monkeypatch):
    from dgx_monarch.cli.comfy_ports import find_comfy_ports

    connection = SimpleNamespace(
        status="LISTEN", laddr=SimpleNamespace(port=8195)
    )
    process = SimpleNamespace(
        info={
            "pid": 4321,
            "cmdline": [
                "python", "main.py", "--listen", "127.0.0.1",
                "--port", "8195", "--disable-pinned-memory",
            ],
        },
        net_connections=lambda kind: [connection],
    )
    psutil = SimpleNamespace(
        CONN_LISTEN="LISTEN",
        Error=RuntimeError,
        process_iter=lambda attrs: [process],
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil)

    assert find_comfy_ports()[8195] == 4321


def test_pack_confirmation_requires_the_stock_init_object_info_schema():
    from dgx_monarch.cli.comfy_ports import _init_schema

    valid = {"DGXMonarchInit": {
        "name": "DGXMonarchInit",
        "category": "DGX Monarch",
        "output": ["DGXM_MESH"],
    }}
    assert _init_schema(valid) is True
    assert _init_schema({}) is False
    assert _init_schema({"DGXMonarchInit": {**valid["DGXMonarchInit"],
                                            "category": "unrelated"}}) is False
    assert _init_schema({**valid, "SomeCatchAll": {}}) is False
