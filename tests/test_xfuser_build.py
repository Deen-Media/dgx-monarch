"""The xFuser builds this release runs on: the requirement, doctor's rows, Worker setup and the Gate.

Stock xFuser 0.8.0 and the patched 0.7.0+dgxm.npuimport1 that 1.4.0 built run;
nothing else does. The requirement cannot say so by itself (a local label never
takes part in matching), so each test here goes through the place that holds
the exact list in production: doctor's driver row and per-host probe, Worker
setup before it imports xFuser, and the Gate's runtime identity.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.specifiers import SpecifierSet

from dgx_monarch import gate_runtime, worker_runtime, xfuser_build
from dgx_monarch.actor import worker_env
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.cli import doctor, xfuser_row
from dgx_monarch.cli.probe_certainty import FAIL, OK, WARN
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from gate_orchestration_helpers import _clear_process_gate_verdicts  # noqa: F401  # autouse fixture import.

REPO = Path(__file__).resolve().parents[1]
PYTHON = "/opt/comfy/bin/python"


@pytest.fixture(autouse=True)
def _no_pinned_build(monkeypatch):
    """Each test starts as a process that has not run Worker setup, and leaves none pinned or held."""
    from dgx_monarch.adapters import attention_capability

    monkeypatch.setattr(xfuser_build, "_PINNED", None)
    monkeypatch.setattr(attention_capability, "_UNBUILT", {})


FA_MISSING = ("xFuserLongContextAttention was asked for AttnType.FA, but flash-attn is not "
              "available here. Leave attn_type unset to use a kernel that is.")


def _xfuser_that_will_not_build_fa(monkeypatch):
    """Stock 0.8.0's constructor on a host without flash-attn, whatever xFuser this process has."""
    import types

    class Refuses:
        def __init__(self, *, use_sync=False, attn_type=None):
            if attn_type is not None and attn_type.name in ("FA", "FA3"):
                raise ImportError(FA_MISSING)
            self.attn_type = attn_type

    module = types.ModuleType("xfuser.core.long_ctx_attention")
    module.xFuserLongContextAttention = Refuses
    monkeypatch.setitem(sys.modules, "xfuser.core.long_ctx_attention", module)


@pytest.mark.parametrize("topology", [{"cfg": 2}, {"dp": 2}, {"ulysses": 1}])
def test_fa_without_flash_attn_sets_up_where_the_kernel_never_runs(monkeypatch, topology):
    """cfg2, dp and single build as on 0.7.0; the kernel's absence waits for a render that runs it."""
    from dgx_monarch.actor.attention_dispatch import _AttentionDispatch

    _xfuser_that_will_not_build_fa(monkeypatch)
    dispatch = _AttentionDispatch()
    dispatch.configure("FA", True, topology=topology, world=2, setup_generation=1)
    assert dispatch.kernel == "FA"


def test_a_render_that_would_run_fa_refuses_in_the_packs_words(monkeypatch):
    from dgx_monarch.actor.attention_dispatch import _AttentionDispatch
    from dgx_monarch.adapters import attention_capability

    _xfuser_that_will_not_build_fa(monkeypatch)
    dispatch = _AttentionDispatch()
    dispatch.configure("FA", True, topology={"ulysses": 2, "ring": 1}, world=2, setup_generation=1)
    # The sequence-parallel load's capability check, before any collective.
    with pytest.raises(UnsupportedModelError) as raised:
        attention_capability.assert_kernel_capability("FA", "chroma", 128, probe=lambda *a, **k: None)
    text = str(raised.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class == RefusalClass.PHYSICS and not tag.waivable
    assert "FA cannot run the sharded attention on this Worker" in text
    assert "flash-attn 2.7.0 or newer" in text
    # xFuser's reason, without its advice: the Init node offers no unset attn_type.
    assert ("it reported: xFuserLongContextAttention was asked for AttnType.FA, but flash-attn "
            "is not available here. ") in text
    assert "Leave attn_type unset" not in text and ".." not in text
    assert "Run this workflow on a cfg or single topology instead" in text
    assert "docs/TROUBLESHOOTING.md #104" in text
    # The stand-in refuses the same way if anything ever calls it.
    with pytest.raises(UnsupportedModelError, match="FA cannot run the sharded attention"):
        dispatch(None, None, None)


def test_any_other_kernel_that_will_not_build_still_fails_setup(monkeypatch):
    from dgx_monarch.adapters import attention_capability

    with pytest.raises(ImportError, match="sage"):
        attention_capability.hold_unbuilt("SAGE_FP8", ImportError("no sage here"))
    assert attention_capability._UNBUILT == {}


def test_stock_0_8_0_without_flash_attn_builds_fa_for_a_render_that_never_runs_it(monkeypatch):
    """Through the real xFuser 0.8.0 package check: needs stock 0.8.0 importable.

    CI installs 0.8.0; on a box that still holds the patched build, put the
    unpacked 0.8.0 wheel first on PYTHONPATH to run this.
    """
    import importlib

    attn_layer = importlib.import_module("xfuser.core.long_ctx_attention.hybrid.attn_layer")
    if not hasattr(attn_layer, "_resolve_attn_type"):
        pytest.skip("this xFuser builds FA without checking for flash-attn (the patched 0.7.0)")
    from dgx_monarch.adapters import attention_capability
    from dgx_monarch.adapters.base import make_usp_attention

    if attn_layer.AttnType is None:
        # xFuser imports yunchang's kernel enum only where CUDA is; a CPU runner gets it here.
        from yunchang.kernels import AttnType

        monkeypatch.setattr(attn_layer, "AttnType", AttnType)
    monkeypatch.setitem(attn_layer.env_info, "has_flash_attn", False)
    with pytest.raises(ImportError, match="flash-attn is not available"):
        attn_layer.xFuserLongContextAttention(attn_type=attn_layer.AttnType.FA)
    stand_in = make_usp_attention("FA", True)
    assert callable(stand_in)
    with pytest.raises(UnsupportedModelError, match=r"flash-attn 2\.7\.0 or newer"):
        attention_capability.raise_if_unbuilt("FA", 128)


def test_the_requirement_is_one_string_in_pyproject_requirements_and_the_module():
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    lines = [line.strip() for line in (REPO / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert [d for d in pyproject if d.startswith("xfuser")] == [xfuser_build.REQUIREMENT]
    assert [d for d in lines if d.startswith("xfuser")] == [xfuser_build.REQUIREMENT]


def test_the_requirement_admits_both_builds_and_only_code_refuses_stock_0_7_0():
    spec = SpecifierSet(xfuser_build.REQUIREMENT.removeprefix("xfuser"))
    for build in xfuser_build.ALLOWED:
        assert spec.contains(build), build
    for build in ("0.6.9", "0.8.0.post1", "0.8.1", "0.9.0"):
        assert not spec.contains(build), build
    # No range can leave stock 0.7.0 out while it admits the patched build,
    # so the exact list in code is what refuses it.
    assert spec.contains("0.7.0")
    assert xfuser_build.verdict("0.7.0", PYTHON)[0] == xfuser_build.FAIL


@pytest.mark.parametrize(("installed", "forced"), [
    ("0.8.0+local1", True), ("0.8.0+dgxm.rebuild", True), ("0.8+local1", True),
    ("0.8.0.0+local1", True), ("0.7.0+dgxm.npuimport1", False), ("0.7.0", False),
    ("0.8.1", False), ("0.8.0rc1", False), ("0.8.0.post1", False), (xfuser_build.MISSING, False)])
def test_the_move_line_installs_stock_0_8_0_by_pips_own_matching(installed, forced):
    """Pip counts a local build of 0.8.0 as meeting ==0.8.0, so a plain line there does nothing.

    PEP 440 pads the release with zeros, so ``0.8+local1`` is such a build too.
    """
    from pip._vendor.packaging.specifiers import SpecifierSet as PipSpecifierSet

    line = xfuser_build.move_command(PYTHON, installed)
    assert line.endswith(" xfuser==0.8.0") and line.startswith(f"{PYTHON} -m pip install ")
    already = installed != xfuser_build.MISSING and PipSpecifierSet("==0.8.0").contains(
        installed, prereleases=True)
    assert already is forced
    if forced:
        # Only a forced reinstall moves it, and only xFuser itself: torch stays.
        assert "--force-reinstall --no-deps" in line
    else:
        assert "--force-reinstall" not in line and "--upgrade-strategy only-if-needed" in line


@pytest.mark.parametrize(("build", "state"), [
    ("0.8", xfuser_build.OK), ("0.8.0.0", xfuser_build.OK),
    ("0.7.0+dgxm-npuimport1", xfuser_build.WARN), ("0.7.0+DGXM.npuimport1", xfuser_build.WARN),
    ("0.7", xfuser_build.FAIL), ("0.8+local1", xfuser_build.FAIL), ("not a version", xfuser_build.FAIL)])
def test_builds_compare_as_pep_440_versions_not_as_strings(build, state):
    """Equal by PEP 440 is the same build: zero padding and the local label's case and separators."""
    found, detail = xfuser_build.verdict(build, PYTHON)
    assert found == state
    assert (xfuser_build.setup_refusal(build, PYTHON) is None) is (state != xfuser_build.FAIL)
    if build == "0.7":
        assert "imports an NPU ring helper that released yunchang lacks" in detail


def test_a_local_build_of_0_8_0_reads_fail_with_the_forced_line():
    state, detail = xfuser_build.verdict("0.8.0+local1", PYTHON)
    assert state == xfuser_build.FAIL
    assert f"`{PYTHON} -m pip install --force-reinstall --no-deps xfuser==0.8.0`" in detail


@pytest.mark.parametrize(("build", "state", "words"), [
    ("0.8.0", xfuser_build.OK, "xFuser 0.8.0"),
    ("0.7.0+dgxm.npuimport1", xfuser_build.WARN, "the patched build 1.4.0 installed"),
    ("0.7.0", xfuser_build.FAIL, "imports an NPU ring helper that released yunchang lacks"),
    ("0.8.1", xfuser_build.FAIL, "no render has been checked on xFuser 0.8.1"),
    (xfuser_build.MISSING, xfuser_build.FAIL, "xFuser is not installed"),
])
def test_each_build_reads_its_verdict_with_cause_and_remedy(build, state, words):
    found, detail = xfuser_build.verdict(build, PYTHON)
    assert found == state
    assert words in detail
    if state != xfuser_build.OK:
        assert f"`{PYTHON} -m pip install --upgrade-strategy only-if-needed xfuser==0.8.0`" in detail
        assert "docs/TROUBLESHOOTING.md #104" in detail


def test_installed_version_reads_metadata_without_importing_xfuser(monkeypatch):
    monkeypatch.delitem(sys.modules, "xfuser", raising=False)
    import importlib.metadata

    try:
        expected = importlib.metadata.distribution("xfuser").version
    except importlib.metadata.PackageNotFoundError:
        expected = xfuser_build.MISSING
    assert xfuser_build.installed_version() == expected
    assert "xfuser" not in sys.modules


def _worker():
    return SimpleNamespace(_setup_key=None, _setup_cleanup_failed=False)


def _env():
    return {"setup_generation": 1, "world": 2, "rank": 1, "gpus_per_host": 1,
            "local_gpu_index": 0, "fabric_env": {}, "comfy_dir": "/comfy",
            "topology": {"ulysses": 2}, "worker_args": {}}


@pytest.mark.parametrize("build", ["0.7.0", "0.8.1", xfuser_build.MISSING])
def test_worker_setup_refuses_any_other_build_before_it_changes_anything(monkeypatch, build):
    monkeypatch.setattr(xfuser_build, "installed_version", lambda: build)
    # Setup's first state change: reaching it means the refusal did not happen first.
    monkeypatch.setattr(worker_env, "_setup_key",
                        lambda *_a: pytest.fail("setup went past the xFuser check"))
    worker = _worker()
    with pytest.raises(UnsupportedModelError) as raised:
        worker_env.setup_impl(worker, _env())
    text = str(raised.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class == RefusalClass.PHYSICS and not tag.waivable
    assert f"this Worker's Python ({sys.executable})" in text
    assert "Use xFuser 0.8.0 on that box" in text
    assert "in local mode, where the actors run inside ComfyUI's own mesh" in text
    assert (f"`{sys.executable} -m pip install --upgrade-strategy only-if-needed "
            "xfuser==0.8.0`") in text
    assert "docs/TROUBLESHOOTING.md #104" in text
    assert worker._setup_key is None and worker._setup_cleanup_failed is False


@pytest.mark.parametrize("build", xfuser_build.ALLOWED)
def test_worker_setup_goes_on_with_either_allowed_build(monkeypatch, build):
    monkeypatch.setattr(xfuser_build, "installed_version", lambda: build)

    class Past(Exception):
        pass

    def past(*_a):
        raise Past

    monkeypatch.setattr(worker_env, "_setup_key", past)
    with pytest.raises(Past):
        worker_env.setup_impl(_worker(), _env())


def test_an_actor_keeps_reporting_the_build_its_setup_checked(monkeypatch):
    """A pip install while the actor runs moves the disk, not the module the actor imported."""
    import importlib.metadata

    on_disk = {"version": "0.7.0+dgxm.npuimport1"}
    monkeypatch.setattr(importlib.metadata, "distribution",
                        lambda name: SimpleNamespace(version=on_disk["version"]))
    from dgx_monarch.actor import xfuser_setup

    xfuser_setup.refuse_unsupported_xfuser()
    on_disk["version"] = "0.8.0"
    assert xfuser_build.installed_version() == "0.7.0+dgxm.npuimport1"
    assert gate_runtime.runtime_build_identity()["xfuser"] == "0.7.0+dgxm.npuimport1"
    # A process that never ran setup, such as the driver, follows its disk.
    monkeypatch.setattr(xfuser_build, "_PINNED", None)
    assert xfuser_build.installed_version() == "0.8.0"


def test_the_driver_row_reads_this_interpreter(monkeypatch):
    monkeypatch.setattr(xfuser_build, "installed_version", lambda: "0.7.0+dgxm.npuimport1")
    status, name, detail = xfuser_row.driver_row()
    assert (status, name) == (WARN, "xfuser")
    assert f"`{sys.executable} -m pip install" in detail
    monkeypatch.setattr(xfuser_build, "installed_version", lambda: "0.7.0")
    assert xfuser_row.driver_row()[0] == FAIL


def _fake_xfuser(tmp_path: Path, version: str) -> Path:
    site = tmp_path / f"site-{version}"
    meta = site / f"xfuser-{version}.dist-info"
    meta.mkdir(parents=True)
    (meta / "METADATA").write_text(f"Metadata-Version: 2.1\nName: xfuser\nVersion: {version}\n")
    return site


@pytest.mark.parametrize(("version", "status"), [
    ("0.8.0", OK), ("0.7.0+dgxm.npuimport1", WARN), ("0.7.0", FAIL)])
def test_the_host_probe_prints_the_line_doctor_selects_and_the_row_reads_it(tmp_path, version, status):
    """Run the payload as doctor does, under a Python that sees one installed xFuser."""
    env = dict(os.environ, PYTHONPATH=str(_fake_xfuser(tmp_path, version)))
    result = subprocess.run([sys.executable, "-S", "-"], input=xfuser_row.PROBE_SOURCE, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    printed = result.stdout.strip().splitlines()
    assert printed == [f"{xfuser_row.LINE_PREFIX}{version}"]
    assert xfuser_row.select(["torchmonarch=0.8.0 torch=2", *printed]) == version
    assert xfuser_row.host_row("sibling", version, PYTHON)[:2] == (status, "sibling xfuser")
    assert "\nEOF" not in xfuser_row.PROBE_SOURCE


def test_a_host_that_prints_no_xfuser_line_reads_unobserved():
    assert xfuser_row.select(["torchmonarch=0.8.0 torch=2"]) == xfuser_row.UNKNOWN
    status, _name, detail = xfuser_row.host_row("sibling", xfuser_row.UNKNOWN, PYTHON)
    assert status == WARN and detail.startswith("unobserved")


def test_the_match_row_warns_on_two_builds_and_stays_out_when_no_host_answered():
    assert xfuser_row.match_row("0.8.0", {}) is None
    assert xfuser_row.match_row("0.8.0", {"head": xfuser_row.UNKNOWN}) is None
    assert xfuser_row.match_row("0.8.0", {"head": "0.8.0", "sibling": "0.8.0"}) == (
        OK, "xfuser build match", "0.8.0")
    status, name, detail = xfuser_row.match_row(
        "0.8.0", {"head": "0.8.0", "sibling": "0.7.0+dgxm.npuimport1"})
    assert (status, name) == (WARN, "xfuser build match")
    assert "driver=0.8.0, head=0.8.0, sibling=0.7.0+dgxm.npuimport1" in detail


def _doctor(monkeypatch, capsys, remote_by_host, driver_build):
    from test_doctor_hostname_row import _stub_driver
    from test_doctor_nccl_row import _cluster

    _stub_driver(monkeypatch, "10.0.0.1")
    monkeypatch.setattr(xfuser_build, "installed_version", lambda: driver_build)
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(doctor, "_tcp_probe", lambda *_a, **_kw: False)
    monkeypatch.setattr(doctor, "passive_worker_health",
                        lambda *_a, **_kw: {"running": True, "listening": True, "healthy": True})
    scripts: list[str] = []

    def fake(_config, host, script, **_kw):
        scripts.append(script)
        return subprocess.CompletedProcess([], 0, remote_by_host[host.name], "")

    monkeypatch.setattr(doctor, "run_on_host", fake)
    ok = doctor.run_doctor(_cluster())
    return ok, capsys.readouterr().out, scripts


def test_doctor_sends_the_probe_and_fails_a_host_on_stock_0_7_0(monkeypatch, capsys):
    from test_doctor_hostname_row import _REMOTE_OK

    remote = {"head": _REMOTE_OK + "xfuser_build=0.8.0\n",
              "worker": _REMOTE_OK + "xfuser_build=0.7.0\n"}
    ok, output, scripts = _doctor(monkeypatch, capsys, remote, "0.8.0")
    assert ok is False
    assert "[ ok ] xfuser: xFuser 0.8.0" in output
    assert "[ ok ] head xfuser: xFuser 0.8.0" in output
    assert "[FAIL] worker xfuser: xFuser 0.7.0: this release runs on xFuser 0.8.0" in output
    assert "[WARN] xfuser build match: two builds: driver=0.8.0, head=0.8.0, worker=0.7.0" in output
    env_scripts = [script for script in scripts if "<<'EOF'" in script]
    assert len(env_scripts) == 2
    assert all(xfuser_row.PROBE_SOURCE in script for script in env_scripts)


def test_doctor_warns_but_passes_on_the_patched_build_everywhere(monkeypatch, capsys):
    from test_doctor_hostname_row import _REMOTE_OK

    line = "xfuser_build=0.7.0+dgxm.npuimport1\n"
    ok, output, _ = _doctor(monkeypatch, capsys, {"head": _REMOTE_OK + line,
                                                  "worker": _REMOTE_OK + line},
                            "0.7.0+dgxm.npuimport1")
    assert ok is True
    assert "[WARN] xfuser: xFuser 0.7.0+dgxm.npuimport1, the patched build 1.4.0 installed" in output
    assert "[WARN] worker xfuser:" in output
    # The metadata refresh is the driver's step alone (INSTALL item 30): only its row names it.
    printed = [line.strip() for line in output.splitlines()]
    rows = {line.split(":", 1)[0]: line for line in printed if line.startswith("[WARN] ")}
    refresh = "first refresh the driver's dgx-monarch metadata"
    assert refresh in rows["[WARN] xfuser"]
    assert refresh not in rows["[WARN] head xfuser"] and refresh not in rows["[WARN] worker xfuser"]
    assert output.count(refresh) == 1
    assert "[ ok ] xfuser build match: 0.7.0+dgxm.npuimport1" in output


def test_the_gate_runtime_identity_names_the_xfuser_build(monkeypatch):
    monkeypatch.setattr(xfuser_build, "installed_version", lambda: "0.8.0")
    identity = gate_runtime.runtime_build_identity()
    assert identity["xfuser"] == "0.8.0"
    assert set(identity) == {"torch", "cuda", "xfuser"}


DRIVER = {"torch": "2.15.0", "cuda": "13.2", "xfuser": "0.8.0"}


def _pair_handle():
    return SimpleNamespace(
        config=SimpleNamespace(
            worker_args={}, source="/configs/cluster.toml",
            hosts=(SimpleNamespace(name="head-host", address="10.0.0.1"),
                   SimpleNamespace(name="sibling-host", address="10.0.0.2"))),
        config_fingerprint="config-a", world=2, n_hosts=2, gpus_per_host=1)


def test_a_pair_on_two_xfuser_builds_holds_no_gate_authority(monkeypatch):
    from dgx_monarch import box_names

    monkeypatch.setattr(box_names, "_is_head", lambda name: name == "head-host")
    patched = dict(DRIVER, xfuser="0.7.0+dgxm.npuimport1")
    text = worker_runtime.mismatch(_pair_handle(), [{"rank": 0, "runtime": dict(DRIVER)},
                                                    {"rank": 1, "runtime": patched}], DRIVER)
    assert text is not None
    assert "the driver runs torch 2.15.0 with CUDA 13.2 and xFuser 0.8.0" in text
    assert "runs torch 2.15.0 with CUDA 13.2 and xFuser 0.7.0+dgxm.npuimport1" in text
    assert worker_runtime.mismatch(_pair_handle(), [{"rank": 0, "runtime": dict(DRIVER)},
                                                    {"rank": 1, "runtime": dict(DRIVER)}],
                                   DRIVER) is None


def test_one_version_spelled_two_ways_is_admitted_on_each_box_but_refuses_at_the_gate(monkeypatch):
    """Admission compares by PEP 440; the Gate's identity names the build exactly, on purpose."""
    from dgx_monarch import box_names

    monkeypatch.setattr(box_names, "_is_head", lambda name: name == "head-host")
    padded = dict(DRIVER, xfuser="0.8")
    for spelling in ("0.8.0", "0.8"):
        assert xfuser_build.setup_refusal(spelling, PYTHON) is None
        assert xfuser_build.verdict(spelling, PYTHON)[0] == xfuser_build.OK
    text = worker_runtime.mismatch(_pair_handle(), [{"rank": 0, "runtime": dict(DRIVER)},
                                                    {"rank": 1, "runtime": padded}], DRIVER)
    assert text is not None
    assert re.search(r"runs torch 2\.15\.0 with CUDA 13\.2 and xFuser 0\.8(?!\.?\d)", text)
    monkeypatch.setattr(gate_runtime, "runtime_build_identity", lambda: dict(DRIVER))
    handle = _pair_handle()
    handle.worker_capabilities = [{"runtime": dict(DRIVER)}, {"runtime": padded}]
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch):
        worker_runtime.cohort_runtime(handle)
    with pytest.raises(RuntimeError, match=r"and the Gate grant names .*xFuser 0\.8(?!\.?\d)"):
        worker_runtime.assert_grant_runtime({"runtime": padded}, DRIVER)
    status, _name, detail = xfuser_row.match_row("0.8.0", {"head": "0.8.0", "sibling": "0.8"})
    assert status == WARN and "sibling=0.8." in detail


def test_with_no_pep_440_parser_builds_compare_as_exact_strings_and_nothing_else_is_admitted(
        monkeypatch):
    """Neither ``packaging`` nor pip's copy imports: fall back to the exact list, never admit."""
    monkeypatch.setitem(sys.modules, "packaging", None)
    monkeypatch.setitem(sys.modules, "pip._vendor.packaging", None)
    assert xfuser_build._pep440() is None
    for build in xfuser_build.ALLOWED:
        assert xfuser_build.setup_refusal(build, PYTHON) is None
    assert xfuser_build.verdict("0.8.0", PYTHON)[0] == xfuser_build.OK
    assert xfuser_build.verdict("0.7.0+dgxm.npuimport1", PYTHON)[0] == xfuser_build.WARN
    for other in ("0.8", "0.8.0.0", "0.7.0+dgxm-npuimport1", "0.7.0", "0.8.1", "not a version",
                  xfuser_build.MISSING):
        assert xfuser_build.setup_refusal(other, PYTHON) is not None, other
        assert xfuser_build.verdict(other, PYTHON)[0] == xfuser_build.FAIL, other
    assert "--force-reinstall --no-deps" in xfuser_build.move_command(PYTHON, "0.8.0+local1")
    for plain in ("0.7.0", xfuser_build.MISSING, "0.8+local1"):
        assert "--upgrade-strategy only-if-needed" in xfuser_build.move_command(PYTHON, plain)


def test_a_box_moved_on_disk_but_not_restarted_is_told_to_restart_its_worker(monkeypatch):
    """The driver reads its disk; an actor reports the build it imported until it restarts."""
    from dgx_monarch import box_names

    monkeypatch.setattr(box_names, "_is_head", lambda name: name == "head-host")
    monkeypatch.setattr(gate_runtime, "runtime_build_identity", lambda: dict(DRIVER))
    handle = _pair_handle()
    pinned = dict(DRIVER, xfuser="0.7.0+dgxm.npuimport1")
    handle.worker_capabilities = [{"runtime": dict(DRIVER)}, {"runtime": pinned}]
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch) as raised:
        worker_runtime.cohort_runtime(handle)
    text = str(raised.value)
    assert "or restart its Worker if that box already holds the build on disk" in text
    assert "in local mode, reset the attached mesh or restart ComfyUI" in text
    assert "Use one torch and CUDA build on every box" in text


def test_a_worker_refuses_a_grant_proven_under_another_xfuser():
    patched = dict(DRIVER, xfuser="0.7.0+dgxm.npuimport1")
    with pytest.raises(RuntimeError, match=r"this Worker runs torch 2\.15\.0 with CUDA 13\.2 and "
                                           r"xFuser 0\.8\.0, and the Gate grant names .*0\.7\.0\+dgxm"):
        worker_runtime.assert_grant_runtime({"runtime": patched}, DRIVER)


def test_an_xfuser_move_makes_the_pass_read_stale_and_the_notice_names_it(tmp_path, monkeypatch):
    import dgx_monarch.gate_ledger as ledger_mod
    from dgx_monarch.gate_ledger import GateLedger, gate_verdict_token
    from test_gate_runtime import _context

    monkeypatch.setattr(ledger_mod.runtime_provenance, "cached_dgx_source_manifest_sha256",
                        lambda: "a" * 64)
    led = GateLedger(str(tmp_path))
    patched = _context(monkeypatch, dict(DRIVER, xfuser="0.7.0+dgxm.npuimport1"))
    led.record("k", "artifact", "commit", "PASS", context=patched)
    moved = _context(monkeypatch, dict(DRIVER))
    assert led.lookup("k", "artifact", "commit", moved) == "stale"
    token = gate_verdict_token("k", "artifact", "commit", moved)
    assert gate_runtime.torch_change(str(tmp_path), token) == (
        "proven under torch 2.15.0 with CUDA 13.2 and xFuser 0.7.0+dgxm.npuimport1, not under "
        "this driver's torch 2.15.0 with CUDA 13.2 and xFuser 0.8.0")
