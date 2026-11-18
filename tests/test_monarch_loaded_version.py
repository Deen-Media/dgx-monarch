"""The attach gate and the admission compare the torchmonarch this process loaded.

docs/TROUBLESHOOTING.md #8. A driver reads torchmonarch's version once, when
it first admits the runtime after importing ``monarch``, and keeps that
reading: an install under a running ComfyUI changes the files on disk, not the
code the process runs. Each case here runs a fresh driver process that loads
the real installed ``monarch`` package from a site whose dist-info this test
owns, relabels that dist-info as such an install would, admits the runtime
twice and attaches. The loop record comes from the production writer, the
listener-generation probe runs in bash, and only the dial itself is a stand-in
that records what it was given.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import dgx_monarch

pytest.importorskip("monarch.actor")

SUFFIX = f"_dgxm_loaded_version_test_{os.getpid()}"
# A version no release carries.
OTHER = "0.0.1"

_DRIVER = r'''
import importlib, json, os, socket, subprocess, sys, threading
from pathlib import Path

site, loaded, later, pin, home, order, reinstall = sys.argv[1:8]
info = next(Path(site).glob("torchmonarch-*.dist-info"))


def label(version):
    lines = (info / "METADATA").read_text().split("\n")
    (info / "METADATA").write_text("\n".join(
        "Version: " + version if line.startswith("Version: ") else line for line in lines))
    importlib.invalidate_caches()


def install_under_this_process():
    """What an install under this running process leaves on disk: new metadata,
    and with a reinstall a new binding file where pip unlinked the loaded one."""
    label(later)
    if reinstall == "1":
        binding = Path(sys.modules["monarch._rust_bindings"].__file__)
        binding.unlink()
        binding.write_bytes(b"the reinstalled binding, never loaded here")


label(loaded)
sys.path.insert(0, site)
import monarch.actor

import dgx_monarch
from dgx_monarch import attach_trace, mesh_attach, mesh_factory, worker_proc_id
from dgx_monarch.cli import actor_ledger, lifecycle
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.monarch_capabilities import assert_supported_runtime, loaded_version

dgx_monarch.TORCHMONARCH_PIN = pin
monarch.actor.unhandled_fault_hook = lambda failure: None


def admit():
    try:
        return {"admitted": assert_supported_runtime()}
    except RuntimeError as exc:
        return {"refused": str(exc)}


out = {}
if order == "admit_first":
    out["first"] = admit()
install_under_this_process()
if order == "import_first":  # something else imported monarch; nothing has read it yet
    out["first"] = admit()
out["second"] = admit()
try:
    out["loaded"] = loaded_version()
except RuntimeError as exc:
    out["loaded"] = f"refused: {exc}"

# A Worker loop of this install, as the production writer records it.
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(1)
address = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
proc_id = worker_proc_id.current()
assert actor_ledger.publish_loop(address, proc_id=proc_id, monarch=pin, home=home)


def run_on_host(_config, _host, script, timeout=60, **_kwargs):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=max(timeout, 60), env=dict(os.environ, HOME=home))


class Log:
    def __init__(self):
        self.lines = []

    def record(self, template, *args):
        self.lines.append(template % args if args else template)

    info = warning = error = debug = record


def raise_attach(addresses, exc, **_detail):
    raise RuntimeError(f"attach failed: {exc!r}") from exc


dials = []
lifecycle.run_on_host = run_on_host
worker_proc_id.prime = lambda addresses, **_kwargs: []  # it sends to the address; no loop answers here
log = Log()
runtime = {
    "mesh_factory": mesh_factory, "MeshAttachError": RuntimeError, "log": log,
    "_TRANSPORT_ENABLE_LOCK": threading.Lock(), "_TRANSPORT_POISON": None,
    "_TRANSPORT_BIND": None, "_attach_once": lambda dial: dials.append(list(dial)) or "hosts",
    "_heal_dead_loops": lambda *_a, **_k: None, "_raise_attach": raise_attach,
    "_raise_released_attach": lambda addresses, exc, **_k: raise_attach(addresses, exc),
}
config = ClusterConfig(hosts=(HostConfig(name="h1", address=address),),
                       client_bind="tcp://127.0.0.1:0", auto_heal=False,
                       transport_security="trusted_fabric", python_bin=sys.executable)
try:
    mesh_attach.attach_cluster(runtime, config, None)
    out["attach"] = "attached"
except BaseException as exc:
    out["attach"] = f"{type(exc).__name__}: {exc}"
out.update(served=worker_proc_id.proc_addr(address, proc_id), dials=dials,
           poison=runtime["_TRANSPORT_POISON"],
           close=[fields for fields in (attach_trace.parse(line) or {} for line in log.lines)
                  if fields.get("event") == "close"])
print(json.dumps(out), flush=True)
os._exit(0)
'''


@pytest.fixture
def driver(tmp_path):
    """Run one driver over a site holding the real ``monarch`` and a copy of its dist-info.

    With ``reinstall`` the site's ``monarch`` is a folder of links to the real
    package's entries and a copy of its native binding, so the driver can
    replace that binding as pip would without touching the real install.
    """
    installed = importlib.metadata.distribution("torchmonarch")
    real = Path(installed.locate_file(""))
    info = next(path for path in real.glob("torchmonarch-*.dist-info")
                if importlib.metadata.Distribution.at(path).version == installed.version)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), CUDA_VISIBLE_DEVICES="", MONARCH_LOG_SUFFIX=SUFFIX,
               PYTHONPATH=str(Path(dgx_monarch.__file__).resolve().parents[1]))

    def run(loaded: str, later: str, *, order: str = "admit_first", reinstall: bool = False) -> dict:
        site = tmp_path / f"site_{order}_{int(reinstall)}"
        site.mkdir()
        if reinstall:
            (site / "monarch").mkdir()
            for entry in (real / "monarch").iterdir():
                if entry.name.startswith("_rust_bindings.") and entry.suffix == ".so":
                    shutil.copy2(entry, site / "monarch" / entry.name)
                else:
                    (site / "monarch" / entry.name).symlink_to(entry)
        else:
            (site / "monarch").symlink_to(real / "monarch")
        shutil.copytree(info, site / info.name)
        # -B: the package is the real one, so no bytecode may be written beside it.
        result = subprocess.run(
            [sys.executable, "-B", "-c", _DRIVER, str(site), loaded, later,
             installed.version, str(home), order, "1" if reinstall else "0"],
            env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr[-3000:]
        return json.loads(result.stdout.strip().splitlines()[-1])

    yield SimpleRun(run, installed.version)
    log_dir = Path("/tmp") / pwd.getpwuid(os.getuid()).pw_name  # noqa: S108
    for path in log_dir.glob(f"monarch_log_*{SUFFIX}*.log"):
        path.unlink(missing_ok=True)


class SimpleRun:
    def __init__(self, run, pin: str) -> None:
        self.run, self.pin = run, pin


def test_a_pin_installed_under_a_running_driver_does_not_admit_it(driver):
    """The driver loaded another torchmonarch, was refused, and then had the
    pin installed under it. Reading the disk again would admit it and dial a
    loop on the pin, which waits out two config-push budgets and poisons the
    session. It must keep refusing, ask for a restart, and refuse before the
    attach dials, naming the version it loaded."""
    out = driver.run(OTHER, driver.pin)
    assert "refused" in out["second"], out["second"]  # the disk now reads the pin
    assert out["loaded"] == OTHER
    assert out["dials"] == []
    for admission in (out["first"], out["second"]):
        assert f"this process loaded torchmonarch {OTHER}" in admission["refused"]
        assert "restart ComfyUI" in admission["refused"]
    assert out["attach"].startswith("WorkerProcIdError: no attach ran: h1 (")
    assert f"runs torchmonarch {driver.pin} and this driver runs {OTHER}" in out["attach"]
    assert out["poison"] is None
    assert [(close["outcome"], close["stage"]) for close in out["close"]] == [("refused", "proc_id")]


def test_an_install_under_a_running_driver_on_the_pin_changes_nothing(driver):
    """The driver loaded the pin; the disk then names another version. The
    process still runs the pin, so it admits and dials the loop's id."""
    out = driver.run(driver.pin, OTHER)
    assert out["first"] == out["second"] == {"admitted": driver.pin}
    assert out["loaded"] == driver.pin
    assert out["attach"] == "attached"
    assert out["dials"] == [[out["served"]]]
    assert out["poison"] is None
    assert [close["outcome"] for close in out["close"]] == ["attached"]


def test_a_reinstall_before_the_first_reading_refuses_whatever_the_disk_reads(driver):
    """Something else imported monarch, then torchmonarch was reinstalled,
    pin and all, before the pack's first reading. The disk reads the pin, so a
    disk reading would admit and dial a loop on the pin with another release
    loaded. The binding this process mapped is no longer the file on disk, so
    the reading refuses, asks for a restart, and nothing dials."""
    out = driver.run(OTHER, driver.pin, order="import_first", reinstall=True)
    assert "refused" in out["first"], out["first"]
    for admission in (out["first"], out["second"]):
        assert "torchmonarch was reinstalled after this process loaded it" in admission["refused"]
        assert "restart ComfyUI" in admission["refused"]
    assert out["loaded"].startswith("refused: Monarch's native binding in this process")
    assert out["dials"] == [] and out["poison"] is None and out["close"] == []
    assert out["attach"].startswith("RuntimeError: Monarch's native binding in this process")


def test_a_first_reading_after_a_foreign_import_with_nothing_reinstalled_admits(driver):
    """The mapped binding reads back as the file on disk, so a first reading
    taken late, with nothing reinstalled, admits and dials as before."""
    out = driver.run(driver.pin, driver.pin, order="import_first")
    assert out["first"] == out["second"] == {"admitted": driver.pin}
    assert out["loaded"] == driver.pin and out["attach"] == "attached"
    assert out["dials"] == [[out["served"]]]
