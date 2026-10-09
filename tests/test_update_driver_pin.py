from __future__ import annotations

import json
import shutil
import stat
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from dgx_monarch.cli.update_driver_pin import (
    DriverPinError,
    DriverPinTransition,
    site_payload_digest,
)
from dgx_monarch.cli.update_transaction import Certainty


def _make_wheel(
    directory: Path,
    pin: str,
    *,
    metadata_name: str = "torchmonarch",
    metadata_version: str | None = None,
    extra_member: str | None = None,
    symlink_member: bool = False,
) -> Path:
    wheel = directory / f"torchmonarch-{pin}-py3-none-any.whl"
    dist_info = f"torchmonarch-{pin}.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torchmonarch/__init__.py", f'VERSION = "{pin}"\n')
        archive.writestr("torchmonarch/runtime.py", f'PIN = "{pin}"\n')
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {metadata_name}\nVersion: {metadata_version or pin}\n",
        )
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\n")
        archive.writestr(
            f"{dist_info}/RECORD",
            "torchmonarch/__init__.py,,\ntorchmonarch/runtime.py,,\n"
            f"{dist_info}/METADATA,,\n{dist_info}/WHEEL,,\n{dist_info}/RECORD,,\n",
        )
        if extra_member is not None:
            archive.writestr(extra_member, b"unsafe")
        if symlink_member:
            member = zipfile.ZipInfo("torchmonarch/link")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, "runtime.py")
    return wheel


def _metadata_version(site: Path) -> str | None:
    candidates = list(site.glob("*.dist-info/METADATA"))
    if len(candidates) != 1:
        return None
    for line in candidates[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ")
    return None


def _clear_install(site: Path) -> None:
    package = site / "torchmonarch"
    if package.is_dir():
        shutil.rmtree(package)
    for metadata in site.glob("torchmonarch-*.dist-info"):
        shutil.rmtree(metadata)


def _extract_wheel(wheel: Path, site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            parts = member.filename.rstrip("/").split("/")
            if parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] == "scripts":
                    continue
                parts = parts[2:]
            destination = site.joinpath(*parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))


class FakeRunner:
    def __init__(self, wheels: Mapping[str, Path], live_site: Path) -> None:
        self.wheels = dict(wheels)
        self.live_site = live_site
        self.calls: list[tuple[list[str], Path | None, Mapping[str, str] | None]] = []
        self.live_actions: list[str] = []
        self.live_installs = 0

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path | None = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, timeout
        args = list(argv)
        self.calls.append((args, None if cwd is None else Path(cwd), env))
        if "download" in args:
            pin = args[-1].split("==", 1)[1]
            wheel = self.wheels.get(pin)
            if wheel is None:
                return subprocess.CompletedProcess(args, 1, "", "missing")
            destination = Path(args[args.index("--dest") + 1]) / wheel.name
            shutil.copy2(wheel, destination)
            return subprocess.CompletedProcess(args, 0, "", "")
        if "install" in args:
            wheel = Path(args[-1])
            if "--target" in args:
                _extract_wheel(wheel, Path(args[args.index("--target") + 1]))
                return subprocess.CompletedProcess(args, 0, "", "")
            return self._live_install(args, wheel)
        if "-c" in args:
            version = _metadata_version(self.live_site)
            if version is None:
                return subprocess.CompletedProcess(args, 3, "", "missing")
            payload = json.dumps({"root": str(self.live_site), "version": version})
            return subprocess.CompletedProcess(args, 0, payload, "")
        raise AssertionError(f"unexpected command: {args}")

    def _live_install(self, args: list[str], wheel: Path) -> subprocess.CompletedProcess[str]:
        self.live_installs += 1
        action = self.live_actions.pop(0) if self.live_actions else "normal"
        if action == "timeout":
            raise subprocess.TimeoutExpired(args, 30)
        if action in ("normal", "target_error"):
            _clear_install(self.live_site)
            _extract_wheel(wheel, self.live_site)
            return subprocess.CompletedProcess(args, action == "target_error", "", "")
        if action == "partial":
            _clear_install(self.live_site)
            with zipfile.ZipFile(wheel) as archive:
                metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
                destination = self.live_site / metadata
                destination.parent.mkdir(parents=True)
                destination.write_bytes(archive.read(metadata))
                package = self.live_site / "torchmonarch"
                package.mkdir()
                (package / "__init__.py").write_bytes(archive.read("torchmonarch/__init__.py"))
            return subprocess.CompletedProcess(args, 1, "", "partial")
        if action == "unchanged_ok":
            return subprocess.CompletedProcess(args, 0, "", "")
        if action == "unchanged_error":
            return subprocess.CompletedProcess(args, 1, "", "failed")
        raise AssertionError(f"unknown live action: {action}")


@pytest.fixture
def transition_fixture(tmp_path: Path) -> tuple[DriverPinTransition, FakeRunner, Path, Path, Path]:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    prior_wheel = _make_wheel(wheel_dir, "0.6.0")
    target_wheel = _make_wheel(wheel_dir, "0.7.0")
    live_site = tmp_path / "live"
    _extract_wheel(prior_wheel, live_site)
    runner = FakeRunner({"0.6.0": prior_wheel, "0.7.0": target_wheel}, live_site)
    transition = DriverPinTransition(python_bin="/venv/bin/python", command_runner=runner)
    release = tmp_path / "release"
    transition.prepare(release, "0.6.0", "0.7.0")
    return transition, runner, release, prior_wheel, target_wheel


def test_prepare_stages_both_exact_wheels_and_private_sites_offline(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
) -> None:
    transition, runner, release, _prior, _target = transition_fixture
    assert transition.prepared
    assert transition.target_site is not None and transition.target_site.is_dir()
    assert transition.target_matches_site(transition.target_site)
    assert stat.S_IMODE(release.stat().st_mode) & 0o077 == 0
    downloads = [args for args, _cwd, _env in runner.calls if "download" in args]
    staged = [args for args, _cwd, _env in runner.calls if "install" in args and "--target" in args]
    assert len(downloads) == len(staged) == 2
    assert all("--no-deps" in args and "--only-binary=:all:" in args for args in downloads)
    assert all("--no-index" in args and "--no-deps" in args for args in staged)
    assert transition.matches_prior()
    probe = next(call for call in runner.calls if "-c" in call[0])
    assert probe[1] == release / "driver-pin"
    assert probe[2] is not None and "PYTHONPATH" not in probe[2] and "PYTHONHOME" not in probe[2]


def test_target_release_site_must_match_the_cached_target_artifact(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
    tmp_path: Path,
) -> None:
    transition, _runner, _release, _prior, _target = transition_fixture
    assert transition.target_site is not None
    candidate = tmp_path / "candidate"
    shutil.copytree(transition.target_site, candidate)
    assert transition.target_matches_site(candidate)
    (candidate / "torchmonarch" / "stale.py").write_text("stale\n", encoding="utf-8")
    assert not transition.target_matches_site(candidate)


def test_explicit_site_payload_digest_detects_remote_copy_drift(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
) -> None:
    transition, _runner, _release, _prior, _target = transition_fixture
    assert transition.target_site is not None
    before = site_payload_digest(transition.target_site, "0.7.0")
    (transition.target_site / "torchmonarch" / "runtime.py").write_text(
        "PIN = 'tampered'\n", encoding="utf-8"
    )
    assert site_payload_digest(transition.target_site, "0.7.0") != before


@pytest.mark.parametrize(
    ("member", "symlink_member"),
    [
        ("../escape", False),
        ("/absolute", False),
        ("torchmonarch\\escape", False),
        ("torchmonarch-0.7.0.data/data/outside", False),
        (None, True),
    ],
)
def test_prepare_rejects_unsafe_or_unverifiable_wheel_members(
    tmp_path: Path, member: str | None, symlink_member: bool,
) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    prior = _make_wheel(wheels, "0.6.0")
    target = _make_wheel(wheels, "0.7.0", extra_member=member, symlink_member=symlink_member)
    live = tmp_path / "live"
    _extract_wheel(prior, live)
    runner = FakeRunner({"0.6.0": prior, "0.7.0": target}, live)
    release = tmp_path / "release"
    with pytest.raises(DriverPinError):
        DriverPinTransition(command_runner=runner).prepare(release, "0.6.0", "0.7.0")
    assert not (release / "driver-pin").exists()
    assert runner.live_installs == 0


@pytest.mark.parametrize(
    ("name", "version"), [("another-project", "0.7.0"), ("torchmonarch", "0.7.1")],
)
def test_prepare_rejects_wrong_wheel_metadata(tmp_path: Path, name: str, version: str) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    prior = _make_wheel(wheels, "0.6.0")
    target = _make_wheel(wheels, "0.7.0", metadata_name=name, metadata_version=version)
    live = tmp_path / "live"
    _extract_wheel(prior, live)
    runner = FakeRunner({"0.6.0": prior, "0.7.0": target}, live)
    with pytest.raises(DriverPinError):
        DriverPinTransition(command_runner=runner).prepare(tmp_path / "release", "0.6.0", "0.7.0")


def test_prepare_rejects_non_private_or_symlink_release_roots(tmp_path: Path) -> None:
    open_root = tmp_path / "open"
    open_root.mkdir(mode=0o755)
    link_root = tmp_path / "link"
    link_root.symlink_to(open_root, target_is_directory=True)
    for root in (open_root, link_root):
        with pytest.raises(DriverPinError):
            DriverPinTransition(command_runner=FakeRunner({}, tmp_path / "live")).prepare(
                root, "0.6.0", "0.7.0",
            )


@pytest.mark.parametrize("ordinary_primary", [False, True])
def test_prepare_cleanup_preserves_strongest_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ordinary_primary: bool,
) -> None:
    primary: BaseException = RuntimeError("stage") if ordinary_primary else KeyboardInterrupt()
    cleanup = KeyboardInterrupt() if ordinary_primary else SystemExit(130)
    expected = cleanup if ordinary_primary else primary
    transition = DriverPinTransition(command_runner=FakeRunner({}, tmp_path / "live"))
    def fail_prepare(*_args: object) -> object:
        raise primary
    def fail_cleanup(*_args: object) -> None:
        raise cleanup
    monkeypatch.setattr(transition, "_prepare_artifact", fail_prepare)
    monkeypatch.setattr(DriverPinTransition, "_remove_stage", staticmethod(fail_cleanup))

    with pytest.raises(BaseException) as raised:
        transition.prepare(tmp_path / "release", "0.6.0", "0.7.0")

    assert raised.value is expected


def test_matches_requires_exact_payload_but_ignores_pyc(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
) -> None:
    transition, runner, _release, _prior, _target = transition_fixture
    cache = runner.live_site / "torchmonarch" / "__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-312.pyc").write_bytes(b"generated")
    assert transition.matches_prior()
    (runner.live_site / "torchmonarch" / "stale.py").write_text("stale = True\n", encoding="utf-8")
    assert not transition.matches_prior()


@pytest.mark.parametrize("damage", ["missing", "modified", "symlink"])
def test_matches_rejects_partial_or_redirected_payload(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path], damage: str,
) -> None:
    transition, runner, _release, _prior, _target = transition_fixture
    runtime = runner.live_site / "torchmonarch" / "runtime.py"
    if damage == "missing":
        runtime.unlink()
    elif damage == "modified":
        runtime.write_text("PIN = 'other'\n", encoding="utf-8")
    else:
        runtime.unlink()
        runtime.symlink_to(runner.live_site / "torchmonarch" / "__init__.py")
    assert not transition.matches_prior()
    assert not transition.matches_target()


def test_promote_succeeds_only_after_exact_offline_target_install(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
) -> None:
    transition, runner, _release, _prior, _target = transition_fixture
    assert transition.promote().certainty == Certainty.SUCCEEDED
    assert transition.matches_target() and not transition.matches_prior()
    live_command = [args for args, _cwd, _env in runner.calls if "--force-reinstall" in args][-1]
    assert "--no-index" in live_command and "--no-deps" in live_command
    assert "download" not in live_command


@pytest.mark.parametrize(
    ("action", "certainty"),
    [
        ("unchanged_error", Certainty.FAILED),
        ("unchanged_ok", Certainty.UNKNOWN),
        ("target_error", Certainty.UNKNOWN),
        ("partial", Certainty.UNKNOWN),
        ("timeout", Certainty.UNKNOWN),
    ],
)
def test_promote_classifies_command_and_exact_postcondition(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
    action: str,
    certainty: Certainty,
) -> None:
    transition, runner, _release, _prior, _target = transition_fixture
    runner.live_actions.append(action)
    assert transition.promote().certainty == certainty


def test_promote_refuses_non_prior_or_tampered_staging_without_mutation(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
) -> None:
    transition, runner, _release, _prior, _target = transition_fixture
    (runner.live_site / "torchmonarch" / "stale.py").write_text("stale\n", encoding="utf-8")
    assert transition.promote().certainty == Certainty.UNKNOWN
    assert runner.live_installs == 0
    (runner.live_site / "torchmonarch" / "stale.py").unlink()
    assert transition.target_site is not None
    (transition.target_site / "torchmonarch" / "runtime.py").write_text("tampered\n", encoding="utf-8")
    assert transition.promote().certainty == Certainty.UNKNOWN
    assert runner.live_installs == 0


def test_restore_prior_is_offline_and_requires_exact_target_precondition(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
) -> None:
    transition, runner, _release, _prior, _target = transition_fixture
    assert transition.restore_prior().certainty == Certainty.SUCCEEDED
    assert runner.live_installs == 0
    assert transition.promote().certainty == Certainty.SUCCEEDED
    assert transition.restore_prior().certainty == Certainty.SUCCEEDED
    assert transition.matches_prior()
    live_commands = [args for args, _cwd, _env in runner.calls if "--force-reinstall" in args]
    assert len(live_commands) == 2
    assert all("--no-index" in args and "--no-deps" in args for args in live_commands)


def test_restore_partial_result_is_unknown(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
) -> None:
    transition, runner, _release, _prior, _target = transition_fixture
    assert transition.promote().certainty == Certainty.SUCCEEDED
    runner.live_actions.append("partial")
    assert transition.restore_prior().certainty == Certainty.UNKNOWN
    assert not transition.matches_prior() and not transition.matches_target()


def test_restore_refuses_ambiguous_live_state_without_mutation(
    transition_fixture: tuple[DriverPinTransition, FakeRunner, Path, Path, Path],
) -> None:
    transition, runner, _release, _prior, _target = transition_fixture
    runtime = runner.live_site / "torchmonarch" / "runtime.py"
    runtime.write_text("partial\n", encoding="utf-8")
    assert transition.restore_prior().certainty == Certainty.UNKNOWN
    assert runner.live_installs == 0
