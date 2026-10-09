from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dgx_monarch.cli import lifecycle
from dgx_monarch.cli import setup_services_platform as platform
from dgx_monarch.cli import setup_services_scripts as scripts
from dgx_monarch.cli.lifecycle_generation import GENERATION_FORMAT, GENERATION_MARKER_REL
from dgx_monarch.cli.lifecycle_lock import locked_script
from dgx_monarch.cli.setup_services import ActivitySnapshot, SetupServiceRequest
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256

TOKEN = "s-" + "c" * 16
UPDATE_TOKEN = "u-" + "b" * 12 + "-" + "e" * 16
BOOT_ID = "11111111-2222-3333-4444-555555555555"
SOURCE = (Path(__file__).parents[1] / "src/dgx_monarch").resolve()


def _run(script: str, env: dict[str, str], *, timeout: int = 40) -> subprocess.CompletedProcess[str]:
    proc = env.get("DGXM_TEST_PROC")
    if proc:
        script = script.replace("pathlib.Path('/proc')", f"pathlib.Path({proc!r})")
        script = script.replace("'/proc/net/tcp6'", repr(f"{proc}/net/tcp6"))
        script = script.replace("'/proc/net/tcp'", repr(f"{proc}/net/tcp"))
    return subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True, env=env, timeout=timeout)


def _fake_commands(root: Path) -> Path:
    binary = root / "bin"
    binary.mkdir()
    systemctl = binary / "systemctl"
    systemctl.write_text(
        """#!/bin/sh
unit="$HOME/.config/systemd/user/dgxm-worker.service"
[ "$1" = --user ] && shift
umask 077
generation="$HOME/.local/state/dgx-monarch/worker-generation"
generation_lock="$HOME/.local/state/dgx-monarch/worker-generation.lock"
[ -d "$DGXM_TEST_PROC/123" ] && alive=true || alive=false
case "$1" in
 show)
  [ -z "${DGXM_TEST_SHOW_HOOK:-}" ] || "$DGXM_TEST_SHOW_HOOK"
  if [ -f "$unit" ]; then echo LoadState=loaded; echo FragmentPath="$unit"
  else echo LoadState=not-found; echo FragmentPath=; fi
  state=""; [ -z "${DGXM_TEST_STATE_FILE:-}" ] || [ ! -f "$DGXM_TEST_STATE_FILE" ] || state=$(cat "$DGXM_TEST_STATE_FILE")
  if [ "$state" = activating ]; then echo ActiveState=activating; echo SubState=start-pre; echo MainPID=0
  elif $alive; then echo ActiveState=active; echo SubState=running; echo MainPID=123
  else echo ActiveState=inactive; echo SubState=dead; echo MainPID=0; fi
  if [ "$state" = activating ]; then echo Job=77; else echo Job="${DGXM_TEST_JOB:-0}"; fi;;
 is-enabled)
  if [ ! -f "$unit" ] && [ -n "${DGXM_TEST_IS_ENABLED_RC:-}" ]; then echo "${DGXM_TEST_IS_ENABLED_TEXT:-}"; exit "$DGXM_TEST_IS_ENABLED_RC"
  elif [ -L "$HOME/.config/systemd/user/default.target.wants/dgxm-worker.service" ]; then echo enabled; exit 0
  else echo disabled; exit 1; fi;;
 show-environment|daemon-reload) exit 0;;
 start)
  /usr/bin/flock --nonblock "$generation_lock" /usr/bin/rm -f -- "$generation" || exit 1
  "$DGXM_TEST_HELPER" start
  (sleep .5; "$DGXM_TEST_HELPER" listen) >/dev/null 2>&1 &
  if [ -n "${DGXM_TEST_START_FENCE_READY:-}" ]; then
   (exec 9<> "$generation_lock"; /usr/bin/flock --exclusive 9; touch "$DGXM_TEST_START_FENCE_READY"; sleep 1) >/dev/null 2>&1 &
   while [ ! -f "$DGXM_TEST_START_FENCE_READY" ]; do sleep .01; done
  fi;;
 stop)
  [ -z "${DGXM_TEST_STOP_COUNT:-}" ] || printf x >> "$DGXM_TEST_STOP_COUNT"
  [ -z "${DGXM_TEST_STATE_FILE:-}" ] || rm -f -- "$DGXM_TEST_STATE_FILE"
  if [ -n "${DGXM_TEST_RACE_RESULT:-}" ]; then
   if /usr/bin/flock --nonblock "$generation_lock" /usr/bin/rm -f -- "$generation"; then
    printf won > "$DGXM_TEST_RACE_RESULT"
   else
    printf blocked > "$DGXM_TEST_RACE_RESULT"
   fi
  fi
  [ "${DGXM_TEST_RETAIN_STOP:-}" = 1 ] || "$DGXM_TEST_HELPER" stop
  [ "${DGXM_TEST_STOP_NONZERO:-}" != 1 ] || exit 1
  [ -z "${DGXM_TEST_STOP_DELAY:-}" ] || sleep "$DGXM_TEST_STOP_DELAY";;
 *) exit 1;;
esac
""",
        encoding="utf-8",
    )
    systemctl.chmod(0o700)
    loginctl = binary / "loginctl"
    loginctl.write_text("#!/bin/sh\necho yes\n", encoding="utf-8")
    loginctl.chmod(0o700)
    helper = binary / "fake-proc-state"
    helper.write_text(
        f"""#!{sys.executable}
import os,pathlib,shutil,sys
root=pathlib.Path(os.environ['DGXM_TEST_PROC']); pid=root/'123'; tcp=root/'net/tcp'
if sys.argv[1]=='stop':
 shutil.rmtree(pid,ignore_errors=True); tcp.write_text('header\\n'); raise SystemExit
if sys.argv[1]=='listen':
 port=int(os.environ['DGXM_TEST_ADDRESS'].rsplit(':',1)[1]); tcp.write_text(f'header\\n 0: 0100007F:{{port:04X}} 00000000:0000 0A 0 0 0 0 0 42\\n'); raise SystemExit
(pid/'fd').mkdir(parents=True); (pid/'cmdline').write_bytes(b'python\\0-m\\0dgx_monarch.cli.worker_loop\\0--address\\0'+os.environ['DGXM_TEST_ADDRESS'].encode()+b'\\0')
(pid/'environ').write_bytes(b''); (pid/'fd/7').symlink_to('socket:[42]')
(pid/'stat').write_text('123 (fake) S '+' '.join(['0']*18+['777']))
""",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    return binary


def _generation_path(env: dict[str, str]) -> Path:
    return Path(env["HOME"]) / GENERATION_MARKER_REL


def _generation_payload(token: str, pid: int, birth: int) -> str:
    return f"{GENERATION_FORMAT}\n{token}\n{BOOT_ID}\n{pid}:{birth}\n"


def _prepare(root: Path, *, local: bool) -> tuple[dict[str, str], str, str, Path]:
    home = root / "home"
    home.mkdir(mode=0o700)
    binary = _fake_commands(root)
    candidate = SOURCE
    if local:
        candidate = root / "checkout/dgx_monarch"
        shutil.copytree(SOURCE, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    digest = dgx_source_manifest_sha256(candidate)
    address = "tcp://127.0.0.1:28651"
    proc = root / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text(BOOT_ID + "\n", encoding="ascii")
    (proc / "net/tcp").write_text("header\n", encoding="ascii")
    (proc / "net/tcp6").write_text("header\n", encoding="ascii")
    env = dict(
        os.environ,
        HOME=str(home),
        PATH=f"{binary}:{os.environ['PATH']}",
        DGXM_TEST_HELPER=str(binary / "fake-proc-state"),
        DGXM_TEST_ADDRESS=address,
        DGXM_TEST_PROC=str(proc),
    )
    reserved = _run(scripts.build_reserve_script(sys.executable, TOKEN, 1, digest), env)
    assert reserved.returncode == 0 and reserved.stdout.strip() == "RESERVED"
    package = home / scripts.release_rel(TOKEN, 1) / "site/dgx_monarch"
    if not local:
        shutil.copytree(SOURCE, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        for directory in (package, *tuple(path for path in package.rglob("*") if path.is_dir())):
            directory.chmod(0o700)
    verified = _run(
        scripts.build_verify_stage_script(
            sys.executable, TOKEN, 1, digest, source_root=str(candidate) if local else None
        ),
        env,
    )
    assert verified.returncode == 0 and verified.stdout.strip() == "MATCH"
    return env, digest, address, candidate


def _activate(
    prepared: tuple[dict[str, str], str, str, Path], *, start: bool, local: bool
) -> subprocess.CompletedProcess[str]:
    env, digest, address, candidate = prepared
    unit = "[Unit]\nDescription=test\n[Service]\nExecStart=/bin/true\n"
    return _run(
        scripts.build_activate_script(
            sys.executable,
            TOKEN,
            1,
            digest,
            unit,
            address,
            start_service=start,
            source_site=str(candidate.parent) if local else None,
        ),
        env,
    )


def _stage(root: Path, *, start: bool, local: bool) -> tuple[dict[str, str], str, str, Path]:
    prepared = _prepare(root, local=local)
    env, digest, address, candidate = prepared
    marker = _generation_path(env)
    marker.write_text(_generation_payload(UPDATE_TOKEN, 999, 999), encoding="ascii")
    marker.chmod(0o600)
    published = _activate(prepared, start=start, local=local)
    assert published.returncode == 0 and published.stdout.strip() == "PUBLISHED"
    identity = (123, 777) if start else (0, 0)
    assert marker.read_text(encoding="ascii") == _generation_payload(TOKEN, *identity)
    return env, digest, address, candidate


@pytest.mark.parametrize(("start", "local"), [(False, False), (True, False), (False, True)])
def test_generated_service_transaction_round_trips_exact_owned_state(tmp_path: Path, start: bool, local: bool) -> None:
    env, digest, _address, candidate = _stage(tmp_path, start=start, local=local)
    marker = _generation_path(env)
    probe = _run(scripts.build_probe_activation_script(sys.executable, TOKEN, 1, digest), env)
    assert probe.returncode == 0 and probe.stdout.strip() == "NEW"
    if local:
        link = Path(env["HOME"]) / ".local/share/dgx-monarch/src"
        archive = Path(env["HOME"]) / f".local/state/dgx-monarch/setup/{TOKEN}-1/local-slot"
        assert os.readlink(link) == str(candidate.parent)
        assert not (Path(env["HOME"]) / scripts.release_rel(TOKEN, 1)).exists()
        assert archive.is_dir() and (archive / "setup.json").is_file()
        (candidate / "__init__.py").write_text("# simulated local update\n", encoding="utf-8")
    if not start and not local:
        holder = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, env=env)
        assert holder.stdin is not None
        holder.stdin.write(locked_script(sys.executable, 'touch "$HOME/lock-ready"; sleep 1').encode())
        holder.stdin.close()
        deadline = time.monotonic() + 2
        while not (Path(env["HOME"]) / "lock-ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        blocked = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), env)
        assert blocked.stdout.strip() == "UNKNOWN"
        assert marker.is_file()
        assert holder.wait(timeout=3) == 0
    compensated = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), env)
    assert compensated.returncode == 0 and compensated.stdout.strip() == "COMPENSATED"
    assert not marker.exists()
    if not start and not local:
        slot = Path(env["HOME"]) / scripts.release_rel(TOKEN, 1)
        quarantine = slot.parent / f".cleanup-{TOKEN}-1"
        slot.rename(quarantine)
        (quarantine / "setup.json").unlink()
    cleaned = _run(scripts.build_cleanup_script(sys.executable, TOKEN, 1, digest), env)
    assert cleaned.returncode == 0 and cleaned.stdout.strip() == "CLEANED"
    assert not (Path(env["HOME"]) / scripts.release_rel(TOKEN, 1)).exists()
    if local:
        assert (candidate / "__init__.py").read_text(encoding="utf-8") == "# simulated local update\n"


def test_rc4_not_found_allows_activation_and_owned_cleanup(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path, local=False)
    env, digest, _address, _candidate = prepared
    env.update(DGXM_TEST_IS_ENABLED_RC="4", DGXM_TEST_IS_ENABLED_TEXT="not-found")
    published = _activate(prepared, start=False, local=False)
    assert published.returncode == 0 and published.stdout.strip() == "PUBLISHED"
    compensated = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), env)
    assert compensated.returncode == 0 and compensated.stdout.strip() == "COMPENSATED"
    cleaned = _run(scripts.build_cleanup_script(sys.executable, TOKEN, 1, digest), env)
    assert cleaned.returncode == 0 and cleaned.stdout.strip() == "CLEANED"


def test_activation_refuses_when_stale_generation_cannot_be_replaced(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path, local=False)
    env, _digest, _address, _candidate = prepared
    marker = _generation_path(env)
    marker.mkdir()

    published = _activate(prepared, start=False, local=False)

    home = Path(env["HOME"])
    assert published.returncode != 0
    assert marker.is_dir()
    assert not (home / ".local/share/dgx-monarch/src").exists()
    assert not (home / ".config/systemd/user/dgxm-worker.service").exists()


def test_active_generation_publication_loses_to_a_busy_start_fence(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path, local=False)
    env, _digest, _address, _candidate = prepared
    ready = tmp_path / "start-fence-ready"
    guarded = dict(env, DGXM_TEST_START_FENCE_READY=str(ready))
    prepared = (guarded, *prepared[1:])

    published = _activate(prepared, start=True, local=False)

    home = Path(env["HOME"])
    assert published.returncode != 0 and ready.is_file()
    assert not _generation_path(env).exists()
    assert (Path(env["DGXM_TEST_PROC"]) / "123").is_dir()
    assert (home / ".local/share/dgx-monarch/src").is_symlink()
    assert (home / ".config/systemd/user/dgxm-worker.service").is_file()


@pytest.mark.parametrize("start", [False, True])
def test_compensation_fence_blocks_concurrent_systemd_prestart(tmp_path: Path, start: bool) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=start, local=False)
    raced = tmp_path / "prestart-result"
    guarded = dict(env, DGXM_TEST_RACE_RESULT=str(raced))

    compensated = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), guarded)

    home = Path(env["HOME"])
    assert compensated.stdout.strip() == "COMPENSATED"
    assert raced.read_text(encoding="ascii") == "blocked"
    assert not (Path(env["DGXM_TEST_PROC"]) / "123").exists()
    assert not (home / ".local/share/dgx-monarch/src").exists()
    assert not (home / ".config/systemd/user/dgxm-worker.service").exists()


def test_install_only_compensation_cancels_start_queued_after_stop_intent(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=False, local=False)
    count = tmp_path / "stop-count"
    calls = tmp_path / "show-count"
    state = tmp_path / "manager-state"
    hook = tmp_path / "show-hook"
    hook.write_text(
        "#!/bin/sh\n"
        f'calls={str(calls)!r}\nstate={str(state)!r}\n'
        'value=0; [ ! -f "$calls" ] || value=$(cat "$calls")\n'
        'value=$((value+1)); printf %s "$value" > "$calls"\n'
        '[ "$value" -ne 2 ] || printf activating > "$state"\n',
        encoding="utf-8",
    )
    hook.chmod(0o700)
    guarded = dict(
        env,
        DGXM_TEST_SHOW_HOOK=str(hook),
        DGXM_TEST_STATE_FILE=str(state),
        DGXM_TEST_STOP_COUNT=str(count),
    )

    compensated = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), guarded)

    home = Path(env["HOME"])
    assert compensated.stdout.strip() == "COMPENSATED"
    assert count.read_text(encoding="ascii") == "x"
    assert not _generation_path(env).exists()
    assert not (home / ".local/share/dgx-monarch/src").exists()
    assert not (home / ".config/systemd/user/dgxm-worker.service").exists()


def test_install_only_compensation_cancels_start_already_queued_at_recheck(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=False, local=False)
    count = tmp_path / "stop-count"
    state = tmp_path / "manager-state"
    state.write_text("activating", encoding="ascii")
    guarded = dict(
        env,
        DGXM_TEST_STATE_FILE=str(state),
        DGXM_TEST_STOP_COUNT=str(count),
    )

    compensated = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), guarded)

    home = Path(env["HOME"])
    assert compensated.stdout.strip() == "COMPENSATED"
    assert count.read_text(encoding="ascii") == "x"
    assert not _generation_path(env).exists()
    assert not (home / ".local/share/dgx-monarch/src").exists()
    assert not (home / ".config/systemd/user/dgxm-worker.service").exists()


def test_ambiguous_stop_is_dispatched_once_and_retry_is_readback_only(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=True, local=False)
    count = tmp_path / "stop-count"
    guarded = dict(env, DGXM_TEST_RETAIN_STOP="1", DGXM_TEST_STOP_COUNT=str(count))
    script = scripts.build_compensate_script(sys.executable, TOKEN, 1, digest)
    fast_script = script.replace("deadline=time.monotonic()+25", "deadline=time.monotonic()+.05")

    first = _run(fast_script, guarded)
    second = _run(fast_script, guarded)

    intent = Path(env["HOME"]) / f".local/state/dgx-monarch/setup/{TOKEN}-1/stop-intent.json"
    assert first.stdout.strip() == second.stdout.strip() == "UNKNOWN"
    assert count.read_text(encoding="ascii") == "x" and intent.is_file()
    subprocess.run([env["DGXM_TEST_HELPER"], "stop"], env=env, check=True)
    settled = _run(script, dict(env, DGXM_TEST_STOP_COUNT=str(count)))
    assert settled.stdout.strip() == "COMPENSATED"
    assert count.read_text(encoding="ascii") == "x"


def test_pending_systemd_job_retains_artifacts_until_stable_readback(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=True, local=False)
    count = tmp_path / "stop-count"
    script = scripts.build_compensate_script(sys.executable, TOKEN, 1, digest)
    pending = dict(env, DGXM_TEST_JOB="77", DGXM_TEST_STOP_COUNT=str(count))

    first = _run(script, pending)
    second = _run(script, pending)

    home = Path(env["HOME"])
    assert first.stdout.strip() == second.stdout.strip() == "UNKNOWN"
    assert count.read_text(encoding="ascii") == "x"
    assert (home / ".local/share/dgx-monarch/src").is_symlink()
    assert (home / ".config/systemd/user/dgxm-worker.service").is_file()
    settled = _run(script, dict(env, DGXM_TEST_STOP_COUNT=str(count)))
    assert settled.stdout.strip() == "COMPENSATED"
    assert count.read_text(encoding="ascii") == "x"


@pytest.mark.parametrize("stop_outcome", ["nonzero", "timeout"])
def test_lost_stop_result_settles_only_after_complete_stable_readback(
    tmp_path: Path, stop_outcome: str
) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=True, local=False)
    count = tmp_path / "stop-count"
    guarded = dict(env, DGXM_TEST_STOP_COUNT=str(count))
    script = scripts.build_compensate_script(sys.executable, TOKEN, 1, digest)
    if stop_outcome == "nonzero":
        guarded["DGXM_TEST_STOP_NONZERO"] = "1"
    else:
        guarded["DGXM_TEST_STOP_DELAY"] = ".2"
        script = script.replace(
            "run(['systemctl','--user','stop','dgxm-worker.service'],30)",
            "run(['systemctl','--user','stop','dgxm-worker.service'],.05)",
        )

    compensated = _run(script, guarded)

    assert compensated.stdout.strip() == "COMPENSATED"
    assert count.read_text(encoding="ascii") == "x"


def test_late_generation_unlink_failure_retains_stop_intent_for_safe_retry(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=True, local=False)
    count = tmp_path / "stop-count"
    guarded = dict(env, DGXM_TEST_STOP_COUNT=str(count))
    script = scripts.build_compensate_script(sys.executable, TOKEN, 1, digest)
    failed_script = script.replace(
        " try: generation.unlink()\n except OSError",
        " try: raise OSError\n except OSError",
    )
    assert failed_script != script

    failed = _run(failed_script, guarded)

    home = Path(env["HOME"])
    txn = home / f".local/state/dgx-monarch/setup/{TOKEN}-1"
    assert failed.stdout.strip() == "UNKNOWN"
    assert count.read_text(encoding="ascii") == "x"
    assert _generation_path(env).is_file() and (txn / "stop-intent.json").is_file()
    assert not (home / ".local/share/dgx-monarch/src").exists()
    cleanup = scripts.build_cleanup_script(sys.executable, TOKEN, 1, digest)
    replacement = _generation_payload(TOKEN, 456, 888)
    _generation_path(env).write_text(replacement, encoding="ascii")
    mismatched = _run(cleanup, env)
    assert mismatched.stdout.strip() == "UNKNOWN"
    assert _generation_path(env).read_text(encoding="ascii") == replacement
    assert (txn / "stop-intent.json").is_file() and not (txn / "closed.json").exists()
    _generation_path(env).write_text(_generation_payload(TOKEN, 123, 777), encoding="ascii")
    failed_cleanup = cleanup.replace(
        "  generation.unlink()\n  parent=os.open(generation.parent",
        "  raise OSError\n  parent=os.open(generation.parent",
    )
    assert failed_cleanup != cleanup
    retained = _run(failed_cleanup, env)
    assert retained.stdout.strip() == "RETAINED"
    assert (txn / "stop-intent.json").is_file() and not (txn / "closed.json").exists()

    _generation_path(env).unlink()  # unlink completed but the prior result was lost
    settled = _run(script, guarded)
    assert settled.stdout.strip() == "COMPENSATED"
    assert count.read_text(encoding="ascii") == "x"
    cleaned = _run(cleanup, env)
    assert cleaned.stdout.strip() == "CLEANED" and not txn.exists()


def test_install_only_compensation_refuses_a_later_ordinary_start(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=False, local=False)
    marker = _generation_path(env)
    marker.unlink()  # ordinary lifecycle up invalidates setup authority before start
    helper = env["DGXM_TEST_HELPER"]
    subprocess.run([helper, "start"], env=env, check=True)
    subprocess.run([helper, "listen"], env=env, check=True)

    probe = _run(scripts.build_probe_activation_script(sys.executable, TOKEN, 1, digest), env)
    compensated = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), env)

    home = Path(env["HOME"])
    assert probe.stdout.strip() == "UNKNOWN"
    assert compensated.stdout.strip() == "UNKNOWN"
    assert (Path(env["DGXM_TEST_PROC"]) / "123").is_dir()
    assert (home / ".local/share/dgx-monarch/src").is_symlink()
    assert (home / ".config/systemd/user/dgxm-worker.service").is_file()


def test_replaced_worker_birth_refuses_stale_probe_and_compensation(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=True, local=False)
    probe_script = scripts.build_probe_activation_script(sys.executable, TOKEN, 1, digest)
    assert _run(probe_script, env).stdout.strip() == "NEW"
    process = Path(env["DGXM_TEST_PROC"]) / "123"
    (process / "stat").write_text(
        "123 (replacement) S " + " ".join(["0"] * 18 + ["888"]), encoding="ascii"
    )

    probe = _run(probe_script, env)
    compensated = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), env)

    home = Path(env["HOME"])
    assert probe.stdout.strip() == "UNKNOWN"
    assert compensated.stdout.strip() == "UNKNOWN"
    assert process.is_dir()
    assert (home / ".local/share/dgx-monarch/src").is_symlink()
    assert (home / ".config/systemd/user/dgxm-worker.service").is_file()
    assert _generation_path(env).read_text(encoding="ascii") == _generation_payload(TOKEN, 123, 777)


def test_explicit_other_user_local_target_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "dgx_monarch"
    source.mkdir()
    request = SetupServiceRequest(
        ClusterConfig(
            hosts=(HostConfig("localhost", "tcp://127.0.0.1:26600", ssh_user="definitely-other"),),
            transport_security="trusted_fabric",
        ),
        source,
        "d" * 64,
        True,
        False,
    )
    with pytest.raises(ValueError, match="different SSH user"):
        platform.SystemSetupServiceOps(request, activity_probe=lambda: ActivitySnapshot(False, False, 0))


def test_managed_install_readback_rejects_a_replaced_source_site(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=False, local=False)
    slot = Path(env["HOME"]) / scripts.release_rel(TOKEN, 1)
    site = slot / "site"
    site.rename(slot / "displaced-site")
    site.mkdir(mode=0o700)
    probe = _run(scripts.build_probe_activation_script(sys.executable, TOKEN, 1, digest), env)
    assert probe.stdout.strip() == "UNKNOWN"


def test_install_only_final_bracket_rejects_an_actor_that_appears_mid_probe(tmp_path: Path) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=False, local=False)
    hook = tmp_path / "bin/show-hook"
    hook.write_text(
        f"""#!{sys.executable}
import os,pathlib
root=pathlib.Path(os.environ['DGXM_TEST_PROC']); count=root/'show-count'
value=int(count.read_text())+1 if count.exists() else 1; count.write_text(str(value))
if value==2:
 pid=root/'456'; (pid/'fd').mkdir(parents=True)
 (pid/'cmdline').write_bytes(b'python\\0-m\\0monarch._src.actor.bootstrap_main\\0')
 (pid/'environ').write_bytes(b''); (pid/'stat').write_text('456 (actor) S '+' '.join(['0']*19))
""",
        encoding="utf-8",
    )
    hook.chmod(0o700)
    guarded_env = dict(env, DGXM_TEST_SHOW_HOOK=str(hook))
    probe = _run(scripts.build_probe_activation_script(sys.executable, TOKEN, 1, digest), guarded_env)
    assert probe.stdout.strip() == "UNKNOWN"


def test_status_parser_rejects_noise_around_an_allowlisted_word() -> None:
    result = subprocess.CompletedProcess(["fake"], 0, "sensitive-path\nMATCH\n", "")
    assert platform._word(result, frozenset({"MATCH"})) is None


def test_locked_shell_does_not_delegate_its_flock_to_long_lived_children() -> None:
    payload = locked_script(sys.executable, "nohup /bin/true &")
    assert "pass_fds" not in payload
    assert "set_inheritable(fd,True)" not in payload


def test_pin_install_uses_the_same_full_mutation_lifecycle_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    scripts_seen: list[str] = []
    config = ClusterConfig(
        hosts=(HostConfig("worker", "tcp://10.0.0.2:26600"),),
        transport_security="trusted_fabric",
    )

    def run_host(_config: object, _host: object, script: str, timeout: int = 60) -> object:
        scripts_seen.append(script)
        return subprocess.CompletedProcess([], 0, "PIN_OK\n", "")

    monkeypatch.setattr(lifecycle, "run_on_host", run_host)
    assert lifecycle.ensure_torchmonarch_pin(config, "0.7.0")
    assert len(scripts_seen) == 1
    assert "DGXM_LIFECYCLE_LOCKED_PAYLOAD" in scripts_seen[0]
    assert "pip install --no-deps" in scripts_seen[0]


def test_reserve_exception_removes_only_exact_owned_bootstrap_state(tmp_path: Path) -> None:
    home = tmp_path / "home"
    foreign = home / scripts.release_rel(TOKEN, 1)
    foreign.mkdir(parents=True, mode=0o700)
    home.chmod(0o700)
    env = dict(os.environ, HOME=str(home))
    failed = _run(scripts.build_reserve_script(sys.executable, TOKEN, 1, "d" * 64), env)
    assert failed.returncode != 0 and foreign.exists()
    assert not (home / f".local/state/dgx-monarch/setup/{TOKEN}-1").exists()


@pytest.mark.parametrize(("code", "text", "expected"), [
    (1, "disabled", True), (1, "not-found", True), (4, "not-found", True),
    (4, "", None), (4, "disabled", None), (5, "not-found", None),
])
def test_ownership_rc4_requires_literal_not_found(
    tmp_path: Path, code: int, text: str, expected: bool | None,
) -> None:
    env, _digest, address, _site = _prepare(tmp_path, local=True)
    env.update(DGXM_TEST_IS_ENABLED_RC=str(code), DGXM_TEST_IS_ENABLED_TEXT=text)
    result = _run(scripts.build_ownership_script(sys.executable, address, 1, local_source=True), env)
    assert result.returncode == 0
    payload = json.loads(next(
        line.removeprefix(scripts.STATE_MARKER) for line in result.stdout.splitlines()
        if line.startswith(scripts.STATE_MARKER)
    ))
    assert payload["enablement_absent"] is expected


@pytest.mark.parametrize(("code", "text"), [(4, ""), (4, "disabled"), (5, "not-found")])
def test_ambiguous_absent_unit_never_publishes_or_cleans_a_reserved_slot(
    tmp_path: Path, code: int, text: str,
) -> None:
    prepared = _prepare(tmp_path, local=False)
    env, digest, _address, _candidate = prepared
    env.update(DGXM_TEST_IS_ENABLED_RC=str(code), DGXM_TEST_IS_ENABLED_TEXT=text)
    published = _activate(prepared, start=False, local=False)
    assert published.returncode != 0
    home = Path(env["HOME"])
    assert not (home / ".config/systemd/user/dgxm-worker.service").exists()
    assert not (home / ".local/share/dgx-monarch/src").is_symlink()
    cleaned = _run(scripts.build_cleanup_script(sys.executable, TOKEN, 1, digest), env)
    assert cleaned.stdout.strip() == "RETAINED"
    assert (home / scripts.release_rel(TOKEN, 1)).is_dir()


@pytest.mark.parametrize(("code", "text"), [(4, ""), (4, "disabled"), (5, "not-found")])
def test_compensation_retains_recovery_after_ambiguous_unit_removal(
    tmp_path: Path, code: int, text: str,
) -> None:
    env, digest, _address, _candidate = _stage(tmp_path, start=False, local=False)
    home = Path(env["HOME"])
    live = home / ".local/share/dgx-monarch/src"
    marker = _generation_path(env)
    txn = home / f".local/state/dgx-monarch/setup/{TOKEN}-1"
    link_target = os.readlink(live)
    marker_bytes = marker.read_bytes()
    recovery = {name: (txn / name).read_bytes() for name in (
        "reservation.json", "slot.json", "transaction.json",
    )}
    count = tmp_path / "stop-count"
    guarded = dict(env, DGXM_TEST_IS_ENABLED_RC=str(code), DGXM_TEST_IS_ENABLED_TEXT=text,
                   DGXM_TEST_STOP_COUNT=str(count))
    script = scripts.build_compensate_script(sys.executable, TOKEN, 1, digest)

    result = _run(script, guarded)

    assert result.returncode == 0 and result.stdout.strip() == "UNKNOWN"
    assert not (home / ".config/systemd/user/dgxm-worker.service").exists()
    assert live.is_symlink() and os.readlink(live) == link_target
    assert marker.read_bytes() == marker_bytes
    assert all((txn / name).read_bytes() == data for name, data in recovery.items())
    assert (txn / "stop-intent.json").is_file()
    assert (home / scripts.release_rel(TOKEN, 1)).is_dir()
    again = _run(script, guarded)
    assert again.stdout.strip() == "UNKNOWN"
    assert count.read_text() == "x"
