"""Select the host's owned Worker release for Doctor's environment probe."""
from __future__ import annotations

import subprocess

from .. import __version__
from ..config import ClusterConfig, HostConfig
from . import release_slot
from .lifecycle_managed import (
    POLICY_STALE,
    POLICY_STALE_EXIT,
    managed_source_probe,
    policy_stale_detail,
    probe_cause,
    probe_failure,
    stderr_tail,
)
from .probe_certainty import FAIL, OK, WARN

POLICY_LINE = "dgxm_worker_policy="
PROBE_EXIT_LINE = "dgxm_worker_probe_exit="


def worker_context(config: ClusterConfig, host: HostConfig) -> str:
    """Keep plain installs unchanged; never fall back from an unverified link.

    The existing source probe validates ownership and the complete generated
    service unit without importing installed code. A managed release shadows
    the configured interpreter's base packages, just as its Worker does. The
    driver's private staged directory must never be forwarded to another host.
    A stale policy authenticates the same release, so Doctor still inspects it
    and reports the verdict on its own line. Any other result stops the check
    before the environment probe runs, after printing the probe's output and
    its exit code on a last line, so Doctor can name the cause as restart and
    a syncing up do. The last line reports where the source link points and
    the package version in a release slot (``release_slot``).
    """
    return f'''DGXM_DOCTOR_SOURCE_KIND=$(
{managed_source_probe(config, host)})
DGXM_DOCTOR_SOURCE_EXIT=$?
case "$DGXM_DOCTOR_SOURCE_EXIT:$DGXM_DOCTOR_SOURCE_KIND" in
  0:MANAGED|{POLICY_STALE_EXIT}:{POLICY_STALE}) export PYTHONPATH="$HOME/.local/share/dgx-monarch/src";;
  0:PLAIN) ;;
  *) printf '%s\\n' "$DGXM_DOCTOR_SOURCE_KIND"; echo "{PROBE_EXIT_LINE}$DGXM_DOCTOR_SOURCE_EXIT"; exit 1;;
esac
echo "{POLICY_LINE}$DGXM_DOCTOR_SOURCE_KIND"
{release_slot.PROBE_COMMAND}
'''


def env_failure_row(host_name: str, result: subprocess.CompletedProcess[str]) -> tuple[str, str, str]:
    """The env row for a host check that exited non-zero.

    When the source probe stopped the check, the row names its cause through
    the classifier and wording that ``dgxm restart`` and a syncing ``dgxm up``
    use. Doctor runs the probe without the lifecycle lock, so an exit without a
    verdict reads FAILED_SOURCE_PROBE, never FAILED_LIFECYCLE_LOCK. Doctor is
    also stricter in one case: an exit 0 with unexpected output, which
    ``probe_kind`` reads as PLAIN, fails here as FAILED_SOURCE_PROBE.
    ``dgxm up --no-sync`` and a verified update's generation start check the
    source through ``managed_start_guard`` instead, which still names every
    other result FAILED_SOURCE_LAYOUT. A failure after the probe passed,
    such as the environment probe itself, keeps the generic line.
    """
    name = f"{host_name} env"
    lines = result.stdout.strip().splitlines()
    code = lines[-1][len(PROBE_EXIT_LINE):] if lines and lines[-1].startswith(PROBE_EXIT_LINE) else ""
    if code.isdigit():
        probe = subprocess.CompletedProcess(result.args, int(code), "\n".join(lines[:-1]), result.stderr)
        tail = stderr_tail(result.stderr)
        return FAIL, name, (f"{probe_cause(probe_failure(probe, locked=False), probe)}; "
                            "the environment check did not run" + (f"; stderr: {tail}" if tail else ""))
    detail = (result.stderr or result.stdout or "no output").strip().splitlines()[-1]
    return FAIL, name, f"remote check exited {result.returncode}: {detail}"


def worker_policy_row(host_name: str, lines: list[str]) -> tuple[str, str, str]:
    """Report whether a managed Worker unit still carries cluster.toml's policy."""
    kind = next((line[len(POLICY_LINE):] for line in lines if line.startswith(POLICY_LINE)), "")
    name = f"{host_name} worker policy"
    if kind == "MANAGED":
        return OK, name, "the managed Worker unit carries the policy in cluster.toml"
    if kind == "PLAIN":
        return OK, name, "plain source; no managed release unit to compare"
    if kind == POLICY_STALE:
        return FAIL, name, f"STALE: {policy_stale_detail()}"
    return WARN, name, "unobserved: the source probe printed no policy verdict"


def worker_release_row(host_name: str, lines: list[str]) -> tuple[str, str, str]:
    """Compare the package version in a host's release slot with the driver's.

    Plain ``dgxm update`` moves the driver and never a slot, so a slot left on
    an earlier version fails here even when its torchmonarch matches the pin.
    """
    name = f"{host_name} worker release version"
    verdict = release_slot.parse(lines)
    if verdict is None:
        return WARN, name, "unobserved: the release probe printed no verdict"
    kind, token, version = verdict
    if kind == "PLAIN":
        return OK, name, "plain source directory, not a release slot; nothing to compare"
    if kind == "OUTSIDE":
        return OK, name, "the source link points outside releases/, so it is not a release slot"
    if version == "UNREADABLE":
        return WARN, name, f"unobserved: release slot {token} has no readable package version"
    if version != __version__:
        return FAIL, name, (
            f"{host_name} runs dgx-monarch {version} from release slot {token}, but the driver "
            f"runs {__version__}. Plain `dgxm update` never changes a release slot; repair the "
            f"split with {release_slot.TROUBLESHOOTING}, then update only with `dgxm update --verify`")
    return OK, name, f"release slot {token} carries dgx-monarch {version}, as the driver does"


def worker_source_rows(host_name: str, lines: list[str]) -> list[tuple[str, str, str]]:
    """The policy row and the release version row, both read from the source probe."""
    return [worker_policy_row(host_name, lines), worker_release_row(host_name, lines)]
