"""Select the host's owned Worker release for Doctor's environment probe."""
from __future__ import annotations

from ..config import ClusterConfig, HostConfig
from .lifecycle_managed import POLICY_STALE, POLICY_STALE_EXIT, managed_source_probe, policy_stale_detail
from .probe_certainty import FAIL, OK, WARN

POLICY_LINE = "dgxm_worker_policy="


def worker_context(config: ClusterConfig, host: HostConfig) -> str:
    """Keep plain installs unchanged; never fall back from an unverified link.

    The existing source probe validates ownership and the complete generated
    service unit without importing installed code. A managed release shadows
    the configured interpreter's base packages, just as its Worker does. The
    driver's private staged directory must never be forwarded to another host.
    A stale policy authenticates the same release, so Doctor still inspects it
    and reports the verdict on its own line.
    """
    return f'''DGXM_DOCTOR_SOURCE_KIND=$(
{managed_source_probe(config, host)})
case "$?:$DGXM_DOCTOR_SOURCE_KIND" in
  0:MANAGED|{POLICY_STALE_EXIT}:{POLICY_STALE}) export PYTHONPATH="$HOME/.local/share/dgx-monarch/src";;
  0:PLAIN) ;;
  *) echo FAILED_SOURCE_LAYOUT; exit 1;;
esac
echo "{POLICY_LINE}$DGXM_DOCTOR_SOURCE_KIND"
'''


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
