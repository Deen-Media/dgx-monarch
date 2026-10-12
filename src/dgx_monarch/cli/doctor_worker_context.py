"""Select the host's owned Worker release for Doctor's environment probe."""
from __future__ import annotations

from ..config import ClusterConfig, HostConfig
from .lifecycle_managed import managed_source_probe


def worker_context(config: ClusterConfig, host: HostConfig) -> str:
    """Keep plain installs unchanged; never fall back from an unverified link.

    The existing source probe validates ownership and the complete generated
    service unit without importing installed code. A managed release shadows
    the configured interpreter's base packages, just as its Worker does. The
    driver's private staged directory must never be forwarded to another host.
    """
    return f'''DGXM_DOCTOR_SOURCE_KIND=$(
{managed_source_probe(config, host)}) || {{ echo FAILED_SOURCE_LAYOUT; exit 1; }}
case "$DGXM_DOCTOR_SOURCE_KIND" in
  MANAGED) export PYTHONPATH="$HOME/.local/share/dgx-monarch/src";;
  PLAIN) ;;
  *) echo FAILED_SOURCE_LAYOUT; exit 1;;
esac
'''
