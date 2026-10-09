"""Canonical legacy systemd unit rendering and ownership evidence."""

from __future__ import annotations

import re
from pathlib import Path

from ..config import ClusterConfig, HostConfig
from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX
from .lifecycle_generation import invalidate_generation_shell, systemd_generation_invalidator
from .lifecycle_hardened import hardened_worker_unit
from .lifecycle_host import is_local, validated_ssh_identity
from .systemd_unit import quote as systemd_quote
from .systemd_unit import resolve_python

UNIT_NAME = "dgxm-worker.service"
MANAGED_SRC_REL = ".local/share/dgx-monarch/src"
UNIT_OWNERSHIP_MARKER = "# dgxm-managed-unit-v1"
# The first line guided setup writes; remote source sync reads the same prefix.
SETUP_UNIT_MARKER = "# dgxm-setup-token=s-"


def package_src_dir() -> Path:
    """The checkout ``src`` directory used by a colocated Worker service."""
    return Path(__file__).resolve().parent.parent.parent


def unit_exec_args(config: ClusterConfig, host: HostConfig) -> str:
    """Quote the unit's ExecStart arguments; this needs no locality answer."""
    python_uses_home = config.python_bin == "~" or config.python_bin.startswith("~/")
    return " ".join([
        systemd_quote(resolve_python(config.python_bin),
                      preserve_home_specifier=python_uses_home),
        systemd_quote("-m"), systemd_quote("dgx_monarch.cli.worker_loop"),
        systemd_quote("--address"), systemd_quote(host.address),
    ])


def validate_unit_arguments(config: ClusterConfig) -> None:
    """Validate unit install or removal arguments before contacting any host.

    Check Python paths and listener addresses for control characters and validate
    SSH names and users for all hosts before resolving or changing any host.
    """
    for host in config.hosts:
        unit_exec_args(config, host)
        validated_ssh_identity(host)


def systemd_worker_unit(
    config: ClusterConfig, host: HostConfig, *, marked: bool = True,
    managed_source: bool = False, update_token: str | None = None,
    site_packages: str | None = None,
) -> str:
    """Render the exact unit this lifecycle installs for one host."""
    if update_token is not None:
        if re.fullmatch(r"u-[0-9a-f]{12}-[0-9a-f]{16}", update_token) is None:
            raise ValueError("invalid update token")
        if not managed_source or not marked:
            raise ValueError("update token requires a marked managed-source unit")
    if site_packages is not None:
        if not managed_source:
            raise ValueError("explicit dependency path requires managed source")
        return hardened_worker_unit(
            config, host, source=f"%h/{MANAGED_SRC_REL}", site_packages=site_packages,
            home="%h", user="%u", marked=marked, update_token=update_token,
        )
    local = False if managed_source else is_local(host)
    pythonpath = str(package_src_dir()) if local else f"%h/{MANAGED_SRC_REL}"
    exec_args = unit_exec_args(config, host)
    environment = systemd_quote(
        f"PYTHONPATH={pythonpath}", preserve_home_specifier=not local)
    bytecode_environment = " ".join((
        systemd_quote("PYTHONDONTWRITEBYTECODE=1"),
        systemd_quote(f"PYTHONPYCACHEPREFIX={SOURCE_ONLY_PYCACHE_PREFIX}"),
    ))
    marker = f"{UNIT_OWNERSHIP_MARKER}\n" if marked else ""
    if update_token is not None:
        marker = f"# dgxm-update-token={update_token}\n" + marker
    return f"""\
{marker}[Unit]
Description=dgx-monarch worker service
Wants=network-online.target
After=network-online.target

[Service]
{systemd_generation_invalidator()}
Environment={environment}
Environment={bytecode_environment}
ExecStart={exec_args}
UMask=0077
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


# Scan both dgxm-worker.service.d and the dash-prefix dgxm-.service.d in
# user-owned search paths. Include XDG_RUNTIME_DIR: runtime overrides survive
# canonical unit replacement. Exclude /etc and /usr/lib, which this user-level
# lifecycle cannot manage, and service.d, which applies beyond this unit.
# An unset XDG_RUNTIME_DIR falls back to an absent path.
_UNIT_SEARCH_DIRS = ("$HOME/.config/systemd/user", "$HOME/.local/share/systemd/user",
                     "${XDG_RUNTIME_DIR:-/nonexistent}/systemd/user")
_DROPIN_DIRS = (f"{UNIT_NAME}.d", f"{UNIT_NAME.split('-', 1)[0]}-.service.d")


def unit_dropin_scan() -> str:
    """Build shell code that lists the unit's drop-in files in ``DGXM_DROPINS``.

    Drop-ins can replace ExecStart even when the unit body matches the canonical
    bytes. This lifecycle creates none, so every discovered file needs review.
    Ignore missing or empty directories. Report paths relative to home, or full
    paths for runtime drop-ins, to distinguish files across search directories.
    """
    candidates = " ".join(f'"{base}/{name}"'
                          for base in _UNIT_SEARCH_DIRS for name in _DROPIN_DIRS)
    return f"""
UNIT_DIR="{_UNIT_SEARCH_DIRS[0]}"
UNIT="$UNIT_DIR/{UNIT_NAME}"
DGXM_DROPINS=""
for DGXM_DROPIN_DIR in {candidates}; do
  [ -d "$DGXM_DROPIN_DIR" ] || continue
  for DGXM_DROPIN in "$DGXM_DROPIN_DIR"/*; do
    [ -e "$DGXM_DROPIN" ] || continue
    DGXM_DROPINS="$DGXM_DROPINS ${{DGXM_DROPIN#"$HOME"/}}"
  done
done
"""


def unit_ownership_probe(legacy_unit: str, *, allow_setup_token: bool = False) -> str:
    """Accept a non-symlink marked unit or an exact pre-marker unit.

    Pre-marker units must match the current renderer, with an optional trailing
    newline from the legacy heredoc writer. Older, unreproducible bodies do not
    establish ownership.

    ``allow_setup_token`` permits removal of a setup-managed unit. Installation
    must refuse it: replacing the body would drop the token used with the managed
    link to skip source sync, allowing a later sync to overwrite the release slot.
    """
    setup_verdict = "UNIT_OWNED\n" if allow_setup_token else "REFUSED_SETUP_UNIT\n  exit 1\n"
    # Check drop-ins even when the unit is absent: a later install activates
    # leftover overrides. Removal reports them but proceeds to remove the unit.
    dropin_verdict = "" if allow_setup_token else """
if [ -n "$DGXM_DROPINS" ]; then
  echo "REFUSED_FOREIGN_DROPIN$DGXM_DROPINS"
  exit 1
fi
"""
    return f"""{unit_dropin_scan()}{dropin_verdict}
DGXM_UNIT_HEAD=""
DGXM_SETUP_UNIT=no
if [ -f "$UNIT" ] && [ ! -L "$UNIT" ]; then
  DGXM_UNIT_HEAD=$(head -n 1 -- "$UNIT")
  case "$DGXM_UNIT_HEAD" in {SETUP_UNIT_MARKER!r}*|'# dgxm-update-token=u-'*) DGXM_SETUP_UNIT=yes;; esac
fi
if [ -L "$UNIT" ]; then
  echo REFUSED_FOREIGN_UNIT
  exit 1
elif [ ! -e "$UNIT" ]; then
  echo UNIT_ABSENT
elif [ ! -f "$UNIT" ]; then
  echo REFUSED_FOREIGN_UNIT
  exit 1
elif [ "$DGXM_UNIT_HEAD" = {UNIT_OWNERSHIP_MARKER!r} ]; then
  echo UNIT_OWNED
elif [ "$DGXM_SETUP_UNIT" = yes ]; then
  echo {setup_verdict}elif cmp -s -- "$UNIT" - <<'DGXM_PRE_MARKER_UNIT'
{legacy_unit}DGXM_PRE_MARKER_UNIT
then
  echo UNIT_OWNED
elif cmp -s -- "$UNIT" - <<'DGXM_PRE_MARKER_UNIT'
{legacy_unit}
DGXM_PRE_MARKER_UNIT
then
  echo UNIT_OWNED
else
  echo REFUSED_FOREIGN_UNIT
  exit 1
fi
"""


def unit_refusal(stdout: str) -> str:
    """Name the cause the probe printed, so an operator reads more than FAILED."""
    for line in stdout.splitlines():
        if line.startswith("REFUSED_FOREIGN_DROPIN"):
            names = line[len("REFUSED_FOREIGN_DROPIN"):].split()
            return (
                f"REFUSED_FOREIGN_DROPIN (systemd reads these drop-ins for "
                f"{UNIT_NAME} and this lifecycle wrote none of them: "
                + ", ".join(names) + ". A drop-in can replace ExecStart, so "
                "the unit body alone proves nothing about the command systemd "
                "runs. Remove those files, then rerun. No source, generation, "
                "service, or unit change)"
            )
    if "REFUSED_SETUP_UNIT" in stdout:
        return (
            "REFUSED_SETUP_UNIT (setup or verified update wrote this unit; publishing the "
            "canonical body would drop the token its source link needs. Run "
            "`dgxm uninstall`, then remove the managed source link it leaves "
            "behind, before you rerun either installer)"
        )
    return (
        "REFUSED_FOREIGN_UNIT (the existing unit is not this lifecycle's "
        "canonical unit; no source, generation, service, or unit change)"
    )


def unit_publish_script(unit: str) -> str:
    """Publish the reviewed unit by replacing its path, never its symlink target."""
    return f"""
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/{UNIT_NAME}"
mkdir -p -- "$UNIT_DIR"
UNIT_TMP=$(mktemp "$UNIT_DIR/.{UNIT_NAME}.tmp.XXXXXX") || exit 1
trap 'rm -f -- "$UNIT_TMP"' EXIT
chmod 0600 -- "$UNIT_TMP"
cat > "$UNIT_TMP" <<'UNIT'
{unit}UNIT
mv -T -- "$UNIT_TMP" "$UNIT"
"""


def unit_remove_script(legacy_unit: str) -> str:
    """Prove ownership, then disable and remove the unit and read the removal back."""
    return f"""
set -u
{unit_ownership_probe(legacy_unit, allow_setup_token=True)}
if [ -f "$UNIT" ]; then
{invalidate_generation_shell()}
  systemctl --user disable --now {UNIT_NAME} || exit 1
  rm -f -- "$UNIT" || exit 1
  systemctl --user daemon-reload || exit 1
fi
if [ -e "$UNIT" ] || systemctl --user is-active --quiet {UNIT_NAME}; then
  echo FAILED_STILL_PRESENT
  exit 1
fi
if [ -n "$DGXM_DROPINS" ]; then
  echo "LEFT_DROPIN$DGXM_DROPINS"
fi
echo DONE
"""


def unit_removal_leftovers(stdout: str) -> str:
    """Return the drop-in paths left after unit removal, or an empty string.

    A later unit install would activate these overrides again, so report each path
    to the operator.
    """
    for line in stdout.splitlines():
        if line.startswith("LEFT_DROPIN"):
            names = line[len("LEFT_DROPIN"):].split()
            if names:
                return (f"systemd still reads these drop-ins for {UNIT_NAME}: "
                        + ", ".join(names)
                        + "; remove those before the next install, which refuses "
                          "a drop-in it did not write")
    return ""
