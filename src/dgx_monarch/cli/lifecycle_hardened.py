"""Isolated systemd launches for installations with explicit dependency paths."""
from __future__ import annotations

from ..config import ClusterConfig, HostConfig
from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX
from .lifecycle_generation import systemd_generation_invalidator
from .systemd_unit import quote, resolve_python

UNSET_ENVIRONMENT = (
    'PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS '
    'LD_ASSUME_KERNEL LD_AUDIT LD_BIND_NOT LD_BIND_NOW LD_DEBUG LD_DEBUG_OUTPUT '
    'LD_DYNAMIC_WEAK LD_HWCAP_MASK LD_LIBRARY_PATH LD_ORIGIN_PATH LD_POINTER_GUARD '
    'LD_PREFER_MAP_32BIT_EXEC LD_PRELOAD LD_PROFILE LD_PROFILE_OUTPUT LD_SHOW_AUXV '
    'LD_TRACE_LOADED_OBJECTS LD_TRACE_PRELINKING LD_USE_LOAD_BIAS LD_VERBOSE LD_WARN '
    'GLIBC_TUNABLES GCONV_PATH LOCPATH NLSPATH'
)


def hardened_worker_unit(
    config: ClusterConfig, host: HostConfig, *, source: str, site_packages: str,
    home: str, user: str, marked: bool = False, update_token: str | None = None,
) -> str:
    """Render an isolated unit, optionally with generation-lock protection.

    Capture the explicit dependency directory from the configured interpreter
    during update preparation. Never infer it from the controller's Python.
    """
    if not site_packages.startswith('/') or ':' in site_packages:
        raise ValueError('site-packages must be an absolute single directory')
    if ':' in source:
        raise ValueError('source must be a single directory')
    def argument(value: str) -> str:
        if value in ('HOME=%h', 'USER=%u', 'LOGNAME=%u'):
            return f'"{value}"'
        return quote(value, preserve_home_specifier=(
            value.startswith('%h/') or value.startswith('PYTHONPATH=%h/')))
    args = ['/usr/bin/env', '-i', f'HOME={home}', f'USER={user}', f'LOGNAME={user}',
            'PATH=/usr/bin:/bin', 'LANG=C.UTF-8', f'PYTHONPATH={source}:{site_packages}',
            'PYTHONNOUSERSITE=1', 'PYTHONSAFEPATH=1', 'PYTHONDONTWRITEBYTECODE=1',
            f'PYTHONPYCACHEPREFIX={SOURCE_ONLY_PYCACHE_PREFIX}', resolve_python(config.python_bin),
            '-S', '-P', '-s', '-B', '-m', 'dgx_monarch.cli.worker_loop', '--address', host.address]
    marker = '# dgxm-managed-unit-v1\n' if marked else ''
    if update_token is not None:
        import re
        if not marked or re.fullmatch(r'u-[0-9a-f]{12}-[0-9a-f]{16}', update_token) is None:
            raise ValueError('invalid marked update token')
        marker = f'# dgxm-update-token={update_token}\n' + marker
    fence = systemd_generation_invalidator() + '\n' if marked else ''
    return f'''{marker}[Unit]
Description=dgx-monarch worker loop
Wants=network-online.target
After=network-online.target

[Service]
{fence}UnsetEnvironment={UNSET_ENVIRONMENT}
ExecStart={' '.join(argument(value) for value in args)}
UMask=0077
Restart=on-failure
RestartSec=3
KillMode=control-group
SendSIGKILL=yes
TimeoutStopSec=30

[Install]
WantedBy=default.target
'''
