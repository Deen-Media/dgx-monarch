"""Strict cluster.toml key and dormant-fabric validation regressions."""
from pathlib import Path

import pytest

from dgx_monarch.config import ClusterConfigError, load_cluster_config

_HOST = '[[hosts]]\nname = "10.0.0.2"\n'


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cluster.toml"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            'transport_security = "trusted_fabric"\n'
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n'
            f"{_HOST}",
            ("top level", "transport_security", "cluster"),
        ),
        (
            '[cluster]\nclient_bnd = "tcp://10.0.0.1:0"\n'
            f"{_HOST}",
            ("[cluster]", "client_bnd", "client_bind"),
        ),
        (
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n'
            '[[hosts]]\nname = "10.0.0.2"\n'
            '[[hosts]]\nname = "10.0.0.3"\ngpu_count = 2\n',
            ("[[hosts]] entry 2", "gpu_count", "gpus"),
        ),
    ],
)
def test_unknown_keys_fail_at_their_exact_config_scope(
    tmp_path: Path, text: str, expected: tuple[str, ...],
):
    with pytest.raises(ClusterConfigError) as raised:
        load_cluster_config(_write(tmp_path, text))

    message = str(raised.value)
    assert all(fragment in message for fragment in expected), message


def test_dormant_fabric_profile_is_validated_before_selection(tmp_path: Path):
    path = _write(
        tmp_path,
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n'
        'fabric_profile = "generic-ib"\n'
        f"{_HOST}"
        '[fabric.future]\nNOT_ALLOWED = "x"\n',
    )

    with pytest.raises(ClusterConfigError, match=r"\[fabric\.future\].*NOT_ALLOWED"):
        load_cluster_config(path)


def test_fabric_scope_accepts_only_profile_tables(tmp_path: Path):
    path = _write(
        tmp_path,
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n'
        f"{_HOST}"
        '[fabric]\nfuture = "not a table"\n',
    )

    with pytest.raises(ClusterConfigError, match=r"fabric entry 'future'.*table"):
        load_cluster_config(path)


def test_valid_dormant_fabric_profile_is_checked_but_not_applied(tmp_path: Path):
    path = _write(
        tmp_path,
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n'
        'fabric_profile = "generic-ib"\n'
        f"{_HOST}"
        '[fabric.future]\nNCCL_DEBUG = "INFO"\nUCX_TLS = "tcp"\n',
    )

    config = load_cluster_config(path)
    assert config.fabric_profile == "generic-ib"
    assert config.fabric_env == {}
    assert config.resolved_fabric_env()["NCCL_DEBUG"] == "WARN"
    assert "UCX_TLS" not in config.resolved_fabric_env()
