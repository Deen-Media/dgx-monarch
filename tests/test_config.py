"""cluster.toml round trip, refusals, local config and fabric profile resolution."""
from pathlib import Path

import pytest

from dgx_monarch import config as config_module
from dgx_monarch.config import (
    FABRIC_PROFILES,
    ClusterConfigError,
    find_config_path,
    format_tcp_address,
    load_cluster_config,
    local_config,
    render_cluster_toml,
)


def test_render_and_load_roundtrip(tmp_path: Path):
    text = render_cluster_toml(
        hosts=[("sparkA", "192.0.2.11", 1), ("sparkB", "192.0.2.12", 1)],
        client_ip="192.0.2.11",
        fabric_profile="dgx-spark-pair",
        master_addr="192.0.2.11",
        ssh_key="~/.ssh/id_ed25519",
        python_bin="~/monarch-env/bin/python",
        transport_security="trusted_fabric",
    )
    path = tmp_path / "cluster.toml"
    path.write_text(text)

    config = load_cluster_config(path)
    assert config.world_size == 2
    assert config.client_bind == "tcp://192.0.2.11:0"
    assert config.hosts[0].address == "tcp://192.0.2.11:26600"
    assert config.hosts[1].name == "sparkB"
    assert config.resolved_master_addr() == "192.0.2.11"
    assert config.python_bin == "~/monarch-env/bin/python"
    assert config.transport_security == "trusted_fabric"

    env = config.resolved_fabric_env()
    assert env["NCCL_IB_GID_INDEX"] == "3"
    assert env["NCCL_NET_GDR_LEVEL"] == "SYS"
    # The FSDP-killer must never ship in a profile.
    assert "NCCL_PROTO" not in env


def test_fabric_override(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text(
        """
[cluster]
client_bind = "tcp://10.0.0.1:0"
fabric_profile = "generic-roce"

[[hosts]]
name = "n1"
address = "tcp://10.0.0.1:26600"

[fabric.generic-roce]
NCCL_SOCKET_IFNAME = "eth7"
"""
    )
    config = load_cluster_config(path)
    env = config.resolved_fabric_env()
    assert env["NCCL_SOCKET_IFNAME"] == "eth7"
    assert env["NCCL_IB_GID_INDEX"] == "3"  # profile base survives override


def test_fabric_boolean_overrides_use_native_environment_spellings(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n'
        'fabric_profile = "custom"\n\n'
        '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.1:26600"\n\n'
        '[fabric.custom]\nNCCL_IB_DISABLE = true\nUCX_MEMTYPE_CACHE = false\n'
    )

    assert load_cluster_config(path).resolved_fabric_env() == {
        "NCCL_IB_DISABLE": "1",
        "UCX_MEMTYPE_CACHE": "0",
    }


def test_master_addr_derived_from_first_host(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text(
        """
[cluster]
client_bind = "tcp://10.0.0.1:0"

[[hosts]]
name = "n1"
address = "tcp://10.0.0.9:26600"
"""
    )
    config = load_cluster_config(path)
    assert config.resolved_master_addr() == "10.0.0.9"


def test_local_config_is_zero_setup(monkeypatch):
    monkeypatch.setattr(config_module, "find_config_path", lambda *a, **k: None)
    config = local_config()
    assert config.hosts == ()
    assert config.resolved_fabric_env() == {}
    assert config.worker_args == {}


def test_local_config_carries_worker_args(tmp_path: Path, monkeypatch):
    # A local worker is an actor proc, so the operator's [worker_args] must
    # reach it. On 2026-08-12 a local wall-capture leg that lost them rendered
    # zero-copy instead of exercising the pinned-staging refusal it set up.
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
        '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.1:26600"\n\n'
        "[worker_args]\ndisable_pinned_memory = false\nreserve_vram_gb = 3.0\n"
    )
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(path))

    config = local_config()
    assert config.worker_args == {
        "disable_pinned_memory": False, "reserve_vram_gb": 3.0}
    # Cluster facts stay out: a local mesh has none.
    assert config.hosts == ()
    assert config.source == ""


def test_local_config_prefers_an_explicitly_named_config(
    tmp_path: Path, monkeypatch,
):
    # get_mesh resolves the operator's config_path first and hands it over, so
    # naming a file in local mode reads that file's table, not the search order's.
    named = tmp_path / "named.toml"
    named.write_text("[worker_args]\nswap_verify = 7\n")
    found = tmp_path / "cluster.toml"
    found.write_text("[worker_args]\nswap_verify = 1\n")
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(found))

    assert local_config(named).worker_args == {"swap_verify": 7}
    assert local_config().worker_args == {"swap_verify": 1}


def test_local_config_takes_no_worker_args_when_the_table_is_absent(
    tmp_path: Path, monkeypatch,
):
    path = tmp_path / "cluster.toml"
    path.write_text('[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n')
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(path))

    assert local_config().worker_args == {}


def test_local_config_refuses_a_malformed_worker_args_table(
    tmp_path: Path, monkeypatch,
):
    path = tmp_path / "cluster.toml"
    path.write_text("[worker_args]\nreserve_vram_gb = \"lots\"\n")
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(path))

    with pytest.raises(ClusterConfigError, match="worker_args"):
        local_config()


def test_local_config_refuses_invalid_toml(tmp_path: Path, monkeypatch):
    path = tmp_path / "cluster.toml"
    path.write_text("[cluster\n")
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(path))

    with pytest.raises(ClusterConfigError, match="invalid TOML"):
        local_config()


def test_all_profiles_free_of_proto_ll():
    for name, env in FABRIC_PROFILES.items():
        assert env.get("NCCL_PROTO", "").upper() != "LL", name


# Each row writes this cluster.toml and expects load_cluster_config(path) to
# raise ClusterConfigError matching the fragment. Each id is the hyphenated name
# of the test the row replaced, plus a suffix where that test took several values.
@pytest.mark.parametrize(
    ("toml", "fragment"),
    [
        pytest.param(
            # DESIGN §5.1: worker binds are explicit fabric IPs, never hostnames.
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n[[hosts]]\nname = "sparkb"\n',
            "address",
            id="hostname-derived-address-is-refused",
        ),
        pytest.param(
            '[cluster]\nfabric_profile = "dgx_spark_pair"\n\n'
            '[[hosts]]\nname = "10.0.0.9"\n',
            "fabric_profile",
            id="unknown-fabric-profile-without-table-is-refused",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.1:26600"\n\n'
            '[[hosts]]\nname = "n2"\naddress = "tcp://10.0.0.2:26600"\n',
            r"single-node.*2 hosts",
            id="multi-host-config-refuses-single-node-fabric-default",
        ),
        pytest.param(
            '[[hosts]]\naddress = "tcp://10.0.0.9:26600"\n',
            "name",
            id="missing-name-is-refused",
        ),
        pytest.param(
            # The loopback trap: an explicit but hostname-based worker address
            # must be rejected at parse, not stored verbatim.
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
            '[[hosts]]\nname = "sparkb"\naddress = "tcp://sparkb:26600"\n',
            "fabric IP",
            id="explicit-hostname-address-is-refused",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://myhost:0"\n\n[[hosts]]\nname = "10.0.0.9"\n',
            "client_bind",
            id="hostname-client-bind-is-refused",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "udp://10.0.0.1:0"\n\n[[hosts]]\nname = "10.0.0.9"\n',
            "tcp://",
            id="non-tcp-bind-is-refused",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://10.0.0.1"\n\n[[hosts]]\nname = "10.0.0.9"\n',
            "port",
            id="portless-bind-is-refused",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n',
            "hosts",
            id="empty-hosts-is-refused",
        ),
        pytest.param(
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.9:0"\n',
            "nonzero",
            id="worker-port-must-be-nonzero",
        ),
        pytest.param(
            '[[hosts]]\nname = "n1"\naddress = "tcp://user@10.0.0.9:26600"\n',
            "user-info",
            id="tcp-bind-rejects-user-info",
        ),
        pytest.param(
            '[cluster]\ntransport_security = "tls"\n\n[[hosts]]\nname = "10.0.0.9"\n',
            "trusted_fabric",
            id="transport-security-enum-is-validated",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.9:26600"\ngpus = 0\n',
            "positive integer",
            id="gpu-count-must-be-a-positive-integer-0",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.9:26600"\ngpus = -1\n',
            "positive integer",
            id="gpu-count-must-be-a-positive-integer-negative",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.9:26600"\ngpus = true\n',
            "positive integer",
            id="gpu-count-must-be-a-positive-integer-true",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.9:26600"\ngpus = "2"\n',
            "positive integer",
            id="gpu-count-must-be-a-positive-integer-quoted-2",
        ),
        pytest.param(
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.9:26600/path"\n',
            "host and port",
            id="tcp-bind-rejects-non-endpoint-components-path",
        ),
        pytest.param(
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.9:26600?query=1"\n',
            "host and port",
            id="tcp-bind-rejects-non-endpoint-components-query",
        ),
        pytest.param(
            '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.9:26600#fragment"\n',
            "host and port",
            id="tcp-bind-rejects-non-endpoint-components-fragment",
        ),
        pytest.param(
            '[cluster]\nnccl_master_port = 0\n\n[[hosts]]\nname = "10.0.0.9"\n',
            "nccl_master_port",
            id="master-port-type-and-range-are-validated-0",
        ),
        pytest.param(
            '[cluster]\nnccl_master_port = 65536\n\n[[hosts]]\nname = "10.0.0.9"\n',
            "nccl_master_port",
            id="master-port-type-and-range-are-validated-65536",
        ),
        pytest.param(
            '[cluster]\nnccl_master_port = true\n\n[[hosts]]\nname = "10.0.0.9"\n',
            "nccl_master_port",
            id="master-port-type-and-range-are-validated-true",
        ),
        pytest.param(
            '[cluster]\nnccl_master_port = "29777"\n\n[[hosts]]\nname = "10.0.0.9"\n',
            "nccl_master_port",
            id="master-port-type-and-range-are-validated-quoted-29777",
        ),
        pytest.param(
            '[[hosts]]\nname = "worker"\naddress = "tcp://0.0.0.0:26600"\n',
            "unicast",
            id="unspecified-and-multicast-worker-binds-are-rejected-unspecified-ipv4",
        ),
        pytest.param(
            '[[hosts]]\nname = "worker"\naddress = "tcp://[::]:26600"\n',
            "unicast",
            id="unspecified-and-multicast-worker-binds-are-rejected-unspecified-ipv6",
        ),
        pytest.param(
            '[[hosts]]\nname = "worker"\naddress = "tcp://224.0.0.1:26600"\n',
            "unicast",
            id="unspecified-and-multicast-worker-binds-are-rejected-multicast-ipv4",
        ),
        pytest.param(
            '[[hosts]]\nname = "worker"\naddress = "tcp://[ff02::1]:26600"\n',
            "unicast",
            id="unspecified-and-multicast-worker-binds-are-rejected-multicast-ipv6",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://0.0.0.0:0"\n\n'
            '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n',
            "unicast",
            id="unspecified-and-multicast-client-binds-are-rejected-unspecified-ipv4",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://[ff02::1]:0"\n\n'
            '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n',
            "unicast",
            id="unspecified-and-multicast-client-binds-are-rejected-multicast-ipv6",
        ),
        # These rows test the IPv4-mapped spellings of 0.0.0.0, 127.0.0.1 and
        # 224.0.0.1. On CPython 3.12.3 their IPv6 objects answer False to
        # is_unspecified, is_loopback and is_multicast; on 3.11.14 and 3.13.11
        # each answers True where its IPv4 form does (checked 2026-10-07).
        # config.unmapped_bind_ip unwraps them, and the bind checks refuse any
        # mapped spelling, so the refusal holds on every Python version. The
        # worker-loop CLI test in tests/test_ops_security.py reads the same
        # three from MAPPED_BINDS.
        pytest.param(
            '[[hosts]]\nname = "worker"\naddress = "tcp://[::ffff:0.0.0.0]:26600"\n',
            "IPv4-mapped",
            id="ipv4-mapped-worker-binds-are-rejected-unspecified",
        ),
        pytest.param(
            '[[hosts]]\nname = "worker"\naddress = "tcp://[::ffff:127.0.0.1]:26600"\n',
            "IPv4-mapped",
            id="ipv4-mapped-worker-binds-are-rejected-loopback",
        ),
        pytest.param(
            '[[hosts]]\nname = "worker"\naddress = "tcp://[::ffff:224.0.0.1]:26600"\n',
            "IPv4-mapped",
            id="ipv4-mapped-worker-binds-are-rejected-multicast",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://[::ffff:0.0.0.0]:0"\n\n'
            '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n',
            "IPv4-mapped",
            id="ipv4-mapped-client-binds-are-rejected-unspecified",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://[::ffff:127.0.0.1]:0"\n\n'
            '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n',
            "IPv4-mapped",
            id="ipv4-mapped-client-binds-are-rejected-loopback",
        ),
        pytest.param(
            '[cluster]\nclient_bind = "tcp://[::ffff:224.0.0.1]:0"\n\n'
            '[[hosts]]\nname = "worker"\naddress = "tcp://10.0.0.2:26600"\n',
            "IPv4-mapped",
            id="ipv4-mapped-client-binds-are-rejected-multicast",
        ),
    ],
)
def test_cluster_toml_refusals(tmp_path: Path, toml: str, fragment: str):
    path = tmp_path / "cluster.toml"
    path.write_text(toml)
    with pytest.raises(ClusterConfigError, match=fragment):
        load_cluster_config(path)


def test_literal_ip_name_may_default_the_address(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text('[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n[[hosts]]\nname = "10.0.0.9"\n')
    config = load_cluster_config(path)
    assert config.hosts[0].address == "tcp://10.0.0.9:26600"


def test_multi_host_config_accepts_explicit_network_fabric(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n'
        'fabric_profile = "generic-roce"\n\n'
        '[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.1:26600"\n\n'
        '[[hosts]]\nname = "n2"\naddress = "tcp://10.0.0.2:26600"\n'
    )

    assert load_cluster_config(path).fabric_profile == "generic-roce"


def test_renderer_refuses_single_node_fabric_for_multiple_hosts():
    with pytest.raises(ClusterConfigError, match=r"single-node.*2 hosts"):
        render_cluster_toml(
            hosts=[("n1", "10.0.0.1", 1), ("n2", "10.0.0.2", 1)],
            client_ip="10.0.0.1",
            fabric_profile="single-node",
            master_addr="10.0.0.1",
        )


def test_bare_hosts_table_is_refused(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text('[hosts]\nname = "10.0.0.9"\n')
    with pytest.raises(ClusterConfigError):
        load_cluster_config(path)


def test_explicit_missing_config_path_raises():
    with pytest.raises(ClusterConfigError):
        find_config_path("/nonexistent/cluster.toml")


def test_valid_explicit_binds_accepted(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n'
        '[[hosts]]\nname = "sparkb"\naddress = "tcp://10.0.0.9:26600"\n'
    )
    config = load_cluster_config(path)
    assert config.hosts[0].address == "tcp://10.0.0.9:26600"
    assert config.client_bind == "tcp://10.0.0.1:0"


def test_ipv6_endpoints_are_bracketed_and_canonical(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text(
        '[cluster]\nclient_bind = "tcp://[2001:0DB8:0:0::1]:0"\n\n'
        '[[hosts]]\nname = "n1"\naddress = "tcp://[2001:0DB8:0:0::2]:26600"\n'
    )
    config = load_cluster_config(path)
    assert config.client_bind == "tcp://[2001:db8::1]:0"
    assert config.hosts[0].address == "tcp://[2001:db8::2]:26600"
    assert config.resolved_master_addr() == "2001:db8::2"
    assert format_tcp_address("2001:db8::3", 42) == "tcp://[2001:db8::3]:42"


@pytest.mark.parametrize("host", [True, False, 1, 42, b"10.0.0.1", None])
def test_format_tcp_address_rejects_every_non_string_host(host):
    with pytest.raises(ValueError, match="TCP host must be a string literal IP address"):
        format_tcp_address(host, 42)


def test_render_ipv6_round_trip(tmp_path: Path):
    path = tmp_path / "cluster.toml"
    path.write_text(render_cluster_toml(
        hosts=[("node", "fd00::2", 1)],
        client_ip="fd00::1",
        fabric_profile="generic-ib",
        master_addr="fd00::1",
    ))
    config = load_cluster_config(path)
    assert config.client_bind == "tcp://[fd00::1]:0"
    assert config.hosts[0].address == "tcp://[fd00::2]:26600"


@pytest.mark.parametrize("mapped", ["::ffff:0.0.0.0", "::ffff:224.0.0.1", "::ffff:192.0.2.1"])
def test_master_addr_rejects_every_ipv4_mapped_ipv6_spelling(tmp_path: Path, mapped: str):
    path = tmp_path / "cluster.toml"
    path.write_text(
        f'[cluster]\nnccl_master_addr = "{mapped}"\n\n[[hosts]]\nname = "10.0.0.9"\n')
    with pytest.raises(ClusterConfigError, match="IPv4-mapped"):
        load_cluster_config(path)
    with pytest.raises(ClusterConfigError, match="IPv4-mapped"):
        render_cluster_toml(
            hosts=[("node", "10.0.0.2", 1)], client_ip="10.0.0.1",
            fabric_profile="generic-roce", master_addr=mapped)


@pytest.mark.parametrize(
    ("field", "invalid", "overrides"),
    [
        ("client_ip", "not-an-ip", {"client_ip": "not-an-ip"}),
        ("master_addr", "not-an-ip", {"master_addr": "not-an-ip"}),
        ("host 'node' address", "not-an-ip", {"hosts": [("node", "not-an-ip", 1)]}),
        ("client_ip", "True", {"client_ip": True}),
        ("master_addr", "1", {"master_addr": 1}),
        ("host 'node' address", "True", {"hosts": [("node", True, 1)]}),
    ],
)
def test_render_invalid_ips_raise_contextual_config_error(
    field: str, invalid: str, overrides: dict,
):
    args = {
        "hosts": [("node", "10.0.0.2", 1)],
        "client_ip": "10.0.0.1",
        "fabric_profile": "generic-roce",
        "master_addr": "10.0.0.1",
    }
    args.update(overrides)
    with pytest.raises(ClusterConfigError) as raised:
        render_cluster_toml(**args)
    assert field in str(raised.value)
    assert invalid in str(raised.value)


def test_render_does_not_claim_network_isolation_by_default():
    text = render_cluster_toml(
        hosts=[("node", "10.0.0.2", 1)],
        client_ip="10.0.0.1",
        fabric_profile="generic-roce",
        master_addr="10.0.0.1",
    )
    assert 'transport_security = ""' in text
