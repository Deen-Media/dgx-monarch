"""Typed policy, process-global identity and native readback boundaries."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dgx_monarch import monarch_policy as policy
from dgx_monarch.config import ClusterConfig, ClusterConfigError, load_cluster_config, local_config
from dgx_monarch.monarch_policy_io import trust_config_fingerprint


@pytest.fixture
def isolated_policy(monkeypatch, isolated_environ):
    monkeypatch.setattr(policy, "_ACTIVE", None)
    for key in policy.policy_environment(policy.MonarchPolicy()):
        monkeypatch.delenv(key, raising=False)


def test_selected_host_resolves_before_serialization():
    value = policy.parse_policy({"rdma": {
        "target": "nic:driver0", "host_targets": {"worker": "nic:worker0"},
        "peer_device_affinity": "groups:driver0,worker0", "qps_per_peer": 2,
    }}, host_names=("worker",))
    child = policy.policy_from_token(policy.encode_policy(value, "worker"))
    assert child.rdma.target == "nic:worker0"
    assert child.rdma.host_targets == ()
    env = policy.policy_environment(value, "worker")
    assert env["MONARCH_RDMA_IBVERBS_TARGET"] == "nic:worker0"
    assert env["MONARCH_RDMA_ALLOW_TCP_FALLBACK"] == "false"
    assert env["MONARCH_RDMA_QPS_PER_PEER"] == "2"
    assert policy.policy_from_token(env[policy.POLICY_ENV]) == child


def test_local_init_cache_tracks_policy_but_ignores_cluster_hosts(tmp_path):
    from dgx_monarch.nodes.init import DGXMonarchInit

    path = tmp_path / "cluster.toml"
    path.write_text('[cluster]\nname="one"\n[monarch.rdma]\napplication_chunks=1\n')
    first = DGXMonarchInit.IS_CHANGED("local", "local", str(path))
    path.write_text('[cluster]\nname="two"\n[monarch.rdma]\napplication_chunks=1\n')
    assert DGXMonarchInit.IS_CHANGED("local", "local", str(path)) == first
    path.write_text('[cluster]\nname="two"\n[monarch.rdma]\napplication_chunks=2\n')
    assert DGXMonarchInit.IS_CHANGED("local", "local", str(path)) != first


@pytest.mark.parametrize("raw", [
    {"startup_timeout_s": True}, {"startup_timeout_s": 0}, {"typo": 3},
    {"rdma": {"target": "nic:a\nsecret"}}, {"rdma": {"target": "gpu:-1"}},
    {"rdma": {"peer_device_affinity": "groups:a,b|b,c"}},
    {"rdma": {"max_nics_per_buffer": 0}}, {"rdma": {"application_chunks": False}},
    {"rdma": {"max_nics_per_buffer": 8, "qps_per_peer": 8}},
    {"rdma": {"host_targets": {"missing": "nic:x"}}},
    {"diagnostics": {"metrics": "false"}},
    {"diagnostics": {"otlp_endpoint": "https://secret@collector.example"}},
])
def test_invalid_policy_refuses_before_native_work(raw):
    with pytest.raises(ClusterConfigError):
        policy.parse_policy(raw, host_names=("worker",))


def test_transport_latches_and_diagnostic_knobs_do_not_change_identity(isolated_policy):
    initial = policy.MonarchPolicy()
    policy.apply_policy(initial)
    diagnostic = replace(initial, diagnostics=policy.DiagnosticsPolicy(metrics=True))
    policy.apply_policy(diagnostic)
    assert policy.transport_identity(initial) == policy.transport_identity(diagnostic)
    changed = replace(initial, rdma=replace(initial.rdma, target="nic:other"))
    with pytest.raises(RuntimeError, match="restart"):
        policy.apply_policy(changed)
    assert os.environ["MONARCH_RDMA_IBVERBS_TARGET"] == ""


def test_ambient_conflict_refused_without_partial_install(isolated_policy, monkeypatch):
    monkeypatch.setenv("MONARCH_RDMA_QPS_PER_PEER", "8")
    with pytest.raises(ClusterConfigError, match="conflicts"):
        policy.apply_policy(policy.MonarchPolicy())
    assert policy.POLICY_ENV not in os.environ


def test_native_readback_mismatch_does_not_mark_verified(isolated_policy, monkeypatch):
    value = policy.MonarchPolicy()
    policy.apply_policy(value)
    monkeypatch.setitem(sys.modules, "monarch.config", SimpleNamespace(get_global_config=lambda: {}))
    with pytest.raises(RuntimeError, match="differs"):
        policy.verify_native_policy()
    assert policy.snapshot()["native_readback"] == "pending"


def test_native_requires_selected_target(isolated_policy):
    with pytest.raises(ClusterConfigError, match="selected target") as caught:
        policy.apply_policy(policy.MonarchPolicy(), native_enabled=True)
    assert "[monarch.rdma] target" in str(caught.value) and "host_targets" in str(caught.value)
    assert "this driver process resolves to upstream selection" in str(caught.value)
    assert policy.POLICY_ENV not in os.environ


@pytest.mark.parametrize("target", ["cpu:0", "gpu:0"])
def test_single_rail_requires_a_named_nic(isolated_policy, target):
    with pytest.raises(ClusterConfigError, match="selected target"):
        policy.apply_policy(policy.MonarchPolicy(rdma=policy.RdmaPolicy(target=target)), native_enabled=True)


@pytest.mark.parametrize("nics", [2, 8])
def test_multi_nic_native_has_no_empty_target_escape(isolated_policy, nics):
    """Upstream auto-selection across several NICs is the unqualified 0.6 failure class."""
    value = policy.parse_policy({"rdma": {"max_nics_per_buffer": nics}})
    with pytest.raises(ClusterConfigError, match=r"selected target.*\[monarch\.rdma\] target"):
        policy.apply_policy(value, native_enabled=True)
    policy.apply_policy(value)  # native off: upstream selection stays allowed


HOSTS = ("192.0.2.1", "192.0.2.2")
Q1_POLICIES = {
    "rig_default": {},
    "global_only": {"rdma": {"target": "nic:mlx5_0"}},
    "host_targets_only": {"rdma": {"host_targets": dict.fromkeys(HOSTS, "nic:mlx5_0")}},
    "global_plus_override": {"rdma": {"target": "nic:global0", "host_targets": {HOSTS[1]: "nic:worker1"}}},
    "global_empty_multi_nic": {"rdma": {"max_nics_per_buffer": 2}},
    "gpu_target": {"rdma": {"target": "gpu:0"}},
}


@pytest.mark.parametrize(("case", "host", "accepted", "target"), [
    # The driver now resolves its own host entry, like the qualifier and each actor.
    ("rig_default", HOSTS[0], False, None),
    ("rig_default", HOSTS[1], False, None),
    ("global_only", HOSTS[0], True, "nic:mlx5_0"),
    ("global_only", HOSTS[1], True, "nic:mlx5_0"),
    ("host_targets_only", HOSTS[0], True, "nic:mlx5_0"),
    ("host_targets_only", HOSTS[1], True, "nic:mlx5_0"),
    ("global_plus_override", HOSTS[0], True, "nic:global0"),
    ("global_plus_override", HOSTS[1], True, "nic:worker1"),
    ("global_empty_multi_nic", HOSTS[0], False, None),
    ("gpu_target", HOSTS[0], False, None),
])
def test_per_host_native_admission_matches_each_participant(isolated_policy, case, host, accepted, target):
    value = policy.parse_policy(Q1_POLICIES[case], host_names=HOSTS)
    if not accepted:
        with pytest.raises(ClusterConfigError, match="selected target"):
            policy.apply_policy(value, host, native_enabled=True)
        with pytest.raises(ClusterConfigError, match="selected target"):
            policy.admit_native_targets(value, HOSTS)
        return
    policy.apply_policy(value, host, native_enabled=True)
    assert os.environ["MONARCH_RDMA_IBVERBS_TARGET"] == target
    policy.admit_native_targets(value, HOSTS)


def test_driver_admission_checks_every_host_before_any_process_starts():
    """A host without a NIC fails at driver admission, never inside a committed actor."""
    partial = policy.parse_policy({"rdma": {"host_targets": {HOSTS[0]: "nic:a0"}}}, host_names=HOSTS)
    policy.admit_native_targets(partial, (HOSTS[0],))
    with pytest.raises(ClusterConfigError, match=f"host '{HOSTS[1]}' resolves to upstream selection"):
        policy.admit_native_targets(partial, HOSTS)
    with pytest.raises(ClusterConfigError, match="the driver resolves to upstream selection"):
        policy.admit_native_targets(partial, ("", *HOSTS[:1]))


@pytest.mark.parametrize("override", ["", "cpu:0", "gpu:3"])
def test_per_host_override_must_name_a_nic_at_parse_time(override):
    with pytest.raises(ClusterConfigError, match="host_targets values must be nic:<device>"):
        policy.parse_policy({"rdma": {"target": "nic:global0", "host_targets": {HOSTS[1]: override}}},
                            host_names=HOSTS)


VALIDATOR_REFUSALS = {
    "nics0": {"rdma": {"max_nics_per_buffer": 0}},
    "qps_neg": {"rdma": {"qps_per_peer": -1}},
    "chunks0": {"rdma": {"application_chunks": 0}},
    "chunks_huge": {"rdma": {"application_chunks": 10**9}},
    "product33": {"rdma": {"max_nics_per_buffer": 4, "qps_per_peer": 8, "application_chunks": 2}},
    "bool_as_int": {"rdma": {"qps_per_peer": True}},
    "float": {"rdma": {"qps_per_peer": 1.0}},
    "affinity_unknown": {"rdma": {"peer_device_affinity": "nearest"}},
    "affinity_empty_groups": {"rdma": {"peer_device_affinity": "groups:"}},
    "affinity_dup": {"rdma": {"peer_device_affinity": "groups:a,b|b,c"}},
    "affinity_trailing_bar": {"rdma": {"peer_device_affinity": "groups:a,b|"}},
    "host_targets_list": {"rdma": {"host_targets": ["a"]}},
    "host_targets_str": {"rdma": {"host_targets": "a"}},
    "host_targets_badval": {"rdma": {"host_targets": {"h": 5}}},
    "host_targets_empty_val": {"rdma": {"host_targets": {"h": ""}}},
    "host_targets_cpu": {"rdma": {"host_targets": {"h": "cpu:0"}}},
    "target_ws": {"rdma": {"target": " nic:mlx5_0"}},
    "target_ws2": {"rdma": {"target": "nic:mlx5_0 "}},
    "target_nl": {"rdma": {"target": "nic:mlx5_0\n"}},
    "target_nic_empty": {"rdma": {"target": "nic:"}},
    "target_gpu_256": {"rdma": {"target": "gpu:256"}},
    "target_gpu_unicode": {"rdma": {"target": "gpu:" + chr(0x661)}},  # ARABIC-INDIC DIGIT ONE
    "target_cpu_neg": {"rdma": {"target": "cpu:-1"}},
    "target_upper_kind": {"rdma": {"target": "NIC:mlx5_0"}},
    "startup_29": {"startup_timeout_s": 29},
    "startup_bool": {"startup_timeout_s": True},
    "unknown_key": {"rdma": {"bogus": 1}},
    "otlp_no_logs": {"diagnostics": {"otlp_endpoint": "http://x:1"}},
    "otlp_cred": {"diagnostics": {"otlp_endpoint": "http://u:p@x:1", "otlp_logs": True}},
    "otlp_port_bad": {"diagnostics": {"otlp_endpoint": "http://x:99999", "otlp_logs": True}},
    "otlp_ftp": {"diagnostics": {"otlp_endpoint": "ftp://x", "otlp_logs": True}},
    "qp_init_0": {"rdma": {"qp_init_timeout_s": 0}},
    "stripe_3": {"rdma": {"min_stripe_size_kb": 3}},
    "fallback_str": {"rdma": {"allow_tcp_fallback": "false"}},
    "root_list": [],
}


@pytest.mark.parametrize("name", sorted(VALIDATOR_REFUSALS))
def test_validator_refusals_are_typed(name):
    with pytest.raises(ClusterConfigError):
        policy.parse_policy(VALIDATOR_REFUSALS[name])


@pytest.mark.parametrize("raw", [
    {"rdma": {"max_nics_per_buffer": 4, "qps_per_peer": 8, "application_chunks": 1}},
    {"rdma": {"min_stripe_size_kb": 5}},
    {"rdma": {"target": "cpu:1"}}, {"rdma": {"target": "gpu:255"}},
    {"rdma": {"target": "nic:mlx5_0", "host_targets": {"h": "nic:mlx5_1"}}},
    {"diagnostics": {"otlp_endpoint": "https://collector.example:4318", "otlp_logs": True}},
])
def test_validator_accepts_supported_values(raw):
    assert isinstance(policy.parse_policy(raw), policy.MonarchPolicy)


def test_application_chunks_cannot_drift_after_admission(isolated_policy):
    from dgx_monarch.transfer_utils import qp_split

    policy.apply_policy(policy.MonarchPolicy())
    os.environ["DGXM_RDMA_QP_SPLIT"] = "8"
    assert qp_split() == 1


def test_unknown_native_override_refused(isolated_policy, monkeypatch):
    monkeypatch.setenv("MONARCH_RDMA_UNKNOWN", "1")
    with pytest.raises(ClusterConfigError, match="unmanaged"):
        policy.apply_policy(policy.MonarchPolicy())


def test_late_native_activation_uses_runtime_and_reads_back(isolated_policy, monkeypatch):
    values = {}
    monkeypatch.setitem(sys.modules, "monarch.config", SimpleNamespace(
        configure=lambda **kwargs: values.update(kwargs), get_global_config=lambda: values))
    monkeypatch.setitem(sys.modules, "monarch._src.rdma.rdma", SimpleNamespace(_rdma_manager_init_cache={}))
    policy.apply_policy(policy.MonarchPolicy())
    result = policy.activate_native_policy()
    assert values["rdma_qps_per_peer"] == 1
    assert values["rdma_disable_ibverbs"] is False
    assert result["native_readback"] == "verified"


def test_prior_rdma_manager_refuses_late_admission(isolated_policy, monkeypatch):
    configured = []
    monkeypatch.setitem(sys.modules, "monarch.config", SimpleNamespace(configure=lambda **kwargs: configured.append(kwargs)))
    monkeypatch.setitem(sys.modules, "monarch._src.rdma.rdma", SimpleNamespace(_rdma_manager_init_cache={"owned": object()}))
    policy.apply_policy(policy.MonarchPolicy())
    with pytest.raises(RuntimeError, match="initialized before"):
        policy.activate_native_policy()
    assert configured == []


def test_config_and_local_mode_read_same_policy(tmp_path):
    path = tmp_path / "cluster.toml"
    path.write_text('[[hosts]]\nname="worker"\naddress="tcp://192.0.2.1:26600"\n'
                    '[monarch.rdma]\ntarget="nic:test0"\n[monarch.diagnostics]\nmetrics=true\n')
    cluster, local = load_cluster_config(path), local_config(path)
    assert cluster.monarch == local.monarch
    assert cluster.monarch.rdma.target == "nic:test0"


def test_gate_identity_excludes_export_settings_and_binds_transport():
    base = ClusterConfig()
    diagnostic = replace(base, monarch=replace(base.monarch, diagnostics=policy.DiagnosticsPolicy(metrics=True)))
    native = replace(base, monarch=replace(base.monarch, rdma=policy.RdmaPolicy(target="nic:x")))
    threshold = replace(base, rdma_min_bytes=1024)
    assert trust_config_fingerprint(base, "") == trust_config_fingerprint(diagnostic, "")
    assert trust_config_fingerprint(base, "") != trust_config_fingerprint(native, "")
    assert trust_config_fingerprint(base, "") != trust_config_fingerprint(threshold, "")


def _cluster() -> ClusterConfig:
    from dgx_monarch.config import HostConfig

    return ClusterConfig(
        hosts=(HostConfig("192.0.2.1", "tcp://192.0.2.1:26600"), HostConfig("192.0.2.2", "tcp://192.0.2.2:26600")),
        client_bind="tcp://192.0.2.1:0", transport_security="trusted_fabric", source="/cfg/cluster.toml")


def _monarch(raw: dict, config: ClusterConfig) -> policy.MonarchPolicy:
    return policy.parse_policy(raw, host_names=tuple(host.name for host in config.hosts))


# Lifecycle, readiness-budget, export and provenance edits keep Gate PASS rows.
FINGERPRINT_SAME = {
    "metrics": lambda c: replace(c, monarch=_monarch({"diagnostics": {"metrics": True}}, c)),
    "otlp": lambda c: replace(c, monarch=_monarch({"diagnostics": {
        "otlp_endpoint": "http://127.0.0.1:4318", "otlp_logs": True}}, c)),
    "startup": lambda c: replace(c, monarch=_monarch({"startup_timeout_s": 240}, c)),
    "explicit_defaults": lambda c: replace(c, monarch=_monarch({"rdma": {"qps_per_peer": 1}}, c)),
    "auto_heal": lambda c: replace(c, auto_heal=not c.auto_heal),
    "ssh_key": lambda c: replace(c, ssh_key="/keys/other"),
    "source": lambda c: replace(c, source="/elsewhere/cluster.toml"),
}
# Anything that can change math, residency or transport re-proves.
FINGERPRINT_CHANGED = {
    "target": lambda c: replace(c, monarch=_monarch({"rdma": {"target": "nic:x"}}, c)),
    "host_target": lambda c: replace(c, monarch=_monarch({"rdma": {"host_targets": {"192.0.2.2": "nic:y"}}}, c)),
    "chunks": lambda c: replace(c, monarch=_monarch({"rdma": {"application_chunks": 2}}, c)),
    "native": lambda c: replace(c, rdma_latent_return=True),
    "threshold": lambda c: replace(c, rdma_min_bytes=1024),
    "worker_args": lambda c: replace(c, worker_args={"slab_weights": "off"}),
    "address": lambda c: replace(c, hosts=(c.hosts[0], replace(c.hosts[1], address="tcp://192.0.2.3:26600"))),
    "gpus": lambda c: replace(c, hosts=(c.hosts[0], replace(c.hosts[1], gpus=2))),
    "host_comfy": lambda c: replace(c, hosts=(c.hosts[0], replace(c.hosts[1], comfy_dir="/other/ComfyUI"))),
    # The Worker runs the ssh_user account's interpreter and ComfyUI tree.
    "ssh_user": lambda c: replace(c, hosts=tuple(replace(h, ssh_user="operator") for h in c.hosts)),
    "python": lambda c: replace(c, python_bin="/other/env/bin/python"),
    "fabric_env": lambda c: replace(c, fabric_env={"NCCL_ALGO": "Ring"}),
    "client_bind": lambda c: replace(c, client_bind="tcp://192.0.2.2:0"),
    "master_port": lambda c: replace(c, nccl_master_port=29778),
}


@pytest.mark.parametrize("name", sorted(FINGERPRINT_SAME))
def test_trust_fingerprint_ignores_lifecycle_budget_and_export_settings(name):
    base = _cluster()
    assert trust_config_fingerprint(FINGERPRINT_SAME[name](base), "fb") == trust_config_fingerprint(base, "fb")


@pytest.mark.parametrize("name", sorted(FINGERPRINT_CHANGED))
def test_trust_fingerprint_binds_math_residency_and_transport(name):
    base = _cluster()
    assert trust_config_fingerprint(FINGERPRINT_CHANGED[name](base), "fb") != trust_config_fingerprint(base, "fb")


def test_every_config_field_is_classified_for_the_trust_fingerprint():
    """A new field is hashed by default; widening the excluded sets needs this test edited."""
    from dataclasses import fields

    from dgx_monarch import monarch_policy_io
    from dgx_monarch.config import HostConfig

    assert monarch_policy_io.UNTRUSTED_FIELDS == {"auto_heal", "ssh_key", "source"}
    assert monarch_policy_io.UNTRUSTED_HOST_FIELDS == frozenset()
    assert monarch_policy_io.UNTRUSTED_MONARCH_FIELDS == {"startup_timeout_s", "diagnostics"}
    assert monarch_policy_io.UNTRUSTED_FIELDS < {f.name for f in fields(ClusterConfig)}
    assert monarch_policy_io.UNTRUSTED_HOST_FIELDS < {f.name for f in fields(HostConfig)}
    assert {f.name for f in fields(policy.MonarchPolicy)} - monarch_policy_io.UNTRUSTED_MONARCH_FIELDS == {"rdma"}


def test_local_mode_fingerprint_reads_the_same_rule(tmp_path):
    path = tmp_path / "cluster.toml"
    path.write_text("[monarch]\nstartup_timeout_s = 240\n")
    assert trust_config_fingerprint(local_config(path), "fb") == trust_config_fingerprint(ClusterConfig(), "fb")
    assert trust_config_fingerprint(object(), "fb") == "fb"


@pytest.fixture
def driver_warnings(monkeypatch):
    import logging

    from dgx_monarch import monarch_policy_io
    from dgx_monarch.log import get_logger

    monkeypatch.setattr(monarch_policy_io, "_UNMATCHED_DRIVER_WARNED", False)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = get_logger("dgx_monarch.monarch_policy_io")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


def test_driver_host_matches_client_bind_to_a_worker_address(driver_warnings):
    from dgx_monarch.config import HostConfig
    from dgx_monarch.monarch_policy_io import driver_host

    config = _cluster()
    assert driver_host(config) == "192.0.2.1"
    assert driver_host(replace(config, client_bind="tcp://192.0.2.2:0")) == "192.0.2.2"
    # The worker bind address decides, not an ssh alias that happens to differ.
    aliased = replace(config, hosts=(HostConfig("head", "tcp://192.0.2.1:26600"),
                                     HostConfig("worker", "tcp://192.0.2.2:26600")))
    assert driver_host(aliased) == "head"
    assert driver_host(ClusterConfig()) == ""  # local mode has no host to match
    assert driver_warnings == []


def test_unmatched_driver_uses_global_target_and_warns_once(driver_warnings):
    import logging

    from dgx_monarch.monarch_policy_io import driver_host

    for bind in ("tcp://192.0.2.9:0", ""):
        assert driver_host(replace(_cluster(), client_bind=bind)) == ""
    assert [record.levelno for record in driver_warnings] == [logging.WARNING]
    assert "global [monarch.rdma] target" in driver_warnings[0].getMessage()


def test_driver_host_on_shared_address_must_agree_on_target():
    from dgx_monarch.config import HostConfig
    from dgx_monarch.monarch_policy_io import driver_host

    shared = ClusterConfig(hosts=(HostConfig("one", "tcp://192.0.2.1:26600"),
                                  HostConfig("two", "tcp://192.0.2.1:26601")),
                           client_bind="tcp://192.0.2.1:0")
    assert driver_host(shared) == "one"
    split = replace(shared, monarch=policy.parse_policy(
        {"rdma": {"host_targets": {"one": "nic:a0", "two": "nic:b0"}}}, host_names=("one", "two")))
    with pytest.raises(ClusterConfigError, match="ambiguous"):
        driver_host(split)


def test_transport_identity_is_the_rdma_block_only(isolated_policy):
    base = policy.MonarchPolicy()
    budget = replace(base, startup_timeout_s=240)
    assert policy.transport_identity(budget) == policy.transport_identity(base)
    policy.apply_policy(base)
    policy.apply_policy(budget)  # a readiness-budget edit no longer demands a new process
    assert policy.snapshot()["transport_sha256"] == policy.transport_identity(base)
    chunked = replace(base, rdma=replace(base.rdma, application_chunks=2))
    assert policy.transport_identity(chunked) != policy.transport_identity(base)


@pytest.mark.parametrize("token", ["%%%", "e30\n=", "A" * 20000])
def test_policy_token_is_bounded_and_strict(token):
    with pytest.raises(ClusterConfigError):
        policy.policy_from_token(token)
