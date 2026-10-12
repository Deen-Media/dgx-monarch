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
    with pytest.raises(ClusterConfigError, match="selected target"):
        policy.apply_policy(policy.MonarchPolicy(), native_enabled=True)


@pytest.mark.parametrize("target", ["cpu:0", "gpu:0"])
def test_single_rail_requires_a_named_nic(isolated_policy, target):
    with pytest.raises(ClusterConfigError, match="selected target"):
        policy.apply_policy(policy.MonarchPolicy(rdma=policy.RdmaPolicy(target=target)), native_enabled=True)


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


@pytest.mark.parametrize("token", ["%%%", "e30\n=", "A" * 20000])
def test_policy_token_is_bounded_and_strict(token):
    with pytest.raises(ClusterConfigError):
        policy.policy_from_token(token)
