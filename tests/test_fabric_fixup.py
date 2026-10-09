"""A fabric profile written for one QSFP port set must not strand a rig cabled
into the other: workers rewrite it to the active RDMA rails (docs/INSTALL.md)."""
from dgx_monarch.config import FABRIC_PROFILES, detect_roce_rails, fixup_fabric_ifaces


def _fake_sysfs(tmp_path, rails, up):
    """rails: {rdma_dev: netdev}; up: set of netdevs whose operstate is 'up'."""
    for rdma_dev, netdev in rails.items():
        (tmp_path / "class" / "infiniband" / rdma_dev / "device" / "net" / netdev).mkdir(parents=True)
        net = tmp_path / "class" / "net" / netdev
        net.mkdir(parents=True, exist_ok=True)
        (net / "operstate").write_text("up\n" if netdev in up else "down\n")
    return str(tmp_path)


def test_detect_rails_only_up_sorted(tmp_path):
    root = _fake_sysfs(
        tmp_path,
        {"rocep1s0f1": "enp1s0f1np1", "roceP2p1s0f1": "enP2p1s0f1np1", "rocep1s0f0": "enp1s0f0np0"},
        up={"enp1s0f1np1", "enP2p1s0f1np1"},
    )
    # The case-insensitive sort puts the non-P2p rail first, as the primary rail.
    assert detect_roce_rails(root) == [("enp1s0f1np1", "rocep1s0f1"), ("enP2p1s0f1np1", "roceP2p1s0f1")]


def test_fixup_passthrough_when_iface_up(tmp_path):
    root = _fake_sysfs(tmp_path, {"rocep1s0f0": "enp1s0f0np0"}, up={"enp1s0f0np0"})
    env = dict(FABRIC_PROFILES["dgx-spark-pair"])
    healed, note = fixup_fabric_ifaces(env, root)
    assert healed == env and note == ""   # a working rig's env passes unchanged


def test_fixup_heals_other_port_set(tmp_path):
    # Profile names the f0 set; the box is cabled into f1.
    root = _fake_sysfs(
        tmp_path,
        {"rocep1s0f1": "enp1s0f1np1", "roceP2p1s0f1": "enP2p1s0f1np1"},
        up={"enp1s0f1np1", "enP2p1s0f1np1"},
    )
    healed, note = fixup_fabric_ifaces(dict(FABRIC_PROFILES["dgx-spark-pair"]), root)
    assert healed["NCCL_SOCKET_IFNAME"] == "enp1s0f1np1"
    assert healed["GLOO_SOCKET_IFNAME"] == "enp1s0f1np1"
    assert healed["UCX_NET_DEVICES"] == "enp1s0f1np1"
    assert healed["NCCL_IB_HCA"] == "rocep1s0f1,roceP2p1s0f1"
    assert "QSFP" in note
    # untouched keys survive
    assert healed["NCCL_IB_GID_INDEX"] == "3"


def test_fixup_no_rails_warns_without_rewrite(tmp_path):
    root = _fake_sysfs(tmp_path, {"rocep1s0f0": "enp1s0f0np0"}, up=set())
    env = dict(FABRIC_PROFILES["dgx-spark-pair"])
    healed, note = fixup_fabric_ifaces(env, root)
    assert healed == env
    assert "no UP RDMA rails" in note


def test_fixup_noop_without_socket_ifname(tmp_path):
    healed, note = fixup_fabric_ifaces({"NCCL_DEBUG": "WARN"}, str(tmp_path))
    assert healed == {"NCCL_DEBUG": "WARN"} and note == ""
