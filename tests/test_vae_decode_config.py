"""``[worker_args] worker_vae_decode``: off by default, read by the driver alone.

The knob takes "off" or "auto". It is a driver-only key: it never reaches an
actor, so it stays out of the published worker policy and its key, out of
every Gate capability context and out of the trust fingerprint. Turning it on
therefore keeps every persisted PASS row, since a decode changes no DiT math.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dgx_monarch.config import load_cluster_config
from dgx_monarch.config_schema import (
    DRIVER_ONLY_KEYS,
    ClusterConfigError,
    actor_worker_args,
    validate_worker_args,
)
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.monarch_policy_io import trust_config_fingerprint

CLUSTER = '[cluster]\nclient_bind = "tcp://192.0.2.11:0"\n[[hosts]]\nname = "192.0.2.11"\n'


def test_the_knob_takes_off_or_auto():
    assert validate_worker_args({"worker_vae_decode": "auto"}) == {"worker_vae_decode": "auto"}
    assert validate_worker_args({"worker_vae_decode": "off"}) == {"worker_vae_decode": "off"}
    for bad in ("on", True, 1, "AUTO"):
        with pytest.raises(ClusterConfigError, match='worker_vae_decode must be "off" or "auto"'):
            validate_worker_args({"worker_vae_decode": bad})


def test_the_knob_is_driver_only():
    assert DRIVER_ONLY_KEYS == frozenset({"worker_vae_decode"})
    assert actor_worker_args({"worker_vae_decode": "auto", "slab_weights": True}) == {"slab_weights": True}
    assert actor_worker_args(None) == {}


def test_the_actors_never_receive_it():
    handle = SimpleNamespace(config=SimpleNamespace(worker_args={"worker_vae_decode": "auto", "swap_verify": 2}))
    effective = MeshHandle.effective_worker_args(handle, {"mmap_fallback": False})
    assert effective == {"swap_verify": 2, "mmap_fallback": False}


def test_turning_it_on_keeps_the_trust_fingerprint(tmp_path):
    plain = tmp_path / "plain.toml"
    plain.write_text(CLUSTER)
    auto = tmp_path / "auto.toml"
    auto.write_text(CLUSTER + '[worker_args]\nworker_vae_decode = "auto"\n')
    config = load_cluster_config(auto)
    assert config.worker_args == {"worker_vae_decode": "auto"}
    assert (trust_config_fingerprint(config, "fallback")
            == trust_config_fingerprint(load_cluster_config(plain), "fallback"))
    other = tmp_path / "other.toml"
    other.write_text(CLUSTER + "[worker_args]\nswap_verify = 4\n")
    assert (trust_config_fingerprint(load_cluster_config(other), "fallback")
            != trust_config_fingerprint(load_cluster_config(plain), "fallback"))


def test_the_example_config_documents_the_knob_off():
    from pathlib import Path

    example = (Path(__file__).resolve().parents[1] / "cluster.example.toml").read_text()
    assert '# worker_vae_decode = "auto"' in example
