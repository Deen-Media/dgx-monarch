"""Refuse artifacts for which this release has no forward, before loading.

The driver and worker share upstream_gate.py's class P message. The gate must
run before policy and footprint cards and before ensure_live can heal or respawn:
no topology, budget or consent can supply the missing forward. It reads raw
header keys, so forcing an adapter cannot bypass it.

Z-Image L2P is refused; the supported DCT PixelSpace and latent Z-Image neighbors
must pass. This prevents the load-before-refusal failure recorded on 2026-09-02.

CPU-only: synthetic safetensors headers, fake folder_paths and a recording
store replace real checkpoints and ModelStore.
"""
from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    _isolated_process_state,
    _RecordingStore,
)
from dgx_monarch import consent_pending, family_select, residency_mode, upstream_gate
from dgx_monarch.actor import store_fsdp
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import loader_preflight
from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag, refusal

# The three surfaces of one Z-Image backbone: latent, L2P and DCT PixelSpace.
ZIMAGE_KEYS = ["cap_embedder.0.weight", "x_pad_token", "layers.0.attention.qkv.weight"]
L2P_KEYS = [*ZIMAGE_KEYS, "all_x_embedder.16-1.weight", "local_decoder.blocks.0.weight"]
DCT_KEYS = [*ZIMAGE_KEYS, "dec_net.blocks.0.weight"]
LTX_KEYS = ["adaln_single.linear.weight", "transformer_blocks.0.attn1.to_q.weight",
            "patchify_proj.weight"]


def _write_checkpoint(path: Path, keys: list[str]) -> Path:
    tensors = {
        key: {"dtype": "BF16", "shape": [1], "data_offsets": [index * 2, index * 2 + 2]}
        for index, key in enumerate(keys)
    }
    header = json.dumps(tensors).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.write(b"\0" * (2 * len(keys)))
    return path


@pytest.fixture
def models(tmp_path, monkeypatch):
    """Three artifacts on disk and the `folder_paths` both sites resolve with."""
    names = {"l2p": L2P_KEYS, "dct": DCT_KEYS, "latent": ZIMAGE_KEYS, "ltx": LTX_KEYS}
    for name, keys in names.items():
        _write_checkpoint(tmp_path / f"{name}.safetensors", keys)

    def get_full_path(kind, name):
        candidate = tmp_path / str(name)
        return str(candidate) if candidate.exists() else None

    fake = types.ModuleType("folder_paths")
    fake.get_full_path = get_full_path
    fake.get_filename_list = lambda _kind: sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    return SimpleNamespace(
        l2p="l2p.safetensors", dct="dct.safetensors", latent="latent.safetensors",
        ltx="ltx.safetensors", missing="absent.safetensors", dir=tmp_path)


def _mesh(**overrides):
    spec = {"topology_preset": "dp2", "handle": None, "worker_args": {}}
    spec.update(overrides)
    return SimpleNamespace(**spec)


def _worker(**overrides):
    spec = {"topology": {}, "world": 1, "store": _RecordingStore()}
    spec.update(overrides)
    return SimpleNamespace(**spec)


def _driver_refusal(models, monkeypatch, name, **mesh_overrides) -> str:
    with pytest.raises(UnsupportedModelError) as raised:
        DGXMonarchUNETLoader().load(_mesh(**mesh_overrides), name)
    return str(raised.value)


@pytest.mark.parametrize("preset", ["single", "dp2"])
def test_the_loader_node_refuses_an_l2p_artifact_before_ensure_live(
        models, monkeypatch, preset):
    """Nothing below the gate may run: no policy card, no price, no heal.

    Not "before the fleet is live": the Init node calls ``get_mesh`` and can
    bring a fleet up before any loader runs. What this pins is that no answer
    the checkpoint reaches runs first, and that ``ensure_live`` never heals or
    respawns for a file this release cannot run.

    Both worlds, named rather than inherited: `single` is world 1 and `dp2` is
    world 2. The refusal is about the file, so it cannot wait for a second box.
    """
    charged: list[str] = []

    def explode(*_args, **_kwargs):
        charged.append("ran")
        raise AssertionError("an answer below the gate ran")

    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology", explode)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", explode)
    from dgx_monarch.nodes import loaders

    monkeypatch.setattr(loaders, "ensure_live", explode)
    with pytest.raises(UnsupportedModelError):
        DGXMonarchUNETLoader().load(_mesh(topology_preset=preset), models.l2p)
    assert charged == [], "the gate let a load reach the answers below it"


def test_the_refusal_is_class_p_with_no_guard_and_no_panel_sentence(
        models, monkeypatch):
    text = _driver_refusal(models, monkeypatch, models.l2p)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None
    assert tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None
    assert tag.waivable is False
    assert "DGX Monarch panel" not in text
    assert consent_pending.pending_cards() == []
    assert "docs/TROUBLESHOOTING.md #91" in text


def test_the_card_names_the_upstream_gate_and_the_working_alternatives(
        models, monkeypatch):
    text = _driver_refusal(models, monkeypatch, models.l2p)
    for token in ("#279", "14055", "local_decoder", "dec_net", "2.8", "129",
                  "nothing was quarantined"):
        assert token in text
    assert "latent Z-Image checkpoint" in text
    assert "DCT PixelSpace checkpoint" in text


def test_the_body_constant_carries_no_tag_of_its_own():
    """The tag is built at the raise, never at import."""
    assert "[dgxm:" not in upstream_gate.ZIMAGE_L2P_REFUSAL
    assert len(upstream_gate.ZIMAGE_L2P_REFUSAL) <= 4000
    assert refusal(RefusalClass.PHYSICS, upstream_gate.ZIMAGE_L2P_REFUSAL)
    assert upstream_gate.TROUBLESHOOTING == 91


def test_the_worker_funnel_refuses_ahead_of_the_stock_preflight(models):
    worker = _worker()
    with pytest.raises(UnsupportedModelError):
        store_fsdp.ensure(worker, models.l2p, {}, [])
    assert worker.store.calls == [], "the store was entered on a refused artifact"


def test_the_worker_sentence_is_byte_for_byte_the_driver_sentence(
        models, monkeypatch):
    driver = _driver_refusal(models, monkeypatch, models.l2p)
    with pytest.raises(UnsupportedModelError) as raised:
        store_fsdp.ensure(_worker(), models.l2p, {}, [])
    assert str(raised.value) == driver


@pytest.mark.parametrize("surface", ["dct", "latent"])
def test_a_working_artifact_loads_untouched(models, monkeypatch, surface):
    """The near neighbours the discriminator separates. Both are validated."""
    name = getattr(models, surface)
    loader_preflight.preflight_upstream_gated_artifact(name)
    worker = _worker()
    store_fsdp.ensure(worker, name, {}, [])
    assert len(worker.store.calls) == 1


def test_a_missing_or_unreadable_path_is_a_no_op_not_a_refusal(models):
    loader_preflight.preflight_upstream_gated_artifact(models.missing)
    worker = _worker()
    store_fsdp.ensure(worker, models.missing, {}, [])
    assert len(worker.store.calls) == 1


def test_a_forced_family_override_cannot_walk_past_the_gate(models, monkeypatch):
    """The predicate reads the header, not the adapter the graph asked for.

    A forced family names which adapter re-expresses the math. It cannot supply
    a forward this release does not carry, and this gate keeps the host alive,
    so no override may bypass it.
    """
    text = _driver_refusal(models, monkeypatch, models.l2p,
                           worker_args={"family_adapter": "chroma"})
    assert "local_decoder" in text
    worker = _worker(family_override="chroma")
    with pytest.raises(UnsupportedModelError):
        store_fsdp.ensure(worker, models.l2p, {"family_adapter": "chroma"}, [])
    assert worker.store.calls == []


def test_the_gate_answers_before_the_comfy_managed_policy_card(models, monkeypatch):
    text = _driver_refusal(models, monkeypatch, models.l2p,
                           topology_preset="uly2+fsdp",
                           worker_args={"comfy_managed": True})
    assert "local_decoder" in text
    assert residency_mode.COMFY_MANAGED_FSDP_REFUSAL not in text


@pytest.mark.parametrize("surface", ["dct", "latent"])
def test_the_gate_adds_a_refusal_and_removes_none(models, surface):
    """The gate removes no earlier refusal: the artifacts it passes still reach
    the comfy-managed policy card, which refuses them on uly2+fsdp."""
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as raised:
        DGXMonarchUNETLoader().load(
            _mesh(topology_preset="uly2+fsdp", worker_args={"comfy_managed": True}),
            getattr(models, surface))
    assert residency_mode.COMFY_MANAGED_FSDP_REFUSAL in str(raised.value)


def test_263_still_reports_and_does_not_refuse(models, caplog):
    """A missing LTX config key has warned since 2026-08-19, and this gate does not refuse it."""
    assert family_select._CONFIG_REQUIRED_FAMILIES == frozenset({"ltx"})
    path = str(models.dir / models.ltx)
    with caplog.at_level("WARNING"):
        family_select.warn_auto_admits_checkpoint("ltx", path)
    assert any("carries no" in record.getMessage() for record in caplog.records)
    assert upstream_gate.refuses(path) is False
    for name in (models.dct, models.latent):
        assert upstream_gate.refuses(str(models.dir / name)) is False
