"""A sentinel identity never composes a gate or consent lookup key.

``artifact_signature`` answers ``unreadable`` or ``unstable`` when it cannot
fingerprint a file, and the reader-side composites write ``missing`` for a name
that resolved to no path. gate_artifacts.refuse_unresolved says why no lookup
key may hold any of the three, and that the writer side refuses them too; these
tests cover the readers.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
import dgx_monarch.nodes.render_quarantine as quarantine_mod
from dgx_monarch.nodes import consent_rescue

UNET = "model.safetensors"
LORA = "lora.safetensors"


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """A fake folder_paths; a test may change one name's resolution."""
    checkpoint = tmp_path / UNET
    checkpoint.write_bytes(b"weights" * 64)
    lora = tmp_path / LORA
    lora.write_bytes(b"delta" * 64)
    output = tmp_path / "output"
    output.mkdir()
    resolved = {UNET: str(checkpoint), LORA: str(lora)}
    module = types.ModuleType("folder_paths")
    module.get_full_path = lambda _kind, name: resolved.get(name)  # type: ignore[attr-defined]
    module.get_output_directory = lambda: str(output)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", module)
    return SimpleNamespace(resolved=resolved, tmp_path=tmp_path)


def _model(loras=()):
    return SimpleNamespace(
        unet_name=UNET,
        options={},
        loras=tuple(loras),
        mesh=SimpleNamespace(worker_args={"lora_low_rss": True}),
    )


def test_a_resolved_stack_still_composes_a_key(folders):
    key, artifacts = gate_mod._combo_of(_model([{"name": LORA}]))
    assert key
    assert len(artifacts.current) == 64
    combo, signature = consent_rescue.combo_identity(
        UNET, {}, folders.resolved[UNET], [{"name": LORA}])
    assert combo == key
    assert signature.current == artifacts.current


@pytest.mark.parametrize("name", [UNET, LORA])
def test_an_unresolved_name_refuses_the_gate_key(folders, name):
    folders.resolved[name] = None
    with pytest.raises(RuntimeError, match="stable model artifact identity"):
        gate_mod._combo_of(_model([{"name": LORA}]))


def test_an_unreadable_file_refuses_the_gate_key(folders):
    folders.resolved[UNET] = str(folders.tmp_path / "gone.safetensors")
    with pytest.raises(RuntimeError, match="stable model artifact identity"):
        gate_mod._combo_of(_model())


@pytest.mark.parametrize("name", [UNET, LORA])
def test_an_unresolved_name_refuses_the_consent_key(folders, name):
    folders.resolved[name] = None
    path = folders.resolved[UNET] or str(folders.tmp_path / "gone.safetensors")
    with pytest.raises(RuntimeError, match="stable model artifact identity"):
        consent_rescue.combo_identity(UNET, {}, path, [{"name": LORA}])


def test_a_sentinel_never_skips_the_persisted_quarantine(folders, monkeypatch):
    """An unidentified artifact never reaches the ledger; the quarantine check turns every guarded residency lever off.

    Without the guard, a missing LoRA path digests to a value no FAIL row
    carries, the lookup reads "nothing recorded", and the render goes out with
    every guarded residency lever on.
    """
    folders.resolved[LORA] = None
    model = _model([{"name": LORA}])
    reads: list[str] = []

    class Ledger:
        def __init__(self, _directory):
            reads.append("constructed")

        def lookup_with_entry(self, *_args):
            reads.append("lookup")
            return "clear", None

    monkeypatch.setattr("dgx_monarch.gate_ledger.GateLedger", Ledger)
    common._AUTO_GATE_ACTIVE.on = False
    quarantine_mod._enforce_persisted_quarantine(model)
    assert not reads, "an unidentified artifact must never reach the ledger"
    assert model.mesh.worker_args["lora_low_rss"] is False
    assert model.mesh.worker_args["slab_weights"] is False
