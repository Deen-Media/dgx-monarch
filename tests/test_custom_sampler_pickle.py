"""Stock custom-sampler noise crosses actor RPCs with an importable identity."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import cloudpickle
import pytest
import torch

from dgx_monarch.nodes import samplers

_NOISE_SOURCE = """
class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0
"""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def comfy_noise_modules(tmp_path, monkeypatch):
    package_dir = tmp_path / "comfy_extras"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    source = package_dir / "nodes_custom_sampler.py"
    source.write_text(_NOISE_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    module_names = ("comfy_extras.nodes_custom_sampler", "comfy_extras")
    previous = {name: sys.modules.pop(name) for name in module_names if name in sys.modules}
    importlib.invalidate_caches()
    canonical = importlib.import_module("comfy_extras.nodes_custom_sampler")
    # Newer cloudpickle builds cannot pickle a path-shaped module name: ``dumps``
    # tries to import that literal path. Keep the noncanonical loader identity
    # under a valid transient package alias and drop it before the raw load.
    path_name = "comfy_extras._path_loaded_nodes_custom_sampler"
    path_loaded = _load_module(path_name, source)
    try:
        yield canonical, path_loaded, path_name
    finally:
        sys.modules.pop(path_name, None)
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


@pytest.mark.parametrize(
    ("class_name", "args", "expected_seed"),
    [
        ("Noise_RandomNoise", (123,), 123),
        ("Noise_EmptyNoise", (), 0),
    ],
)
def test_path_loaded_stock_noise_fails_raw_but_canonical_copy_unpickles(
    comfy_noise_modules, class_name, args, expected_seed,
):
    _, path_loaded, path_name = comfy_noise_modules
    noise = getattr(path_loaded, class_name)(*args)
    raw_payload = cloudpickle.dumps(noise)

    wire_noise = samplers._canonical_noise_for_wire(noise)
    wire_payload = cloudpickle.dumps(wire_noise)
    assert type(wire_noise).__module__ == "comfy_extras.nodes_custom_sampler"
    assert wire_noise.seed == expected_seed

    sys.modules.pop(path_name)
    sys.modules.pop("comfy_extras.nodes_custom_sampler")
    sys.modules.pop("comfy_extras")
    with pytest.raises(ModuleNotFoundError):
        cloudpickle.loads(raw_payload)
    restored = cloudpickle.loads(wire_payload)
    assert type(restored).__module__ == "comfy_extras.nodes_custom_sampler"
    assert restored.seed == expected_seed


def test_canonical_and_third_party_noise_objects_are_unchanged(comfy_noise_modules, tmp_path):
    canonical, _, _ = comfy_noise_modules
    stock = canonical.Noise_RandomNoise(7)
    assert samplers._canonical_noise_for_wire(stock) is stock

    other_source = tmp_path / "third_party_noise.py"
    other_source.write_text("class Noise_RandomNoise:\n    pass\n", encoding="utf-8")
    third_party_module = _load_module(str(other_source.with_suffix("")), other_source)
    third_party = third_party_module.Noise_RandomNoise()
    try:
        assert samplers._canonical_noise_for_wire(third_party) is third_party
    finally:
        sys.modules.pop(third_party_module.__name__, None)


def test_sampler_custom_changes_only_stock_noise_identity(
    comfy_noise_modules, monkeypatch,
):
    _, path_loaded, _ = comfy_noise_modules
    captured = {}
    model = object()
    sampler = object()
    noise = path_loaded.Noise_RandomNoise(9)

    def run_render(_model, request, latent, **kwargs):
        captured.update(request=request, latent=latent, kwargs=kwargs)
        return {"samples": latent["samples"]}

    monkeypatch.setattr(samplers, "run_render", run_render)
    latent = {"samples": torch.zeros(1, 4, 2, 2)}
    guider = {"model": model, "spec": {"kind": "basic"}, "cfg": 1.0}
    samplers.DGXMonarchSamplerCustom().sample(
        noise, guider, sampler, torch.tensor([1.0, 0.0]), latent,
    )

    request = captured["request"]
    assert request["sampler_object"] is sampler
    assert request["noise_object"].seed == 9
    assert type(request["noise_object"]).__module__ == "comfy_extras.nodes_custom_sampler"
