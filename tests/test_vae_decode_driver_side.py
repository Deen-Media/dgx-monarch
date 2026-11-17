"""Real ComfyUI: what the driver decides before a Worker-assisted VAE decode splits.

A small real Wan 2.1 VAE file, loaded by ComfyUI's own Load VAE from a vae
folder, goes through the installed decode wrappers on the CPU path:

* the file a Worker loads is named as Load VAE lists it, through a symlinked
  file or folder and for a name that starts with two dots;
* a decode called with ``samples_in`` by keyword reaches ComfyUI's decode
  through the decode relief and the split's wrapper alike;
* a driver whose VAE decodes on the CPU (``--cpu-vae`` or ``--cpu``) keeps
  the whole decode, since a Worker decodes on its GPU;
* a decode environment the driver cannot read ends the split, never the
  stock decode;
* a VAE with quantized layers stays on the driver, since Load VAE builds them
  in a compute dtype of its own and a Worker builds them in the VAE dtype.

Skipped where ComfyUI is not importable, as in the ordinary CI job.
"""
from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import vae_decode_relief, vae_decode_remote, vae_decode_route
from dgx_monarch.gate_artifacts import artifact_signature
from dgx_monarch.vae_decode_families import DECODE, DECODE_TILED, classify
from test_vae_decode_split_comfy import _handle, _Ranks, same_bytes
from test_vae_decode_split_comfy import comfy as comfy  # the fixture tests ask for by name

TILED = {"tile_x": 8, "tile_y": 8, "overlap": 2, "tile_t": 2, "overlap_t": 1}


@pytest.fixture
def vae_file(comfy, tmp_path):
    """A small real Wan 2.1 VAE (ComfyUI reads its width from the file) in a store outside the vae folder."""
    from comfy.ldm.wan.vae import WanVAE
    from safetensors.torch import save_file

    model = WanVAE(dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                   temperal_downsample=[False, True, True])
    generator = torch.Generator().manual_seed(5)
    state = {key: torch.randn(value.shape, generator=generator) * 0.05 for key, value in model.state_dict().items()}
    store = tmp_path / "store"
    store.mkdir()
    path = store / "wan_small.safetensors"
    save_file(state, str(path))
    return path


@pytest.fixture
def vae_folder(comfy, tmp_path, monkeypatch):
    """Point ComfyUI's vae folder at an empty folder of this test."""
    import folder_paths

    root = tmp_path / "models" / "vae"
    root.mkdir(parents=True)
    _paths, extensions = folder_paths.folder_names_and_paths["vae"]
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "vae", ([str(root)], extensions))
    return root


def load_vae(name):
    """ComfyUI's Load VAE node, as a graph runs it."""
    return __import__("nodes").VAELoader().load_vae(name)[0]


@pytest.fixture
def wrappers(comfy, monkeypatch):
    """Install the decode relief and the split's wrappers as a driver does; put ComfyUI's back after."""
    import comfy.samplers

    monkeypatch.setattr(comfy.sd.VAE, "decode", comfy.sd.VAE.decode)
    monkeypatch.setattr(comfy.sd.VAE, "decode_tiled", comfy.sd.VAE.decode_tiled)
    monkeypatch.setattr(comfy.utils, "tiled_scale_multidim", comfy.utils.tiled_scale_multidim)
    monkeypatch.setattr(comfy.samplers.CFGGuider, "sample", comfy.samplers.CFGGuider.sample)
    vae_decode_relief.install()
    vae_decode_route.install()


@pytest.fixture
def armed(monkeypatch, ranks_seen):
    """An armed, set-up mesh with one Worker rank on another GPU, its units recorded."""
    monkeypatch.setattr(vae_decode_route, "_ARMED", (lambda: None, "cluster.toml"))
    monkeypatch.setattr(vae_decode_route, "armed_handle", _handle)
    monkeypatch.setattr(vae_decode_remote, "remote_ranks", lambda _handle: [1])
    return ranks_seen


@pytest.fixture
def ranks_seen(monkeypatch):
    simulated = _Ranks(lambda latent: latent)
    monkeypatch.setattr(vae_decode_remote, "sender", simulated.sender)
    monkeypatch.setattr(vae_decode_remote, "waiter", simulated.waiter)
    yield simulated
    simulated.pool.shutdown(wait=True)


def not_split_lines(caplog):
    return [record.getMessage() for record in caplog.records if "not split" in record.getMessage()]


def test_load_vae_names_each_file_as_it_lists_it_through_symlinks_and_dotted_names(comfy, vae_file, vae_folder):
    """The Worker resolves the name it is sent in its own vae folder; the name is the one Load VAE lists."""
    import folder_paths

    from dgx_monarch.actor.comfy_bridge import resolve_model_path

    store = vae_file.parent
    (store / "blobs").mkdir()
    (store / "blobs" / "0a1b2c").write_bytes(vae_file.read_bytes())
    (store / "videos").mkdir()
    (store / "videos" / "clip.safetensors").write_bytes(vae_file.read_bytes())
    (vae_folder / "plain.safetensors").write_bytes(vae_file.read_bytes())
    # A file and a folder kept on other storage, and a relative link into a download cache.
    (vae_folder / "linked.safetensors").symlink_to(vae_file)
    (vae_folder / "video").symlink_to(store / "videos", target_is_directory=True)
    (vae_folder / "cached.safetensors").symlink_to(os.path.relpath(store / "blobs" / "0a1b2c", vae_folder))
    # A second name for a file in the same folder, and names that start with two dots.
    (vae_folder / "sub").mkdir()
    (vae_folder / "sub" / "real_blob.safetensors").write_bytes(vae_file.read_bytes())
    (vae_folder / "alias.safetensors").symlink_to("sub/real_blob.safetensors")
    (vae_folder / "..dots.safetensors").write_bytes(vae_file.read_bytes())
    (vae_folder / "..archive").mkdir()
    (vae_folder / "..archive" / "old.safetensors").write_bytes(vae_file.read_bytes())

    listed = set(folder_paths.get_filename_list("vae"))
    names = ["plain.safetensors", "linked.safetensors", "video/clip.safetensors", "cached.safetensors",
             "alias.safetensors", "..dots.safetensors", "..archive/old.safetensors"]
    assert set(names) <= listed, listed
    signature = artifact_signature(str(vae_file))
    for name in names:
        source = vae_decode_remote.vae_source(load_vae(name))
        assert isinstance(source, dict), (name, source)
        assert (source["folder"], source["name"], source["signature"]) == ("vae", name, signature), source
        # A Worker with the same layout finds the same bytes under that name.
        assert artifact_signature(resolve_model_path("vae", source["name"])) == signature


def test_the_signature_is_of_the_file_load_vae_opened_through_a_root_with_dotdot_after_a_symlink(
        comfy, vae_file, tmp_path, monkeypatch):
    """A vae folder registered as ``<symlink>/../vae``, as a custom node can register one from a
    symlinked pack folder: the OS resolves the link before the "..", so Load VAE opens one file,
    while the same path read as text names another. The signature the Workers check is the one
    of the file Load VAE opened."""
    import folder_paths

    (tmp_path / "pack" / "nodes").mkdir(parents=True)
    (tmp_path / "pack" / "vae").mkdir()
    (tmp_path / "vae").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "pack" / "nodes", target_is_directory=True)
    opened = tmp_path / "pack" / "vae" / "wan_small.safetensors"
    opened.write_bytes(vae_file.read_bytes())
    # The path read as text, with the ".." taken off before the link: another file.
    (tmp_path / "vae" / "wan_small.safetensors").write_bytes(b"\0" * 4096)
    _paths, extensions = folder_paths.folder_names_and_paths["vae"]
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "vae",
                        ([os.path.join(str(tmp_path / "link"), os.pardir, "vae")], extensions))
    source = vae_decode_remote.vae_source(load_vae("wan_small.safetensors"))
    assert isinstance(source, dict), source
    assert source["name"] == "wan_small.safetensors"
    assert source["signature"] == artifact_signature(str(opened))


def test_a_keyword_samples_in_reaches_comfyuis_decode_through_both_wrappers(comfy, vae_file, vae_folder,
                                                                             wrappers, armed, caplog):
    (vae_folder / "wan_small.safetensors").write_bytes(vae_file.read_bytes())
    vae = load_vae("wan_small.safetensors")
    latent = torch.randn(1, 16, 3, 12, 12, generator=torch.Generator().manual_seed(6))
    caplog.set_level(logging.INFO, logger="dgx_monarch.vae_decode")
    with torch.inference_mode():
        positional = vae.decode(latent)
        assert same_bytes(vae.decode(samples_in=latent), positional)
        assert same_bytes(vae.decode(samples_in=latent, vae_options={}), positional)
        assert same_bytes(vae.decode_tiled(samples=latent, **TILED), vae.decode_tiled(latent, **TILED))
    # The keyword call was read as a decode like any other: Wan splits only through VAE Decode (Tiled).
    assert any("use VAE Decode (Tiled) to split it" in line for line in not_split_lines(caplog))
    # Off, the wrappers pass the call through as it came.
    vae_decode_route._ARMED = None
    with torch.inference_mode():
        assert same_bytes(vae.decode(samples_in=latent), positional)


def _plain(stock):
    def wrapper(self, *args, **kwargs):
        return stock(self, *args, **kwargs)
    return wrapper


def _named_default(stock):
    def wrapper(self, x=None, *args, **kwargs):
        return stock(self, *((x,) if x is not None else ()), *args, **kwargs)
    return wrapper


@pytest.mark.parametrize("shape", [_plain, _named_default], ids=["(self, *a, **k)", "(self, x=None, *a, **k)"])
def test_the_latent_is_read_under_another_custom_nodes_plain_wrapper(comfy, vae_file, vae_folder, armed,
                                                                      monkeypatch, caplog, shape):
    """Another custom node wrapped VAE.decode and VAE.decode_tiled before the split installed, with
    wrappers that set no __wrapped__, so the signature the split reads is theirs: (self, *args,
    **kwargs), or one that names the latent with a default and so leaves it unbound when the
    caller passes samples_in or samples by name. The latent is still the one the caller passed,
    by position or by name."""
    import comfy.utils

    for entry in ("decode", "decode_tiled"):
        monkeypatch.setattr(comfy.sd.VAE, entry, shape(getattr(comfy.sd.VAE, entry)))
    monkeypatch.setattr(comfy.utils, "tiled_scale_multidim", comfy.utils.tiled_scale_multidim)
    vae_decode_route.install()
    seen = []
    real_open = vae_decode_route._open

    def recorded(vae, samples, entry):
        seen.append(samples)
        return real_open(vae, samples, entry)

    monkeypatch.setattr(vae_decode_route, "_open", recorded)
    (vae_folder / "wan_small.safetensors").write_bytes(vae_file.read_bytes())
    vae = load_vae("wan_small.safetensors")
    # A still of 96x96 pixels: the split keeps it on the driver by the small-still rule.
    still = torch.randn(1, 16, 1, 12, 12, generator=torch.Generator().manual_seed(9))
    caplog.set_level(logging.INFO, logger="dgx_monarch.vae_decode")
    with torch.inference_mode():
        vae.decode(still)
        vae.decode(samples_in=still)
        vae.decode_tiled(still, **TILED)
        vae.decode_tiled(samples=still, **TILED)
    assert len(seen) == 4 and all(samples is still for samples in seen), [type(item) for item in seen]
    tiled = [line for line in not_split_lines(caplog) if "(wan21, VAE Decode (Tiled))" in line]
    assert len(tiled) == 2 and all("stills at or under 1.2 MP" in line for line in tiled), tiled


def test_a_driver_vae_on_the_cpu_keeps_the_whole_decode(comfy, vae_file, vae_folder, wrappers, armed, caplog):
    (vae_folder / "wan_small.safetensors").write_bytes(vae_file.read_bytes())
    vae = load_vae("wan_small.safetensors")
    # ComfyUI's own placement under --cpu here; --cpu-vae gives the same on a GPU driver.
    assert vae.device == comfy.sd.model_management.vae_device() == torch.device("cpu")
    family, _why = classify(vae)
    assert family is not None and family.name == "wan21"
    latent = torch.randn(1, 16, 3, 12, 12, generator=torch.Generator().manual_seed(7))
    caplog.set_level(logging.INFO, logger="dgx_monarch.vae_decode")
    with torch.inference_mode():
        split = vae.decode_tiled(latent, **TILED)
        stock = comfy.sd.VAE.decode_tiled.__wrapped__(vae, latent, **TILED)
    assert same_bytes(split, stock)
    assert armed.seen == [], "no unit went to a Worker"
    lines = not_split_lines(caplog)
    assert len(lines) == 1 and "decodes this VAE on the CPU" in lines[0], lines
    # The same VAE on a GPU driver opens a route.
    vae.device = torch.device("cuda", 0)
    assert vae_decode_route._open(vae, latent, DECODE_TILED) is not None


def test_a_decode_environment_the_driver_cannot_read_ends_the_split_not_the_decode(comfy, vae_file, vae_folder,
                                                                                    armed, monkeypatch, caplog):
    """Torch's precision getter raises once code mixed its legacy and new TF32 settings."""
    (vae_folder / "wan_small.safetensors").write_bytes(vae_file.read_bytes())
    vae = load_vae("wan_small.safetensors")
    vae.device = torch.device("cuda", 0)

    def mixed():
        raise RuntimeError("PyTorch is checking the matmul precision without a specific backend name, Current "
                           "status indicate that you have used mix of the legacy and new APIs to set the matmul "
                           "precision.")

    monkeypatch.setattr(torch, "get_float32_matmul_precision", mixed)
    stock = []
    wrapped = vae_decode_route._entry_wrapper(lambda self, samples, **kwargs: stock.append(samples) or "stock",
                                              DECODE_TILED)
    latent = torch.zeros(1, 16, 3, 12, 12)
    assert wrapped(vae, latent, **TILED) == "stock"
    assert len(stock) == 1 and stock[0] is latent
    assert armed.seen == []
    assert any("runs here" in record.getMessage() and record.levelno == logging.WARNING
               for record in caplog.records), caplog.text


def test_a_vae_with_quantized_layers_stays_on_the_driver(comfy, vae_file):
    """A layer ComfyUI's mixed-precision ops load from a comfy_quant entry marks the VAE."""
    import comfy.ops
    from comfy.ldm.wan.vae import WanVAE

    ops = comfy.ops.mixed_precision_ops(comfy.utils.detect_layer_quantization({"x.comfy_quant": 0}, ""),
                                        torch.float16)
    layer = ops.Linear(16, 8, bias=False)
    layer.load_state_dict({
        "weight": torch.randn(8, 16).to(torch.float8_e4m3fn), "weight_scale": torch.tensor(0.5),
        "comfy_quant": torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8),
    }, strict=False)
    plain = ops.Linear(16, 8, bias=False)
    plain.load_state_dict({"weight": torch.randn(8, 16)}, strict=False)

    def wan(extra):
        fsm = WanVAE(dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                     temperal_downsample=[False, True, True])
        fsm.extra = extra
        return SimpleNamespace(first_stage_model=fsm, latent_dim=3, handles_tiling=False,
                               upscale_index_formula=(4, 8, 8), not_video=False)

    assert classify(wan(plain))[0] is not None
    family, why = classify(wan(layer))
    assert family is None and "quantized layers" in why, why
    # A VAE that stays on the driver for a reason of its own keeps that reason: the quantized
    # Sol-H3-VAE decoder still names its whole-clip decode.
    import inspect

    if "noncausal_decoder" in inspect.signature(comfy.ldm.minimax.vae.MiniMaxH3VideoVAE.__init__).parameters:
        sol = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE(encoder=False, decoder=False, noncausal_decoder=True)
        sol.extra = layer
        family, why = classify(SimpleNamespace(first_stage_model=sol, handles_tiling=True, latent_dim=3,
                                               not_video=False))
        assert family is None and "Sol-H3-VAE" in why, why


def test_a_quantized_stock_minimax_h3_decoder_stays_on_the_driver(comfy):
    """ComfyUI builds an int8 or fp8 MiniMax H3 video VAE's decoder with mixed-precision ops
    (comfy/sd.py) and loads each block linear the file marks with a comfy_quant entry as a
    quantized weight. A one-block stock decoder built that way, one block linear loaded from
    such an entry, declines on the H3 branch; the same decoder unquantized splits."""
    import comfy.ops
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    ops = comfy.ops.mixed_precision_ops(comfy.utils.detect_layer_quantization({"x.comfy_quant": 0}, ""),
                                        torch.float16)
    h3 = MiniMaxH3VideoVAE(operations=ops, num_layers=1, encoder=False, decoder=True)
    vae = SimpleNamespace(first_stage_model=h3, handles_tiling=True, latent_dim=3, not_video=False)
    family, why = classify(vae)
    assert family is not None and family.name == "minimax_h3", why
    linear = h3.decoder.transformer_blocks[0].attn.to_out
    shape = (linear.out_features, linear.in_features)
    missing, _unexpected = h3.load_state_dict({
        "decoder.transformer_blocks.0.attn.to_out.weight": torch.randn(shape).to(torch.float8_e4m3fn),
        "decoder.transformer_blocks.0.attn.to_out.weight_scale": torch.tensor(0.5),
        "decoder.transformer_blocks.0.attn.to_out.comfy_quant":
            torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8),
    }, strict=False)
    assert linear.quant_format == "float8_e4m3fn"
    assert "decoder.transformer_blocks.0.attn.to_out.weight" not in missing
    family, why = classify(vae)
    assert family is None and "quantized layers" in why, why


def test_the_decode_entries_take_the_latent_by_its_stock_name(comfy):
    """The split's entry wrapper reads the latent under ComfyUI's own parameter names."""
    import inspect

    assert list(inspect.signature(comfy.sd.VAE.decode).parameters)[:2] == ["self", "samples_in"]
    assert list(inspect.signature(comfy.sd.VAE.decode_tiled).parameters)[:2] == ["self", "samples"]
    assert DECODE == "decode" and DECODE_TILED == "decode_tiled"
