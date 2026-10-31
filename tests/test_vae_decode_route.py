"""Which decodes split, which stay on the driver, and what each says (no ComfyUI needed).

The family table admits MiniMax H3 through both stock decode nodes and the
Wan, HunyuanVideo and LTX video VAEs through VAE Decode (Tiled) only. A still at
or under 1.2 MP, an unknown family, a VAE with no loader file, a mesh that is
not set up and a mesh with no rank on another GPU all run the reference decode
and log one INFO line naming why. The knob defaults to off. The wire carries
the unit's latent, the VAE's file identity and dtype, the output dtype and the
driver's decode environment, and a reply is used only as an ``ok`` CPU tensor
of the expected dtype and shape.
"""
from __future__ import annotations

import logging
import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import vae_decode_remote, vae_decode_route, vae_decode_telemetry
from dgx_monarch.vae_decode_families import (
    DECODE,
    DECODE_TILED,
    H3_CHUNK,
    MULTIDIM,
    STILL_SPLIT_MIN_PIXELS,
    classify,
    small_still,
)
from dgx_monarch.vae_decode_telemetry import DecodeRecord


def _fsm(qualified, **attributes):
    module, _, name = qualified.rpartition(".")
    cls = type(name, (), {"modules": lambda self: iter([self, *attributes.pop("_children", [])])})
    cls.__module__ = module
    instance = cls()
    for key, value in attributes.items():
        setattr(instance, key, value)
    return instance


def _vae(qualified, *, latent_dim=3, formula=None, channels=16, handles_tiling=False, not_video=False,
         ratio=8, **fsm_attributes):
    return SimpleNamespace(first_stage_model=_fsm(qualified, **fsm_attributes), latent_dim=latent_dim,
                           upscale_index_formula=formula, latent_channels=channels, handles_tiling=handles_tiling,
                           not_video=not_video, spacial_compression_decode=lambda: ratio)


def test_the_family_table():
    h3, why = classify(_vae("comfy.ldm.minimax.vae.MiniMaxH3VideoVAE", handles_tiling=True, decoder=object()))
    assert h3 is not None and h3.name == "minimax_h3" and h3.kind == H3_CHUNK
    assert h3.entries == {DECODE, DECODE_TILED} and not why
    cases = {
        "wan21": _vae("comfy.ldm.wan.vae.WanVAE", formula=(4, 8, 8)),
        "wan22": _vae("comfy.ldm.wan.vae2_2.WanVAE", formula=(4, 16, 16), channels=48),
        "ltx": _vae("comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE", formula=(8, 32, 32),
                    timestep_conditioning=False),
        "hunyuan_video": _vae("comfy.ldm.models.autoencoder.AutoencoderKL", formula=(4, 8, 8)),
    }
    for name, vae in cases.items():
        family, why = classify(vae)
        assert family is not None and family.name == name and family.kind == MULTIDIM, (name, why)
        assert family.entries == {DECODE_TILED}
    decoder = _fsm("comfy.ldm.hunyuan_video.vae_refiner.Decoder")
    family, _ = classify(_vae("comfy.ldm.models.autoencoder.AutoencodingEngine", formula=(4, 16, 16),
                              channels=32, decoder=decoder))
    assert family is not None and family.name == "hunyuan_video15"
    diffusion, _ = classify(_vae("comfy.ldm.lightricks.vae.na_diffusion_decoder.CausalDiffusionVAE",
                                 formula=(8, 32, 32)))
    assert diffusion is not None and diffusion.name == "ltx"


@pytest.mark.parametrize("vae,expected", [
    (_vae("comfy.ldm.cogvideo.vae.AutoencoderKLCogVideoX"), "CogVideoX tiles frames inside its spatial loop"),
    (_vae("comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE", timestep_conditioning=True),
     "draws unseeded noise"),
    (_vae("comfy.ldm.wan.vae2_2.WanVAE", latent_dim=2), "no split for comfy.ldm.wan.vae2_2.WanVAE"),
    (_vae("comfy.ldm.models.autoencoder.AutoencodingEngine", latent_dim=2), "no split for"),
    (_vae("comfy.ldm.minimax.vae.MiniMaxH3VideoVAE", handles_tiling=False, decoder=object()),
     "some other way than its own chunked decode"),
    (SimpleNamespace(first_stage_model=None), "the VAE has no decoder model"),
])
def test_families_that_stay_on_the_driver_say_why(vae, expected):
    family, why = classify(vae)
    assert family is None and expected in why


def test_the_small_still_rule():
    vae = _vae("comfy.ldm.wan.vae.WanVAE", ratio=8)
    assert small_still(vae, torch.empty(1, 16, 1, 128, 128)) == 1024 * 1024
    assert small_still(vae, torch.empty(1, 16, 128, 128)) == 1024 * 1024
    assert small_still(vae, torch.empty(1, 16, 1, 192, 192)) is None  # 2.36 MP
    assert small_still(vae, torch.empty(1, 16, 5, 64, 64)) is None  # video
    assert STILL_SPLIT_MIN_PIXELS == 1_200_000


def test_the_knob_defaults_to_off():
    assert vae_decode_route.knob(SimpleNamespace(config=SimpleNamespace(worker_args={}))) == "off"
    assert vae_decode_route.knob(SimpleNamespace(config=SimpleNamespace(worker_args=None))) == "off"
    on = SimpleNamespace(config=SimpleNamespace(worker_args={"worker_vae_decode": "auto"}))
    assert vae_decode_route.knob(on) == "auto"


class _Handle(SimpleNamespace):
    """A mesh handle double the route can hold by weak reference, as it holds a MeshHandle."""


def _handle(**overrides):
    values = {"lock": threading.RLock(), "setup_key": ("topology",), "worker_args_key": ("policy",),
              "setup_generation": 1, "setup_cleanup_state": None, "defunct": False, "teardown_complete": False,
              "world": 2, "gpus_per_host": 1,
              "config": SimpleNamespace(worker_args={"worker_vae_decode": "auto"}, source="c.toml", hosts=())}
    values.update(overrides)
    return _Handle(**values)


def test_arm_installs_only_when_the_knob_is_auto(monkeypatch, caplog):
    installed = []
    monkeypatch.setattr(vae_decode_route, "install", lambda: installed.append(True))
    monkeypatch.setattr(vae_decode_route, "_ARMED", None)
    caplog.set_level(logging.INFO, logger="dgx_monarch.vae_decode")
    off = _handle(config=SimpleNamespace(worker_args={}, source="c.toml", hosts=()))
    vae_decode_route.arm(off)
    assert not installed and vae_decode_route.armed_handle() is None
    on = _handle()
    vae_decode_route.arm(on)
    assert installed and vae_decode_route.armed_handle() is on
    assert 'Worker-assisted VAE decode is on ([worker_args] worker_vae_decode = "auto")' in caplog.text
    vae_decode_route.arm(off)
    assert vae_decode_route.armed_handle() is None
    assert "Worker-assisted VAE decode is off" in caplog.text


def test_arm_never_raises_into_init(monkeypatch, caplog):
    monkeypatch.setattr(vae_decode_route, "install", lambda: (_ for _ in ()).throw(ImportError("no comfy")))
    monkeypatch.setattr(vae_decode_route, "_ARMED", None)
    vae_decode_route.arm(_handle())
    assert "could not arm" in caplog.text


def _open(monkeypatch, vae, samples, entry, *, handle="armed", ranks=(1,), source=None):
    armed = _handle() if handle == "armed" else handle
    monkeypatch.setattr(vae_decode_route, "armed_handle", lambda: armed)
    monkeypatch.setattr(vae_decode_remote, "remote_ranks", lambda _handle: list(ranks))
    monkeypatch.setattr(vae_decode_remote, "vae_source", lambda _vae: source or {
        "folder": "vae", "name": "wan_2.1_vae.safetensors", "signature": "abc", "metadata": None,
        "dtype": "torch.bfloat16"})
    monkeypatch.setattr("dgx_monarch.vae_decode_env.describe", lambda _vae: {"vae_class": "x"})
    return vae_decode_route._open(vae, samples, entry)


@pytest.mark.parametrize("case,expected", [
    ("unknown", "no split for comfy.ldm.models.autoencoder.AutoencodingEngine"),
    ("vae_decode_on_wan", "use VAE Decode (Tiled) to split it"),
    ("small_still", "stills at or under 1.2 MP keep the one-host decode"),
    ("no_mesh", "no live DGX Monarch mesh"),
    ("not_set_up", "the mesh has not finished setup"),
    ("no_ranks", "no Worker rank sits on a GPU other than this driver's"),
    ("no_file", "did not come from a VAE loader file"),
])
def test_a_decode_that_stays_here_logs_one_info_line(monkeypatch, caplog, case, expected):
    caplog.set_level(logging.INFO, logger="dgx_monarch.vae_decode")
    wan = _vae("comfy.ldm.wan.vae.WanVAE", formula=(4, 8, 8))
    video = torch.empty(1, 16, 21, 60, 104)
    kwargs: dict = {}
    vae, samples, entry = wan, video, DECODE_TILED
    if case == "unknown":
        vae = _vae("comfy.ldm.models.autoencoder.AutoencodingEngine", latent_dim=2)
    elif case == "vae_decode_on_wan":
        entry = DECODE
    elif case == "small_still":
        samples = torch.empty(1, 16, 1, 128, 128)
    elif case == "no_mesh":
        kwargs["handle"] = None
    elif case == "not_set_up":
        kwargs["handle"] = _handle(setup_key=None)
    elif case == "no_ranks":
        kwargs["ranks"] = ()
    elif case == "no_file":
        kwargs["source"] = "the VAE did not come from a VAE loader file, so no Worker can load the same bytes"
    assert _open(monkeypatch, vae, samples, entry, **kwargs) is None
    lines = [record.getMessage() for record in caplog.records if record.name == "dgx_monarch.vae_decode"]
    assert len(lines) == 1 and lines[0].startswith("VAE decode not split (") and expected in lines[0]
    assert vae_decode_telemetry.last()["split"] is False


def test_a_minimax_h3_still_stays_here(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="dgx_monarch.vae_decode")
    h3 = _vae("comfy.ldm.minimax.vae.MiniMaxH3VideoVAE", handles_tiling=True, decoder=object(), ratio=16)
    assert _open(monkeypatch, h3, torch.empty(1, 24, 1, 96, 96), DECODE) is None
    assert "a MiniMax H3 still is one temporal chunk" in caplog.text
    assert _open(monkeypatch, h3, torch.empty(1, 24, 37, 48, 84), DECODE) is not None


def test_an_eligible_decode_opens_a_route(monkeypatch):
    route = _open(monkeypatch, _vae("comfy.ldm.wan.vae.WanVAE", formula=(4, 8, 8)),
                  torch.empty(1, 16, 21, 60, 104), DECODE_TILED, ranks=(1, 2, 3))
    assert route is not None and route.family.name == "wan21" and route.ranks == [1, 2, 3]
    assert route.source["name"] == "wan_2.1_vae.safetensors"


def test_remote_ranks_skip_the_drivers_own_gpu(monkeypatch):
    monkeypatch.setattr(vae_decode_remote, "_driver_gpu_index", lambda: 0)
    hosts = (SimpleNamespace(name="head"), SimpleNamespace(name="sibling"))
    pair = _handle(world=2, gpus_per_host=1, config=SimpleNamespace(worker_args={}, source="c", hosts=hosts))
    monkeypatch.setattr("dgx_monarch.monarch_policy_io.driver_host", lambda config: "head")
    assert vae_decode_remote.remote_ranks(pair) == [1]
    monkeypatch.setattr("dgx_monarch.monarch_policy_io.driver_host", lambda config: "sibling")
    assert vae_decode_remote.remote_ranks(pair) == [0]
    monkeypatch.setattr("dgx_monarch.monarch_policy_io.driver_host", lambda config: "")
    assert vae_decode_remote.remote_ranks(pair) == [0, 1]
    local = _handle(world=4, gpus_per_host=4, config=SimpleNamespace(worker_args={}, source="", hosts=()))
    assert vae_decode_remote.remote_ranks(local) == [1, 2, 3]


def test_the_wire_shape():
    base = vae_decode_remote.request_base(
        MULTIDIM, {"folder": "vae", "name": "a.safetensors", "signature": "s", "metadata": {"k": "v"},
                   "dtype": "torch.bfloat16"}, {"vae_class": "c"}, torch.float32)
    assert set(base) == {"schema", "kind", "vae", "env", "output_dtype"}
    assert base["schema"] == 1 and base["kind"] == MULTIDIM and base["output_dtype"] == "torch.float32"
    assert vae_decode_remote.request_base(H3_CHUNK, {}, {}, None)["output_dtype"] is None
    sent = {}

    class Endpoint:
        def call_one(self, request):
            sent.update(request)
            return "future"

    worker = SimpleNamespace(vae_decode=Endpoint())
    handle = _handle(world=2, gpus_per_host=1, workers=SimpleNamespace(
        extent=SimpleNamespace(labels=["hosts", "gpus"]), slice=lambda **coords: worker))
    latent = torch.zeros(1, 16, 5, 8, 8, dtype=torch.bfloat16)
    send = vae_decode_remote.sender(handle, None, base, [latent])
    calls = []
    import dgx_monarch.mesh_setup as mesh_setup

    original = mesh_setup.dispatch_endpoint
    try:
        mesh_setup.dispatch_endpoint = lambda handle, name, send_, setup_token=None: (
            calls.append((name, setup_token)) or send_())
        assert send(1, 0) == "future"
    finally:
        mesh_setup.dispatch_endpoint = original
    assert calls == [("vae_decode", None)]
    assert sent["unit"] == 0 and sent["input"] is latent and sent["input"].dtype == torch.bfloat16


def test_a_reply_is_used_only_as_the_expected_tensor():
    accept = vae_decode_remote.acceptor(torch.float32, lambda unit: (1, 3, None, None))
    good = {"schema": 1, "ok": True, "unit": 2, "rank": 1, "host": "sibling", "output": torch.zeros(1, 3, 8, 9)}
    assert accept(good) is None
    assert "declined: not the driver's file" in accept({**good, "ok": False, "reason": "not the driver's file"})
    assert "returned torch.float16" in accept({**good, "output": torch.zeros(1, 3, 8, 9, dtype=torch.float16)})
    assert "returned shape (1, 4, 8, 9)" in accept({**good, "output": torch.zeros(1, 4, 8, 9)})
    assert "no schema-1 reply" in accept({**good, "schema": 2})
    assert "no CPU tensor" in accept({**good, "output": [1, 2]})


def test_the_vae_source_names_the_loader_file(monkeypatch, tmp_path):
    root = tmp_path / "vae"
    (root / "video").mkdir(parents=True)
    path = root / "video" / "wan.safetensors"
    path.write_bytes(b"\0" * 4096)

    def load_vae_patcher(vae_path, metadata=None, device=None):
        return None

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_folder_paths = lambda folder: [str(root)] if folder == "vae" else []
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    vae = SimpleNamespace(vae_dtype=torch.bfloat16, patcher=SimpleNamespace(
        cached_patcher_init=(load_vae_patcher, (str(path), {"config": "{}"}, None))))
    source = vae_decode_remote.vae_source(vae)
    assert isinstance(source, dict)
    assert source["folder"] == "vae" and source["name"] == "video/wan.safetensors"
    assert source["metadata"] == {"config": "{}"} and source["dtype"] == "torch.bfloat16"
    assert source["signature"] not in ("unreadable", "unstable")
    checkpoint = SimpleNamespace(patcher=SimpleNamespace(cached_patcher_init=(object, (str(path),), 2)))
    assert "did not come from a VAE loader file" in vae_decode_remote.vae_source(checkpoint)
    outside = SimpleNamespace(vae_dtype=torch.float16, patcher=SimpleNamespace(
        cached_patcher_init=(load_vae_patcher, (str(tmp_path / "elsewhere.safetensors"), None, None))))
    assert "outside ComfyUI's vae folders" in vae_decode_remote.vae_source(outside)


def test_the_decode_record_and_its_lines(caplog):
    caplog.set_level(logging.INFO, logger="dgx_monarch.vae_decode")
    record = DecodeRecord(family="minimax_h3", entry="decode", units=7, local=4, remote={1: 3}, seconds=33.1)
    record.split = True
    assert vae_decode_telemetry.info_line(record) == (
        "VAE decode split over 2 ranks (minimax_h3, VAE Decode): 7 units, 4 here, 3 on rank 1, 33.1 s")
    assert vae_decode_telemetry.warning_line(record) is None
    record.failures.append("rank 1: unit 6: TimeoutError: no reply")
    record.fallback = 1
    assert vae_decode_telemetry.warning_line(record).startswith(
        "Worker-assisted VAE decode fell back to this driver for 1 of 7 units, so the image is the one-host "
        "decode: rank 1: unit 6")
    vae_decode_telemetry.publish(record)
    assert "VAE decode split over 2 ranks" in caplog.text and "fell back to this driver" in caplog.text
    last = vae_decode_telemetry.last()
    assert last["ranks"] == 2 and last["remote"] == {"1": 3} and last["split"] is True
    render = vae_decode_telemetry.with_vae_decode({"last": {}})
    assert render["vae_decode"]["family"] == "minimax_h3"
    from dgx_monarch import telemetry

    assert any(event.get("kind") == "vae_decode" for event in telemetry.events_tail(16))


def test_each_tile_has_its_full_expected_shape_and_its_bytes_on_the_board():
    from dgx_monarch.vae_decode_plan import Geometry, multidim_output_shapes

    plan = [Geometry(0, (1, 16, 2, 8, 8), (1, 1, 1, 1, 1)), Geometry(5, (1, 16, 1, 6, 8), (1, 1, 1, 1, 1))]
    wan = (lambda a: max(0, a * 4 - 3), 8, 8)
    assert multidim_output_shapes(plan, wan, 3) == [(1, 3, 5, 64, 64), (1, 3, 1, 48, 64)]
    flat = [Geometry(0, (1, 4, 37, 53), (1, 1, 1, 1))]
    assert multidim_output_shapes(flat, 8, 3) == [(1, 3, 296, 424)]
    board = vae_decode_route._Route._board([(1, 3, 5, 64, 64), (1, 3, 1, 48, 64)], torch.float32)
    assert board._unit_bytes == [3 * 5 * 64 * 64 * 4, 3 * 48 * 64 * 4]
    assert board._budget == vae_decode_remote.REPLY_BUDGET_BYTES == 2 << 30
    bf16 = vae_decode_route._Route._board([(1, 3, 28, 768, 1344)], torch.bfloat16)
    assert bf16._unit_bytes == [3 * 28 * 768 * 1344 * 2]
