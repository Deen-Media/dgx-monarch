"""Real ComfyUI: a Worker-assisted VAE decode is byte-identical to the stock tiled decode.

The reference is ComfyUI's own one-host tiled decode. These tests run it, then
run the split (vae_decode_route.py) on the same latent with simulated Worker
ranks: threads that compute each unit they claim with the same deterministic
fake decode, exactly as a Worker would with the same VAE file, dtype and
attention. Every case compares bytes (``same_bytes``):

* ``comfy.utils.tiled_scale_multidim`` for 1-D, 2-D and 3-D tiles, odd sizes,
  overlaps, a batch of two, the Wan and MiniMax H3 index formulas, a single
  tile, more ranks than tiles and more tiles than ranks;
* the stock decode function from ``VAE.decode_tiled_3d`` through the installed
  wrappers, which is the path VAE Decode (Tiled) takes;
* MiniMax H3's own chunked decode (``MiniMaxH3VideoVAE.decode``), with and
  without the preallocated output buffer VAE Decode passes;
* fail-open: a rank that raises, declines, answers in the wrong dtype or
  never answers, and a plan that the stock loop leaves.

What a hardware leg must still show: that a GB10 Worker's real decode of a
unit equals the driver's (docs/VALIDATION.md, Worker-assisted VAE decode).
Skipped where ComfyUI is not importable, as in the ordinary CI job; the
comfy-canary job runs it against comfy master and the newest release.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import vae_decode_remote, vae_decode_route
from dgx_monarch.vae_decode_families import H3_CHUNK, MULTIDIM, Family
from dgx_monarch.vae_decode_telemetry import DecodeRecord
from module_location_helpers import from_checkout


def _comfy_dir() -> str:
    value = os.environ.get("COMFYUI_DIR") or os.environ.get("COMFY_DIR") or "~/ComfyUI"
    return os.path.abspath(os.path.expanduser(value))


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


def same_bytes(left, right) -> bool:
    """Byte identity: dtype, shape and every byte, NaN payloads included."""
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


@pytest.fixture(scope="module")
def comfy():
    """Import real comfy on its CPU path, then put sys.modules back as found."""
    comfy_dir = _comfy_dir()
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-vae-decode-split", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI checkout on sys.path")
            options.enable_args_parsing()
            utils = pytest.importorskip("comfy.utils")
            sd = pytest.importorskip("comfy.sd")
            minimax = pytest.importorskip("comfy.ldm.minimax.vae")
        finally:
            sys.argv = original_argv
        yield SimpleNamespace(utils=utils, sd=sd, minimax=minimax,
                              tiled=utils.tiled_scale_multidim, decode_tiled_3d=sd.VAE.decode_tiled_3d)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def _sizes(shape, factors):
    out = []
    for size, factor in zip(shape, factors, strict=True):
        out.append(round(factor(size)) if callable(factor) else round(factor * size))
    return out


def _fake_decode(factors, out_channels=3):
    """Deterministic, per-sample and per-element: a stand-in for a VAE decoder."""
    def decode(x):
        mixed = torch.stack([torch.tanh(x[:, c % x.shape[1]] * (0.7 + 0.1 * c) + 0.05 * c)
                             for c in range(out_channels)], dim=1)
        sizes = _sizes(x.shape[2:], factors)
        up = torch.nn.functional.interpolate(mixed.float(), size=sizes, mode="nearest").to(mixed.dtype)
        ramp = torch.linspace(0.9, 1.1, up.shape[-1], dtype=up.dtype)
        return up * ramp
    return decode


class _Ranks:
    """Simulated Worker ranks: each claimed unit is computed in a thread, as a Worker would."""

    def __init__(self, compute, *, fail=None, delay_s=0.0):
        self.compute = compute
        self.fail = fail or (lambda rank, unit: None)
        self.delay_s = delay_s
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.seen: list[tuple[int, int]] = []

    def sender(self, handle, token, base, inputs):
        def send(rank, unit):
            self.seen.append((rank, unit))
            return self.pool.submit(self._reply, rank, unit, inputs[unit], base)
        return send

    def _reply(self, rank, unit, latent, base):
        time.sleep(self.delay_s)
        failure = self.fail(rank, unit)
        if isinstance(failure, BaseException):
            raise failure
        if failure == "decline":
            return {"schema": 1, "ok": False, "unit": unit, "rank": rank, "host": "sibling",
                    "reason": "this host's vae/x.safetensors is not the driver's file"}
        output = self.compute(latent)
        if failure == "dtype":
            output = output.to(torch.float16)
        return {"schema": 1, "ok": True, "unit": unit, "rank": rank, "host": "sibling", "output": output}

    def waiter(self, handle):
        return lambda future, timeout_s: future.result(timeout=timeout_s)


def _handle(world=2):
    return SimpleNamespace(
        lock=threading.RLock(), setup_key=("topology",), worker_args_key=("policy",), setup_generation=4,
        setup_cleanup_state=None, defunct=False, teardown_complete=False, world=world, gpus_per_host=1,
        config=SimpleNamespace(worker_args={"worker_vae_decode": "auto"}, source="cluster.toml", hosts=()))


def _route(vae, kind, ranks, world=None):
    family = Family("test", kind, frozenset({"decode", "decode_tiled"}))
    return vae_decode_route._Route(_handle(world or len(ranks) + 1), family, vae, list(ranks),
                                   {"folder": "vae", "name": "x.safetensors", "signature": "s"},
                                   {"vae_class": "test"}, DecodeRecord(family="test", entry="decode_tiled"))


@pytest.fixture
def ranks(monkeypatch):
    holder = {}

    def install(simulated):
        monkeypatch.setattr(vae_decode_remote, "sender", simulated.sender)
        monkeypatch.setattr(vae_decode_remote, "waiter", simulated.waiter)
        holder["ranks"] = simulated
        return simulated

    yield install
    if "ranks" in holder:
        holder["ranks"].pool.shutdown(wait=True)


def _slow_first(function, delay_s=0.5):
    """Hold the driver on unit 0 so the simulated ranks claim before it moves on."""
    calls = {"n": 0}

    def wrapped(a):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(delay_s)
        return function(a)
    return wrapped


# (shape, tile, overlap, upscale_amount, index_formulas, dtype)
CASES = {
    "2d": ((1, 4, 37, 53), (16, 16), 4, 8, None, torch.float32),
    "2d-batch2-bf16": ((2, 4, 29, 41), (12, 20), 3, 8, None, torch.bfloat16),
    "1d": ((1, 8, 301), (64,), 16, 4, None, torch.float32),
    "3d-wan": ((1, 16, 13, 23, 31), (5, 8, 8), (1, 2, 2), (lambda a: max(0, a * 4 - 3), 8, 8), (4, 8, 8),
               torch.bfloat16),
    "3d-h3-formulas": ((1, 24, 17, 9, 11), (7, 4, 4), (1, 1, 1),
                       (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16), (4, 16, 16), torch.float32),
    "3d-odd-overlap": ((1, 6, 9, 19, 17), (4, 7, 6), (2, 3, 1), (lambda a: max(0, a * 4 - 3), 8, 8), (4, 8, 8),
                       torch.float32),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("remote", [1, 3])
def test_multidim_split_is_byte_identical_to_stock(comfy, ranks, case, remote):
    shape, tile, overlap, upscale, formulas, dtype = CASES[case]
    torch.manual_seed(7)
    samples = torch.randn(shape)
    dims = len(tile)
    factors = list(upscale) if isinstance(upscale, tuple) else [upscale] * dims
    fake = _fake_decode(factors)
    function = lambda a: fake(a.to(dtype)).to(dtype=torch.float32)  # noqa: E731 - the stock lambda's shape
    stock = comfy.tiled(samples, function, tile=tile, overlap=overlap, upscale_amount=upscale, out_channels=3,
                        output_device="cpu", index_formulas=formulas)
    vae = SimpleNamespace(vae_dtype=dtype, vae_output_dtype=lambda: torch.float32)
    ranks(_Ranks(lambda latent: fake(latent).to(torch.float32)))
    route = _route(vae, MULTIDIM, range(1, remote + 1))
    split = route.split_multidim(comfy.tiled, {
        "samples": samples, "function": _slow_first(function), "tile": tile, "overlap": overlap,
        "upscale_amount": upscale, "out_channels": 3, "output_device": "cpu", "downscale": False,
        "index_formulas": formulas, "pbar": None})
    assert same_bytes(split, stock)
    assert route.record.units > 1 and sum(route.record.remote.values()) >= 1
    assert route.record.local + sum(route.record.remote.values()) == route.record.units
    assert not route.record.failures


def test_more_ranks_than_tiles_and_a_single_tile(comfy, ranks):
    samples = torch.randn(1, 4, 20, 12)
    fake = _fake_decode([8, 8])
    function = lambda a: fake(a).to(torch.float32)  # noqa: E731
    vae = SimpleNamespace(vae_dtype=torch.float32, vae_output_dtype=lambda: torch.float32)
    ranks(_Ranks(lambda latent: fake(latent).to(torch.float32)))
    two = {"samples": samples, "function": _slow_first(function), "tile": (12, 12), "overlap": 4,
           "upscale_amount": 8, "out_channels": 3, "output_device": "cpu", "downscale": False,
           "index_formulas": None, "pbar": None}
    route = _route(vae, MULTIDIM, [1, 2, 3, 4])
    assert same_bytes(route.split_multidim(comfy.tiled, two), comfy.tiled(**{**two, "function": function}))
    assert route.record.units == 2 and route.record.local == 1 and sum(route.record.remote.values()) == 1
    single = {**two, "tile": (64, 64)}
    route = _route(vae, MULTIDIM, [1])
    assert same_bytes(route.split_multidim(comfy.tiled, single), comfy.tiled(**{**single, "function": function}))
    assert route.record.units == 0 and route._notes == ["the tiled decode is one tile here, so there is nothing to split"]


@pytest.mark.parametrize("failure", ["raise", "decline", "dtype", "silent"])
def test_a_failing_rank_falls_back_with_the_stock_bytes(comfy, ranks, monkeypatch, caplog, failure):
    samples = torch.randn(1, 4, 37, 41)
    fake = _fake_decode([8, 8])
    function = lambda a: fake(a).to(torch.float32)  # noqa: E731
    vae = SimpleNamespace(vae_dtype=torch.float32, vae_output_dtype=lambda: torch.float32)
    monkeypatch.setattr(vae_decode_remote, "FIRST_UNIT_TIMEOUT_S", 0.5)
    monkeypatch.setattr(vae_decode_remote, "UNIT_TIMEOUT_S", 0.5)

    def fail(rank, unit):
        if failure == "raise":
            return RuntimeError("actor on the sibling died")
        if failure == "silent":
            time.sleep(1.5)
        return failure if failure in ("decline", "dtype") else None

    ranks(_Ranks(lambda latent: fake(latent).to(torch.float32), fail=fail))
    arguments = {"samples": samples, "function": _slow_first(function), "tile": (12, 12), "overlap": 4,
                 "upscale_amount": 8, "out_channels": 3, "output_device": "cpu", "downscale": False,
                 "index_formulas": None, "pbar": None}
    route = _route(vae, MULTIDIM, [1])
    caplog.set_level(logging.INFO, logger="dgx_monarch.vae_decode")
    split = route.split_multidim(comfy.tiled, arguments)
    assert same_bytes(split, comfy.tiled(**{**arguments, "function": function}))
    assert route.record.failures and route.record.failures[0].startswith("rank 1: ")
    assert sum(route.record.remote.values()) == 0
    from dgx_monarch.vae_decode_telemetry import publish

    publish(route.record)
    assert "Worker-assisted VAE decode fell back to this driver for" in caplog.text


def test_a_plan_the_stock_loop_leaves_decodes_the_rest_here(comfy, ranks, monkeypatch):
    from dgx_monarch import vae_decode_plan

    real_plan = vae_decode_plan.multidim_plan

    def shifted(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        first = plan[1]
        return [plan[0], vae_decode_plan.Geometry(first.offset + 1, first.size, first.stride), *plan[2:]]

    monkeypatch.setattr(vae_decode_plan, "multidim_plan", shifted)
    samples = torch.randn(1, 4, 33, 33)
    fake = _fake_decode([8, 8])
    function = lambda a: fake(a).to(torch.float32)  # noqa: E731
    vae = SimpleNamespace(vae_dtype=torch.float32, vae_output_dtype=lambda: torch.float32)
    ranks(_Ranks(lambda latent: fake(latent).to(torch.float32)))
    arguments = {"samples": samples, "function": function, "tile": (12, 12), "overlap": 4, "upscale_amount": 8,
                 "out_channels": 3, "output_device": "cpu", "downscale": False, "index_formulas": None,
                 "pbar": None}
    route = _route(vae, MULTIDIM, [1])
    assert same_bytes(route.split_multidim(comfy.tiled, arguments), comfy.tiled(**{**arguments}))
    assert any("left the plan" in failure for failure in route.record.failures)
    assert route.record.local + sum(route.record.remote.values()) == route.record.units


class _FakeVae:
    """The attributes VAE.decode_tiled_3d reads, around a deterministic decoder."""

    def __init__(self, dtype):
        self.first_stage_model = SimpleNamespace(decode=_fake_decode([lambda a: max(0, a * 4 - 3), 8, 8]))
        self.vae_dtype = dtype
        self.device = torch.device("cpu")
        self.output_device = torch.device("cpu")
        self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
        self.upscale_index_formula = (4, 8, 8)
        self.output_channels = 3
        self.process_output = lambda image: image

    def vae_output_dtype(self):
        return torch.float32


def test_the_stock_decode_tiled_3d_path_splits_through_the_installed_wrappers(comfy, ranks):
    vae_decode_route.install()
    vae_decode_route.install()  # keyed: a second install stacks nothing
    assert not getattr(comfy.utils.tiled_scale_multidim.__wrapped__, "_dgxm_vae_decode_route", False)
    fake = _FakeVae(torch.bfloat16)
    samples = torch.randn(1, 16, 9, 21, 27)
    stock = comfy.decode_tiled_3d(fake, samples, tile_t=4, tile_x=8, tile_y=8, overlap=(1, 2, 2))
    simulated = ranks(_Ranks(lambda latent: fake.first_stage_model.decode(latent).to(torch.float32), delay_s=0.0))
    route = _route(fake, MULTIDIM, [1])
    split = route.run(lambda: comfy.sd.VAE.decode_tiled_3d(fake, samples, tile_t=4, tile_x=8, tile_y=8,
                                                          overlap=(1, 2, 2)))
    assert same_bytes(split, stock)
    assert route.record.units > 1 and route.record.local + sum(route.record.remote.values()) == route.record.units
    assert all(rank == 1 for rank, _unit in simulated.seen)


class _Decoder(torch.nn.Module):
    """What decode_output_shape reads off the real ViT decoder."""

    out_channels = 3


def _h3(comfy, dtype=torch.float32):
    """The real MiniMax H3 VAE class with 64 px tiles and a fake per-tile decoder."""
    fsm = comfy.minimax.MiniMaxH3VideoVAE(encoder=False, decoder=False, tile_size=64, tile_overlap_min=16)
    fsm.decoder = _Decoder()

    def pixels(z):
        mixed = torch.stack([torch.tanh(z[:, c] * 0.9 + z[:, c + 3] * 0.2) for c in range(3)], dim=1)
        up = mixed.repeat_interleave(4, dim=2).repeat_interleave(16, dim=3).repeat_interleave(16, dim=4)
        return up * torch.linspace(0.95, 1.05, up.shape[-1], dtype=up.dtype)

    slow = {"armed": False}

    def first_slow(z):
        # Once armed, hold the next call (the driver's chunk 0) so a simulated
        # rank claims a chunk first, however loaded the machine is.
        if slow["armed"]:
            slow["armed"] = False
            time.sleep(0.5)
        return pixels(z)

    fsm._decode_pixels = first_slow
    fsm.arm_slow_first = lambda: slow.update(armed=True)
    return fsm.to(dtype)


@pytest.mark.parametrize("tokens", [12, 27])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("buffered", [False, True])
def test_h3_chunked_decode_split_is_byte_identical(comfy, ranks, tokens, dtype, buffered):
    torch.manual_seed(11)
    z = torch.randn(1, 24, tokens, 6, 7).to(dtype)
    driver, worker = _h3(comfy, dtype), _h3(comfy, dtype)
    shape = driver.decode_output_shape(z.shape)
    if buffered:
        stock = torch.empty(shape)
        driver.decode(z, output_buffer=stock)
    else:
        stock = driver.decode(z)
    ranks(_Ranks(lambda window: type(worker)._adaptive_decode(worker, window), delay_s=0.05))
    route = _route(SimpleNamespace(first_stage_model=driver), H3_CHUNK, [1])
    driver.arm_slow_first()
    if buffered:
        split = torch.full(shape, float("nan"))
        route.run(lambda: driver.decode(z, output_buffer=split))
    else:
        split = route.run(lambda: driver.decode(z))
    assert same_bytes(split, stock)
    assert "decode_temporal" not in driver.__dict__ and "_adaptive_decode" not in driver.__dict__
    assert route.record.units == driver._decode_temporal_chunks(tokens)[1]
    assert route.record.local + sum(route.record.remote.values()) == route.record.units
    assert sum(route.record.remote.values()) >= 1


def test_h3_still_and_a_failing_rank_keep_the_stock_bytes(comfy, ranks):
    z = torch.randn(1, 24, 27, 6, 7)
    driver, worker = _h3(comfy), _h3(comfy)
    stock = driver.decode(z)
    ranks(_Ranks(lambda window: type(worker)._adaptive_decode(worker, window),
                 fail=lambda rank, unit: RuntimeError("sibling lost") if unit >= 3 else None))
    route = _route(SimpleNamespace(first_stage_model=driver), H3_CHUNK, [1])
    driver.arm_slow_first()
    assert same_bytes(route.run(lambda: driver.decode(z)), stock)
    assert route.record.failures and "sibling lost" in route.record.failures[0]
    still = torch.randn(1, 24, 1, 6, 7)
    route = _route(SimpleNamespace(first_stage_model=driver), H3_CHUNK, [1])
    assert same_bytes(route.run(lambda: driver.decode(still)), driver.decode(still))
    assert route.record.units == 0 and not route.record.split


def test_a_real_vae_file_splits_through_the_stock_decode_and_the_worker_body(comfy, ranks):
    """End to end on CPU: Load VAE's object, the installed wrappers, and a rank running the real ``run()``.

    The rank builds its own VAE from the same file by name, checks its
    signature and decodes each unit it claims, as a Worker does; only Monarch
    transport is simulated. Skipped where the file is absent, as in CI.
    """
    import folder_paths

    from dgx_monarch.actor import vae_decode

    name = "wan_2.1_vae.safetensors"
    if folder_paths.get_full_path("vae", name) is None:
        pytest.skip(f"no models/vae/{name} in this ComfyUI checkout")
    loader = __import__("nodes").VAELoader()
    vae = loader.load_vae(name)[0]
    worker = SimpleNamespace(rank=1, _setup_key=("topology",), _active_worker_args={}, uma_reserve_gb=0.0)
    gpu = threading.Lock()  # the actor's GPU lock: one unit at a time

    class _WorkerRanks(_Ranks):
        def sender(self, handle, token, base, inputs):
            def send(rank, index):
                request = {**base, "unit": index, "input": inputs[index]}

                def run():
                    with gpu:
                        return vae_decode.run(worker, request)
                return self.pool.submit(run)
            return send

    ranks(_WorkerRanks(compute=None))
    vae_decode_route.install()
    torch.manual_seed(13)
    latent = torch.randn(1, 16, 3, 12, 12)
    tiled = {"tile_x": 8, "tile_y": 8, "overlap": 2, "tile_t": 2, "overlap_t": 1}
    with torch.inference_mode():
        stock = vae.decode_tiled(latent, **tiled)
        route = _route(vae, MULTIDIM, [1])
        route.source = vae_decode_remote.vae_source(vae)
        from dgx_monarch.vae_decode_env import describe

        route.env = describe(vae)
        split = route.run(lambda: vae.decode_tiled(latent, **tiled))
    vae_decode.drop_resident(worker)
    assert same_bytes(split, stock)
    assert sum(route.record.remote.values()) >= 1 and not route.record.failures


def test_stock_seams_the_split_replays(comfy):
    """The canary half: the stock names, signatures and decode lambdas the split depends on."""
    import inspect

    assert list(inspect.signature(comfy.tiled).parameters) == list(vae_decode_route._SPLIT_ARGUMENTS)
    source = inspect.getsource(comfy.sd.VAE.decode_tiled_3d)
    assert "self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())" in source
    assert "comfy.utils.tiled_scale_multidim(" in source
    temporal = inspect.getsource(comfy.minimax.MiniMaxH3VideoVAE.decode_temporal)
    assert "self._adaptive_decode(clip_z)" in temporal
    assert "self._decode_temporal_chunks(z.shape[2])" in temporal
    assert "self.decode_temporal(z, output_buffer)" in inspect.getsource(comfy.minimax.MiniMaxH3VideoVAE.decode)
    # vae_decode_remote.vae_source names the file from the loader's reload factory.
    assert list(inspect.signature(comfy.sd.load_vae_patcher).parameters)[:3] == ["vae_path", "metadata", "device"]
    loader = (os.path.join(_comfy_dir(), "nodes.py"))
    with open(loader, encoding="utf-8") as handle:
        assert "cached_patcher_init = (comfy.sd.load_vae_patcher, (vae_path, metadata, None))" in handle.read()
