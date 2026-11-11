"""A damaged safetensors header cannot cancel a refusal of the upstream gate.

The gate refuses four kinds of file from the header alone (TROUBLESHOOTING #91,
#120, #149 and #150). Its strict reader refuses headers stock ComfyUI still
reads: a key written twice, an empty tensor name, tensors that share bytes,
metadata that is not all strings. Before, any failed read meant no verdict, so
such a file loaded past every refusal. Now the gate reads the header again as
ComfyUI's own reader does and the same predicates answer; a file no stock
reader can read keeps no verdict; and a read that fails with no parse verdict
refuses class P (#151).

Every file here is hand-made: a header and a few zero bytes, no weights. The
stock side (what safetensors and ComfyUI's two load paths read from the same
damages) is held against a real ComfyUI by the last two tests, which the
comfy-canary job runs and ordinary CI skips.
"""
from __future__ import annotations

import json
import re
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import minimax_h3_image_headers as h3i
from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    _isolated_process_state,
    _RecordingStore,
)
from dgx_monarch import upstream_gate
from dgx_monarch.actor import store_fsdp
from dgx_monarch.adapters import detect
from dgx_monarch.adapters import minimax_h3_image_header as h3_image
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import loader_preflight
from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader
from dgx_monarch.refusal import RefusalClass, RefusalTag, parse_leading_refusal_tag
from dgx_monarch.safetensors_header import SafetensorsHeaderError, read_safetensors_header
from test_dropped_scale_gate import PASSES, REFUSALS, _write_header
from test_dropped_scale_gate import real_comfy as real_comfy  # the real-ComfyUI fixture

L2P, SCALES = upstream_gate.TROUBLESHOOTING, upstream_gate.DROPPED_SCALE_TROUBLESHOOTING
GRID, UNREAD = upstream_gate.H3_IMAGE_GRID_TROUBLESHOOTING, upstream_gate.UNREAD_HEADER_TROUBLESHOOTING
# What an intact MiniMax-H3-Image file with its sigma grid gets: today the family
# refusal (#150). The one line to change when that refusal retires with the family's
# adapter: set it to None (the file then passes the gate), and every test below
# holds a damaged copy to the same answer, a pass included.
H3_IMAGE_WITH_GRID: int | None = upstream_gate.H3_IMAGE_FAMILY_TROUBLESHOOTING
TROUBLESHOOTING = Path(__file__).resolve().parents[1] / "docs" / "TROUBLESHOOTING.md"
WIDTHS = {"BF16": 2, "F32": 4, "I8": 1, "U8": 1, "F6_E2M3": 1}
DEEP = "[" * 200_000 + "]" * 200_000  # past every Python's JSON nesting limit
CONFIG = json.dumps(h3i.CONFIG)
WEIGHT = "single_blocks.0.qkv.weight"
_ZIMAGE = ["cap_embedder.0.weight", "x_pad_token", "layers.0.attention.qkv.weight"]
_H3_IMAGE = ["video_patch_proj.weight", "condition_proj.weight", "blocks.0.attn.qkv_proj.weight"]

# One intact header per refusal: (tensors, metadata, the entry it refuses with).
BASES: dict[str, tuple[list[tuple[str, str, list[int]]], dict[str, str] | None, int | None]] = {
    "l2p": ([(key, "BF16", [1]) for key in
             [*_ZIMAGE, "all_x_embedder.16-1.weight", "local_decoder.blocks.0.weight"]], None, L2P),
    "scales": ([(WEIGHT, "I8", [2, 2]), ("single_blocks.0.qkv.weight_scale", "F32", [2, 1]),
                ("single_blocks.0.norm.query_norm.scale", "BF16", [2])], None, SCALES),
    "grid": ([(key, "BF16", [1]) for key in _H3_IMAGE], None, GRID),
    "family": ([(key, "BF16", [1]) for key in _H3_IMAGE], {"config": CONFIG}, H3_IMAGE_WITH_GRID),
}


def _pairs(base: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], int]:
    """(the ``__metadata__`` pair if any, the tensor pairs, the data size), values as JSON text."""
    tensors, metadata, _number = BASES[base]
    pairs, offset = [], 0
    for name, dtype, shape in tensors:
        size = WIDTHS[dtype]
        for dim in shape:
            size *= dim
        pairs.append((name, json.dumps(
            {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]})))
        offset += size
    meta = [("__metadata__", json.dumps(metadata))] if metadata is not None else []
    return meta, pairs, offset


def _raw(pairs: list[tuple[str, str]]) -> bytes:
    """A JSON object from (key, value text) pairs, so one key can be written twice."""
    return ("{" + ",".join(f"{json.dumps(key)}:{value}" for key, value in pairs) + "}").encode()


def _file(header: bytes, size: int, declared: int | None = None) -> bytes:
    return struct.pack("<Q", len(header) if declared is None else declared) + header + b"\0" * size


def _descriptor(dtype: str, shape: list[int], begin: int, end: int) -> str:
    return json.dumps({"dtype": dtype, "shape": shape, "data_offsets": [begin, end]})


def _with_metadata(base: str, text: str) -> bytes:
    _meta, pairs, size = _pairs(base)
    return _file(_raw([("__metadata__", text), *pairs]), size)


def _twice(pairs: list[tuple[str, str]]) -> str:
    return "{" + ",".join(f"{json.dumps(key)}:{json.dumps(value)}" for key, value in pairs) + "}"


def _damage(base: str, damage: str) -> bytes:
    """One damage a stock reader still reads, written over ``base``."""
    meta, pairs, size = _pairs(base)
    metadata = BASES[base][1] or {"note": "x"}
    first, first_info = pairs[0][0], json.loads(pairs[0][1])
    if damage == "intact":
        return _file(_raw(meta + pairs), size)
    if damage == "a tensor entry twice":
        return _file(_raw(meta + pairs + [pairs[0]]), size)
    if damage == "__metadata__ twice":
        twice = meta or [("__metadata__", "{}")]
        return _file(_raw(twice + twice + pairs), size)
    if damage == "a metadata key twice":
        key = next(iter(metadata))
        return _with_metadata(base, _twice([*metadata.items(), (key, metadata[key])]))
    if damage == "a descriptor field twice":
        text = pairs[0][1].replace('{"dtype"', '{"dtype": ' + json.dumps(first_info["dtype"]) + ', "dtype"', 1)
        return _file(_raw([*meta, (first, text), *pairs[1:]]), size)
    if damage == "an empty tensor name":
        return _file(_raw(meta + pairs + [("", _descriptor("BF16", [0], size, size))]), size)
    if damage == "two tensors on the same bytes":
        twin = _descriptor(first_info["dtype"], first_info["shape"], *first_info["data_offsets"])
        return _file(_raw(meta + pairs + [("zz_alias.weight", twin)]), size)
    if damage == "a metadata value that is no string":
        return _with_metadata(base, json.dumps({**(BASES[base][1] or {}), "note": 5}))
    assert damage == "a dtype the strict reader does not table", damage
    return _file(_raw(meta + pairs + [("zz_f6.weight", _descriptor("F6_E2M3", [1], size, size + 1))]),
                 size + 1)


# Each cancels all four refusals on dev c1f0ba8. safetensors reads the first, third and
# fifth; ComfyUI's own reader reads all but the last, which no stock reader loads.
DAMAGES = ("a tensor entry twice", "__metadata__ twice", "a metadata key twice",
           "a descriptor field twice", "an empty tensor name", "two tensors on the same bytes",
           "a metadata value that is no string", "a dtype the strict reader does not table")


class _PastTheGate(Exception):
    """The loader went on to an answer below the gate."""


@pytest.fixture
def models(tmp_path, monkeypatch):
    """A models folder both funnels resolve through, with everything below the gate marked."""
    def get_full_path(_kind, name):
        candidate = tmp_path / str(name)
        return str(candidate) if candidate.exists() else None

    fake = types.ModuleType("folder_paths")
    fake.get_full_path = get_full_path
    fake.get_filename_list = lambda _kind: sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setitem(sys.modules, "folder_paths", fake)

    def past(*_args, **_kwargs):
        raise _PastTheGate

    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology", past)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", past)
    from dgx_monarch.nodes import loaders

    monkeypatch.setattr(loaders, "ensure_live", past)
    return tmp_path


def _driver(name: str) -> str | None:
    """The driver loader's refusal text, or None when the file passes the gate."""
    try:
        DGXMonarchUNETLoader().load(
            SimpleNamespace(topology_preset="single", handle=None, worker_args={}), name)
    except UnsupportedModelError as exc:
        return str(exc)
    except _PastTheGate:
        return None
    raise AssertionError("the loader returned with every later stage patched to raise")


def _worker(name: str) -> str | None:
    """The Worker funnel's refusal text, or None when the store was entered."""
    worker = SimpleNamespace(topology={}, world=1, store=_RecordingStore())
    try:
        store_fsdp.ensure(worker, name, {}, [])
    except UnsupportedModelError as exc:
        assert worker.store.calls == [], "the store was entered on a refused artifact"
        return str(exc)
    assert len(worker.store.calls) == 1
    return None


def _write(models: Path, name: str, data: bytes) -> str:
    (models / name).write_bytes(data)
    return name


def _number(models: Path, name: str) -> int | None:
    found = upstream_gate.refusal_for(str(models / name))
    return None if found is None else found[1]


def _assert_on_both_paths(models: Path, name: str, number: int | None) -> str | None:
    """Gate, driver and Worker refuse class P with one sentence, which is returned;
    or, for ``None``, all three let the file through."""
    found = upstream_gate.refusal_for(str(models / name))
    if number is None:
        assert found is None and _driver(name) is None and _worker(name) is None, found
        return None
    assert found is not None and found[1] == number, found
    driver, worker = _driver(name), _worker(name)
    assert driver is not None, "the driver loader let the file past the gate"
    assert worker == driver
    assert found[0] in driver and driver.endswith(f"See docs/TROUBLESHOOTING.md #{number}.")
    assert parse_leading_refusal_tag(driver) == RefusalTag(
        refusal_class=RefusalClass.PHYSICS, guard=None, waivable=False)
    return driver


@pytest.mark.parametrize("damage", DAMAGES)
@pytest.mark.parametrize("base", sorted(BASES))
def test_a_damaged_header_gets_the_refusal_its_intact_header_gets_on_both_paths(models, base, damage):
    """Each damage against each of the four refusals, on the driver and the Worker."""
    number = BASES[base][2]
    reference = _assert_on_both_paths(models, _write(models, "intact.safetensors",
                                                             _damage(base, "intact")), number)
    name = _write(models, "damaged.safetensors", _damage(base, damage))
    with pytest.raises(SafetensorsHeaderError):
        read_safetensors_header(models / name)  # the damage is one the strict reader refuses
    assert _assert_on_both_paths(models, name, number) == reference


def test_the_last_copy_of_a_key_written_twice_is_the_one_the_gate_reads(models):
    """As every stock reader that takes the file reads it (run on both ComfyUI clones)."""
    _meta, pairs, size = _pairs("scales")
    as_bf16 = (WEIGHT, _descriptor("BF16", [2], 0, 4))
    for label, order, number in (("i8-last", [as_bf16, pairs[0]], SCALES),
                                 ("bf16-last", [pairs[0], as_bf16], None)):
        name = _write(models, f"tensor-{label}.safetensors",
                      _file(_raw([order[0], *pairs[1:], order[1]]), size))
        assert _number(models, name) == number, label
        assert (_driver(name) is None) == (number is None) == (_worker(name) is None), label
    # The same inside one descriptor, which only ComfyUI's own reader takes.
    field = '{"dtype":"I8","dtype":"BF16","shape":[2],"data_offsets":[0,4]}'
    name = _write(models, "field.safetensors", _file(_raw([(WEIGHT, field), *pairs[1:]]), size))
    assert _number(models, name) is None and _driver(name) is None and _worker(name) is None
    grid, none = ("config", CONFIG), ("config", "{}")
    for label, order, number in (("no-grid-last", [grid, none], GRID),
                                 ("grid-last", [none, grid], H3_IMAGE_WITH_GRID)):
        inner = _write(models, f"config-{label}.safetensors", _with_metadata("family", _twice(order)))
        _assert_on_both_paths(models, inner, number)
        _meta, tensors, data = _pairs("family")
        outer = _write(models, f"metadata-{label}.safetensors", _file(_raw(
            [("__metadata__", _twice([pair])) for pair in order] + tensors), data))
        _assert_on_both_paths(models, outer, number)


@pytest.mark.parametrize("base", ["grid", "family"])
@pytest.mark.parametrize("config", [
    DEEP, '{"transformer":' + json.dumps(h3i.CONFIG["transformer"]) + ',"junk":' + DEEP + "}",
], ids=["deep", "deep-beside-a-grid"])
def test_a_config_nested_past_the_parsers_limit_counts_as_a_missing_grid(models, base, config):
    """RecursionError left ``grid_missing`` and cancelled #149 and #150; stock's own
    ``json.loads`` of the same config fails, so the file has no grid ComfyUI can use."""
    with pytest.raises(RecursionError):
        json.loads(config)
    assert not h3_image.grid_missing({"config": CONFIG})
    assert h3_image.grid_missing({"config": config})
    name = _write(models, "deep.safetensors", _with_metadata(base, json.dumps({"config": config})))
    read_safetensors_header(models / name)  # an intact header: only the config is deep
    _assert_on_both_paths(models, name, GRID)


def test_metadata_is_read_as_comfys_own_reader_hands_it_on(models):
    """A value that is no string is dropped alone; metadata that is no map is no metadata.

    ComfyUI's own reader hands every entry on and can parse only the strings. Dropping
    the whole map for one stray value would turn the family refusal into the grid one,
    and refuse a file whose ``_quantization_metadata`` covers its layers."""
    stray = _write(models, "stray.safetensors",
                   _with_metadata("family", json.dumps({"config": CONFIG, "note": 5})))
    _assert_on_both_paths(models, stray, H3_IMAGE_WITH_GRID)
    for text in ("[]", '"config"', "7"):
        name = _write(models, "no-map.safetensors", _with_metadata("family", text))
        _assert_on_both_paths(models, name, GRID)
        for base in ("l2p", "scales", "grid"):
            name = _write(models, f"no-map-{base}.safetensors", _with_metadata(base, text))
            _assert_on_both_paths(models, name, BASES[base][2])
    # A covered nvfp4 layer stays covered beside a stray number, and still passes.
    tensors, metadata = PASSES["nvfp4_with_its_metadata"]
    sound = Path(_write_header(models / "sound.safetensors", tensors, metadata)).read_bytes()
    length = struct.unpack("<Q", sound[:8])[0]
    header = json.loads(sound[8:8 + length])
    header["__metadata__"]["note"] = 5
    covered = _write(models, "covered.safetensors",
                     _file(json.dumps(header).encode(), len(sound) - 8 - length))
    with pytest.raises(SafetensorsHeaderError):
        read_safetensors_header(models / covered)
    assert _number(models, covered) is None and _driver(covered) is None and _worker(covered) is None


def _no_header(base: str, damage: str) -> bytes:
    meta, pairs, size = _pairs(base)
    header = _raw(meta + pairs)
    if damage == "shorter than 8 bytes":
        return _file(header, size)[:7]
    if damage == "length zero":
        return _file(header, size, declared=0)
    if damage == "length past the file":
        return _file(header, size, declared=len(header) + size + 64)
    if damage == "length 2**62":
        return _file(header, size, declared=2**62)
    if damage == "length that cuts the JSON short":
        return _file(header, size, declared=len(header) - 3)
    if damage == "not JSON":
        return _file(b"{not json" + header[9:], size)
    if damage == "a trailing comma":
        return _file(header[:-1] + b",}", size)
    if damage == "a list at the top":
        return _file(b"[" + header + b"]", 0)
    assert damage == "not UTF-8", damage
    return _file(header.replace(b"weight", b"wei\xffht", 1), size)


NO_HEADER = ("shorter than 8 bytes", "length zero", "length past the file", "length 2**62",
             "length that cuts the JSON short", "not JSON", "a trailing comma", "a list at the top",
             "not UTF-8")


@pytest.mark.parametrize("damage", NO_HEADER)
@pytest.mark.parametrize("base", sorted(BASES))
def test_a_file_no_stock_reader_can_read_keeps_no_verdict(models, base, damage):
    """Not a refusal: safetensors and ComfyUI's own reader both fail on each of these
    (held against a real ComfyUI in tests/test_dropped_scale_gate.py), so the load fails
    by itself. This guards the other side of the fix: a gate that refused every failed
    read would refuse here."""
    name = _write(models, "none.safetensors", _no_header(base, damage))
    assert upstream_gate._lenient_header(str(models / name)) is None
    with pytest.raises(SafetensorsHeaderError):  # the strict reader's own error stands
        upstream_gate.uncovered_scaled_layers(str(models / name))
    assert upstream_gate.refusal_for(str(models / name)) is None
    assert _driver(name) is None and _worker(name) is None


def _heading(number: int) -> str:
    headings = dict(re.findall(r"^## (\d+)\. (.+)$", TROUBLESHOOTING.read_text(), re.MULTILINE))
    return headings[str(number)]


@pytest.mark.parametrize("base", sorted(BASES))
def test_a_read_that_fails_with_no_parse_verdict_refuses_class_p(models, base, monkeypatch):
    """The closed default (#151): no proof that stock cannot read the file, so no load.

    A junk value nested past the parser's limit raises RecursionError in the strict
    reader and in the lenient one. Whether ComfyUI's reader fails on it too depends on
    the interpreter, not on the file, so the gate refuses."""
    meta, pairs, size = _pairs(base)
    junk = pairs[0][1][:-1] + ',"junk":' + DEEP + "}"
    name = _write(models, "junk.safetensors", _file(_raw([*meta, (pairs[0][0], junk), *pairs[1:]]), size))
    with pytest.raises(RecursionError):
        read_safetensors_header(models / name)
    text = _assert_on_both_paths(models, name, UNREAD)
    assert UNREAD == 151 and _heading(UNREAD).startswith("A render refuses because the header gate could not read")
    for token in ("could not read this checkpoint's safetensors header", "Nothing was loaded",
                  "no waiver", "stock ComfyUI on one GPU", "Load Diffusion Model", "restore the file"):
        assert token in text
    assert "[dgxm:" not in upstream_gate.UNREAD_HEADER_REFUSAL  # the tag is built at the raise

    # Any other failure of the second read is the same case: memory, or an I/O error after the open.
    for error in (MemoryError(), OSError("read error")):
        def fail(_path, error=error):
            raise error

        damaged = _write(models, f"second-{type(error).__name__}.safetensors",
                         _damage(base, "a tensor entry twice"))
        monkeypatch.setattr(upstream_gate, "_lenient_header", fail)
        assert _assert_on_both_paths(models, damaged, UNREAD) == text


def test_quantization_metadata_nested_past_the_limit_refuses_but_a_parse_error_does_not(models):
    """The scale check gives no verdict only on the errors ComfyUI's own
    ``convert_old_quants`` raises on the same text (not JSON, no layers map). Any other
    failure is no proof that stock fails, so it refuses (#151)."""
    tensors, _metadata = PASSES["nvfp4_with_its_metadata"]
    deep = Path(_write_header(models / "deep.safetensors", tensors, {"_quantization_metadata": DEEP})).name
    _assert_on_both_paths(models, deep, UNREAD)
    for text in ("{not json", json.dumps({"format": "x"}), json.dumps({"layers": ["a"]}), "7", '"layers"'):
        name = Path(_write_header(models / "parse.safetensors", tensors,
                                  {"_quantization_metadata": text})).name
        with pytest.raises(upstream_gate.QuantMetadataError):
            upstream_gate.uncovered_scaled_layers(str(models / name))
        assert _number(models, name) is None and _driver(name) is None and _worker(name) is None
    # The same damaged table beside a damaged header: still no verdict, as before.
    _meta, pairs, size = _pairs("scales")
    table = ("__metadata__", json.dumps({"_quantization_metadata": "{not json"}))
    both = _write(models, "both.safetensors", _file(_raw([table, *pairs, pairs[0]]), size))
    with pytest.raises(SafetensorsHeaderError):
        read_safetensors_header(models / both)
    assert _number(models, both) is None and _driver(both) is None and _worker(both) is None


def test_a_process_that_cannot_import_detection_gets_no_verdict(models, monkeypatch):
    """A headless caller with no torch (the sweep matrix, a tool) loads nothing, so a
    failed import is no refusal there. Any other error out of the read is the closed default."""
    name = _write(models, "l2p.safetensors", _damage("l2p", "intact"))

    def cannot_import(*_identity):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(upstream_gate, "_read_by_identity", cannot_import)
    assert upstream_gate.refusal_for(str(models / name)) is None

    def anything_else(*_identity):
        raise RuntimeError("not a verdict on the file")

    monkeypatch.setattr(upstream_gate, "_read_by_identity", anything_else)
    assert _number(models, name) == UNREAD


def test_a_damaged_l2p_header_logs_the_key_paths_the_gate_read(models, caplog):
    """#91 tells the reader to confirm the spelling from the driver log; the strict
    sniff that writes that line cannot read a damaged header, so the gate writes it."""
    name = _write(models, "l2p.safetensors", _damage("l2p", "an empty tensor name"))
    assert detect.sniff_zimage_l2p(str(models / name)) is False
    with caplog.at_level("INFO"):
        assert _number(models, name) == L2P
    lines = [record.getMessage() for record in caplog.records if "upstream gate:" in record.getMessage()]
    assert any("failed the strict header read" in line and "non-empty strings" in line for line in lines)
    assert any("carries the Z-Image L2P decoder (local_decoder.blocks.0.weight) and no dec_net head"
               in line for line in lines)
    assert upstream_gate.uncovered_scaled_layers(str(models / name)) == ()
    scales = _write(models, "scales.safetensors", _damage("scales", "a tensor entry twice"))
    assert upstream_gate.uncovered_scaled_layers(str(models / scales)) == ("single_blocks.0.qkv",)


def _signature_keys() -> list[tuple[str, list[str]]]:
    """One header per row of the detector's signature table, from the row's own needles."""
    return [(family, [needle + "weight" if needle.endswith(".") else needle for needle in needles])
            for family, needles in detect._SIGNATURES]


def test_a_sound_header_never_reaches_the_second_read_and_keeps_its_verdict(models, monkeypatch):
    """No new refusal for a sound file: with the lenient read patched to raise, every
    family's synthetic header, every dropped-scale fixture and the published
    MiniMax-H3-Image headers answer exactly what they answered before."""
    def entered(_path):
        raise AssertionError("the lenient read ran on a header the strict reader accepts")

    monkeypatch.setattr(upstream_gate, "_lenient_header", entered)
    seen = set()
    for index, (family, keys) in enumerate(_signature_keys()):
        path = _write_header(models / f"family-{index}.safetensors", {key: ("BF16", [1]) for key in keys})
        assert detect.detect_family_from_keys(keys) == family
        seen.add(family)
        # MiniMax-H3-Image with no metadata has no grid; nothing else refuses.
        assert _number(models, Path(path).name) == (GRID if family == "minimax_h3_image" else None), family
    assert seen == {family for family, _needles in detect._SIGNATURES} and len(seen) >= 25
    for case, (tensors, metadata) in PASSES.items():
        assert upstream_gate.refusal_for(_write_header(models / "pass.safetensors", tensors, metadata)) is None, case
    for case, (tensors, metadata, layers) in REFUSALS.items():
        path = _write_header(models / "refuse.safetensors", tensors, metadata)
        assert upstream_gate.uncovered_scaled_layers(path) == layers, case
        assert upstream_gate.refusal_for(path)[1] == SCALES, case
    for kind in h3i.KINDS:
        path = h3i.write(models / f"{kind}.safetensors", kind, blocks=2)
        assert _number(models, path.name) == H3_IMAGE_WITH_GRID
        assert upstream_gate.refusal_for(str(h3i.write(models / f"{kind}-h3.safetensors", kind, blocks=2,
                                                        audio=True))) is None
    for base, (_tensors, _metadata, number) in BASES.items():
        name = _write(models, f"{base}.safetensors", _damage(base, "intact"))
        assert _number(models, name) == number
    assert _number(models, "absent.safetensors") is None
    assert upstream_gate.refusal_for(str(models / "model.ckpt")) is None


# Real ComfyUI, CPU only. Keep these last in the file: the conftest comfy guard
# checks the namespace once this module's fixtures are torn down.
class _PlainMMAP:
    """Stands in for comfy-aimdo's ModelMMAP, whose library hooks CUDA and is not
    loaded on CPU. ``comfy.utils.load_safetensors`` asks it for an address and a
    file handle only; the header parse and every check stay ComfyUI's code."""

    def __init__(self, path):
        import ctypes
        import mmap

        self._file = open(path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        self._array = (ctypes.c_uint8 * len(self._map)).from_buffer(self._map)
        self._address = ctypes.addressof(self._array)

    def get(self):
        return self._address

    def get_file_handle(self):
        return self._file.fileno()


@pytest.fixture
def stock(real_comfy, monkeypatch):
    """``read(path)``: what ComfyUI's two load paths hand on, or None where one fails.

    ``safetensors`` is ``load_torch_file`` without DynamicVRAM (safetensors' own
    reader); ``own`` is the same call with it on (``comfy.utils.load_safetensors``).
    Each answer is ``(name -> safetensors dtype name, the string metadata entries)``."""
    import comfy_aimdo.model_mmap as model_mmap  # ComfyUI requires it; real_comfy skipped otherwise

    monkeypatch.setattr(model_mmap, "ModelMMAP", _PlainMMAP)
    names = {dtype: name for name, dtype in real_comfy.utils._TYPES.items()}

    def read(path) -> dict[str, tuple[dict[str, str], dict[str, str]] | None]:
        out = {}
        for reader, aimdo in (("safetensors", False), ("own", True)):
            monkeypatch.setattr(real_comfy.memory_management, "aimdo_enabled", aimdo)
            try:
                state, metadata = real_comfy.utils.load_torch_file(str(path), return_metadata=True)
            except Exception:
                out[reader] = None
                continue
            strings = ({key: value for key, value in metadata.items() if isinstance(value, str)}
                       if isinstance(metadata, dict) else {})
            out[reader] = ({key: names[value.dtype] for key, value in state.items()}, strings)
        return out

    return read


def _lenient(path) -> tuple[dict[str, str], dict[str, str]]:
    view = upstream_gate._lenient_header(str(path))
    return {key: info.dtype for key, info in view.tensors.items()}, view.metadata


def test_the_second_read_names_what_comfys_own_readers_hand_on(stock, tmp_path):
    """The premise of the fix, against ComfyUI itself: every damage above that the
    strict reader refuses, ComfyUI's own reader still loads, and the gate's second
    read hands its predicates the same names, dtypes and metadata strings."""
    files = {(base, damage): _damage(base, damage) for base in BASES for damage in DAMAGES}
    _meta, pairs, size = _pairs("scales")
    as_bf16 = (WEIGHT, _descriptor("BF16", [2], 0, 4))
    files["scales", "i8 last"] = _file(_raw([as_bf16, *pairs[1:], pairs[0]]), size)
    files["scales", "bf16 last"] = _file(_raw([pairs[0], *pairs[1:], as_bf16]), size)
    for label, order in (("no grid last", [("config", CONFIG), ("config", "{}")]),
                         ("grid last", [("config", "{}"), ("config", CONFIG)])):
        files["family", f"config, {label}"] = _with_metadata("family", _twice(order))
        _m, tensors, data = _pairs("family")
        files["family", f"metadata, {label}"] = _file(_raw(
            [("__metadata__", _twice([pair])) for pair in order] + tensors), data)
    read_by = {"safetensors": set(), "own": set()}
    for (base, damage), data in files.items():
        path = tmp_path / "damaged.safetensors"
        path.write_bytes(data)
        with pytest.raises(SafetensorsHeaderError):
            read_safetensors_header(path)
        answers = stock(path)
        for reader, answer in answers.items():
            if answer is not None:
                read_by[reader].add(damage)
                assert answer == _lenient(path), (base, damage, reader)
        if damage == "a dtype the strict reader does not table":
            assert answers == {"safetensors": None, "own": None}  # a refusal added where stock fails
        else:
            assert answers["own"] is not None, (base, damage)
    # safetensors' own reader takes fewer of them; none is one it reads and ComfyUI's does not.
    assert read_by["safetensors"] < read_by["own"]
    assert {"a tensor entry twice", "an empty tensor name"} <= read_by["safetensors"]


def test_what_the_gate_calls_no_header_no_stock_reader_reads(stock, real_comfy, tmp_path):
    """The other half of "no verdict only if stock cannot read the file either"."""
    from dgx_monarch.safetensors_header import MAX_HEADER_BYTES

    for base in BASES:
        for damage in NO_HEADER:
            path = tmp_path / "none.safetensors"
            path.write_bytes(_no_header(base, damage))
            assert stock(path) == {"safetensors": None, "own": None}, (base, damage)
            assert upstream_gate._lenient_header(str(path)) is None
    # Both stock readers stop at a smaller declared header than the gate's bound.
    assert real_comfy.utils._SAFETENSORS_MAX_HEADER_SIZE <= MAX_HEADER_BYTES
    path = tmp_path / "long.safetensors"
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", MAX_HEADER_BYTES + 1) + _raw(_pairs("l2p")[1]))
        handle.truncate(8 + MAX_HEADER_BYTES + 2)  # a hole: nothing is written or read
    assert stock(path) == {"safetensors": None, "own": None}
    assert upstream_gate._lenient_header(str(path)) is None
    assert upstream_gate._lenient_header(str(tmp_path / "absent.safetensors")) is None
