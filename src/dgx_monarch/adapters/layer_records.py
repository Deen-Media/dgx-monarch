"""Name a checkpoint's quantization kind from its per-layer ``comfy_quant`` records.

A safetensors header tells some kinds apart and not others
(adapters/detect.py ``detect_quant_from_header``): a ConvRot W4A4 file reads
``int8`` as an INT8 file does, and an FP8, an MXFP8 and an NVFP4 file all read
``fp8``. The file's own layer records say which it is. Each record is a U8
tensor of a few dozen bytes holding JSON, read here at the offset the header
gives; no weight is read. The driver uses this before dispatch, to refuse
where a Worker would refuse after a full load and to describe the file it
holds (nodes/model_patch_preflight.py).
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

from ..safetensors_header import (
    SafetensorsFileIdentity,
    SafetensorsHeaderError,
    read_safetensors_header,
)

# ComfyUI bounds one per-layer comfy_quant record far above its real size, tens of bytes.
_LAYER_RECORD_MAX_BYTES = 4096
# What each file read so far names, by the identity of the file that was read.
# The key is the device, inode, size and times of the one descriptor the header
# and the records came from, never those of a path looked at beforehand: a
# file put in the path's place between a look and a read is then filed under
# its own identity, and cannot answer for the file it replaced.
_CACHE_SIZE = 64
_FORMATS_BY_IDENTITY: OrderedDict[SafetensorsFileIdentity, frozenset[str] | None] = OrderedDict()


def _open_readonly(path: str | Path) -> int:
    return os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))


def _records_of(fd: int, identity: SafetensorsFileIdentity) -> frozenset[str] | None:
    """The formats the records of the file behind ``fd`` name; None when it carries none readable.

    The header is parsed through the descriptor's own /proc entry, which names
    the open file and not the path, and its identity must be the descriptor's.
    Raises OSError or ValueError where the file cannot be read as that file.
    """
    parsed = read_safetensors_header(f"/proc/self/fd/{fd}")
    if parsed.file_identity != identity:
        raise OSError("the file changed while its header was read")
    records = [descriptor for name, descriptor in parsed.tensors.items()
               if name == "comfy_quant" or name.endswith(".comfy_quant")]
    if not records:
        return None
    formats: set[str] = set()
    for descriptor in records:
        if descriptor.dtype != "U8" or not 0 < descriptor.nbytes <= _LAYER_RECORD_MAX_BYTES:
            return None
        raw = os.pread(fd, descriptor.nbytes, descriptor.start)
        record = json.loads(raw) if len(raw) == descriptor.nbytes else None
        fmt = record.get("format") if isinstance(record, dict) else None
        if not isinstance(fmt, str):
            return None
        formats.add(fmt)
    return frozenset(formats)


def sniff_layer_formats(path: str | Path) -> frozenset[str] | None:
    """The formats a file's per-layer ``comfy_quant`` records name, or None when it has none readable.

    None means the file carries no record, or one of them could not be read
    or decoded, so the caller knows the answer is incomplete. A file already
    read is answered from its identity with no open. Any other is opened
    once: the identity comes from that descriptor, the header and the records
    are read through it, and the answer is kept under that identity only if
    the descriptor still shows it afterwards. A file that cannot be read is
    None and is not kept.
    """
    try:
        seen = SafetensorsFileIdentity.from_stat(os.stat(path))
    except OSError:
        return None
    if seen in _FORMATS_BY_IDENTITY:
        _FORMATS_BY_IDENTITY.move_to_end(seen)
        return _FORMATS_BY_IDENTITY[seen]
    try:
        fd = _open_readonly(path)
    except OSError:
        return None
    try:
        identity = SafetensorsFileIdentity.from_stat(os.fstat(fd))
        formats = _records_of(fd, identity)
        if SafetensorsFileIdentity.from_stat(os.fstat(fd)) != identity:
            return None
    except (SafetensorsHeaderError, OSError, ValueError):
        return None
    finally:
        os.close(fd)
    _FORMATS_BY_IDENTITY[identity] = formats
    while len(_FORMATS_BY_IDENTITY) > _CACHE_SIZE:
        _FORMATS_BY_IDENTITY.popitem(last=False)
    return formats


def name_int8_header_kind(path: str | Path) -> str | None:
    """What a file whose header reads ``int8`` is by its own layer records.

    A ConvRot W4A4 file stores I8 weights beside a ``weight_scale`` as an INT8
    file does, so the header cannot tell them apart; the records can. Returns
    ``"w4a4"`` when a record names ``convrot_w4a4``, as one such layer names
    the whole file on a Worker (actor/store_detect.py), ``"int8"`` when the
    records are readable and none does, and None when the file has no readable
    record, which leaves the question open.
    """
    formats = sniff_layer_formats(path)
    if formats is None:
        return None
    return "w4a4" if "convrot_w4a4" in formats else "int8"


# The kind each record format gives a file whose header reads fp8.
_FP8_HEADER_RECORD_KINDS = {"float8_e4m3fn": "fp8", "float8_e5m2": "fp8",
                            "mxfp8": "mxfp8", "nvfp4": "nvfp4"}


def name_fp8_header_kind(path: str | Path) -> str | None:
    """What a file whose header reads ``fp8`` is by its own layer records.

    Returns ``"fp8"``, ``"mxfp8"`` or ``"nvfp4"`` when every record names that
    one kind, and None when the file has no readable record, names a format
    outside these, or mixes kinds: a Worker names a mixed file by its first
    quantized layer (actor/store_detect.py), an order the header does not hold.
    """
    formats = sniff_layer_formats(path)
    if not formats:
        return None
    kinds = {_FP8_HEADER_RECORD_KINDS.get(fmt) for fmt in formats}
    return kinds.pop() if len(kinds) == 1 else None


__all__ = ["name_fp8_header_kind", "name_int8_header_kind", "sniff_layer_formats"]
