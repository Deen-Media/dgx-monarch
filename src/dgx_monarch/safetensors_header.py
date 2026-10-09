"""Strict, allocation-bounded safetensors header validation.

Family detection, the file-backed weight paths and the capacity and LoRA
pricing parse headers here, so they share one trust boundary: no descriptor is
returned until its dtype, shape, byte range and the complete file layout pass.
"""
from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .log import get_logger

MAX_HEADER_BYTES = 256 * 1024 * 1024

# Bits per logical element. Any other dtype (safetensors 0.8.0 also defines
# F6_E2M3 and F6_E3M2) raises UnsupportedSafetensorsDtypeError once the layout
# checks pass. F4 packs two values per byte, so its serialized shape has twice
# the final dimension of the packed torch tensor.
DTYPE_BITS: dict[str, int] = {
    "F4": 4,
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "C64": 64,
}

log = get_logger(__name__)


class SafetensorsHeaderError(ValueError):
    """The file is not a structurally valid safetensors container."""


class UnsupportedSafetensorsDtypeError(SafetensorsHeaderError):
    """The container is valid so far, but its dtype is unavailable here.

    Slab callers may retry this capability miss through the stock safetensors
    loader. It stays a header error so strict consumers such as topology
    sniffing and unbake capture fail closed unless they opt into that fallback.
    """


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int
    data_start: int

    @property
    def start(self) -> int:
        return self.data_start + self.begin

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    def as_header_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offsets": [self.begin, self.end],
        }


@dataclass(frozen=True, slots=True)
class SafetensorsFileIdentity:
    """Compact identity for the exact inode whose header was validated."""

    file_dev: int
    file_ino: int
    file_size: int
    file_mtime_ns: int
    file_ctime_ns: int

    @classmethod
    def from_stat(cls, stat: os.stat_result) -> SafetensorsFileIdentity:
        return cls(
            file_dev=stat.st_dev,
            file_ino=stat.st_ino,
            file_size=stat.st_size,
            file_mtime_ns=stat.st_mtime_ns,
            file_ctime_ns=stat.st_ctime_ns,
        )


@dataclass(frozen=True)
class SafetensorsHeader:
    tensors: dict[str, TensorDescriptor]
    metadata: dict[str, str]
    header_len: int
    file_size: int
    file_dev: int
    file_ino: int
    file_mtime_ns: int
    file_ctime_ns: int

    @property
    def file_identity(self) -> SafetensorsFileIdentity:
        return SafetensorsFileIdentity(
            file_dev=self.file_dev,
            file_ino=self.file_ino,
            file_size=self.file_size,
            file_mtime_ns=self.file_mtime_ns,
            file_ctime_ns=self.file_ctime_ns,
        )


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SafetensorsHeaderError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _is_uint(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _read_exact(fd: int, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = os.pread(fd, remaining, offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_safetensors_header(path: str | os.PathLike[str]) -> SafetensorsHeader:
    """Parse and validate a safetensors header without reading tensor bytes.

    Validation covers a bounded header allocation, complete descriptors, byte
    ranges that match dtype and shape, in-file offsets, non-overlapping ranges
    (padding gaps are legal) and a file identity that holds for the whole read.
    """
    display = Path(path).name
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SafetensorsHeaderError(f"{display}: cannot open checkpoint ({exc})") from exc

    try:
        before = os.fstat(fd)
        prefix = _read_exact(fd, 0, 8)
        if len(prefix) != 8:
            raise SafetensorsHeaderError(
                f"{display}: file too small for a safetensors header"
            )
        header_len = struct.unpack("<Q", prefix)[0]
        if not 0 < header_len <= MAX_HEADER_BYTES:
            raise SafetensorsHeaderError(
                f"{display}: implausible safetensors header length {header_len}"
            )
        data_start = 8 + header_len
        if data_start > before.st_size:
            raise SafetensorsHeaderError(
                f"{display}: truncated safetensors header (declared {header_len} bytes, "
                f"file has {max(0, before.st_size - 8)})"
            )
        raw = _read_exact(fd, 8, header_len)
        if len(raw) != header_len:
            raise SafetensorsHeaderError(
                f"{display}: truncated safetensors header (declared {header_len} bytes, "
                f"read {len(raw)})"
            )
        try:
            header = json.loads(raw, object_pairs_hook=_json_object)
        except SafetensorsHeaderError:
            raise
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise SafetensorsHeaderError(
                f"{display}: could not parse safetensors JSON header ({exc})"
            ) from exc
        after_fd = os.fstat(fd)
        try:
            after_path = os.stat(path)
        except OSError as exc:
            raise SafetensorsHeaderError(
                f"{display}: checkpoint path changed while reading header"
            ) from exc
        identity = SafetensorsFileIdentity.from_stat(before)
        if (
            SafetensorsFileIdentity.from_stat(after_fd) != identity
            or SafetensorsFileIdentity.from_stat(after_path) != identity
        ):
            raise SafetensorsHeaderError(f"{display}: checkpoint changed while reading header")
    finally:
        os.close(fd)

    if not isinstance(header, dict):
        raise SafetensorsHeaderError(
            f"{display}: safetensors JSON header must be an object, "
            f"got {type(header).__name__}"
        )

    raw_metadata = header.get("__metadata__", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, dict):
        raise SafetensorsHeaderError(
            f"{display}: safetensors __metadata__ must be an object"
        )
    if any(not isinstance(k, str) or not isinstance(v, str)
           for k, v in raw_metadata.items()):
        raise SafetensorsHeaderError(
            f"{display}: safetensors __metadata__ keys and values must be strings"
        )

    tensors: dict[str, TensorDescriptor] = {}
    unsupported_dtypes: list[tuple[str, str]] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name:
            raise SafetensorsHeaderError(f"{display}: tensor names must be non-empty strings")
        if not isinstance(info, dict):
            raise SafetensorsHeaderError(
                f"{display}: tensor descriptor for {name!r} must be an object"
            )
        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if (not isinstance(shape, list)
                or not all(_is_uint(dimension) for dimension in shape)):
            raise SafetensorsHeaderError(
                f"{display}: malformed safetensors tensor descriptor for {name!r}"
            )
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(_is_uint(offset) for offset in offsets)
                or offsets[0] > offsets[1]):
            raise SafetensorsHeaderError(
                f"{display}: malformed safetensors tensor descriptor for {name!r}"
            )
        if not isinstance(dtype, str) or not dtype:
            raise SafetensorsHeaderError(
                f"{display}: malformed safetensors tensor descriptor for {name!r}"
            )
        begin, end = offsets
        dtype_bits = DTYPE_BITS.get(dtype)
        if dtype_bits is None:
            # An unknown dtype is a capability miss only once the structure
            # that does not depend on dtype proves sound. Keep parsing so bad
            # bounds and overlaps stay structural errors and never reach the
            # stock-loader fallback.
            unsupported_dtypes.append((name, dtype))
        else:
            expected_bits = math.prod(shape) * dtype_bits
            if expected_bits % 8:
                raise SafetensorsHeaderError(
                    f"{display}: packed tensor {name!r} has a non-byte-aligned shape"
                )
            expected = expected_bits // 8
            if end - begin != expected:
                raise SafetensorsHeaderError(
                    f"{display}: tensor {name!r} byte range is {end - begin}, "
                    f"but dtype/shape require {expected}"
                )
        if data_start + end > before.st_size:
            raise SafetensorsHeaderError(
                f"{display}: tensor {name!r} data range exceeds file size"
            )
        tensors[name] = TensorDescriptor(
            dtype=dtype,
            shape=tuple(shape),
            begin=begin,
            end=end,
            data_start=data_start,
        )

    cursor = 0
    for name, desc in sorted(tensors.items(), key=lambda item: (item[1].begin, item[1].end)):
        if desc.begin < cursor:
            raise SafetensorsHeaderError(
                f"{display}: tensor {name!r} overlaps byte offset {cursor}"
            )
        if desc.begin > cursor:
            # Some producers emit alignment padding between tensors, a legal
            # layout. Each descriptor was bounds- and size-checked above, and
            # overlap stays an error.
            log.debug("%s: %d alignment-padding bytes before tensor %r",
                      display, desc.begin - cursor, name)
        cursor = desc.end
    data_size = before.st_size - data_start
    if cursor > data_size:
        raise SafetensorsHeaderError(
            f"{display}: tensor ranges cover {cursor} data bytes, file contains {data_size}"
        )
    if cursor < data_size:
        log.debug("%s: %d trailing padding bytes after the last tensor",
                  display, data_size - cursor)

    if unsupported_dtypes:
        name, dtype = unsupported_dtypes[0]
        raise UnsupportedSafetensorsDtypeError(
            f"{display}: unsupported safetensors dtype {dtype!r} for {name!r}"
        )

    return SafetensorsHeader(
        tensors=tensors,
        metadata=dict(raw_metadata),
        header_len=header_len,
        file_size=before.st_size,
        file_dev=before.st_dev,
        file_ino=before.st_ino,
        file_mtime_ns=before.st_mtime_ns,
        file_ctime_ns=before.st_ctime_ns,
    )
