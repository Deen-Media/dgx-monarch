"""Property checks for the allocation-bounded safetensors trust boundary."""
from __future__ import annotations

import json
import math
import struct

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dgx_monarch.safetensors_header import (
    DTYPE_BITS,
    MAX_HEADER_BYTES,
    SafetensorsHeaderError,
    read_safetensors_header,
)


def _assert_parse_contract(path) -> None:
    try:
        parsed = read_safetensors_header(path)
    except SafetensorsHeaderError:
        return

    assert 0 < parsed.header_len <= MAX_HEADER_BYTES
    assert parsed.file_size == path.stat().st_size
    assert all(isinstance(key, str) and isinstance(value, str)
               for key, value in parsed.metadata.items())

    data_start = 8 + parsed.header_len
    cursor = 0
    for descriptor in sorted(parsed.tensors.values(), key=lambda item: (item.begin, item.end)):
        assert descriptor.data_start == data_start
        assert 0 <= descriptor.begin <= descriptor.end
        assert descriptor.begin >= cursor
        assert descriptor.start >= data_start
        assert descriptor.data_start + descriptor.end <= parsed.file_size
        expected_bits = math.prod(descriptor.shape) * DTYPE_BITS[descriptor.dtype]
        assert expected_bits % 8 == 0
        assert descriptor.nbytes == expected_bits // 8
        cursor = descriptor.end


_PROPERTY_SETTINGS = settings(
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@_PROPERTY_SETTINGS
@given(payload=st.binary(max_size=2048))
def test_arbitrary_small_files_return_validated_headers_or_typed_errors(tmp_path, payload):
    path = tmp_path / "arbitrary.safetensors"
    path.write_bytes(payload)
    _assert_parse_contract(path)


_VALID_HEADER = json.dumps({
    "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
    "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
}).encode()
_VALID_CONTAINER = struct.pack("<Q", len(_VALID_HEADER)) + _VALID_HEADER + bytes(10)


@_PROPERTY_SETTINGS
@given(
    prefix_len=st.integers(min_value=0, max_value=len(_VALID_CONTAINER)),
    mutations=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=len(_VALID_CONTAINER) - 1),
            st.integers(min_value=0, max_value=255),
        ),
        max_size=16,
    ),
    suffix=st.binary(max_size=32),
)
def test_byte_mangled_valid_containers_fail_typed_or_preserve_all_bounds(
        tmp_path, prefix_len, mutations, suffix):
    payload = bytearray(_VALID_CONTAINER)
    for index, value in mutations:
        payload[index] = value
    path = tmp_path / "mangled.safetensors"
    path.write_bytes(bytes(payload[:prefix_len]) + suffix)
    _assert_parse_contract(path)


@_PROPERTY_SETTINGS
@given(
    header_len=st.one_of(
        st.sampled_from([0, 1, MAX_HEADER_BYTES, MAX_HEADER_BYTES + 1, 2**64 - 1]),
        st.integers(min_value=0, max_value=2**64 - 1),
    ),
    suffix=st.binary(max_size=64),
)
def test_declared_u64_header_lengths_never_escape_the_typed_boundary(
        tmp_path, header_len, suffix):
    path = tmp_path / "declared-length.safetensors"
    path.write_bytes(struct.pack("<Q", header_len) + suffix)
    _assert_parse_contract(path)
