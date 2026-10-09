"""Wire contract for Qwen Image 2.1's per-sampling-run prefix cache."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict


class QwenImage21CacheOptions(TypedDict):
    device: str
    dtype: str


_DEVICES = frozenset({"auto", "gpu", "cpu", "off"})
_DTYPES = frozenset({"default", "int8", "int4"})


def normalize_qwen_image21_cache(value: object | None) -> QwenImage21CacheOptions | None:
    """Validate and copy the render-local cache policy.

    ``None`` means the adapter's cache-off default. An explicit ``off`` remains
    serializable so saved workflows say why the cache is idle.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"device", "dtype"}:
        raise ValueError("qwen_image21_cache must contain exactly 'device' and 'dtype'")
    device, dtype = value["device"], value["dtype"]
    if not isinstance(device, str) or device not in _DEVICES:
        raise ValueError("qwen_image21_cache device must be auto, gpu, cpu, or off")
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ValueError("qwen_image21_cache dtype must be default, int8, or int4")
    return {"device": device, "dtype": dtype}
