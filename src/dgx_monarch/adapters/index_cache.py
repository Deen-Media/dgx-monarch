"""A small bounded cache for the integer index tensors of the Ulysses pad path.

The pad rows and the joint key order of a padded Ulysses attention depend only
on integer row counts, which stay fixed across every block and step of one
forward. Building their index tensors once per layout, on the device that
uses them, takes the per-call host reads out of the attention path. The values
are the same integers the per-call build produced, so the output does not
change; this memoizes index arithmetic, never activations.

Each key names its device, so an entry never serves another device. The bound
keeps a long session with many prompt lengths from growing the cache: one
render uses a handful of layouts, and the least recently used entry goes first.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

import torch

Value = TypeVar("Value")


def device_key(device) -> torch.device:
    """The concrete device an index lives on, for use in a cache key.

    A string and a ``torch.device`` name the same device, and a CUDA device
    without an index means the current one, so all three forms share a key.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


class BoundedIndexCache(Generic[Value]):
    """A least recently used map with a fixed number of entries."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._entries: OrderedDict[Hashable, Value] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Value | None:
        """The entry for ``key``, marked as just used, or ``None``."""
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def peek(self, key: Hashable) -> Value | None:
        """The entry for ``key`` without changing its place in the order."""
        with self._lock:
            return self._entries.get(key)

    def put(self, key: Hashable, value: Value) -> Value:
        """Store ``value``, drop the oldest entries past the bound, return it."""
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.limit:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
