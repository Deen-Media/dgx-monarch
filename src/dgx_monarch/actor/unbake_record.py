"""The un-bake record: where every patched key's pristine bytes live, and the file it binds.

``actor.unbake`` captures a record before a LoRA bake and restores from it on
the next stack change; ``actor.fsdp_lora`` builds the same record for sharded
weights. The record is data and one check: whether the checkpoint file is
still the one it was captured from. ``actor.unbake`` re-exports both classes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from .unbake_file import _FileTensor


@dataclass
class QuantSpec:
    """Restore recipe for one quantized weight wrapper.

    module_key: attribute path of the owning op ("" allowed);
    layer_conf: this layer's entry from the file's _quantization_metadata;
    components: state-dict suffix ("weight", "weight_scale", ...) -> file
    slice; resident: components that failed byte-verify, kept as snapshots."""
    module_key: str
    layer_conf: dict
    components: dict[str, _FileTensor] = field(default_factory=dict)
    resident: dict[str, torch.Tensor] = field(default_factory=dict)
    full_precision_mm_config: bool | None = None


@dataclass
class UnbakeRecord:
    """Where every patched key's pristine bytes live.

    mapped:   plain-tensor patcher key -> checkpoint slice (byte-verified)
    quant:    quantized patcher key -> QuantSpec (per-component slices)
    resident: plain patcher key -> pristine tensor snapshot (mapping failed)
    """
    path: str
    file_size: int
    file_mtime_ns: int
    file_dev: int | None = None
    file_ino: int | None = None
    file_ctime_ns: int | None = None
    mapped: dict[str, _FileTensor] = field(default_factory=dict)
    quant: dict[str, QuantSpec] = field(default_factory=dict)
    resident: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def mapped_bytes(self) -> int:
        return (sum(ft.nbytes for ft in self.mapped.values())
                + sum(ft.nbytes for spec in self.quant.values()
                      for ft in spec.components.values()))

    @property
    def resident_bytes(self) -> int:
        tensors = list(self.resident.values())
        tensors += [t for spec in self.quant.values() for t in spec.resident.values()]
        return sum(t.numel() * t.element_size() for t in tensors)

    @property
    def resident_count(self) -> int:
        return len(self.resident) + sum(len(s.resident) for s in self.quant.values())

    def matches_stat(self, st: os.stat_result) -> bool:
        basic = st.st_size == self.file_size and st.st_mtime_ns == self.file_mtime_ns
        if not basic:
            return False
        # Captures pin filesystem identity as well as timestamp and size.
        # Optional fields permit synthetic records used by callers and tests.
        if self.file_dev is not None and st.st_dev != self.file_dev:
            return False
        if self.file_ino is not None and st.st_ino != self.file_ino:
            return False
        # Device and inode catch replacement; size and mtime catch an ordinary
        # rewrite. The change time catches the rest: a tool can put an old size
        # and mtime back after rewriting the same inode, but no tool can set
        # the change time back, and a record must never vouch for bytes that
        # changed. A metadata-only change (chmod, chown, a new hard link) moves
        # it too and costs one reload from disk instead of a lazy swap. The
        # record lives only in this Worker's memory, so nothing on another
        # machine or after a copy depends on it.
        if self.file_ctime_ns is not None and st.st_ctime_ns != self.file_ctime_ns:
            return False
        return True

    def stat_ok(self) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        return self.matches_stat(st)
