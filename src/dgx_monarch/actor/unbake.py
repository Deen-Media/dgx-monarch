"""Restore pristine LoRA weights from file-backed checkpoint references.

ComfyUI keeps a pristine copy of every patched weight. On unified-memory hosts
that backup is memory the host cannot reclaim. This module records verified
checkpoint slices before baking, then restores only patched tensors without a
full model reload.

Quantized records include data and scale components needed to reconstruct each
wrapper. Components that cannot be verified stay resident as CPU snapshots;
records over the resident cap are discarded in favor of a full reload. Bulk
reads evict clean checkpoint pages afterward. ComfyUI imports stay local for
CPU-only tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from ..log import get_logger
from .unbake_file import (
    _ST_TO_TORCH as _ST_TO_TORCH,
)
from .unbake_file import (
    _bytes_match,
    _candidate_file_bindings,
    _candidate_file_keys,
    _cast_match,
    _decode_live_quant_conf,
    _direct_quant_conf_reason,
    _direct_quant_module_supported,
    _FileTensor,
    _InvalidQuantConfError,
    _quant_conf_matches_live,
    _quant_layer_conf,
    read_safetensors_header,
)

log = get_logger(__name__)

# Discard records whose resident fallback costs more than a full reload.
_RESIDENT_CAP_BYTES = 2 << 30


# Refuse unknown formats during capture before any partial restore attempt.
RESTORABLE_QUANT_FORMATS = frozenset(
    {"float8_e4m3fn", "float8_e5m2", "mxfp8", "nvfp4", "int8_tensorwise"})


class UnbakeError(RuntimeError):
    """A lazy un-bake cannot proceed (stale file, shape drift, read failure).

    The model may be partially restored, so the caller must discard it and
    perform a full reload.
    """


class CaptureAborted(RuntimeError):
    """Capture discarded the record: the checkpoint changed, a quantized layer cannot be rebuilt
    exactly from one checkpoint binding and its live contract, or the resident set went over the cap."""


def drop_file_cache(path: str) -> None:
    """Best-effort eviction of clean file pages with POSIX_FADV_DONTNEED."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


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
        # ctime is not compared: metadata-only changes move it without changing
        # contents. Size and mtime catch mutation; device and inode catch replacement.
        return True

    def stat_ok(self) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        return self.matches_stat(st)


def live_tensor(model: Any, key: str) -> torch.Tensor:
    """The live tensor at a state-dict-style attribute path (nn.Module walk)."""
    obj = model
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def _is_quantized(tensor: Any) -> bool:
    """Duck-typed QuantizedTensor check (comfy_kitchen wrapper subclass)."""
    return hasattr(tensor, "layout_cls") and hasattr(tensor, "params")


def capture_unbake_record(model: Any, keys, base_path: str) -> UnbakeRecord:
    """Build a record from pristine live tensors before baking.

    Plain keys map when shape and sampled bytes match, at the file's dtype or
    after a cast to the live one. Quantized keys need one checkpoint binding
    whose quantization metadata matches the live layer. Unmatched tensors and
    components stay as CPU snapshots; excess resident memory aborts capture.
    """
    header, metadata, parsed = read_safetensors_header(
        base_path, want_metadata=True, want_identity=True
    )
    record = UnbakeRecord(
        path=base_path,
        file_size=parsed.file_size,
        file_mtime_ns=parsed.file_mtime_ns,
        file_dev=parsed.file_dev,
        file_ino=parsed.file_ino,
        file_ctime_ns=parsed.file_ctime_ns,
    )

    def _keep(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to("cpu").clone()

    def _capture_identity_matches(st: os.stat_result) -> bool:
        return record.matches_stat(st)

    try:
        with open(base_path, "rb") as f:
            fd = f.fileno()
            try:
                path_before = os.stat(base_path)
            except OSError as exc:
                raise CaptureAborted(
                    "checkpoint path changed after its header was validated"
                ) from exc
            if not _capture_identity_matches(os.fstat(fd)) or not _capture_identity_matches(
                path_before
            ):
                raise CaptureAborted(
                    "checkpoint changed after its header was validated"
                )

            def _map_plain(key: str, tensor: torch.Tensor) -> bool:
                for cand in _candidate_file_keys(key):
                    ft = header.get(cand)
                    if (
                        ft is None
                        or ft.dtype != tensor.dtype
                        or ft.shape != tuple(tensor.shape)
                    ):
                        continue
                    if _bytes_match(fd, ft, tensor):
                        record.mapped[key] = ft
                        return True
                for cand in _candidate_file_keys(key):
                    ft = header.get(cand)
                    if ft is None:
                        continue
                    if _cast_match(fd, ft, tensor):
                        record.mapped[key] = _FileTensor(
                            dtype=ft.dtype,
                            shape=ft.shape,
                            start=ft.start,
                            nbytes=ft.nbytes,
                            cast_to=tensor.dtype,
                        )
                        return True
                return False

            def _capture_quant(key: str, tensor: torch.Tensor) -> None:
                module_key, _, param_name = key.rpartition(".")
                module = live_tensor(model, module_key) if module_key else model
                bindings = []
                for candidate, model_prefix in _candidate_file_bindings(key):
                    if not candidate.endswith(".weight") or candidate not in header:
                        continue
                    file_layer = candidate[:-len(".weight")]
                    try:
                        conf, conf_source = _quant_layer_conf(
                            metadata, header, file_layer, model_prefix, fd
                        )
                    except _InvalidQuantConfError as exc:
                        raise CaptureAborted(
                            f"quantized key {key} has malformed checkpoint "
                            "quantization authority; record discarded, reload "
                            "contract applies"
                        ) from exc
                    if conf is not None:
                        bindings.append((file_layer, conf, conf_source))
                if len(bindings) != 1:
                    raise CaptureAborted(
                        f"quantized key {key} has {len(bindings)} coherent checkpoint "
                        "bindings; record discarded, reload contract applies"
                    )
                file_layer, conf, conf_source = bindings[0]
                legacy_scaled_fp8 = conf_source == "legacy"
                if param_name != "weight" or not hasattr(module, "state_dict"):
                    # Without matching quant metadata, reconstruction would
                    # require a lossy dequantize and requantize cycle.
                    raise CaptureAborted(
                        f"quantized key {key} has no matching quantization metadata in the "
                        "checkpoint; record discarded, reload contract applies"
                    )
                if conf.get("format") not in RESTORABLE_QUANT_FORMATS:
                    raise CaptureAborted(
                        f"quant format {conf.get('format')!r} on {key} has no exact restore "
                        "recipe yet; record discarded, reload contract applies"
                    )
                comps = module.state_dict()
                live_conf = _decode_live_quant_conf(comps.get("comfy_quant"))
                if conf_source == "direct":
                    reason = _direct_quant_conf_reason(conf)
                    if reason is None:
                        reason = (
                            "the live layer marker did not decode"
                            if live_conf is None
                            else _direct_quant_conf_reason(live_conf)
                        )
                    if reason is None and not _direct_quant_module_supported(
                        module, conf, comps
                    ):
                        reason = (
                            "the owning module does not carry the weight and "
                            "scale state the restore reads"
                        )
                    if reason is not None:
                        raise CaptureAborted(
                            f"direct quantization contract on {key} cannot be "
                            f"reproduced by generic un-bake ({reason}); record "
                            "discarded, reload contract applies"
                        )
                recorded_full_precision = conf.get(
                    "full_precision_matrix_mult", False
                )
                if not isinstance(recorded_full_precision, bool):
                    raise CaptureAborted(
                        f"quantized layer {key} has a non-boolean "
                        "full-precision matmul contract; record discarded, "
                        "reload contract applies"
                    )
                if legacy_scaled_fp8:
                    expected_full_precision = recorded_full_precision
                    required = {"weight", "weight_scale"}
                    unexpected_scales = {
                        name
                        for name in comps
                        if name.startswith("weight_scale_")
                    }
                    if (
                        any(
                            not isinstance(comps.get(name), torch.Tensor)
                            for name in required
                        )
                        or unexpected_scales
                        or f"{file_layer}.weight_scale" in header
                        or getattr(module, "quant_format", None) != conf["format"]
                        or getattr(module, "_full_precision_mm_config", None)
                        is not expected_full_precision
                        or live_conf != conf
                    ):
                        raise CaptureAborted(
                            f"legacy scaled_fp8 layer {key} does not match Comfy's "
                            "live converted quantization contract; record discarded, "
                            "reload contract applies"
                        )
                elif (
                    getattr(module, "quant_format", None) != conf["format"]
                    or not _quant_conf_matches_live(
                        conf,
                        live_conf,
                    )
                    or getattr(module, "_full_precision_mm_config", None)
                    is not recorded_full_precision
                ):
                    raise CaptureAborted(
                        f"quantized layer {key} does not match Comfy's live "
                        "quantization contract; record discarded, reload "
                        "contract applies"
                    )
                spec = QuantSpec(
                    module_key=module_key,
                    layer_conf=conf,
                    full_precision_mm_config=recorded_full_precision,
                )
                for comp_name, comp in comps.items():
                    if comp_name == "comfy_quant" or comp is None:
                        continue
                    if comp_name != param_name and not comp_name.startswith(
                        param_name + "_scale"
                    ):
                        continue  # Bake does not modify bias or input scale.
                    file_component = (
                        "scale_weight"
                        if legacy_scaled_fp8 and comp_name == "weight_scale"
                        else comp_name
                    )
                    ft = header.get(f"{file_layer}.{file_component}")
                    if ft is not None and _bytes_match(fd, ft, comp):
                        spec.components[comp_name] = ft
                    else:
                        spec.resident[comp_name] = _keep(comp)
                record.quant[key] = spec

            for key in keys:
                tensor = live_tensor(model, key)
                if _is_quantized(tensor):
                    _capture_quant(key, tensor)
                elif not _map_plain(key, tensor):
                    record.resident[key] = _keep(tensor)

            try:
                path_after = os.stat(base_path)
            except OSError as exc:
                raise CaptureAborted(
                    "checkpoint path changed while capturing the un-bake record"
                ) from exc
            if not _capture_identity_matches(os.fstat(fd)) or not _capture_identity_matches(
                path_after
            ):
                raise CaptureAborted(
                    "checkpoint changed while capturing the un-bake record"
                )
    finally:
        drop_file_cache(base_path)
    if record.resident_bytes > _RESIDENT_CAP_BYTES:
        raise CaptureAborted(
            f"{record.resident_count} keys ({record.resident_bytes / 2**30:.2f} GiB, over the resident "
            "cap) failed byte-verify against the checkpoint; record discarded, reload contract applies"
        )
    if record.resident_count:
        names = list(record.resident)
        names += [f"{k}::{c}" for k, spec in record.quant.items() for c in spec.resident]
        log.warning(
            "unbake capture: %d/%d keys did not verify against the checkpoint and stay "
            "resident (%.2f GiB kept, %.2f GiB file-backed): %s%s",
            record.resident_count, len(record.mapped) + len(record.quant) + len(record.resident),
            record.resident_bytes / 2**30, record.mapped_bytes / 2**30,
            ", ".join(names[:8]), "..." if len(names) > 8 else "",
        )
    else:
        log.info(
            "unbake capture: %d keys file-backed (%.2f GiB of backup replaced by "
            "checkpoint references)", len(record.mapped) + len(record.quant),
            record.mapped_bytes / 2**30,
        )
    return record


def _copy_into(model: Any, key: str, src: torch.Tensor) -> None:
    target = live_tensor(model, key)
    if target.dtype != src.dtype or tuple(target.shape) != tuple(src.shape):
        raise UnbakeError(
            f"live tensor {key} is {target.dtype}/{tuple(target.shape)} but the record "
            f"holds {src.dtype}/{tuple(src.shape)}; the model changed after capture"
        )
    dst = target.data if isinstance(target, torch.nn.Parameter) else target
    dst.copy_(src.to(dst.device))


_DIO_ALIGN = 4096  # Conservative O_DIRECT block alignment.


def _open_for_restore(path: str) -> tuple[int, bool]:
    """Open a checkpoint for bulk reads, preferring O_DIRECT.

    Direct I/O skips the page cache, so restore reads do not compete with kernel
    reclaim when another process holds the pool.
    Fall back where unsupported.
    """
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT), True
    except OSError:
        return os.open(path, os.O_RDONLY), False


def _aligned_buffer(nbytes: int) -> torch.Tensor:
    """A uint8 CPU tensor whose data pointer is _DIO_ALIGN-aligned."""
    raw = torch.empty(nbytes + _DIO_ALIGN, dtype=torch.uint8)
    shift = (-raw.data_ptr()) % _DIO_ALIGN
    return raw[shift:shift + nbytes]


def _pread_direct(fd: int, start: int, nbytes: int, buf: torch.Tensor,
                  dst_off: int) -> int:
    """O_DIRECT read covering [start, start+nbytes) into buf at dst_off.

    O_DIRECT requires aligned offsets, addresses, and lengths. The read starts
    at the aligned floor and returns the payload's leading pad.
    """
    astart = start & ~(_DIO_ALIGN - 1)
    pad = start - astart
    need = pad + nbytes
    want = need + (-need) % _DIO_ALIGN
    mv = buf.numpy().data[dst_off:dst_off + want]
    pos = 0
    while pos < need:
        n = os.preadv(fd, [mv[pos:]], astart + pos)
        if n <= 0:
            raise UnbakeError(f"short direct read at file offset {astart + pos}")
        pos += n
    return pad


def _pread_into(fd: int, offset: int, nbytes: int, scratch: torch.Tensor) -> None:
    """pread exactly nbytes at offset into the head of a uint8 CPU tensor, with no intermediate copy."""
    _pread_into_at(fd, offset, nbytes, scratch, 0)


def _pread_into_at(fd: int, offset: int, nbytes: int, scratch: torch.Tensor,
                   dst_off: int) -> None:
    """pread exactly nbytes at offset into scratch[dst_off:dst_off+nbytes]."""
    mv = scratch.numpy().data[dst_off:dst_off + nbytes]
    pos = 0
    while pos < nbytes:
        n = os.preadv(fd, [mv[pos:]], offset + pos)
        if n <= 0:
            raise UnbakeError(f"short read at file offset {offset + pos}")
        pos += n


def _restore_quant(model: Any, key: str, spec: QuantSpec, comps: dict) -> None:
    """Rebuild a quantized weight's wrapper from pristine checkpoint bytes.

    ``comps`` contains CPU tensors in checkpoint dtype and shape; resident
    snapshots fill gaps. Layout constructors define each format's scale views
    and parameter fields.
    """
    import comfy.quant_ops as quant_ops

    # Quantized owners are duck-typed ComfyUI modules rather than tensors.
    module: Any = live_tensor(model, spec.module_key) if spec.module_key else model
    fmt = spec.layer_conf.get("format")
    if fmt is None or getattr(module, "quant_format", None) != fmt:
        raise UnbakeError(
            f"quantized layer {spec.module_key}: live format "
            f"{getattr(module, 'quant_format', None)!r} != record format {fmt!r}"
        )
    if (
        spec.full_precision_mm_config is not None
        and getattr(module, "_full_precision_mm_config", None)
        is not spec.full_precision_mm_config
    ):
        raise UnbakeError(
            f"quantized layer {spec.module_key}: live full-precision matmul config "
            "does not match the un-bake record"
        )
    qconfig = quant_ops.QUANT_ALGOS[fmt]
    device = module.weight.device

    def comp(name: str) -> torch.Tensor:
        if name in comps:
            return comps[name].to(device)
        if name in spec.resident:
            return spec.resident[name].to(device)
        raise UnbakeError(f"quantized layer {spec.module_key}: missing component {name}")

    weight = comp("weight").view(qconfig["storage_t"])
    if fmt in ("float8_e4m3fn", "float8_e5m2"):
        scales: dict[str, Any] = {"scale": comp("weight_scale").float()}
    elif fmt == "mxfp8":
        scales = {"scale": comp("weight_scale").view(torch.float8_e8m0fnu)}
    elif fmt == "nvfp4":
        scales = {"scale": comp("weight_scale_2").float(),
                  "block_scale": comp("weight_scale").view(torch.float8_e4m3fn)}
    elif fmt == "int8_tensorwise":
        scales = {"scale": comp("weight_scale").float()}
        params_conf = spec.layer_conf.get("params", {})
        if not isinstance(params_conf, dict):
            params_conf = {}
        if spec.layer_conf.get("convrot", params_conf.get("convrot", False)):
            scales["convrot"] = True
            scales["convrot_groupsize"] = int(
                spec.layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
    else:
        raise UnbakeError(f"unsupported quant format {fmt!r} in record")

    layout_type = qconfig["comfy_tensor_layout"]
    layout_cls = quant_ops.get_layout_class(layout_type)
    params = layout_cls.Params(
        **scales, orig_dtype=module.factory_kwargs["dtype"], orig_shape=module._orig_shape)
    module.weight = torch.nn.Parameter(
        quant_ops.QuantizedTensor(weight, layout_type, params), requires_grad=False)


def restore_pristine(model: Any, record: UnbakeRecord, after_key=None) -> dict:
    """Overwrite every recorded key's live state with its pristine bytes.

    Read plain keys in file order, rebuild quantized wrappers from their
    components, and copy resident snapshots. A reader thread reads ahead so
    ``after_key`` can overlap GPU baking with disk I/O. Any inconsistency may
    leave partial state; the caller must discard the model and reload fully.
    """
    import queue
    import threading

    if not record.stat_ok():
        raise UnbakeError(
            f"checkpoint {record.path} changed since capture "
            "(filesystem identity/size/time mismatch)"
        )
    restored = 0
    plain = sorted(record.mapped.items(), key=lambda kv: kv[1].start)
    q: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()

    # Three reused aligned buffers: no per-key mappings or repeated page faults,
    # and the reader can fill ahead (up to two keys) while the consumer copies.
    def _unit_bytes(items):
        return sum(ft.nbytes + 3 * _DIO_ALIGN for ft in items)

    unit_sizes = [_unit_bytes([ft]) for _, ft in plain]
    unit_sizes += [_unit_bytes(spec.components.values()) for spec in record.quant.values()]
    pool: queue.Queue = queue.Queue()
    for _ in range(3):
        pool.put(_aligned_buffer(max(unit_sizes, default=0)))

    def _take(src_q: queue.Queue):
        while not stop.is_set():
            try:
                return src_q.get(timeout=0.5)
            except queue.Empty:
                continue
        return None

    def _put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _fill(fd: int, direct: bool, buf: torch.Tensor, items: dict) -> dict:
        """Read components into ``buf`` and return typed payload views."""
        out, off = {}, 0
        for name, ft in items.items():
            if direct:
                pad = _pread_direct(fd, ft.start, ft.nbytes, buf, off)
            else:
                _pread_into_at(fd, ft.start, ft.nbytes, buf, off)
                pad = 0
            view = buf[off + pad:off + pad + ft.nbytes]
            elem = torch.empty((), dtype=ft.dtype).element_size()
            if view.data_ptr() % elem:
                view = view.clone()  # rare: payload landed dtype-misaligned
            out[name] = view.view(ft.dtype).view(ft.shape)
            step = pad + ft.nbytes
            off += step + (-step) % _DIO_ALIGN
        return out

    def _reader() -> None:
        fd = None
        try:
            # Report open races through the reader channel so the consumer
            # always receives a terminal item.
            fd, direct = _open_for_restore(record.path)
            # Validate the opened inode and current path to close the stat/open
            # race. Later path replacement cannot change this descriptor.
            opened = os.fstat(fd)
            current = os.stat(record.path)
            if not record.matches_stat(opened) or not record.matches_stat(current):
                raise UnbakeError(
                    f"checkpoint {record.path} changed while opening it for restore"
                )
            for key, ft in plain:
                buf = _take(pool)
                if buf is None:
                    return
                views = _fill(fd, direct, buf, {"": ft})
                if not _put(("plain", key, ft, (buf, views[""]))):
                    return
            for key, spec in record.quant.items():
                buf = _take(pool)
                if buf is None:
                    return
                views = _fill(fd, direct, buf, spec.components)
                if not _put(("quant", key, spec, (buf, views))):
                    return
            # Recheck after reads to detect in-place mutation or path replacement.
            opened_after = os.fstat(fd)
            current_after = os.stat(record.path)
            if (not record.matches_stat(opened_after)
                    or not record.matches_stat(current_after)):
                raise UnbakeError(
                    f"checkpoint {record.path} changed while restore reads were in progress"
                )
            _put(("done", None, None, None))
        except BaseException as exc:  # Preserve failures for the consumer.
            _put(("error", exc, None, None))
        finally:
            if fd is not None:
                os.close(fd)

    reader = threading.Thread(target=_reader, name="unbake-reader", daemon=True)
    reader.start()
    try:
        while True:
            try:
                kind, key, meta, payload = q.get(timeout=0.5)
            except queue.Empty:
                if not reader.is_alive():
                    raise UnbakeError(
                        "checkpoint reader exited without a terminal result"
                    ) from None
                continue
            if kind == "done":
                break
            if kind == "error":
                if isinstance(key, OSError):
                    raise UnbakeError(f"checkpoint read failed: {key!r}") from key
                raise key
            buf, views = payload
            if kind == "plain":
                src = views if meta.cast_to is None else views.to(meta.cast_to)
                _copy_into(model, key, src)
            else:
                _restore_quant(model, key, meta, views)
            pool.put(buf)  # Device copies finished; the reader may refill it.
            restored += 1
            if after_key is not None:
                after_key(key)
    finally:
        stop.set()
        reader.join(timeout=10)
        drop_file_cache(record.path)
    for key, tensor in record.resident.items():
        _copy_into(model, key, tensor)
        restored += 1
        if after_key is not None:
            after_key(key)
    return {"restored_keys": restored, "file_backed_gib": round(record.mapped_bytes / 2**30, 2)}
