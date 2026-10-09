"""Minimal DLPack wrapping for slab-owned byte ranges."""
from __future__ import annotations

import ctypes
import threading

import torch


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


_KDL_CPU = 1
_KDL_CUDA = 2

# DLManagedTensor and its deleter must outlive every wrapped tensor: torch keeps
# the struct pointer for collection, even after slab closure on a failure path.
# Freeing it early can crash the collector. Torch copies the shape, pointer, and
# device at construction, so one process-lifetime struct with a fixed deleter
# serves all wraps. Set its fields under the lock before calling from_dlpack;
# this retains constant-size metadata rather than allocating it per load.
_NOOP_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: None)
_WRAP_LOCK = threading.Lock()
_SHARED_SHAPE = (ctypes.c_int64 * 1)(0)
_SHARED_MT = _DLManagedTensor()
_SHARED_MT.dl_tensor.ndim = 1
_SHARED_MT.dl_tensor.dtype = _DLDataType(1, 8, 1)
_SHARED_MT.dl_tensor.shape = _SHARED_SHAPE
_SHARED_MT.dl_tensor.strides = None
_SHARED_MT.dl_tensor.byte_offset = 0
_SHARED_MT.manager_ctx = None
_SHARED_MT.deleter = _NOOP_DELETER


def wrap_u8(ptr: int, nbytes: int, *, cuda: bool) -> torch.Tensor:
    """Wrap a slab-owned byte range as a flat, zero-copy uint8 tensor."""
    pycapi = ctypes.pythonapi
    pycapi.PyCapsule_New.restype = ctypes.py_object
    pycapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    with _WRAP_LOCK:
        _SHARED_SHAPE[0] = nbytes
        _SHARED_MT.dl_tensor.data = ptr
        _SHARED_MT.dl_tensor.device = _DLDevice(
            _KDL_CUDA if cuda else _KDL_CPU,
            0,
        )
        capsule = pycapi.PyCapsule_New(
            ctypes.byref(_SHARED_MT),
            b"dltensor",
            None,
        )
        return torch.utils.dlpack.from_dlpack(capsule)
