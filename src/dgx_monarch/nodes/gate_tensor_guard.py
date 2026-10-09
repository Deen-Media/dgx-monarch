"""Bounded full-content fingerprints for Identity Gate transaction tensors."""
from __future__ import annotations

import hashlib
from typing import Any, cast

_HASH_CHUNK_BYTES = 4 * 1024 * 1024


def _update_storage_fingerprint(digest: Any, tensor: Any) -> None:
    """Hash one tensor storage with bounded host scratch memory."""
    import torch

    storage = tensor.untyped_storage()
    nbytes = int(storage.nbytes())
    # StorageImpl identity closes the zero-numel case where two independent
    # storages both expose data_ptr=0 and nbytes=0 after ``tensor.data = ...``.
    digest.update(
        repr((int(storage._cdata), int(storage.data_ptr()), nbytes)).encode()
    )
    raw = torch.empty(
        0, dtype=torch.uint8, device=tensor.device
    ).set_(storage, 0, (nbytes,), (1,))
    for start in range(0, nbytes, _HASH_CHUNK_BYTES):
        cpu = raw[start:start + _HASH_CHUNK_BYTES].to("cpu")
        digest.update(memoryview(cast(Any, cpu.contiguous().numpy())))


def _update_tensor_fingerprint(digest: Any, tensor: Any) -> None:
    """Hash observable metadata and backing bytes for one tensor."""
    import torch

    layout = str(tensor.layout)
    stride = tuple(tensor.stride()) if tensor.layout == torch.strided else None
    offset = int(tensor.storage_offset()) if tensor.layout == torch.strided else None
    metadata = (
        type(tensor).__module__,
        type(tensor).__qualname__,
        tuple(tensor.shape),
        stride,
        offset,
        str(tensor.dtype),
        layout,
        str(tensor.device),
        bool(tensor.requires_grad),
        bool(tensor.is_conj()),
        bool(tensor.is_neg()),
    )
    digest.update(repr(metadata).encode())
    if tensor.device.type == "meta":
        return
    if tensor.layout == torch.sparse_coo:
        digest.update(b"sparse-coo")
        _update_tensor_fingerprint(digest, tensor._indices())
        _update_tensor_fingerprint(digest, tensor._values())
        return
    compressed_parts = {
        "torch.sparse_csr": ("crow_indices", "col_indices"),
        "torch.sparse_bsr": ("crow_indices", "col_indices"),
        "torch.sparse_csc": ("ccol_indices", "row_indices"),
        "torch.sparse_bsc": ("ccol_indices", "row_indices"),
    }.get(layout)
    if compressed_parts is not None:
        digest.update(layout.encode())
        for accessor in (*compressed_parts, "values"):
            _update_tensor_fingerprint(digest, getattr(tensor, accessor)())
        return
    _update_storage_fingerprint(digest, tensor)


def transaction_tensor_fingerprint(tensor: Any) -> bytes:
    """Return a full-content digest without another tensor-sized allocation."""
    import torch

    if type(tensor) is not torch.Tensor:
        raise TypeError(
            "identity-gate fingerprints require exact torch.Tensor values"
        )
    digest = hashlib.sha256()
    try:
        _update_tensor_fingerprint(digest, tensor)
    except Exception as exc:
        raise RuntimeError(
            "identity-gate could not fingerprint a captured request/latent tensor"
        ) from exc
    return digest.digest()
