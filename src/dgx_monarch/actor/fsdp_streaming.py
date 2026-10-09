"""Build FSDP shards from file-backed parameters without a full anonymous copy.

ComfyUI's copying loader materializes the entire checkpoint before sharding.
This window uses ``load_model_weights(..., assign=True)`` to retain mmap-backed
parameters, then ``adapters/fsdp_shard_build.py`` moves only this rank's rows
to the device. The streaming high-water is the local shards plus one chunk.
Quantized files use FSDP2's direct wrap, which stages each full block before
chunking it and therefore retains that larger transient. The capacity model
still charges full materialization when world size is unknown or the file is
neither bf16 nor fp16.

Assignment preserves file dtypes, whereas copying preserves constructed
dtypes. Recast only parameters and buffers whose dtypes differ so stock fp32
islands, such as Wan's input convolution, retain the copying loader's math.
The bulk of the model stays file-backed until shard construction finishes;
``release_materialize_window`` then releases the mapping's pages.
"""
from __future__ import annotations

import contextlib
import gc
import itertools
import os
from typing import Any

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)


def constructed_dtypes(module: Any) -> dict[str, Any]:
    """{qualified name: dtype} for every parameter and buffer of ``module``."""
    named = itertools.chain(module.named_parameters(), module.named_buffers())
    return {name: tensor.dtype for name, tensor in named}


def recast_to_constructed(module: Any, constructed: dict[str, Any]) -> tuple[int, int]:
    """Give every assigned tensor the dtype the model constructed it with.

    Returns (tensor count, bytes) of what was recast. Tensors the model did not
    construct (file keys with no constructed tensor of that name) keep the
    file's dtype, as they would under the copying load.
    """
    count = 0
    nbytes = 0
    for name, parameter in list(module.named_parameters()):
        want = constructed.get(name)
        if want is None or parameter.dtype == want:
            continue
        parameter.data = parameter.data.to(want)
        count += 1
        nbytes += parameter.numel() * parameter.element_size()
    for name, buffer in list(module.named_buffers()):
        want = constructed.get(name)
        if want is None or buffer.dtype == want:
            continue
        owner_name, _, attr = name.rpartition(".")
        owner = module.get_submodule(owner_name) if owner_name else module
        owner._buffers[attr] = buffer.to(want)
        count += 1
        nbytes += buffer.numel() * buffer.element_size()
    return count, nbytes


@contextlib.contextmanager
def assign_load_window():
    """Make Comfy assign file tensors as parameters for one FSDP model load."""
    try:
        import comfy.model_base as model_base
    except ImportError as exc:
        # Only a stand-in comfy lacks this module (the real loader imports it
        # before any model load); there is nothing to stream through then.
        safe_call(log.warning, "FSDP streaming build unavailable: %s",
                  failure_summary(exc))
        yield
        return

    original = model_base.BaseModel.load_model_weights

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        diffusion_model = self.diffusion_model
        constructed = constructed_dtypes(diffusion_model)
        result = original(self, sd, unet_prefix, assign=True)
        count, nbytes = recast_to_constructed(diffusion_model, constructed)
        log.info(
            "FSDP streaming build: assigned the checkpoint's tensors as "
            "parameters (file-backed, nothing committed); recast %d tensor(s) "
            "(%.3f GiB) to their constructed dtype",
            count, nbytes / 2**30,
        )
        return result

    model_base.BaseModel.load_model_weights = load_model_weights
    try:
        yield
    finally:
        model_base.BaseModel.load_model_weights = original


def release_materialize_window(model: Any, path: str) -> int:
    """Release checkpoint mappings after sharding and before ComfyUI loads.

    Replicated islands, scalar parameters, auxiliary modules, and unchanged-
    dtype buffers can still retain the mapping. Clone these into anonymous
    memory, collect the state dict and load closures, release any remaining
    clean mapped pages with ``madvise``, then evict page cache with
    ``posix_fadvise``. Eviction must run last because it cannot evict mapped
    pages. Return the bytes mapped and resident at entry.

    Measure the files backing the tensors plus the resolved checkpoint path:
    ``/proc/self/fd`` aliases do not match the real paths in kernel mappings.

    Count anonymous mapped pages before copying can remove the mapping. They
    indicate a device copy bypassed the bounce buffer; preserve and log them
    because their contents are not proven unmodified.
    """
    from . import mapped_pages

    try:
        target = getattr(model, "model", model)
        paths = mapped_pages.backing_paths(
            itertools.chain(target.parameters(), target.buffers()))
        paths |= {os.path.realpath(path)}
        held = mapped_pages.resident(paths)
        before = sum(rss for _mapping, rss, _anonymous in held)
        anonymous = sum(anon for _mapping, _rss, anon in held)
        count, copied, detached = mapped_pages.detach_from_files(target)
        gc.collect()
        paths |= detached
        rows = mapped_pages.resident(paths)
        residual = sum(rss for _mapping, rss, _anonymous in rows)
        dropped = mapped_pages.drop_mapped_pages(paths) if residual else 0
        mapped_pages.evict_page_cache(paths)
        log.info(
            "FSDP load: checkpoint mapping released, %.1f GiB was mapped and "
            "resident; detached %d file-backed tensor(s) (%.3f GiB copied into "
            "anonymous memory), %.1f GiB still mapped afterwards, %.1f GiB of it "
            "given back, %.3f GiB copy-on-write",
            before / 2**30, count, copied / 2**30, residual / 2**30,
            dropped / 2**30, anonymous / 2**30)
        if anonymous:
            log.warning(
                "FSDP load: %.3f GiB of the checkpoint mapping is anonymous: this load "
                "bypassed the bounce copy in fsdp_shard_build.py, so rows were copied to "
                "the device straight from the mmap and the CUDA driver broke copy-on-write "
                "(docs/VALIDATION.md, 2026-09-07; issue #406)", anonymous / 2**30)
        return before
    except Exception as exc:  # never turn an accounting step into a failed load
        safe_call(log.warning, "checkpoint mapping release skipped for %s (%s)",
                  path, failure_summary(exc))
        return 0
