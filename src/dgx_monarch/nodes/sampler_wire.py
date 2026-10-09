"""Wire-side helpers the sampler nodes call: output merge and noise rebuild."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path

from ..accuracy_waiver import STAMPED_RESULT_KEY
from .latent_outputs import concatenate_latent_batches, is_direct_nested_tensor


def _aggregate_render_outputs(outputs: list[dict], path: str) -> dict:
    """Combine sampler results without collapsing batch or waiver metadata.

    Absent ``batch_index`` stays absent; otherwise indices concatenate beside
    samples. Waiver provenance stays flat and deduplicates by run and guard.
    """
    if not outputs:
        raise RuntimeError(f"{path} returned no latent batches")

    combined = dict(outputs[0])
    combined["samples"] = concatenate_latent_batches(
        [output["samples"] for output in outputs], path)
    indexed = ["batch_index" in output for output in outputs]
    if any(indexed) and not all(indexed):
        raise RuntimeError(f"{path} inconsistently returned batch_index metadata")
    if all(indexed):
        batch_index = []
        for index, output in enumerate(outputs):
            value = output["batch_index"]
            if not isinstance(value, list):
                raise RuntimeError(f"{path}[{index}] batch_index must be a list")
            if any(type(item) is not int or item < 0 for item in value):
                raise RuntimeError(
                    f"{path}[{index}] batch_index entries must be nonnegative integers")
            samples = output["samples"]
            if is_direct_nested_tensor(samples, f"{path}[{index}] samples"):
                samples = samples.unbind()[0]
            sample_batch = int(samples.shape[0])
            if len(value) != sample_batch:
                raise RuntimeError(
                    f"{path}[{index}] batch_index length {len(value)} does not "
                    f"match samples batch {sample_batch}")
            batch_index.extend(value)
        combined["batch_index"] = batch_index
    else:
        combined.pop("batch_index", None)

    provenance = []
    seen_provenance = set()
    for output_index, output in enumerate(outputs):
        entries = output[STAMPED_RESULT_KEY] if STAMPED_RESULT_KEY in output else ()
        if type(entries) not in (list, tuple):
            raise RuntimeError(
                f"{path}[{output_index}] waiver provenance must be a list or tuple")
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise RuntimeError(
                    f"{path}[{output_index}] waiver provenance entry "
                    f"{entry_index} must be a mapping"
                )
            run_id, guard = entry.get("run_id"), entry.get("guard")
            if not all(isinstance(value, str) and value for value in (run_id, guard)):
                raise RuntimeError(
                    f"{path}[{output_index}] waiver provenance entry {entry_index} must "
                    "carry nonempty string 'run_id' and 'guard'")
            identity = (run_id, guard)
            if identity in seen_provenance:
                continue
            seen_provenance.add(identity)
            provenance.append(dict(entry))
    if provenance:
        combined[STAMPED_RESULT_KEY] = provenance
    else:
        combined.pop(STAMPED_RESULT_KEY, None)
    return combined


def _module_source_path(module):
    try:
        return Path(module.__file__).resolve(strict=False)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _canonical_noise_for_wire(noise):
    """Rebuild stock extra-node noise under its importable canonical module.

    Comfy's extra-node loader executes ``nodes_custom_sampler.py`` under an
    absolute-path module key. Cloudpickle records that key for RandomNoise and
    DisableNoise outputs, and workers cannot import it. Rebuild only on an exact
    stock class name and source file match; any other noise object still
    pickles by reference.
    """
    cls = type(noise)
    if cls.__name__ not in ("Noise_RandomNoise", "Noise_EmptyNoise"):
        return noise
    source_module = sys.modules.get(cls.__module__)
    if source_module is None:
        return noise
    canonical = importlib.import_module("comfy_extras.nodes_custom_sampler")
    if _module_source_path(source_module) != _module_source_path(canonical):
        return noise
    canonical_cls = getattr(canonical, cls.__name__)
    if cls is canonical_cls:
        return noise
    if cls.__name__ == "Noise_RandomNoise":
        return canonical_cls(int(noise.seed))
    return canonical_cls()
