"""The Init node's `family_adapter` override, read on the driver side.

Two independent decisions take a family, and both follow this widget:

* the driver reads the checkpoint's safetensors header to pick an AUTO_TABLE
  row and to arm the family-keyed preflights (``adapters/detect.py``);
* the worker matches the model object ComfyUI built to an adapter
  (``adapters.adapter_for``).

The header sniff is the half that misreads a finetune. It needs every listed
key path to be present, so a derivative that renamed or dropped one falls
through to a looser signature rather than to "unknown", and an export under an
unrecognized top-level prefix reads "unknown" and loses its table row and its
preflights. Naming the family restores both.

`auto` emits no worker-args key, so an unforced graph keeps its worker args,
its gate capability context and its existing PASS rows.
"""
from __future__ import annotations

import threading
from typing import Any

from .log import get_logger

log = get_logger(__name__)

AUTO = "auto"

# Init widget -> worker args. Also the gate capability context key, because
# worker args are that context: a forced run can never share a ledger row with
# an auto run of the same file.
WORKER_ARG_KEY = "family_override"

# LTX carries its model config in the safetensors header and ComfyUI's
# constructor defaults disagree with it, so a file that lost the header builds
# a structurally different model and still accepts most of the weights
# (docs/MODELS.md). Naming the family cannot put the config back.
_CONFIG_METADATA_KEY = "config"
_CONFIG_REQUIRED_FAMILIES = frozenset({"ltx"})

_auto_warned: set[str] = set()  # proc lifetime, keyed by path, never evicted
_auto_warned_lock = threading.Lock()


def _first_auto_warn(path: str) -> bool:
    """True exactly once per file: locked test-and-add, so overlapping renders
    cannot race the membership check."""
    with _auto_warned_lock:
        if path in _auto_warned:
            return False
        _auto_warned.add(path)
        return True


def override_from_worker_args(worker_args: Any) -> str | None:
    """The forced family, or None when the graph left `family_adapter` on auto."""
    if not isinstance(worker_args, dict):
        return None
    value = worker_args.get(WORKER_ARG_KEY)
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value == AUTO:
        return None
    return value


def override_for_model(model: Any) -> str | None:
    """The forced family reached through a ModelSpec's mesh."""
    mesh = getattr(model, "mesh", None)
    return override_from_worker_args(getattr(mesh, "worker_args", None))


def override_for_worker(worker: Any) -> str | None:
    """The forced family a worker's model store carries, or None for auto.

    Read defensively because adapter binding is also exercised by probes and
    unit tests that stand a bare namespace in for the worker; those carry no
    store and belong on the auto path.
    """
    return getattr(getattr(worker, "store", None), "family_override", None)


def effective_family(sniffed: str, override: str | None) -> str:
    """The family the driver should act on: the override, else the sniff.

    Quantization is not part of this: the override names an architecture, and
    a nonstandard quant export still misreads its precision. For that, use an
    explicit topology preset (docs/MODELS.md).
    """
    return override or sniffed


def assert_override_admits_checkpoint(family: str, path: str) -> None:
    """Refuse a forced family the file cannot carry.

    Override path only; auto warns instead (warn_auto_admits_checkpoint). Only
    LTX is in _CONFIG_REQUIRED_FAMILIES, and the fault is the file, not the
    name: the adapter the override names is right, but the model ComfyUI
    builds has the wrong shape. No waiver can supply a missing config, so this
    is class P and the fix is a re-export.
    """
    if family not in _CONFIG_REQUIRED_FAMILIES:
        return
    if not path.lower().endswith((".safetensors", ".sft")):
        return  # no header to read; the loader's own errors remain the report
    from .safetensors_header import SafetensorsHeaderError, read_safetensors_header

    try:
        metadata = read_safetensors_header(path).metadata
    except (SafetensorsHeaderError, OSError):
        return  # unreadable header: leave the load path's own report in place
    if _CONFIG_METADATA_KEY in metadata:
        return
    from .adapters.base import UnsupportedModelError
    from .refusal import RefusalClass, refusal

    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"family_adapter={family!r} cannot admit this checkpoint: its safetensors "
        f"header carries no {_CONFIG_METADATA_KEY!r} metadata. An {family} file stores "
        "its model config in that header, and ComfyUI's constructor defaults disagree "
        "with it, so a file without it builds a structurally different model that still "
        "accepts most of the weights and renders wrong output with no error. Naming the "
        "family does not fix it: that binds the right adapter to the wrong architecture. "
        "Leaving family_adapter on 'auto' does not fix it either. Use an export that keeps "
        "the header metadata instead (docs/MODELS.md).",
    ))


def warn_auto_admits_checkpoint(family: str, path: str) -> None:
    """Say once that a configless checkpoint of a config-carrying family loaded.

    The override path refuses this (a named family is a claim the file cannot
    honor). Auto only knows the key is missing, which is not proof the built
    architecture is wrong, so this reports and lets the render run.
    """
    if family not in _CONFIG_REQUIRED_FAMILIES:
        return
    if not path.lower().endswith((".safetensors", ".sft")):
        return  # no header to read; the loader's own errors remain the report
    from .adapters.detect import CheckpointSniffError, sniff_metadata_keys

    try:
        metadata_keys = sniff_metadata_keys(path)
    except (CheckpointSniffError, OSError):
        return  # unreadable header: leave the load path's own report in place
    if _CONFIG_METADATA_KEY in metadata_keys:
        return
    if not _first_auto_warn(path):
        return
    log.warning(
        "This checkpoint's tensor keys identify it as %s, but its safetensors header "
        "carries no %r metadata. ComfyUI reads that key to size an %s model, so this "
        "load falls back to constructor defaults and can finish without an error "
        "while the architecture is not the file's. Not refused here: auto mode knows "
        "only that the key is missing, not that the built model is wrong. Re-export "
        "with the header metadata kept (docs/MODELS.md).",
        family, _CONFIG_METADATA_KEY, family)
