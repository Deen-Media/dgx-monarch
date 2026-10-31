"""Report the actors' FP8 matmul setting, ``[worker_args] fp8_matrix_mult``.

Doctor and ``dgxm status`` show what cluster.toml asks of the actors. Neither
attaches a mesh or reads a running process, so neither sees the operations
class a build picked: the ``FP8 compute:`` line that every model build logs, in
the driver log and in each Worker journal, names that
(``src/dgx_monarch/fp8_compute.py``). The setting is never wrong in itself, so
the row never warns.
"""
from __future__ import annotations

from collections.abc import Mapping

from ..fp8_compute import AUTO, OFF, ON, WORKER_ARG, setting
from .probe_certainty import OK

NAME = "FP8 matmul (actors)"
ENTRY = "docs/TROUBLESHOOTING.md #141"
_MEANING = {
    AUTO: ("the actors pass ComfyUI's fp8_optimizations only where a loader's weight_dtype is "
           "fp8_e4m3fn_fast, as stock ComfyUI does without --fast"),
    ON: ("every model the actors build passes ComfyUI's fp8_optimizations, the actors' form of "
         "--fast fp8_matrix_mult, so plain FP8 checkpoints run torch._scaled_mm"),
    OFF: ("no model the actors build passes ComfyUI's fp8_optimizations, so a loader's "
          "fp8_e4m3fn_fast builds as fp8_e4m3fn and FP8 weights are dequantized for every matmul"),
}


def _worker_args(config: object) -> Mapping[str, object] | None:
    args = getattr(config, "worker_args", None)
    return args if isinstance(args, Mapping) else None


def doctor_row(config: object) -> tuple[str, str, str]:
    """One (status, name, detail) row for the actors, from doctor's config."""
    mode, source = setting(_worker_args(config))
    return (OK, NAME, f"{WORKER_ARG}={mode} from {source}: {_MEANING[mode]}. A checkpoint with "
                      "comfy-kitchen quantization metadata runs comfy-kitchen's kernels whatever this "
                      "says, and ComfyUI's own --fast flags reach only the driver's loads. Doctor reads "
                      "cluster.toml, not a running process: the `FP8 compute:` line each model build "
                      f"logs names the class ComfyUI picked ({ENTRY})")


def status_fields(config: object) -> dict[str, str]:
    """The setting and its source for the ``dgxm status --json`` payload."""
    mode, source = setting(_worker_args(config))
    return {"value": mode, "source": source}


def status_line(config: object) -> str | None:
    """The line ``dgxm status`` prints under the mesh line, or None for an unset key.

    A file that leaves the key out keeps status at its host rows and one mesh
    line; ``--json`` carries the setting either way.
    """
    args = _worker_args(config)
    if not args or WORKER_ARG not in args:
        return None
    mode, source = setting(args)
    return f"actors {WORKER_ARG}={mode} ({source}; `dgxm doctor` says what it does)"
