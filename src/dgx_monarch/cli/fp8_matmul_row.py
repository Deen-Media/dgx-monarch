"""Report the actors' FP8 matmul setting, ``[worker_args] fp8_matrix_mult``.

Doctor and ``dgxm status`` show what cluster.toml asks of the actors. Neither
attaches a mesh or reads a running process, so neither sees the operations
class a build picked: the ``FP8 compute:`` line that every model build logs, in
the driver log and in each Worker journal, names that
(``src/dgx_monarch/fp8_compute.py``). The row warns in one case only, the one
MODELS.md documents as unsafe: ``"on"`` where a host reports a GB10 GPU,
compute capability 12.1, read from ``nvidia-smi`` by doctor's per-host probe.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping

from ..fp8_compute import AUTO, OFF, ON, WORKER_ARG, setting
from .probe_certainty import OK, WARN

NAME = "FP8 matmul (actors)"
ENTRY = "docs/TROUBLESHOOTING.md #141"
MODELS_ENTRY = "docs/MODELS.md#fp8-on-gb10"
GB10_CAPABILITY = "12.1"
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


def local_capability() -> str:
    """This box's GPU compute capability as ``nvidia-smi`` prints it, or ``unknown``."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15)
    except Exception:
        return "unknown"
    fields = result.stdout.split() if result.returncode == 0 else []
    return fields[0] if fields else "unknown"


def _known(capability: str) -> bool:
    parts = capability.split(".")
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def doctor_row(
    config: object, caps: Mapping[str, str] | None = None, *,
    probe: Callable[[], str] = local_capability,
) -> tuple[str, str, str]:
    """One (status, name, detail) row for the actors, from doctor's config.

    ``caps`` maps each host to the compute capability its probe reported. With
    none, as in local mode, the row reads this box, and only for ``"on"``.
    """
    mode, source = setting(_worker_args(config))
    plain = (OK, NAME, f"{WORKER_ARG}={mode} from {source}: {_MEANING[mode]}. A checkpoint with "
                       "comfy-kitchen quantization metadata runs comfy-kitchen's kernels whatever this "
                       "says, and ComfyUI's own --fast flags reach only the driver's loads. Doctor reads "
                       "cluster.toml, not a running process: the `FP8 compute:` line each model build "
                       f"logs names the class ComfyUI picked ({ENTRY})")
    if mode != ON:
        return plain
    caps = {"this box": probe()} if caps is None else caps
    gb10 = sorted(host for host, cap in caps.items() if cap == GB10_CAPABILITY)
    if gb10:
        return (WARN, NAME, f"{WORKER_ARG}=on from {source}, and a GPU is GB10 (compute capability "
                            f"12.1) on {', '.join(gb10)}. There every plain FP8 checkpoint the actors "
                            "build takes fp8_ops, which MODELS.md documents as unsafe on GB10: "
                            "Qwen-Image 2512's plain FP8 file rendered noise, though HunyuanImage 2.1's "
                            "stayed inside the floor, so not every plain FP8 file is affected. A checkpoint "
                            "with comfy-kitchen quantization metadata and a BF16 checkpoint compute as "
                            f"before. Leave the key out for auto ({MODELS_ENTRY})")
    blind = sorted(host for host, cap in caps.items() if not _known(cap))
    if blind or not caps:
        return (WARN, NAME, f"unobserved: {WORKER_ARG}=on from {source}, and "
                            f"{', '.join(blind) or 'no host'} reported no GPU compute capability, so "
                            "this row cannot say whether the setting meets the case MODELS.md documents "
                            f"as unsafe on GB10 ({MODELS_ENTRY})")
    return plain


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
