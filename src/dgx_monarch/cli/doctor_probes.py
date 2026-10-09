"""Read swap state and check SageAttention for ``dgxm doctor``.

The remote swap probe emits prefixed lines in doctor's per-host heredoc. The
Sage kernel probe checks the same ``nvidia-smi`` process list as the
co-resident GPU row and skips when another GPU process is observed.

The kernel probe is doctor's only CUDA workload. Doctor checks CUDA
availability and imports SageAttention without allocating tensors, then runs
the kernel in a bounded subprocess of the caller's interpreter. A stale-ABI
crash therefore terminates the subprocess rather than doctor.
"""
from __future__ import annotations

import os

from .probe_certainty import FAIL, OK, WARN

SAGE_NAME = "sageattention"
FALLBACK_NAME = "sageattention fallbacks"
TROUBLESHOOTING_ENTRY = "docs/TROUBLESHOOTING.md #89"
_REBUILD_HINT = f"rebuild sageattention against the installed torch; {TROUBLESHOOTING_ENTRY}"

KERNEL_PROBE_TIMEOUT_S = 20.0

# Run standalone as `python_bin -c KERNEL_PROBE_SOURCE`, never embedded in
# another f-string, so it needs no brace escaping. head_dim 128 catches a
# missing or shadowed kernel; head_dim 64 adds the head_dim-conditional path.
KERNEL_PROBE_SOURCE = """import torch
from sageattention import sageattn
for _hd in (128, 64):
    _q = torch.randn(1, 8, 1024, _hd, dtype=torch.float16, device="cuda")
    _o = sageattn(_q, _q, _q, tensor_layout="HND")
    torch.cuda.synchronize()
    assert _o.shape == _q.shape and torch.isfinite(_o).all()
print("SAGE_KERNEL_PROBE_OK")"""

# Doctor selects the remote swap line from the per-host script output by this prefix.
SWAP_LINE_PREFIX = "swap_kib="

# Substituted into doctor.py's per-host f-string, like sol_attn_row.PROBE_SOURCE.
# It is a plain string, so its own f-string braces reach the child as written.
SWAP_PROBE_SOURCE = """try:
    with open("/proc/swaps") as _swf:
        _swap_lines = _swf.read().splitlines()[1:]
except OSError:
    _swap_lines = []
_swap_kib = 0
_swap_paths = []
for _sl in _swap_lines:
    _sp = _sl.split()
    if len(_sp) >= 3:
        _swap_paths.append(_sp[0])
        try:
            _swap_kib += int(_sp[2])
        except ValueError:
            pass
print(f"swap_kib={_swap_kib} swap_paths={','.join(_swap_paths) or 'none'}")"""

DEFAULT_FALLBACK_LOG = "~/comfyui-monarch.log"
_FALLBACK_ERROR_LINE = "Error running sage attention"
_FALLBACK_BENIGN = "Unsupported head_dim"


def _read_proc_swaps() -> str:
    """The local box's `/proc/swaps`. No sudo; readable by any user."""
    with open("/proc/swaps") as f:
        return f.read()


def _parse_swap(content: str) -> tuple[float, list[str]]:
    """`(gib, paths)` over every active swap device in `/proc/swaps` text."""
    gib_total = 0
    paths: list[str] = []
    for line in content.splitlines()[1:]:  # header row first
        parts = line.split()
        if len(parts) < 3:
            continue
        paths.append(parts[0])
        try:
            gib_total += int(parts[2])
        except ValueError:
            continue
    return gib_total / (1024 * 1024), paths


def _swap_warn_detail(gib: float, paths: list[str]) -> str:
    return (f"{gib:.1f} GiB active on {', '.join(paths)}. A load near the memory limit "
            "pages weights through swap and can freeze the box; with swap off the same "
            "overrun is a clean OOM kill. See docs/INSTALL.md")


def swap_row() -> tuple[str, str, str]:
    """`(status, name, detail)` for the local `swap` row. Never FAIL: swap on
    is a risk, not a broken install."""
    try:
        content = _read_proc_swaps()
    except Exception as exc:
        return (WARN, "swap", f"unobserved: /proc/swaps unreadable ({exc!r})")
    gib, paths = _parse_swap(content)
    if not paths:
        return (OK, "swap", "off")
    return (WARN, "swap", _swap_warn_detail(gib, paths))


def remote_swap_row(host_name: str, line: str) -> tuple[str, str, str]:
    """`(status, name, detail)` for one worker host's swap state, parsed from
    the `swap_kib=... swap_paths=...` line its per-host probe prints."""
    name = f"{host_name} swap"
    if not line:
        return (WARN, name, "unobserved: the swap probe did not answer")
    fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
    try:
        kib = int(fields.get("swap_kib", "0"))
    except ValueError:
        kib = 0
    paths_field = fields.get("swap_paths", "none")
    paths = [] if paths_field in ("none", "") else paths_field.split(",")
    if not paths or kib <= 0:
        return (OK, name, "off")
    return (WARN, name, _swap_warn_detail(kib / (1024 * 1024), paths))


def gpu_busy() -> bool | None:
    """Return whether another GPU compute process is currently running.

    Use ``nvidia-smi --query-compute-apps``, as the co-resident GPU row does.
    Return ``None`` when the command is missing, times out or fails; a failed probe
    must not report the GPU as free.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def run_kernel_probe(python_bin: str):
    """Run the kernel probe in a subprocess; tests monkeypatch this function."""
    import subprocess

    return subprocess.run([python_bin, "-c", KERNEL_PROBE_SOURCE],
                          capture_output=True, text=True, timeout=KERNEL_PROBE_TIMEOUT_S)


def sage_kernel_row(python_bin: str) -> tuple[str, str, str]:
    """`(status, name, detail)` for the `sageattention` row.

    Import state first (cheap, no CUDA tensor). Only when a CUDA device is
    present, the package imports, and no co-resident GPU process is running
    does this dispatch the bounded subprocess kernel probe; a live render
    reads a WARN-free skip instead of pausing behind it.
    """
    try:
        import torch
    except Exception as exc:
        return (WARN, SAGE_NAME, f"probe unavailable ({exc!r})")
    try:
        if not torch.cuda.is_available():
            return (OK, SAGE_NAME, "skipped: no CUDA device on this host")
    except Exception as exc:
        return (WARN, SAGE_NAME, f"probe unavailable ({exc!r})")

    try:
        import sageattention  # noqa: F401
    except ImportError:
        return (OK, SAGE_NAME, "not installed (ComfyUI's default attention does not need it; "
                               "--use-sage-attention and the SAGE_* kernels do)")
    except Exception as exc:
        return (FAIL, SAGE_NAME, f"installed but broken on this torch ({exc!r}); {_REBUILD_HINT}")

    busy = gpu_busy()
    if busy is None:
        return (WARN, SAGE_NAME,
                "installed; kernel probe NOT RUN (the co-resident GPU process "
                "check did not answer, so a kernel run here could disturb a "
                "live render)")
    if busy:
        return (OK, SAGE_NAME, "skipped: GPU busy")

    try:
        result = run_kernel_probe(python_bin)
    except Exception as exc:
        timed_out = type(exc).__name__ == "TimeoutExpired"
        reason = f"timed out after {KERNEL_PROBE_TIMEOUT_S:.0f}s" if timed_out else repr(exc)
        return (FAIL, SAGE_NAME, f"kernel probe failed: {reason} ({_REBUILD_HINT})")
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "no output").strip().splitlines()
        reason = tail[-1] if tail else "no output"
        return (FAIL, SAGE_NAME, f"kernel probe failed: {reason} ({_REBUILD_HINT})")
    return (OK, SAGE_NAME, "kernel ran (head_dim 128 and 64)")


def sage_fallback_row(log_path: str = DEFAULT_FALLBACK_LOG) -> tuple[str, str, str]:
    """`(status, name, detail)` counting real sage-to-torch fallbacks in the driver
    log on this box only; no other host's log is read. It counts the whole file, so
    it resets when the log rotates."""
    expanded = os.path.expanduser(log_path)
    try:
        with open(expanded, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return (OK, FALLBACK_NAME,
                f"skipped: {log_path} is missing or unreadable, so this row cannot count fallbacks")
    real = sum(1 for line in lines
               if _FALLBACK_ERROR_LINE in line and _FALLBACK_BENIGN not in line)
    if real:
        return (WARN, FALLBACK_NAME,
                f"{real} real fallback(s) in {log_path} (sampling ran on torch "
                "attention; the sweep records the kernel per cell)")
    return (OK, FALLBACK_NAME, f"0 real fallback(s) in {log_path}")
