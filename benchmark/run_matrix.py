#!/usr/bin/env python3
"""Run benchmark cases through DGX Monarch nodes without a ComfyUI server.

Encode prompts on the driver with stock ComfyUI, render through the cluster,
and record timings, quality gates, and provenance for each case.

Run in a ComfyUI environment with Worker services available. Copy and edit
matrix.example.toml first:

    python benchmark/run_matrix.py --matrix benchmark/matrix.toml \
        --config cluster.toml [--out results.json]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import ntpath
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

COMFY_DIR = os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))
# comfy.cli_args parses argv at import time: capture this script's args now, but
# do not mutate process-global argv until main() has parsed them, so unit tests
# and other tooling can import the provenance helpers.
CLI_ARGS = sys.argv[1:]


def _git_commit(directory: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def _git_provenance(directory: str | Path) -> dict:
    """Commit plus working-tree state; a commit alone is not reproducible."""
    provenance = {"commit": _git_commit(directory), "dirty": "unknown", "changes": []}
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return provenance
    if result.returncode != 0:
        return provenance
    changes = [line for line in result.stdout.splitlines() if line]
    provenance.update({"dirty": bool(changes), "changes": changes})
    return provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_provenance(path_value: str) -> dict:
    path = Path(path_value).expanduser().resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        # Preserve comments, ordering, and every byte-significant setting. A
        # hash proves identity; embedded TOML remains useful if the file moves.
        "content": path.read_bytes().decode("utf-8"),
    }


def _config_provenance(explicit: str) -> dict:
    try:
        from dgx_monarch.config import find_config_path

        path = find_config_path(explicit or None)
        if path is None:
            return {"path": "unknown", "size_bytes": "unknown", "sha256": "unknown"}
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    except Exception:
        return {"path": "unknown", "size_bytes": "unknown", "sha256": "unknown"}


def _artifact_provenance(name: str, category: str) -> dict:
    result = {
        "name": name,
        "path": "unknown",
        "size_bytes": "unknown",
        "bounded_signature": "unknown",
        "signature_algorithm": "sha256(size + start/middle/end 1MiB), first 24 hex",
    }
    try:
        import folder_paths

        path_value = folder_paths.get_full_path(category, name)
        if not path_value:
            return result
        path = Path(path_value).resolve()
        from dgx_monarch.gate_ledger import artifact_signature

        result.update({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "bounded_signature": artifact_signature(str(path)),
        })
    except Exception:
        pass
    return result


def _case_artifact_provenance(cases: list[dict], defaults: dict) -> dict:
    artifacts = {}
    for case in cases:
        merged = {**defaults, **case}
        name = merged.get("name", "unknown")
        unet = merged.get("unet")
        row = {
            "unet": (
                _artifact_provenance(str(unet), "diffusion_models")
                if unet
                else {"name": "unknown", "bounded_signature": "unknown"}
            ),
        }
        te = merged.get("te")
        if isinstance(te, dict) and te.get("name"):
            te_names = te["name"]
            if isinstance(te_names, list):
                row["text_encoder"] = [
                    _artifact_provenance(str(name), "text_encoders")
                    for name in te_names
                ]
            else:
                # One encoder keeps the scalar shape, so readers of
                # ``text_encoder.path`` keep working; a list appears only when
                # the matrix names several encoders.
                row["text_encoder"] = _artifact_provenance(
                    str(te_names), "text_encoders"
                )
        loras = []
        for entry in merged.get("loras", []):
            lora_name = entry.get("name") if isinstance(entry, dict) else entry
            if lora_name:
                loras.append(_artifact_provenance(str(lora_name), "loras"))
        if loras:
            row["loras"] = loras
        artifacts[name] = row
    return artifacts


def _json_safe(value):
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return str(value)


def _case_settings(case: dict, defaults: dict) -> dict:
    """The complete requested cell, not a lossy hand-picked field subset."""
    return _json_safe({**defaults, **case})


def _sampling_path(settings: dict) -> str:
    """Select the runner route; ksampler is the default."""
    path = settings.get("sampling_path", "ksampler")
    if not isinstance(path, str) or path not in {"ksampler", "flux2"}:
        raise ValueError("sampling_path must be 'ksampler' or 'flux2'")
    return path


def _route_sampling(settings: dict, *, ksampler, flux2):
    """Invoke only the selected sampling route (dependency-injected for tests)."""
    return {"ksampler": ksampler, "flux2": flux2}[_sampling_path(settings)]()


def _init_options(settings: dict) -> dict:
    """Matrix-facing DGXMonarchInit options and their governed defaults."""
    governed = {
        "auto_gate": (("first_use", "off"), "first_use"),
        "lora_low_rss": (("auto", "on", "off"), "auto"),
        "slab_weights": (("auto", "on", "off"), "auto"),
    }
    resolved = {}
    for name, (choices, default) in governed.items():
        value = settings.get(name, default)
        if not isinstance(value, str) or value not in choices:
            allowed = ", ".join(repr(choice) for choice in choices)
            raise ValueError(f"{name} must be one of: {allowed}")
        resolved[name] = value
    return resolved


def _model_sampling_sd3_shift(settings: dict) -> float | None:
    """Validate the optional matrix-facing render-time SD3 sampling shift."""
    field = "model_sampling_sd3_shift"
    if field not in settings:
        return None
    value = settings[field]
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 100.0
    ):
        raise ValueError(f"{field} must be a finite number from 0 through 100")
    return float(value)


def _apply_model_sampling_sd3(model, settings: dict, *, patch=None):
    """Apply the optional node-equivalent SD3 patch before topology binding."""
    shift = _model_sampling_sd3_shift(settings)
    if shift is None:
        return model
    if patch is None:
        from dgx_monarch.nodes.model_sampling import DGXMonarchModelSamplingSD3

        patch = DGXMonarchModelSamplingSD3().patch
    return patch(model, shift)[0]


def _integer_setting(
    settings: dict, name: str, default: int, *, minimum: int
) -> int:
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        requirement = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {requirement} integer")
    return value


def _latent_downscale(settings: dict) -> int:
    return _integer_setting(settings, "latent_downscale", 8, minimum=1)


def _latent_shape(settings: dict, batch: int) -> tuple[int, ...]:
    """Resolve an image or video empty-latent shape from matrix settings."""
    channels = _integer_setting(settings, "latent_channels", 4, minimum=1)
    downscale = _latent_downscale(settings)
    temporal = _integer_setting(settings, "latent_t", 0, minimum=0)
    spatial = (settings["height"] // downscale, settings["width"] // downscale)
    if temporal:
        return (batch, channels, temporal, *spatial)
    return (batch, channels, *spatial)


def _benchmark_latent(
    settings: dict,
    batch: int,
    *,
    zeros,
    metadata_key: str,
) -> tuple[dict, int]:
    """Build the latent and retain its pixel scale for auto-topology policy."""
    downscale = _latent_downscale(settings)
    latent = {
        "samples": zeros(_latent_shape(settings, batch)),
        metadata_key: downscale,
    }
    return latent, downscale


def _latent_sha256(samples) -> str:
    """SHA-256 over the tensor's logical, contiguous raw bytes."""
    import torch

    contiguous = samples.detach().to(device="cpu").contiguous()
    byte_view = contiguous.view(torch.uint8)
    return hashlib.sha256(memoryview(byte_view.numpy())).hexdigest()


def _flux2_schedule(get_schedule, *, steps: int, width: int, height: int):
    """Call the public Flux2 scheduler surface and unwrap its first output."""
    return get_schedule(steps=int(steps), width=int(width), height=int(height))[0]


def _topology_provenance(topology, *, sage: bool, reason: str, attention: str) -> dict:
    return {
        "name": topology.describe(),
        "ulysses": int(topology.ulysses),
        "ring": int(topology.ring),
        "cfg": int(topology.cfg),
        "dp": int(topology.dp),
        "fsdp": bool(topology.fsdp),
        "world": int(topology.world),
        "attention": "SAGE_AUTO" if sage else attention,
        "reason": reason,
    }


def _gpu_hardware() -> list[str] | str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return rows if result.returncode == 0 and rows else "unknown"


def _gpu_uuids(gpu_hardware: object) -> list[str]:
    """The raw uuid field (index 1) of each nvidia-smi CSV row: the exact
    strings sanitize_report must redact."""
    if not isinstance(gpu_hardware, list):
        return []
    uuids = []
    for row in gpu_hardware:
        if not isinstance(row, str):
            continue
        fields = [f.strip() for f in row.split(",")]
        if len(fields) >= 2 and fields[1]:
            uuids.append(fields[1])
    return uuids


def _redact_gpu_hardware(value: object) -> object:
    """nvidia-smi rows are 'name, uuid, memory.total'; keep the generic
    hardware name and memory size, redact only the uuid (field index 1)."""
    if not isinstance(value, list):
        return value
    rows = []
    for row in value:
        if not isinstance(row, str):
            rows.append(row)
            continue
        fields = [f.strip() for f in row.split(",")]
        if len(fields) >= 2:
            fields[1] = "[UUID redacted]"
        rows.append(", ".join(fields))
    return rows


_PATH_FIELDS = frozenset({"path", "comfy_dir", "config", "config_path"})
_EMBEDDED_PATH_PATTERNS = (
    re.compile(r'''(?<![A-Za-z0-9_.:/-])/(?!/)[^\s"'`<>{}\[\](),;:]+'''),
    re.compile(r'''(?<![A-Za-z0-9_.-])[A-Za-z]:\\[^\s"'`<>{}\[\](),;]+'''),
    re.compile(r'''(?<![A-Za-z0-9_.-])~[/\\][^\s"'`<>{}\[\](),;]+'''),
)


def _is_path_field(key: str) -> bool:
    return key in _PATH_FIELDS or key.endswith(("_path", "_dir"))


def _redacted_path(value: str) -> str:
    """Keep only the non-private leaf of a POSIX, Windows, or ~/ path."""
    stripped = value.rstrip("/\\")
    if not stripped:
        return "[path redacted]"
    leaf = ntpath.basename(stripped) if "\\" in stripped else os.path.basename(stripped)
    return leaf or "[path redacted]"


def _looks_private_path(value: str, *, path_field: bool) -> bool:
    if not value or value == "unknown":
        return False
    return (
        path_field
        or os.path.isabs(value)
        or ntpath.isabs(value)
        or value.startswith(("~/", "~\\"))
    ) and _redacted_path(value) != value


def _embedded_private_paths(value: str) -> list[str]:
    found = []
    for pattern in _EMBEDDED_PATH_PATTERNS:
        for match in pattern.finditer(value):
            candidate = match.group(0).rstrip(".")
            if candidate and _redacted_path(candidate) != candidate:
                found.append(candidate)
    return found


def private_path_secrets(value: object, *, _key: str = "") -> list[str]:
    """Collect path values whose private prefix sanitization must remove.

    The key-aware cases cover relative config paths as well as the absolute
    paths discovered generically. Returned values are the original strings so
    the final serialized report can be checked for exact leaks before a file is
    opened for writing.
    """
    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            found.extend(private_path_secrets(sub, _key=str(key)))
    elif isinstance(value, list | tuple):
        for item in value:
            found.extend(private_path_secrets(item, _key=_key))
    elif isinstance(value, str):
        if _looks_private_path(value, path_field=_is_path_field(_key)):
            found.append(value)
        found.extend(_embedded_private_paths(value))
    # Longest first prevents a parent path from partially masking a child
    # before the child is replaced in free-form diagnostic strings.
    return sorted(set(found), key=lambda item: (-len(item), item))


def _redact_value(value: object, *, path_secrets: tuple[str, ...] = ()) -> object:
    if isinstance(value, dict):
        out = {}
        for key, sub in value.items():
            if key == "host" and isinstance(sub, str):
                out[key] = "[hostname redacted]"
            elif _is_path_field(str(key)) and isinstance(sub, str):
                out[key] = _redacted_path(sub)
            elif key == "gpu_hardware":
                out[key] = _redact_gpu_hardware(sub)
            else:
                out[key] = _redact_value(sub, path_secrets=path_secrets)
        return out
    if isinstance(value, list):
        return [_redact_value(item, path_secrets=path_secrets) for item in value]
    if isinstance(value, str):
        # A path can be duplicated inside a diagnostic or embedded matrix
        # string under a non-path key. Redact every already-discovered private
        # path there too, rather than trusting only the field that introduced it.
        for secret in path_secrets:
            value = value.replace(secret, _redacted_path(secret))
        if _looks_private_path(value, path_field=False):
            return _redacted_path(value)
    return value


def sanitize_report(payload: dict) -> dict:
    """Deep-copy with hostname, GPU uuid, and private path values redacted.

    This is applied automatically before any write to a report path that is not
    gitignored, unless --no-redact overrides it. The posture is canonical in
    docs/THREAT_MODEL.md. Never mutates the input.
    """
    secrets = tuple(private_path_secrets(payload))
    redacted = _redact_value(copy.deepcopy(payload), path_secrets=secrets)
    redacted["sanitization"] = (
        "auto-redacted (host/uuid/private paths) before non-gitignored write"
    )
    return redacted


def leaked_secrets(serialized: str, secrets: list[str]) -> list[str]:
    """Which of the given real (pre-redaction) secret strings still appear
    verbatim in serialized report text. This leak assertion backs
    sanitize_report. A redaction miss must fail before the destination is
    opened, not ship quietly or briefly touch a commit-reachable path."""
    return [s for s in secrets if s and s in serialized]


class ReportRedactionError(RuntimeError):
    def __init__(self, count: int) -> None:
        super().__init__(f"{count} forbidden secret value(s) remain after redaction")
        self.count = count


def _serialize_report(payload: dict, forbidden_secrets: list[str]) -> str:
    try:
        serialized = json.dumps(payload, indent=2, allow_nan=False)
    except ValueError as exc:
        raise ReportSerializationError(
            "report contains a non-finite numeric value"
        ) from exc
    leaked = leaked_secrets(serialized, forbidden_secrets)
    if leaked:
        raise ReportRedactionError(len(leaked))
    return serialized


class ReportSerializationError(RuntimeError):
    """The report cannot be represented as strict JSON."""


def _latent_statistics(samples) -> dict[str, float]:
    """Return finite latent statistics or refuse to attest a successful cell."""
    finite = samples.isfinite()
    if not bool(finite.all().item()):
        raise ValueError("sample latent contains a non-finite value")
    values = {
        "mean": float(samples.mean()),
        "std": float(samples.std()),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("sample latent statistics are non-finite")
    return values


def _not_available_result(
    name: str,
    case: dict,
    defaults: dict,
    error: BaseException,
) -> dict[str, object]:
    """Build a durable failure row without rendering attacker-controlled text.

    The raise site (the innermost frame's file name and line) rides beside the
    type: it names where to look without carrying the exception's message.
    """
    site = ""
    tb = error.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is not None:
        site = f"{Path(tb.tb_frame.f_code.co_filename).name}:{tb.tb_lineno}"
    return {
        "name": name,
        "status": "N/A",
        "reason": "case execution failed",
        "error_type": type(error).__name__,
        "error_site": site,
        "settings": _case_settings(case, defaults),
    }


def _is_gitignored(path: Path) -> bool:
    """Fail closed: any git/subprocess failure (no git, path outside a repo,
    binary missing) counts as not ignored, so sanitization still runs."""
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=str(REPO), capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def encode_prompts(te_spec: dict, prompt: str, negative: str, *, encode_negative: bool = True):
    import comfy.sd as comfy_sd
    import folder_paths

    te_names = te_spec["name"] if isinstance(te_spec["name"], list) else [te_spec["name"]]
    te_paths = [folder_paths.get_full_path("text_encoders", n) for n in te_names]
    clip = comfy_sd.load_clip(
        ckpt_paths=te_paths,
        embedding_directory=None,
        clip_type=getattr(comfy_sd.CLIPType, te_spec["type"].upper()),
        model_options={},
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative)) if encode_negative else None
    del clip
    import comfy.model_management as mm

    mm.soft_empty_cache()
    return positive, neg


def _init_tracked_mesh(
    settings: dict,
    mesh_handles: list,
    *,
    get_mesh_fn=None,
    init_node=None,
):
    """Acquire and record the headless mesh before Init can fail setup."""
    init_options = _init_options(settings)
    mode = settings.get("mode", "cluster")
    config_path = settings.get("config", "")
    if get_mesh_fn is None:
        from dgx_monarch.mesh import get_mesh as get_mesh_fn
    if init_node is None:
        from dgx_monarch.nodes.init import DGXMonarchInit

        init_node = DGXMonarchInit()

    # DGXMonarchInit obtains this exact singleton before topology setup. Do it
    # once up front so a topology/setup exception cannot strand an untracked
    # client-owned ProcMesh until the process-global atexit ordering runs.
    handle = get_mesh_fn(
        config_path=config_path,
        mode=mode,
        gpus_per_host=0,
    )
    mesh_handles.append(handle)
    (mesh,) = init_node.init(
        topology=settings["topology"],
        mode=mode,
        config_path=config_path,
        **init_options,
    )
    if mesh.handle is not handle:
        # Defensive against a future Init implementation changing its cache
        # key or replacing the handle between acquisition and return.
        mesh_handles.append(mesh.handle)
    return mesh


def _fidelity_gate(samples, merged: dict, results: dict) -> dict:
    """Score one candidate against the reference cell its case names.

    Two regimes (benchmark/gates.py):
      * `reference = "<case name>"` with steps == 1: cross-topology per-step
        fidelity under the normalized floor (gates.DEFAULT_STEP_NRMS says why
        one step);
      * the same reference over more steps: the strict same-math gate.

    N/A discipline covers the gate too: a mistyped or misordered reference, or a
    reference that itself failed, fails this cell instead of skipping the
    comparison. Both sides already hold the form their own row named, because
    every case reduces its own latent as it renders, so the slice below takes
    batch row 0 of each and compares like with like.
    """
    ref_name = merged["reference"]
    ref = results.get(ref_name)
    if ref is None:
        return {
            "pass": False, "reference": ref_name,
            "detail": f"reference case {ref_name!r} not found: check its name, and list it before this case"}
    if "latent" not in ref:
        return {
            "pass": False, "reference": ref_name,
            "detail": f"reference case {ref_name!r} produced no latent (status {ref.get('status')})"}
    from gates import gate_fidelity, gate_step_fidelity

    cand_slice, ref_slice = samples[:1], ref["latent"][:1]  # batch slice 0 on both sides
    if int(merged["steps"]) == 1:
        ok, detail = gate_step_fidelity(cand_slice, ref_slice)
    else:
        ok, detail = gate_fidelity(cand_slice, ref_slice)
    return {"pass": ok, "detail": detail, "reference": ref_name}


def run_case(
    case: dict,
    defaults: dict,
    results: dict,
    mesh_handles: list,
) -> dict:
    merged = _case_settings(case, defaults)
    sampling_path = _sampling_path(merged)

    import torch
    from gates import scored_latent

    from dgx_monarch.nodes.common import (
        LATENT_DOWNSCALE_METADATA_KEY,
        resolve_topology,
    )
    from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader
    from dgx_monarch.nodes.samplers import DGXMonarchKSampler

    name = merged["name"]
    print(f"\n=== {name} ===", flush=True)

    mesh = _init_tracked_mesh(merged, mesh_handles)
    (model,) = DGXMonarchUNETLoader().load(mesh, merged["unet"], merged.get("weight_dtype", "default"))
    model = _apply_model_sampling_sd3(model, merged)

    positive, negative = encode_prompts(
        merged["te"], merged["prompt"], merged["negative"],
        encode_negative=sampling_path == "ksampler",
    )
    batch = int(merged.get("batch", 1))  # dp reference cells need batch == dp degree
    latent, latent_downscale = _benchmark_latent(
        merged,
        batch,
        zeros=torch.zeros,
        metadata_key=LATENT_DOWNSCALE_METADATA_KEY,
    )
    sampling_cfg = float(merged["cfg"]) if sampling_path == "ksampler" else 1.0
    effective_topology, sage, topology_reason = resolve_topology(
        model,
        latent["samples"],
        sampling_cfg,
        latent_downscale=latent_downscale,
    )

    def sample_ksampler():
        (out,) = DGXMonarchKSampler().sample(
            model, int(merged["seed"]), int(merged["steps"]), float(merged["cfg"]),
            merged["sampler"], merged["scheduler"], positive, negative, latent,
            denoise=1.0,
        )
        return out

    def sample_flux2():
        import node_helpers
        from comfy_extras.nodes_custom_sampler import KSamplerSelect, RandomNoise
        from comfy_extras.nodes_flux import Flux2Scheduler

        from dgx_monarch.nodes.guiders import DGXMonarchBasicGuider
        from dgx_monarch.nodes.samplers import DGXMonarchSamplerCustom

        guided_positive = node_helpers.conditioning_set_values(
            positive, {"guidance": float(merged.get("guidance", 4.0))}
        )
        noise = RandomNoise.execute(noise_seed=int(merged["seed"]))[0]
        sampler = KSamplerSelect.execute(sampler_name=merged["sampler"])[0]
        sigmas = _flux2_schedule(
            Flux2Scheduler.execute,
            steps=int(merged["steps"]),
            width=int(merged["width"]),
            height=int(merged["height"]),
        )
        (guider,) = DGXMonarchBasicGuider().get_guider(model, guided_positive)
        out, _denoised = DGXMonarchSamplerCustom().sample(
            noise, guider, sampler, sigmas, latent
        )
        return out

    times = []
    samples = None
    for i in range(int(merged["repeats"])):
        t0 = time.perf_counter()
        out = _route_sampling(
            merged, ksampler=sample_ksampler, flux2=sample_flux2,
        )
        times.append(time.perf_counter() - t0)
        samples = out["samples"]
        print(f"  run {i}: {times[-1]:.1f}s", flush=True)

    # Reduce a returned cond/uncond pair before anything reads the latent, so
    # the statistics, the hash, the .npy sidecar and the gate all attest the
    # one form the row names.
    samples, latent_form = scored_latent(samples, batch, sampling_cfg)
    latent_stats = _latent_statistics(samples)
    row = {
        "name": name,
        "topology": merged["topology"],
        "world_size": mesh.world,
        "median_s": round(statistics.median(times), 2),
        "times_s": [round(t, 2) for t in times],
        "latent_form": latent_form,
        "latent_stats": latent_stats,
        "latent_sha256": _latent_sha256(samples),
        "settings": merged,
        "effective_topology": _topology_provenance(
            effective_topology,
            sage=sage,
            reason=topology_reason,
            attention=mesh.attention,
        ),
        "effective_worker_args": dict(
            mesh.handle.active_worker_args
            or mesh.handle.effective_worker_args(mesh.worker_args)
        ),
        "status": "ok",
    }

    ref_name = merged.get("reference")
    if ref_name:
        row["fidelity_gate"] = _fidelity_gate(samples, merged, results)
        gate = row["fidelity_gate"]
        print(f"  fidelity vs {ref_name}: {'PASS' if gate['pass'] else 'FAIL'} ({gate['detail']})")
    row["latent"] = samples  # kept in-memory for reference cells; stripped before JSON
    return row


def _save_latent_npy(results: dict[str, dict], keep_dir: str) -> None:
    """Write each case's latent as a float32 .npy sidecar before main() strips it.

    A no-op without --keep-latent-npy. A caller comparing this matrix's
    candidate leg against an independently produced reference needs the tensor,
    not just latent_sha256/latent_stats. The file holds the form the row scored,
    which the row's `latent_form` names: a cfg pair arrives already combined.
    """
    if not keep_dir:
        return
    import numpy as np

    out_dir = Path(keep_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in results.values():
        samples = row.get("latent")
        if samples is None:
            continue
        array = samples.detach().to(device="cpu").contiguous().float().numpy()
        np.save(out_dir / f"{row['name']}_latent.npy", array)


def _recycle_mesh_handles(handles: list) -> list[dict[str, object]]:
    """Recycle each distinct headless mesh once and return sanitized facts."""
    outcomes: list[dict[str, object]] = []
    seen: set[int] = set()
    for handle in handles:
        identity = id(handle)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            recycled = bool(handle.recycle())
        except BaseException as exc:
            # Cleanup must continue across every fleet and must not serialize
            # exception reprs that may contain private endpoints or paths.
            outcomes.append({
                "recycled": False,
                "error_type": type(exc).__name__,
            })
        else:
            outcomes.append({"recycled": recycled})
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(REPO / "benchmark" / "matrix.toml"))
    parser.add_argument("--out", default=str(REPO / "benchmark" / "results.json"))
    parser.add_argument("--config", default="", help="cluster.toml used for benchmark cells")
    parser.add_argument("--mode", choices=("cluster", "local", "auto"), default="cluster")
    parser.add_argument("--no-redact", action="store_true",
        help="skip the host, GPU UUID and path redaction when --out is not gitignored "
             "(prints a warning; the report keeps the raw host name, UUIDs and paths)")
    parser.add_argument("--keep-latent-npy", default="",
        help="directory to save each case's scored latent in, as a float32 "
             "<case>_latent.npy (off by default; the JSON report is the same "
             "either way)")
    args = parser.parse_args(CLI_ARGS)

    # From here onward imports may reach comfy.cli_args. Hide the benchmark's
    # flags so Comfy sees only the program name, as a headless driver does.
    sys.argv = [sys.argv[0]]
    repo_src = str(REPO / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    if COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)

    sys.path.insert(0, str(REPO / "benchmark"))
    with open(args.matrix, "rb") as f:
        matrix = tomllib.load(f)

    defaults = {**matrix.get("defaults", {}), "config": args.config, "mode": args.mode}
    results: dict[str, dict] = {}
    mesh_handles: list = []
    cases = matrix.get("case", [])
    names = [case.get("name") for case in cases]
    if not cases or None in names or len(names) != len(set(names)):
        print("the matrix needs at least one case, and every case needs a unique name", file=sys.stderr)
        return 2
    try:
        for case in cases:
            name = case["name"]
            try:
                results[name] = run_case(
                    case, defaults, results, mesh_handles=mesh_handles,
                )
            except Exception as exc:  # N/A discipline: record, do not silently skip
                # Exception strings can contain private endpoints, paths,
                # credentials, or model names. Keep durable and console
                # diagnostics typed and content-free.
                results[name] = _not_available_result(name, case, defaults, exc)
                print(f"  N/A: {type(exc).__name__}", flush=True)
    finally:
        mesh_cleanup = _recycle_mesh_handles(mesh_handles)

    _save_latent_npy(results, args.keep_latent_npy)
    for row in results.values():
        row.pop("latent", None)
    def version(name: str) -> str:
        try:
            from importlib.metadata import version as package_version

            return package_version(name)
        except Exception:
            return "missing"

    repo_git = _git_provenance(REPO)
    comfy_git = _git_provenance(COMFY_DIR)
    host = socket.gethostname()
    gpu_hardware = _gpu_hardware()
    payload = {
        "provenance": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # repo_commit and comfy_commit repeat the commits in repo_git and
            # comfy_git, for readers that read only the scalar fields.
            "repo_commit": repo_git["commit"],
            "comfy_commit": comfy_git["commit"],
            "repo_git": repo_git,
            "comfy_git": comfy_git,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": host,
            "comfy_dir": COMFY_DIR,
            "matrix": _matrix_provenance(args.matrix),
            "config": _config_provenance(args.config),
            "mode": args.mode,
            "gpu_hardware": gpu_hardware,
            "case_artifacts": _case_artifact_provenance(cases, defaults),
            "versions": {name: version(name) for name in
                         ("dgx-monarch", "torch", "torchmonarch", "xfuser", "yunchang")},
            "mesh_cleanup": mesh_cleanup,
        },
        "results": list(results.values()),
    }

    out_path = Path(args.out).resolve()
    ignored = _is_gitignored(out_path)
    to_write = payload
    if ignored:
        pass  # gitignored destination: full fidelity (docs/THREAT_MODEL.md)
    elif args.no_redact:
        print(f"WARNING: --no-redact: writing un-redacted hostname/UUID/paths to "
              f"a non-gitignored path ({out_path})", file=sys.stderr)
    else:
        to_write = sanitize_report(payload)

    forbidden_secrets: list[str] = []
    if to_write is not payload:
        forbidden_secrets = [
            host,
            *_gpu_uuids(gpu_hardware),
            *private_path_secrets(payload),
        ]
    try:
        serialized = _serialize_report(to_write, forbidden_secrets)
    except (ReportRedactionError, ReportSerializationError) as exc:
        if isinstance(exc, ReportRedactionError):
            detail = (
                f"{exc.count} secret value(s) remain despite sanitize_report"
            )
        else:
            detail = "report contains non-finite numeric values"
        print(f"REPORT FAILURE: {detail}; destination was not written", file=sys.stderr)
        return 3

    with open(args.out, "w") as f:
        f.write(serialized)
        f.write("\n")
    print(f"\nwrote {args.out}")

    failed = [r for r in results.values()
              if r.get("fidelity_gate", {}).get("pass") is False or r["status"] == "N/A"]
    if any(not outcome.get("recycled") for outcome in mesh_cleanup):
        failed.append({"status": "mesh cleanup failed"})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
