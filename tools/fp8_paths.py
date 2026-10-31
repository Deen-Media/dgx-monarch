#!/usr/bin/env python3
"""Report which operations class ComfyUI picks for each diffusion checkpoint, in four settings.

    python tools/fp8_paths.py ~/ComfyUI/models/diffusion_models
    python tools/fp8_paths.py --json FILE [FILE ...]
    python tools/fp8_paths.py --comfy-dir ~/ComfyUI ~/ComfyUI/models/diffusion_models

It reads safetensors headers, plus the few bytes of each per-layer
``comfy_quant`` record a header points at, and imports no ComfyUI, torch or
CUDA, so it runs on a CPU box and beside a live render. Directories are walked
for ``.safetensors`` and ``.sft`` files; rows name files, never paths. For
each file it reports:

- comfy-kitchen quantization metadata: where ComfyUI finds it (the header's
  ``_quantization_metadata``, per-layer ``comfy_quant`` tensors, or the legacy
  ``scaled_fp8`` marker) and the formats it names;
- the weight dtypes present, by bytes, and the one ComfyUI's ``weight_dtype``
  picks (the dtype with the most elements);
- the operations class ComfyUI picks in four settings (plain,
  ``--fast fp8_matrix_mult``, the loader's ``weight_dtype=fp8_e4m3fn_fast``,
  and both) and what that class does to the matmuls.

The decision is ComfyUI's own. Its functions cannot be imported without
``comfy.model_management``, which sets up CUDA, so this file reproduces them.
Line numbers are ComfyUI 0.43.0 (426f5eea5):

- nodes.py 1003-1012, UNETLoader: ``fp8_e4m3fn_fast`` sets dtype
  float8_e4m3fn and ``fp8_optimizations``.
- comfy/sd.py 2355-2443, load_diffusion_model_state_dict: convert_old_quants,
  the unet prefix strip, weight_dtype, quant_config, unet_dtype,
  unet_manual_cast, then ``model_options["fp8_optimizations"]``.
- comfy/utils.py 254-264 (weight_dtype), 1467-1532 (detect_layer_quantization:
  any ``<layer>.comfy_quant`` key sets quant_config; convert_old_quants).
- comfy/model_management.py 1136-1213 (unet_dtype keeps an FP8 weight dtype
  when supports_fp8_compute; unet_manual_cast), 2030-2054
  (supports_fp8_compute: sm_89 and newer, so GB10's sm_121).
- comfy/model_base.py 183-185: pick_operations(unet dtype, manual cast dtype,
  fp8_optimizations=optimizations["fp8"], model_config).
- comfy/ops.py 1902-1929, pick_operations: quant_config gives
  mixed_precision_ops; FP8 compute and (fp8_optimizations or --fast
  fp8_matrix_mult) gives fp8_ops; then disable_weight_init when the weight
  dtype is the compute dtype, else manual_cast. 926-981: fp8_ops sends only
  float8_e4m3fn weights through torch._scaled_mm. 1871-1885: formats the
  device cannot compute dequantize.
- comfy/cli_args.py 197-203, 327-334: ``--fast fp8_matrix_mult`` adds one
  PerformanceFeature; a bare ``--fast`` adds all of them, autotune included.

``--comfy-dir`` re-reads those clauses from a checkout as text and warns when
one is gone, so a ComfyUI update cannot silently outdate the table.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The sibling src, so the tool runs from any directory and a worktree reads its
# own package. The entry leaves sys.path once the imports are done, as in
# tools/gen_templates.py: a process that loads this file as a library keeps
# the sys.path it had.
_SRC = str(ROOT / "src")
_ADDED_SRC = _SRC not in sys.path
if _ADDED_SRC:
    sys.path.insert(0, _SRC)
try:
    from dgx_monarch.safetensors_header import read_safetensors_header
    from dgx_monarch.upstream_gate import comfy_load_keys
finally:
    if _ADDED_SRC:
        sys.path.remove(_SRC)

SUFFIXES = (".safetensors", ".sft")
SETTINGS = (
    ("plain", False, False),
    ("--fast fp8_matrix_mult", True, False),
    ("weight_dtype=fp8_e4m3fn_fast", False, True),
    ("both", True, True),
)
# safetensors dtype names as torch spells them.
TORCH_NAMES = {
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "F8_E8M0": "float8_e8m0fnu",
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64",
    "F4": "float4_e2m1fn_x2", "BOOL": "bool",
}
FP8 = ("float8_e4m3fn", "float8_e5m2")
# The common name of each comfy-kitchen format a release file carries.
FORMAT_NAMES = {"float8_e4m3fn": "fp8_scaled", "float8_e5m2": "fp8_scaled (e5m2)",
                "int8_tensorwise": "int8", "mxfp8": "mxfp8", "nvfp4": "nvfp4"}
# Comfy's bound for one comfy_quant record is generous; real ones are tens of bytes.
MAX_RECORD_BYTES = 4096
# comfy.model_detection.unet_prefix_from_state_dict's candidates.
PREFIXES = ("model.diffusion_model.", "model.model.", "net.", "model.", "")


@dataclass
class Facts:
    name: str
    size_bytes: int = 0
    kitchen: bool = False
    sources: list[str] = field(default_factory=list)
    formats: dict[str, int] = field(default_factory=dict)
    convrot_layers: int = 0
    full_precision_layers: int = 0
    dtype_bytes: dict[str, int] = field(default_factory=dict)
    weight_dtype: str | None = None
    error: str = ""


@dataclass
class Pick:
    operations: str
    matmul: str
    notes: list[str] = field(default_factory=list)


def _numel(descriptor) -> int:
    count = math.prod(descriptor.shape) if descriptor.shape else 1
    # torch loads F4 as float4_e2m1fn_x2, two values per element.
    return count // 2 if descriptor.dtype == "F4" else count


def _record(path: Path, descriptor) -> dict | None:
    """One per-layer comfy_quant record: a short UTF-8 JSON object in a 1-D U8 tensor."""
    if descriptor.dtype != "U8" or len(descriptor.shape) != 1 or not 0 < descriptor.nbytes <= MAX_RECORD_BYTES:
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        raw = os.pread(fd, descriptor.nbytes, descriptor.start)
    finally:
        os.close(fd)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _legacy_marker(tensors: dict) -> object | None:
    """The scaled_fp8 marker convert_old_quants finds at the root or under a unet prefix."""
    for prefix in PREFIXES:
        if f"{prefix}scaled_fp8" in tensors:
            return tensors[f"{prefix}scaled_fp8"]
    return None


def _layer_confs(path: Path, header) -> tuple[dict[str, dict], list[str]]:
    """Each quantized layer's config by file layer name, and where ComfyUI found them.

    Precedence is convert_old_quants': the header's _quantization_metadata
    names its layers and suppresses the legacy conversion; without it, the
    legacy scaled_fp8 marker covers every layer with a scale_weight; a direct
    per-layer record stands for any layer neither of those names.
    """
    tensors, metadata = header.tensors, header.metadata
    confs: dict[str, dict] = {}
    sources: list[str] = []
    raw = metadata.get("_quantization_metadata")
    if isinstance(raw, str):
        try:
            layers = json.loads(raw).get("layers", {})
        except (ValueError, AttributeError):
            layers = {}
        confs.update({name: conf for name, conf in layers.items() if isinstance(conf, dict)})
        sources.append("header _quantization_metadata")
    else:
        marker = _legacy_marker(tensors)
        if marker is not None:
            numel = math.prod(marker.shape) if marker.shape else 1
            for key in tensors:
                if key.endswith(".scale_weight"):
                    conf = {"format": "float8_e4m3fn"}
                    if numel == 2:
                        conf["full_precision_matrix_mult"] = True
                    confs[key[: -len(".scale_weight")]] = conf
            sources.append("legacy scaled_fp8 marker")
    direct = 0
    for key, descriptor in tensors.items():
        if key.endswith(".comfy_quant") and key[: -len(".comfy_quant")] not in confs:
            conf = _record(path, descriptor)
            if conf is not None:
                confs[key[: -len(".comfy_quant")]] = conf
                direct += 1
    if direct:
        sources.append("per-layer comfy_quant tensors")
    return confs, sources


def _weight_dtype(header, keys: dict[str, str]) -> str | None:
    """comfy.utils.weight_dtype over the keys the load hands the model: most elements wins."""
    counts: Counter[str] = Counter()
    for key in keys:
        descriptor = next((header.tensors[p + key] for p in PREFIXES if p + key in header.tensors), None)
        if descriptor is not None:
            counts[descriptor.dtype] += _numel(descriptor)
    if not counts:
        return None
    return TORCH_NAMES.get(counts.most_common(1)[0][0], counts.most_common(1)[0][0])


def read_facts(path: Path) -> Facts:
    facts = Facts(name=path.name)
    try:
        facts.size_bytes = path.stat().st_size
        header = read_safetensors_header(path)
        keys = comfy_load_keys({key: d.dtype for key, d in header.tensors.items()}, header.metadata)
        facts.kitchen = any(key.endswith(".comfy_quant") for key in keys)
        confs, facts.sources = _layer_confs(path, header)
        formats = Counter(str(conf.get("format")) for conf in confs.values())
        facts.formats = dict(sorted(formats.items()))
        facts.convrot_layers = sum(1 for conf in confs.values()
                                   if conf.get("convrot") or (conf.get("params") or {}).get("convrot"))
        facts.full_precision_layers = sum(1 for conf in confs.values() if conf.get("full_precision_matrix_mult"))
        by_dtype: Counter[str] = Counter()
        for descriptor in header.tensors.values():
            by_dtype[TORCH_NAMES.get(descriptor.dtype, descriptor.dtype)] += descriptor.nbytes
        facts.dtype_bytes = dict(by_dtype.most_common())
        facts.weight_dtype = _weight_dtype(header, keys)
    except Exception as exc:  # one unreadable file never stops the table
        facts.error = f"{type(exc).__name__}: {exc}"[:200]
    return facts


def kitchen_label(facts: Facts) -> str:
    """The common name of the file's kitchen formats, e.g. fp8_scaled or int8_convrot.

    Layers that set full_precision_matrix_mult are counted, because their
    matmuls dequantize to the compute dtype whatever the flags say.
    """
    names = []
    for fmt in facts.formats:
        name = FORMAT_NAMES.get(fmt, fmt)
        if fmt == "int8_tensorwise" and facts.convrot_layers:
            name = "int8_convrot"
        names.append(name)
    label = " + ".join(names)
    if facts.full_precision_layers:
        label += f", {facts.full_precision_layers} of {sum(facts.formats.values())} layers full precision"
    return label


def predict(facts: Facts, fast: bool, loader_fast: bool, fp8_compute: bool = True) -> Pick:
    """The class pick_operations returns for this file, and what it does to the matmuls."""
    if facts.kitchen:
        formats = ", ".join(facts.formats) or "unnamed"
        notes = []
        if facts.full_precision_layers:
            notes.append(f"{facts.full_precision_layers} layer(s) set full_precision_matrix_mult and "
                         "dequantize to the compute dtype")
        if not fp8_compute and any(fmt in FP8 for fmt in facts.formats):
            notes.append("without FP8 compute the FP8 layers dequantize")
        if loader_fast:
            notes.append("the loader's float8_e4m3fn dtype still reaches the weights comfy-kitchen does "
                         "not quantize, and adds no FP8 matmul")
        return Pick("mixed_precision_ops", f"comfy-kitchen GEMMs ({formats})", notes)
    notes = []
    if loader_fast:
        stored = "float8_e4m3fn"
        if facts.weight_dtype not in FP8:
            notes.append(f"the load casts {facts.weight_dtype} weights to float8_e4m3fn with no scale")
    elif facts.weight_dtype in FP8:
        stored = facts.weight_dtype if fp8_compute else "float8 or an upcast, by free memory"
    else:
        stored = "full precision"
    if fp8_compute and (fast or loader_fast):
        if stored == "float8_e4m3fn":
            return Pick("fp8_ops", "torch._scaled_mm: FP8 weights, activations cast to FP8 with unit scale", notes)
        return Pick("fp8_ops", "cast path: fp8_linear runs only float8_e4m3fn weights, so the matmul "
                               "stays in the compute dtype", notes)
    if stored in FP8 or stored.startswith("float8"):
        return Pick("manual_cast", "FP8 weights dequantized to the compute dtype for every matmul", notes)
    return Pick("disable_weight_init", "native matmul in the stored dtype", notes)


SHORT = {
    "mixed_precision_ops": "kitchen",
    "fp8_ops": "FP8 scaled_mm",
    "manual_cast": "dequant",
    "disable_weight_init": "native",
}


def cell(pick: Pick) -> str:
    tag = SHORT.get(pick.operations, "")
    if pick.operations == "fp8_ops" and pick.matmul.startswith("cast path"):
        tag = "cast path, no FP8 matmul"
    return f"{pick.operations} ({tag})" + (" *" if pick.notes else "")


def iter_files(targets: list[Path]) -> list[Path]:
    """Every safetensors file under the targets, sorted by name; symlinks followed."""
    found: dict[str, Path] = {}
    for target in targets:
        if target.is_dir():
            for root, dirs, files in os.walk(target, followlinks=True):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for name in files:
                    if name.lower().endswith(SUFFIXES):
                        found.setdefault(os.path.join(root, name), Path(root, name))
        elif target.name.lower().endswith(SUFFIXES):
            found.setdefault(str(target), target)
    return sorted(found.values(), key=lambda p: p.name.lower())


def table(rows: list[tuple[Facts, list[Pick]]]) -> str:
    out = ["| File | GiB | Kitchen metadata | Weight dtype | " + " | ".join(s[0] for s in SETTINGS) + " |",
           "|---|---:|---|---|" + "---|" * len(SETTINGS)]
    for facts, picks in rows:
        if facts.error:
            out.append(f"| {facts.name} | | unreadable: {facts.error} | |" + " |" * len(SETTINGS))
            continue
        meta = f"{kitchen_label(facts)} ({'; '.join(facts.sources)})" if facts.kitchen else "none"
        out.append(f"| {facts.name} | {facts.size_bytes / 2**30:.1f} | {meta} | {facts.weight_dtype} | "
                   + " | ".join(cell(p) for p in picks) + " |")
    return "\n".join(out)


# The clauses of ComfyUI's rule this file reproduces, by the file and function
# that must still hold them (whitespace-normalized source text).
CLAUSES = (
    ("comfy/ops.py", "pick_operations", "model_config.quant_config"),
    ("comfy/ops.py", "pick_operations", "return mixed_precision_ops("),
    ("comfy/ops.py", "pick_operations",
     "fp8_optimizations or PerformanceFeature.Fp8MatrixMultiplication in args.fast"),
    ("comfy/ops.py", "pick_operations", "return fp8_ops"),
    ("comfy/ops.py", "pick_operations", "if compute_dtype is None or weight_dtype == compute_dtype:"),
    ("comfy/ops.py", "fp8_linear", "if dtype not in [torch.float8_e4m3fn]:"),
    ("nodes.py", "UNETLoader.load_unet", 'model_options["fp8_optimizations"] = True'),
    ("comfy/sd.py", "load_diffusion_model_state_dict", 'if model_options.get("fp8_optimizations", False):'),
    ("comfy/model_management.py", "unet_dtype", "if supports_fp8_compute(device):"),
    ("comfy/utils.py", "detect_layer_quantization", 'k.endswith(".comfy_quant")'),
)


def _function_text(source: str, qualname: str) -> str | None:
    tree = ast.parse(source)
    parts = qualname.split(".")
    body = tree.body
    node = None
    for part in parts:
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part), None)
        if node is None:
            return None
        body = node.body
    text = ast.get_source_segment(source, node) if node is not None else None
    return re.sub(r"\s+", " ", text) if text else None


def check_rule(comfy_dir: Path) -> list[str]:
    """Clauses the checkout no longer holds; empty when the reproduced rule still matches."""
    missing = []
    cache: dict[str, str] = {}
    for relative, qualname, clause in CLAUSES:
        path = comfy_dir / relative
        if relative not in cache:
            cache[relative] = path.read_text(errors="replace") if path.is_file() else ""
        text = _function_text(cache[relative], qualname) if cache[relative] else None
        if text is None or re.sub(r"\s+", " ", clause) not in text:
            missing.append(f"{relative} {qualname}: {clause}")
    return missing


def comfy_version(comfy_dir: Path) -> str:
    match = re.search(r'__version__\s*=\s*"([^"]+)"', (comfy_dir / "comfyui_version.py").read_text()) \
        if (comfy_dir / "comfyui_version.py").is_file() else None
    return match.group(1) if match else "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/fp8_paths.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", nargs="+", type=Path, help="checkpoint files or directories")
    parser.add_argument("--json", action="store_true", help="print one JSON document instead of the table")
    parser.add_argument("--no-fp8-compute", action="store_true",
                        help="answer for a device without FP8 compute (below sm_89); GB10 has it")
    parser.add_argument("--comfy-dir", type=Path, help="check the reproduced rule against this ComfyUI checkout")
    args = parser.parse_args(argv)
    fp8_compute = not args.no_fp8_compute
    rows = []
    for path in iter_files([t.expanduser() for t in args.targets]):
        facts = read_facts(path)
        rows.append((facts, [predict(facts, fast, loader_fast, fp8_compute) for _, fast, loader_fast in SETTINGS]))
    drift = check_rule(args.comfy_dir.expanduser()) if args.comfy_dir else None
    if args.json:
        print(json.dumps({
            "fp8_compute": fp8_compute,
            "comfy": None if args.comfy_dir is None else {"version": comfy_version(args.comfy_dir.expanduser()),
                                                          "missing_clauses": drift},
            "files": [{**asdict(facts), "kitchen_label": kitchen_label(facts) if facts.kitchen else None,
                       "picks": {name: asdict(pick) for (name, _, _), pick in zip(SETTINGS, picks, strict=True)}}
                      for facts, picks in rows],
        }, indent=1))
    else:
        print(table(rows))
        noted = [(facts.name, name, note) for facts, picks in rows
                 for (name, _, _), pick in zip(SETTINGS, picks, strict=True) for note in pick.notes]
        if noted:
            print("\n* " + "\n* ".join(f"{name}, {setting}: {note}" for name, setting, note in noted))
        if args.comfy_dir:
            version = comfy_version(args.comfy_dir.expanduser())
            print(f"\nComfyUI {version}: " + ("the reproduced rule holds." if not drift else
                  "these clauses are gone, so the table may be wrong: " + "; ".join(drift)))
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
