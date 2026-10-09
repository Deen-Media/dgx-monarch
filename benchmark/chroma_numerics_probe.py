#!/usr/bin/env python3
"""Probe Chroma nvfp4 quantization and unequal-prompt CFG error on one GPU.

Probe A measures layer-level terms behind the uly2 one-step NRMS gap:
Chroma nvfp4 measured 0.120 with shared activation scale, versus 0.032 at bf16
and 0.011 for Krea2 nvfp4 (docs/VALIDATION.md, 2026-09-03 and 2026-09-05).
Per-rank amax accounted for about 0.01. Tap each nvfp4 Linear's first-step
input and measure:

* ``shard_amax_ratio``: whole-tensor amax divided by the smallest shard amax;
  this identifies layers affected by independent shard scales.
* ``nrms_shared_scale``: whole-call quantization versus two shards using the
  shared amax. Plain torch arithmetic gives zero because token sharding leaves
  feature blocks intact. It does not test the packed kernel's row padding or
  scale swizzle; those need separate measurements in docs/VALIDATION.md.
* ``amplification``: output change divided by an input perturbation at bf16
  scale. Compare Chroma with the Krea2 nvfp4 control: similar amplification
  in both families cannot explain their different sharded errors.

``LayerTerms.nrms_shard_scale`` supplies the per-rank term reported as A2.

Probe B separates two possible causes of Chroma's unequal-prompt result:
0.150 at bf16 versus 0.031 for the even-token template on the same build
(docs/VALIDATION.md, 2026-09-06). Separate model calls round independently and
CFG weights them by 3.5 and 2.5; alternatively, the 28-row stream could diverge
on its own. Analyze the cases in chroma_cfg_amplification_matrix.toml.

Synthetic probe, requiring no GPU:

    python benchmark/chroma_numerics_probe.py --dry-run

Real probe in a ComfyUI environment on one GPU, with no workers running:

    python benchmark/chroma_numerics_probe.py --unet Chroma1-HD-nvfp4.safetensors \
        --te t5xxl_fp8_e4m3fn_scaled.safetensors --out probe.md
    python benchmark/chroma_numerics_probe.py --unet krea2_raw_nvfp4.safetensors \
        --te qwen3vl_4b_bf16.safetensors --te-type krea2 --out control.md

Supply measured CFG cases to probe B:

    python benchmark/chroma_numerics_probe.py --dry-run \
        --nrms gate-bf16-cfg35-unequal=0.150 --nrms gate-bf16-cfg35-equal=0.031

docs/VALIDATION.md records acceptance criteria (2026-09-09) and results
(reconciled 2026-09-11): both families amplified similarly, and every CFG case
was below the floor.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent

# comfy's nvfp4 constants; the two maxima keep their comfy_kitchen.float_utils names.
F8_E4M3_MAX = 448.0
F4_E2M1_MAX = 6.0
NVFP4_BLOCK = 16
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# A bf16 mantissa is 8 bits, so a sharded computation hands the next layer an
# input that differs in the last place, about 4e-3 relative. That is the move
# probe A pushes through the quantizer.
BF16_RELATIVE_STEP = 4e-3

# Chroma's sharded 1-step readings at uly2: bf16 on 2026-09-03, nvfp4 with the
# shared scale on 2026-09-05 (docs/VALIDATION.md).
MEASURED_BF16_NRMS = 0.032
MEASURED_NVFP4_NRMS = 0.120

# The harness's one-step floor (gates.DEFAULT_STEP_NRMS). A leg at or under it
# reads low and a leg above it reads high, the words the matrix file uses for
# its three cases.
CFG_LEG_FLOOR = 0.10


def nvfp4_tensor_scale(tensor: torch.Tensor) -> float:
    """The scale comfy falls back to when an nvfp4 module ships none."""
    return float(tensor.abs().amax()) / (F8_E4M3_MAX * F4_E2M1_MAX)


def quantize_nvfp4(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Round one activation the way comfy's nvfp4 layout rounds it.

    A per-tensor scale, then a per-16-element block scale held in e4m3, then
    E2M1 elements. comfy runs this as a packed CUDA kernel that needs a
    Blackwell part (comfy_kitchen.quantize_nvfp4); this is the same arithmetic
    in plain torch, which returns the dequantized values and runs anywhere. The
    kernel's tie rule is not pinned here, so read a layer whose numbers turn on
    single ties as unproven.
    """
    if scale <= 0.0:
        return torch.zeros_like(tensor)
    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    rows, columns = flat.shape
    pad = (-columns) % NVFP4_BLOCK
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(rows, -1, NVFP4_BLOCK)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    step = (block_amax / F4_E2M1_MAX / scale).to(torch.float8_e4m3fn).float() * scale
    levels = torch.tensor(E2M1_LEVELS, dtype=torch.float32, device=tensor.device)
    midpoints = (levels[1:] + levels[:-1]) / 2
    units = blocks / step.clamp_min(torch.finfo(torch.float32).tiny)
    rounded = levels[torch.bucketize(units.abs(), midpoints)] * units.sign()
    out = (rounded * step).reshape(rows, -1)[:, :columns]
    return out.reshape(tensor.shape).to(tensor.dtype)


def shard_axis(activation: torch.Tensor) -> int:
    """Which axis ulysses splits: the tokens, never the batch."""
    return 1 if activation.ndim >= 3 else 0


def token_shards(activation: torch.Tensor, world: int) -> list[torch.Tensor]:
    """The tokens each rank of a ulysses group holds for a Linear.

    A folded cond/uncond call reaches a Linear as batch 2, and splitting its
    leading axis would emulate cfg-parallel rather than ulysses. So a
    (batch, tokens, features) activation splits on the token axis and every
    rank keeps both halves of the pair, which is what ulysses does.
    """
    axis = shard_axis(activation)
    length = int(activation.shape[axis])
    if world < 1 or length % world:
        raise ValueError(f"{length} tokens do not split into {world} equal shards")
    return list(activation.split(length // world, dim=axis))


def normalized_rms(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """The same reading the fidelity gate takes, on a layer instead of a latent."""
    diff = candidate.float() - reference.float()
    return float(diff.pow(2).mean().sqrt() / reference.float().std().clamp_min(1e-8))


@dataclass(frozen=True)
class LayerTerms:
    """One nvfp4 Linear's shard terms at step one."""

    path: str
    tokens: int
    amax_whole: float
    amax_shard_min: float
    shard_amax_ratio: float
    nrms_shard_scale: float
    nrms_shared_scale: float
    amplification: float


def layer_terms(path: str, activation: torch.Tensor, *, world: int = 2,
                quantize=quantize_nvfp4, generator=None) -> LayerTerms:
    """Price one layer's shard terms from its own step-one input."""
    activation = activation.float()
    axis = shard_axis(activation)
    whole_scale = nvfp4_tensor_scale(activation)
    whole = quantize(activation, whole_scale)
    shards = token_shards(activation, world)
    local = [nvfp4_tensor_scale(shard) for shard in shards]
    shared = max(local)  # the reduction the hook installs: max over ranks
    per_rank = torch.cat(
        [quantize(s, own) for s, own in zip(shards, local, strict=True)], dim=axis)
    common = torch.cat([quantize(s, shared) for s in shards], dim=axis)
    move = BF16_RELATIVE_STEP * float(activation.abs().mean()) * torch.randn(
        activation.shape, generator=generator, dtype=torch.float32)
    moved = quantize(activation + move, nvfp4_tensor_scale(activation + move))
    carried = float((moved - whole).pow(2).mean().sqrt())
    return LayerTerms(
        path=path,
        tokens=int(activation.shape[axis]),
        amax_whole=whole_scale * F8_E4M3_MAX * F4_E2M1_MAX,
        amax_shard_min=min(local) * F8_E4M3_MAX * F4_E2M1_MAX,
        shard_amax_ratio=whole_scale / max(min(local), 1e-30),
        nrms_shard_scale=normalized_rms(per_rank, whole),
        nrms_shared_scale=normalized_rms(common, whole),
        amplification=carried / max(float(move.pow(2).mean().sqrt()), 1e-30),
    )


def price_layers(layers: list[tuple[str, torch.Tensor]], *, world: int = 2,
                 generator=None) -> tuple[list[LayerTerms], list[tuple[str, str]]]:
    """Price every tapped layer that shards, and name the ones that do not.

    A tapped input whose token count does not divide by `world`, or one with no
    token axis at all, cannot be split into equal shards. Such a layer is
    skipped and named in the note, and the run keeps its other layers, so one
    odd shape does not cost the whole leg.
    """
    terms: list[LayerTerms] = []
    skipped: list[tuple[str, str]] = []
    for path, activation in layers:
        try:
            terms.append(layer_terms(path, activation, world=world, generator=generator))
        except ValueError as refusal:
            skipped.append((path, str(refusal)))
    return terms, skipped


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def layer_verdicts(terms: list[LayerTerms], asked_ratio: float | None = None) -> list[str]:
    """Read probe A's three accept criteria off the per-layer table.

    `asked_ratio` is the family's own sharded nvfp4 reading over its sharded
    bf16 reading, which is what the amplification has to account for. It
    defaults to chroma's 0.120 over 0.032. Pass krea2's own ratio for the
    control leg (the module docstring says why).
    """
    if not terms:
        return ["no nvfp4 layer was tapped, so probe A read nothing"]
    moved = [t for t in terms if t.shard_amax_ratio > 1.000001]
    exact = [t for t in terms if t.nrms_shared_scale > 0.0]
    amplification = _median([t.amplification for t in terms])
    wanted = MEASURED_NVFP4_NRMS / MEASURED_BF16_NRMS if asked_ratio is None else asked_ratio
    named = f"; the first that did not is {exact[0].path}" if exact else ""
    lines = [
        f"A1 shared scale exact: {len(terms) - len(exact)} of {len(terms)} layers "
        f"read 0 for the split call against the whole call{named}. This column "
        f"reads 0 for every input the default quantizer takes, because the amax "
        f"over a partition is the max of the parts and the blocks lie along the "
        f"features while the shard splits the tokens. It confirms the arithmetic "
        f"and cannot see the packed kernel's own row pad or block-scale swizzle, "
        f"which needs a hardware leg of its own",
        f"A2 per-rank amax term: {len(moved)} of {len(terms)} layers have shards "
        f"that disagree on the amax; median layer nrms {_median([t.nrms_shard_scale for t in terms]):.4f}",
        f"A3 bin-flip amplification: median {amplification:.2f} against the "
        f"{wanted:.2f} this family's own nvfp4 over bf16 readings ask for",
    ]
    if amplification >= wanted:
        lines.append(
            "A3 reads at or above the asked factor, so the bin flips carry the "
            "gap on this family; read the control leg before you accept that, "
            "because a factor that reads alike on krea2 explains neither")
    else:
        lines.append(
            "A3 reads under the asked factor, so the bin flips do not carry the "
            "whole gap and hypothesis 2 or 3 must explain the rest")
    return lines


@dataclass(frozen=True)
class CfgLeg:
    """One leg of the cfg amplification matrix and what it read."""

    name: str
    folded: bool  # whether comfy folds its pair into one model call
    tokens: tuple[int, int]
    nrms: float | None = None


# The legs the matrix file defines, in its own order and with its own counts.
CFG_LEGS = (
    CfgLeg("gate-bf16-cfg35-unequal", folded=False, tokens=(100, 28)),
    CfgLeg("gate-bf16-cfg35-equal", folded=True, tokens=(44, 44)),
    CfgLeg("gate-fp8mixed-cfg35-unequal", folded=False, tokens=(100, 28)),
    CfgLeg("gate-bf16-cfg35-short-equal", folded=True, tokens=(28, 28)),
)


# The shipped negative's own row count, and with it the line between a leg whose
# streams are both short and one whose folded pair is the even template's.
SHORT_STREAM_TOKENS = 28


def is_short(leg: CfgLeg) -> bool:
    """Whether both of this leg's streams are as short as the negative."""
    return max(leg.tokens) <= SHORT_STREAM_TOKENS


def combine_gain(cfg: float, folded: bool) -> float:
    """How much the cfg combine multiplies one leg's own deviation.

    One model call carrying both legs rounds them alike, and the combine
    `uncond + (cond - uncond) * cfg` subtracts that common part, leaving the
    deviation itself. Two calls round independently, so the same per-leg
    deviation adds in quadrature weighted cfg and cfg - 1.
    """
    return 1.0 if folded else math.hypot(cfg, cfg - 1.0)


def cfg_split_rows(legs: list[CfgLeg], cfg: float) -> list[dict]:
    """Divide each leg's reading by its own combine gain."""
    rows = []
    for leg in legs:
        gain = combine_gain(cfg, leg.folded)
        rows.append({
            "leg": leg.name,
            "tokens": f"{leg.tokens[0]} and {leg.tokens[1]}",
            "calls": "one" if leg.folded else "two",
            "gain": round(gain, 2),
            "nrms": leg.nrms,
            "per_leg": None if leg.nrms is None else round(leg.nrms / gain, 4),
        })
    return rows


def cfg_verdict(legs: list[CfgLeg], floor: float = CFG_LEG_FLOOR) -> str:
    """The reading the matrix file asks for, in its own three cases."""
    scored = [leg for leg in legs if leg.nrms is not None]
    if not scored:
        return "no leg carries a reading yet, so probe B says nothing"
    split = [leg for leg in scored if not leg.folded]
    equal = [leg for leg in scored if leg.folded and not is_short(leg)]
    short = [leg for leg in scored if leg.folded and is_short(leg)]
    if not (split and equal and short):
        return ("the three cases need a split leg, an equal leg and a short equal "
                "leg; one of them has no reading yet")
    if all(leg.nrms <= floor for leg in split + equal + short):
        return "every leg reads low, so the term is in the sweep's own probe"
    if any(leg.nrms > floor for leg in short):
        return "the short stream carries the term, not the split model call"
    if any(leg.nrms > floor for leg in split) and all(
            leg.nrms <= floor for leg in equal + short):
        return "the split model call carries the term"
    return "the legs read in no pattern the matrix file named; read the table"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def report(terms: list[LayerTerms], legs: list[CfgLeg], cfg: float,
           *, synthetic: bool = False, asked_ratio: float | None = None,
           skipped: list[tuple[str, str]] | None = None) -> str:
    """The whole probe as one markdown note."""
    layer_rows = [[
        t.path, str(t.tokens), f"{t.amax_whole:.4g}", f"{t.amax_shard_min:.4g}",
        f"{t.shard_amax_ratio:.3f}", f"{t.nrms_shard_scale:.4f}",
        f"{t.nrms_shared_scale:.4f}", f"{t.amplification:.2f}",
    ] for t in terms]
    split_rows = [[
        row["leg"], row["tokens"], row["calls"], f"{row['gain']:.2f}",
        "not run" if row["nrms"] is None else f"{row['nrms']:.3f}",
        "not run" if row["per_leg"] is None else f"{row['per_leg']:.4f}",
    ] for row in cfg_split_rows(legs, cfg)]
    parts = [
        "# Chroma sharded numerics probe",
        "",
        *(["SYNTHETIC INPUT. Probe A ran on invented tensors, so its numbers "
           "test the arithmetic and say nothing about the model. Only a run "
           "on a real checkpoint says anything about the chroma nvfp4 gap.", ""]
          if synthetic else []),
        "## Probe A, per nvfp4 layer at step one (the chroma nvfp4 gap, docs/VALIDATION.md, 2026-09-09)",
        "",
        markdown_table(
            ["layer", "tokens", "amax whole", "amax min shard", "ratio",
             "nrms per-rank scale", "nrms shared scale", "amplification"],
            layer_rows) if layer_rows else "No layer was tapped.",
        "",
        *[f"* skipped {path}: {reason}" for path, reason in (skipped or [])],
        *[f"* {line}" for line in layer_verdicts(terms, asked_ratio)],
        "",
        f"## Probe B, the cfg combine at cfg {cfg:g} (the unequal-prompt term, docs/VALIDATION.md, 2026-09-06)",
        "",
        markdown_table(
            ["leg", "tokens", "model calls", "combine gain", "1-step nrms",
             "implied per-leg deviation"],
            split_rows),
        "",
        f"* {cfg_verdict(legs)}",
        "",
    ]
    return "\n".join(parts)


def synthetic_layers(count: int, *, tokens: int = 64, width: int = 64,
                     seed: int = 0) -> list[tuple[str, torch.Tensor]]:
    """Stand-in activations for the dry leg: real shapes, invented numbers.

    A Linear sees (batch, tokens, features), and the batch is 2 where comfy
    folded the pair into one call, so both shapes appear here. Every second
    layer carries one outlier token in its first half, the shape that makes two
    ranks disagree on the amax.
    """
    generator = torch.Generator().manual_seed(seed)
    layers = []
    for index in range(count):
        batch = 2 if index % 3 == 0 else 1  # a folded pair, and a single call
        activation = torch.randn(batch, tokens, width, generator=generator)
        if index % 2 == 0:
            activation[:, 0] *= 40.0
        layers.append((f"double_blocks.{index}.img_mlp.0", activation))
    return layers


def capture_layer_inputs(unet: str, te: str, prompt: str, negative: str, *,
                         steps: int, cfg: float, sampler: str, scheduler: str,
                         width: int, height: int, seed: int,
                         limit: int, te_type: str = "chroma") -> list[tuple[str, torch.Tensor]]:
    """Tap every shard-dependent Linear's step-one input on one GPU.

    Stock comfy, no mesh and no workers: the shard is emulated from the whole
    tensor afterwards, which is exactly what a ulysses rank holds for a Linear.
    This does not carry the attention reordering, which reaches later layers as
    a different input; that term needs the cross-box leg, which
    docs/VALIDATION.md lists as owed.

    The tap keeps each layer's first call. Chroma and krea2 publish their text
    conditioning (`c_crossattn`) as a `CONDRegular`, which folds only an equal
    pair. An unequal pair, such as the shipped 100 and 28 rows, runs two calls,
    cond first, so the 28-row stream is never tapped and the table prices the
    long stream. An equal pair runs one call, and the tapped tensor carries
    both legs in its batch.
    """
    sys.path.insert(0, str(REPO / "benchmark"))
    import comfy.sd
    import folder_paths
    import nodes
    from run_matrix import encode_prompts

    from dgx_monarch.adapters.quant_activation_scale import SHARD_DEPENDENT_LAYOUTS

    positive, negated = encode_prompts({"name": te, "type": te_type}, prompt, negative)
    path = folder_paths.get_full_path("diffusion_models", unet)
    if path is None:
        raise FileNotFoundError(f"diffusion model {unet!r} is in no ComfyUI diffusion_models folder")
    model = comfy.sd.load_diffusion_model(path)
    taps: dict[str, torch.Tensor] = {}
    handles = []

    def tap(name):
        def hook(_module, args, _output):
            if name not in taps and args and torch.is_tensor(args[0]):
                taps[name] = args[0].detach().to(device="cpu", dtype=torch.float32)
        return hook

    for name, module in model.model.diffusion_model.named_modules():
        # layout_type holds the layout's class name: the same test the
        # shared-scale hook's install walks the tree with.
        if getattr(module, "layout_type", None) in SHARD_DEPENDENT_LAYOUTS:
            handles.append(module.register_forward_hook(tap(name)))
    try:
        latent = {"samples": torch.zeros(1, 4, height // 8, width // 8)}
        nodes.common_ksampler(model, seed, steps, cfg, sampler, scheduler,
                              positive, negated, latent, denoise=1.0)
    finally:
        for handle in handles:
            handle.remove()
    ordered = list(taps.items())
    return ordered[:limit] if limit else ordered


def parse_reading(value: str) -> tuple[str, float]:
    name, _, number = value.partition("=")
    if not name or not number:
        raise argparse.ArgumentTypeError("write a reading as LEG=NRMS")
    return name, float(number)


def legs_with_readings(readings: list[tuple[str, float]]) -> list[CfgLeg]:
    known = {leg.name: leg for leg in CFG_LEGS}
    legs = list(CFG_LEGS)
    for name, value in readings:
        if name not in known:
            raise SystemExit(f"{name!r} is not a leg of the cfg matrix")
        index = legs.index(known[name])
        legs[index] = CfgLeg(name, known[name].folded, known[name].tokens, value)
    return legs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="use synthetic activations on the CPU, with no comfy and no GPU")
    parser.add_argument("--unet", default="", help="nvfp4 diffusion model to tap")
    parser.add_argument("--te", default="t5xxl_fp8_e4m3fn_scaled.safetensors")
    parser.add_argument("--te-type", default="chroma",
                        help="CLIPLoader type for --te: chroma for the chroma files, "
                             "krea2 for the control leg's qwen3vl encoder")
    parser.add_argument("--prompt", default="a close-up photograph of a tiger's eye")
    parser.add_argument("--negative", default="low quality, blurry, watermark")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="beta")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world", type=int, default=2, help="ulysses degree to emulate")
    parser.add_argument("--asked-ratio", type=float, default=None,
                        help="the factor A3 must reach: this family's sharded nvfp4 1-step NRMS "
                             "over its sharded bf16 one (default chroma's 0.120 over 0.032; "
                             "for the krea2 control leg, pass 0.011 divided by krea2's own sharded "
                             "bf16 reading)")
    parser.add_argument("--layers", type=int, default=8,
                        help="how many tapped layers to price; 0 prices every one, "
                             "and the dry run builds at least one")
    parser.add_argument("--nrms", type=parse_reading, action="append", default=[],
                        metavar="LEG=NRMS", help="a cfg matrix leg's 1-step NRMS; repeat for each leg")
    parser.add_argument("--out", default="", help="also write the note to this file")
    parser.add_argument("--json", default="", help="write the rows to this file as JSON")
    args = parser.parse_args(argv)

    if args.dry_run == bool(args.unet):
        print("pass exactly one of --dry-run and --unet", file=sys.stderr)
        return 2
    if args.dry_run:
        layers = synthetic_layers(max(args.layers, 1) or 8, seed=args.seed)
    else:
        layers = capture_layer_inputs(
            args.unet, args.te, args.prompt, args.negative, steps=args.steps,
            cfg=args.cfg, sampler=args.sampler, scheduler=args.scheduler,
            width=args.width, height=args.height, seed=args.seed,
            limit=args.layers, te_type=args.te_type)

    generator = torch.Generator().manual_seed(args.seed)
    terms, skipped = price_layers(layers, world=args.world, generator=generator)
    legs = legs_with_readings(args.nrms)
    note = report(terms, legs, args.cfg, synthetic=args.dry_run,
                  asked_ratio=args.asked_ratio, skipped=skipped)
    print(note)
    if args.out:
        Path(args.out).write_text(note + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "layers": [asdict(term) for term in terms],
            "skipped": [{"layer": path, "reason": reason} for path, reason in skipped],
            "cfg_legs": cfg_split_rows(legs, args.cfg),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
