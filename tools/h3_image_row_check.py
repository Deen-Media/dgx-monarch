#!/usr/bin/env python3
"""Record whether a MiniMax H3 block's row-keyed calls give stock's rows at one rank's rows.

Under uly2 each rank runs a block on its share of the packed rows (about 700 a
rank at 1024 square for MiniMax-H3-Image), where stock runs the whole stream.
A call whose kernel picks its tiling, its split or its activation scale by the
row count can round a row differently at the shard's size. This tool runs each
row-keyed call of block 0 on the full stream and on every rank's rows (the tail
rank carries the divisibility pad rows, zeros, as the shard does) and compares
the rank's real rows bit for bit with the same rows of the full call:

- ``qkv_proj`` and ``out_proj``: the attention projections (INT8 ConvRot or
  W4A8 GEMMs in the quantized files);
- ``rms_rope_split_half_``: comfy-kitchen's fused q/k RMS norm and rope;
- ``mlp``: ``linear_input_act(fc2, fc1(x), "swiglu")``, the fused MLP.

The canary seam "minimax h3 image row calls" holds that these are the calls a
block makes. ``--checkpoint`` loads a real file through ComfyUI's own loader
(run it on the hardware it is to vouch for); ``--tiny`` builds a small random
model and always runs on CPU. ``--self-check`` adds a negative control, a call
whose rows depend on every row it sees, which the comparison must report
unequal; the tool exits 2 if it does not. The record is JSON: one row per call and rank with
``equal`` and ``max_abs_diff``, plus the shape and device it ran on.

Usage:
  python tools/h3_image_row_check.py --comfy <ComfyUI dir> --checkpoint <dit.safetensors> \
      --latent 64 64 --text-rows 300 --world 2 --out row-check.json
  python tools/h3_image_row_check.py --comfy <ComfyUI dir> --tiny --out smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _comfy(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)
    sys.argv = [sys.argv[0], "--cpu"] if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else [sys.argv[0]]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.ldm.minimax.model as h3

    return h3


def _tiny_block(h3, device):
    import comfy.ops
    import torch

    torch.manual_seed(0)
    model = h3.MiniMaxH3ImageModel(
        pdd_sigmas=[1.0, 0.5, 0.0], pdd_steps=2, hidden_size=48, num_layers=1,
        token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=24,
        ffn_hidden_size=64, text_dim=40, time_embed_dim=8, rope_inv_freq_len=4,
        adaln_curve_grid=1025, dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if name.endswith("inv_freq"):
                tensor.copy_(1.0 / (10000 ** (torch.arange(tensor.numel()) / tensor.numel())))
            else:
                tensor.copy_(0.2 * torch.randn_like(tensor))
    return model.requires_grad_(False).to(device), torch.float32


def _checkpoint_block(path: str, device):
    import comfy.model_management
    import comfy.sd

    patcher = comfy.sd.load_diffusion_model(path)
    comfy.model_management.load_models_gpu([patcher])
    model = patcher.model.diffusion_model
    return model, patcher.model.get_dtype_inference()


def _shards(rows: int, world: int) -> list[tuple[int, int]]:
    local = -(-rows // world)
    return [(rank * local, min((rank + 1) * local, rows)) for rank in range(world)]


def _local(x, start: int, stop: int, local: int):
    import torch

    part = x[start:stop]
    if part.shape[0] == local:
        return part
    pad = torch.zeros((local - part.shape[0], *part.shape[1:]), dtype=x.dtype, device=x.device)
    return torch.cat([part, pad])


def run(h3, model, dtype, latent: tuple[int, int], text_rows: int, world: int,
        self_check: bool = False) -> dict:
    import comfy.quant_ops
    import torch

    device = next(model.parameters()).device
    layout = h3.PackedLayout(text_rows, 1, latent[0], latent[1], 0)
    rows = layout.seq_len
    local = -(-rows // world)
    block = model.blocks[0]
    attn = block.attn
    inner = attn.heads * attn.head_dim
    generator = torch.Generator(device="cpu").manual_seed(5)
    x = torch.randn(rows, model.hidden_size, generator=generator).to(device=device, dtype=dtype)
    attended = torch.randn(rows, inner, generator=generator).to(device=device, dtype=dtype)
    # The shard builds its rope table from its own positions, pad rows at zero.
    angles = model.rope_freqs(layout.position_ids, device)
    qw = attn.q_norm.weight.to(device)
    kw = attn.k_norm.weight.to(device)

    def rms_rope(hidden, part_angles):
        table = h3.rope_rotation_table(part_angles, dtype)
        s = hidden.shape[0]
        # As stock: in place on the strided views of the qkv buffer.
        q, k, _v = attn.qkv_proj(hidden).split(inner, dim=-1)
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, table, qw, kw, epsilon=attn.q_norm.eps,
                                                rot_dim=table.shape[-3] * 2)
        return torch.cat([q[0].reshape(s, -1), k[0].reshape(s, -1)], dim=-1)

    calls = {
        "qkv_proj": (x, lambda t, _a: attn.qkv_proj(t)),
        "rms_rope_split_half_": (x, rms_rope),
        "out_proj": (attended, lambda t, _a: attn.out_proj(t)),
        "mlp": (x, lambda t, _a: block.mlp(t)),
    }
    if self_check:
        # A negative control: a call whose rows depend on every row it sees, so the
        # comparison must report it unequal on some rank.
        calls["self_check_row_dependent"] = (x, lambda t, _a: t / t.abs().amax())
    record = {"rows": rows, "world": world, "local_rows": local, "latent": list(latent),
              "text_rows": text_rows, "device": str(device), "dtype": str(dtype), "calls": []}
    with torch.inference_mode():
        for name, (source, call) in calls.items():
            full = call(source, angles)
            for rank, (start, stop) in enumerate(_shards(rows, world)):
                part = call(_local(source, start, stop, local), _local(angles, start, stop, local))
                real = stop - start
                ours, stock = part[:real], full[start:stop]
                diff = (ours.float() - stock.float()).abs().max().item() if real else 0.0
                record["calls"].append({"call": name, "rank": rank, "real_rows": real,
                                        "equal": bool(torch.equal(ours, stock)),
                                        "max_abs_diff": diff})
    record["all_equal"] = all(row["equal"] for row in record["calls"]
                              if row["call"] != "self_check_row_dependent")
    if self_check:
        record["self_check_seen"] = not all(
            row["equal"] for row in record["calls"] if row["call"] == "self_check_row_dependent")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--comfy", required=True, help="the ComfyUI checkout to import")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="a MiniMax H3 or MiniMax-H3-Image DiT file")
    source.add_argument("--tiny", action="store_true", help="a small random model (CPU smoke)")
    parser.add_argument("--latent", type=int, nargs=2, default=(64, 64), metavar=("H", "W"),
                        help="the latent size, even (1024 square is 64 64)")
    parser.add_argument("--text-rows", type=int, default=300)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--self-check", action="store_true",
                        help="add a row-dependent call the comparison must report unequal")
    parser.add_argument("--out", required=True, help="the JSON record to write")
    args = parser.parse_args(argv)
    if args.tiny:
        # The smoke runs on CPU even on a GPU host.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    h3 = _comfy(os.path.abspath(args.comfy))
    import torch

    if args.tiny:
        model, dtype = _tiny_block(h3, torch.device("cpu"))
    else:
        model, dtype = _checkpoint_block(args.checkpoint, None)
    with torch.inference_mode():
        record = run(h3, model.requires_grad_(False), dtype, tuple(args.latent),
                     args.text_rows, args.world, self_check=args.self_check)
    record["source"] = "tiny" if args.tiny else Path(args.checkpoint).name
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(f"{'all calls equal' if record['all_equal'] else 'a call differs'}: {args.out}")
    if args.self_check and not record["self_check_seen"]:
        print("the negative control compared equal: the comparison is broken")
        return 2
    return 0 if record["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
