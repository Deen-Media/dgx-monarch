"""CPU slab-hook canary: the comfy surfaces slab_weights patches depend on.

comfy_bridge.slab_load() monkeypatches two comfy call sites and relies on
several structural behaviors of comfy's loader. None of them are covered by
comfy's own API stability promises, so a nightly rename would break the
zero-copy load at render time. This canary fails first, against whatever
comfy is on sys.path (comfy master in the comfy-canary workflow):

  1. comfy.utils.load_torch_file keeps its (ckpt, safe_load, device,
     return_metadata) surface, which is the interception target,
  2. BaseModel.load_model_weights keeps its `assign` kwarg and
     load_diffusion_model_state_dict still routes through it and still reads
     `offload_device` from model_options, with comfy's default when the key is
     absent. The pack never passes that key:
     comfy_bridge.load_diffusion_model_slab overwrites the patcher's
     offload_device with the GPU after the load,
  3. torch's Module.load_state_dict keeps `assign`,
  4. comfy.quant_ops keeps QUANT_ALGOS entries (storage_t +
     comfy_tensor_layout) and get_layout_class/QuantizedTensor for every
     restorable format, which is the quant reabsorb/rebuild recipe,
  5. comfy.utils.convert_old_quants keeps the exact legacy scaled-FP8
     conversion that file-backed un-bake mirrors,
  6. functional: current per-layer markers capture and restore exactly through
     Comfy's real mixed-precision Linear implementation: FP8, and
     int8-convrot in both spellings Comfy reads (flat and under `params`),
     with a marker naming anything the restore cannot replay refused,
  7. functional: slab_load() over a real (tiny) safetensors file returns
     slab-backed tensors through comfy's own load_torch_file, forces
     assign=True, and restores both patches on exit.

Runs in the comfy-canary workflow and as a pytest wherever comfy is
importable (COMFY_DIR overrides the location).
"""
import inspect
import json
import os
import struct
import sys
import tempfile
from types import SimpleNamespace


def _ensure_comfy():
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR", "../ComfyUI"))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    sys.argv = ["slab-hook-canary", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()


def main() -> None:
    _ensure_comfy()
    import comfy.model_base
    import comfy.ops
    import comfy.quant_ops
    import comfy.sd
    import comfy.utils
    import torch
    from safetensors.torch import save_file

    # 1. the load_torch_file interception target
    params = inspect.signature(comfy.utils.load_torch_file).parameters
    for name in ("ckpt", "safe_load", "device", "return_metadata"):
        assert name in params, f"load_torch_file lost param {name!r}"

    # 2. the assign path and the offload_device model_options key
    lmw = inspect.signature(comfy.model_base.BaseModel.load_model_weights)
    assert "assign" in lmw.parameters, "BaseModel.load_model_weights lost `assign`"
    src = inspect.getsource(comfy.sd.load_diffusion_model_state_dict)
    assert "load_model_weights(" in src, \
        "load_diffusion_model_state_dict no longer routes load_model_weights"
    assert 'model_options.get("offload_device"' in src, \
        "load_diffusion_model_state_dict no longer honors offload_device"

    # 3. torch's assign kwarg, which comfy passes the forced assign to
    lsd = inspect.signature(torch.nn.Module.load_state_dict)
    assert "assign" in lsd.parameters, "torch Module.load_state_dict lost `assign`"

    # 4. the quant rebuild surface (reabsorb and lazy un-bake recipe)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    from dgx_monarch.actor.unbake import (
        RESTORABLE_QUANT_FORMATS,
        CaptureAborted,
        capture_unbake_record,
        restore_pristine,
    )

    quant_checked = 0
    for fmt in RESTORABLE_QUANT_FORMATS:
        algo = comfy.quant_ops.QUANT_ALGOS.get(fmt)
        assert algo is not None, f"comfy lost QUANT_ALGOS[{fmt!r}]"
        for field in ("storage_t", "comfy_tensor_layout"):
            assert field in algo, f"QUANT_ALGOS[{fmt!r}] lost {field!r}"
        comfy.quant_ops.get_layout_class(algo["comfy_tensor_layout"])
        quant_checked += 1
    if quant_checked == 0:
        raise SystemExit("RESTORABLE_QUANT_FORMATS is empty; quant surface canary is vacuous")
    assert hasattr(comfy.quant_ops, "QuantizedTensor"), "comfy lost QuantizedTensor"
    # the quant reabsorb (actor/slab.py) reads _qdata, _layout_cls and _params
    # off a wrapper and rebuilds it through the (qdata, layout_cls, params) ctor
    qt_init = inspect.signature(comfy.quant_ops.QuantizedTensor.__init__)
    for pname in ("qdata", "layout_cls", "params"):
        assert pname in qt_init.parameters, f"QuantizedTensor ctor lost {pname!r}"
    qt_src = inspect.getsource(comfy.quant_ops.QuantizedTensor.__init__)
    for attr in ("_qdata", "_layout_cls", "_params"):
        assert attr in qt_src, f"QuantizedTensor no longer stores {attr}"

    # 5. the exact legacy scaled-FP8 conversion that actor/unbake_file.py mirrors
    #    (actor/unbake.py maps weight_scale back to the file's scale_weight)
    convert_params = inspect.signature(comfy.utils.convert_old_quants).parameters
    for name in ("state_dict", "model_prefix", "metadata"):
        assert name in convert_params, f"convert_old_quants lost param {name!r}"

    def marker_conf(value):
        return json.loads(bytes(value.detach().cpu().tolist()))

    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(
        torch.float8_e4m3fn
    )
    scale = torch.tensor(0.25, dtype=torch.float32)
    root, _ = comfy.utils.convert_old_quants(
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float32),
            "blk.weight": weight.clone(),
            "blk.scale_weight": scale.clone(),
        },
        model_prefix="",
        metadata={},
    )
    assert "scaled_fp8" not in root and "blk.scale_weight" not in root
    assert torch.equal(root["blk.weight_scale"], scale)
    assert marker_conf(root["blk.comfy_quant"]) == {
        "format": "float8_e4m3fn"
    }

    prefix = "model.diffusion_model."
    prefixed, _ = comfy.utils.convert_old_quants(
        {
            f"{prefix}scaled_fp8": torch.ones(
                2, dtype=torch.float8_e4m3fn
            ),
            f"{prefix}blk.weight": weight.clone(),
            f"{prefix}blk.scale_weight": scale.clone(),
        },
        model_prefix=prefix,
        metadata={},
    )
    assert marker_conf(prefixed[f"{prefix}blk.comfy_quant"]) == {
        "format": "float8_e4m3fn",
        "full_precision_matrix_mult": True,
    }
    assert torch.equal(prefixed[f"{prefix}blk.weight_scale"], scale)

    embedded_conf = {"layers": {"blk": {"format": "float8_e5m2"}}}
    embedded, _ = comfy.utils.convert_old_quants(
        {
            "scaled_fp8": torch.ones(1, dtype=torch.float8_e4m3fn),
            "blk.weight": weight.clone(),
            "blk.scale_weight": scale.clone(),
        },
        model_prefix="",
        metadata={"_quantization_metadata": json.dumps(embedded_conf)},
    )
    assert "blk.scale_weight" in embedded and "blk.weight_scale" not in embedded
    assert marker_conf(embedded["blk.comfy_quant"]) == embedded_conf["layers"]["blk"]

    # 6. direct FP8 marker capture and restore through real Comfy ops
    direct_conf = {
        "format": "float8_e4m3fn",
        "full_precision_matrix_mult": True,
    }
    direct_marker = torch.tensor(
        list(json.dumps(direct_conf).encode("utf-8")),
        dtype=torch.uint8,
    )
    operations = comfy.ops.mixed_precision_ops(
        {"mixed_ops": True},
        torch.bfloat16,
    )
    module = operations.Linear(
        4,
        4,
        bias=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    module.load_state_dict(
        {
            "weight": weight.clone(),
            "weight_scale": scale.clone(),
            "comfy_quant": direct_marker.clone(),
        },
        strict=True,
    )
    model = SimpleNamespace(
        diffusion_model=SimpleNamespace(blk=module),
    )
    before = module.state_dict()
    with tempfile.TemporaryDirectory() as directory:
        direct_path = os.path.join(directory, "direct.safetensors")
        save_file(
            {
                "blk.weight": weight.clone(),
                "blk.weight_scale": scale.clone(),
                "blk.comfy_quant": direct_marker.clone(),
            },
            direct_path,
        )
        record = capture_unbake_record(
            model,
            ["diffusion_model.blk.weight"],
            direct_path,
        )
        restored = restore_pristine(model, record)
    after = module.state_dict()
    spec = record.quant["diffusion_model.blk.weight"]
    assert set(spec.components) == {"weight", "weight_scale"}
    assert spec.full_precision_mm_config is True
    assert restored["restored_keys"] == 1
    assert torch.equal(
        before["weight"].view(torch.uint8),
        after["weight"].view(torch.uint8),
    )
    assert torch.equal(before["weight_scale"], after["weight_scale"])

    # 6b. direct int8-convrot marker: the pruned MiniMax H3 spelling, where
    # the per-layer marker is the whole quantization authority. Comfy reads
    # the two convrot settings flat or under `params` and re-emits the live
    # marker flat either way; the capture gate's closed allowlist and
    # _restore_quant's Params kwargs are written against that, so both
    # spellings run here against real Comfy, not a stub.
    int8_weight = (
        (torch.arange(256 * 256, dtype=torch.int32) % 255) - 127
    ).to(torch.int8).reshape(256, 256)
    int8_scale = torch.linspace(0.5, 1.5, 256, dtype=torch.float32).reshape(256, 1)
    flat_conf = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    }

    def int8_convrot_roundtrip(conf, name, directory):
        marker = torch.tensor(
            list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8
        )
        module = comfy.ops.mixed_precision_ops(
            {"mixed_ops": True},
            torch.bfloat16,
        ).Linear(
            256,
            256,
            bias=False,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        module.load_state_dict(
            {
                "weight": int8_weight.clone(),
                "weight_scale": int8_scale.clone(),
                "comfy_quant": marker.clone(),
            },
            strict=True,
        )
        assert marker_conf(module.state_dict()["comfy_quant"]) == flat_conf, \
            f"comfy no longer re-emits the {name} int8-convrot marker flat"
        for setting in ("convrot", "convrot_groupsize"):
            assert hasattr(module.weight._params, setting), \
                f"int8 layout Params lost {setting!r}"
        path = os.path.join(directory, f"int8-{name}.safetensors")
        save_file(
            {
                "blk.weight": int8_weight.clone(),
                "blk.weight_scale": int8_scale.clone(),
                "blk.comfy_quant": marker.clone(),
            },
            path,
        )
        holder = SimpleNamespace(diffusion_model=SimpleNamespace(blk=module))
        record = capture_unbake_record(
            holder, ["diffusion_model.blk.weight"], path
        )
        spec = record.quant["diffusion_model.blk.weight"]
        assert set(spec.components) == {"weight", "weight_scale"}, \
            f"{name} int8-convrot capture did not bind both components"
        assert not spec.resident, f"{name} int8-convrot capture held resident bytes"
        # a bake requantizes data and the per-row scales: dirty both, so the
        # restore has to fetch the checkpoint's own bytes to come back equal
        module.weight._qdata.fill_(1)
        module.weight._params.scale.fill_(9.0)
        restored = restore_pristine(holder, record)
        after_int8 = module.state_dict()
        assert restored["restored_keys"] == 1
        assert torch.equal(
            after_int8["weight"].view(torch.uint8),
            int8_weight.view(torch.uint8),
        ), f"{name} int8-convrot restore did not replay the checkpoint bytes"
        assert torch.equal(after_int8["weight_scale"], int8_scale), \
            f"{name} int8-convrot restore did not replay the per-row scales"
        assert marker_conf(after_int8["comfy_quant"]) == flat_conf, \
            f"{name} int8-convrot restore changed the live marker"
        for setting, value in (("convrot", True), ("convrot_groupsize", 256)):
            assert getattr(module.weight._params, setting) == value, \
                f"{name} int8-convrot restore lost {setting!r}"

    with tempfile.TemporaryDirectory() as directory:
        int8_convrot_roundtrip(flat_conf, "flat", directory)
        int8_convrot_roundtrip(
            {"format": "int8_tensorwise",
             "params": {"convrot": True, "convrot_groupsize": 256}},
            "nested",
            directory,
        )
        # and a setting no restore replays still falls back to reload
        try:
            int8_convrot_roundtrip(
                {**flat_conf, "per_row": True}, "per-row", directory
            )
        except CaptureAborted as exc:
            assert "'per_row'" in str(exc), \
                f"the refusal no longer names the key that caused it: {exc}"
        else:
            raise SystemExit(
                "an int8 marker naming 'per_row' was captured instead of refused"
            )

    # 7. functional: slab_load over comfy's own load_torch_file, CPU
    from dgx_monarch.actor.comfy_bridge import slab_load

    t = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    header = {"w": {"dtype": "F32", "shape": [8, 8],
                    "data_offsets": [0, t.numel() * 4]},
              "__metadata__": {"canary": "1"}}
    hj = json.dumps(header).encode()
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(struct.pack("<Q", len(hj)) + hj)
        f.write(t.numpy().tobytes())
        path = f.name
    try:
        orig_ltf = comfy.utils.load_torch_file
        orig_lmw = comfy.model_base.BaseModel.load_model_weights
        with slab_load(path) as slab:
            sd, md = comfy.utils.load_torch_file(path, return_metadata=True)
            assert md == {"canary": "1"}, "metadata lost through the slab loader"
            assert slab.contains(sd["w"].data_ptr()), "state_dict not slab-backed"
            assert torch.equal(sd["w"].cpu(), t), "slab bytes diverge"
            # the assign forcing is asserted structurally: the installed hook
            # must forward with assign=True (running it needs a full
            # model_config, out of canary scope)
            hook_src = inspect.getsource(
                comfy.model_base.BaseModel.load_model_weights)
            assert "assign=True" in hook_src, "slab_load did not force assign"
        assert comfy.utils.load_torch_file is orig_ltf, "load_torch_file not restored"
        assert comfy.model_base.BaseModel.load_model_weights is orig_lmw, \
            "load_model_weights not restored"
        del sd
        slab.close()
    finally:
        os.unlink(path)

    print("slab-hook canary green")


if __name__ == "__main__":
    main()
