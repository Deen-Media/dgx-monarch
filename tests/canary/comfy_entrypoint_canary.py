#!/usr/bin/env python3
"""Import the repository the way ComfyUI imports a custom-node pack.

Loading the repository-root ``__init__.py``, not only ``dgx_monarch.nodes``,
covers the packaging bridge, fault-hook installation, WEB_DIRECTORY, browser
assets and src-layout package discovery in one run. The daily comfy-canary
workflow runs it on CPU against ComfyUI master.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
COMFY_DIR = Path(os.environ.get("COMFYUI_DIR", "../ComfyUI")).expanduser().resolve()


def main() -> None:
    sys.argv = ["dgxm-comfy-entrypoint-canary", "--cpu"]
    sys.path.insert(0, str(COMFY_DIR))

    # Comfy ignores --cpu unless parsing is enabled before comfy.cli_args is
    # first imported (main.py does this for a real server process).
    import comfy.options

    comfy.options.enable_args_parsing()

    # A headless import has no PromptServer.instance and registers no routes.
    # Install an unstarted instance with aiohttp's route table to check route
    # registration and the instance's send_sync signature.
    import server
    from aiohttp import web

    missing = object()
    previous_instance = getattr(server.PromptServer, "instance", missing)
    prompt_server = _headless_prompt_server(server.PromptServer, web.RouteTableDef)
    server.PromptServer.instance = prompt_server
    real_gethostname = socket.gethostname
    socket.gethostname = lambda: "comfy-canary"
    try:
        _assert_entrypoint(prompt_server)
    finally:
        socket.gethostname = real_gethostname
        if previous_instance is missing:
            del server.PromptServer.instance
        else:
            server.PromptServer.instance = previous_instance


def _assert_entrypoint(prompt_server) -> None:
    module_name = "dgx_monarch_custom_node_canary"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO / "__init__.py",
        submodule_search_locations=[str(REPO)],
    )
    assert spec and spec.loader, "could not construct custom-node entrypoint spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    assert module.NODE_CLASS_MAPPINGS, "custom-node entrypoint exported no nodes"
    assert set(module.NODE_CLASS_MAPPINGS) == set(module.NODE_DISPLAY_NAME_MAPPINGS)
    assert module.WEB_DIRECTORY == "./web/js"

    web_dir = (REPO / module.WEB_DIRECTORY).resolve()
    assert web_dir.is_dir(), f"WEB_DIRECTORY does not exist: {web_dir}"
    expected_js = {
        "dgx_monarch.js",
        "dgx_monarch_consent.js",
        "dgx_monarch_consent_model.js",
        "dgx_monarch_panel.js",
        "dgx_monarch_readiness_model.js",
        "dgx_monarch_segments.js",
        "dgx_monarch_timeline.js",
        "dgx_monarch_timeline_model.js",
        "dgx_monarch_ui.js",
        "dgx_monarch_widgets.js",
    }
    assert expected_js <= {path.name for path in web_dir.glob("*.js")}

    package_spec = importlib.util.find_spec("dgx_monarch")
    assert package_spec and package_spec.origin
    package_path = Path(package_spec.origin).resolve()
    assert package_path.is_relative_to(REPO / "src"), (
        f"entrypoint imported dgx_monarch from outside this checkout: {package_path}"
    )

    from dgx_monarch.comfy_surface import assert_comfy_surface

    touchpoints_checked = assert_comfy_surface()
    _assert_model_sampling_clone_lifecycle()
    _assert_real_nested_latent_contract()
    assert getattr(prompt_server, "_dgxm_routes", False), (
        "custom-node entrypoint did not register its PromptServer routes"
    )
    assert len(prompt_server.on_prompt_handlers) == 1, (
        "custom-node entrypoint did not register the queue-time graph advisor"
    )

    for name, cls in module.NODE_CLASS_MAPPINGS.items():
        cls.INPUT_TYPES()
        print(f"schema ok: {name}")
    print(
        f"custom-node entrypoint green: root={REPO} package={package_path} "
        f"web={web_dir} comfy_touchpoints={touchpoints_checked}"
    )


def _headless_prompt_server(prompt_server_type, route_table_def):
    """Build only the existing-instance surface this node pack consumes.

    The pack registers routes and a queue-time prompt handler on the existing
    ``PromptServer.instance`` and sends events through its ``send_sync``. Do not
    call ``PromptServer.__init__``: it sets up managers, user directories and the
    frontend, side effects this CPU import check must not have.
    """
    prompt_server = object.__new__(prompt_server_type)
    prompt_server.routes = route_table_def()
    # PromptServer.__init__ creates this list; the queue-time graph advisor
    # appends to it through the real add_on_prompt_handler.
    prompt_server.on_prompt_handlers = []
    return prompt_server


def _assert_model_sampling_clone_lifecycle() -> None:
    """Exercise the real Comfy object-patch handoff that prevents shift leaks."""
    import torch
    from comfy.model_patcher import ModelPatcher
    from comfy.model_sampling import ModelSamplingDiscreteFlow

    from dgx_monarch.actor.sampling import model_sampling_render_clone

    class MinimalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_config = SimpleNamespace(sampling_settings={})
            self.model_sampling = ModelSamplingDiscreteFlow(self.model_config)
            self.device = torch.device("cpu")

    model = MinimalModel()
    stock = model.model_sampling
    resident = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"), size=1)

    def activate(value):
        patcher = model_sampling_render_clone(resident, value)
        patcher.patch_model(load_weights=False)
        return patcher

    shifted = activate({"kind": "sd3", "shift": 5.0})
    assert model.model_sampling.shift == 5.0
    shifted.detach(unpatch_all=False)
    assert model.model_sampling.shift == 5.0

    default = activate(None)
    assert default.get_model_object("model_sampling") is stock
    assert model.model_sampling is stock
    default.detach(unpatch_all=False)

    shifted = activate({"kind": "sd3", "shift": 3.0})
    assert model.model_sampling.shift == 3.0
    shifted.detach(unpatch_all=False)
    default = activate(None)
    assert model.model_sampling is stock
    assert resident.object_patches == {}
    assert model.model_loaded_weight_memory == 0
    default.unpatch_model(unpatch_weights=False)
    print("model sampling clone lifecycle ok")


def _assert_real_nested_latent_contract() -> None:
    """Run Comfy's installed NestedTensor through the output, message, transaction and zero-step seams."""
    import pickle

    import torch
    from comfy.nested_tensor import NestedTensor

    from dgx_monarch.actor.latent_outputs import zero_step_outputs
    from dgx_monarch.actor.sampling import (
        _custom_denoised_output,
        _normalize_sample_output,
    )
    from dgx_monarch.nodes.gate_identity import (
        copy_transaction,
        freeze_transaction,
        require_transaction_unchanged,
        transaction_tensor_versions,
    )
    from dgx_monarch.nodes.latent_identity import compare_latents
    from dgx_monarch.sampling_contract import custom_schedule_steps
    from dgx_monarch.transfer import LatentReturn, read_latent_result

    source = NestedTensor((
        torch.arange(8, dtype=torch.float32)
        .reshape(1, 2, 4)[:, :, ::2]
        .requires_grad_(),
        torch.arange(6, dtype=torch.float32)
        .reshape(1, 3, 2)[:, :, 0]
        .requires_grad_(),
    ))
    normalized = _normalize_sample_output(source, "Comfy canary packed output")
    parts = tuple(normalized.unbind())
    assert type(normalized) is NestedTensor
    assert [tuple(part.shape) for part in parts] == [(1, 2, 2), (1, 3)]
    assert all(
        part.device.type == "cpu" and part.is_contiguous() and not part.requires_grad
        for part in parts
    )

    latent_return = LatentReturn("rdma", min_bytes=1)
    descriptor = latent_return.pack("samples", normalized)
    # Trusted canary object: only simulate the actor serialization boundary.
    wire_descriptor = pickle.loads(pickle.dumps(descriptor))  # noqa: S301
    restored = read_latent_result(wire_descriptor)
    assert descriptor["kind"] == "message"
    assert type(restored) is NestedTensor
    assert latent_return._keepalive == {}
    assert compare_latents(normalized, restored) == (True, 0.0)
    divergent = NestedTensor((parts[0].clone(), parts[1] + 1))
    assert compare_latents(normalized, divergent) == (False, 1.0)

    frozen = freeze_transaction({"samples": normalized})
    versions = transaction_tensor_versions(frozen)
    frozen_leg = copy_transaction(frozen)
    frozen_audio = frozen_leg["samples"].unbind()[1]
    frozen_audio_version = int(getattr(frozen_audio, "_version", 0))
    frozen_audio.data.add_(1)
    assert int(getattr(frozen_audio, "_version", 0)) == frozen_audio_version
    try:
        require_transaction_unchanged(versions)
    except RuntimeError as exc:
        assert "mutated a captured" in str(exc)
    else:
        raise AssertionError(
            "production transaction guard accepted version-bypassing audio mutation"
        )

    assert custom_schedule_steps(torch.tensor([1.0])) == 0
    zero_primary, zero_denoised = zero_step_outputs(
        normalized, _normalize_sample_output
    )
    assert compare_latents(normalized, zero_primary) == (True, 0.0)
    assert compare_latents(normalized, zero_denoised) == (True, 0.0)
    zero_primary_parts = tuple(zero_primary.unbind())
    zero_denoised_parts = tuple(zero_denoised.unbind())
    assert all(
        primary.untyped_storage()._cdata != preview.untyped_storage()._cdata
        for primary, preview in zip(
            zero_primary_parts, zero_denoised_parts, strict=True
        )
    )
    assert all(
        original.untyped_storage()._cdata
        not in {
            primary.untyped_storage()._cdata,
            preview.untyped_storage()._cdata,
        }
        for original, primary, preview in zip(
            parts, zero_primary_parts, zero_denoised_parts, strict=True
        )
    )

    class IdentityLatentModel:
        @staticmethod
        def process_latent_out(value):
            return value

    guider = SimpleNamespace(
        model_patcher=SimpleNamespace(model=IdentityLatentModel())
    )
    packed_x0 = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
    denoised = _custom_denoised_output(guider, normalized, packed_x0)
    denoised_parts = tuple(denoised.unbind())
    assert type(denoised) is NestedTensor
    assert [tuple(part.shape) for part in denoised_parts] == [(1, 2, 2), (1, 3)]
    assert [part.dtype for part in denoised_parts] == [torch.float32, torch.float32]
    assert torch.equal(
        denoised_parts[0], torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    )
    assert torch.equal(
        denoised_parts[1], torch.arange(4, 7, dtype=torch.float32).reshape(1, 3)
    )
    denoised_storage = [part.untyped_storage().data_ptr() for part in denoised_parts]
    assert len(set(denoised_storage)) == len(denoised_storage)
    assert all(
        pointer != packed_x0.untyped_storage().data_ptr()
        for pointer in denoised_storage
    )
    assert all(
        denoised_part.untyped_storage().data_ptr()
        != sample_part.untyped_storage().data_ptr()
        for denoised_part, sample_part in zip(
            denoised_parts, parts, strict=True
        )
    )

    malformed_x0 = torch.zeros((1, 1, 8), dtype=torch.float32)
    try:
        _custom_denoised_output(guider, normalized, malformed_x0)
    except RuntimeError as exc:
        assert "malformed packed denoised shape" in str(exc)
    else:
        raise AssertionError(
            "production custom-denoised output accepted malformed packed width"
        )

    # The other x0 shape, and the one a packed render gets: comfy wraps the
    # step callback for every latent that packs more than one modality and
    # hands it the nested view. The contract is the arithmetic stock
    # process_latent_out applies to that view, so the stub model's `/ 0.5`
    # below runs through the installed NestedTensor's own operators.
    class ScaledLatentModel:
        @staticmethod
        def process_latent_out(value):
            return value / 0.5

    scaled_guider = SimpleNamespace(
        model_patcher=SimpleNamespace(model=ScaledLatentModel())
    )
    nested_x0 = NestedTensor(tuple(part.clone() for part in parts))
    nested_denoised = _custom_denoised_output(scaled_guider, normalized, nested_x0)
    nested_parts = tuple(nested_denoised.unbind())
    assert type(nested_denoised) is NestedTensor
    assert [tuple(part.shape) for part in nested_parts] == [(1, 2, 2), (1, 3)]
    assert [part.dtype for part in nested_parts] == [torch.float32, torch.float32]
    assert all(
        torch.equal(rebuilt, sampled * 2)
        for rebuilt, sampled in zip(nested_parts, parts, strict=True)
    )
    assert all(
        rebuilt.untyped_storage().data_ptr() != sampled.untyped_storage().data_ptr()
        for rebuilt, sampled in zip(nested_parts, parts, strict=True)
    )

    dropped_x0 = NestedTensor((parts[0].clone(),))
    try:
        _custom_denoised_output(scaled_guider, normalized, dropped_x0)
    except RuntimeError as exc:
        assert "wrong number of NestedTensor modalities" in str(exc)
    else:
        raise AssertionError(
            "production custom-denoised output accepted a short nested x0"
        )
    print(
        "real Comfy NestedTensor transaction/zero-step/message/output contract ok"
    )


if __name__ == "__main__":
    main()
