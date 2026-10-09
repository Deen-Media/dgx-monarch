"""Node-surface contract snapshot (DESIGN.md §5.7).

NODE_CLASS_MAPPINGS keys, input names and defaults are semver-governed API:
saved workflows must keep loading across releases. This file pins the keys and,
per node, the ordered input names, Comfy types and effective defaults
(node_input_contract.json), so a widget shift fails here; building INPUT_TYPES,
as the daily Comfy canary's entrypoint check does, cannot catch one.
"""
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from dgx_monarch.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

EXPECTED_NODE_KEYS = {
    "DGXMonarchInit",
    "DGXMonarchUNETLoader",
    "DGXMonarchUncondUNETLoader",
    "DGXMonarchLoraLoader",
    "DGXMonarchModelSamplingSD3",
    "DGXMonarchQwenImage21Cache",
    "DGXMonarchKSampler",
    "DGXMonarchKSamplerAdvanced",
    "DGXMonarchKSamplerPipeline",
    "DGXMonarchSamplerCustom",
    "DGXMonarchBasicScheduler",
    "DGXMonarchBasicGuider",
    "DGXMonarchCFGGuider",
    "DGXMonarchDualModelGuider",
    "DGXMonarchIdentityGate",
    "DGXMonarchFleetKSampler",
    "DGXMonarchStatus",
    "DGXMonarchClearVRAM",
}

EXPECTED_INPUT_CONTRACT = json.loads(
    (Path(__file__).with_name("node_input_contract.json")).read_text()
)
_DYNAMIC_COMBOS = {"unet_name", "lora_name", "sampler_name", "scheduler"}


@contextmanager
def _comfy_surface_stubs():
    """Provide fixed dropdown sources, then restore module state.

    Other tests install partial ``comfy`` modules. Never use setdefault here: it
    attaches the stub child module to the wrong parent, or to none, and leaks that
    hybrid into later tests. Save and restore all three entries instead.
    """
    names = ("folder_paths", "comfy", "comfy.samplers")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    fp = types.ModuleType("folder_paths")
    fp.get_filename_list = lambda kind: ["stub.safetensors"]
    fp.get_output_directory = lambda: "/tmp"  # noqa: S108 (inert stub)
    fp.base_path = "/tmp"  # noqa: S108 (inert stub)
    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")

    class _KS:
        SAMPLERS = ["euler"]
        SCHEDULERS = ["simple"]

    samplers.KSampler = _KS
    comfy.samplers = samplers
    sys.modules.update({"folder_paths": fp, "comfy": comfy, "comfy.samplers": samplers})
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _input_contract(cls) -> dict[str, list[list[object]]]:
    contract = {}
    for section in ("required", "optional", "hidden"):
        rows = []
        for name, spec in cls.INPUT_TYPES().get(section, {}).items():
            if isinstance(spec, tuple):
                input_type = spec[0]
                options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if isinstance(input_type, list):
                    # Model file and sampler lists vary by install. The contract
                    # pins the rule (first available option), not the string
                    # that sits first on this machine.
                    default = (
                        "<first-option>"
                        if name in _DYNAMIC_COMBOS and "default" not in options
                        else options.get("default", input_type[0] if input_type else None)
                    )
                    input_type = "COMBO"
                else:
                    default = options.get("default")
            else:  # Comfy hidden inputs use the compact ``name: type`` form.
                input_type = spec
                default = None
            rows.append([name, input_type, default])
        if rows:
            contract[section] = rows
    return contract


def test_node_class_mapping_keys_frozen():
    assert set(NODE_CLASS_MAPPINGS) == EXPECTED_NODE_KEYS


def test_display_names_cover_every_node():
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == set(NODE_CLASS_MAPPINGS)
    assert all(NODE_DISPLAY_NAME_MAPPINGS.values())


def test_ordered_input_names_types_and_defaults_are_frozen():
    with _comfy_surface_stubs():
        actual = {name: _input_contract(cls) for name, cls in NODE_CLASS_MAPPINGS.items()}
    assert actual == EXPECTED_INPUT_CONTRACT


def test_every_node_declares_the_comfy_surface():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        assert hasattr(cls, "INPUT_TYPES"), name
        assert hasattr(cls, "RETURN_TYPES"), name
        assert hasattr(cls, "FUNCTION"), name
        assert getattr(cls, "CATEGORY", "").startswith("DGX Monarch"), name


def test_function_signatures_accept_every_declared_input():
    """A declared input with no FUNCTION parameter crashes the node at run time
    with 'unexpected keyword argument' (seen 2026-07-07 on the gate node's strict
    widget)."""
    import inspect
    with _comfy_surface_stubs():
        from dgx_monarch.nodes import NODE_CLASS_MAPPINGS

        for name, cls in NODE_CLASS_MAPPINGS.items():
            fn = getattr(cls, cls.FUNCTION)
            params = inspect.signature(fn).parameters
            has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            if has_kwargs:
                continue
            declared = cls.INPUT_TYPES()
            names = list(declared.get("required", {})) + list(declared.get("optional", {}))
            names += list(declared.get("hidden", {}))
            missing = [n for n in names if n not in params]
            assert not missing, f"{name}.{cls.FUNCTION}() missing parameters for inputs: {missing}"
            # execution.get_input_data drops values not declared in INPUT_TYPES.
            # An undeclared parameter would therefore ignore the prompt's widget value.
            undeclared = [pname for pname in params
                          if pname not in ("self",) and pname not in names]
            assert not undeclared, (
                f"{name}.{cls.FUNCTION}() has parameters with no declared input "
                f"(phantom knobs, silently stripped from API prompts): {undeclared}")
