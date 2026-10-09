"""Executable regression contract for the ComfyUI touchpoint manifest."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.actor.unbake import RESTORABLE_QUANT_FORMATS
from dgx_monarch.adapters import ADAPTERS
from dgx_monarch.adapters.detect import (
    model_base_touchpoints,
    model_detection_touchpoints,
)
from dgx_monarch.comfy_forward_contracts import (
    ADAPTER_BIND_SITES,
    ADDITIONAL_REBOUND_MODEL_BASE_TOUCHPOINTS,
    BROAD_ADAPTER_SUBCLASS_CONTRACTS,
    REBOUND_CALLER_TOUCHPOINTS,
    REBOUND_FORWARD_TOUCHPOINTS,
    REBOUND_METHOD_CONTRACTS,
    REWRITTEN_FORWARD_CALLEE_TOUCHPOINTS,
    ReboundMethodContract,
)
from dgx_monarch.comfy_rebound_validation import rebound_signature_incompatibility
from dgx_monarch.comfy_surface import (
    GUIDER_INSTANCE_ATTRIBUTES,
    MODEL_DETECTION_ANCHORS,
    MODEL_PATCHER_INSTANCE_ATTRIBUTES,
    OPTIONAL_TOUCHPOINTS,
    QUANT_ALGORITHM_FIELDS,
    QUANT_ALGORITHM_FORMATS,
    QUANT_LAYOUT_PARAMETERS,
    QUANTIZED_TENSOR_INSTANCE_ATTRIBUTES,
    TOUCHPOINTS,
    Touchpoint,
    assert_comfy_surface,
)

EXPECTED_MODEL_BASE_TOUCHPOINTS = {
    "Anima",
    "Boogu",
    "Chroma",
    "ChromaRadiance",
    "CogVideoX",
    "ErnieImage",
    "Flux",
    "Flux2",
    "HunyuanImage21",
    "HunyuanImage21Refiner",
    "HunyuanVideo",
    "HunyuanVideo15",
    "HunyuanVideo15_SR_Distilled",
    "Ideogram4",
    "Kandinsky5",
    "Kandinsky5Image",
    "Krea2",
    "Lens",
    "LongCatImage",
    "MageFlow",
    "LTXAV",
    "LTXV",
    "Lumina2",
    "MiniMaxH3",
    "Omnigen2",
    "PiD",
    "PixelDiTT2I",
    "QwenImage",
    "QwenImage21",
    "WAN21",
    "WAN21_FlowRVS",
    "WAN21_SCAIL",
    "WAN21_SCAIL2",
    "WAN_Animate2",
    "WAN22",
    "WAN22_WanDancer",
    "ZImagePixelSpace",
}

ADAPTER_SOURCES = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch" / "adapters"
REBOUND_CONTRACTS_BY_PATH = {
    contract.path: contract for contract in REBOUND_METHOD_CONTRACTS
}


def _canary_model_base_touchpoints() -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        (*model_base_touchpoints(ADAPTERS), *ADDITIONAL_REBOUND_MODEL_BASE_TOUCHPOINTS)
    ))


class _Namespace(SimpleNamespace):
    pass


class _SourceBaseModel:
    def __init__(self):
        self.diffusion_model = object()

    def load_model_weights(self, sd, *, unet_prefix="", assign=False):
        del sd, unet_prefix, assign

    def process_latent_out(self, latent):
        return latent


class _SourceBaseModelMissingDiffusion(_SourceBaseModel):
    def __init__(self):
        pass


class _SourceNextDiT:
    def __init__(self, *, pad_tokens_multiple=None):
        self.pad_tokens_multiple = pad_tokens_multiple

    def _forward(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        ref_latents=None,
        ref_contexts=None,
        siglip_feats=None,
        transformer_options=None,
        **kwargs,
    ):
        del (
            x,
            timesteps,
            context,
            num_tokens,
            attention_mask,
            ref_latents,
            ref_contexts,
            siglip_feats,
            transformer_options,
            kwargs,
        )


def _callable(parameters: tuple[str, ...] = (), positional: tuple[str, ...] = ()):
    def value(*args, **kwargs):
        del args, kwargs

    value.__signature__ = inspect.Signature(
        [
            *(inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY) for name in positional),
            *(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in parameters),
        ]
    )
    return value


def _rebound_callable(contract: ReboundMethodContract):
    positional = contract.positional_only + contract.positional_or_keyword
    explicit = [
        inspect.Parameter(
            name,
            (
                inspect.Parameter.POSITIONAL_ONLY
                if index < len(contract.positional_only)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
            default=(
                inspect.Parameter.empty
                if index < contract.required_positional
                else None
            ),
        )
        for index, name in enumerate(positional)
    ]
    if contract.variadic_positional is not None:
        explicit.append(
            inspect.Parameter(
                contract.variadic_positional,
                inspect.Parameter.VAR_POSITIONAL,
            )
        )
    required_keyword_only = set(contract.required_keyword_only)
    explicit.extend(
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=(inspect.Parameter.empty if name in required_keyword_only else None),
        )
        for name in contract.keyword_only
    )
    if contract.variadic_keyword is not None:
        explicit.append(
            inspect.Parameter(contract.variadic_keyword, inspect.Parameter.VAR_KEYWORD)
        )

    def value(*args, **kwargs):
        del args, kwargs

    value.__signature__ = inspect.Signature(explicit)
    return value


def _touchpoint_callable(touchpoint: Touchpoint):
    contract = REBOUND_CONTRACTS_BY_PATH.get(touchpoint.path)
    if contract is not None:
        return _rebound_callable(contract)
    return _callable(touchpoint.parameters, touchpoint.positional)


def _resolve(root: object, attribute: str) -> object:
    value = root
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def _install(root: object, touchpoint: Touchpoint) -> None:
    parts = touchpoint.attribute.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            setattr(parent, part, _Namespace())
        parent = getattr(parent, part)
    leaf = parts[-1]
    if not hasattr(parent, leaf):
        setattr(
            parent,
            leaf,
            _touchpoint_callable(touchpoint) if touchpoint.callable else _Namespace(),
        )
    elif touchpoint.callable:
        current = getattr(parent, leaf)
        if not callable(current):
            replacement = _touchpoint_callable(touchpoint)
            replacement.__dict__.update(vars(current))
            setattr(parent, leaf, replacement)
        else:
            current.__signature__ = _touchpoint_callable(touchpoint).__signature__


def _fake_surface():
    modules = {name: _Namespace() for name in {entry.module for entry in TOUCHPOINTS}}
    for touchpoint in sorted(TOUCHPOINTS, key=lambda entry: entry.attribute.count(".")):
        _install(modules[touchpoint.module], touchpoint)

    quant_ops = modules["comfy.quant_ops"]
    layouts = {
        f"layout_{format_name}": _Namespace(Params=_callable(QUANT_LAYOUT_PARAMETERS[format_name]))
        for format_name in QUANT_ALGORITHM_FORMATS
    }
    quant_ops.QUANT_ALGOS = {
        format_name: {
            "storage_t": object(),
            "comfy_tensor_layout": f"layout_{format_name}",
        }
        for format_name in QUANT_ALGORITHM_FORMATS
    }

    def get_layout_class(name):
        return layouts[name]

    get_layout_class.__signature__ = _callable(positional=("name",)).__signature__
    quant_ops.get_layout_class = get_layout_class

    model_base = modules["comfy.model_base"]
    for name in _canary_model_base_touchpoints():
        if not hasattr(model_base, name):
            setattr(model_base, name, type(name, (), {}))
    for contract in BROAD_ADAPTER_SUBCLASS_CONTRACTS:
        root = getattr(model_base, contract.root_model_base)
        for branch in contract.branches:
            if branch.model_base != contract.root_model_base:
                setattr(model_base, branch.model_base, type(branch.model_base, (root,), {}))

    patcher = _Namespace()
    for attribute in MODEL_PATCHER_INSTANCE_ATTRIBUTES:
        setattr(patcher, attribute, object())
    detection = _Namespace(
        diffusion_model=_Namespace(pad_tokens_multiple=32),
    )
    quantized = _Namespace()
    for attribute in QUANTIZED_TENSOR_INSTANCE_ATTRIBUTES:
        setattr(quantized, attribute, object())
    return modules, patcher, detection, quantized


def _delete(root: object, attribute: str) -> None:
    parts = attribute.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    delattr(parent, parts[-1])


def _assert_fake_surface(
    modules: dict[str, object],
    patcher: object,
    detection: object,
    quantized: object,
    guider: object | None = None,
    check_detection_sources: bool = False,
) -> int:
    if guider is None:
        guider = _Namespace(model_patcher=patcher)
    return assert_comfy_surface(
        import_module=modules.__getitem__,
        adapters=ADAPTERS,
        model_patcher=patcher,
        model_detection_object=detection,
        quantized_tensor=quantized,
        guider=guider,
        check_detection_sources=check_detection_sources,
    )


def _enclosing_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names: list[str] = []
    cursor = node
    while cursor in parents:
        cursor = parents[cursor]
        if isinstance(cursor, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(cursor.name)
    return ".".join(reversed(names))


BindSiteKey = tuple[str, str, str, str, str, str]
FunctionDefinition = ast.FunctionDef | ast.AsyncFunctionDef
DefinitionMap = dict[tuple[str, str], list[FunctionDefinition]]
AssignmentMap = dict[tuple[str, str], list[ast.AST]]
BindRecord = tuple[
    BindSiteKey,
    ast.Call,
    DefinitionMap,
    AssignmentMap,
]


def _bind_site_key(
    path: Path,
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> BindSiteKey | None:
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Attribute)
        or node.func.attr != "bind"
    ):
        return None
    binder = ast.unparse(node.func)
    if binder not in {"self.bind", "Adapter.bind"}:
        raise AssertionError(
            f"{path.name}:{node.lineno}: unsupported adapter .bind binder {binder!r}; "
            "declare and scan a supported binding form before using it"
        )
    if (
        len(node.args) != 3
        or node.keywords
        or not isinstance(node.args[1], ast.Constant)
        or not isinstance(node.args[1].value, str)
    ):
        raise AssertionError(
            f"{path.name}:{node.lineno}: adapter bind must keep three static arguments"
        )
    return (
        path.name,
        _enclosing_scope(node, parents),
        binder,
        ast.unparse(node.args[0]),
        node.args[1].value,
        ast.unparse(node.args[2]),
    )


def _bind_records_for_tree(path: Path, tree: ast.AST) -> list[BindRecord]:
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    definitions: DefinitionMap = {}
    assignments: AssignmentMap = {}
    for node in ast.walk(tree):
        scope = _enclosing_scope(node, parents)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault((scope, node.name), []).append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault((scope, target.id), []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments.setdefault((scope, node.target.id), []).append(node.value)

    records: list[BindRecord] = []
    for node in ast.walk(tree):
        key = _bind_site_key(path, node, parents)
        if key is not None:
            records.append((key, node, definitions, assignments))
    return records


def _adapter_bind_records() -> list[BindRecord]:
    records: list[BindRecord] = []
    for path in sorted(ADAPTER_SOURCES.glob("*.py")):
        records.extend(_bind_records_for_tree(path, ast.parse(path.read_text())))
    return records


def _adapter_bind_site_keys() -> list[BindSiteKey]:
    return [key for key, _node, _definitions, _assignments in _adapter_bind_records()]


def _replacement_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.IfExp):
        return (*_replacement_names(node.body), *_replacement_names(node.orelse))
    raise AssertionError(
        f"adapter bind replacement {ast.unparse(node)!r} is not a static function/conditional"
    )


def _replacement_definition(
    definitions: DefinitionMap,
    assignments: AssignmentMap,
    scope: str,
    name: str,
    seen: frozenset[tuple[str, str]] = frozenset(),
) -> FunctionDefinition:
    candidate_scope = scope
    while True:
        lookup = (candidate_scope, name)
        if lookup in seen:
            raise AssertionError(f"replacement alias cycle at {lookup}")
        candidates = definitions.get((candidate_scope, name), [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise AssertionError(
                f"replacement function {name!r} is ambiguous in scope {candidate_scope!r}"
            )
        values = assignments.get((candidate_scope, name), [])
        if len(values) > 1:
            raise AssertionError(
                f"replacement alias {name!r} is ambiguous in scope {candidate_scope!r}"
            )
        if values:
            value = values[0]
            if isinstance(value, ast.Name):
                return _replacement_definition(
                    definitions,
                    assignments,
                    candidate_scope,
                    value.id,
                    seen | {lookup},
                )
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                factory = _replacement_definition(
                    definitions,
                    assignments,
                    candidate_scope,
                    value.func.id,
                    seen | {lookup},
                )
                returns = [
                    statement.value
                    for statement in factory.body
                    if isinstance(statement, ast.Return) and statement.value is not None
                ]
                if len(returns) != 1 or not isinstance(returns[0], ast.Name):
                    raise AssertionError(
                        f"replacement factory {factory.name!r} must return one local function"
                    )
                factory_parent = next(
                    parent_scope
                    for (parent_scope, function_name), definitions_here in definitions.items()
                    if function_name == factory.name and factory in definitions_here
                )
                factory_scope = ".".join(
                    part for part in (factory_parent, factory.name) if part
                )
                produced = definitions.get((factory_scope, returns[0].id), [])
                if len(produced) != 1:
                    raise AssertionError(
                        f"replacement factory {factory.name!r} return is not one local function"
                    )
                return produced[0]
            raise AssertionError(
                f"replacement alias {name!r} has unsupported value {ast.unparse(value)!r}"
            )
        if not candidate_scope:
            break
        candidate_scope = candidate_scope.rpartition(".")[0]
    raise AssertionError(f"replacement function {name!r} is not defined in scope {scope!r}")


def _ast_signature(definition: FunctionDefinition) -> inspect.Signature:
    arguments = definition.args
    positional = (*arguments.posonlyargs, *arguments.args)
    first_default = len(positional) - len(arguments.defaults)
    parameters = [
        inspect.Parameter(
            argument.arg,
            (
                inspect.Parameter.POSITIONAL_ONLY
                if index < len(arguments.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
            default=(inspect.Parameter.empty if index < first_default else object()),
        )
        for index, argument in enumerate(positional)
    ]
    if arguments.vararg is not None:
        parameters.append(
            inspect.Parameter(arguments.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )
    parameters.extend(
        inspect.Parameter(
            argument.arg,
            inspect.Parameter.KEYWORD_ONLY,
            default=(
                inspect.Parameter.empty
                if default is None
                else object()
            ),
        )
        for argument, default in zip(
            arguments.kwonlyargs,
            arguments.kw_defaults,
            strict=True,
        )
    )
    if arguments.kwarg is not None:
        parameters.append(
            inspect.Parameter(arguments.kwarg.arg, inspect.Parameter.VAR_KEYWORD)
        )
    return inspect.Signature(parameters)


def _target_call_shapes(
    contract: ReboundMethodContract,
) -> tuple[tuple[str, list[object], dict[str, object]], ...]:
    sentinel = object()
    positional = contract.positional_only + contract.positional_or_keyword
    minimal_args = [sentinel] * contract.required_positional
    minimal_kwargs = dict.fromkeys(contract.required_keyword_only, sentinel)

    full_args = [sentinel] * contract.required_positional
    full_kwargs = dict.fromkeys(
        positional[contract.required_positional:], sentinel
    )
    # A tolerated trailing or infix parameter is a real stock call shape on the
    # newer tree, so every replacement has to name or absorb it too.
    full_kwargs.update(dict.fromkeys(contract.optional_trailing, sentinel))
    full_kwargs.update(dict.fromkeys(contract.optional_infix, sentinel))
    full_kwargs.update(dict.fromkeys(contract.keyword_only, sentinel))
    if contract.variadic_keyword is not None:
        full_kwargs["__dgxm_declared_variadic_keyword__"] = sentinel

    named_args = [sentinel] * len(contract.positional_only)
    named_positional_or_keyword = contract.positional_or_keyword
    if not contract.positional_only and named_positional_or_keyword[:1] == ("self",):
        named_args.append(sentinel)
        named_positional_or_keyword = named_positional_or_keyword[1:]
    named_kwargs = dict.fromkeys(named_positional_or_keyword, sentinel)
    named_kwargs.update(dict.fromkeys(contract.optional_trailing, sentinel))
    named_kwargs.update(dict.fromkeys(contract.optional_infix, sentinel))
    named_kwargs.update(dict.fromkeys(contract.keyword_only, sentinel))
    if contract.variadic_keyword is not None:
        named_kwargs["__dgxm_declared_variadic_keyword__"] = sentinel

    shapes = [
        ("minimal positional", minimal_args, minimal_kwargs),
        ("full optional-keyword", full_args, full_kwargs),
        ("full named", named_args, named_kwargs),
    ]
    if contract.variadic_positional is not None:
        variadic_kwargs = dict.fromkeys(contract.keyword_only, sentinel)
        if contract.variadic_keyword is not None:
            variadic_kwargs["__dgxm_declared_variadic_keyword__"] = sentinel
        shapes.append(
            (
                "declared variadic positional",
                [sentinel] * (len(positional) + 1),
                variadic_kwargs,
            )
        )
    return tuple(shapes)


def test_adapter_detection_metadata_covers_every_current_model_base_class():
    assert set(_canary_model_base_touchpoints()) == EXPECTED_MODEL_BASE_TOUCHPOINTS
    assert set(ADDITIONAL_REBOUND_MODEL_BASE_TOUCHPOINTS) == {"ChromaRadiance"}
    assert model_detection_touchpoints(ADAPTERS) == (
        "diffusion_model",
        "diffusion_model.pad_tokens_multiple",
    )
    assert set(MODEL_DETECTION_ANCHORS) == set(model_detection_touchpoints(ADAPTERS))
    assert {metadata[0] for metadata in MODEL_DETECTION_ANCHORS.values()} <= {
        touchpoint.path for touchpoint in TOUCHPOINTS
    }


def test_broad_adapter_subclass_branches_name_their_exact_rebound_targets():
    actual = {
        (
            contract.family,
            contract.root_model_base,
            branch.model_base,
            branch.rebound_targets,
        )
        for contract in BROAD_ADAPTER_SUBCLASS_CONTRACTS
        for branch in contract.branches
    }
    assert actual == {
        (
            "chroma",
            "Chroma",
            "Chroma",
            ("comfy.ldm.chroma.model.Chroma.forward_orig",),
        ),
        (
            "chroma",
            "Chroma",
            "ChromaRadiance",
            ("comfy.ldm.chroma_radiance.model.ChromaRadiance.forward_orig",),
        ),
        (
            "krea2",
            "Krea2",
            "Krea2",
            ("comfy.ldm.krea2.model.SingleStreamDiT._forward",),
        ),
        (
            "ideogram4",
            "Ideogram4",
            "Ideogram4",
            ("comfy.ldm.ideogram4.model.Ideogram4Transformer2DModel._forward",),
        ),
    }
    contract_families = {
        contract.path: set(contract.families)
        for contract in REBOUND_METHOD_CONTRACTS
    }
    for family, _root, _model_base, targets in actual:
        assert all(family in contract_families[target] for target in targets)


def test_adapter_bind_sites_and_rebound_contracts_are_a_bijection():
    actual_sites = _adapter_bind_site_keys()
    declared_sites = [site.ast_key for site in ADAPTER_BIND_SITES]
    assert len(actual_sites) == len(set(actual_sites)) == 44
    assert len(declared_sites) == len(set(declared_sites)) == 44
    assert set(actual_sites) == set(declared_sites), (
        "Adapter.bind/self.bind sites differ from ADAPTER_BIND_SITES: "
        f"undeclared={sorted(set(actual_sites) - set(declared_sites))}, "
        f"stale={sorted(set(declared_sites) - set(actual_sites))}"
    )

    contract_paths = [contract.path for contract in REBOUND_METHOD_CONTRACTS]
    assert len(contract_paths) == len(set(contract_paths)) == 38
    referenced_targets = {
        target for site in ADAPTER_BIND_SITES for target in site.targets
    }
    assert referenced_targets == set(contract_paths)
    assert all(site.families and site.branch and site.targets for site in ADAPTER_BIND_SITES)
    assert all(
        contract.families and contract.branches
        for contract in REBOUND_METHOD_CONTRACTS
    )
    adapter_families = {adapter.family for adapter in ADAPTERS}
    assert {
        family
        for contract in REBOUND_METHOD_CONTRACTS
        for family in contract.families
    } <= adapter_families
    for contract in REBOUND_METHOD_CONTRACTS:
        site_families = {
            family
            for site in ADAPTER_BIND_SITES
            if contract.path in site.targets
            for family in site.families
        }
        assert site_families == set(contract.families), contract.path
    assert set(contract_paths) <= {touchpoint.path for touchpoint in TOUCHPOINTS}

    rebound_paths = {touchpoint.path for touchpoint in REBOUND_FORWARD_TOUCHPOINTS}
    caller_paths = {touchpoint.path for touchpoint in REBOUND_CALLER_TOUCHPOINTS}
    callee_paths = {
        touchpoint.path for touchpoint in REWRITTEN_FORWARD_CALLEE_TOUCHPOINTS
    }
    assert rebound_paths == set(contract_paths)
    assert (len(rebound_paths), len(caller_paths), len(callee_paths)) == (38, 2, 7)
    assert rebound_paths.isdisjoint(caller_paths | callee_paths)


@pytest.mark.parametrize(
    "binder",
    ("alias.bind", "super().bind", "factory().bind"),
)
def test_adapter_bind_scan_rejects_every_undeclared_binder_form(binder: str):
    tree = ast.parse(
        "def probe(self, model, replacement):\n"
        f"    {binder}(model, 'forward', replacement)\n"
    )
    with pytest.raises(AssertionError, match=r"unsupported adapter \.bind binder"):
        _bind_records_for_tree(Path("synthetic_adapter.py"), tree)


def test_every_declared_replacement_accepts_its_target_stock_call_shapes():
    """Statically bind local replacement signatures without importing Comfy."""
    sites = {site.ast_key: site for site in ADAPTER_BIND_SITES}
    contracts = {
        contract.path: contract for contract in REBOUND_METHOD_CONTRACTS
    }
    checked: list[tuple[BindSiteKey, str, str]] = []
    for key, node, definitions, assignments in _adapter_bind_records():
        site = sites[key]
        for replacement_name in _replacement_names(node.args[2]):
            definition = _replacement_definition(
                definitions,
                assignments,
                site.scope,
                replacement_name,
            )
            replacement_signature = _ast_signature(definition)
            for target in site.targets:
                contract = contracts[target]
                for shape, args, kwargs in _target_call_shapes(contract):
                    try:
                        replacement_signature.bind(*args, **kwargs)
                    except TypeError as exc:
                        pytest.fail(
                            f"{key[0]}:{definition.lineno} replacement "
                            f"{replacement_name}{replacement_signature} rejects {shape} "
                            f"shape for {target}: {exc}"
                        )
                checked.append((key, replacement_name, target))
    assert len(checked) == len(set(checked)) == 52


def test_model_detection_source_anchors_are_executable_and_name_missing_assignment():
    modules, patcher, detection, quantized = _fake_surface()
    modules["comfy.model_base"].BaseModel = _SourceBaseModel
    modules["comfy.ldm.lumina.model"].NextDiT = _SourceNextDiT
    assert _assert_fake_surface(
        modules, patcher, detection, quantized, check_detection_sources=True
    ) > 0

    modules["comfy.model_base"].BaseModel = _SourceBaseModelMissingDiffusion
    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(
            modules, patcher, detection, quantized, check_detection_sources=True
        )
    assert str(error.value) == (
        "Comfy touchpoint missing: comfy.model_base.BaseModel.diffusion_model"
    )


def test_manifest_paths_are_unique_and_non_vacuous():
    paths = [touchpoint.path for touchpoint in TOUCHPOINTS]
    assert len(paths) == len(set(paths))
    assert len(paths) >= 75
    assert set(OPTIONAL_TOUCHPOINTS) == {
        "comfy.model_management.current_loaded_models",
        "comfy.memory_management.aimdo_enabled",
        "comfy.utils.DISABLE_MMAP",
        "comfy.model_patcher.ModelPatcher.model.model_loaded_weight_memory",
        "comfy.model_patcher.ModelPatcher.model.model_lowvram",
        "comfy.model_patcher.ModelPatcher.load_device",
        "comfy.ops.mixed_precision_ops.Linear.input_scale",
    }
    assert set(QUANT_ALGORITHM_FORMATS) == RESTORABLE_QUANT_FORMATS
    modules, patcher, detection, quantized = _fake_surface()
    assert _assert_fake_surface(modules, patcher, detection, quantized) == (
        len(TOUCHPOINTS)
        + len(EXPECTED_MODEL_BASE_TOUCHPOINTS)
        + len(BROAD_ADAPTER_SUBCLASS_CONTRACTS)
        + len(MODEL_PATCHER_INSTANCE_ATTRIBUTES)
        + len(model_detection_touchpoints(ADAPTERS))
        + len(QUANT_ALGORITHM_FORMATS) * (len(QUANT_ALGORITHM_FIELDS) + 1)
        + len(QUANTIZED_TENSOR_INSTANCE_ATTRIBUTES)
        + len(GUIDER_INSTANCE_ATTRIBUTES)
    )


def test_surface_uses_existing_prompt_server_instance_without_constructing_it(monkeypatch):
    pytest.importorskip("aiohttp")
    from aiohttp import web

    modules, patcher, detection, quantized = _fake_surface()
    instance = modules["server"].PromptServer.instance

    class PromptServer:
        def __init__(self, loop, asset_manager):
            raise AssertionError("the pack must not construct PromptServer")

        def add_on_prompt_handler(self, handler):
            self.on_prompt_handlers.append(handler)

        def send_sync(self, event, data):
            return None

    PromptServer.instance = instance
    modules["server"].PromptServer = PromptServer

    canary_path = Path(__file__).parents[1] / "tests" / "canary" / "comfy_entrypoint_canary.py"
    spec = importlib.util.spec_from_file_location("prompt_server_canary_fixture", canary_path)
    assert spec and spec.loader
    canary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canary)

    server = types.ModuleType("server")
    server.PromptServer = PromptServer
    monkeypatch.setitem(sys.modules, "server", server)
    prompt_server = canary._headless_prompt_server(PromptServer, web.RouteTableDef)
    PromptServer.instance = prompt_server

    from dgx_monarch.nodes import routes

    routes.register()

    assert _assert_fake_surface(modules, patcher, detection, quantized) > 0
    assert getattr(prompt_server, "_dgxm_routes", False) is True
    assert len(prompt_server.on_prompt_handlers) == 1
    assert {route.path for route in prompt_server.routes} >= {
        "/dgxm/telemetry", "/dgxm/metrics", "/dgxm/recycle", "/dgxm/consents", "/dgxm/consent",
    }


def test_module_import_failure_names_the_first_touchpoint():
    first = TOUCHPOINTS[0]

    def fail_import(_name: str):
        raise RuntimeError("synthetic import failure")

    with pytest.raises(RuntimeError) as error:
        assert_comfy_surface(import_module=fail_import, adapters=ADAPTERS)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {first.path} (module import failed)"
    )


@pytest.mark.parametrize("touchpoint", TOUCHPOINTS, ids=lambda entry: entry.path)
def test_deleting_any_manifest_attribute_names_the_exact_touchpoint(touchpoint: Touchpoint):
    modules, patcher, detection, quantized = _fake_surface()
    _delete(modules[touchpoint.module], touchpoint.attribute)

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == f"Comfy touchpoint missing: {touchpoint.path}"


SIGNATURE_CASES = [(touchpoint, parameter) for touchpoint in TOUCHPOINTS for parameter in touchpoint.parameters]


@pytest.mark.parametrize(
    ("touchpoint", "parameter"),
    SIGNATURE_CASES,
    ids=lambda value: value.path if isinstance(value, Touchpoint) else value,
)
def test_deleting_any_manifest_parameter_names_the_exact_touchpoint(
    touchpoint: Touchpoint,
    parameter: str,
):
    modules, patcher, detection, quantized = _fake_surface()
    value = _resolve(modules[touchpoint.module], touchpoint.attribute)
    parameters = [item for item in inspect.signature(value).parameters.values() if item.name != parameter]
    value.__signature__ = inspect.Signature(parameters)

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {touchpoint.path} (signature missing parameter {parameter!r})"
    )


@pytest.mark.parametrize(
    "touchpoint",
    [
        touchpoint
        for touchpoint in TOUCHPOINTS
        if touchpoint.positional and touchpoint.path not in REBOUND_CONTRACTS_BY_PATH
    ],
    ids=lambda entry: entry.path,
)
def test_rejecting_a_declared_positional_call_names_the_exact_touchpoint(
    touchpoint: Touchpoint,
):
    modules, patcher, detection, quantized = _fake_surface()
    value = _resolve(modules[touchpoint.module], touchpoint.attribute)
    parameters = list(inspect.signature(value).parameters.values())
    first_keyword = next(
        (index for index, item in enumerate(parameters) if item.kind is inspect.Parameter.KEYWORD_ONLY),
        len(parameters),
    )
    del parameters[first_keyword - 1]
    value.__signature__ = inspect.Signature(parameters)

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {touchpoint.path} (positional parameter order changed)"
    )


def test_reordering_positional_parameters_names_the_exact_touchpoint():
    modules, patcher, detection, quantized = _fake_surface()
    path = "comfy.ldm.cogvideo.model.get_timestep_embedding"
    value = _resolve(modules["comfy.ldm.cogvideo.model"], "get_timestep_embedding")
    parameters = list(inspect.signature(value).parameters.values())
    parameters[2], parameters[3] = parameters[3], parameters[2]
    value.__signature__ = inspect.Signature(parameters)

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {path} (positional parameter order changed)"
    )


@pytest.mark.parametrize(
    "contract",
    REBOUND_METHOD_CONTRACTS,
    ids=lambda contract: contract.path,
)
def test_rebound_contract_rejects_an_added_explicit_parameter(
    contract: ReboundMethodContract,
):
    """An optional stock kwarg can otherwise disappear into a replacement's **kwargs."""
    modules, patcher, detection, quantized = _fake_surface()
    value = _resolve(modules[contract.module], contract.attribute)
    parameters = list(inspect.signature(value).parameters.values())
    insert_at = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind
            in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            )
        ),
        len(parameters),
    )
    parameters.insert(
        insert_at,
        inspect.Parameter(
            "synthetic_optional",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
        ),
    )
    value.__signature__ = inspect.Signature(parameters)

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {contract.path} "
        "(exact explicit parameter list changed)"
    )


def _signature_with_trailing(
    contract: ReboundMethodContract,
    names: tuple[str, ...],
) -> inspect.Signature:
    """The declared signature with `names` appended before the variadics."""
    parameters = list(inspect.signature(_rebound_callable(contract)).parameters.values())
    insert_at = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind
            in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            )
        ),
        len(parameters),
    )
    for offset, name in enumerate(names):
        parameters.insert(
            insert_at + offset,
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            ),
        )
    return inspect.Signature(parameters)


def test_version_tolerant_contracts_are_exactly_the_reviewed_set():
    """A tolerated shape admits only reviewed upstream signatures, never a general loosening."""
    assert {
        contract.path: contract.optional_trailing
        for contract in REBOUND_METHOD_CONTRACTS
        if contract.optional_trailing
    } == {
        "comfy.ldm.minimax.model.MiniMaxH3Model._forward": (
            "denoise_mask", "audio_denoise_mask",
        ),
    }


def test_a_tolerated_trailing_suffix_admits_both_pinned_signatures():
    """Comfy ff6c8a8 appended H3's mask parameters; both trees stay supported."""
    contract = next(
        contract
        for contract in REBOUND_METHOD_CONTRACTS
        if contract.optional_trailing
    )
    declared = inspect.signature(_rebound_callable(contract))
    assert rebound_signature_incompatibility(declared, contract) is None
    tolerated = _signature_with_trailing(contract, contract.optional_trailing)
    assert rebound_signature_incompatibility(tolerated, contract) is None

    for rejected in (
        contract.optional_trailing[:1],
        (*contract.optional_trailing, "third_mask"),
        tuple(reversed(contract.optional_trailing)),
    ):
        assert rebound_signature_incompatibility(
            _signature_with_trailing(contract, rejected), contract
        ) == "exact explicit parameter list changed", rejected

    # The tolerated shape still goes through the defaults gate: dropping the
    # defaults keeps the admitted parameter list and fails on the next check.
    required = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in contract.positional_or_keyword + contract.optional_trailing
        ]
        + [inspect.Parameter(contract.variadic_keyword, inspect.Parameter.VAR_KEYWORD)]
    )
    assert rebound_signature_incompatibility(required, contract) == (
        "exact parameter defaults changed"
    )


def _signature_with_infix(
    contract: ReboundMethodContract,
    names: tuple[str, ...],
) -> inspect.Signature:
    """The declared signature with `names` inserted before its last declared name
    (transformer_options in every infix contract so far), not before the variadics
    as a trailing suffix is."""
    parameters = list(inspect.signature(_rebound_callable(contract)).parameters.values())
    declared_count = len(contract.positional_only) + len(contract.positional_or_keyword)
    insert_at = declared_count - 1
    for offset, name in enumerate(names):
        parameters.insert(
            insert_at + offset,
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            ),
        )
    return inspect.Signature(parameters)


def test_version_tolerant_infix_contracts_are_exactly_the_reviewed_set():
    """A tolerated infix admits only reviewed upstream signatures, never a general loosening."""
    assert {
        contract.path: contract.optional_infix
        for contract in REBOUND_METHOD_CONTRACTS
        if contract.optional_infix
    } == {
        "comfy.ldm.lumina.model.NextDiT._forward": ("direct_context", "ref_frames"),
    }


def test_a_tolerated_infix_admits_both_pinned_signatures():
    """Comfy 3b4c0b0e/1568e6cf inserted Ming-Image params before transformer_options;
    both trees stay supported, and any other insertion fails."""
    contract = next(
        contract
        for contract in REBOUND_METHOD_CONTRACTS
        if contract.optional_infix
    )
    declared = inspect.signature(_rebound_callable(contract))
    assert rebound_signature_incompatibility(declared, contract) is None
    tolerated = _signature_with_infix(contract, contract.optional_infix)
    assert rebound_signature_incompatibility(tolerated, contract) is None

    for rejected in (
        contract.optional_infix[:1],
        (*contract.optional_infix, "third_param"),
        tuple(reversed(contract.optional_infix)),
    ):
        assert rebound_signature_incompatibility(
            _signature_with_infix(contract, rejected), contract
        ) == "exact explicit parameter list changed", rejected

    # The same two names after transformer_options instead of before it are an
    # unreviewed shape and must fail.
    trailing_instead = _signature_with_trailing(contract, contract.optional_infix)
    assert rebound_signature_incompatibility(trailing_instead, contract) == (
        "exact explicit parameter list changed"
    )


@pytest.mark.parametrize(
    "path",
    (
        "comfy.ldm.chroma_radiance.model.ChromaRadiance.forward_orig",
        "comfy.ldm.wan.model.SCAIL2WanModel.forward_orig",
    ),
)
def test_inherited_rebound_path_rejects_a_future_concrete_override(path: str):
    """Each shipped subclass path remains guarded after it gains an override."""
    modules, patcher, detection, quantized = _fake_surface()
    module, attribute = next(
        (contract.module, contract.attribute)
        for contract in REBOUND_METHOD_CONTRACTS
        if contract.path == path
    )
    value = _resolve(modules[module], attribute)
    parameters = list(inspect.signature(value).parameters.values())
    insert_at = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
    parameters.insert(
        insert_at,
        inspect.Parameter(
            "future_subclass_argument",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
        ),
    )
    value.__signature__ = inspect.Signature(parameters)

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {path} "
        "(exact explicit parameter list changed)"
    )


@pytest.mark.parametrize(
    ("root_name", "future_name"),
    (
        ("Chroma", "FutureChroma"),
        ("Krea2", "FutureKrea2"),
        ("Ideogram4", "FutureIdeogram4"),
    ),
)
def test_broad_adapter_rejects_an_unmanifested_future_subclass(
    root_name: str,
    future_name: str,
):
    modules, patcher, detection, quantized = _fake_surface()
    model_base = modules["comfy.model_base"]
    root = getattr(model_base, root_name)
    setattr(model_base, future_name, type(future_name, (root,), {}))

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: comfy.model_base.{root_name} "
        f"(accepted subclass set changed: added=['{future_name}'], removed=[])"
    )


def test_rebound_contract_rejects_a_same_arity_parameter_reorder():
    """Same-kind optional arguments can swap values without raising at the call boundary."""
    modules, patcher, detection, quantized = _fake_surface()
    path = "comfy.ldm.krea2.model.SingleStreamDiT._forward"
    value = _resolve(modules["comfy.ldm.krea2.model"], "SingleStreamDiT._forward")
    parameters = list(inspect.signature(value).parameters.values())
    by_name = {parameter.name: index for index, parameter in enumerate(parameters)}
    left = by_name["attention_mask"]
    right = by_name["ref_latents"]
    parameters[left], parameters[right] = parameters[right], parameters[left]
    value.__signature__ = inspect.Signature(parameters)

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {path} "
        "(exact explicit parameter list changed)"
    )


def test_rebound_contract_rejects_a_removed_explicit_parameter():
    modules, patcher, detection, quantized = _fake_surface()
    path = "comfy.ldm.qwen_image.model.QwenImageTransformer2DModel._forward"
    value = _resolve(
        modules["comfy.ldm.qwen_image.model"],
        "QwenImageTransformer2DModel._forward",
    )
    value.__signature__ = inspect.Signature(
        parameter
        for parameter in inspect.signature(value).parameters.values()
        if parameter.name != "additional_t_cond"
    )

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {path} "
        "(exact explicit parameter list changed)"
    )


def test_rebound_contract_allows_only_its_declared_variadics():
    modules, patcher, detection, quantized = _fake_surface()
    assert _assert_fake_surface(modules, patcher, detection, quantized) > 0

    path = "comfy.ldm.anima.model.Anima._forward"
    value = _resolve(modules["comfy.ldm.anima.model"], "Anima._forward")
    value.__signature__ = inspect.Signature(
        parameter
        for parameter in inspect.signature(value).parameters.values()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    )
    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {path} (exact variadic parameters changed)"
    )


def test_assign_must_remain_keyword_capable():
    modules, patcher, detection, quantized = _fake_surface()
    path = "comfy.model_base.BaseModel.load_model_weights"
    value = _resolve(modules["comfy.model_base"], "BaseModel.load_model_weights")
    signature = inspect.signature(value)
    positional = [
        parameter for parameter in signature.parameters.values() if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    value.__signature__ = inspect.Signature(
        [
            *positional,
            inspect.Parameter("assign", inspect.Parameter.POSITIONAL_ONLY),
            inspect.Parameter("unet_prefix", inspect.Parameter.KEYWORD_ONLY),
        ]
    )

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (f"Comfy touchpoint incompatible: {path} (parameter 'assign' is positional-only)")


def test_unused_optional_upstream_parameter_does_not_fail_a_compatible_call():
    modules, patcher, detection, quantized = _fake_surface()
    value = _resolve(modules["comfy.utils"], "ProgressBar")
    parameters = list(inspect.signature(value).parameters.values())
    parameters.append(inspect.Parameter("unused_optional", inspect.Parameter.KEYWORD_ONLY, default=None))
    value.__signature__ = inspect.Signature(parameters)
    assert _assert_fake_surface(modules, patcher, detection, quantized) > 0


def test_zero_argument_call_rejects_a_new_required_parameter():
    modules, patcher, detection, quantized = _fake_surface()
    path = "comfy.model_management.get_torch_device"
    value = _resolve(modules["comfy.model_management"], "get_torch_device")
    value.__signature__ = inspect.Signature(
        [inspect.Parameter("new_required", inspect.Parameter.POSITIONAL_ONLY)]
    )

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {path} (production call shape rejected)"
    )


@pytest.mark.parametrize("class_name", sorted(EXPECTED_MODEL_BASE_TOUCHPOINTS))
def test_deleting_any_adapter_model_class_names_the_exact_touchpoint(class_name: str):
    modules, patcher, detection, quantized = _fake_surface()
    delattr(modules["comfy.model_base"], class_name)
    path = f"comfy.model_base.{class_name}"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == f"Comfy touchpoint missing: {path}"


def test_adapter_model_touchpoint_must_be_a_class_for_isinstance():
    modules, patcher, detection, quantized = _fake_surface()
    modules["comfy.model_base"].Flux = _callable()

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == ("Comfy touchpoint incompatible: comfy.model_base.Flux (not a class)")


@pytest.mark.parametrize("attribute", MODEL_PATCHER_INSTANCE_ATTRIBUTES)
def test_deleting_any_model_patcher_attribute_names_the_exact_touchpoint(attribute: str):
    modules, patcher, detection, quantized = _fake_surface()
    delattr(patcher, attribute)
    path = f"comfy.model_patcher.ModelPatcher.{attribute}"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == f"Comfy touchpoint missing: {path}"


@pytest.mark.parametrize("attribute", model_detection_touchpoints(ADAPTERS))
def test_deleting_any_model_detection_attribute_names_the_exact_touchpoint(attribute: str):
    modules, patcher, detection, quantized = _fake_surface()
    _delete(detection, attribute)
    path = f"comfy.model_base.BaseModel.{attribute}"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == f"Comfy touchpoint missing: {path}"


@pytest.mark.parametrize("format_name", QUANT_ALGORITHM_FORMATS)
def test_deleting_any_quant_algorithm_names_the_exact_touchpoint(format_name: str):
    modules, patcher, detection, quantized = _fake_surface()
    del modules["comfy.quant_ops"].QUANT_ALGOS[format_name]
    path = f"comfy.quant_ops.QUANT_ALGOS[{format_name}]"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == f"Comfy touchpoint missing: {path}"


@pytest.mark.parametrize(
    ("format_name", "field"),
    [(format_name, field) for format_name in QUANT_ALGORITHM_FORMATS for field in QUANT_ALGORITHM_FIELDS],
)
def test_deleting_any_quant_algorithm_field_names_the_exact_touchpoint(
    format_name: str,
    field: str,
):
    modules, patcher, detection, quantized = _fake_surface()
    del modules["comfy.quant_ops"].QUANT_ALGOS[format_name][field]
    path = f"comfy.quant_ops.QUANT_ALGOS[{format_name}].{field}"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == f"Comfy touchpoint missing: {path}"


@pytest.mark.parametrize("format_name", QUANT_ALGORITHM_FORMATS)
def test_deleting_any_quant_layout_params_names_the_exact_touchpoint(format_name: str):
    modules, patcher, detection, quantized = _fake_surface()
    quant_ops = modules["comfy.quant_ops"]
    layout_name = quant_ops.QUANT_ALGOS[format_name]["comfy_tensor_layout"]
    layout = quant_ops.get_layout_class(layout_name)
    del layout.Params
    path = f"comfy.quant_ops.get_layout_class({layout_name!r}).Params"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == f"Comfy touchpoint missing: {path}"


@pytest.mark.parametrize(
    ("format_name", "parameter"),
    [
        (format_name, parameter)
        for format_name, parameters in QUANT_LAYOUT_PARAMETERS.items()
        for parameter in parameters
    ],
)
def test_deleting_any_quant_layout_parameter_names_the_exact_touchpoint(
    format_name: str,
    parameter: str,
):
    modules, patcher, detection, quantized = _fake_surface()
    quant_ops = modules["comfy.quant_ops"]
    layout_name = quant_ops.QUANT_ALGOS[format_name]["comfy_tensor_layout"]
    params = quant_ops.get_layout_class(layout_name).Params
    params.__signature__ = inspect.Signature(
        item for item in inspect.signature(params).parameters.values() if item.name != parameter
    )
    path = f"comfy.quant_ops.get_layout_class({layout_name!r}).Params"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == (
        f"Comfy touchpoint incompatible: {path} (production call shape rejected)"
    )


@pytest.mark.parametrize("attribute", QUANTIZED_TENSOR_INSTANCE_ATTRIBUTES)
def test_deleting_any_quantized_tensor_attribute_names_the_exact_touchpoint(attribute: str):
    modules, patcher, detection, quantized = _fake_surface()
    delattr(quantized, attribute)
    path = f"comfy.quant_ops.QuantizedTensor.{attribute}"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized)
    assert str(error.value) == f"Comfy touchpoint missing: {path}"


@pytest.mark.parametrize("attribute", GUIDER_INSTANCE_ATTRIBUTES)
def test_deleting_any_guider_attribute_names_the_exact_touchpoint(attribute: str):
    modules, patcher, detection, quantized = _fake_surface()
    guider = _Namespace(model_patcher=patcher)
    delattr(guider, attribute)
    path = f"comfy.samplers.CFGGuider.{attribute}"

    with pytest.raises(RuntimeError) as error:
        _assert_fake_surface(modules, patcher, detection, quantized, guider)
    assert str(error.value) == f"Comfy touchpoint missing: {path}"
