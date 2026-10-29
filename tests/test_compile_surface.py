"""compile_dit's surface: which families it compiles, and the residency rule that follows.

The block compiler (actor/worker_compile.py) compiles a top-level ``blocks``
list and nothing else. compile_policy.KNOWN_COMPILE_NOOP_FAMILIES lists the
families whose DiT has no such list, so compile_dit leaves them eager and does
not take their slab residency away. Three things hold that list true:

* every registered adapter family sits on exactly one side, so a new family
  cannot be forgotten and the list holds no stale name;
* each block list a supported family uses, faked here, compiles nothing and is
  named in the eager warning, while a ``blocks`` list beside it compiles;
* where a ComfyUI checkout is present (COMFYUI_DIR, default ~/ComfyUI), each
  family's DiT class assigns ``self.blocks`` exactly when the family is on the
  compiled side, and comfy.model_base still builds that very class (its
  ``unet_model=`` target), so a class ComfyUI stops using cannot stand in for
  the one it moved to. The source is parsed, never imported, so this runs on
  any CPU.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from torch import nn

from dgx_monarch import compile_policy
from dgx_monarch.actor import worker_compile

# The families whose DiT has a top-level `blocks` list on ComfyUI 0.43.0.
COMPILED_FAMILIES = frozenset({
    "anima", "cogvideo", "krea2", "minimax_h3", "wan", "wan_animate2", "wan_dancer", "wan_scail",
})

# Each family's DiT classes, as comfy.model_base builds them: the file under
# comfy/ldm, the class, and the block lists it or a base assigns.
_FLUX = (("flux/model.py", "Flux", ("double_blocks", "single_blocks")),)
_REFINERS = ("noise_refiner", "context_refiner", "layers")
DIT_CLASSES: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "krea2": (("krea2/model.py", "SingleStreamDiT", ("blocks",)),),
    "minimax_h3": (("minimax/model.py", "MiniMaxH3Model", ("blocks",)),),
    "wan": (("wan/model.py", "WanModel", ("blocks",)),),
    "wan_scail": (("wan/model.py", "SCAILWanModel", ("blocks",)),
                  ("wan/model.py", "SCAIL2WanModel", ("blocks",))),
    "wan_animate2": (("wan/model_animate2.py", "WanAnimate2Model", ("blocks",)),),
    "wan_dancer": (("wan/model_wandancer.py", "WanDancerModel", ("blocks",)),),
    "cogvideo": (("cogvideo/model.py", "CogVideoXTransformer3DModel", ("blocks",)),),
    "anima": (("anima/model.py", "Anima", ("blocks",)),),
    "qwen_image": (("qwen_image/model.py", "QwenImageTransformer2DModel", ("transformer_blocks",)),),
    "qwen_image21": (("qwen_image21/model.py", "QwenImage21Transformer2DModel", ("transformer_blocks",)),),
    "mage_flow": (("mage_flow/model.py", "MageFlowTransformer2DModel", ("transformer_blocks",)),),
    "lens": (("lens/model.py", "LensTransformer2DModel", ("transformer_blocks",)),),
    "ltx": (("lightricks/model.py", "LTXVModel", ("transformer_blocks",)),
            ("lightricks/av_model.py", "LTXAVModel", ("transformer_blocks",))),
    "flux": _FLUX,
    "flux2": _FLUX,
    "longcat": _FLUX,
    "chroma": (("chroma/model.py", "Chroma", ("double_blocks", "single_blocks")),
               ("chroma_radiance/model.py", "ChromaRadiance",
                ("double_blocks", "single_blocks", "nerf_blocks"))),
    "hunyuan": (("hunyuan_video/model.py", "HunyuanVideo", ("double_blocks", "single_blocks")),),
    "ideogram4": (("ideogram4/model.py", "Ideogram4Transformer2DModel", ("layers",)),),
    "moss_image": (("moss_image/model.py", "MOSSImage", ("layers",)),),
    "ernie": (("ernie/model.py", "ErnieImageModel", ("layers",)),),
    "zimage": (("lumina/model.py", "NextDiT", _REFINERS),
               ("lumina/model.py", "NextDiTPixelSpace", _REFINERS)),
    "omnigen2": (("omnigen/omnigen2.py", "OmniGen2Transformer2DModel",
                  ("noise_refiner", "ref_image_refiner", "context_refiner", "layers")),),
    "boogu": (("boogu/model.py", "BooguTransformer2DModel",
               ("noise_refiner", "ref_image_refiner", "context_refiner",
                "double_stream_layers", "single_stream_layers")),),
    "flux3": (("flux3/model.py", "Flux3", ("content_mode_blocks", "txt_mode_blocks", "single_blocks")),),
    "pixeldit_comfy": (("pixeldit/model.py", "PixDiT_T2I", ("patch_blocks", "pixel_blocks")),
                       ("pixeldit/pid.py", "PidNet", ("patch_blocks", "pixel_blocks"))),
    "kandinsky5": (("kandinsky5/model.py", "Kandinsky5",
                    ("text_transformer_blocks", "visual_transformer_blocks")),),
}

# Every block list a supported family names other than `blocks`.
OTHER_LISTS = sorted({name for rows in DIT_CLASSES.values() for _f, _c, lists in rows
                      for name in lists} - {"blocks"})


def test_every_registered_family_sits_on_exactly_one_side():
    from dgx_monarch.adapters import ADAPTERS

    families = {adapter.family for adapter in ADAPTERS}
    noop = compile_policy.KNOWN_COMPILE_NOOP_FAMILIES
    assert not COMPILED_FAMILIES & noop
    assert families - COMPILED_FAMILIES - noop == set(), "a registered family is on neither side"
    assert (COMPILED_FAMILIES | noop) - families == set(), "a listed family is not registered"
    assert set(DIT_CLASSES) == families


@pytest.mark.parametrize("family", sorted(DIT_CLASSES))
def test_the_table_has_a_blocks_list_exactly_on_the_compiled_side(family):
    has_blocks = any("blocks" in lists for _f, _c, lists in DIT_CLASSES[family])
    assert has_blocks is (family in COMPILED_FAMILIES)


@pytest.mark.parametrize("family", sorted(DIT_CLASSES))
def test_compile_keeps_only_a_compiled_family_stock_resident(family):
    compiled = family in COMPILED_FAMILIES
    assert compile_policy.compile_dit_is_known_noop(family) is not compiled
    assert compile_policy.compile_dit_blocks_slab(family, True) is compiled
    assert compile_policy.compile_dit_blocks_slab(family, False) is False


@pytest.mark.parametrize("family", (None, "unknown", "a_family_added_later"))
def test_a_family_not_known_eager_stays_stock_under_compile(family):
    assert not compile_policy.compile_dit_is_known_noop(family)
    assert compile_policy.compile_dit_blocks_slab(family, True) is True


class _Log:
    def __init__(self):
        self.warnings, self.infos = [], []

    def warning(self, message, *args):
        self.warnings.append(message % args)

    def info(self, message, *args):
        self.infos.append(message % args)


class _Block(nn.Module):
    """A block whose compile only records the call."""

    def __init__(self):
        super().__init__()
        self.compiled = []

    def compile(self, *args, **kwargs):
        self.compiled.append(kwargs)


def _block_list(name: str) -> nn.Module:
    if name == "content_mode_blocks":  # FLUX.3: one list per content mode
        return nn.ModuleDict({"image": nn.ModuleList([_Block(), _Block()])})
    return nn.ModuleList([_Block(), _Block()])


def _blocks_of(model: nn.Module) -> list[_Block]:
    return [module for module in model.modules() if isinstance(module, _Block)]


@pytest.fixture
def compile_on(monkeypatch):
    log = _Log()
    monkeypatch.setattr(worker_compile, "log", log)
    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")
    return log


@pytest.mark.parametrize("name", OTHER_LISTS)
def test_each_other_block_list_compiles_nothing_and_is_named(compile_on, name):
    model = nn.Module()
    model.add_module(name, _block_list(name))

    worker_compile.maybe_compile_dit(model)

    assert not any(block.compiled for block in _blocks_of(model))
    assert compile_on.warnings == [
        f"compile_dit: Module has no blocks list (its blocks are in {name}), so nothing was "
        "compiled; running eager (docs/TROUBLESHOOTING.md #14)"]
    assert not compile_on.infos


def test_the_warning_names_only_block_lists():
    model = nn.Module()
    model.add_module("double_blocks", _block_list("double_blocks"))
    model.add_module("single_blocks", _block_list("single_blocks"))
    model.add_module("txt_norm", nn.ModuleList([nn.Identity()]))  # Lens: a list, not blocks
    model.add_module("emb_in", nn.ModuleDict({"image": nn.Identity()}))  # FLUX.3: Linears by mode
    model.add_module("final_layer", nn.Identity())

    assert worker_compile._block_lists(model) == "double_blocks, single_blocks"
    assert worker_compile._block_lists(object()) == ""


def test_a_blocks_list_compiles_and_a_list_beside_it_stays_eager(compile_on, monkeypatch):
    import torch._inductor.config as inductor_config

    # The compile path sets both process-wide; monkeypatch restores them.
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    monkeypatch.setattr(inductor_config, "max_autotune_gemm_backends",
                        inductor_config.max_autotune_gemm_backends)
    model = nn.Module()
    model.add_module("blocks", _block_list("blocks"))
    model.add_module("refiner_blocks", _block_list("refiner_blocks"))

    worker_compile.maybe_compile_dit(model)

    assert all(block.compiled == [{"mode": "max-autotune-no-cudagraphs", "dynamic": False}]
               for block in model.blocks)
    assert not any(block.compiled for block in model.refiner_blocks)
    assert not compile_on.warnings
    assert compile_on.infos == ["DiT: 2 blocks compiled (max-autotune-no-cudagraphs)"]


# ComfyUI's own source, parsed: the classes each family's DiT is built from.

def _ldm() -> Path | None:
    root = Path(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    ldm = root / "comfy" / "ldm"
    return ldm if ldm.is_dir() else None


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


@pytest.fixture(scope="module")
def comfy_classes():
    """Top-level classes under comfy/ldm, keyed by (file, name), plus each file's tree."""
    ldm = _ldm()
    if ldm is None:
        pytest.skip("no ComfyUI checkout at COMFYUI_DIR or ~/ComfyUI")
    classes: dict[tuple[Path, str], ast.ClassDef] = {}
    trees: dict[Path, ast.Module] = {}
    for path in sorted(ldm.rglob("*.py")):
        tree = _parse(path)
        if tree is None:
            continue
        trees[path] = tree
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[(path, node.name)] = node
    return ldm, classes, trees


def _self_assigned(cls: ast.ClassDef) -> set[str]:
    """The attributes the class's own methods assign on self.

    Every method counts, not only __init__: LTX builds its list in a helper
    that __init__ calls.
    """
    names: set[str] = set()
    for item in cls.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        for node in ast.walk(item):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    names.add(target.attr)
    return names


def _base_file(ldm: Path, tree: ast.Module, path: Path, name: str) -> Path | None:
    """Where `name` comes from in this file: an import of it, or the file itself."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(alias.name == name for alias in node.names):
            if node.level:
                package = path.parent
                for _ in range(node.level - 1):
                    package = package.parent
                return package / (node.module or "").replace(".", "/")
            if node.module and node.module.startswith("comfy.ldm."):
                return ldm / node.module.removeprefix("comfy.ldm.").replace(".", "/")
    return path.with_suffix("")


def _chain_assigned(ldm: Path, classes, trees, path: Path, cls: ast.ClassDef) -> set[str]:
    """What the class and its bases under comfy/ldm assign on self."""
    names: set[str] = set()
    pending, seen = [(path, cls)], set()
    while pending:
        path, cls = pending.pop()
        if (path, cls.name) in seen:
            continue
        seen.add((path, cls.name))
        names |= _self_assigned(cls)
        for base in cls.bases:
            if not isinstance(base, ast.Name):
                continue
            stem = _base_file(ldm, trees[path], path, base.id)
            found = classes.get((stem.with_suffix(".py"), base.id)) if stem is not None else None
            if found is not None:
                pending.append((stem.with_suffix(".py"), found))
    return names


def _dotted(node: ast.expr) -> list[str]:
    """``a.b.C`` as ``["a", "b", "C"]``; empty for anything but a name chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return []
    parts.append(node.id)
    return parts[::-1]


@pytest.fixture(scope="module")
def model_base_targets(comfy_classes) -> set[tuple[str, str]]:
    """(comfy/ldm-relative file, class) of every ``unet_model=`` comfy.model_base passes.

    model_base builds a family's DiT by handing the class to BaseModel as
    ``unet_model=comfy.ldm.<module>.<Class>``. This is the set of classes
    ComfyUI builds today, so the table above cannot keep checking a class
    ComfyUI has moved away from.
    """
    ldm, _classes, _trees = comfy_classes
    tree = _parse(ldm.parents[1] / "comfy" / "model_base.py")
    assert tree is not None, "comfy/model_base.py is missing or does not parse"
    # The class arrives either as a keyword of the super().__init__ call or as
    # the default of the model's own ``unet_model`` parameter (Flux, Chroma and
    # Qwen Image pass it that way), so both forms count.
    values: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            values += [keyword.value for keyword in node.keywords if keyword.arg == "unet_model"]
        elif isinstance(node, ast.FunctionDef):
            positional = node.args.posonlyargs + node.args.args
            for arg, default in zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults, strict=True):
                if arg.arg == "unet_model":
                    values.append(default)
            values += [default for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
                       if arg.arg == "unet_model" and default is not None]
    targets: set[tuple[str, str]] = set()
    for value in values:
        parts = _dotted(value)
        if len(parts) > 3 and parts[:2] == ["comfy", "ldm"]:
            targets.add(("/".join(parts[2:-1]) + ".py", parts[-1]))
    assert targets, "comfy/model_base.py names no comfy.ldm class as unet_model"
    return targets


@pytest.mark.parametrize("family", sorted(DIT_CLASSES))
def test_each_familys_comfy_dit_has_blocks_exactly_on_the_compiled_side(comfy_classes, model_base_targets, family):
    ldm, classes, trees = comfy_classes
    for relative, name, lists in DIT_CLASSES[family]:
        path = ldm / relative
        cls = classes.get((path, name))
        assert cls is not None, f"{family}: ComfyUI has no class {name} in comfy/ldm/{relative}"
        assert (relative, name) in model_base_targets, (
            f"{family}: comfy.model_base no longer builds {name} from comfy/ldm/{relative}; "
            "the family moved to another class, so this table checks a class ComfyUI does not use")
        assigned = _chain_assigned(ldm, classes, trees, path, cls)
        assert ("blocks" in assigned) is (family in COMPILED_FAMILIES), (
            f"{family}: {name} {'assigns' if 'blocks' in assigned else 'does not assign'} "
            "self.blocks, so compile_policy.KNOWN_COMPILE_NOOP_FAMILIES is wrong for it")
        assert set(lists) <= assigned, f"{family}: {name} assigns none of {sorted(set(lists) - assigned)}"
