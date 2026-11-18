"""ComfyUI's checkpoint prefix rule on key names, and the part a supported release still has.

``comfy.model_detection.unet_prefix_from_state_dict`` picks the prefix that
ComfyUI's diffusion-model load strips; ``unet_prefix`` replays it for the
upstream gate and follows ComfyUI master, as this project does (DESIGN.md
section 6.3). Through release 0.45.5 that function also gives no prefix to a
file with both ``SENSENOVA_KEYS``, SenseNova's layout with its diffusion and
language backbones at the top level, so the load keeps every key of it. Master
dropped the rule on 2026-11-17 (ComfyUI #17790): its checkpoint loader converts
such a file first, and its diffusion-model load picks the prefix by count, as
for any other file. The two answers differ only for a file that holds both keys
and also keys under the prefix master picks. ``keeps_sensenova_whole`` says
which rule the ComfyUI an answer is for has. The rule leaves this file when the
oldest ComfyUI this project supports no longer has it.

Stdlib only, as ``upstream_gate`` is: headless callers read this rule too.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Collection
from functools import lru_cache
from pathlib import Path

# comfy.model_detection.unet_prefix_from_state_dict's candidates, in its order.
_UNET_PREFIXES = ("model.diffusion_model.", "model.model.", "net.")
SENSENOVA_KEYS = (
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
    "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
)
# The other three keys master's checkpoint conversion requires of a SenseNova
# file (comfy/checkpoint_convert.py). A rule keyed on any of them is another rule.
SENSENOVA_OTHER_KEYS = (
    "fm_modules.fm_head.conv1.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.model.norm.weight",
)
_UNDER_NET = tuple(f"net.blocks.{i}.weight" for i in range(6))
# The rule's own two keys, then a whole SenseNova file, each beside six keys under
# net.: the release rule gives both no prefix, master's gives both net., and any
# other pair of answers is a rule this replay does not hold.
PROBES = ((*SENSENOVA_KEYS, *_UNDER_NET), (*SENSENOVA_KEYS, *SENSENOVA_OTHER_KEYS, *_UNDER_NET))
_ANSWERS = {"": True, "net.": False}


def keeps_sensenova_whole(comfy_dir: str | os.PathLike[str] | None = None) -> bool | None:
    """True where the ComfyUI an answer is for has the release rule, False where it
    has master's, None where that cannot be told. It returns, rather than raises, on
    whatever it cannot read.

    Once this process has loaded ``comfy.model_detection``, as the driver has, its
    own function answers ``PROBES``, since that code is what runs: the same answer
    to both is the rule, and any other is None. Otherwise, as at a Worker's first
    load, the source the process would import is read as text (``_rule_in``),
    and a named checkout is read the same way. That reading is of the names and
    the one clause the function holds, not of what it does; the canary holds it
    against the function itself. No ComfyUI to read and a source or function it
    cannot read give None, and the gate then refuses a file if either rule refuses it.
    """
    if comfy_dir:
        return _source_rule([str(Path(comfy_dir).expanduser() / "comfy")])
    live = getattr(sys.modules.get("comfy.model_detection"), "unet_prefix_from_state_dict", None)
    if callable(live):
        try:
            answers = {live(dict.fromkeys(probe)) for probe in PROBES}
        except Exception:  # a rule that reads more than key names
            return None
        return _ANSWERS.get(answers.pop()) if len(answers) == 1 else None
    try:
        package = sys.modules.get("comfy")
        if package is None:
            from importlib.machinery import PathFinder

            spec = PathFinder.find_spec("comfy")
            locations = list(spec.submodule_search_locations or ()) if spec is not None else []
        else:
            locations = list(package.__path__)
    except Exception:  # a stand-in module, or a path entry that cannot be read
        return None
    return _source_rule(locations) if locations else None


def _source_rule(locations: list[str]) -> bool | None:
    """The rule in the source ``comfy.model_detection`` would load from on this search path."""
    from importlib.machinery import PathFinder

    try:
        spec = PathFinder.find_spec("comfy.model_detection", locations)
        get_source = getattr(spec.loader, "get_source", None) if spec is not None else None
        source = get_source(spec.name) if spec is not None and get_source is not None else None
    except Exception:  # an unreadable file, or a loader with no source to give
        return None
    return _rule_in(source) if isinstance(source, str) else None


@lru_cache(maxsize=8)
def _rule_in(source: str) -> bool | None:
    """What the last ``unet_prefix_from_state_dict`` this source defines, the one
    Python runs, names. True when it names both rule keys, no other key of the
    file, and a statement of its own body that reads
    ``if <one key> in <its argument> and <the other> in <its argument>: return ""``.
    False when it names none of the file's keys and no fragment of one. None for
    anything else, and for a function that reads a name its module binds (a
    helper, a constant, an import), whose rule is not all in its text. This reads
    names and one clause, not behaviour: the canary holds the reading against the
    function itself on master and the newest release."""
    import ast

    def keeps_whole(node: ast.stmt, argument: str) -> bool:
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
                and isinstance(node.test.op, ast.And) and len(node.body) == 1
                and isinstance(node.body[0], ast.Return) and isinstance(node.body[0].value, ast.Constant)
                and node.body[0].value.value == ""):
            return False
        keys = {term.left.value for term in node.test.values
                if isinstance(term, ast.Compare) and isinstance(term.left, ast.Constant)
                and len(term.ops) == 1 and isinstance(term.ops[0], ast.In)
                and isinstance(term.comparators[0], ast.Name) and term.comparators[0].id == argument}
        return len(node.test.values) == len(SENSENOVA_KEYS) and keys == set(SENSENOVA_KEYS)

    try:
        tree = ast.parse(source)
    except Exception:  # not Python, or too deep or too large to parse
        return None
    rules = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "unet_prefix_from_state_dict"]
    if not rules or not rules[-1].args.args:
        return None
    bound = {alias.asname or alias.name.split(".")[0] for node in tree.body
             if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    bound |= {node.name for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    bound |= {target.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
              for target in ast.walk(node) if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)}
    if any(isinstance(node, ast.Name) and node.id in bound - {rules[-1].name} for node in ast.walk(rules[-1])):
        return None
    named = {node.value for node in ast.walk(rules[-1])
             if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    if named.isdisjoint((*SENSENOVA_KEYS, *SENSENOVA_OTHER_KEYS)) and not any(
            "_mot_gen" in text for text in named):
        return False
    argument = rules[-1].args.args[0].arg
    return True if (named.issuperset(SENSENOVA_KEYS) and named.isdisjoint(SENSENOVA_OTHER_KEYS)
                    and any(keeps_whole(node, argument) for node in rules[-1].body)) else None


def unet_prefix(keys: Collection[str], sensenova_whole: bool | None = None) -> str:
    """``comfy.model_detection.unet_prefix_from_state_dict`` on key names, under the
    release rule if ``sensenova_whole`` and master's if not. None asks
    ``keeps_sensenova_whole`` and takes master's where that cannot tell; the gate's
    own verdict reads a file under both rules then."""
    if (any(key.startswith("detector.") for key in keys)
            and any(key.startswith("tracker.") for key in keys)):
        return ""
    if all(key in keys for key in SENSENOVA_KEYS) and (
            keeps_sensenova_whole() is True if sensenova_whole is None else sensenova_whole):
        return ""
    counts = dict.fromkeys(_UNET_PREFIXES, 0)
    for key in keys:
        for prefix in _UNET_PREFIXES:
            if key.startswith(prefix):
                counts[prefix] += 1
                break
    top = max(counts, key=counts.__getitem__)
    return top if counts[top] > 5 else "model."
