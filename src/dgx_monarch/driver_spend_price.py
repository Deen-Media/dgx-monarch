"""Price what a node spends on the driver's box before the sampler, where the UNET loader did not charge it.

On a re-queue in the same driver ComfyUI keeps the UNET loader's output when
nothing upstream of it changed, so the loader's graph price
(``nodes/loader_preflight``) does not run, and a node downstream that changed
spends on the driver before any price sees it: a still swapped for a clip at
1344x768 encodes for 94 GiB or more. ComfyUI does not promise to run the loader
before such a node on a first queue either. So the spend is priced at the node
that makes it (docs/DESIGN.md section 5.9):

- What the loader charged for each spend of the prompt is marked
  (``driver_spend_marks``). A spend charged at a figure at least its own steps
  aside; every other spend is priced live, and only in a running graph whose
  DGX Monarch UNET Loader is of a family the loader prices a driver stack for.
- A marked spend is never refused. An unmarked spend is refused only where it
  would take the box under its floor (``driver_footprint.reserve_bytes``)
  after every rescue: kernel MemAvailable plus this process's unallocated torch
  pool, after ``driver_unload.check_with_unload`` unloads what ComfyUI holds
  and trims the Workers' idle caches.
- Priced: Add Guide (MiniMax H3) on its exact canvas and frame count, a clip
  with ``CLIP_MARGIN_BYTES`` more; Reference to Video (MiniMax H3)'s reference
  video; and the stock loader nodes in ``LOADER_CLASSES``, at what the loader
  charges for their inputs. Only these classes' node methods are wrapped, so
  other packs' loaders, ComfyUI's own reloads and every Worker path are out.
- The refusal is class C under the driver-host guard
  ``driver_footprint_preflight``, with no card, as its own type, counted in
  ``driver_unload.CAPACITY_REFUSALS``. Nothing is spent when it refuses.

ComfyUI runs a v3 node through ``EXECUTE_NORMALIZED`` from the class object in
``nodes.NODE_CLASS_MAPPINGS``, which calls ``cls.execute`` on a per-run clone,
and a v1 node through ``getattr(instance, FUNCTION)``; so a method replaced on
those classes at this pack's import is what runs (canary seam 64). A class
another pack registers over one of these names is not wrapped, and the first
loader price logs a WARNING naming it. A Worker installs nothing (its
environment marker); the loader's own switches turn it off, and so does
``DGXM_DISABLE_DRIVER_SPEND_PRICE=1``, for diagnosis. It never raises anything
but its refusal.
"""
from __future__ import annotations

import functools
import inspect
import os
import sys
import threading
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from . import mesh_safety
from .driver_spend_marks import (
    GUIDE_TERM,
    REFERENCE_TERM,
    charged,
    load_bytes,
    node_term,
    remember,
    running_item,
)
from .log import get_logger
from .refusal import RefusalClass, refusal

log = get_logger(__name__)

DISABLE_ENV = "DGXM_DISABLE_DRIVER_SPEND_PRICE"
# Only an actor carries this (cli/proc_identity.BOOTSTRAP_ENV_MARKER); not DGXM_PYTHONPATH,
# which the driver sets at every bring-up and a ComfyUI-Manager restart re-execs with.
WORKER_ENV_MARKERS = ("HYPERACTOR_MESH_BOOTSTRAP_MODE",)
GUARD = "driver_footprint_preflight"
GUIDE_CLASS = "MiniMaxH3AddGuide"
REFERENCE_CLASS = "MiniMaxH3ReferenceToVideo"
# The stock loader nodes whose inputs the loader charges as a text encoder or a
# driver model. Left out: DiffusersLoader (a folder, which the loader charges
# nothing for), CreateHookModelAsLora (its checkpoint becomes hook patches on
# the model the graph renders with), CLIPVisionLoader and the ControlNet,
# upscale and geometry loaders (inputs the loader does not charge).
LOADER_CLASSES = ("CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "QuadrupleCLIPLoader",
                  "LTXAVTextEncoderLoader", "CheckpointLoaderSimple", "CheckpointLoader",
                  "unCLIPCheckpointLoader", "ImageOnlyCheckpointLoader", "UNETLoader")
TROUBLESHOOTING_ENCODE = 82
TROUBLESHOOTING_LOAD = 56
_GIB = 2 ** 30
# Above a clip's measured charge, at the floor of the proof's formula
# (ceil(reading - 17) + 1, at least 1): on 2026-11-13 a 5-frame clip at 448x256,
# DiT resident, DynamicVRAM on, raised the driver's per-process figure 1.16 GiB
# and took the box down 1.23 GiB against its 17.0 charge. Inside the loader's
# charge for a clip, which adds H3's activations (an 11 GiB floor) to it.
CLIP_MARGIN_BYTES = 1 * _GIB

# What this process did, for doctor and the tests.
INSTALLED = "installed"
OFF = "off"
WORKER = "worker"
NOT_LINUX = "not_linux"
NOT_INTEGRATED = "not_integrated"
NO_COMFY = "no_comfy"
UNSET = "unset"

_KEY = "_dgxm_driver_spend_price"
_LOCK = threading.Lock()
_STATE = {"state": UNSET, "warned": ""}  # proc lifetime: written by install_driver and warn_if_unwrapped
# Session lifetime, capped as the marks are: the (prompt id, node id) pairs
# whose step-aside line was logged, so a prompt says it once per node.
_TOLD: dict[str, None] = {}


class DriverSpendCapacityError(RuntimeError):
    """Typed refusal: a node would spend on the driver's box more than it has, where the loader did not charge it."""


class Spend(NamedTuple):
    """One spend: its words, its mark term, the figure the mark is held to, its live charge, how it is said."""

    what: str
    term: str
    own: int
    charge: Callable[[bool], int]  # True once a rescue has run: a VAE it unloaded loads again
    verb: str
    remedy: str
    troubleshooting: int


def state() -> str:
    """What the rule is in this process: one of the names above."""
    return _STATE["state"]


def _set(value: str) -> str:
    _STATE["state"] = value
    return value


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


def guide_frames(image_frames: int) -> int:
    """The frames Add Guide encodes from an image batch of this length: one below five, else 17k+5."""
    if image_frames < 5:
        return 1
    frames = image_frames
    while frames % 17 != 5:
        frames -= 1
    return frames


def running_graph(prompt_id: str | None) -> Mapping[str, Any] | None:
    """The API-format graph of the prompt ComfyUI is running now, by its id; None where none is in hand."""
    item = running_item(prompt_id)
    return item[2] if item is not None and isinstance(item[2], Mapping) else None


def priced_family(graph: Mapping[str, Any]) -> str | None:
    """The first family the loader prices a stack for, among the graph's DGX Monarch UNET Loaders."""
    import folder_paths

    from .adapters.detect import CheckpointSniffError, sniff_checkpoint
    from .driver_footprint import DRIVER_STACK_FAMILIES
    from .family_select import effective_family

    for node in graph.values():
        if not isinstance(node, Mapping) or not str(node.get("class_type", "")).startswith("DGXMonarch"):
            continue
        name = (node.get("inputs") or {}).get("unet_name")
        if not isinstance(name, str) or not name:
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if not path:
            continue
        try:
            family = effective_family(sniff_checkpoint(path)[0], None)
        except CheckpointSniffError:
            continue
        if family in DRIVER_STACK_FAMILIES:
            return family
    return None


def _pool_free_bytes() -> int:
    """What this process's torch pool holds reserved and unallocated; 0 where unread."""
    try:
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return 0
        return max(0, int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated()))
    except Exception:
        return 0


def _active() -> bool:
    from . import driver_footprint

    return (state() == INSTALLED and os.environ.get(DISABLE_ENV) != "1"
            and not driver_footprint.preflight_disabled() and mesh_safety.gpu_is_integrated())


def _told(prompt_id: str, node: str) -> bool:
    key = f"{prompt_id}\x00{node}"
    if key in _TOLD:
        return True
    remember(_TOLD, key, None)
    return False


def _context() -> tuple[str | None, str]:
    """The executing prompt's id and node id, or (None, "")."""
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
    except Exception:
        return None, ""
    prompt_id = getattr(context, "prompt_id", None)
    return (prompt_id if isinstance(prompt_id, str) and prompt_id else None,
            str(getattr(context, "node_id", "") or ""))


def _refusal_text(spend: Spend, needed: int, room: int, floor: int, reached: bool) -> str:
    why = ("The UNET Loader priced this prompt but did not charge this at its size."
           if reached else
           "The UNET Loader has not priced this prompt: ComfyUI kept its output from an earlier queue, "
           "or runs this node before it.")
    return (f"{spend.what} needs {_gib(needed)} GiB on the driver's box, and the box has "
            f"{_gib(max(0, room - floor))} GiB above its {_gib(floor)} GiB floor after ComfyUI's models were "
            f"unloaded and the Workers' idle caches released. {why} Nothing was {spend.verb}. {spend.remedy} "
            "Clearing the Sparks (Reset attached mesh in the sidebar) also frees what the Workers hold on this "
            "box, and the next queue then runs the loader, which itemises the whole graph's figures.")


def check(spend: Spend, prompt_id: str | None = None) -> None:
    """Admit ``spend.charge(rescued)`` bytes now, or refuse after the driver's rescues; log what admitted it."""
    from . import driver_footprint, driver_unload, vae_decode_relief
    from .nodes import graph_load_price

    floor = driver_footprint.reserve_bytes(None)
    reached = prompt_id is not None and graph_load_price.priced_any_this_prompt(prompt_id)
    seen: list[tuple[int, int, int]] = []
    runs = [0]  # check_with_unload runs the step again only after a rescue

    def step() -> None:
        runs[0] += 1
        avail = mesh_safety.mem_available_bytes()
        if avail is None:
            return  # off Linux: MemAvailable bounds nothing here
        pool = _pool_free_bytes()
        needed = spend.charge(runs[0] > 1)
        if avail + pool - floor >= needed:
            seen.append((needed, avail, pool))
            return
        raise DriverSpendCapacityError(refusal(
            RefusalClass.CAPACITY, _refusal_text(spend, needed, avail + pool, floor, reached),
            guard=GUARD, waivable=False, troubleshooting=spend.troubleshooting))

    try:
        fleet, _why = vae_decode_relief._serving()
    except Exception:
        fleet = None
    driver_unload.check_with_unload([step], handle=fleet, subject=spend.what)
    if seen:
        needed, avail, pool = seen[-1]
        log.info("driver spend price: %s admitted: %s GiB charged against %s GiB of room above the %s GiB "
                 "floor (MemAvailable %s GiB, this process's free torch pool %s GiB)", spend.what,
                 _gib(needed), _gib(avail + pool - floor), _gib(floor), _gib(avail), _gib(pool))


def _price(name: str, spend_of: Callable[[Mapping[str, Any], str], Spend | None]) -> None:
    """Price this node's spend where the loader did not charge it; any fault but the refusal steps aside."""
    try:
        if not _active():
            return
        prompt_id, node = _context()
        graph = running_graph(prompt_id)
        spend = None if prompt_id is None or graph is None else spend_of(graph, node)
        if spend is None:
            return
        held = charged(prompt_id, spend.term)  # held to the encode alone: the loader charges the VAE as a model
        if held and held >= spend.own:
            if not _told(prompt_id, node):
                log.info("driver spend price: the loader priced this prompt; nothing priced here (%s, node %s: "
                         "%s GiB charged, %s GiB needed)", name, node, _gib(held), _gib(spend.own))
            return
        if priced_family(graph) is None:
            log.debug("driver spend price: not a graph of a family the loader prices; %s is not priced", name)
            return
    except Exception as exc:
        log.debug("driver spend price stepped aside for %s (%r)", name, exc)
        return
    try:
        check(spend, prompt_id)
    except DriverSpendCapacityError:
        raise
    except Exception as exc:
        log.warning("driver spend price stepped aside for %s (%s: %s); ComfyUI's own call runs unpriced",
                    spend.what, type(exc).__name__, exc)


ENCODE_REMEDY = "Use a shorter clip or a smaller canvas, or free memory on this box."
LOAD_REMEDY = "Pick a smaller file, or free memory on this box."


def _vae_reload_bytes(vae: Any) -> int:
    """The VAE's size where ComfyUI does not hold it loaded (a rescue unloaded it), else 0."""
    try:
        mm = sys.modules.get("comfy.model_management")
        patcher = getattr(vae, "patcher", None)
        if mm is None or patcher is None:
            return 0
        if any(getattr(entry, "model", None) is patcher for entry in mm.current_loaded_models):
            return 0
        return int(patcher.model_size())
    except Exception:
        return 0


def _guide_spend(arguments: Mapping[str, Any]) -> Callable[[Mapping[str, Any], str], Spend | None]:
    def spend_of(graph: Mapping[str, Any], node: str) -> Spend | None:
        from . import h3_activation
        from .h3_calibration import H3_GUIDE_CLIP_FLOOR_FRAMES
        from .nodes import loader_graph

        image, latent = arguments.get("image"), arguments.get("latent")
        if image is None or not isinstance(latent, Mapping):
            return None  # an audio guide encodes no frames here
        video = latent["samples"].tensors[0]
        height, width = int(video.shape[3]) * 16, int(video.shape[4]) * 16
        frames = guide_frames(int(image.shape[0]))
        base = h3_activation.guide_encode_bytes(width * height, frames)
        clip = frames >= H3_GUIDE_CLIP_FLOOR_FRAMES
        if clip and loader_graph.graph_activation_bytes(graph, 1, family=h3_activation.H3_FAMILY)[0] > 0:
            base += CLIP_MARGIN_BYTES
        kind = f"this {frames}-frame clip" if clip else "this still"
        vae = arguments.get("vae")
        return Spend(f"Add Guide (MiniMax H3) for {kind} at {width}x{height}", GUIDE_TERM, base,
                     lambda rescued: base + (_vae_reload_bytes(vae) if rescued else 0), "encoded", ENCODE_REMEDY,
                     TROUBLESHOOTING_ENCODE)
    return spend_of


def _reference_spend(arguments: Mapping[str, Any]) -> Callable[[Mapping[str, Any], str], Spend | None]:
    def spend_of(graph: Mapping[str, Any], node: str) -> Spend | None:
        from .nodes import loader_graph

        videos = arguments.get("ref_videos")
        if not (isinstance(videos, Mapping) and any(value is not None for value in videos.values())):
            return None
        figure = loader_graph.REF_VIDEO_ENCODE_BYTES
        return Spend("Reference to Video (MiniMax H3) for its reference video", REFERENCE_TERM, figure,
                     lambda rescued: figure, "encoded", ENCODE_REMEDY, TROUBLESHOOTING_ENCODE)
    return spend_of


def _load_spend(arguments: Mapping[str, Any], name: str) -> Callable[[Mapping[str, Any], str], Spend | None]:
    def spend_of(graph: Mapping[str, Any], node: str) -> Spend | None:
        size = load_bytes(arguments, name)
        if not size:
            return None
        files = ", ".join(os.path.basename(str(value)) for key, value in arguments.items()
                          if isinstance(key, str) and isinstance(value, str) and value
                          and (key.startswith(("clip_name", "text_encoder")) or key in ("ckpt_name", "unet_name")))
        return Spend(f"{name} loading {files}", node_term(node), size, lambda rescued: size, "loaded", LOAD_REMEDY,
                     TROUBLESHOOTING_LOAD)
    return spend_of


def _spend_for(name: str, arguments: Mapping[str, Any]) -> Callable[[Mapping[str, Any], str], Spend | None]:
    if name == GUIDE_CLASS:
        return _guide_spend(arguments)
    if name == REFERENCE_CLASS:
        return _reference_spend(arguments)
    return _load_spend(arguments, name)


def _wrap_classmethod(original: Callable[..., Any], name: str) -> Callable[..., Any]:
    """A v3 node's ``execute`` with its spend priced before it; nothing else changes."""
    signature = inspect.signature(original)

    def priced(cls: Any, args: tuple, kwargs: dict) -> None:
        try:
            arguments = signature.bind(cls, *args, **kwargs).arguments
        except TypeError:
            return  # ComfyUI's own call raises the same fault
        _price(name, _spend_for(name, arguments))

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def execute_async(cls: Any, *args: Any, **kwargs: Any) -> Any:
            priced(cls, args, kwargs)
            return await original(cls, *args, **kwargs)
        setattr(execute_async, _KEY, True)
        return execute_async

    @functools.wraps(original)
    def execute(cls: Any, *args: Any, **kwargs: Any) -> Any:
        priced(cls, args, kwargs)
        return original(cls, *args, **kwargs)

    setattr(execute, _KEY, True)
    return execute


def _wrap_method(original: Callable[..., Any], name: str) -> Callable[..., Any]:
    """A v1 node's ``FUNCTION`` method with its spend priced before it; nothing else changes."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            arguments = signature.bind(self, *args, **kwargs).arguments
        except TypeError:
            arguments = None  # ComfyUI's own call raises the same fault
        if arguments is not None:
            _price(name, _spend_for(name, arguments))
        return original(self, *args, **kwargs)

    setattr(method, _KEY, True)
    return method


WRAPPED_CLASSES = (GUIDE_CLASS, REFERENCE_CLASS, *LOADER_CLASSES)


def _wrapped(cls: Any) -> bool | None:
    """Whether ``cls``'s node method carries the wrap; None where it has no method this can wrap."""
    execute = cls.__dict__.get("execute")
    if isinstance(execute, classmethod):
        return bool(getattr(execute.__func__, _KEY, False))
    function = getattr(cls, "FUNCTION", None)
    method = cls.__dict__.get(function) if isinstance(function, str) else None
    return bool(getattr(method, _KEY, False)) if inspect.isfunction(method) else None


def install() -> list[str]:
    """Wrap the node method of each class in ``WRAPPED_CLASSES`` that ComfyUI's mapping holds, once.

    Returns what it wrapped now.
    """
    wrapped: list[str] = []
    mapping = getattr(sys.modules.get("nodes"), "NODE_CLASS_MAPPINGS", None) or {}
    with _LOCK:
        for name in WRAPPED_CLASSES:
            cls = mapping.get(name)
            if cls is None or _wrapped(cls) is not False:
                continue
            execute = cls.__dict__.get("execute")
            if isinstance(execute, classmethod):
                cls.execute = classmethod(_wrap_classmethod(execute.__func__, name))
            else:
                setattr(cls, cls.FUNCTION, _wrap_method(cls.__dict__[cls.FUNCTION], name))
            wrapped.append(name)
    return wrapped


def warn_if_unwrapped() -> None:
    """Once a process: name each class in ComfyUI's mapping that does not carry the wrap. Never raises."""
    if state() != INSTALLED or _STATE["warned"]:
        return
    try:
        mapping = getattr(sys.modules.get("nodes"), "NODE_CLASS_MAPPINGS", None) or {}
        found = {name: _wrapped(mapping[name]) for name in WRAPPED_CLASSES if name in mapping}
    except Exception:
        return
    _STATE["warned"] = "checked"
    for state_of, why in ((False, "another pack registered over it, or a node replacement"),
                          (None, "its node method is inherited, so this pack could not wrap it")):
        missed = [name for name, wrapped in found.items() if wrapped is state_of]
        if missed:
            log.warning("driver spend price: %s in ComfyUI's node mapping is not wrapped (%s), so its spend on "
                        "the driver is not priced where the loader did not charge it", ", ".join(missed), why)


def install_driver() -> str:
    """The ComfyUI process's hook, at node import. Never raises; returns what it did, kept for ``state``."""
    if any(os.environ.get(marker) for marker in WORKER_ENV_MARKERS):
        log.info("this process is a Worker's actor (%s is set): the driver spend price is not installed here",
                 ", ".join(WORKER_ENV_MARKERS))
        return _set(WORKER)
    if os.environ.get(DISABLE_ENV) == "1":
        log.info("%s=1: nothing prices a node's spend on the driver where the loader did not", DISABLE_ENV)
        return _set(OFF)
    if not sys.platform.startswith("linux"):
        return _set(NOT_LINUX)
    from . import driver_uma_memory

    try:
        model_management, torch = driver_uma_memory.comfy_modules()
        integrated = driver_uma_memory.integrated_total_bytes(torch, model_management.get_torch_device())
    except Exception:
        return _set(NO_COMFY)
    if integrated is None:
        return _set(NOT_INTEGRATED)
    try:
        wrapped = install()
    except Exception as exc:
        log.warning("the driver spend price could not be installed (%s: %s); nothing prices a node's "
                    "spend where the loader did not", type(exc).__name__, exc)
        return _set(NO_COMFY)
    log.info("unified memory: a node's spend on the driver that the loader did not charge is priced where it "
             "happens (%s; %s=1 turns this off)", ", ".join(wrapped) or "already in place", DISABLE_ENV)
    return _set(INSTALLED)
