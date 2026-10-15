"""Attention kernels a model pins per module: detection, the render record, the waiver.

ComfyUI lets a model file pin an attention kernel for each attention module.
Its ``comfy_attention`` child (a ``ComfyAttention``) reads a uint8 JSON tensor
at ``<module>.comfy_attention.config`` when the checkpoint loads, keeps the JSON
in ``config`` and binds the kernel in ``function``, or leaves ``function`` None
when that kernel is unknown or cannot run on this device. A LoRA can carry the
same key: ComfyUI's patcher stages a new ``ComfyAttention`` as an object patch
at ``diffusion_model.<module>.comfy_attention``, which ``patch_model`` puts in
place before a render. ``wrap_attn`` runs ``function`` unless the call carries
an ``optimized_attention_override``.

Under sequence parallelism no attention call runs a pin (docs/ADAPTERS.md).
Every topology without it (on a two-rank pair: single, cfg2, dp2, cfg2+fsdp and
dp2+fsdp) keeps every stock call, so a pin runs as it does on one GPU, except
where the FSDP shard build cleared it (``note_injected``). This module finds the
pins, says in the render record which ran, and refuses a render whose pin would
run an approximate kernel until the operator grants the class-K Sol-Attn waiver
(docs/TROUBLESHOOTING.md #117).

Cost: one ``named_modules`` walk per fresh load (``scan_loaded``), cached on the
loaded model. Each render reads that cache and the patcher's object patches, a
dict of a few entries; nothing runs per step.

Standard library only at import, so the driver reads the record helpers
without loading torch.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final

# The names ComfyUI gives the pin; canary seam 39 holds each against the tree.
PIN_MODULE: Final = "comfy_attention"
CONFIG_ATTR: Final = "config"
FUNCTION_ATTR: Final = "function"
# The kernels ComfyUI's ComfyAttention binds, by the function it binds.
KERNEL_FUNCTIONS: Final = {
    "attention_comfy_kitchen_int8": "comfy_kitchen_int8",
    "attention_comfy_kitchen_sol": "comfy_kitchen_sol",
}
KNOWN_KERNELS: Final = frozenset(KERNEL_FUNCTIONS.values())
# Kernels known to compute exact attention. Both ComfyUI kernels are
# approximate by construction: comfy_kitchen_sol attends a routed subset of key
# blocks exactly and pools the rest, and comfy_kitchen_int8 rounds q, k and v
# to signed 8-bit integers and the softmax weights to unsigned ones
# (comfy_kitchen/sage_attention.py, int8_attention). A kernel not listed here
# counts as approximate, so a new one cannot run unasked.
EXACT_KERNELS: frozenset[str] = frozenset()

# The cached load scan, set on the loaded model so every clone of its patcher
# reads the same one.
_SCAN_ATTR: Final = "_dgxm_attention_pin_scan"
# One guard for the pin route, under the Sol-Attn waiver kind. The refusal
# ledger reads it from this constant.
SOL_ATTN_PIN_GUARD = "sol_attn:checkpoint_pin"
RESULT_KEY: Final = "attention_pins"
RECORD_FIELDS: Final = ("attention_pins", "attention_pins_ran", "attention_pins_dropped")

DROPPED_SEQUENCE_PARALLEL: Final = "sequence parallelism"
DROPPED_FSDP: Final = "FSDP shard build"
DROPPED_UNBOUND: Final = "kernel unavailable on this host"

_KERNEL_RISK: Final = {
    "comfy_kitchen_sol": ("comfy_kitchen_sol attends a routed subset of key blocks exactly "
                          "and pools the rest, so it is not identity preserving"),
    "comfy_kitchen_int8": ("comfy_kitchen_int8 rounds q, k, v and the attention weights "
                           "to 8-bit integers"),
}


@dataclass(frozen=True, slots=True)
class BlockPin:
    """One attention module's pin: the kernel, its tau, and whether it can run."""

    kernel: str
    tau: float | None = None
    bound: bool = False
    cleared: bool = False

    @property
    def label(self) -> str:
        return self.kernel if self.tau is None else f"{self.kernel} tau={self.tau:g}"


def _tau(value: Any) -> float | None:
    """A Sol tau as a float, or None for a value that is not a number.

    ComfyUI binds whatever the JSON holds; a malformed tau fails in its kernel
    call, where the pin runs, never in this read.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _declared(config: Any) -> list[tuple[str, float | None]]:
    """Every kernel a config names, in ComfyUI's order (a dict or a list of them)."""
    out: list[tuple[str, float | None]] = []
    for entry in config if isinstance(config, list) else [config]:
        method = entry.get("attention") if isinstance(entry, Mapping) else None
        if isinstance(method, str) and method:
            tau = entry.get("tau", 1.0) if method == "comfy_kitchen_sol" else None
            out.append((method, _tau(tau)))
    return out


def block_pin(module: Any) -> BlockPin | None:
    """Read one ``comfy_attention`` module, or None when it pins nothing.

    A bound ``function`` names the kernel ComfyUI will run. With none bound,
    the first kernel the config names that ComfyUI knows is the one it tried;
    a config naming only unknown kernels pins nothing, as ComfyUI ignores it.
    """
    config = getattr(module, CONFIG_ATTR, None)
    if config is None:
        return None
    declared = _declared(config)
    function = getattr(module, FUNCTION_ATTR, None)
    if function is not None:
        target = getattr(function, "func", function)
        kernel = KERNEL_FUNCTIONS.get(getattr(target, "__name__", ""))
        if kernel is not None:
            keywords = getattr(function, "keywords", None)
            tau = keywords.get("tau") if isinstance(keywords, Mapping) else None
            return BlockPin(kernel, _tau(tau), bound=True)
        name, tau = declared[0] if declared else (getattr(target, "__name__", "unknown"), None)
        return BlockPin(name, tau, bound=True)
    for name, tau in declared:
        if name in KNOWN_KERNELS:
            return BlockPin(name, tau, bound=False)
    return None


def scan(root: Any, prefix: str = "diffusion_model") -> dict[str, BlockPin]:
    """Walk ``root`` once and return every pinned ``comfy_attention`` by path."""
    walk = getattr(root, "named_modules", None)
    if walk is None:
        return {}
    pins: dict[str, BlockPin] = {}
    for name, module in walk(prefix=prefix):
        if name.rpartition(".")[2] == PIN_MODULE:
            pin = block_pin(module)
            if pin is not None:
                pins[name] = pin
    return pins


def scan_loaded(base_patcher: Any) -> dict[str, BlockPin]:
    """Cache the checkpoint's pins on a freshly loaded model, before injection.

    It runs before any LoRA is staged or baked, so it reads the checkpoint
    alone, and before the FSDP shard build, which can clear a pin.
    """
    model = getattr(base_patcher, "model", None)
    if model is None:
        return {}
    pins = scan(getattr(model, "diffusion_model", None))
    _cache(model, pins)
    return pins


def _cache(model: Any, pins: dict[str, BlockPin]) -> None:
    """Keep the scan on the model, which every clone of its patcher shares.

    A ComfyUI model is a torch module and takes the attribute; a stand-in
    that cannot is scanned again at each read instead.
    """
    try:
        setattr(model, _SCAN_ATTR, pins)
    except (AttributeError, TypeError):
        pass


def _resolve(model: Any, path: str) -> Any:
    """The module at a dotted path under ``model``, or None."""
    node = model
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def note_injected(base_patcher: Any) -> None:
    """Mark the pins injection cleared. Reads nothing when the checkpoint pins nothing.

    The FSDP shard build of a bf16 or fp16 checkpoint (``build_shards``)
    reloads the sharded weights through ``load_state_dict``, and
    ``ComfyAttention`` clears its config on every load whose state dict lacks
    the key, so those pins no longer run. An fp8 or int8 checkpoint takes the
    direct wrap, which reloads nothing, so its pins stay and run. This rereads
    each live config rather than assume either path.
    """
    model = getattr(base_patcher, "model", None)
    pins = getattr(model, _SCAN_ATTR, None) if model is not None else None
    if not pins:
        return
    for path, pin in list(pins.items()):
        if getattr(_resolve(model, path), CONFIG_ATTR, None) is None:
            pins[path] = replace(pin, cleared=True, bound=False)


def _checkpoint_pins(patcher: Any) -> dict[str, BlockPin]:
    """The load scan, or one walk now for a model loaded before it existed."""
    model = getattr(patcher, "model", None)
    if model is None:
        return {}
    pins = getattr(model, _SCAN_ATTR, None)
    if pins is None:
        pins = scan(getattr(model, "diffusion_model", None))
        # A live module may be a staged LoRA's; its backup is the checkpoint's.
        for path, original in (getattr(patcher, "object_patches_backup", None) or {}).items():
            if path.rpartition(".")[2] == PIN_MODULE:
                pins.pop(path, None)
                pin = block_pin(original)
                if pin is not None:
                    pins[path] = pin
        _cache(model, pins)
    return pins


def effective_pins(patcher: Any) -> dict[str, BlockPin]:
    """The checkpoint's pins with the patcher's LoRA object patches laid over them."""
    pins = dict(_checkpoint_pins(patcher))
    for path, value in (getattr(patcher, "object_patches", None) or {}).items():
        if not isinstance(path, str) or path.rpartition(".")[2] != PIN_MODULE:
            continue
        pin = block_pin(value)
        if pin is None:
            pins.pop(path, None)
        else:
            pins[path] = pin
    return pins


def _approximate(counts: Mapping[str, int]) -> dict[str, int]:
    return {label: count for label, count in counts.items()
            if label.split(" ", 1)[0] not in EXACT_KERNELS}


@dataclass(frozen=True, slots=True)
class PinSummary:
    """What one rank renders: pinned attention modules per kernel, and which run.

    ``live`` counts the pins nothing drops (no sequence parallelism, no FSDP
    clear), which run wherever ComfyUI bound the kernel. ``running`` is the
    part this host bound; the record counts an unbound pin as dropped.
    """

    kernels: dict[str, int]
    running: dict[str, int]
    dropped: dict[str, int]
    live: dict[str, int] | None = None

    @property
    def ran(self) -> bool:
        return not self.dropped

    @property
    def approximate_running(self) -> dict[str, int]:
        return _approximate(self.running)

    @property
    def approximate_live(self) -> dict[str, int]:
        return _approximate(self.running if self.live is None else self.live)

    def public(self) -> dict[str, Any]:
        return {"attention_pins": dict(self.kernels), "attention_pins_ran": self.ran,
                "attention_pins_dropped": dict(self.dropped) or None}


def summarize(patchers: Iterable[Any], *, sequence_parallel: bool) -> PinSummary | None:
    """Summarize the pins of the patchers this rank renders, or None when none pin."""
    kernels: dict[str, int] = {}
    live: dict[str, int] = {}
    running: dict[str, int] = {}
    dropped: dict[str, int] = {}
    for patcher in patchers:
        if patcher is None:
            continue
        for pin in effective_pins(patcher).values():
            kernels[pin.label] = kernels.get(pin.label, 0) + 1
            reason = (DROPPED_SEQUENCE_PARALLEL if sequence_parallel
                      else DROPPED_FSDP if pin.cleared else None)
            if reason is None:
                live[pin.label] = live.get(pin.label, 0) + 1
                reason = None if pin.bound else DROPPED_UNBOUND
            if reason is None:
                running[pin.label] = running.get(pin.label, 0) + 1
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
    if not kernels:
        return None
    return PinSummary(kernels, running, dropped, live)


def render_summary(worker: Any, patchers: Iterable[Any]) -> PinSummary | None:
    """This render's summary under the worker's topology."""
    topology = getattr(worker, "topology", {}) or {}
    sp = int(topology.get("ulysses", 1) or 1) * int(topology.get("ring", 1) or 1)
    return summarize(patchers, sequence_parallel=sp > 1)


def _modules(count: int) -> str:
    return f"{count} attention module{'s' if count != 1 else ''}"


def _phrase(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{label} on {_modules(count)}" for label, count in sorted(counts.items()))


def _source(pinned_by: str | None) -> str:
    return "the checkpoint" if not pinned_by else f"the unconditional checkpoint {pinned_by}"


def card_detail(counts: Mapping[str, int], topology_label: str,
                pinned_by: str | None = None) -> str:
    """The card's line: the pin, its module count, and what a sequence-parallel topology does."""
    return (f"pinned by {_source(pinned_by)} or a LoRA in its stack: {_phrase(counts)}"
            f", which runs on {topology_label}; a sequence-parallel topology drops the pin instead")


def assert_waived(summary: PinSummary | None, topology: Mapping[str, Any] | None, *,
                  family: str | None = None, alone: bool = False,
                  pinned_by: str | None = None) -> None:
    """Refuse an approximate pinned kernel that would run, unless the waiver is granted.

    Class K, raised on the sample path inside the load guard, before the first
    collective. The verdict reads the pins, the LoRA stack and the topology,
    never whether this host could bind the kernel, so every rank of a fleet
    decides alike. That matters under cfg2+fsdp and dp2+fsdp, which run no
    readiness exchange: a rank that refused alone there would leave its peer
    waiting at the first collective until the group timeout. Only a rank that
    renders ``alone`` (world 1) reads its own binding, since no peer can run
    the kernel; an unbound pin then runs ComfyUI's default backend and asks
    nothing. ``pinned_by`` names the unconditional checkpoint under dual-model
    cfg2, whose rank refuses on its own pins while the card names the
    conditional checkpoint every grant of that render is keyed on. A grant
    fires the ledger's use row once per dispatch and stamps the output.
    """
    from . import accuracy_waiver
    from .refusal import RefusalClass, refusal

    if summary is None:
        return
    approximate = summary.approximate_running if alone else summary.approximate_live
    if not approximate or accuracy_waiver.waived(SOL_ATTN_PIN_GUARD):
        return
    label = accuracy_waiver.topology_label(topology)
    total = (f", {_modules(sum(approximate.values()))} in all" if len(approximate) > 1 else "")
    kernels = sorted({name.split(" ", 1)[0] for name in approximate})
    risks = "; ".join(_KERNEL_RISK.get(name, f"{name} is a kernel this project has not "
                                       "classified, so it counts as approximate")
                      for name in kernels)
    from .adapters.base import UnsupportedModelError

    raise UnsupportedModelError(refusal(
        RefusalClass.KNOWN_WRONG,
        "this model pins an approximate kernel on its attention modules: "
        f"{_phrase(approximate)}{total}, set by "
        f"{_source(pinned_by)} or a LoRA in its stack through ComfyUI's comfy_attention "
        f"config. Topology {label} runs no sequence parallelism, so ComfyUI runs each "
        f"pinned kernel as it does on one GPU. What each kernel does: {risks}. An "
        "approximate kernel is known-wrong math by construction, so this refuses by "
        "default. Two remedies: grant the Sol-Attn accuracy waiver for this checkpoint, "
        "topology and world, which stamps the output, or pick a sequence-parallel "
        "topology such as uly2, where dgx-monarch drops the pin and the Init widget's "
        "kernel runs (exact under TORCH_FLASH).",
        guard=SOL_ATTN_PIN_GUARD,
        waivable=True,
        panel_action=accuracy_waiver.panel_action(SOL_ATTN_PIN_GUARD),
        troubleshooting=117,
    ) + accuracy_waiver.card_tail(SOL_ATTN_PIN_GUARD, family_hint=family,
                                  detail=card_detail(approximate, label, pinned_by)))


def merge(rows: Iterable[Any]) -> dict[str, Any] | None:
    """Fold every rank's summary into the render record's fields, or None.

    Ranks of one checkpoint report the same pins. Under dual-model cfg2 each
    rank holds a different checkpoint, so the record keeps the larger count per
    kernel and reads ran only when every rank that pins ran its pins.
    """
    kernels: dict[str, int] = {}
    dropped: dict[str, int] = {}
    ran = True
    seen = False
    for row in rows:
        pins = row.get(RESULT_KEY) if isinstance(row, Mapping) else None
        if not isinstance(pins, Mapping):
            continue
        counts = pins.get("attention_pins")
        if not isinstance(counts, Mapping) or not counts:
            continue
        seen = True
        for label, count in counts.items():
            if isinstance(label, str) and isinstance(count, int):
                kernels[label] = max(kernels.get(label, 0), count)
        reasons = pins.get("attention_pins_dropped")
        for reason, count in (reasons if isinstance(reasons, Mapping) else {}).items():
            if isinstance(reason, str) and isinstance(count, int):
                dropped[reason] = max(dropped.get(reason, 0), count)
        ran = ran and pins.get("attention_pins_ran") is True
    if not seen:
        return None
    return {"attention_pins": kernels, "attention_pins_ran": ran,
            "attention_pins_dropped": dropped or None}


def describe(record: Mapping[str, Any] | None) -> str:
    """One line for a result row or a log: "pinned: comfy_kitchen_sol x 40 modules, ran"."""
    counts = record.get("attention_pins") if isinstance(record, Mapping) else None
    if not isinstance(record, Mapping) or not isinstance(counts, Mapping) or not counts:
        return ""
    pinned = ", ".join(f"{label} x {count} module{'s' if count != 1 else ''}"
                       for label, count in sorted(counts.items()))
    if record.get("attention_pins_ran") is True:
        return f"pinned: {pinned}, ran"
    reasons = record.get("attention_pins_dropped")
    reasons = reasons if isinstance(reasons, Mapping) else {}
    why = ", ".join(sorted(str(reason) for reason in reasons))
    dropped = sum(n for n in reasons.values() if isinstance(n, int))
    total = sum(n for n in counts.values() if isinstance(n, int))
    if not why or dropped >= total:
        return f"pinned: {pinned}, dropped" + (f" ({why})" if why else "")
    return f"pinned: {pinned}, ran on {total - dropped}, dropped on {dropped} ({why})"
