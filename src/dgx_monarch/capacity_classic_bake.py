"""Price what a LoRA stack's bake adds beside the weights on unified memory.

ComfyUI's classic bake (``lora_low_rss`` off) moves each patched weight from
the host onto the device among that key's temporaries, and the process's GPU
figure ends above the weights by an amount that follows how many weights the
stack patches, not the stack's size. Measured on one GPU of a DGX Spark under
its clock cap (max 2100 MHz), as the rise over the same model's load with no
stack (docs/DESIGN.md section 5.9 holds the table and the dates):

- LTX 2.3 scaled FP8, 1660 patched weights (18.07 GiB of them): 12.34 GiB;
- the same model, 480 patched weights (12.0 GiB): 5.03 GiB;
- MiniMax H3 INT8 ConvRot, 208 patched weights: 1.21 GiB;
- Qwen Image plain FP8, 720 patched weights: 0.84 GiB.

The line through the two LTX points, 2 GiB plus 7.5 MiB a patched weight,
covers both (5.52 and 14.16 GiB). A patched bias counts as a weight, as it
does in the count the Worker logs. The stock price (``capacity_fit``) had no
term for any of it, and a 7.1 GiB stack took a load 8.7 GiB past its price.

The rises say what to charge for; what a load TOOK says how much. Three
hardware blocks (2026-11-12, the same Spark and cap) read the box's lowest
memory after the Worker's price line, for a whole load with its stack and its
render, against what the stock price promised (docs/DESIGN.md section 5.9,
the second table). ``classic_charge`` is sized by format from those records,
and ``tests/test_classic_bake_records.py`` holds each record against it:

- **The stack's files, and which reading the price is held against.** The
  Worker reads every LoRA file of a stack into host memory, one read for each
  entry, and holds them through the bake. A price read BEFORE those reads (a
  load into an empty slot, and every quote) charges the files, each as many
  times as the stack lists it. A price read AFTER them charges none: when the
  store tries its hot swap on a held base it reads the stack first, and if
  the swap then declines, the fresh load's walls read a box those bytes have
  already left (``stack_on_host``). LTX's own stack shows both: from an empty
  Worker the load took 62.5 GiB; held against the reading after the file was
  read, 55.1 to 55.6. The same 7.08 GiB file listed twice took 69.5 from an
  empty Worker, 7 more. The first version charged the files on both readings
  and refused held-base states that had rendered; the second charged them on
  neither and came up 2.4 and 4.7 GiB short on the empty Worker.
- **Scaled FP8** (an FP8 weight with its scale or ComfyUI's quant record):
  the line. It leaves 3.7 to 5.0 GiB above take plus floor on seven recorded
  loads over both stack sizes and both readings.
- **INT8 ConvRot**: the fixed 2 GiB, and 4 MiB a patched key
  (``LIGHT_SHARE_BYTES``) only where ComfyUI's own whole-load condition does
  not already keep the floor. Three records: with 1261 patched keys a load
  ComfyUI placed partly, and the partial-load route then baked whole, took
  33.6 GiB, 1.0 over the stock price; the same stack loaded WHOLE took 39.4,
  6.8 over (ComfyUI's backup of the 15.5 GiB the stack patches, the same
  bytes on the device, 6.7 of sampler memory); with 208 patched keys a whole
  load took 1.3 to 2.2 over. The fixed part covers the route's take. The
  whole load's is covered another way. ComfyUI places a model whole only
  when its free reading is at least the model, the Worker's reserve R and
  its own estimate for the latent (``load_models_gpu``: model + R + max(0.8
  GiB, A)); under that it places it partly, and on one GPU the route carries
  it. So a whole load starts with R still free, and what it then takes
  beyond ComfyUI's accounting comes out of R: 2.3 GiB on the one stack where
  both sides were read (whole from a reading of 45.1 at R = 8, 39.4 taken:
  the lowest whole-load low is 5.7). Where R is at least the floor and that
  2.3 (``reserve_keeps_floor``), every whole load leaves the floor and the
  share is not charged. A lower reserve (the Init widget ``reserve_vram_gb``
  or ``[worker_args]``; the default is 8) lets ComfyUI place the model whole
  at readings where it does not, so the share is charged and the price
  itself covers the whole load: 4 MiB a key is a fit through that load, with
  0.1 GiB to spare. The 2.3 is read on one stack, which patches its whole
  model (the most a backup can hold) with 1254 keys at the share; a stack
  with more such keys is past the record and always pays the share. A price
  given no reserve pays it too.
- **Plain FP8** (an FP8 weight with no marker): the fixed 2 GiB, no share.
  Two stacks on one model (720 and 240 patched keys) took 5.0 to 2.7 GiB
  LESS than that price allows.
- **Small keys beside such weights.** In a stack whose patched bytes are
  mostly INT8 ConvRot or plain FP8 weights, a patched key under
  ``SMALL_KEY_BYTES`` (a bias, a norm weight) takes that format's share and
  not the line's; a larger weight of another format keeps its own. Every
  other stack is counted as the line was fitted, its biases included. A
  stack is read by the format that holds most of its patched bytes, so one
  stray key of another format does not change how it is charged.
- **Full precision** (BF16, FP16, FP32): the line. A Klein BF16
  stack of 121 patched weights rose 1.00 GiB against a line of 2.89, and the
  BF16 Wan-Animate 2 template's stack of 1261 rose at most 7.4 against 11.24;
  that template's load took 13.3 GiB over the stock price, 1.3 more than the
  line and the files charge, so nothing is trimmed there.
- **The bytes guard** (0.75 of the patched bytes of a format, for a weight
  larger than any measured) has no record behind it and is off:
  ``GUARDED_BY_BYTES`` is empty. Both LTX stacks' largest patched weight is
  64.0 MiB; a file with larger ones is past the records, and the constant is
  the one place to change when a cell reads one.
- A stack whose LoRA file cannot be read or mapped is charged for every weight
  of its checkpoint. A checkpoint whose own header cannot be read gives no
  weights to count, so its stack is charged the line for
  ``UNREADABLE_STACK_WEIGHTS``, the largest count measured, and its files.
  That is a stand-in and not a bound: a larger model may patch more.

The in-place bake (what ``lora_low_rss`` does on a loaded model, and what the
store falls back to when the classic price does not fit and the identity gate
has passed the combination) leaves almost none of the classic bake's memory:
it rose 8.28 GiB for a 7.08 GiB stack staged on the device and 2.25 for a
0.305 GiB one, the stack's own bytes plus about 2 GiB of working memory. The
fallback runs with ``lora_low_rss`` off, where the stack stays on the host, so
``in_place_bake_bytes`` charges one copy of the stack and that working memory.

No torch and no ComfyUI import: the driver's quote and the Worker's walls
call the same functions with the same arguments, so one load has one price.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from .capacity_floor import ABSOLUTE_HOST_FLOOR_BYTES
from .capacity_lora_bake import _base_index, _find_base_key, _lora_targets, stack_file_bytes
from .safetensors_header import TensorDescriptor, read_safetensors_header

GIB = 1 << 30
MIB = 1 << 20

# The line through the two scaled FP8 rises: 5.52 GiB for the 5.03 measured, 14.16 for the 12.34.
CLASSIC_BAKE_FIXED_BYTES = 2 * GIB
CLASSIC_BAKE_BYTES_PER_WEIGHT = int(7.5 * MIB)
# The larger share of the patched bytes measured (0.68), rounded up.
CLASSIC_BAKE_GUARD_RATIO = 0.75
FP8_QUANTIZED = "fp8_quantized"
INT8_QUANTIZED = "int8_quantized"
PLAIN_FP8 = "plain_fp8"
FULL_PRECISION = "full_precision"
OTHER = "other"
# The classes with a per-weight share of their own in place of the line's, in bytes a
# patched key. INT8 ConvRot, 4 MiB, the share that makes the price cover a WHOLE classic
# load: with 1261 patched keys one took 6.8 GiB over the stock price (39.4 against a promise
# of 32.6; block C, 2026-11-12, priced after the stack's file was read). 2 GiB, 4 MiB for
# each of 1254 keys and 7 large weights on the line come to 6.95, 0.1 over. It is charged
# only where ``reserve_keeps_floor`` says ComfyUI's own condition does not keep the floor.
# With 208 keys the term is then 2.84 against a take of 1.3 to 2.2. Plain FP8, nothing: two
# stacks (720 and 240 keys) took less than the fixed part allows.
LIGHT_SHARE_BYTES: dict[str, int] = {INT8_QUANTIZED: 4 * MIB, PLAIN_FP8: 0}
# What a whole classic load takes beyond what ComfyUI's whole-load condition sets aside for
# it (comfy/model_management.py ``load_models_gpu``: whole only when the free reading is at
# least model + R + max(0.8 GiB, A), R the Worker's reserve, A its estimate for the latent).
# A whole load therefore starts with R free and ends with R less this. Read on the shipped
# INT8 Wan-Animate 2 stack at R = 8 GiB: whole from a reading of 45.1, 39.4 taken, so 2.3
# (about 1.7 the load holds beside the model, its backup and the sampler's memory, and
# about 0.5 by which the sampler's 6.7 passed ComfyUI's estimate at that clip). The backup
# is the one part that follows the stack and it cannot pass the model, which the condition
# already counts; that stack patches its whole model. No further gap is allowed for: at the
# default reserve 0.7 GiB is left for a clip whose sampler passes the estimate by more.
WHOLE_LOAD_BEYOND_COMFYUI_BYTES = int(2.3 * GIB)
# The keys at a share on that stack (480 INT8 weights and 774 small keys). A stack with
# more is past the one record the figure above is read from.
RECORDED_WHOLE_LOAD_SHARE_KEYS = 1254
# In a stack whose patched bytes are mostly weights of those classes, a patched key smaller
# than this takes that class's share and not the line's. The records are of whole stacks, and
# most of a stack's keys there are the biases and norm weights of the same blocks (774 of
# the 1261 on Wan-Animate 2, none over 640 KiB; the file's next tensor up is 1.4 MiB).
SMALL_KEY_BYTES = MIB
# The formats whose patched bytes are guarded by ``CLASSIC_BAKE_GUARD_RATIO`` when a weight
# is larger than the row below. Empty: no record needs the guard (both LTX stacks' largest
# patched weight is 64.0 MiB, and BF16's two cells sit under the line). Naming a format here
# is the whole change if a cell on a file with larger weights calls for it.
GUARDED_BY_BYTES: tuple[str, ...] = ()
# The largest patched weight under each format's measured points. A guarded format whose
# largest patched weight is no larger is charged by the line alone; a guarded format with no
# row here is always guarded. The FP8 row is read from the LTX 2.3 scaled FP8 file's header
# (2026-11-12): the largest weight either measured stack patches is 64.0 MiB.
MEASURED_LARGEST_WEIGHT_BYTES = {FP8_QUANTIZED: 64 * MIB}
# What a stack is counted as when its checkpoint's header cannot be read: the largest count
# of patched weights measured (LTX 2.3's own stack). A stand-in, not a bound: a larger model
# whose header cannot be read may patch more.
UNREADABLE_STACK_WEIGHTS = 1660
# The bake's working memory beside the stack's own bytes (1.9 and 1.2 GiB measured).
IN_PLACE_BAKE_FIXED_BYTES = 2 * GIB

_FP8_DTYPES = frozenset({"F8_E4M3", "F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FNUZ"})
_INT8_DTYPES = frozenset({"I8"})
_FULL_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_QUANT_MARKERS = (".weight_scale", ".scale_weight", ".comfy_quant")
_WEIGHT_SUFFIXES = (".weight", ".bias")

# The sample request's field for the driver's permission (nodes/in_place_bake_permission.py).
# Only the value True grants it; absent, an older driver, a requeue or a recycle all read as no.
IN_PLACE_BAKE_PROVEN = "_dgxm_in_place_bake_proven"

# Why an in-place bake was not priced. Each is a whole clause; a refusal prints it after the
# classic price's terms (``StockFit.in_place_note``, ``stock_words.bake_sentences``).
GATE_HAS_NOT_PASSED = ("baking it in place was not tried: no identity-gate PASS for this combination's in-place "
                "bake came with this load; render it once with lora_low_rss on so the gate can prove it")


def weight_class(key: str, tensors: Mapping[str, TensorDescriptor]) -> str:
    """How the classic bake's price reads one checkpoint weight, from the header alone.

    A quantized weight is one with a quantization marker beside it (its
    scale, or ComfyUI's quant record): scaled FP8 and INT8 ConvRot. An FP8
    weight with no marker is plain FP8. An INT8 weight with no marker, and
    any other dtype, has no record and reads as ``OTHER``, charged by the line.
    """
    dtype = tensors[key].dtype
    if dtype in _FULL_DTYPES:
        return FULL_PRECISION
    stem = key.rsplit(".", 1)[0]
    quantized = any(stem + marker in tensors for marker in _QUANT_MARKERS)
    if dtype in _FP8_DTYPES:
        return FP8_QUANTIZED if quantized else PLAIN_FP8
    if dtype in _INT8_DTYPES and quantized:
        return INT8_QUANTIZED
    return OTHER


def _light_lead(by_class: Mapping[str, list[int]]) -> str | None:
    """The ``LIGHT_SHARE_BYTES`` class a stack is read as, or ``None``.

    A stack is read by the format that holds most of what it patches: where
    more than half of its patched bytes are weights of the light classes, the
    one of them that holds the most. One stray key of another format cannot
    change that. A stack with no bytes to read (an unreadable header) is of
    no format.
    """
    held = {kind: sum(sizes) for kind, sizes in by_class.items()}
    light = {kind: size for kind, size in held.items() if kind in LIGHT_SHARE_BYTES}
    if 2 * sum(light.values()) <= sum(held.values()):
        return None
    return max(sorted(light), key=lambda kind: light[kind])


def reserve_keeps_floor(comfy_reserve_bytes: int | None, share_keys: int) -> bool:
    """Whether ComfyUI's whole-load condition already keeps the floor under a whole classic load.

    ComfyUI places a model whole only with its reserve still free beyond the
    model and the latent's estimate, and the whole load then takes
    ``WHOLE_LOAD_BEYOND_COMFYUI_BYTES`` of that reserve. So the floor is kept
    when the reserve is at least the floor and that figure. ``share_keys`` is
    how many of the stack's keys sit at a ``LIGHT_SHARE_BYTES`` share: above
    the recorded stack's count nothing is on record, and the answer is no.
    ``None`` is a price that was not told the reserve: no.
    """
    if comfy_reserve_bytes is None or share_keys > RECORDED_WHOLE_LOAD_SHARE_KEYS:
        return False
    return int(comfy_reserve_bytes) >= ABSOLUTE_HOST_FLOOR_BYTES + WHOLE_LOAD_BEYOND_COMFYUI_BYTES


def term_from_weights(weights: Iterable[tuple[str, int]], *, guard: bool = True,
                      comfy_reserve_bytes: int | None = None) -> int:
    """The classic bake's charge for patched weights given as (class, stored bytes).

    The fixed part, and a per-weight share for each key: its own class's
    where ``LIGHT_SHARE_BYTES`` names one, else the line's. In a stack read
    as a light class (``_light_lead``) a key of another class under
    ``SMALL_KEY_BYTES`` takes that class's share. Every other stack is
    counted whole at the line's share, its biases included, as the line was
    fitted. ``comfy_reserve_bytes`` is the Worker's ComfyUI reserve: where it keeps
    the floor under a whole load (``reserve_keeps_floor``) the light classes'
    shares are not charged.

    ``guard=False`` charges the line whatever ``GUARDED_BY_BYTES`` names. The
    partial-load route asks for that: its BF16 cell ran on hardware under the
    rule without this term, and the bytes guard has no measurement behind it.
    """
    count = 0
    by_class: dict[str, list[int]] = {}
    for kind, nbytes in weights:
        count += 1
        by_class.setdefault(kind, []).append(max(int(nbytes), 0))
    if not count:
        return 0
    lead = _light_lead(by_class)
    shares = light = light_keys = 0
    for kind, sizes in by_class.items():
        share = LIGHT_SHARE_BYTES.get(kind)
        if share is not None:
            light, light_keys = light + share * len(sizes), light_keys + (len(sizes) if share else 0)
            continue
        small = sum(1 for size in sizes if size < SMALL_KEY_BYTES) if lead is not None else 0
        share = LIGHT_SHARE_BYTES[lead] if lead is not None else 0
        light, light_keys = light + share * small, light_keys + (small if share else 0)
        shares += CLASSIC_BAKE_BYTES_PER_WEIGHT * (len(sizes) - small)
    if not reserve_keeps_floor(comfy_reserve_bytes, light_keys):
        shares += light
    line = CLASSIC_BAKE_FIXED_BYTES + shares
    if not guard:
        return line
    guarded = 0
    for kind in GUARDED_BY_BYTES:
        sizes = by_class.get(kind)
        if not sizes:
            continue
        measured = MEASURED_LARGEST_WEIGHT_BYTES.get(kind)
        if measured is None or max(sizes) > measured:
            guarded += sum(sizes)
    return max(line, int(CLASSIC_BAKE_GUARD_RATIO * guarded))


def _every_weight(tensors: Mapping[str, TensorDescriptor]) -> set[str]:
    return {key for key in tensors if key.endswith(_WEIGHT_SUFFIXES)}


def patched_weights(tensors: Mapping[str, TensorDescriptor], lora_stack: list | None,
                    resolve_lora_path: Callable[[str], str]) -> set[str]:
    """The checkpoint keys this stack patches; every weight when a file or a target cannot be read."""
    index = _base_index(tensors)
    keys: set[str] = set()
    for entry in lora_stack or ():
        name = entry.get("name") if isinstance(entry, dict) else None
        if not name:
            continue
        try:
            targets = _lora_targets(read_safetensors_header(resolve_lora_path(str(name))).tensors)
        except Exception:
            return _every_weight(tensors)
        if targets is None:
            return _every_weight(tensors)
        for stem, suffix in targets:
            found = _find_base_key(stem, suffix, index)
            if found is None:
                return _every_weight(tensors)
            keys.add(found)
    return keys


def _patched(checkpoint_path: str, lora_stack: list,
             resolve_lora_path: Callable[[str], str]) -> list[tuple[str, int]]:
    """(class, stored bytes) for each checkpoint key the stack patches.

    A checkpoint whose header cannot be read gives no weights to count or to
    class, so the stack stands as ``UNREADABLE_STACK_WEIGHTS`` unclassed ones.
    """
    try:
        tensors = read_safetensors_header(checkpoint_path).tensors
    except Exception:
        return [(OTHER, 0)] * UNREADABLE_STACK_WEIGHTS
    keys = patched_weights(tensors, lora_stack, resolve_lora_path)
    return [(weight_class(key, tensors), tensors[key].nbytes) for key in keys]


def classic_bake_bytes(checkpoint_path: str, lora_stack: list | None,
                       resolve_lora_path: Callable[[str], str]) -> int:
    """The term alone: what ``term_from_weights`` charges for the weights ``lora_stack`` patches."""
    if not lora_stack:
        return 0
    return term_from_weights(_patched(checkpoint_path, lora_stack, resolve_lora_path))


def stack_bytes(lora_stack: list | None, resolve_lora_path: Callable[[str], str]) -> int:
    """The stack's LoRA files on disk; a file that cannot be sized counts as nothing."""
    return int(stack_file_bytes((lora_stack, resolve_lora_path)) or 0) if lora_stack else 0


def charge_from_weights(weights: Iterable[tuple[str, int]], stack_files: int, *,
                        stack_on_host: bool = False, comfy_reserve_bytes: int | None = None) -> int:
    """What a classic price adds for a stack: the term, and the stack's files unless they are already read.

    The Worker reads every LoRA file of the stack into host memory, one read
    for each entry, and holds them through the bake
    (``ModelStore._build_active``). ``stack_files`` is their sum as
    ``stack_bytes`` gives it: a file listed twice counts twice. A price held
    against a reading taken BEFORE those reads charges them, for every class.
    ``stack_on_host`` says the reading was taken AFTER them: the store's hot
    swap read the stack and declined, and the fresh load that follows prices
    against a box those bytes have already left. Then nothing is charged for
    the files. ``tests/test_classic_bake_records.py`` holds both readings
    against what each recorded load took. ``comfy_reserve_bytes`` is the Worker's
    ComfyUI reserve (``term_from_weights``).
    """
    return (term_from_weights(weights, comfy_reserve_bytes=comfy_reserve_bytes)
            + (0 if stack_on_host else max(int(stack_files), 0)))


def classic_charge(checkpoint_path: str, lora_stack: list | None, resolve_lora_path: Callable[[str], str],
                   *, stack_on_host: bool = False, comfy_reserve_bytes: int | None = None) -> int:
    """What the classic bake of ``lora_stack`` adds to this checkpoint's stock price (``charge_from_weights``).

    The one figure the quote, the Worker's two walls and the in-place choice
    read for a classic bake. Only the site that knows the stack was read
    before its reading passes ``stack_on_host``: the Worker, after a declined
    hot swap. A quote never does. Every site passes the Worker's reserve.
    """
    if not lora_stack:
        return 0
    return charge_from_weights(_patched(checkpoint_path, lora_stack, resolve_lora_path),
                               stack_bytes(lora_stack, resolve_lora_path), stack_on_host=stack_on_host,
                               comfy_reserve_bytes=comfy_reserve_bytes)


def in_place_bake_bytes(stack_files: int) -> int:
    """What baking a stack in place on a loaded model takes with ``lora_low_rss`` off.

    One copy of the stack, read into host memory for the bake, and the bake's
    working memory. Nothing is staged on the device in that mode.
    """
    return max(int(stack_files), 0) + IN_PLACE_BAKE_FIXED_BYTES


def _gib(value: int) -> float:
    return round(value / GIB, 1)


def priced(fit: Any, checkpoint_path: str, lora_stack: list | None, resolve_lora_path: Callable[[str], str],
           *, held_bytes: int = 0, in_place: str | bool = "", bare_load: bool = False, charged: int = 0,
           stack_on_host: bool = False, comfy_reserve_bytes: int | None = None) -> Any:
    """``fit`` (a ``capacity_fit.StockFit``) with the classic bake charged, or re-priced as an in-place bake.

    Every site that prices one stack change calls this with the same
    arguments, so the quote, the store and the Worker's walls agree.

    ``held_bytes`` is what dropping the slot's resident returns: the classic
    change is compared with the reading plus that, and so is an in-place bake
    that must first reload the base bare. ``in_place`` is ``True`` when the
    store may bake this stack in place, else the clause that says why not (""
    when the question does not arise). ``bare_load`` says the slot does not
    hold this base bare: it is dropped, the base loads bare at the stock
    price, and the bake then needs its room beside the loaded copy.

    The in-place price is taken only when the classic one, so credited, does
    not fit and the in-place one does. A classic price that misses carries
    what the in-place bake would have needed and why it was not taken.

    ``charged`` is what this request's earlier slots already take out of the
    reading (the quote prices a pair's second model after its first). The
    bake is chosen against what is left, as the store chooses it when that
    slot's load runs beside the first model. The fit returned keeps the
    reading as read: ``capacity_quote._charge`` takes the charge off once.

    ``stack_on_host`` says the reading was taken after the Worker had read
    the stack's files (``charge_from_weights``): the classic price then
    charges none of them. The in-place price charges them either way.

    ``comfy_reserve_bytes`` is the Worker's ComfyUI reserve, the R of ComfyUI's
    whole-load condition (``reserve_keeps_floor``). The quote, both walls
    and ``take`` run in the Worker and pass the one it runs with
    (``in_place_fallback.worker_reserve``). ``None`` charges as if no
    reserve kept the floor.
    """
    if not lora_stack or not getattr(fit, "applies", False) or getattr(fit, "avail_bytes", None) is None:
        return fit
    read, spent = int(fit.avail_bytes), max(int(charged), 0)
    fit = replace(fit, avail_bytes=read - spent)
    return replace(_chosen(fit, checkpoint_path, lora_stack, resolve_lora_path, max(int(held_bytes), 0),
                           in_place, bare_load, "was left beside this request's other model" if spent else "was read",
                           bool(stack_on_host), comfy_reserve_bytes),
                   avail_bytes=read)


def _chosen(fit: Any, checkpoint_path: str, lora_stack: list, resolve_lora_path: Callable[[str], str],
            held: int, in_place: str | bool, bare_load: bool, reading_words: str,
            stack_on_host: bool = False, comfy_reserve_bytes: int | None = None) -> Any:
    """``priced``'s choice between the two bakes, against ``fit.avail_bytes`` as given."""
    files = stack_bytes(lora_stack, resolve_lora_path)
    classic = replace(fit, bake_bytes=classic_charge(
        checkpoint_path, lora_stack, resolve_lora_path, stack_on_host=stack_on_host,
        comfy_reserve_bytes=comfy_reserve_bytes))
    classic = replace(classic, fits=classic.required_bytes <= classic.avail_bytes)
    if classic.required_bytes - held <= classic.avail_bytes or not in_place:
        return classic
    if in_place is not True:
        return replace(classic, in_place_note="; " + str(in_place))
    need = in_place_bake_bytes(files)
    if bare_load:
        need = max(int(fit.required_bytes) - ABSOLUTE_HOST_FLOOR_BYTES, int(fit.resident_bytes) + need)
        need = max(need - held, IN_PLACE_BAKE_FIXED_BYTES)  # the drop returns the resident first
    taken = replace(classic, in_place_bytes=need)
    if taken.required_bytes <= taken.avail_bytes:
        return replace(taken, fits=True)
    beside = "" if bare_load else f" beside the {_gib(held)} GiB loaded copy"
    return replace(classic, in_place_note=f"; baking it in place would need {taken.required_gib} GiB{beside}, "
                   f"and {taken.avail_gib} GiB {reading_words}")


def probe(lora_stack: list | None, resolve_lora_path: Callable[[str], str], *, held_bytes: int = 0,
          in_place: str | bool = "", bare_load: bool = False, charged: int = 0, stack_on_host: bool = False,
          comfy_reserve_bytes: int | None = None,
          stock_load_fit: Callable[[str, dict], Any] | None = None) -> Callable[[str, dict], Any]:
    """A stock probe (``capacity_quote.price``'s ``fit_probe``) that charges this load's classic bake."""
    def fit_probe(path: str, model_options: dict) -> Any:
        from . import capacity_fit

        return priced((stock_load_fit or capacity_fit.stock_load_fit)(path, model_options), path, lora_stack,
                      resolve_lora_path, held_bytes=held_bytes, in_place=in_place, bare_load=bare_load,
                      charged=charged, stack_on_host=stack_on_host, comfy_reserve_bytes=comfy_reserve_bytes)

    return fit_probe


__all__ = ["CLASSIC_BAKE_BYTES_PER_WEIGHT", "CLASSIC_BAKE_FIXED_BYTES", "CLASSIC_BAKE_GUARD_RATIO",
           "FP8_QUANTIZED", "FULL_PRECISION", "GATE_HAS_NOT_PASSED", "GUARDED_BY_BYTES",
           "INT8_QUANTIZED", "IN_PLACE_BAKE_FIXED_BYTES", "IN_PLACE_BAKE_PROVEN", "LIGHT_SHARE_BYTES",
           "MEASURED_LARGEST_WEIGHT_BYTES", "OTHER", "PLAIN_FP8", "RECORDED_WHOLE_LOAD_SHARE_KEYS", "SMALL_KEY_BYTES",
           "UNREADABLE_STACK_WEIGHTS", "WHOLE_LOAD_BEYOND_COMFYUI_BYTES", "charge_from_weights",
           "classic_bake_bytes", "classic_charge", "in_place_bake_bytes", "patched_weights", "priced", "probe",
           "reserve_keeps_floor", "stack_bytes", "term_from_weights", "weight_class"]
