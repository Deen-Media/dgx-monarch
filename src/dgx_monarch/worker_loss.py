"""The cause of a lost Worker, in plain words, from Monarch's own texts.

When a Worker's process ends or the driver loses contact with its box, Monarch
says so in two places: the fault the driver's hook receives (a ``MeshFailure``
whose ``mesh_name`` names the actor mesh) and the error a pending or later call
raises (a ``SupervisionError`` that names the actor's point in the mesh). Both
carry the cause in Monarch's words, which read the same in every text captured
on CPU on 2026-11-17 against torchmonarch 0.7.0 and 0.8.0:

- ``Killed(sig=N)`` (``Killed(sig=N, core)`` with a core dump): a signal ended
  the process. The kernel's out-of-memory killer sends signal 9, but so does
  ``kill -9``, so the words point at the check and never state the cause.
- ``exited with non-zero code N``: the Worker exited by itself.
- ``supervision liveness failed``, ``status request to monitored actor ... timed
  out``, ``host unreachable``, ``timeout waiting for message from proc mesh
  agent``, ``no status received from process`` or a monitor that ``stopped
  before reporting a failure``: the driver lost contact with the box.

``read`` turns one text into a ``WorkerLoss`` and ``words`` into a sentence a
user reads, naming the box the way ``box_names`` does (no host name, address or
rank number). Only the pack's own Worker mesh counts: Monarch's system meshes on
the same process (``log_level``, ``logger``, ``setup``) fail with the same words
for the same death, so they are left out and one death is worded once.

A fault names its mesh. The pack records each Worker mesh's name and its
ranks' boxes at bring-up (``remember``), and the fault hook words a fault by
that record (``fault_words``): an idle Worker's fault can reach the hook about
30 s after the death, when the fleet may already have been replaced, so a mesh
the driver still holds is worded with its own boxes, one it no longer holds as
an earlier fleet's, never with the current fleet's boxes, and a mesh the pack
never recorded keeps Monarch's words. A call's error names no mesh; the caller
passes the boxes of the fleet it called (``explain_error``).

Leaf module: the standard library only.
"""
from __future__ import annotations

import re
import signal as _signal
import threading
from collections import OrderedDict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

WORKER_MESH = "dgxm_worker"
SIGNAL, EXIT_CODE, LOST_CONTACT = "signal", "exit_code", "lost_contact"

_SIGNAL = re.compile(r"Killed\(sig=(\d{1,3})(?:, core)?\)")
_EXIT = re.compile(r"exited with non-zero code (-?\d{1,4})")
_LOST = ("supervision liveness failed", "status request to monitored actor",
         "host unreachable", "timeout waiting for message from proc mesh agent",
         "no status received from process", "stopped before reporting a failure")
# A MeshFailure's own fields, and a mesh named in a proc-agent timeout.
_MESH_NAME = re.compile(r"mesh_name=([^\s,()]+)")
_QUOTED_MESH = re.compile(r'"(' + WORKER_MESH + r'-[^"\s]+)"')
_CRASHED = re.compile(r"crashed_ranks=\[([0-9, ]*)\]")
# The actor's point: dgxm_worker{'gpus': 1/2} or dgxm_worker{'hosts': 0/1, 'gpus': 1/2}.
_POINT = re.compile(WORKER_MESH + r"\{([^{}]*)\}")
_COORD = re.compile(r"'(\w+)':\s*(\d+)/(\d+)")
# Never the words "out of memory": mesh_safety.is_memory_exhaustion and the gate
# read them as a fact about the failure, and this hint states no fact.
_OOM_HINT = ("; the kernel's out-of-memory killer sends signal 9, and so does `kill -9`: "
             "look for an out-of-memory line in `dmesg` or `journalctl -k` on {where} "
             "before you blame memory")
_KEEP = 64
_LOCK = threading.Lock()
# Each Worker mesh the pack brought up, by its name, with its ranks in plain
# words, recorded at bring-up (mesh_factory) so the hook thread reads no mesh.
_FLEETS: OrderedDict[str, tuple[str, ...]] = OrderedDict()


@dataclass(frozen=True)
class WorkerLoss:
    """One lost Worker as Monarch reported it; machine fields, not user words."""

    kind: str
    ranks: tuple[int, ...] = ()
    mesh: str | None = None
    signal: int | None = None
    code: int | None = None


class WorkerLostError(RuntimeError):
    """A render's error that a lost Worker caused: the cause in plain words, then Monarch's own text.

    The node layer raises it chained from Monarch's error (nodes/refusal_lead.py),
    so ComfyUI names this type and Studio's error code reads
    ``dgx_monarch.worker_loss.WorkerLostError``; before 1.5.0 both read Monarch's
    ``monarch._rust_bindings.monarch_hyperactor.supervision.SupervisionError``.
    A RuntimeError, as that one is. It reports a fault and refuses nothing.
    """


def mesh_uid(name: object) -> str | None:
    """The uid in ``dgxm_worker-<uid>``, the name a fault and ``ActorMesh._name`` carry."""
    prefix = f"{WORKER_MESH}-"
    if not isinstance(name, str) or not name.startswith(prefix):
        return None
    uid = name[len(prefix):]
    return uid if re.fullmatch(r"[^<>\s]+", uid) else None


def _kind(text: str) -> tuple[str, int | None, int | None] | None:
    found = _SIGNAL.search(text)
    if found:
        return SIGNAL, int(found.group(1)), None
    found = _EXIT.search(text)
    if found:
        return EXIT_CODE, None, int(found.group(1))
    if any(marker in text for marker in _LOST):
        return LOST_CONTACT, None, None
    return None


def _ranks(text: str) -> tuple[int, ...]:
    crashed = _CRASHED.search(text)
    if crashed and crashed.group(1).strip():
        return tuple(dict.fromkeys(int(part) for part in crashed.group(1).split(",")))
    point = _POINT.search(text)
    if point is None:
        return ()
    coords = {label: (int(index), int(size)) for label, index, size in _COORD.findall(point.group(1))}
    if "gpus" not in coords:
        return ()
    gpu, gpus = coords["gpus"]
    host = coords.get("hosts", (0, 1))[0]
    return (host * gpus + gpu,)


def read(text: str, mesh_name: str | None = None) -> WorkerLoss | None:
    """The lost Worker one fault or error text describes, or None.

    None for anything that is not the pack's Worker mesh losing a process or
    contact: a system mesh's copy of the same death, a Worker's own exception,
    a deliberate stop, an undeliverable message.
    """
    if not isinstance(text, str) or WORKER_MESH not in text:
        return None
    named = mesh_name if isinstance(mesh_name, str) else None
    if named is None:
        found = _MESH_NAME.search(text)
        named = found.group(1) if found else None
    if named is not None and not named.startswith(WORKER_MESH):
        return None  # a system mesh (log_level, logger, setup): the same death, worded once
    if named is None:
        quoted = _QUOTED_MESH.search(text)
        named = quoted.group(1) if quoted else None
    kind = _kind(text)
    if kind is None:
        return None
    return WorkerLoss(kind[0], _ranks(text), mesh_uid(named), kind[1], kind[2])


def _signal_name(number: int) -> str:
    try:
        return f" ({_signal.Signals(number).name})"
    except ValueError:
        return ""


def words(loss: WorkerLoss, labels: Sequence[str] | None, *, earlier_fleet: bool = False) -> str:
    """One sentence for a lost Worker.

    ``labels`` are the fleet's ranks in plain words (``box_names.fleet_labels``);
    None, or a rank past them, words the Worker without its box. ``earlier_fleet``
    says the fault named a mesh the driver no longer holds. Callers word only a
    loss they did not attribute to a deliberate stop: a stop that outlives its
    grace ends the process with signal 9 too.
    """
    boxes = list(dict.fromkeys(labels[rank] for rank in loss.ranks
                               if labels is not None and 0 <= rank < len(labels)))
    several = len(boxes) > 1 and not earlier_fleet
    if earlier_fleet:
        subject = "a Worker of an earlier fleet"
    elif boxes:
        subject = ("the Workers " if several else "the Worker ") + _join([_at(box) for box in boxes])
    else:
        subject = "a Worker"
    if loss.kind == SIGNAL and loss.signal is not None:
        hosts = len({_box(box) for box in boxes}) if not earlier_fleet else 0
        where = "that box" if hosts == 1 else "those boxes" if hosts else "each box of the fleet"
        sentence = (f"{subject} {'were' if several else 'was'} stopped by signal "
                    f"{loss.signal}{_signal_name(loss.signal)}"
                    + (_OOM_HINT.format(where=where) if loss.signal == 9 else ""))
    elif loss.kind == EXIT_CODE:
        sentence = f"{subject} exited with code {loss.code}"
    else:
        # Contact is lost with a box, not with one GPU on it.
        where = (_join(list(dict.fromkeys(_box(box) for box in boxes)))
                 if boxes and not earlier_fleet else subject)
        sentence = f"the driver lost contact with {where}"
    return sentence[:1].upper() + sentence[1:] + "."


def _at(label: str) -> str:
    """'on the sibling', or 'for GPU 1 on the head' for a box that holds several ranks."""
    return ("for " if label.startswith("GPU ") else "on ") + label


def _box(label: str) -> str:
    return label.split(" on ", 1)[1] if label.startswith("GPU ") and " on " in label else label


def _join(items: Sequence[str]) -> str:
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


def explain_fault(text: str, mesh_name: str | None,
                  fleets: Mapping[str, Sequence[str]]) -> str | None:
    """Plain words for a fault the hook received, keyed on the mesh it names.

    ``fleets`` maps the uid of each Worker mesh the driver holds to its rank
    labels. A fault for any other mesh is an earlier fleet's.
    """
    loss = read(text, mesh_name)
    if loss is None:
        return None
    labels = fleets.get(loss.mesh) if loss.mesh is not None else None
    return words(loss, labels, earlier_fleet=loss.mesh is not None and labels is None)


def explain_error(text: str, labels: Sequence[str] | None) -> str | None:
    """Plain words for a call's error, on the fleet the caller called."""
    loss = read(text)
    return None if loss is None else words(loss, labels)


def remember(name: object, labels: Sequence[str] | None) -> None:
    """Record a Worker mesh's name and its ranks' words at bring-up; never raises."""
    if not isinstance(name, str) or mesh_uid(name) is None or labels is None:
        return
    with _LOCK:
        _FLEETS[name] = tuple(str(label) for label in labels)
        _FLEETS.move_to_end(name)
        while len(_FLEETS) > _KEEP:
            _FLEETS.popitem(last=False)


def fault_words(text: str, mesh_name: str | None, held: Collection[str]) -> str | None:
    """Plain words for a fault, keyed on the mesh it names, or None to keep Monarch's.

    ``held`` names the Worker meshes behind the driver's current handles. A
    recorded mesh the driver still holds is worded with its own boxes, read at
    its bring-up; one it no longer holds is an earlier fleet's. A mesh the pack
    never recorded gets no words: nothing says whose it was.
    """
    loss = read(text, mesh_name)
    if loss is None or loss.mesh is None:
        return None
    name = f"{WORKER_MESH}-{loss.mesh}"
    with _LOCK:
        labels = _FLEETS.get(name)
    if labels is None:
        return None
    return words(loss, labels) if name in held else words(loss, None, earlier_fleet=True)


def hook_words(failure: object, held: Callable[[], Collection[str]] | None) -> str | None:
    """``fault_words`` for the fault hook, from the fault itself; never raises."""
    try:
        name = getattr(failure, "mesh_name", None)
        return fault_words(str(failure), name if isinstance(name, str) else None,
                           held() if held is not None else ())
    except Exception:
        return None


def reset() -> None:
    """Forget every recorded mesh (tests)."""
    with _LOCK:
        _FLEETS.clear()
