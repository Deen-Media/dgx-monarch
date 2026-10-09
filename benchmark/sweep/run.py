"""Run a sweep session against a live ComfyUI driver.

Local and cluster sessions use separate driver processes. Each cell queues a
cold run and, after it or its waived retry renders, a warm run with a changed
seed and the requested references. One JSON record per cell supports resume
after interruption.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import shlex
import subprocess
import threading
import time
import urllib.error
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple

from .compare import audio_compare_available, compare_outputs, frames_for
from .convert import set_probe_steps
from .driver import (
    GraphRejected,
    SessionWedged,
    cached_nodes,
    classify_history,
    clear_graph,
    drain,
    error_text,
    free_cache,
    get,
    grant_pending_consent,
    queue_and_wait,
    recycle,
    waiver_run_ids,
)
from .matrix import FULL_RENDER_BAR, INIT_CLASS, LORA_CLASS, REPO, WAIVED_BAR, load_config

SEED_KEYS, SAVE_CLASSES = ("seed", "noise_seed"), ("SaveImage", "SaveAudioAdvanced")
# mesh_safety.py raises the activation wall (docs/TROUBLESHOOTING.md #47)
# untagged, so it reads refuse:untyped. The driver-side wall (driver_footprint.py,
# and nodes/loader_preflight.py at the loader node; docs/TROUBLESHOOTING.md #52)
# is class C, or class K where a measured identity-gate FAIL quarantines slab
# residency, so verdict_for matches its text as well as refuse:C.
CAPACITY_WALLS = ("activation footprint preflight refuses ",
                  "driver-side footprint preflight")
# A framed driver message ends with the refusal card, after the remote type
# name and any traceback, so the record keeps the tail.
MESSAGE_KEEP = 8000
NRMS_FLOOR = 0.10  # step-nrms bar; the null A/B calibrates the real number
JOURNAL_KEEP = ("FSDP capacity mode", "FSDP streaming build", "LoRA on FSDP shards",
                "ambient verify", "identity gate", "refus", "CAPACITY", "Traceback",
                "sharded", "slab", "comfy_managed", "sol-attn kernel", "backend=",
                "USP attention kernel", "unavailable (")


def patch_graph(graph: dict, cell: dict, preset: str, auto_gate: str, seed_shift: int,
                prefix: str, steps: int | None = None) -> dict:
    """Apply the sweep variables and nothing else."""
    out = copy.deepcopy(graph)
    for node_id, name in cell["unets"].items():
        out[node_id]["inputs"]["unet_name"] = name
    keep = {lora["node"]: lora for lora in cell["loras"]}
    for node_id in [nid for nid, node in out.items() if node["class_type"] == LORA_CLASS]:
        if node_id in keep:
            out[node_id]["inputs"]["lora_name"] = keep[node_id]["name"]
            out[node_id]["inputs"]["strength_model"] = keep[node_id]["strength"]
            continue
        source = out[node_id]["inputs"].get("model")
        del out[node_id]
        for node in out.values():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and value and value[0] == node_id:
                    node["inputs"][name] = source
    for node_id, node in out.items():
        inputs = node["inputs"]
        if node["class_type"] == INIT_CLASS:
            inputs.update({"topology": preset, "mode": cell["mode"], "auto_gate": auto_gate,
                           "attention": cell["attention"], **cell["levers"]})
            if cell["gpus_per_host"]:
                inputs["gpus_per_host"] = cell["gpus_per_host"]
        if node["class_type"] in SAVE_CLASSES:
            inputs["filename_prefix"] = prefix
        # Only the node the matrix read the batch from, and only on a cell the
        # batch toggle made: a second batch_size widget is another modality's.
        if cell["batch_toggled"] and node_id == cell["batch_node"]:
            inputs["batch_size"] = cell["batch"]
        for key in SEED_KEYS:
            if isinstance(inputs.get(key), int):
                inputs[key] += seed_shift
    # The step count is not always a widget called steps on a sampler: a
    # scheduler node of the family's own can hold it, and the LTX pack holds a
    # whole sigma list. convert.STEP_WIDGETS is the one table that says where.
    if steps is not None:
        set_probe_steps(out, steps)
    return out


def _on(host: str, command: list[str], timeout: float) -> tuple[str, str]:
    """Stdout and an error string.

    ssh joins its argv with spaces and the login shell on the far side re-parses
    the result, so a remote command travels as one quoted string. Nothing reads
    the operator's terminal either: an inherited stdin turns a remote read into
    a wait for the timeout.
    """
    if host:
        command = ["ssh", "-n", "-o", "ConnectTimeout=5", host, shlex.join(command)]
    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if done.returncode:
        return done.stdout, f"rc {done.returncode}: {done.stderr.strip()[:200]}"
    return done.stdout, ""


# The meminfo fields a capacity leg is stated in, in the order the capacity
# design lists them. Records before 2026-09-03 hold only MemAvailable and
# AnonPages, so keep both names for readers of those files. The rest tell a
# shared-memory slab, a page cache fill and an undeclared copy apart.
MEMINFO_FIELDS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Buffers", "Shmem",
                  "AnonPages", "Dirty", "Writeback", "Active(anon)", "Inactive(anon)",
                  "Active(file)", "Inactive(file)", "Unevictable", "Mlocked",
                  "SReclaimable", "SwapTotal", "SwapFree")
# One grep per sample, anchored at both ends of the name. Unanchored, Cached
# would also match SwapCached, and Shmem would match three lines. The parens in
# the LRU names are ERE groups unless escaped, and Active(anon) would match no line.
MEMINFO_PATTERN = "^(" + "|".join(
    name.replace("(", r"\(").replace(")", r"\)") for name in MEMINFO_FIELDS) + "):"
MEM_DECIMALS = 3  # 1 MiB of resolution; the legs are read to two


def parse_meminfo(text: str) -> dict[str, float]:
    """The named fields in GiB, in MEMINFO_FIELDS order.

    A field this kernel does not carry is absent, not zero: a zero would read as
    a measurement, and a leg would price a figure nobody took.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        amount = rest.split()
        if name in MEMINFO_FIELDS and amount and amount[0].isdigit():
            values[name] = round(int(amount[0]) / 2**20, MEM_DECIMALS)
    return {name: values[name] for name in MEMINFO_FIELDS if name in values}


def _mem_gib(host: str) -> tuple[dict, str]:
    """Every MEMINFO_FIELDS reading in GiB, and why a box gave none of them."""
    text, error = _on(host, ["grep", "-E", MEMINFO_PATTERN, "/proc/meminfo"], 10)
    values = parse_meminfo(text)
    return values, error or ("" if values else "no meminfo line came back")


def session_hosts(config: dict, cell: dict) -> list[str]:
    """The boxes this cell's own session runs on.

    The memory floor waits on these, and the record keeps their worker
    journals. A world 1 cell that waited on the second box would stall on
    memory it never asks for and file another run's journal lines as its own.
    """
    return ["", config["sibling"]] if config["sibling"] and cell["session"] == "cluster" else [""]


def sampling_hosts(config: dict) -> list[str]:
    """Every box the memory record covers, world 1 legs included.

    The idle box is the control on the sampler itself: a figure that moves on
    a box running nothing is the sampler, not the load under test.
    """
    return ["", config["sibling"]] if config["sibling"] else [""]


class Sampler(threading.Thread):
    """Every MEMINFO_FIELDS reading at 1 Hz on every box.

    The summary updates on every row, so it is valid the moment the leg ends.
    """

    def __init__(self, hosts: list[str], path: Path, label: str = "") -> None:
        super().__init__(daemon=True)
        self.hosts, self.path, self.stop = hosts, path, threading.Event()
        self.label = label
        self.summary: dict = {}
        self.read_errors: dict = {}
        self.error = ""

    def _fold(self, name: str, row: dict) -> None:
        if self.stop.is_set():
            return  # close() may have replaced summary with an error dict
        held = self.summary.setdefault(name, {})
        for field, value in row.items():
            seen = held.setdefault(field, {"min": value, "peak": value, "last": value})
            seen["min"] = round(min(seen["min"], value), 2)
            seen["peak"] = round(max(seen["peak"], value), 2)
            seen["last"] = round(value, 2)

    def sample_once(self, handle) -> dict:
        """One row across every box, folded into the summary and written.

        A box that answers nothing records its error and the row is still
        written: when a peer stops answering in a capacity event, the other
        box's readings are the record.
        """
        row: dict = {"t": time.time()}
        if self.label:
            row["leg"] = self.label
        for host in self.hosts:
            name = host or "head"
            row[name], failure = _mem_gib(host)
            if failure:
                self.read_errors[name] = failure
            self._fold(name, row[name])
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        return row

    def run(self) -> None:
        try:
            with open(self.path, "a") as handle:
                while not self.stop.is_set():
                    self.sample_once(handle)
                    self.stop.wait(1.0)
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        """Stop and join past the sibling ssh bound, and say so if it hangs."""
        self.stop.set()
        self.join(timeout=20)
        for name, failure in self.read_errors.items():
            self.summary.setdefault(name, {})["read_error"] = failure
        if self.is_alive():
            self.summary = {"memory_error": "the sampler thread did not stop"}
        elif self.error:
            self.summary = {"memory_error": self.error}


def ledger_rows_since(ledger: Path, offset: int) -> tuple[list[dict], int]:
    if not ledger.exists():
        return [], 0
    data = ledger.read_bytes()
    rows = [json.loads(line) for line in data[offset:].splitlines() if line.strip()]
    return rows, len(data)


def journal_lines(host: str, since: str) -> list[str]:
    text, error = _on(host, ["journalctl", "--user", "-u", "dgxm-worker.service",
                             "--since", since, "-o", "cat"], 30)
    lines = [line for line in text.splitlines() if any(key in line for key in JOURNAL_KEEP)]
    return lines + ([f"journal read failed: {error}"] if error else [])


FLOOR_STALL_POLLS = 12  # about three minutes at the 15 s cadence
FLOOR_STALL_GIB = 0.1  # the resolution the countdown line already prints


def floor_stalled(lows: list[float], polls: int = FLOOR_STALL_POLLS,
                  eps: float = FLOOR_STALL_GIB) -> bool:
    """Whether the lowest reading has stopped moving.

    A floor that drifts is a box still releasing memory and worth waiting out.
    Every floor pinned to one value for minutes in the first W0 wave
    (2026-09-01) was a worker fleet holding the previous checkpoint, which only
    a recycle frees.
    """
    if len(lows) < polls:
        return False
    recent = lows[-polls:]
    return max(recent) - min(recent) <= eps


def wait_for_memory(hosts: list[str], floor_gib: float, bounded_s: float) -> dict:
    """Wait for MemAvailable on every box. A box that gives no reading is
    recorded and the wait returns at once: read as zero, it would hold every
    cell to the deadline and report the floor unmet. A floor that has not moved
    for FLOOR_STALL_POLLS polls returns early, marked stalled, so the caller
    can replace the fleet rather than wait out the window."""
    deadline = time.time() + bounded_s
    lows: list[float] = []
    while True:
        readings, errors = {}, {}
        for host in hosts:
            values, failure = _mem_gib(host)
            if failure:
                errors[host or "head"] = failure
            else:
                readings[host or "head"] = values.get("MemAvailable", 0.0)
        low = min(readings.values(), default=0.0)
        if errors or low >= floor_gib or time.time() > deadline:
            return {"readings": readings, "errors": errors, "floor_gib": floor_gib,
                    "cleared": bool(readings) and not errors and low >= floor_gib}
        lows.append(round(low, 1))
        if floor_stalled(lows):
            return {"readings": readings, "errors": errors, "floor_gib": floor_gib,
                    "cleared": False, "stalled": len(lows)}
        print(f"memory floor: {low:.1f} GiB available < {floor_gib:.0f} floor "
              f"({', '.join(f'{k} {v:.1f}' for k, v in readings.items())}); "
              f"{int(deadline - time.time())} s left", flush=True)
        time.sleep(15)


def _commit(directory: Path) -> str:
    return _on("", ["git", "-C", str(directory), "rev-parse", "HEAD"], 10)[0].strip() or "unknown"


def _package(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _nccl_version() -> tuple[str, str]:
    """The NCCL library this process has loaded, as major.minor.patch.

    torch reports the NCCL it was built against, and a source build symlinked
    over the wheel moves only the loaded library, so a swap would go unseen.
    This reads the harness process's own library, not the workers'. It falls
    back to torch's build-time value when the library cannot answer.
    """
    try:
        import torch

        built = ".".join(str(part) for part in torch.cuda.nccl.version())
    except (ImportError, AttributeError, RuntimeError):
        return _package("nvidia-nccl-cu13"), "harness-env"
    try:
        code = ctypes.c_int()
        # The soname resolves only after torch has loaded the library.
        if ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(code)) == 0 and code.value > 0:
            return f"{code.value // 10000}.{code.value // 100 % 100}.{code.value % 100}", "loaded-library"
    except (OSError, AttributeError):
        pass
    return built, "torch-runtime"


def source_digest(repo: Path = REPO) -> str:
    """Hash the runtime source used to derive cell labels, including file names.

    Use the same module set as refusal-prefix discovery. A repository commit
    would also change for harness-only edits and invalidate settled records
    unnecessarily (176 contract cells in the 2026-09-01 wave).
    """
    digest = hashlib.sha256()
    for path in sorted((repo / "src" / "dgx_monarch").rglob("*.py")):
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def session_pin(config: dict) -> dict:
    """Record environment versions and the process each was read from."""
    torch_version, torch_source = _package("torch"), "harness-env"
    try:
        torch_version = get(f"{config['driver']}/system_stats", timeout=30)["system"][
            "pytorch_version"]
        torch_source = "driver"
    except (OSError, KeyError, ValueError):
        pass
    nccl, nccl_source = _nccl_version()
    source = (REPO / "src" / "dgx_monarch" / "gate_ledger.py").read_text()
    protocol = next((line.split("=")[1].strip() for line in source.splitlines()
                     if line.startswith("GATE_PROTOCOL_VERSION")), "unknown")
    return {"comfy_commit": _commit(config["comfy_dir"]), "monarch_commit": _commit(REPO),
            "source_digest": source_digest(),
            "torch": torch_version, "torch_source": torch_source,
            "nccl": nccl, "nccl_source": nccl_source,
            "gate_protocol_version": protocol, "driver": config["driver"],
            "world": config["world"]}


def graph_digest(out: Path, template: str) -> str:
    """The converted graph's own bytes. A pin of commits cannot see an edited
    or re-converted template, so a resume compares this too."""
    path = out / "graphs" / f"{template}.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else "missing"


def leg_plan(cell: dict) -> list[tuple[str, str, str, int, int | None]]:
    """(leg, preset, auto_gate, seed shift, steps) for one cell."""
    reference, gate = cell["reference"], cell["auto_gate"]
    plan = [("cold", cell["preset"], gate, 0, None), ("warm", cell["preset"], gate, 1, None)]
    if reference["kind"] == "resident-leg":
        plan.append(("resident", reference["preset"], "off", 1, None))
    if reference.get("probe_steps") and cell["probe"]:
        plan.append(("probe", cell["preset"], gate, 0, reference["probe_steps"]))
    return plan


def verdict_for(cell: dict, observed: str, message: str = "") -> tuple[str, str]:
    if cell["label"] == observed:
        return "PASS", ""
    detail = f"expected {cell['label']}, got {observed}"
    if cell.get("capacity_risk") and (observed == "refuse:C"
                                      or any(wall in message for wall in CAPACITY_WALLS)):
        # Every wall prices bytes against free memory, which the matrix cannot
        # know: activations for a ref-pose family, the driver host's own stack
        # beside the weights, the whole checkpoint for a 60 GiB-class file.
        # None of those answers is a fault.
        return "PASS-capacity", detail + "; a capacity boundary answered first"
    if cell.get("capacity_risk") and observed.startswith("refuse:"):
        detail += "; this cell sits at a capacity boundary the matrix does not price"
    return "FINDING", detail


def waivable(cell: dict, outcome: str) -> bool:
    """A waivable guard fired: the labelled one, or a class K refusal on a cell
    with waiver guards, such as a ring_pad the matrix could only flag because
    its trigger is a token count the driver owns."""
    return bool((cell["waiver"] and outcome == cell["label"])
                or (outcome == "refuse:K" and cell["waiver_guards"]))


def seeded_nodes(graph: dict) -> list[str]:
    """The nodes whose seed the warm leg shifts: the ones the cache must not serve."""
    return sorted(node_id for node_id, node in graph.items()
                  if any(isinstance((node.get("inputs") or {}).get(key), int) for key in SEED_KEYS))


def run_leg(cell: dict, config: dict, client_id: str, leg: str, patched: dict,
            prefix: str, timeout: float, sampled: list[str], out: Path) -> dict:
    """Queue one graph, sample memory around it, and describe what happened."""
    driver, output_dir = config["driver"], config["comfy_dir"] / "output"
    sampler = Sampler(sampled, out / "mem" / f"{cell['id']}_{leg}.jsonl", label=leg)
    sampler.start()
    record = {"prefix": prefix}
    try:
        wall, entry, state, prompt_id = queue_and_wait(driver, patched, client_id, timeout)
        outcome, message = classify_history(entry, state == "timeout")
        # What the history says the cache served, kept per leg so the warm leg
        # can be read for a served seed rather than guessed from its wall.
        record["cached_seeded"] = sorted(set(cached_nodes(entry or {})) & set(seeded_nodes(patched)))
        record["waiver_run_ids"] = waiver_run_ids(entry or {})
        if state and state != "timeout":
            outcome, message = "cell-error", state
        record["exception_type"] = error_text(entry or {}, "exception_type")[:400]
        if state and prompt_id:
            # A prompt that timed out or lost its poll may still be running, and
            # the next leg would queue behind it and inherit its wall.
            record["drain"] = drain(driver, prompt_id)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        if exc.code == 400:
            raise GraphRejected(f"{exc.code} {exc.reason}: {detail}") from exc
        wall, outcome, message = 0.0, "cell-error", f"http {exc.code}: {exc.reason}: {detail}"
    except (OSError, KeyError, ValueError) as exc:
        wall, outcome, message = 0.0, "cell-error", f"{type(exc).__name__}: {exc}"
    finally:
        sampler.close()
    record.update({"wall_s": round(wall, 1), "outcome": outcome,
                   "message": message[-MESSAGE_KEEP:],
                   "memory": sampler.summary, "frames": len(frames_for(output_dir, prefix))})
    print(f"[{cell['id']}] {leg}: {outcome} in {wall:.1f} s", flush=True)
    return record


def waived_guard_used(record: dict, cell: dict, leg: str) -> str | None:
    """The listed class-K guard spent by this prompt's rendered leg, if any."""
    guards = set(cell.get("waiver_guards") or ())
    if not guards or cell.get("label") not in {"render", "refuse:K"}:
        return None
    leg_record = (record.get("legs") or {}).get(leg) or {}
    run_ids = leg_record.get("waiver_run_ids") if isinstance(leg_record, dict) else None
    if not isinstance(run_ids, list):
        return None
    run_ids = {run_id for run_id in run_ids if isinstance(run_id, str) and run_id}
    if not run_ids:
        return None
    for row in record.get("ledger_rows") or ():
        if not isinstance(row, dict):
            continue
        if (isinstance(row.get("target_guard"), str)
                and row.get("verdict") == "WAIVER" and row.get("waiver_class") == "K"
                and row.get("action") == "use" and row.get("target_guard") in guards
                and isinstance(row.get("run_id"), str) and row["run_id"] in run_ids):
            return str(row["target_guard"])
    return None


def runner_granted_render(record: dict, cell: dict) -> bool:
    """A runtime-discovered K refusal was granted and its retry rendered."""
    return bool(
        (record.get("legs") or {}).get("cold", {}).get("outcome") == "refuse:K"
        and (record.get("legs") or {}).get("waived", {}).get("outcome") == "render"
        and (record.get("consent") or {}).get("granted") is True
        and waived_guard_used(record, cell, "waived")
    )


def carried_grant_render(record: dict, cell: dict) -> bool:
    """Whether the cold leg rendered under a grant an earlier cell left live.

    A class K memo is scoped to the combination, the topology and the world and
    stays live until an operator revokes it, so the first cell of a shape to be
    waived leaves every later cell of that shape rendering under the same grant.
    Such a cell never refuses, so it reaches neither the waived leg nor a
    consent this runner granted. Only the ledger says what happened: the guard
    wrote a WAIVER row with action ``use`` for one of the label's own guards.
    Those rows tell a carried grant from a bar that silently stopped firing.
    """
    if (record.get("legs") or {}).get("cold", {}).get("outcome") != "render":
        return False
    return waived_guard_used(record, cell, "cold") is not None


def waived_leg_bar(record: dict, cell: dict, bar: str) -> bool:
    """Return whether a granted class K waiver replaces the step-nrms bar.

    A sol kernel is known at matrix-build time. Other class K deviations become
    known only when a guard fires and the operator grants a waiver, or when the
    render uses an earlier grant. Both report deviation without a fidelity floor.

    Only step-nrms changes. Bit-identical remains an exact pixel contract; bars
    that already have no floor stay unchanged.
    """
    return (bar == "step-nrms"
            and (runner_granted_render(record, cell) or carried_grant_render(record, cell)))


def stale_reference(record: dict, reference: dict, out: Path | None) -> str:
    """Why the reference cell's last render cannot be read against this one, or "".

    The reference image on disk is whatever build last rendered that cell. A
    named run that leaves the reference cell out compares this build's render
    with another build's, and a family whose output legitimately moved between
    the two reads as a fault (W3c, 2026-09-08: a chroma cfg2 cell scored
    0.63 NRMS against a reference rendered six days and three cfg fixes
    earlier). The record's pin decides, on the same keys a resume reads.
    """
    if reference.get("kind") == "resident-leg" or out is None:
        return ""
    path = out / "cells" / f"{reference['cell']}.json"
    if not path.is_file():
        return (f"reference cell {reference['cell']} has no record in this session or "
                "the other; run it in its own session before this cell")
    done = json.loads(path.read_text())
    old = done.get("pin") or {}
    mine = record.get("pin") or {}
    moved = [key for key in RESUME_KEYS if old.get(key) != mine.get(key)]
    # The converted graph is part of what rendered the image: W3c's chroma
    # references were rendered on a graph the convert step at driver boundary 3
    # replaced.
    if done.get("graph_digest") != record.get("graph_digest"):
        moved.append("graph_digest")
    if not moved:
        return ""
    return (f"reference cell {reference['cell']} last rendered on another build "
            f"({', '.join(moved)} moved); its image on disk is not this build's, so "
            "name it to re-render before this cell")


def _fidelity(record: dict, cell: dict, output_dir: Path, floor: float,
              out: Path | None = None) -> None:
    reference = cell["reference"]
    leg = "warm"
    if reference["kind"] == "resident-leg":
        against = f"sweep_{cell['id']}_resident"
    else:
        leg = "probe" if reference.get("probe_steps") and cell["probe"] else "warm"
        against = f"sweep_{reference['cell']}_{leg}"
    if stale := stale_reference(record, reference, out):
        fidelity: dict = {"error": stale}
    else:
        fidelity = compare_outputs(output_dir, f"sweep_{cell['id']}_{leg}", against, cell["audio"])
    bar = WAIVED_BAR if waived_leg_bar(record, cell, reference["bar"]) else reference["bar"]
    fidelity["bar"] = bar
    record["fidelity"] = fidelity
    if fidelity.get("error"):
        record["findings"].append(f"reference compare: {fidelity['error']}")
    elif bar == "bit-identical" and not fidelity.get("identical"):
        record["verdict"] = "FINDING"
        record["findings"].append(f"expected bit-identical, got {fidelity}")
    elif bar == FULL_RENDER_BAR:
        # No probe leg, so this compares two full warm renders. The step-nrms
        # floor was calibrated at one step, where divergence has had one step to
        # grow, so it does not answer here and the number is reported against no
        # floor.
        line = (f"{FULL_RENDER_BAR} {fidelity.get('nrms')} against "
                f"{fidelity.get('reference_file', 'the reference')}: no probe leg is available "
                f"on this cell, so no floor applies; "
                f"{cell.get('probe_reason') or 'this reference spec plans no probe leg'}")
        if fidelity.get("identical"):
            # Two renders identical to the pixel need no floor: the PASS stands
            # and the number rides as a note.
            record["notes"].append(line)
        else:
            if record["verdict"].startswith("PASS"):
                record["verdict"] = "CHECK"
            # The report moves this line out of the findings list by matching
            # the copy stored here, not by its wording.
            record["full_render"] = line
            record["findings"].append(line)
    elif bar == WAIVED_BAR:
        # A render known wrong on purpose is the PASS, and this number is what
        # the cell measures. No floor answers for a deviation that was asked
        # for, and a CHECK would file the result as a doubt. The line names the
        # cause: an approximate kernel the matrix named before the run, or the
        # guard a granted card waived.
        if bar != reference["bar"] and carried_grant_render(record, cell):
            why = (f"a class K grant for {'/'.join(cell['waiver_guards'])} was "
                   "already live and this leg rendered under it")
        elif bar != reference["bar"]:
            why = (f"the runner granted the class K card for "
                   f"{'/'.join(cell['waiver_guards'])} and this leg rendered under it")
        else:
            why = (f"{cell.get('attention') or 'this kernel'} is an approximation "
                   "the class K waiver admits")
        record["waived_render"] = line = (
            f"{WAIVED_BAR} {fidelity.get('nrms')} against "
            f"{fidelity.get('reference_file', 'the reference')}: {why}, so the "
            "deviation is the measurement and no floor applies")
        record["notes"].append(line)
    elif bar == "step-nrms" and (fidelity.get("nrms") or 0.0) > floor:
        # Not a FINDING: the null A/B calibrates what this number means first.
        if record["verdict"].startswith("PASS"):
            record["verdict"] = "CHECK"
        record["findings"].append(
            f"step-nrms {fidelity['nrms']} over the {floor} floor against "
            f"{fidelity.get('reference_file', 'the reference')}")
    if not fidelity.get("error") and (audio := (fidelity.get("audio") or {}).get("error")):
        # The frames compared and the audio did not. A one-step probe on an
        # audio template can render near silence, which no envelope
        # correlates, so this is an unproven measurement (CHECK), never a
        # fault, and carries the `reference compare: ` prefix that says so.
        record["findings"].append(f"reference compare: audio envelope: {audio}")
    if out is not None and (split := cfg_split_note(record, cell, out)):
        # A cfg cell that never split renders the right pixels on both ranks,
        # so no fidelity bar can see it; only the wall shows it.
        if record["verdict"].startswith("PASS"):
            record["verdict"] = "CHECK"
        record["findings"].append(split)


# How close a cfg cell's warm wall may sit to its single reference's before a
# split that never reached the model call is the likelier reading. Every rank
# rendering the whole batch takes one rank's wall; a split that landed reads
# 1.7 to 2.4x on this fleet (2026-09-04), so 10 percent leaves the gap wide.
CFG_SPLIT_RATIO = 0.9


def cfg_split_note(record: dict, cell: dict, out: Path) -> str:
    """The line for a cfg cell whose warm wall matches its single reference.

    Comfy applies a DIFFUSION_MODEL wrapper only where the family's own forward
    builds the executor, so a split installed on a family that builds none is
    inert: both ranks render the whole batch, the output is correct, and only
    the wall says the topology did nothing. The single reference is the one leg
    that prices one rank's work, so a cell with no such reference makes no
    claim here.
    """
    if int(cell.get("resolved_cfg") or 1) < 2 or cell["reference"].get("kind") != "cell":
        return ""
    warm = ((record.get("legs") or {}).get("warm") or {})
    if warm.get("outcome") != "render" or not warm.get("wall_s"):
        return ""
    path = out / "cells" / f"{cell['reference']['cell']}.json"
    if not path.is_file() or stale_reference(record, cell["reference"], out):
        return ""  # a wall from another build prices nothing here
    against = json.loads(path.read_text())
    if int(against["cell"].get("resolved_cfg") or 1) != 1:
        return ""  # a lever cell scores against the auto row, which may be cfg too
    single = ((against.get("legs") or {}).get("warm") or {})
    if single.get("outcome") != "render" or not single.get("wall_s"):
        return ""
    mine, theirs = float(warm["wall_s"]), float(single["wall_s"])
    if mine < theirs * CFG_SPLIT_RATIO:
        return ""
    return (f"cfg2 did not split: a {mine} s warm leg against a {theirs} s single "
            f"reference on {against['cell']['id']}. Two ranks that each render the "
            "whole cond/uncond batch take one rank's wall, and the pixels are right "
            "either way, so no fidelity bar sees it")


# A refusal that arrives after the weights are on the box leaves the fleet
# holding them, and the next cell can then answer on the residue rather than
# on its own guard. Two signals, because neither sees both loads: a stock load
# grows anonymous memory, while a slab load is a shared memfd mapping that moves
# no AnonPages, so only the wall records it.
LOAD_ANON_GIB = 3.0
LOAD_WALL_S = 30.0


def load_evidence(record: dict) -> str:
    """Why the cell looks like it loaded weights, or "" for no sign of one.

    The memory sample is the evidence, not the journal: the journal keeps only
    lines that match JOURNAL_KEEP, and an ordinary stock load leaves no trace in
    it. AnonPages is read peak minus min rather than against the first row,
    because a fleet still draining the previous load delivers its residue in
    that first row and the growth would read zero on the cells this rule is
    for. The waived leg counts as well as the cold one: a guard that refuses in
    two seconds and then renders under its card has loaded the checkpoint too.
    """
    legs = record.get("legs") or {}
    for name in ("cold", "waived"):
        leg = legs.get(name) or {}
        memory = leg.get("memory") or {}
        if memory.get("memory_error"):
            return f"the {name} memory sample is unreadable"
        for host, fields in memory.items():
            if not isinstance(fields, dict):
                continue
            if fields.get("read_error"):
                return f"{host} gave no memory reading"
            anon = fields.get("AnonPages")
            if not isinstance(anon, dict):
                continue
            grew = round(float(anon.get("peak", 0.0)) - float(anon.get("min", 0.0)), 2)
            if grew > LOAD_ANON_GIB:
                return f"{host}'s anonymous memory peaked {grew} GiB above its low"
        wall = float(leg.get("wall_s") or 0.0)
        if wall > LOAD_WALL_S:
            return f"a {wall} s {name} leg"
    return ""


class LastCell(NamedTuple):
    """What the cell before this one left the fleet holding.

    ``available_gib`` is what the box had left, not the growth this cell
    caused: a cell that inherited a load from a kept fleet and added nothing
    would read zero growth, while this figure still shows the inherited load.
    """

    template: str
    managed: str
    observed: str
    load: str
    available_gib: float | None


def final_available_gib(record: dict) -> float | None:
    """The lowest of each box's last MemAvailable reading on this cell's last sampled leg.

    That is the box the next cell inherits when the fleet is not replaced: the
    weights the fleet still holds are already out of the reading, and so is
    every earlier cell's residue. The last leg that sampled at all ran nearest
    the handover; every leg samples or none do, so in a real record it is the
    last leg. A broken sample answers None rather than a number, because a
    missing reading is not a full box, and the caller replaces the fleet on it.
    """
    sampled = [leg for leg in (record.get("legs") or {}).values() if (leg or {}).get("memory")]
    if not sampled:
        return None
    memory = sampled[-1]["memory"]
    if memory.get("memory_error"):
        return None
    readings = []
    for fields in memory.values():
        if not isinstance(fields, dict) or fields.get("read_error"):
            return None
        available = fields.get("MemAvailable")
        if not isinstance(available, dict):
            return None
        readings.append(float(available.get("last", 0.0)))
    return round(min(readings), 2) if readings else None


def fleet_need_gib(cell: dict, floor_gib: float) -> float:
    """Return the greater of the runner's memory floor and stock-load requirement.

    The runtime's stock price includes checkpoint bytes, a host copy, and its
    absolute host floor. The next residency mode is unknown; slab and FSDP shard
    builds cost less, while comfy-managed pinned staging can cost more. The latter
    depends on disable_pinned_memory, which no matrix lever sets, and cannot be
    estimated here. Overestimating may recycle unnecessarily; underestimating
    can leave the next cell failing capacity because of the previous load.
    """
    # The runtime owns the stock price and its floor (capacity_fit.py,
    # capacity_floor.py), so the harness calls it rather than restating it.
    from dgx_monarch.capacity_fit import stock_required_bytes

    size = int(float(cell.get("checkpoint_gib") or 0.0) * (1 << 30))
    return round(max(float(floor_gib), stock_required_bytes(size) / (1 << 30)), 1)


def needs_fleet_reset(last: LastCell | None, cell: dict, floor_gib: float) -> str:
    """Return a replacement reason, or an empty string to reuse the fleet.

    Always replace for a new template or comfy_managed value, or if the previous
    cell did not render or supplied no memory reading. Failed proofs and late
    refusals can leave models resident. For example, the 2026-09-01 ideogram4
    uly2+fsdp refusal left both checkpoints loaded and only 51.8 GiB available on
    the head, below the runner's 60 GiB floor.

    For a completed render with the same template and latch, policy updates drop
    incompatible residents. Reuse is then limited by remaining MemAvailable:
    replace when it is below either the runner floor or this cell's estimated
    stock-load requirement. Otherwise old allocations cannot cross those two
    thresholds. Checking here also avoids waiting for the floor poll to stall.
    """
    if last is None:
        return ""
    if last.template != cell["template"]:
        return "template"
    if last.managed != str(cell["levers"].get("comfy_managed", "off")):
        return "comfy_managed"
    if last.observed != "render":
        return f"after {last.observed}" + (f" ({last.load})" if last.load else "")
    if last.available_gib is None:
        return "the previous cell gave no memory reading"
    need = fleet_need_gib(cell, floor_gib)
    if last.available_gib < need:
        return (f"the fleet left {last.available_gib} GiB available, under the "
                f"{need} GiB this cell needs")
    return ""


# driver.recycle's answer when the route refuses the replacement outright.
RECYCLE_BUSY = "http 503"
WEDGE_RECYCLE_RUN = 2
# mesh.py raises both clauses and mesh_health.py publishes the first: a fleet
# the driver will not replace, and a transport poisoned by a partial bring-up
# or a creation failure. Neither clears without a driver restart, so the
# session ends on one rather than filing a finding against every cell left in
# the wave. The second clause is anchored past "reusable" because a transport
# that is not reusable while another mesh creation is in progress is a wait,
# not a wedge.
WEDGE_TEXTS = ("replacement is blocked", "not reusable after")


class SessionWatch:
    """The fleet state between cells, and the two signals it ends a session on.

    A wedged driver answers every later cell in two seconds, so a runner that
    keeps going files findings against cells that never ran. Two signals here
    say the rig, not the cell, is the answer: a recycle the driver refuses
    twice in a row, and a driver message naming a fleet no attach can replace
    or a transport no attach can reuse. The mesh block the driver publishes is
    read after every replacement and reported with either of them, but it never
    raises one of its own; wait_to_attach says why.
    """

    def __init__(self, driver: str, pace_s: float) -> None:
        self.driver, self.pace_s = driver, float(pace_s)
        self.refused_recycles = 0
        self.replaced_at = 0.0
        self.dirty_block: dict = {}

    def recycled(self, answer: str) -> None:
        """Record one recycle answer and start the pace the next attach owes."""
        self.replaced_at = time.time()
        self.refused_recycles = self.refused_recycles + 1 if answer == RECYCLE_BUSY else 0
        if self.refused_recycles >= WEDGE_RECYCLE_RUN:
            raise self._wedged(
                f"the recycle answered {answer} on {self.refused_recycles} tries in a row, "
                "so the old fleet stays up and every later cell would run on it "
                "instead of a fresh one")

    def message(self, text: str) -> None:
        """Raise when a driver message names a fleet or transport no reset clears."""
        hit = next((phrase for phrase in WEDGE_TEXTS if phrase in (text or "")), "")
        if hit:
            raise self._wedged(f"the driver answered {hit!r}, which no reset clears")

    def _wedged(self, reason: str) -> SessionWedged:
        """One wedge reason, carrying the dirty mesh block when there was one."""
        if self.dirty_block:
            reason += f"; the mesh block read {json.dumps(self.dirty_block)[:400]}"
        return SessionWedged(f"{reason}; restart the driver and the worker loops")

    def wait_to_attach(self) -> None:
        """Hold the attach pace a replacement owes, then read the mesh block.

        The pace is monarch's own attach budget, read by
        matrix.attach_pace_default: an attach right behind a replacement meets a
        fleet still coming up. It is a deadline from the recycle rather than a
        sleep, so the memory floor poll before the attach counts against it.

        The block is evidence, never the trigger. The one dirty row a harness
        meets, a handle created and never published, clears on the next
        get_mesh (mesh_health.py MeshHealth.dirty), so a second read before an
        attach would prove no more than the first. The two states no reset
        clears, a blocked replacement and a poisoned transport, make the next
        attach raise a WEDGE_TEXTS message. The wedge is the text match and the
        503 run; a dirty read rides in their reason.
        """
        if not self.replaced_at:
            return
        left = self.pace_s - (time.time() - self.replaced_at)
        self.replaced_at = 0.0
        if left > 0:
            print(f"attach pace: waiting {left:.0f} s after the fleet replacement", flush=True)
            time.sleep(left)
        block = self.mesh_block()
        self.dirty_block = block if block and fleet_is_dirty(block) else {}
        if self.dirty_block:
            print(f"the mesh block reads dirty after the pace: "
                  f"{json.dumps(self.dirty_block)[:400]}", flush=True)

    def mesh_block(self) -> dict:
        """The driver's own mesh block, or {} when it did not answer.

        An unanswered read proves nothing about the fleet, so it is not a wedge.
        """
        try:
            block = get(f"{self.driver}/dgxm/telemetry", timeout=30).get("mesh")
        except (OSError, ValueError, KeyError, AttributeError):
            return {}
        return block if isinstance(block, dict) else {}


def fleet_is_dirty(block: dict) -> bool:
    """The runtime's own predicate over one mesh telemetry block.

    The harness calls it rather than restating it: two readers that disagree on
    one block let a dirty fleet earn a green exit code.
    """
    from dgx_monarch.mesh_evidence import mesh_block_dirty

    return mesh_block_dirty(block)


def reset_fleet(config: dict, client_id: str, managed: str, mode: str,
                watch: SessionWatch | None = None) -> dict:
    """Clear the driver's residency, then replace the fleet.

    comfy_managed latches for the life of a worker process, so a cell that moves
    it needs new workers or it refuses on the latch instead of on its own guard.
    The clear graph carries the outgoing value for the same reason, and the
    recycle after it resets the latch for the next cell.
    """
    driver = config["driver"]
    if watch is not None:
        watch.wait_to_attach()
    wall, entry, state, prompt_id = queue_and_wait(
        driver, clear_graph(driver, mode, managed), client_id, 300)
    reset = {"clear_s": round(wall, 1), "clear_state": state, "comfy_managed_was": managed}
    if watch is not None:
        # The clear graph attaches too, so it meets the wedge before the recycle
        # that cannot clear it does.
        watch.message(error_text(entry or {}))
    if state:
        reset["drain"] = drain(driver, prompt_id)
        if reset["drain"] != "drained":
            raise SessionWedged(f"the clear graph did not leave the queue ({reset['drain']})")
    reset["recycle"] = recycle(driver)
    if watch is not None:
        watch.recycled(reset["recycle"])
    return reset


def probe_transport(config: dict, session: str, client_id: str,
                    watch: SessionWatch | None = None) -> str:
    """Open one mesh at the session's transport before any cell is scored. A
    session on a driver serving the other transport refuses every cell, so it
    ends here rather than filing a whole findings list. Telemetry cannot answer
    it: the worker list stays empty until a mesh exists (nodes/routes.py)."""
    driver, mode = config["driver"], "local" if session == "local" else "cluster"
    if watch is not None:
        watch.wait_to_attach()
    try:
        _wall, entry, state, prompt_id = queue_and_wait(
            driver, clear_graph(driver, mode), client_id, 300)
    except urllib.error.HTTPError as exc:
        # Built from the Init node's defaults, so a rejection is this harness,
        # not the rig: every cell runs a template instead.
        return f"probe graph rejected (http {exc.code}), transport unchecked"
    except (OSError, KeyError, ValueError) as exc:
        raise SessionWedged(f"the driver did not answer the probe: "
                            f"{type(exc).__name__}: {exc}") from exc
    outcome, message = classify_history(entry, state == "timeout")
    if state:
        drain(driver, prompt_id)
        outcome = state if state != "timeout" else outcome
    if watch is not None:
        watch.message(message)
    if outcome != "render":
        raise SessionWedged(
            f"the driver would not open a {mode} mesh ({outcome}); run the {session} "
            f"session against a driver started for it. {message[:400]}")
    return mode


def run_cell(cell: dict, config: dict, pin: dict, client_id: str, args,
             reset: dict | None = None, watch: SessionWatch | None = None) -> dict:
    driver, out = config["driver"], config["out_dir"]
    output_dir = config["comfy_dir"] / "output"
    hosts, sampled = session_hosts(config, cell), sampling_hosts(config)
    graph = json.loads((out / "graphs" / f"{cell['template']}.json").read_text())
    ledger = output_dir / "dgxm_gate_ledger.jsonl"
    timeout = args.timeout_s or cell["timeout_s"]
    _rows, offset = ledger_rows_since(ledger, 0)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    # comfy counts a save prefix up from what is on disk, so a re-run would
    # append to the last one and compare frames across two pins.
    stale = [path for path in output_dir.glob(f"sweep_{cell['id']}_*") if path.is_file()]
    for path in stale:
        path.unlink()
    began, bound = time.time(), 600.0
    floor = wait_for_memory(hosts, config["mem_floor_gib"], bound)
    floor_reset: dict = {}
    if floor.get("stalled"):
        # A stalled floor is a fleet holding the previous checkpoint
        # (floor_stalled). Replace it once, then read again on what is left of
        # the window rather than expiring into it.
        floor_reset = reset_fleet(config, client_id,
                                  str(cell["levers"].get("comfy_managed", "off")),
                                  cell["mode"], watch)
        floor_reset["reason"] = f"the floor stopped moving within {floor['stalled']} polls"
        print(f"[{cell['id']}] floor reset: recycle {floor_reset['recycle']}", flush=True)
        floor = wait_for_memory(hosts, config["mem_floor_gib"],
                                max(60.0, bound - (time.time() - began)))
    record: dict = {"cell": cell, "pin": pin, "started": started, "legs": {},
                    "graph_digest": graph_digest(out, cell["template"]),
                    "reset": reset or {}, "floor_reset": floor_reset,
                    "findings": [], "notes": [],
                    "cleared_stale_files": len(stale), "memory_floor": floor}
    (out / "mem").mkdir(parents=True, exist_ok=True)
    if watch is not None:
        watch.wait_to_attach()
    observed, rendered, wedged = "not-run", False, ""
    for leg, preset, gate, shift, steps in leg_plan(cell):
        prefix = f"sweep_{cell['id']}_{leg}"
        patched = patch_graph(graph, cell, preset, gate, shift, prefix, steps)
        # Never before the warm leg: the reset unloads every model too, which
        # would make the warm wall a second cold one. Its seed shift is enough.
        if leg not in ("cold", "warm"):
            record.setdefault("cache_bust", {})[leg] = free_cache(driver)
        try:
            record["legs"][leg] = leg_record = run_leg(
                cell, config, client_id, leg, patched, prefix, timeout, sampled, out)
        except GraphRejected as exc:
            # One bad widget value 400s, and every cell on this graph fails the
            # same way. main skips the template; rejection_scope decides the rest.
            observed = "graph-rejected"
            record["rejection"] = str(exc)[:2000]
            record["findings"].append(record["rejection"])
            break
        if leg_record.get("drain", "drained") != "drained":
            wedged = f"{cell['id']} {leg}: the queue would not clear ({leg_record['drain']})"
            break
        if leg != "cold":
            continue
        observed, rendered = leg_record["outcome"], leg_record["outcome"] == "render"
        if not rendered:
            # A waivable guard refused and offered a consent card: grant it,
            # free the execution cache, and queue the same graph.
            if not waivable(cell, observed):
                break
            record["consent"] = grant_pending_consent(driver, cell)
            record.setdefault("cache_bust", {})["waived"] = free_cache(driver)
            waived_prefix = f"sweep_{cell['id']}_waived"
            record["legs"]["waived"] = waived = run_leg(
                cell, config, client_id, "waived",
                patch_graph(graph, cell, preset, gate, shift, waived_prefix),
                waived_prefix, timeout, sampled, out)
            rendered = waived["outcome"] == "render"
            if not rendered:
                granted = record["consent"].get("granted")
                record["findings"].append(
                    f"waiver {'granted' if granted else 'not granted'}, "
                    f"the render still {waived['outcome']}")
                break
    record["ledger_rows"], _end = ledger_rows_since(ledger, offset)
    record["journal"] = {host or "head": journal_lines(host, started) for host in hosts}
    record["observed"] = observed
    record["verdict"], detail = verdict_for(
        cell, observed, record["legs"].get("cold", {}).get("message", ""))
    waived_leg = record["legs"].get("waived", {}).get("outcome")
    if detail and observed == "refuse:K" and cell["waiver_guards"] and waived_leg == "render":
        # The guard fired on a token count only the driver sees, the card was
        # granted, and the waived leg rendered. That leg is the result.
        record["verdict"], detail = "PASS", ""
        record["notes"].append(f"{'/'.join(cell['waiver_guards'])} waived; "
                               "the score comes from the waived leg")
    elif detail and carried_grant_render(record, cell):
        # The cell could only be labelled refuse:K, and it rendered because a
        # grant an earlier cell of this shape left live cleared the guard before
        # the cold leg reached it. The ledger's `use` row is the proof, and the
        # render it describes is stamped, so the label is met.
        record["verdict"], detail = "PASS", ""
        record["notes"].append(
            f"{'/'.join(cell['waiver_guards'])} was already granted; the cold "
            "leg rendered under that live grant and is stamped")
    if detail:
        record["notes" if record["verdict"].startswith("PASS") else "findings"].append(detail)
    # The verdict above never reads the warm, resident or probe legs, so each
    # one the cold leg earned must render too, or a warm crash would read PASS.
    reference_refused = ""
    for name, leg_record in record["legs"].items():
        if name == "cold":
            continue  # the verdict above already scored it
        if name == "resident" and leg_record["outcome"] != "render":
            # The resident leg is this cell's reference, not a leg under test,
            # and the capacity wall it hits is the reason the FSDP row exists.
            reference_refused = leg_record["outcome"]
            record["notes"].append(f"resident reference {reference_refused}")
            continue
        if leg_record["outcome"] == "refuse:K" and (record.get("consent") or {}).get("granted"):
            # Whether one grant carries to a later leg is the driver's
            # question, not the matrix's, so it rides as a note.
            record["notes"].append(f"{name} leg refused K again after the waiver")
            continue
        if rendered and leg_record["outcome"] != "render":
            record["verdict"] = "FINDING"
            record["findings"].append(f"{name} leg: {leg_record['outcome']}")
        elif leg_record["outcome"] == "render" and not leg_record["frames"]:
            record["verdict"] = "FINDING"
            record["findings"].append(f"{name} leg rendered and saved no frame")
    if flagged := cache_flag(record):
        record["findings"].append(flagged)
    if reference_refused:
        record["fidelity"] = {"skipped": f"resident reference {reference_refused}"}
    elif rendered and cell["reference"]["kind"] != "none":
        _fidelity(record, cell, output_dir,
                  float(config["reference"].get("nrms_floor", NRMS_FLOOR)), out)
    settle_verdict(record)
    write_record(out, cell["id"], record)
    if watch is not None:
        # After the record is on disk: a wedge ends the session, and the cell
        # that met it is the evidence of what the driver was answering.
        for leg_record in record["legs"].values():
            watch.message(f"{leg_record.get('message', '')}\n"
                          f"{leg_record.get('exception_type', '')}")
    if wedged:
        raise SessionWedged(wedged)
    return record


# A warm wall under this share of its baseline wall more likely measured the
# execution cache than a render.
CACHE_RATIO = 0.2
# Prefer the resident reference when checking warm time for cached output.
# FSDP cold time includes a first-use clean reload that the warm run skips;
# comparing those times can incorrectly flag a real render as cached.
# Without a resident reference, use cold time because records do not isolate
# the first-use check's duration.
CACHE_BASELINE_LEGS = ("resident", "cold")


def cache_flag(record: dict) -> str:
    """The line for a warm leg the execution cache may have served, or "".

    ComfyUI serves an identical prompt out of its execution cache. The runner
    moves the seed to defeat that, and this says when the move did not take:
    from the history's execution_cached message where the leg kept it, else
    from the warm wall against CACHE_BASELINE_LEGS.
    """
    legs = record.get("legs") or {}
    warm = legs.get("warm") or {}
    if warm.get("outcome") != "render":
        return ""
    if "cached_seeded" in warm:
        # The history decides. A seeded node the cache served is the shift not
        # taking; a warm leg that ran its seeded nodes is a render, however fast
        # it was beside a cold leg that paid a ceremony, a compile or a slab
        # build the warm leg never pays.
        served = [str(node) for node in warm.get("cached_seeded") or []]
        if served:
            return (f"warm leg: the execution cache served seeded node(s) {', '.join(served)}; "
                    "the seed shift did not take")
        return ""
    # Records written before 2026-09-07 carry no history signal: the wall ratio.
    for name in CACHE_BASELINE_LEGS:
        leg = legs.get(name) or {}
        if leg.get("outcome") != "render":
            continue
        wall = float(leg.get("wall_s") or 0.0)
        if float(warm.get("wall_s") or 0.0) < wall * CACHE_RATIO:
            return (f"warm leg ran in {warm['wall_s']} s against a {wall} s {name} leg; "
                    "the execution cache probably served it")
        return ""
    return ""


# Findings that report an unproven measurement rather than a fault: a compare
# the harness could not make, such as one against a reference leg that did not
# run in this wave, and a warm leg the execution cache served or may have
# served. The harness already scores a fidelity question at CHECK, and these
# read the same way.
UNPROVEN_FINDINGS = ("reference compare: ", "warm leg ran in", "warm leg: the execution cache served")


def settle_verdict(record: dict) -> str:
    """No PASS survives a finding filed against the same cell.

    The cold leg earns the verdict, so without this a cold refusal that matches
    its label, takes its waiver and then crashes keeps PASS with the crash in
    its findings (2026-09-02). The findings list is the defect list, so it
    decides, except where every line reports something unproven
    (UNPROVEN_FINDINGS), which reads CHECK. A CHECK stands: the null A/B must
    calibrate the step-nrms bar before it can name a fault.
    """
    if record["findings"] and record["verdict"].startswith("PASS"):
        record["verdict"] = "CHECK" if all(
            line.startswith(UNPROVEN_FINDINGS) for line in record["findings"]) else "FINDING"
    return record["verdict"]


def write_record(out: Path, cell_id: str, record: dict) -> None:
    (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "cells" / f"{cell_id}.json").write_text(json.dumps(record, indent=2, default=str))


def write_summary(out: Path, session: str, summary: dict) -> Path:
    """How the session itself ended, beside the per-cell records.

    The outer shell reads this: the runner exits non-zero on a wedge, and the
    reason names what a driver and worker-loop restart has to clear.
    """
    path = out / f"run_summary_{session}.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def keep_attempt(out: Path, cell_id: str) -> Path:
    """Retire the written record under the next attempt number.

    An unsettled attempt is evidence: the crash that a fresh fleet answers is
    how the fleet is known to have been dirty.
    """
    attempts = len(list((out / "cells").glob(f"{cell_id}.attempt*.json")))
    kept = out / "cells" / f"{cell_id}.attempt{attempts + 1}.json"
    (out / "cells" / f"{cell_id}.json").rename(kept)
    return kept


RECOVERABLE_CRASHES = ("SampleResultBusyError", "MeshAttachError", "FsdpGateProofError")


def recoverable_crash(record: dict) -> bool:
    """A cold-leg crash a fresh fleet has answered before.

    These types name a mesh the previous cell left dirty rather than anything
    this cell asked for: the cached mesh cannot be used or replaced, an
    abandoned sample may still be running, or, on an FSDP cell, the
    clean-reload proof ran on such a mesh and raised FsdpGateProofError in
    place of the fault that stopped it. Since 2026-09-03 driver.py labels that
    abort refuse:untyped by its opening, so FsdpGateProofError reaches this
    check only when no untagged prefix matches its message. In the first W0
    wave (2026-09-01) two MeshAttachError crashes and two SampleResultBusyError
    crashes ran again the same day on a replaced fleet, and each answered the
    untyped refusal its label predicted. The one FsdpGateProofError crash that
    ran again answered the driver's process-local INCONCLUSIVE denial, which
    the fleet replacement before it did not clear.
    """
    cold = (record.get("legs") or {}).get("cold") or {}
    if cold.get("outcome") != "crash":
        return False
    return any(line.strip().endswith(RECOVERABLE_CRASHES)
               for line in str(cold.get("exception_type", "")).splitlines())


REJECT_WINDOW_S = 60.0
REJECT_RUN = 3


def rejection_scope(message: str) -> str:
    """Whether a rejected graph condemns one template or the whole session.

    A 400 from the validator names the node whose widget it could not read. One
    node of a template's own fails only that template, and the rest of the wave
    still runs. No node at all, or the Init node every template carries, says
    the driver is refusing graphs as such and there is nothing left to run.
    """
    start = message.find("{")
    try:
        payload = json.loads(message[start:]) if start >= 0 else {}
    except ValueError:
        # A long body is cut at 2000 characters and cannot be read.
        return "unknown"
    nodes = payload.get("node_errors") if isinstance(payload, dict) else None
    if not isinstance(nodes, dict):
        return "unknown"
    if not nodes:
        return "session"
    named = {str(node.get("class_type", "")) for node in nodes.values()
             if isinstance(node, dict)}
    return "session" if INIT_CLASS in named else "template"


def select(cells: list[dict], args) -> tuple[list[dict], dict]:
    """The cells this session runs, and every count the selection cut."""
    session = [cell for cell in cells if cell["session"] == args.session]
    pruned: dict = {"sampled_out": [cell["id"] for cell in session if cell["sampled_out"]],
                    "skipped": [cell["id"] for cell in session
                                if not cell["sampled_out"] and cell["label"].startswith("skip:")]}
    live = [cell for cell in session
            if not cell["sampled_out"] and not cell["label"].startswith("skip:")]
    if args.cells and args.cells != "all":
        wanted = {token.strip() for token in args.cells.split(",") if token.strip()}
        named, waivers = [], []
        for cell in live:
            by_name = bool(wanted & {cell["id"], cell["template"]})
            if not by_name and cell["label"] not in wanted:
                continue
            # A label token names a contract class. A waivable cell answers that
            # class only to be granted its card and render, which is the sol
            # wave's question, not the contract's. An id or a template names the
            # cell itself, so it is never dropped.
            if not by_name and not args.include_waivers \
                    and (cell["waiver"] or cell["waiver_guards"]):
                waivers.append(cell["id"])
                continue
            named.append(cell)
        pruned["waivers_excluded"] = waivers
        keep = {cell["id"] for cell in named} | set(waivers)
        pruned["not_named"] = [cell["id"] for cell in live if cell["id"] not in keep]
        live = named
    if args.limit:
        pruned["over_limit"] = [cell["id"] for cell in live[args.limit:]]
        live = live[:args.limit]
    else:
        # A video cell costs hours: one per run unless the operator says more.
        video = [cell for cell in live if cell["klass"] == "video"]
        pruned["video_beyond_first"] = [cell["id"] for cell in video[1:]]
        live = [cell for cell in live if cell["klass"] != "video"] + video[:1]
    return live, pruned


def compares_against_reference(cell: dict) -> bool:
    """Whether this cell can reach a fidelity compare against its reference cell.

    ``run_cell`` calls ``_fidelity`` only when the cold leg rendered or a
    waived retry did (``waivable``). A cell labelled ``render`` takes the first
    path; one the matrix marked waivable, or whose ``waiver_guards`` name a
    class K guard the driver discovers, can take the second although its label
    is a typed refusal. Every other typed refusal is expected to answer without
    reading the reference spec ``matrix._reference`` gives every cell, so
    leaving that reference unpulled, or blocked, costs such a cell nothing.
    """
    return bool(cell["label"] == "render" or cell["waiver"] or cell["waiver_guards"])


def with_stale_references(runnable: list[dict], cells: list[dict], out: Path,
                          pin: dict, *, skip_crashed: bool = False,
                          ) -> tuple[list[dict], list[str], dict[str, str]]:
    """Order stale references before dependents and report blocked cells.

    Pull each non-resumable reference once, before its first dependent. A selected
    runnable reference is treated as fresh unless its own chain is blocked. Walk
    chains to their roots so shared references have one consistent outcome.

    With ``skip_crashed``, retain deterministic crashes for the same pin instead of
    rerunning them in each dependent's session. Block only dependents that can
    reach a fidelity comparison; non-waivable typed refusals never need their
    reference and can still run. Propagate a crashed reference through its chain.

    Return the runnable order, IDs pulled in, and a map from blocked dependent IDs
    to the crashed reference IDs.
    """
    by_id = {cell["id"]: cell for cell in cells}
    runnable_ids = {cell["id"] for cell in runnable}
    queued: set[str] = set()
    ordered: list[dict] = []
    pulled: list[str] = []
    blocked: dict[str, str] = {}
    memo: dict[str, str | None] = {}

    def blocker_for(cell: dict) -> str | None:
        # A reference cell reads its own reference (a cfg cell scores against
        # its single), so the chain is walked to its root before the cell. The
        # memo is per cell id, so a reference several dependents share, or a
        # runnable cell another selected cell also reads, answers the same way
        # whichever one the loop below reaches first.
        cell_id = cell["id"]
        if cell_id in memo:
            return memo[cell_id]
        memo[cell_id] = None  # provisional: a cycle reads as clear, not a hang
        reference = cell.get("reference") or {}
        target = (by_id.get(reference.get("cell", ""))
                  if reference.get("kind") != "resident-leg" else None)
        if (target is None or target["session"] != cell["session"]
                or target["sampled_out"] or target["label"].startswith("skip:")):
            return None
        if target["id"] not in runnable_ids:
            digest = graph_digest(out, target["template"])
            path = out / "cells" / f"{target['id']}.json"
            done = json.loads(path.read_text()) if path.is_file() else {}
            if resumable(done, pin, digest, target["label"], target["reference"],
                         target.get("capacity_risk_basis")):
                return None
            if skip_crashed and crashed_on_pin(done, pin, digest, target["label"],
                                               target["reference"],
                                               target.get("capacity_risk_basis")):
                if not compares_against_reference(cell):
                    return None
                blocked[cell_id] = target["id"]
                memo[cell_id] = target["id"]
                return target["id"]
        if blocker := blocker_for(target):
            if not compares_against_reference(cell):
                return None
            blocked[cell_id] = blocker
            memo[cell_id] = blocker
            return blocker
        if target["id"] not in runnable_ids and target["id"] not in queued:
            queued.add(target["id"])
            ordered.append(target)
            pulled.append(target["id"])
        return None

    for cell in runnable:
        if blocker_for(cell) is None:
            ordered.append(cell)
    return ordered, pulled, blocked


def wedged_answer(done: dict) -> str:
    """Why the rig wrote this record rather than the cell, or "" for the cell.

    A wedged driver answers every cell in about two seconds with one message,
    so a record written under one is the rig's answer whatever class it was
    scored as, and it re-runs (the krea2 cfg2 records the 2026-09-02 wedge left
    behind). The signals are the runner's own, so a session that ends on one
    and a record that carries one cannot disagree: the two driver messages no
    reset clears, a leg whose queue would not drain, and a replacement the
    driver refused, which leaves the cell answering on the fleet before it.
    """
    for name, leg in (done.get("legs") or {}).items():
        text = f"{leg.get('message', '')}\n{leg.get('exception_type', '')}"
        if hit := next((phrase for phrase in WEDGE_TEXTS if phrase in text), ""):
            return f"the {name} leg answered {hit!r}, which no reset clears"
        if leg.get("drain", "drained") != "drained":
            return f"the {name} leg queue would not clear ({leg['drain']})"
    for name in ("reset", "floor_reset"):
        if (done.get(name) or {}).get("recycle") == RECYCLE_BUSY:
            return f"the {name} recycle answered {RECYCLE_BUSY}, so the old fleet stayed"
    return ""


def settled(done: dict) -> bool:
    """A record worth resuming past: a render or a refusal no wedged driver
    wrote. A timeout or a transport error says the rig was unavailable, not
    that the cell answered, and a crash is not settled either."""
    observed = str(done.get("observed", ""))
    if not (observed == "render" or observed.startswith("refuse:")):
        return False
    return not wedged_answer(done)


RESUME_KEYS = ("comfy_commit", "source_digest", "torch", "nccl", "world")
# What the record was compared against: which cell or leg, at which bar, over
# how many steps. The preset a resident-leg spec names is out, because it is
# cut from the cell's own preset, which is part of the cell id: it cannot move
# under a settled record without moving the record to another file.
REFERENCE_KEYS = ("kind", "cell", "bar", "probe_steps")


def record_current_for(done: dict, pin: dict, digest: str, label: str, reference: dict,
                       capacity_risk_basis: str | None = None) -> bool:
    """Check whether a record describes the current cell and environment.

    Compare ComfyUI, runtime-source digest, torch, NCCL, world, graph digest, label,
    capacity-risk basis, and reference plan. A changed label or capacity rule can
    change the verdict; a changed reference or probe plan changes the comparison.

    Exclude monarch_commit (harness edits can change it), torch_source and
    nccl_source (version provenance), gate_protocol_version (covered by the source
    digest), and driver URL (operator configuration).

    ``resumable`` additionally requires settlement; ``crashed_on_pin`` requires a
    crash. Both use this check to agree on record identity.
    """
    old = done.get("pin") or {}
    if any(old.get(key) != pin.get(key) for key in RESUME_KEYS):
        return False
    cell = done.get("cell") or {}
    if cell.get("label") != label:
        return False
    if cell.get("capacity_risk_basis") != capacity_risk_basis:
        return False
    was = cell.get("reference") or {}
    if any(was.get(key) != reference.get(key) for key in REFERENCE_KEYS):
        return False
    return bool(done.get("graph_digest") == digest and done.get("verdict"))


def resumable(done: dict, pin: dict, digest: str, label: str, reference: dict,
              capacity_risk_basis: str | None = None) -> bool:
    """Whether a written record still answers for this session: it describes
    this exact question (``record_current_for``) and it settled."""
    return record_current_for(done, pin, digest, label, reference,
                              capacity_risk_basis) and settled(done)


def crashed_on_pin(done: dict, pin: dict, digest: str, label: str, reference: dict,
                   capacity_risk_basis: str | None = None) -> bool:
    """Whether a reference's own record is a deterministic crash this exact
    question can still speak for (``--skip-crashed-references``).

    The question is ``record_current_for``'s. A crash recorded against another
    build, an older converted graph, a different label, or a reference plan the
    matrix has since moved says nothing about what this reference would answer
    now, so it reads like any other stale record: pulled in and tried again.
    So do a ``recoverable_crash`` (a dirty mesh an earlier cell left behind,
    not this reference failing) and a ``wedged_answer`` (the rig's answer, not
    the cell's): both would clear on a fresh attempt.
    """
    if done.get("observed") != "crash":
        return False
    if not record_current_for(done, pin, digest, label, reference, capacity_risk_basis):
        return False
    if recoverable_crash(done):
        return False
    return not wedged_answer(done)


def blocked_record(cell: dict, pin: dict, digest: str, crashed_id: str) -> dict:
    """The smallest record report.py's row builder can render for a dependent
    ``--skip-crashed-references`` left without a reference to compare against."""
    return {
        "cell": cell, "pin": pin, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph_digest": digest, "legs": {}, "observed": "blocked", "verdict": "FINDING",
        "notes": [],
        "findings": [f"reference cell {crashed_id} crashed on this pin; "
                    "--skip-crashed-references did not pull it in, so this cell was "
                    "not run"],
    }


def block_crashed_dependents(out: Path, cells: list[dict], pin: dict,
                             blocked: dict[str, str]) -> list[str]:
    """Write a blocked record for every dependent a crashed, unpulled reference
    left without one, unless the dependent's own record already resumes past
    it. Returns the ids actually written."""
    by_id = {cell["id"]: cell for cell in cells}
    written = []
    for cell_id, crashed_id in blocked.items():
        cell = by_id[cell_id]
        digest = graph_digest(out, cell["template"])
        path = out / "cells" / f"{cell_id}.json"
        done = json.loads(path.read_text()) if path.is_file() else {}
        if resumable(done, pin, digest, cell["label"], cell["reference"],
                     cell.get("capacity_risk_basis")):
            print(f"[{cell_id}] resume: already {done['verdict']}", flush=True)
            continue
        if done:
            keep_attempt(out, cell_id)
        write_record(out, cell_id, blocked_record(cell, pin, digest, crashed_id))
        written.append(cell_id)
        print(f"[{cell_id}] blocked: reference {crashed_id} crashed on this pin and "
              "was not pulled in (--skip-crashed-references)", flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.sweep.run",
        description="Run one session of sweep cells against a live driver.")
    parser.add_argument("--config", required=True, help="sweep TOML (see sweep.example.toml)")
    parser.add_argument("--session", required=True, choices=("local", "cluster"),
                        help="one transport per driver process; run the two sessions apart")
    parser.add_argument("--cells", default="all",
                        help="comma-separated cell ids, labels or template names; a label "
                             "token skips that label's waivable cells (see --include-waivers)")
    parser.add_argument("--include-waivers", action="store_true",
                        help="keep the waivable cells a label token would otherwise leave out")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N selected cells; without it, every selected image "
                             "cell runs and only the first video cell")
    parser.add_argument("--cooldown-s", type=float, default=None,
                        help="seconds to pause between cells (default: cooldown_s in the config)")
    parser.add_argument("--timeout-s", type=float, default=0.0,
                        help="seconds each leg may run (default: image_timeout_s or video_timeout_s in the config)")
    parser.add_argument("--skip-crashed-references", action="store_true",
                        help="do not pull in a reference cell whose current record on this pin is "
                             "a deterministic crash; record each dependent that would compare "
                             "against it, directly or down a reference chain, as blocked instead "
                             "(for a caller that isolates OOM-prone cells one per session)")
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    out = config["out_dir"]
    cells = [json.loads(line) for line in (out / "cells.jsonl").read_text().splitlines() if line]
    runnable, pruned = select(cells, args)
    pin, client_id = session_pin(config), str(uuid.uuid4())
    runnable, pruned["references_pulled"], blocked = with_stale_references(
        runnable, cells, out, pin, skip_crashed=args.skip_crashed_references)
    if blocked:
        pruned["blocked_on_crashed_reference"] = block_crashed_dependents(out, cells, pin, blocked)
    (out / f"run_pruned_{args.session}.json").write_text(json.dumps(pruned, indent=2))
    cooldown = config["cooldown_s"] if args.cooldown_s is None else args.cooldown_s
    watch = SessionWatch(config["driver"], config["attach_pace_s"])
    summary: dict = {"session": args.session, "pin": pin, "runnable": len(runnable),
                     "ran": 0, "wedged": "", "rejected_templates": [],
                     "blocked": len(pruned.get("blocked_on_crashed_reference") or [])}
    print(f"{len(runnable)} runnable cells in the {args.session} session; pin {pin}")
    if any(cell.get("audio") for cell in runnable) and not audio_compare_available():
        print("WARNING: soundfile is not installed, so no audio leg can be compared and an audio cell "
              "that reaches a reference compare scores CHECK at best; install the dev extra first", flush=True)
    print("pruned: " + ", ".join(f"{len(ids)} {reason}" for reason, ids in pruned.items())
          + f"; names in {out / f'run_pruned_{args.session}.json'}")
    # An earlier session may have left the fleet latched on a comfy_managed
    # value, and that latch outlives every graph.
    if runnable:
        try:
            answer = recycle(config["driver"])
            watch.recycled(answer)
            print(f"session open recycle: {answer}", flush=True)
            print("transport probe: "
                  f"{probe_transport(config, args.session, client_id, watch)}", flush=True)
        except SessionWedged as exc:
            print(f"SessionWedged: {exc}", flush=True)
            summary["wedged"] = str(exc)
            print(f"wedged; summary in {write_summary(out, args.session, summary)}")
            return 1
    last: LastCell | None = None
    reset: dict | None = None
    rejected: dict[str, list[str]] = {}
    unscoped: list[float] = []
    for cell in runnable:
        if cell["template"] in rejected:
            rejected[cell["template"]].append(cell["id"])
            print(f"[{cell['id']}] skipped: {cell['template']} was rejected", flush=True)
            continue
        existing = out / "cells" / f"{cell['id']}.json"
        done = json.loads(existing.read_text()) if existing.exists() else {}
        digest = graph_digest(out, cell["template"])
        if resumable(done, pin, digest, cell["label"], cell["reference"],
                     cell.get("capacity_risk_basis")):
            print(f"[{cell['id']}] resume: already {done['verdict']}", flush=True)
            continue
        if done:
            # Keep the unsettled attempt: an infra failure is evidence too.
            if why := wedged_answer(done):
                print(f"[{cell['id']}] re-running: {why}", flush=True)
            keep_attempt(out, cell["id"])
        managed = str(cell["levers"].get("comfy_managed", "off"))
        reason = needs_fleet_reset(last, cell, config["mem_floor_gib"])
        try:
            if reason and last is not None:
                reset = reset_fleet(config, client_id, last.managed, cell["mode"], watch)
                reset["reason"] = reason
                print(f"reset ({reason}): recycle {reset['recycle']}", flush=True)
            elif last is not None:
                # The fleet the previous cell rendered on runs this one too, so
                # its cold leg is a first render on a fleet that already holds
                # the model, not a first load. The stamp is how a reader tells.
                reset = {"fleet_kept": True, "available_gib": last.available_gib,
                         "need_gib": fleet_need_gib(cell, config["mem_floor_gib"])}
                print(f"[{cell['id']}] fleet kept: {last.available_gib} GiB available, "
                      f"{reset['need_gib']} GiB needed", flush=True)
            record = run_cell(cell, config, pin, client_id, args, reset, watch)
            reset = None
            summary["ran"] += 1
            print(f"[{cell['id']}] {cell['template']} {cell['preset']} -> {record['verdict']}",
                  flush=True)
            if recoverable_crash(record):
                # The dirty mesh belongs to the cell before this one, so the
                # crash is charged to the wrong cell. Replace the fleet and
                # score the re-run; one retry, so a rig a fresh fleet cannot fix
                # still ends.
                retry = reset_fleet(config, client_id, managed, cell["mode"], watch)
                retry["reason"] = "retry after a recoverable crash"
                if retry["recycle"] != "ok":
                    # The old fleet is still there, so the re-run would meet the
                    # same dirty mesh and spend the cell's one retry on it.
                    record["findings"].append(
                        f"the retry was skipped: the recycle answered {retry['recycle']}")
                    settle_verdict(record)
                    write_record(out, cell["id"], record)
                    print(f"[{cell['id']}] retry skipped: recycle {retry['recycle']}",
                          flush=True)
                else:
                    kept = keep_attempt(out, cell["id"])
                    print(f"[{cell['id']}] retrying on a fresh fleet; the crash is kept as "
                          f"{kept.name}", flush=True)
                    record = run_cell(cell, config, pin, client_id, args, retry, watch)
                    print(f"[{cell['id']}] retry -> {record['verdict']}", flush=True)
        except SessionWedged as exc:
            print(f"[{cell['id']}] {type(exc).__name__}: {exc}", flush=True)
            summary["wedged"] = f"[{cell['id']}] {exc}"
            break
        last = LastCell(cell["template"], managed, record["observed"],
                        load_evidence(record), final_available_gib(record))
        if record["observed"] == "graph-rejected":
            rejected.setdefault(cell["template"], [])
            scope = rejection_scope(record.get("rejection", ""))
            now = time.time()
            unscoped = [seen for seen in unscoped if now - seen <= REJECT_WINDOW_S]
            if scope != "template":
                unscoped.append(now)
            if scope == "session":
                print("the rejection names no node of this template's own, so no graph "
                      "would run; stopping the session", flush=True)
                break
            if len(unscoped) >= REJECT_RUN:
                print(f"{len(unscoped)} rejections inside {int(REJECT_WINDOW_S)} s that name "
                      "no node the harness can read; stopping the session", flush=True)
                break
            print(f"[{cell['id']}] {cell['template']} rejected; its remaining cells are "
                  "skipped and the session continues", flush=True)
        time.sleep(cooldown)
    # Release the mesh before the operator stops the driver: a driver killed
    # with a live mesh strands its actors and the next session's first render
    # spins in a collective until the stall guard evicts it.
    try:
        print(f"session close recycle: {recycle(config['driver'])}", flush=True)
    except (SessionWedged, OSError) as exc:
        print(f"session close recycle failed: {exc}", flush=True)
    if rejected:
        path = out / f"run_rejected_{args.session}.json"
        path.write_text(json.dumps(rejected, indent=2))
        print(f"{len(rejected)} template(s) rejected, "
              f"{sum(len(ids) for ids in rejected.values())} cells skipped; names in {path}")
    summary["rejected_templates"] = sorted(rejected)
    print(f"session summary in {write_summary(out, args.session, summary)}")
    return 1 if (summary["wedged"] or rejected) else 0


if __name__ == "__main__":
    raise SystemExit(main())
