"""CPU-only observer microbenchmark; no ComfyUI, Torch, driver or Worker required.

Write results to a new JSON file in an owner-private directory outside this
repository, then cite the file by the path and SHA-256 the harness prints. The
context fixture below is stdlib-only and explicitly synthetic. Timings are not
generation overhead or hardware acceptance; tracemalloc measures retained Python
allocations, not RSS.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.abc
import json
import os
import platform
import stat
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dgx_monarch import execution_events as events  # noqa: E402
from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def private_output(requested: Path) -> Path:
    """A new .json path whose directory only this user can read, outside the repo.

    Evidence under the repository would be swept into commits and archives, and
    evidence in a shared or world-writable directory can be read, replaced or
    removed by other accounts, so both are refused before any measurement.
    """
    output = requested.absolute()
    require(output.suffix == ".json", "Output must be a .json file")
    parent = output.parent.resolve()
    require(not parent.is_relative_to(ROOT.resolve()), "Output must be outside the repository")
    require(parent.is_dir(), f"Output directory does not exist: {parent}")
    info = parent.stat()
    require(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) & 0o077 == 0,
            f"Output directory must belong to this user with no group or other access (mode 0700): {parent}")
    require(not output.exists() and not output.is_symlink(), f"Output already exists: {output}")
    return parent / output.name


def write_record(output: Path, result: dict[str, Any]) -> str:
    """Write the record once, flush it to disk and return its SHA-256."""
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def worktree_identity() -> dict[str, Any]:
    """Hash current tracked and nonignored untracked bytes, never Git HEAD."""
    output = subprocess.run(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=ROOT,
                            check=True, capture_output=True, timeout=30).stdout
    manifest = hashlib.sha256(b"DGXM_OBSERVER_BENCH_WORKTREE_V1\0")
    names = sorted({name.decode() for name in output.split(b"\0") if name})
    total = 0
    for name in names:
        path = ROOT / name
        if not path.exists() and not path.is_symlink():
            manifest.update(name.encode() + b"\0deleted\n")
            continue
        content = hashlib.sha256()
        if path.is_symlink():
            raw = os.readlink(path).encode()
            content.update(raw)
            total += len(raw)
        else:
            require(path.is_file(), f"Unsupported source entry: {name}")
            before = path.stat()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content.update(chunk)
                    total += len(chunk)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f"Source changed while hashing: {name}")
        manifest.update(name.encode() + b"\0" + content.hexdigest().encode() + b"\n")
    return {"sha256": manifest.hexdigest(), "files": len(names), "bytes": total,
            "method": "working-tree tracked and nonignored untracked file bytes; symlink target text; deleted marker"}


def distribution(values: list[float]) -> dict[str, Any]:
    return {"median_ns_per_call": statistics.median(values), "min_ns_per_call": min(values),
            "max_ns_per_call": max(values), "stdev_ns_per_call": statistics.stdev(values), "samples": values}


def measure(callback: Callable[[], Any], iterations: int, repeats: int,
            before: Callable[[], Any] | None = None) -> dict[str, Any]:
    wall, cpu = [], []
    if before:
        before()
    for _ in range(min(iterations, 1000)):
        callback()
    for _ in range(repeats):
        if before:
            before()
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()
            for _ in range(iterations):
                callback()
            cpu.append((time.process_time_ns() - start_cpu) / iterations)
            wall.append((time.perf_counter_ns() - start_wall) / iterations)
        finally:
            if enabled:
                gc.enable()
    return {"iterations_per_repeat": iterations, "repeats": repeats,
            "wall": distribution(wall), "process_cpu": distribution(cpu)}


class BlockComfy(importlib.abc.MetaPathFinder):
    """Prevent accidental loading of real ComfyUI on any developer machine."""
    attempts = 0

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == "comfy_execution" or fullname.startswith("comfy_execution."):
            self.attempts += 1
            raise ModuleNotFoundError("ComfyUI intentionally unavailable in CPU observer benchmark")
        return None


def timings(iterations: int, repeats: int) -> dict[str, Any]:
    require(not any(name == "comfy_execution" or name.startswith("comfy_execution.") for name in sys.modules),
            "Run in a fresh Python process without ComfyUI")
    original = events.journal
    blocker = BlockComfy()
    sys.meta_path.insert(0, blocker)
    scope = events.Scope("synthetic-prompt", "synthetic-node", 0, "synthetic-render")
    facts = {"outcome": "validated_output", "fixture": True}
    result = {}
    try:
        disabled = events.Journal()
        events.journal = disabled
        result["capture_disabled"] = measure(lambda: events.capture("synthetic-render"), iterations, repeats)
        result["publish_disabled"] = measure(lambda: disabled.publish(scope, "result_validated", facts), iterations, repeats)
        require(blocker.attempts == 0 and disabled._sequence == 0, "Disabled observer performed capture/publication")
        active = events.Journal()
        events.journal = active
        renew = lambda: active.read(active._sequence, active.producer_id)  # noqa: E731
        result["capture_enabled_context_unavailable"] = measure(
            lambda: events.capture("synthetic-render"), iterations, repeats, renew)
        require(blocker.attempts > 0 and active._sequence == 0, "Missing Comfy context was not refused")
        context = SimpleNamespace(prompt_id="synthetic-prompt", node_id="synthetic-node", list_index=0)
        fixture = ModuleType("comfy_execution.utils")
        fixture.get_executing_context = lambda: context
        sys.modules["comfy_execution.utils"] = fixture
        result["capture_enabled_synthetic_context"] = measure(
            lambda: events.capture("synthetic-render"), iterations, repeats, renew)
        require(events.capture("synthetic-render") == scope, "Explicit fixture identity was lost")
        result["publish_enabled"] = measure(
            lambda: active.publish(scope, "result_validated", facts), iterations, repeats, renew)
        require(active.active() and active._sequence > 0, "Observer lease expired while measuring")
        result["replay_first_page"] = measure(lambda: active.read(0), min(iterations, 1000), repeats)
        result["capture_import_attempts"] = blocker.attempts
        result["context_fixture"] = "stdlib synthetic get_executing_context; no ComfyUI code imported"
    finally:
        events.journal = original
        sys.modules.pop("comfy_execution.utils", None)
        sys.meta_path.remove(blocker)
    require(not any(name == "torch" or name.startswith("torch.") or name.startswith("comfy") for name in sys.modules),
            "Benchmark unexpectedly imported Torch or ComfyUI")
    return result


def retained(case: str, payload_size: int, wave: int) -> dict[str, Any]:
    journal = events.Journal()
    scope = events.Scope("synthetic-prompt", "synthetic-node", 0, "synthetic-render")
    journal.read()
    facts = {"fixture_payload": "x" * payload_size}
    gc.collect()
    tracemalloc.start()
    try:
        samples = []
        for _ in range(2):
            journal.read(journal._sequence)
            for _ in range(wave):
                journal.publish(scope, "worker_results", facts)
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            require(journal.active(), "Observer lease expired during retention measurement")
            require(len(journal._events) <= events.MAX_EVENTS and journal._bytes <= events.MAX_BYTES,
                    "Journal exceeded its event/payload retention bound")
            samples.append({"published": journal._sequence, "retained_events": len(journal._events),
                            "encoded_payload_bytes": journal._bytes, "traced_current_bytes": current,
                            "traced_peak_bytes": peak})
        page = journal.read(0)
        require(page["gap"] and page["oldest_sequence"] > 1, "Overflow did not expose a replay gap")
        require(abs(samples[0]["retained_events"] - samples[1]["retained_events"]) <= 2, "Retained event count did not plateau")
        require(samples[1]["traced_current_bytes"] - samples[0]["traced_current_bytes"] < 1024 * 1024,
                "Second overflow wave retained more than 1 MiB of additional Python allocations")
        if case == "count":
            require(len(journal._events) == events.MAX_EVENTS, "Count-bound case did not hit the count cap")
        else:
            require(len(journal._events) < events.MAX_EVENTS and journal._bytes > events.MAX_BYTES - 20000,
                    "Byte-bound case did not hit the payload cap")
        sequence = journal._sequence
        journal.publish(scope, "worker_results", {"oversize": "x" * events.MAX_EVENT_BYTES})
        require(journal._omitted == 1 and journal._sequence == sequence, "Oversized event omission was not counted")
        return {"case": case, "fixture_payload_bytes": payload_size, "events_per_wave": wave,
                "waves": samples, "oldest_sequence": page["oldest_sequence"], "gap": page["gap"],
                "omitted_events": journal._omitted, "assertions": "passed"}
    finally:
        tracemalloc.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new .json file in an owner-private (0700) directory outside the repository")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    require(100 <= args.iterations <= 100000 and 3 <= args.repeats <= 15, "Use 100..100000 iterations and 3..15 repeats")
    try:
        output = private_output(args.output)
    except RuntimeError as exc:
        parser.error(str(exc))
    tree_start = worktree_identity()
    package_start = dgx_source_manifest_sha256()
    result = {"schema_version": 1, "scope": "CPU-only synthetic observer microbenchmark, not Spark/native/generation acceptance",
              "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "host_load_before": os.getloadavg() if hasattr(os, "getloadavg") else None,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_tree_before": tree_start, "package_source_manifest_before": package_start,
              "timings": timings(args.iterations, args.repeats),
              "retention": [retained("count", 32, events.MAX_EVENTS * 2), retained("bytes", 8192, 2048)]}
    result["source_tree_after"] = worktree_identity()
    result["package_source_manifest_after"] = dgx_source_manifest_sha256()
    result["source_tree_changed_during_run"] = tree_start != result["source_tree_after"]
    require(package_start == result["package_source_manifest_after"], "Runtime package changed during benchmark")
    result["assertions"] = "passed"
    digest = write_record(output, result)
    print(json.dumps({"output": str(output), "sha256": digest}))


if __name__ == "__main__":
    main()
