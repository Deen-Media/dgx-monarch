"""Headless custom-node discovery for Comfy worker processes."""
from __future__ import annotations

import importlib.util
import os
import sys

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call
from .server_stub import ensure_prompt_server_stub

log = get_logger(__name__)

# Read-only after import: default exclusions reused for every worker preload.
_SKIP_NODE_PACKS = {
    "ComfyUI-Manager",
    "comfyui-manager",
    "dgx-monarch",
}


def preload_failure_hint(exc: BaseException) -> str:
    """Name the likely fix for a worker-side custom-node import failure."""
    seen: set[int] = set()
    cause: BaseException | None = exc
    stub_gap = False
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, ImportError):
            return (
                "a python dependency is missing in the worker env; install the "
                f"pack's requirements into this python ({sys.executable})"
            )
        if isinstance(cause, AttributeError):
            try:
                stub_gap = stub_gap or "PromptServer" in str(cause)
            except BaseException:
                pass
        try:
            linked = cause.__cause__
            cause = linked if linked is not None else cause.__context__
        except BaseException:
            break
    if stub_gap:
        return (
            "the pack uses PromptServer surface the worker's headless stub does not "
            "implement; report it or extend _HeadlessPromptServer (actor/server_stub.py)"
        )
    return "add it to DGXM_SKIP_NODE_PACKS so the worker skips it"


def load_custom_node_modules() -> None:
    """Import host custom-node packs under ComfyUI's exact module keys."""
    if os.environ.get("DGXM_NO_CUSTOM_NODES"):
        safe_call(
            log.info,
            "custom node preload disabled by DGXM_NO_CUSTOM_NODES",
        )
        return
    try:
        import folder_paths

        bases = folder_paths.get_folder_paths("custom_nodes")
    except Exception as exc:
        safe_call(
            log.warning,
            "custom node preload skipped (%s)",
            failure_summary(exc),
        )
        return

    ensure_prompt_server_stub()
    skip = _SKIP_NODE_PACKS | {
        value.strip()
        for value in os.environ.get("DGXM_SKIP_NODE_PACKS", "").split(",")
        if value.strip()
    }
    eligible = 0
    already_loaded = 0
    failed = 0
    loaded = 0
    for base in bases:
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            if (
                entry.startswith((".", "__"))
                or entry.endswith(".disabled")
                or entry in skip
            ):
                continue
            module_path = os.path.join(base, entry)
            if os.path.isfile(module_path) and module_path.endswith(".py"):
                sys_name = os.path.splitext(module_path)[0]
                init_file = module_path
            elif os.path.isdir(module_path) and os.path.isfile(
                os.path.join(module_path, "__init__.py")
            ):
                sys_name = module_path.replace(".", "_x_")
                init_file = os.path.join(module_path, "__init__.py")
            else:
                continue
            eligible += 1
            if sys_name in sys.modules:
                already_loaded += 1
                continue
            try:
                spec = importlib.util.spec_from_file_location(sys_name, init_file)
                if spec is None or spec.loader is None:
                    raise ImportError(
                        f"could not construct an import spec for custom node {init_file}"
                    )
                module = importlib.util.module_from_spec(spec)
                sys.modules[sys_name] = module
                spec.loader.exec_module(module)
                loaded += 1
            except BaseException as exc:
                sys.modules.pop(sys_name, None)
                if not isinstance(exc, Exception):
                    raise
                failed += 1
                try:
                    log.warning(
                        "custom node pack %s failed to import in the worker (%s); "
                        "its samplers/guiders will not deserialize here and its "
                        "sampler/scheduler names will not resolve "
                        "(docs/TROUBLESHOOTING.md #11); %s",
                        entry,
                        failure_summary(exc),
                        preload_failure_hint(exc),
                    )
                except BaseException:
                    pass
    if loaded:
        safe_call(
            log.info,
            "preloaded %d custom node pack(s) for by-reference deserialization",
            loaded,
        )
    elif eligible == 0:
        safe_call(
            log.info,
            "custom node preload found no eligible packs; custom-node "
            "directories were empty or every entry was skipped/disabled",
        )
    else:
        safe_call(
            log.info,
            "preloaded 0 custom node pack(s) for by-reference deserialization "
            "(%d already loaded, %d failed)",
            already_loaded,
            failed,
        )
