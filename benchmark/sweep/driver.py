"""Communicate with a live ComfyUI driver and classify its responses.

Refusals usually include a class tag. For untagged refusals, read identifying
prefixes from runtime raise sites and supplement them with UNTAGGED_LITERALS
for expressions the AST walk cannot resolve.
"""
from __future__ import annotations

import ast
import json
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from .matrix import INIT_CLASS, REPO, refusal_label

# Untagged refusals that ride a bare ValueError or RuntimeError, or format
# their text from a table, so the raise-site walk cannot read them. Each names
# its site.
UNTAGGED_LITERALS = (
    "explicit topology preset ",  # nodes/render_validation.py _preset_label
    "resolved topology for preset ",  # the same guard under an auto preset
    "cfg-parallel got a model call of batch=",  # adapters/cfg_parallel.py
    "cross-rank latent identity FAILED:",  # render_validation.verify_cross_rank_signatures
    "automatic FSDP ",  # nodes/auto_gate.py _FSDP_DENIAL_REASONS, gate_identity
    "comfy_managed=on ",  # nodes/init.py, both residency conflicts
    # nodes/gate_ceremony.py and nodes/gate_session.py: the FSDP proof abort and
    # the incomplete-proof stop both raise through a runtime subscript or a
    # rebound name, so the AST walk below cannot see either literal.
    "FSDP clean-reload proof ",
)
# A worker refusal reaches /history inside monarch's ActorError, which prints
# the remote type name ahead of the clause every guard opens with.
FRAMING = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception): ")
UNTAGGED_TYPES = ("UnsupportedModelError", "StockLoadCapacityError", "FsdpGateProofError",
                  "PackedCfgParallelError", "PackedDataParallelError")
MIN_PREFIX = 8  # a shorter literal is not a clause and could match a crash's text


class GraphRejected(RuntimeError):
    """The driver would not take the graph. Every cell on it fails the same."""


class SessionWedged(RuntimeError):
    """A timed-out prompt would not clear, so nothing after it is measurable."""


def post(url: str, body: dict, headers: dict | None = None, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def get(url: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def action_headers(driver: str, action: str) -> dict:
    return {"X-DGXM-Action": action, "Content-Type": "application/json", "Origin": driver}


@lru_cache(maxsize=1)
def untagged_prefixes(repo: Path = REPO) -> tuple[str, ...]:
    """Every untagged refusal opening: UNTAGGED_LITERALS plus the runtime's raise sites.

    A hand-copied list alone misses every guard added after it, so this also
    walks the raises of the refusal-shaped exception types: a raise whose
    argument is not ``refusal(...)`` carries no class tag, and its leading
    literal is the only discriminator its message gives.
    """
    found = set(UNTAGGED_LITERALS)
    for path in sorted((repo / "src" / "dgx_monarch").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            name = getattr(node.exc.func, "id", getattr(node.exc.func, "attr", ""))
            arg = node.exc.args[0] if node.exc.args else None
            if name not in UNTAGGED_TYPES or (
                    isinstance(arg, ast.Call) and getattr(arg.func, "id", "") == "refusal"):
                continue
            if isinstance(arg, ast.JoinedStr):
                arg = arg.values[0] if arg.values else None
            if isinstance(arg, ast.Constant) and len(str(arg.value)) >= MIN_PREFIX:
                found.add(str(arg.value))
    return tuple(sorted(found))


def error_text(entry: dict, key: str = "exception_message") -> str:
    """One field of every status message, joined. The class rides the message;
    the exception type is secondary evidence and keeps its own field."""
    payloads = [message[1] if isinstance(message, list) and len(message) > 1 else message
                for message in ((entry or {}).get("status") or {}).get("messages") or []]
    return "\n".join(str(payload[key]) for payload in payloads
                     if isinstance(payload, dict) and payload.get(key))


def cached_nodes(entry: dict) -> list[str]:
    """Node ids ComfyUI served out of its execution cache for this prompt.

    The history carries one ``execution_cached`` message per prompt naming the
    nodes it did not run; a warm leg whose seeded node is among them measured
    the cache, not a render.
    """
    found: list[str] = []
    for message in ((entry or {}).get("status") or {}).get("messages") or []:
        if not (isinstance(message, list) and len(message) > 1 and message[0] == "execution_cached"):
            continue
        payload = message[1]
        nodes = payload.get("nodes") if isinstance(payload, dict) else None
        found.extend(str(node) for node in (nodes or []))
    return sorted(set(found))


def waiver_run_ids(entry: dict) -> list[str]:
    """Class-K dispatch ids published by sampler UI for this prompt only.

    The gate ledger's ``run_id`` names a render dispatch, while the sweep only
    receives Comfy's prompt id.  Sampler UI is retained in that exact prompt's
    history, making this a narrow ledger-to-render join.  Invalid UI data is
    ignored rather than becoming provenance.
    """
    found: set[str] = set()
    outputs = (entry or {}).get("outputs") or {}
    if not isinstance(outputs, dict):
        return []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        runs = output.get("dgxm_waiver_runs")
        if not isinstance(runs, list):
            continue
        for run_id in runs:
            if isinstance(run_id, str) and run_id:
                found.add(run_id)
    return sorted(found)


def _clauses(text: str) -> list[str]:
    """Every message line, and the same line with its remote framing removed."""
    out = []
    for raw in text.splitlines():
        line = raw.lstrip()
        out.append(line)
        if (bare := FRAMING.sub("", line, count=1)) != line:
            out.append(bare)
    return out


def classify_history(entry: dict | None, timed_out: bool = False) -> tuple[str, str]:
    """Outcome and message for one /history entry."""
    if timed_out:
        return "timeout", ""
    status = (entry or {}).get("status") or {}
    if status.get("status_str") == "error":
        text = error_text(entry or {})
        label = refusal_label(text)
        if label:
            return label, text
        prefixes = untagged_prefixes()
        if any(clause.startswith(prefix)
               for clause in _clauses(text) for prefix in prefixes):
            return "refuse:untyped", text
        return "crash", text
    if status.get("completed") is True:
        return "render", ""
    return "incomplete", json.dumps(status)[:400]


def clear_graph(driver: str, mode: str, comfy_managed: str = "off") -> dict:
    """A ClearVRAM hard graph built from the Init node's own defaults.

    It carries the outgoing cell's comfy_managed: the worker latches that
    setting for its process lifetime, so an Init that contradicts the live
    fleet would refuse instead of clearing.
    """
    inputs = {"topology": "auto"}
    try:
        info = get(f"{driver}/object_info/{INIT_CLASS}")[INIT_CLASS]
        for name, spec in (info["input"].get("required") or {}).items():
            if len(spec) > 1 and isinstance(spec[1], dict) and spec[1].get("default") is not None:
                inputs[name] = spec[1]["default"]
    except (OSError, KeyError, ValueError):
        pass
    inputs.update({"mode": mode, "comfy_managed": comfy_managed})
    return {"1": {"class_type": INIT_CLASS, "inputs": inputs},
            "2": {"class_type": "DGXMonarchClearVRAM",
                  "inputs": {"mesh": ["1", 0], "level": "hard", "include_driver": True}}}


def recycle(driver: str, attempts: int = 3) -> str:
    """Replace the worker fleet. The 429 is retried: a recycle that never runs
    leaves the old process and its latched bootstrap, so a rate limit that
    outlasts every retry ends the session instead of letting the next cell run
    on the old fleet."""
    for attempt in range(attempts):
        try:
            post(f"{driver}/dgxm/recycle", {}, action_headers(driver, "recycle"), timeout=300)
            return "ok"
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                return f"http {exc.code}"
            if attempt + 1 < attempts:
                time.sleep(float(exc.headers.get("Retry-After") or 2))
        except OSError as exc:
            return f"{type(exc).__name__}: {exc}"
    raise SessionWedged(f"the recycle stayed rate limited over {attempts} attempts, so the "
                        "next cell would run on the old fleet and its latched bootstrap")


def grant_pending_consent(driver: str, cell: dict) -> dict:
    """Accept the card this cell's guard raised, by its key and its pending id.

    A row names its guard in ``kind``; ``key`` is an opaque memo hash that
    matches nothing. A stale card from an earlier waiver must not be granted in
    its place, so an unmatched or ambiguous list is a finding rather than a
    guess, and a rate-limited POST is retried once after Retry-After.
    """
    from dgx_monarch.refusal import GUARDS

    try:
        pending = get(f"{driver}/dgxm/consents").get("pending") or []
    except (OSError, ValueError) as exc:
        return {"granted": False, "detail": f"{type(exc).__name__}: {exc}"}
    kinds = {GUARDS[guard].consent_kind for guard in cell["waiver_guards"] if guard in GUARDS}
    cards = [row for row in pending if str(row.get("kind", "")) in kinds]
    if len(cards) != 1:
        return {"granted": False, "pending": len(pending),
                "detail": f"{len(cards)} of {len(pending)} pending cards carry "
                          f"{sorted(kinds)}; the grant needs exactly one"}
    body = {"action": "accept", "key": cards[0].get("key"), "id": cards[0].get("id")}
    for attempt in (0, 1):
        try:
            return {"granted": True, "key": body["key"],
                    "response": post(f"{driver}/dgxm/consent", body,
                                     action_headers(driver, "consent"), timeout=60)}
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                time.sleep(float(exc.headers.get("Retry-After") or 2))
                continue
            return {"granted": False, "detail": f"http {exc.code}: {exc.reason}"}
        except (OSError, ValueError) as exc:
            return {"granted": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"granted": False, "detail": "the consent POST was still rate limited after one retry"}


def queue_and_wait(driver: str, graph: dict, client_id: str,
                   timeout_s: float) -> tuple[float, dict | None, str, str]:
    """(wall, entry, state, prompt_id). state is empty when the prompt settled,
    "timeout", or a poll transport error. Under either the prompt may still be
    running, so the caller drains it. A submit whose response is lost is
    different: without a prompt id the runner cannot safely drain it, so it
    wedges the session rather than let the next cell inherit an unknown queue.

    ``timeout_s`` is one absolute deadline for this prompt, and run_cell passes
    the same value to each leg. Every submit and history request receives only
    the time still available, so the leg timeout (--timeout-s, or the config's
    image_timeout_s or video_timeout_s) also sets the HTTP deadline, not the
    120 s and 60 s defaults of post and get.
    """
    started = time.perf_counter()
    deadline = started + timeout_s

    def remaining() -> float:
        return deadline - time.perf_counter()

    submit_timeout = remaining()
    if submit_timeout <= 0:
        return 0.0, None, "timeout", ""
    try:
        response = post(f"{driver}/prompt", {"prompt": graph, "client_id": client_id},
                        timeout=submit_timeout)
        if not isinstance(response, dict):
            raise ValueError("the driver returned a non-object prompt response")
        prompt_id = response["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError("the driver returned no usable prompt_id")
    except urllib.error.HTTPError:
        # An HTTP response is a definite driver answer; retain run_leg's 400
        # graph-rejection handling and its visible status detail.
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SessionWedged(
            "the prompt submission outcome is unknown "
            f"({type(exc).__name__}: {exc}); refusing to queue another cell") from exc
    while True:
        poll_timeout = remaining()
        if poll_timeout <= 0:
            return time.perf_counter() - started, None, "timeout", prompt_id
        try:
            entry = get(f"{driver}/history/{prompt_id}", timeout=poll_timeout).get(prompt_id)
        except (OSError, ValueError) as exc:
            return (time.perf_counter() - started, None,
                    f"poll transport {type(exc).__name__}: {exc}", prompt_id)
        status = (entry or {}).get("status") or {}
        if status.get("status_str") == "error" or status.get("completed") is True:
            return time.perf_counter() - started, entry, "", prompt_id
        if remaining() <= 0:
            return time.perf_counter() - started, entry, "timeout", prompt_id
        time.sleep(max(0.0, min(2.0, remaining())))


def idle(driver: str, bounded_s: float) -> bool:
    deadline = time.time() + bounded_s
    while time.time() < deadline:
        queue = get(f"{driver}/queue")
        if not (queue.get("queue_running") or queue.get("queue_pending")):
            return True
        time.sleep(5)
    return False


def drain(driver: str, prompt_id: str, bounded_s: float = 600.0) -> str:
    """Stop a timed-out prompt and empty the queue behind it.

    Without this the prompt keeps running, every later cell queues behind it,
    and one timeout records thousands of phantom ones.
    """
    try:
        post(f"{driver}/interrupt", {"prompt_id": prompt_id}, timeout=60)
        post(f"{driver}/queue", {"clear": True}, timeout=60)
        return "drained" if idle(driver, bounded_s) else "still busy"
    except (OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"


def free_cache(driver: str, bounded_s: float = 120.0) -> str:
    """Reset the execution cache, which also unloads every model.

    comfy reads the unload flag as ``flags.get("unload_models", free_memory)``
    (ComfyUI main.py), so the reset always unloads too, and a false
    unload_models cannot turn that off because server.py sets that flag only
    when it is true. The next leg therefore loads cold, so the runner never
    calls this before the warm leg, whose moved seed misses the cache instead.
    The flag is read at the end of a loop turn, hence the wait.
    """
    try:
        post(f"{driver}/free", {"free_memory": True}, timeout=60)
        if not idle(driver, bounded_s):
            return "still busy"
        time.sleep(2)  # one worker turn past idle
        return "idle"
    except (OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
