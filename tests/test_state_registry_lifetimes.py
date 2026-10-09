"""Keep related registries consistent across their lifetimes.

A fact stored twice can diverge when the stores expire differently. Regression
cases cover a ceremony flag inherited by an unrelated pipelined render, and a
capped no-material memo whose uncapped denial map outlived it.

The source scan requires an explicit lifetime for every registry in the audited
files. Paired-store tests check shared key spaces and drain points. Calibration
cases ensure those checks detect both regressions.
"""
from __future__ import annotations

import ast
import io
import threading
import tokenize
from pathlib import Path
from types import SimpleNamespace

import pytest

import dgx_monarch.nodes.common as common
from dgx_monarch import first_render, telemetry
from dgx_monarch.actor import slab_lifetime
from dgx_monarch.nodes import (
    consent_projection,
    gate_inconclusive,
    gate_process_state,
    loader_graph,
    loader_preflight,
    recycle_drain,
    render_preflight,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"

# The two packages that own render-time decisions, plus the render tracker
# (telemetry.py) and the first-render notices (first_render.py) outside them.
AUDITED = (
    *sorted((SRC / "nodes").rglob("*.py")),
    *sorted((SRC / "actor").rglob("*.py")),
    SRC / "telemetry.py",
    SRC / "first_render.py",
)

# Each call builds a store that is keyed and read later. Locks and conditions
# hold no keys, so they carry no lifetime of their own and are not scanned.
MUTABLE_CALLS = frozenset({
    "dict", "set", "list", "deque", "defaultdict", "OrderedDict", "Counter",
    "local", "WeakValueDictionary", "WeakKeyDictionary",
})

# The stated lifetime comes from a closed vocabulary, so the scan checks a fact,
# not prose. A registry that fits none of these has no decided point at which it
# empties, which is how this class of defect begins.
LIFETIME_PHRASES = (
    "proc lifetime",
    "process lifetime",
    "session lifetime",
    "render lifetime",
    "ceremony lifetime",
    "handle lifetime",
    "instance lifetime",
    "read-only after import",
    "outlives the proc",
)


def _comment_lines(source: str) -> dict[int, tuple[str, bool]]:
    """Every comment, with whether it owns its line.

    A trailing comment belongs to the code on its own line and to nothing
    under it. The block walk below has to know the difference, or a registry
    declared directly beneath a lifetime-carrying line inherits a lifetime it
    never stated, and the scan misses the likeliest place for a new store.
    """
    comments: dict[int, tuple[str, bool]] = {}
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            own_line = not token.line[:token.start[1]].strip()
            comments[token.start[0]] = (token.string, own_line)
    return comments


def _module_registries(path: Path) -> list[tuple[str, int, int]]:
    """Module-level stores: mutable bindings, plus every scalar a `global` rebinds."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    rebound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            rebound.update(node.names)

    found: list[tuple[str, int, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is None:
            continue
        mutable = isinstance(value, (ast.Dict, ast.Set, ast.List))
        if isinstance(value, ast.Call):
            called = getattr(value.func, "id", None) or getattr(value.func, "attr", None)
            mutable = mutable or called in MUTABLE_CALLS
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id.startswith("__") and target.id.endswith("__"):
                continue  # `__all__` and friends are the module's own surface
            if mutable or target.id in rebound:
                found.append((target.id, node.lineno, node.end_lineno or node.lineno))
    return found


def _stated_lifetime(
    comments: dict[int, tuple[str, bool]], start: int, end: int,
) -> str:
    """Every comment attached to this binding: the block above, then trailing.

    They join in reading order with the markers stripped, so a phrase that wraps
    onto the next comment line still reads as one phrase. The walk climbs only
    through comments that own their line, so a neighbor's trailing lifetime
    stops at that neighbor.
    """
    above = start - 1
    while above in comments and comments[above][1]:
        above -= 1
    lines = range(above + 1, end + 1)
    attached = [
        comments[line][0].lstrip("#").strip() for line in lines if line in comments
    ]
    return " ".join(" ".join(attached).lower().split())


def test_every_audited_registry_states_its_lifetime_in_place():
    """The scan: a new registry with no stated lifetime fails here, not in a
    later render."""
    missing: list[str] = []
    for path in AUDITED:
        comments = _comment_lines(path.read_text())
        for name, start, end in _module_registries(path):
            stated = _stated_lifetime(comments, start, end)
            if not any(phrase in stated for phrase in LIFETIME_PHRASES):
                missing.append(f"{path.relative_to(SRC).as_posix()}:{start} {name}")
    assert not missing, (
        "state with no stated lifetime (issue #197). Put one sentence next to "
        "each, using one of "
        + ", ".join(repr(phrase) for phrase in LIFETIME_PHRASES)
        + ":\n" + "\n".join(missing)
    )


def test_the_scan_would_catch_a_registry_added_without_a_lifetime(tmp_path):
    """A bare module dict and a `global`-rebound scalar must both fail the scan."""
    module = tmp_path / "new_leaf.py"
    module.write_text("_REGISTRY: dict = {}\n_COUNT = 0\n\n\ndef bump():\n"
                      "    global _COUNT\n    _COUNT += 1\n")
    comments = _comment_lines(module.read_text())
    found = _module_registries(module)
    assert [name for name, _start, _end in found] == ["_REGISTRY", "_COUNT"]
    for _name, start, end in found:
        stated = _stated_lifetime(comments, start, end)
        assert not any(phrase in stated for phrase in LIFETIME_PHRASES)


def test_a_neighbors_trailing_lifetime_does_not_cover_the_line_below(tmp_path):
    """A trailing lifetime covers only the binding on its own line.

    A registry declared on the next line states nothing, so it must fail;
    otherwise every trailing lifetime would also cover the line under it, the
    likeliest place for a new store."""
    module = tmp_path / "shielded.py"
    module.write_text("_FIRST: dict = {}  # session lifetime\n_SECOND: dict = {}\n")
    comments = _comment_lines(module.read_text())
    stated = {
        name: _stated_lifetime(comments, start, end)
        for name, start, end in _module_registries(module)
    }

    assert "session lifetime" in stated["_FIRST"]
    assert not any(phrase in stated["_SECOND"] for phrase in LIFETIME_PHRASES)


# The eight Gate names nodes/common forwards: one home, and the facade only reads.

# Every spelling of the facade module that an assignment could name.
_COMMON_PATHS = ("common", "nodes.common", "dgx_monarch.nodes.common")


def _dotted(node: ast.expr) -> str | None:
    """`a.b.c` as a string; None for anything that is not a plain name chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _common_owners(tree: ast.Module) -> set[str]:
    """Names this module can reach nodes/common through, aliases included."""
    owners = set(_COMMON_PATHS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "common" and (node.module or "").endswith("nodes"):
                    owners.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dgx_monarch.nodes.common" and alias.asname:
                    owners.add(alias.asname)
    return owners


def _facade_shadow_sites(source: str, label: str) -> list[str]:
    """Assignments that would bind a Gate name on nodes/common itself.

    Reads through the facade forward to the home. A write does not: it binds a
    new module attribute, the forward stops answering for that name, and every
    later reader silently gets the shadow instead of the home. Mutating the
    object the forward hands back stays legal, so the match requires the
    assignment target to be the name itself.
    """
    tree = ast.parse(source, filename=label)
    owners = _common_owners(tree)
    sites: list[tuple[int, str]] = []

    def flag(node: ast.AST, name: str) -> None:
        line = getattr(node, "lineno", 0)
        sites.append((line, f"{label}:{line} sets common.{name}"))

    def leaves(target: ast.expr) -> list[ast.expr]:
        """A tuple target unpacks; nothing else does.

        The walk stops at the target itself, so `common._X.attr = v` and
        `common._X[key] = v` reach through the forward and stay legal.
        """
        if isinstance(target, (ast.Tuple, ast.List)):
            return [leaf for element in target.elts for leaf in leaves(element)]
        if isinstance(target, ast.Starred):
            return leaves(target.value)
        return [target]

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "setattr" and len(node.args) >= 2:
                owner = _dotted(node.args[0])
                attribute = node.args[1]
                if (owner in owners and isinstance(attribute, ast.Constant)
                        and attribute.value in common._GATE_PROCESS_STATE_NAMES):
                    flag(node, str(attribute.value))
            continue
        for target in targets:
            for leaf in leaves(target):
                if (isinstance(leaf, ast.Attribute)
                        and leaf.attr in common._GATE_PROCESS_STATE_NAMES
                        and _dotted(leaf.value) in owners):
                    flag(leaf, leaf.attr)
    return [text for _line, text in sorted(sites)]


def test_no_module_binds_a_gate_name_on_the_facade_instead_of_its_home():
    """The write side of the forward in nodes/common.

    A write would shadow the forward and go unnoticed until two readers
    disagreed. Rebinds belong on nodes/gate_process_state.
    """
    scanned = sorted({*SRC.rglob("*.py"), *(REPO / "tests").rglob("*.py")})
    shadows: list[str] = []
    for path in scanned:
        shadows += _facade_shadow_sites(
            path.read_text(), path.relative_to(REPO).as_posix())
    assert not shadows, (
        "rebind these on dgx_monarch.nodes.gate_process_state, the home the "
        "forward reads:\n" + "\n".join(shadows)
    )


def test_the_facade_scan_separates_a_shadowing_write_from_a_legal_mutation():
    """The scan flags each shadowing write and passes each legal mutation.

    The source is a string because its four shadowing writes would otherwise
    fail the scan on this file itself.
    """
    source = (
        "from dgx_monarch.nodes import common\n"
        "from dgx_monarch.nodes import common as facade\n"
        "from dgx_monarch.nodes import gate_process_state\n"
        "common._AUTO_GATE_SESSION[token] = 'PASS'\n"
        "common._AUTO_GATE_ACTIVE.on = False\n"
        "common._AUTO_GATE_SESSION.clear()\n"
        "gate_process_state._PROCESS_GATE_DENIALS = {}\n"
        "monkeypatch.setattr(gate_process_state, '_AUTO_GATE_SESSION', {})\n"
        "common._AUTO_GATE_SESSION = {}\n"
        "facade._PROCESS_GATE_DENIALS = {}\n"
        "monkeypatch.setattr(common, '_AUTO_GATE_WAIT_S', 1.0)\n"
        "common._AUTO_GATE_RUNNING, keep = set(), 1\n"
    )
    assert _facade_shadow_sites(source, "synthetic.py") == [
        "synthetic.py:9 sets common._AUTO_GATE_SESSION",
        "synthetic.py:10 sets common._PROCESS_GATE_DENIALS",
        "synthetic.py:11 sets common._AUTO_GATE_WAIT_S",
        "synthetic.py:12 sets common._AUTO_GATE_RUNNING",
    ]


# Key space: the gate token. Calibration case: the capped no-material memo.

_TOKEN = ("combo", "artifact", "commit", "context")


@pytest.fixture(autouse=True)
def _restore_process_state():
    denials = dict(gate_process_state._PROCESS_GATE_DENIALS)
    session = dict(common._AUTO_GATE_SESSION)
    memo = dict(gate_inconclusive._NO_MATERIAL_TOKENS)
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
    common._AUTO_GATE_SESSION.clear()
    first_render.reset()
    yield
    first_render.reset()
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
    gate_inconclusive._NO_MATERIAL_TOKENS.update(memo)
    gate_process_state._PROCESS_GATE_DENIALS = denials
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_SESSION.update(session)


def _no_material(token=_TOKEN):
    return {"verdict": "INCONCLUSIVE", "inconclusive_kind": "no_material",
            "_gate_token": token}


def _flood(count: int) -> None:
    """Publish `count` later combinations through the publication path.

    Each is a no-material INCONCLUSIVE: a flood of PASSes grows the session
    cache but leaves the denial map and the memo at one entry each, so it
    cannot see a cap on either of them. A PASS flood would have passed with
    the 128-entry memo cap in place.
    """
    for index in range(count):
        other = ("combo", f"artifact-{index}", "commit", "context")
        common._record_process_gate_verdicts(
            [other], "INCONCLUSIVE", _no_material(other))


def test_the_denial_map_and_the_no_material_memo_cannot_diverge_by_lifetime():
    """The capped memo, re-derived from the lifetimes alone.

    Both stores gain at most one entry per distinct combination, always in the
    same locked publication step. Write past the session cap and the first
    entry must still read the same in both: a cap on one alone is what let an
    earned no-material outcome fall back to a session quarantine.
    """
    common._record_process_gate_verdicts([_TOKEN], "INCONCLUSIVE", _no_material())
    _flood(gate_process_state._AUTO_GATE_SESSION_LIMIT + 64)

    assert gate_process_state._PROCESS_GATE_DENIALS.get(_TOKEN) == "INCONCLUSIVE"
    assert gate_inconclusive._NO_MATERIAL_TOKENS.get(_TOKEN) is True
    assert gate_inconclusive.cached_verdict(_TOKEN, "INCONCLUSIVE") == (
        gate_inconclusive.NO_MATERIAL_VERDICT)


def test_the_one_capped_gate_store_can_never_outvote_the_uncapped_denial_map():
    """Why the session cache alone may carry a cap: it is read second.

    Evicting a denial would hand back a lever the gate took away. Evicting a
    session row cannot, because the denial map answers first.
    """
    common._record_process_gate_verdicts([_TOKEN], "FAIL")
    _flood(gate_process_state._AUTO_GATE_SESSION_LIMIT + 8)

    assert _TOKEN not in common._AUTO_GATE_SESSION  # the cap did evict it
    assert common._process_gate_verdict(_TOKEN) == "FAIL"


def test_a_dropped_session_pass_costs_a_re_ceremony_and_nothing_else():
    """The other side of the same cap: an evicted PASS re-runs the gate."""
    common._record_process_gate_verdicts([_TOKEN], "PASS")
    _flood(gate_process_state._AUTO_GATE_SESSION_LIMIT + 8)

    assert common._process_gate_verdict(_TOKEN) is None
    assert _TOKEN not in gate_process_state._PROCESS_GATE_DENIALS


def test_a_pass_over_a_denial_clears_both_stores_in_one_step():
    """Publication writes both stores in one locked step, so the pair moves together."""
    common._record_process_gate_verdicts([_TOKEN], "INCONCLUSIVE", _no_material())
    common._record_process_gate_verdicts([_TOKEN], "PASS")

    assert _TOKEN not in gate_process_state._PROCESS_GATE_DENIALS
    assert _TOKEN not in gate_inconclusive._NO_MATERIAL_TOKENS
    assert common._process_gate_verdict(_TOKEN) == "PASS"


# Key space: the render token and the thread ident. Calibration case: the
# inherited ceremony flag.


def _tracker_state() -> dict:
    return telemetry.render_progress._state


def test_ceremony_membership_stays_a_subset_of_the_active_renders():
    """The inherited ceremony flag, re-derived from the lifetimes alone.

    `_tokens` and `_ceremony_tokens` share the render-token key space, so the
    second must be a subset of the first at every step of any interleaving and
    both must empty at the last finish.
    """
    tracker = telemetry.render_progress
    tracker.note_ceremony()
    proof = tracker.start(2, {"model": "proof"}, token=object())
    tracker.close_ceremony()
    warm = tracker.start(20, {"model": "warm"}, token=object())
    operator = tracker.start(20, {"model": "operator"}, token=object())

    for _step in range(3):
        state = _tracker_state()
        assert set(state["_ceremony_tokens"]) <= set(state["_tokens"])

    tracker.finish(warm)
    assert set(_tracker_state()["_ceremony_tokens"]) <= set(_tracker_state()["_tokens"])
    tracker.finish(proof)
    tracker.finish(operator)
    assert not _tracker_state()["_tokens"]
    assert not _tracker_state().get("_ceremony_tokens")


def test_a_concurrent_render_never_inherits_the_ceremony_flag():
    """The defect itself: a render that owns no window and spends no claim."""
    tracker = telemetry.render_progress
    tracker.note_ceremony()
    proof = tracker.start(2, {"model": "proof"}, token=object())

    intruder: list[object] = []

    def other_thread() -> None:
        intruder.append(tracker.start(20, {"model": "fleet"}, token=object()))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()

    assert set(_tracker_state()["_ceremony_tokens"]) == {proof}
    tracker.finish(intruder[0])
    tracker.finish(proof)
    tracker.close_ceremony()


def test_the_bundle_record_reports_any_member_not_the_last_finisher():
    """Overlapping renders drain into one record that says whether any was the ceremony."""
    tracker = telemetry.render_progress
    tracker.note_ceremony()
    proof = tracker.start(2, {"model": "proof"}, token=object())
    tracker.close_ceremony()
    warm = tracker.start(20, {"model": "warm"}, token=object())

    tracker.finish(proof)
    tracker.finish(warm)  # the ceremony render does not finish last

    assert tracker.snapshot()["last"]["ceremony"] is True


def test_a_closed_window_hands_over_exactly_one_claim():
    """`_ceremony_windows` and `_ceremony_claim` share the thread-ident key
    space: the close moves the fact from the first to the second, and the next
    start spends it once."""
    tracker = telemetry.render_progress
    ident = threading.get_ident()
    tracker.note_ceremony()
    assert _tracker_state()["_ceremony_windows"].get(ident) == 1
    tracker.close_ceremony()
    assert ident not in _tracker_state()["_ceremony_windows"]
    assert ident in _tracker_state()["_ceremony_claim"]

    first = tracker.start(20, {"model": "operator"}, token=object())
    assert ident not in _tracker_state()["_ceremony_claim"]
    second = tracker.start(20, {"model": "next"}, token=object())
    assert set(_tracker_state()["_ceremony_tokens"]) == {first}

    tracker.finish(first)
    tracker.finish(second)


def test_the_ceremony_window_reads_the_same_in_both_of_its_homes():
    """`first_render._CEREMONY` and the tracker's `_ceremony_windows` hold one
    fact keyed by the same thread ident, so the two calls that write it must
    leave them agreeing."""
    ident = threading.get_ident()

    first_render.gate_started("unknown")
    assert getattr(first_render._CEREMONY, "active", False) is True
    assert _tracker_state()["_ceremony_windows"].get(ident) == 1

    first_render.gate_finished({"verdict": "PASS", "wall_s": 1.0})
    assert getattr(first_render._CEREMONY, "active", False) is False
    assert ident not in _tracker_state()["_ceremony_windows"]


# Key space: the residency memos, drained together by a recycle.


@pytest.fixture
def _restore_residency_memos():
    yield
    recycle_drain.drop_residency_memos()


def test_a_recycle_drains_every_loader_site_memo():
    """Three stores with three key spaces and one drain point. A fourth memo
    that lands without being wired into `reset_memos` outlives its siblings
    across a recycle, which is this class again."""
    loader_graph.remember(loader_preflight._CLEARED_UNETS, "model.safetensors")
    loader_graph.remember(loader_graph._CHARGED_STACK_ARTIFACTS, "te.safetensors")
    consent_projection._AUDITED_NON_MEMO_GRANTS.add(("rescue-slab", "combo", "panel"))

    loader_preflight.reset_memos()

    assert not loader_preflight._CLEARED_UNETS
    assert not loader_graph._CHARGED_STACK_ARTIFACTS
    assert not consent_projection._AUDITED_NON_MEMO_GRANTS


def test_a_recycle_does_not_re_arm_the_headless_warning(monkeypatch):
    """The fourth loader-site store stays out of the drain.

    The other three memo residency, which a recycle makes false. This one
    records that a caller reached the wall with no graph, which a recycle does
    not change: respawned workers hand a headless caller no prompt either.
    Draining it would reprint one line per recycle and say nothing new, so it
    states proc lifetime.
    """
    monkeypatch.setattr(loader_preflight, "_UNPRICED_STACK_WARNED", {"minimax_h3"})
    loader_graph.remember(loader_preflight._CLEARED_UNETS, "model.safetensors")
    loader_graph.remember(loader_graph._CHARGED_STACK_ARTIFACTS, "te.safetensors")
    consent_projection._AUDITED_NON_MEMO_GRANTS.add(("rescue-slab", "combo", "panel"))

    loader_preflight.reset_memos()

    assert not loader_preflight._CLEARED_UNETS
    assert not loader_graph._CHARGED_STACK_ARTIFACTS
    assert not consent_projection._AUDITED_NON_MEMO_GRANTS
    assert loader_preflight._UNPRICED_STACK_WARNED == {"minimax_h3"}


def test_the_capped_loader_memos_fail_toward_a_refusal_never_toward_a_load():
    """Both loader memos may carry a cap because a missing entry re-charges.
    An over-charge refuses loudly; an under-charge lets an oversized load
    reach the allocator, which is why the verdict stores are uncapped."""
    for index in range(loader_graph._MEMO_LIMIT + 10):
        loader_graph.remember(loader_preflight._CLEARED_UNETS, f"m{index}")
        loader_graph.remember(loader_graph._CHARGED_STACK_ARTIFACTS, f"a{index}")

    assert len(loader_preflight._CLEARED_UNETS) == loader_graph._MEMO_LIMIT
    assert len(loader_graph._CHARGED_STACK_ARTIFACTS) == loader_graph._MEMO_LIMIT
    assert "m0" not in loader_preflight._CLEARED_UNETS  # evicted: it re-charges
    loader_preflight.reset_memos()


def test_a_recycle_drains_both_homes_of_the_checkpoint_key_space(
        _restore_residency_memos):
    """The loader site and the render site memo the same fact about the same
    `unet_name`, and a recycle makes it false at both. Drain one and the other
    keeps crediting `weights_resident` for a checkpoint the respawned fleet
    never loaded: an under-charge, which fails open."""
    loader_graph.remember(loader_preflight._CLEARED_UNETS, "model.safetensors")
    render_preflight.note_successful_render("model.safetensors")
    render_preflight.note_submitted_render("next.safetensors")
    render_preflight.note_driver_charged_render("third.safetensors")

    recycle_drain.drop_residency_memos()

    assert not loader_preflight._CLEARED_UNETS
    assert render_preflight._LAST_RENDERED_UNET is None
    assert render_preflight._LAST_SUBMITTED_UNET is None
    assert render_preflight._LAST_DRIVER_CHARGED_UNET is None


def test_every_door_that_stops_the_procs_runs_the_drain():
    """Three sites in nodes/ call the handle's recycle API: the /dgxm/recycle
    route, the ClearVRAM node at level=recycle, and the abandoned-lease heal in
    render_submit. A door that skips the drain keeps the memos the others drop,
    which is this class one level up."""
    for path in sorted((SRC / "nodes").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [
                call.func.attr for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
            ]
            if not {"recycle", "recycle_detailed"}.intersection(calls):
                continue
            assert "drop_residency_memos" in calls, (
                f"{path.relative_to(SRC).as_posix()}:{node.lineno} {node.name} "
                "stops the worker procs without draining the residency memos")


def _retire(handle, _exc, *, holding_lock: bool = False) -> None:
    handle.defunct = True


def _leave_live(_handle, _exc, *, holding_lock: bool = False) -> None:
    return None


def test_the_render_collect_arm_evicts_through_the_draining_door():
    """The AST guard above keys on the recycle API, which this door never
    calls, so the wiring itself is pinned here instead."""
    from dgx_monarch.nodes import pending

    assert pending.evict_render_fleet is recycle_drain.evict_render_fleet


def test_an_eviction_that_retires_the_fleet_drains_the_memos(
        monkeypatch, _restore_residency_memos):
    """A stall eviction stops the worker procs and the driver session survives
    it: the next render respawns a fleet holding nothing while the memos still
    credit it with the checkpoint the killed procs held."""
    from dgx_monarch import mesh_helpers

    handle = SimpleNamespace(defunct=False)
    monkeypatch.setattr(mesh_helpers, "mark_defunct_preserving_primary", _retire)
    loader_graph.remember(loader_preflight._CLEARED_UNETS, "model.safetensors")
    render_preflight.note_successful_render("model.safetensors")
    render_preflight.note_submitted_render("next.safetensors")
    render_preflight.note_driver_charged_render("third.safetensors")

    recycle_drain.evict_render_fleet(handle, RuntimeError("render stalled"))

    assert not loader_preflight._CLEARED_UNETS
    assert render_preflight._LAST_RENDERED_UNET is None
    assert render_preflight._LAST_SUBMITTED_UNET is None
    assert render_preflight._LAST_DRIVER_CHARGED_UNET is None


def test_a_refusal_that_leaves_the_fleet_live_keeps_the_memos(
        monkeypatch, _restore_residency_memos):
    """A worker refusal is an endpoint error: the actor is alive, the mesh
    keeps its weights, and draining there would double-charge the next estimate
    into a refusal the hardware does not support. The drain reads the handle,
    never the exception."""
    from dgx_monarch import mesh_helpers

    handle = SimpleNamespace(defunct=False)
    monkeypatch.setattr(mesh_helpers, "mark_defunct_preserving_primary",
                        _leave_live)
    render_preflight.note_successful_render("model.safetensors")

    recycle_drain.evict_render_fleet(handle, RuntimeError("worker refused"))

    assert render_preflight._LAST_RENDERED_UNET == "model.safetensors"


def test_a_broken_drain_never_reshapes_the_render_failure(monkeypatch):
    """This door runs inside an exception handler that reconciles whatever it
    raises into the surfaced error. A drain that raised would rewrite the
    render failure the operator is meant to read.

    No memo fixture here: this test writes no memo, and the drain it installs
    raises, so a restoring teardown would run the explosion."""
    from dgx_monarch import mesh_helpers

    def explode() -> None:
        raise RuntimeError("drain exploded")

    monkeypatch.setattr(mesh_helpers, "mark_defunct_preserving_primary", _retire)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", explode)

    recycle_drain.evict_render_fleet(
        SimpleNamespace(defunct=False), RuntimeError("render stalled"))


# Key space: retained failed-load ownership and its poison latch.


def test_retained_ownership_and_the_poison_latch_cannot_disagree():
    """The latch is the third home of the two retained lists: anything owned
    means the latch is set, so a reload cannot start over a mapping nobody has
    proved is dead."""
    assert not slab_lifetime.cleanup_pending()

    class _Slab:
        def close(self) -> None:
            pass

    slab_lifetime.retain_failed_load_resource(_Slab())
    try:
        assert slab_lifetime.cleanup_pending()
        assert slab_lifetime.cleanup_poisoned()
        assert slab_lifetime.retained_resource_count() == 1
    finally:
        slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES.clear()
        slab_lifetime._FAILED_LOAD_CLEANUP_POISONED = False
