"""Check typed-refusal classifications, the unclassified backlog and guard names.

docs/DESIGN.md section 5.9 defines P (physics), C (capacity-protective),
U (unproven correctness) and K (known-wrong math). dgx_monarch.refusal.refusal
encodes the class in the message so it survives worker-to-driver wrapping.

The ledger pins classified sites and per-module unclassified counts. A
reclassification requires a reviewed update; new untagged refusals fail the
check, and the grandfathered backlog can only shrink. Guard names must also
agree with the consent subsystem's waiver registry.

Regenerate the ledger after a reviewed change:

    PYTHONPATH=src python tests/test_refusal_classes.py --write
"""

from __future__ import annotations

import ast
import copy
import functools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"
LEDGER = Path(__file__).with_name("refusal_class_ledger.json")

sys.path.insert(0, str(REPO / "src"))

from dgx_monarch.refusal import (  # noqa: E402
    GUARDS,
    PanelAction,
    RefusalClass,
    parse_leading_refusal_tag,
    parse_refusal_tag,
    refusal,
)

# The typed-refusal classes the walker recognizes, named rather than found by
# base class: a base class would sweep in MeshAttachError and miss every
# ValueError-derived refusal. Every exception class declared in src must be in
# exactly one of this set and NOT_REFUSAL_CLASSES.
REFUSAL_CLASSES = frozenset({
    "UnsupportedModelError",
    "StockLoadCapacityError",
    "ComfyManagedResidencyError",
    "DriverFootprintCapacityError",
    "ConsentRequiredError",
    "SlabResidencyRescueOffer",
    "SlabCertificateError",
    "ArtifactBindingError",
    "CheckpointSniffError",
    "SafetensorsHeaderError",
    "UnsupportedSafetensorsDtypeError",
    "PackedDataParallelError",
    "PackedCfgParallelError",
    "PartialLoadDivergenceError",
    "RenderMemoryPriceError",
    "DualModelLoadDivergenceError",
    # The post-load readiness exchange (actor/rank_readiness.py). Class P on
    # the healthy rank: MIN carries agreement and not identity, so this rank
    # cannot know the peer's cause and no consent offered here would change
    # the answer. The rank that failed keeps its own class.
    "PeerLoadFailureError",
    # The debug load fault (actor/load_fault.py), which fails one rank on
    # purpose so the readiness exchange above can run on hardware. Class P on
    # the rank it names: that box was told to fail, so no consent could change
    # the answer, and its own text says the fault was injected rather than
    # borrowing the healthy rank's message about a peer.
    "InjectedLoadFaultError",
    # The pre-load fleet agreement (capacity_agreement.py). Both are class C and
    # both name the rank; they are two types rather than one because only the
    # shortfall is a capacity answer. A rank that never answered has not said
    # it cannot load anything, so its refusal stays outside the
    # StockLoadCapacityError family the cross-mode ceremony converts.
    "FleetCapacityError",
    "FleetQuoteUnavailableError",
})

# Declared exceptions that are not user-facing model/capacity refusals: control
# flow, lifecycle faults, IO faults and internal invariants. Adding an exception
# class to src forces a choice between these two sets.
NOT_REFUSAL_CLASSES = frozenset({
    "CaptureAborted",
    # Lifecycle faults from the sample stall guard and the poisoned-context
    # latch (2026-08-21): the stall is a driver-side eviction decision and the
    # latch is host-local, so neither carries a refusal class tag (the lease
    # doctrine's rank-symmetry rule; actor/failure.py states it for the latch).
    "SampleStallError",
    "CudaContextPoisonedError",
    "ClusterConfigError",
    "ClusterSmokeError",
    "ConsentStoreError",
    "DriverPinError",
    "PayloadOwnershipError",
    "GateAuditError",
    "ProfileRefusal",
    "PriorTeardownUnknownError",
    "ReleaseLayoutError",
    "RepairError",
    "SetupCliInputError",
    "SetupConfigLockUnavailable",
    "StageFailure",
    "UpdateLockUnavailable",
    "UpdatePreconditionError",
    "WaiverNotAuditedError",
    "ConcurrentRenderSessionError",
    "FsdpGateProofError",
    "GateLedgerReadError",
    "GateLedgerWriteError",
    "GateProvenanceError",
    "InvalidCleanupResult",
    "LifecycleBusyError",
    "MeshAttachError",
    "RenderCancelledError",
    "ResidentAdoptionEvidenceError",
    "RetiredRenderHandleError",
    "SampleResultBusyError",
    "SetupVerdictError",
    "StaleSetupGenerationError",
    "TopologyTransitionError",
    "UnbakeError",
    # An internal signal, not an operator refusal: it says this box cannot
    # fingerprint an artifact, and every caller decides what that means. The
    # one caller where it decides an outcome, the loader-site preflight,
    # converts it into a tagged class C refusal (2026-09-04).
    "UnresolvedArtifactIdentityError",
    "WorkerSpawnError",
    "_Abort",
    "_InvalidQuantConfError",
})

# A class P message must point somewhere reachable. One of these must appear.
ALTERNATIVE_TOKENS = (
    "Use ",
    "use a ",
    "Run this workflow on",
    "Run reference-image renders on",
    "run on topology",
    "instead",
    "take a true single-GPU reference",
)

# A class K message with no guard can never be waived, and must say so.
NO_WAIVER_TOKENS = (
    "no waiver for this refusal",
    "cannot be waived",
)


class _Decl:
    """One `refusal(...)` call, read statically."""

    __slots__ = ("guard", "panel_action", "refusal_class", "text", "waivable")

    def __init__(self, refusal_class: str, guard: str | None, waivable: bool,
                 panel_action: bool, text: str) -> None:
        self.refusal_class = refusal_class
        self.guard = guard
        self.waivable = waivable
        self.panel_action = panel_action
        self.text = text

    def as_row(self) -> dict[str, object]:
        return {
            "class": self.refusal_class,
            "guard": self.guard,
            "waivable": self.waivable,
        }


def _literal(node: ast.expr | None) -> object:
    try:
        return ast.literal_eval(node) if node is not None else None
    except (ValueError, TypeError, SyntaxError):
        return None


def _static_text(node: ast.expr, names: dict[str, str] | None = None,
                 package: dict[str, dict[str, str]] | None = None) -> str:
    """The statically knowable part of a message: literals, joined.

    Module-level string constants resolve too, against the same table the guard
    ids resolve against. A family of refusals that share one tail (the
    comfy-managed bring-up binds its "what works instead" sentence once and
    raises it from four sites) would otherwise read as four messages with no
    alternative in them, and the class rules below would be checking a message
    no operator ever sees. ``package`` extends that to a constant another module
    owns, read here as ``<module>.<NAME>``: one message raised by a driver site
    and a worker site is bound once so the two cannot drift.
    """
    parts: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            parts.append(sub.value)
        elif isinstance(sub, ast.Name) and names and sub.id in names:
            parts.append(names[sub.id])
        elif (isinstance(sub, ast.Attribute) and package
                and isinstance(sub.value, ast.Name)
                and sub.attr in package.get(sub.value.id, {})):
            parts.append(package[sub.value.id][sub.attr])
    return " ".join(parts)


def _read_refusal_call(node: ast.Call, names: dict[str, str] | None = None,
                       package: dict[str, dict[str, str]] | None = None) -> _Decl | None:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name != "refusal" or not node.args:
        return None
    cls_arg = node.args[0]
    cls_value = None
    if isinstance(cls_arg, ast.Attribute):
        cls_value = getattr(RefusalClass, cls_arg.attr, None)
    if cls_value is None:
        return None
    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
    guard_node = kwargs.get("guard")
    guard = _literal(guard_node)
    if guard is None and isinstance(guard_node, ast.Name) and names:
        # A site may bind its guard id to a module constant, which is the
        # better style when one guard has several arms.
        guard = names.get(guard_node.id)
    waivable = _literal(kwargs.get("waivable")) is True
    text = _static_text(node.args[1], names, package) if len(node.args) > 1 else ""
    return _Decl(cls_value.value, guard if isinstance(guard, str) else None,
                 waivable, "panel_action" in kwargs, text)


def _module_name(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


class _ModuleWalk:
    """Constants bound to a `refusal(...)` call, plus every typed raise."""

    def __init__(self, path: Path, tree: ast.Module) -> None:
        self.module = _module_name(path)
        self.tree = tree
        self.constants: dict[str, _Decl] = {}
        self.strings: dict[str, str] = {}
        self.imports: dict[str, tuple[str, str]] = {}
        self.raises: list[dict[str, object]] = []

    def collect_strings(self) -> None:
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.strings[target.id] = node.value.value

    def collect_constants(self, package: dict[str, dict[str, str]]) -> None:
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                decl = _read_refusal_call(node.value, self.strings, package)
                if decl is None:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.constants[target.id] = decl
        # Function-local imports count: the driver-side preflights import the
        # adapters' already-tagged constants inside the function that raises.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.imports.setdefault(
                        alias.asname or alias.name,
                        (_resolve_import(self.module, node), alias.name),
                    )

    def collect_raises(self) -> None:
        counts: dict[tuple[str, str], int] = {}
        stack: list[str] = []

        def visit(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.append(node.name)
                for child in ast.iter_child_nodes(node):
                    visit(child)
                stack.pop()
                return
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc
                call = exc if isinstance(exc, ast.Call) else None
                target = call.func if call is not None else exc
                exc_name = (
                    target.attr if isinstance(target, ast.Attribute)
                    else getattr(target, "id", None)
                )
                if exc_name in REFUSAL_CLASSES:
                    qual = ".".join(stack) or "<module>"
                    key = (qual, exc_name)
                    counts[key] = counts.get(key, 0) + 1
                    self.raises.append({
                        "module": self.module,
                        "qual": qual,
                        "exception": exc_name,
                        "ordinal": counts[key] - 1,
                        "line": node.lineno,
                        "arg": call.args[0] if call is not None and call.args else None,
                    })
            for child in ast.iter_child_nodes(node):
                visit(child)

        for node in self.tree.body:
            visit(node)


def _resolve_import(module: str, node: ast.ImportFrom) -> str:
    """Best-effort relative-import resolution inside the package."""
    if node.level == 0:
        return (node.module or "").replace("dgx_monarch.", "").replace(".", "/") + ".py"
    parts = module.split("/")[:-1]
    for _ in range(node.level - 1):
        if parts:
            parts.pop()
    if node.module:
        parts.extend(node.module.split("."))
    return "/".join(parts) + ".py"


def _package_strings(walks: dict[str, _ModuleWalk]) -> dict[str, dict[str, str]]:
    """Module-level string constants, keyed by the name an importer writes.

    Only unambiguous stems: two modules of the same basename would make
    ``<module>.<NAME>`` mean two things, and a checker that guessed would read a
    message no operator sees.
    """
    by_stem: dict[str, list[dict[str, str]]] = {}
    for walk in walks.values():
        stem = walk.module.rsplit("/", 1)[-1].removesuffix(".py")
        by_stem.setdefault(stem, []).append(walk.strings)
    return {stem: tables[0] for stem, tables in by_stem.items() if len(tables) == 1}


def _walk_tree() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    # Four checks read the same walk of the source tree. Walk it once per run
    # and hand each caller its own copy, so no check can see another's edits.
    return copy.deepcopy(_walk_tree_once())


@functools.cache
def _walk_tree_once() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    walks: dict[str, _ModuleWalk] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        walk = _ModuleWalk(path, tree)
        walk.collect_strings()
        walks[walk.module] = walk
    package = _package_strings(walks)
    for walk in walks.values():
        walk.collect_constants(package)
        walk.collect_raises()

    tagged: list[dict[str, object]] = []
    untagged: list[dict[str, object]] = []
    for walk in walks.values():
        for site in walk.raises:
            decl = _decl_for(site["arg"], walk, walks, package)
            row = {
                "site": f"{site['module']}::{site['qual']}::{site['exception']}#{site['ordinal']}",
                "line": site["line"],
            }
            if decl is None:
                untagged.append(row)
                continue
            row.update(decl.as_row())
            row["panel_action"] = decl.panel_action
            row["text"] = decl.text
            tagged.append(row)
    return tagged, untagged


def _decl_for(arg: object, walk: _ModuleWalk, walks: dict[str, _ModuleWalk],
              package: dict[str, dict[str, str]]) -> _Decl | None:
    if arg is None:
        return None
    node = arg
    if isinstance(node, ast.Call):
        return _read_refusal_call(node, walk.strings, package)
    if isinstance(node, ast.BinOp):
        for side in (node.left, node.right):
            found = _decl_for(side, walk, walks, package)
            if found is not None:
                return found
        return None
    name = node.id if isinstance(node, ast.Name) else None
    if name is None:
        return None
    if name in walk.constants:
        return walk.constants[name]
    origin = walk.imports.get(name)
    if origin is None:
        return None
    source, original = origin
    other = walks.get(source)
    if other is None:
        return None
    return other.constants.get(original)


def _ledger() -> dict[str, object]:
    return json.loads(LEDGER.read_text())


def _untagged_by_module(untagged: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in untagged:
        module = str(row["site"]).split("::", 1)[0]
        counts[module] = counts.get(module, 0) + 1
    return dict(sorted(counts.items()))


def _base_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Attribute):
            names.add(base.attr)
        elif isinstance(base, ast.Name):
            names.add(base.id)
    return names


def test_every_declared_exception_is_classified_as_refusal_or_not():
    """An exception class is caught by what it derives from, not by its name.

    Naming is not a contract: ``SlabResidencyRescueOffer`` is a refusal and
    ends in neither Error nor Aborted. Sweeping on the base class means a new
    typed refusal cannot escape classification by being called something else,
    which is how the walker would otherwise stop seeing its raise sites.
    """
    declared: set[str] = set()
    known = REFUSAL_CLASSES | NOT_REFUSAL_CLASSES
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        local: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = _base_names(node)
            derives = any(
                base.endswith(("Error", "Exception", "Aborted")) or base in known | local
                for base in bases
            )
            if derives or node.name.endswith(("Error", "Aborted")):
                declared.add(node.name)
                local.add(node.name)
    unclassified = declared - REFUSAL_CLASSES - NOT_REFUSAL_CLASSES
    assert not unclassified, (
        "new exception classes must join REFUSAL_CLASSES (a user-facing typed "
        f"refusal, so its raise sites need a class tag) or NOT_REFUSAL_CLASSES: {sorted(unclassified)}"
    )


def test_classified_refusal_sites_match_the_ledger():
    tagged, _ = _walk_tree()
    actual = {row["site"]: {k: row[k] for k in ("class", "guard", "waivable")} for row in tagged}
    expected = {row["site"]: {k: row[k] for k in ("class", "guard", "waivable")}
                for row in _ledger()["tagged"]}
    assert actual == expected, (
        "refusal class ledger drift; regenerate with "
        "`PYTHONPATH=src python tests/test_refusal_classes.py --write` after review"
    )


def test_untagged_refusal_sites_never_grow():
    _, untagged = _walk_tree()
    actual = _untagged_by_module(untagged)
    ledger = _ledger()
    expected = ledger["untagged_by_module"]
    assert actual == expected, (
        "a refusal site was added or removed without a class tag. Every NEW "
        "typed refusal must call dgx_monarch.refusal.refusal(...); a site that "
        "gained a tag decrements its module's count here.\n"
        f"actual={actual}\nledger={expected}"
    )
    pending = ledger.get("new_sites_awaiting_a_class", {})
    backlog = sum(count for module, count in actual.items() if module not in pending)
    assert backlog <= ledger["grandfathered_ceiling"], (
        f"the grandfathered backlog is {backlog}, over the dated ceiling "
        f"{ledger['grandfathered_ceiling']} in DESIGN.md section 7. That number "
        "may only fall: sites added on this branch belong in "
        "new_sites_awaiting_a_class, not in the backlog."
    )


def test_no_new_refusal_site_is_waiting_for_a_class():
    """The grandfathered backlog covers only sites that predate the taxonomy.

    A refusal added on a branch is never grandfathered. A site that lands
    untagged mid-branch goes in the ledger's new_sites_awaiting_a_class, keyed
    by module, so the count stays true while this test stays red. Clear an
    entry by writing the tag, not by deleting the line.
    """
    pending = _ledger().get("new_sites_awaiting_a_class", {})
    assert not pending, (
        "typed refusals added on this branch still carry no class tag. Wrap the "
        "message in dgx_monarch.refusal.refusal(...), then regenerate the ledger "
        "and remove the entry:\n"
        + "\n".join(f"  {module}: {reason}" for module, reason in sorted(pending.items()))
    )


def test_class_rules_hold_at_every_classified_site():
    tagged, _ = _walk_tree()
    failures: list[str] = []
    for row in tagged:
        site, cls, guard, waivable = (
            row["site"], row["class"], row["guard"], row["waivable"],
        )
        text = str(row["text"])
        if cls == "P":
            if guard is not None or waivable:
                failures.append(f"{site}: class P has no guard and is never waivable")
            if not any(token in text for token in ALTERNATIVE_TOKENS):
                failures.append(f"{site}: class P must name the working alternative")
        elif cls in ("C", "U"):
            if guard is None:
                failures.append(f"{site}: class {cls} must name the boundary it protects")
            if waivable and not row["panel_action"]:
                failures.append(f"{site}: a waivable class {cls} refusal must name the panel action")
        elif cls == "K":
            if guard is None:
                if waivable:
                    failures.append(f"{site}: a waivable class K refusal must declare its guard")
                elif not any(token in text for token in NO_WAIVER_TOKENS):
                    failures.append(
                        f"{site}: an unwaivable class K refusal must say there is no waiver"
                    )
            if waivable and not row["panel_action"]:
                failures.append(f"{site}: a waivable class K refusal must name the panel action")
        if guard is not None and guard not in GUARDS:
            failures.append(f"{site}: guard {guard!r} is not in refusal.GUARDS")
        elif guard is not None and GUARDS[guard].refusal_class.value != cls:
            failures.append(f"{site}: guard {guard!r} is class {GUARDS[guard].refusal_class.value}")
        if guard is not None and waivable and not GUARDS[guard].waivable_now:
            failures.append(f"{site}: guard {guard!r} has no consent wired in this release")
    assert not failures, "refusal class rule violations:\n" + "\n".join(failures)


def test_guard_vocabulary_is_claimed_or_explicitly_reserved():
    tagged, _ = _walk_tree()
    claimed = {row["guard"] for row in tagged if row["guard"]}
    reserved = set(_ledger()["reserved_guards"])
    assert claimed & reserved == set(), (
        "a reserved guard is now claimed by a refusal site; move it out of "
        f"reserved_guards: {sorted(claimed & reserved)}"
    )
    assert set(GUARDS) == claimed | reserved, (
        "refusal.GUARDS must be exactly the guards claimed by a site plus the "
        f"reserved ones. claimed={sorted(claimed)} reserved={sorted(reserved)} "
        f"registry={sorted(GUARDS)}"
    )


def test_guard_vocabulary_agrees_with_the_frozen_ledger_vocabulary():
    """The tag registry and the WAIVER row's guard allow-list must agree.

    Two frozen tables drift unless a test compares them, and the failure is
    silent: a refusal names a guard the ledger then refuses to record, so the
    bypass happens and the audit row does not.
    """
    from dgx_monarch.gate_audit_vocab import KIND_CLASS, check_guard

    failures: list[str] = []
    for guard, spec in GUARDS.items():
        kind = spec.consent_kind
        if kind not in KIND_CLASS:
            failures.append(f"{guard}: consent kind {kind!r} is not a v9 waiver kind")
            continue
        if KIND_CLASS[kind] != spec.refusal_class.value:
            failures.append(
                f"{guard}: refusal.GUARDS says class {spec.refusal_class.value}, "
                f"the ledger vocabulary says {KIND_CLASS[kind]} for kind {kind!r}"
            )
        try:
            check_guard(kind, guard)
        except (ValueError, TypeError) as exc:
            failures.append(f"{guard}: the ledger vocabulary rejects it under {kind!r}: {exc}")
    assert not failures, "refusal/ledger guard vocabulary drift:\n" + "\n".join(failures)


def test_every_waivable_guard_message_names_the_panel_and_the_env_fallback():
    for guard, spec in GUARDS.items():
        if not spec.waivable_now:
            continue
        action = PanelAction("Load with slab residency", "DGXM_ALLOW_SLAB_RESCUE")
        message = refusal(
            spec.refusal_class, "measured numbers go here.", guard=guard,
            waivable=True, panel_action=action, troubleshooting=53,
        )
        assert 'click "Load with slab residency"' in message
        assert "DGX Monarch panel" in message
        assert "DGXM_ALLOW_SLAB_RESCUE=1" in message
        assert message.index("DGX Monarch panel") < message.index("Headless:")
        assert "docs/TROUBLESHOOTING.md #53" in message


def test_tag_survives_the_wire_wrapping():
    message = refusal(
        RefusalClass.KNOWN_WRONG, "measured wrong math.", guard="ring_pad:minimax_h3",
    )
    wrapped = RuntimeError(f"ActorError: UnsupportedModelError: {message}\n  at rank 1")
    tag = parse_refusal_tag(str(wrapped))
    assert tag is not None
    assert tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == "ring_pad:minimax_h3"
    assert tag.waivable is False


def test_only_a_canonical_leading_tag_proves_a_worker_refusal():
    message = refusal(
        RefusalClass.KNOWN_WRONG, "measured wrong math.", guard="ring_pad:minimax_h3",
    )
    tag = parse_leading_refusal_tag(message)
    assert tag is not None
    assert tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert parse_leading_refusal_tag(f"ActorError: {message}") is None
    assert parse_leading_refusal_tag(f"CUDA crash while loading artifact/{message}") is None
    assert parse_leading_refusal_tag(None) is None  # type: ignore[arg-type]


def test_tag_shaped_literal_text_cannot_mint_an_ownership_tag():
    message = refusal(
        RefusalClass.CAPACITY,
        "Refuses models/[dgxm:P]checkpoint.safetensors. Use a smaller artifact.",
        guard="driver_footprint_preflight",
    )
    assert message.startswith("[dgxm:C guard=driver_footprint_preflight waivable=0] ")
    assert "models/[dgxm;P]checkpoint.safetensors" in message
    assert message.count("[dgxm:") == 1
    broad = parse_refusal_tag(message)
    leading = parse_leading_refusal_tag(message)
    assert broad is not None and broad.refusal_class is RefusalClass.CAPACITY
    assert leading == broad

    leading_name = refusal(
        RefusalClass.PHYSICS,
        "[dgxm:P] checkpoint.safetensors is malformed. Use a valid artifact.",
    )
    assert leading_name.startswith("[dgxm:P] [dgxm;P] checkpoint.safetensors")
    assert parse_leading_refusal_tag(leading_name) is not None


def test_two_tags_in_one_text_refuse_to_parse():
    one = refusal(RefusalClass.PHYSICS, "Use topology 'single'.")
    assert parse_refusal_tag(one) is not None
    assert parse_refusal_tag(one + " " + one) is None
    assert parse_refusal_tag("no tag here") is None
    assert parse_refusal_tag(None) is None  # type: ignore[arg-type]


def test_refusal_refuses_an_inconsistent_declaration():
    import pytest

    with pytest.raises(ValueError):  # class P has no guard
        refusal(RefusalClass.PHYSICS, "Use x.", guard="ring_pad")
    with pytest.raises(ValueError):  # waivable with no guard
        refusal(RefusalClass.CAPACITY, "x.", waivable=True)
    with pytest.raises(ValueError):  # waivable with no panel action
        refusal(RefusalClass.CAPACITY, "x.", guard="stock_load_preflight", waivable=True)
    with pytest.raises(ValueError):  # a panel action on a refusal nobody can clear
        refusal(RefusalClass.KNOWN_WRONG, "x.", guard="ring_pad",
                panel_action=PanelAction("Do it", "DGXM_X"))
    with pytest.raises(ValueError):  # guard with no consent wired
        refusal(RefusalClass.UNPROVEN, "x.", guard="first_load_stock_memo", waivable=True,
                panel_action=PanelAction("Do it", "DGXM_X"))
    with pytest.raises(ValueError):  # unknown guard
        refusal(RefusalClass.CAPACITY, "x.", guard="not_a_guard")
    with pytest.raises(ValueError):  # class C must name its boundary
        refusal(RefusalClass.CAPACITY, "x.")
    nested = refusal(RefusalClass.PHYSICS, refusal(RefusalClass.PHYSICS, "Use x."))
    assert nested == "[dgxm:P] [dgxm;P] Use x."
    assert parse_refusal_tag(nested) is not None
    with pytest.raises(ValueError):  # empty text
        refusal(RefusalClass.PHYSICS, "   ")


def _write_ledger() -> None:
    tagged, untagged = _walk_tree()
    existing = _ledger() if LEDGER.exists() else {}
    claimed = {row["guard"] for row in tagged if row["guard"]}
    payload = {
        "note": existing.get("note", ""),
        "regenerate": "PYTHONPATH=src python tests/test_refusal_classes.py --write",
        "grandfathered_ceiling": existing.get("grandfathered_ceiling", len(untagged)),
        "grandfathered_dated": existing.get("grandfathered_dated", ""),
        "new_sites_awaiting_a_class": existing.get("new_sites_awaiting_a_class", {}),
        "reserved_guards": {
            guard: GUARDS[guard].note for guard in sorted(set(GUARDS) - claimed)
        },
        "untagged_by_module": _untagged_by_module(untagged),
        "tagged": sorted(
            ({k: row[k] for k in ("site", "class", "guard", "waivable")} for row in tagged),
            key=lambda row: str(row["site"]),
        ),
    }
    LEDGER.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {LEDGER} ({len(payload['tagged'])} tagged, "
          f"{sum(payload['untagged_by_module'].values())} untagged)")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write_ledger()
    else:
        print("pass --write to regenerate tests/refusal_class_ledger.json")
